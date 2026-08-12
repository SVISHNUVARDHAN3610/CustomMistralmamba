"""Production training for Hybrid Mamba-MoE with streaming tokenized shards.

Consumes binary shards produced by ``utils.dataset.TokenizedShardProducer`` via
``MmapShardDataset``. Validation data loading is intentionally omitted here;
wire that in when a held-out pipeline is ready.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from model.core.builders import count_trainable_params
from model.core.config import HybridMambaMoEConfig
from model.core.constants import MEMORY_NAN_FIX_ID
from model.hybrid.losses import _aux_loss_schedule, _expert_loss_schedule
from model.hybrid.mamba import (
    fused_mamba_scan_available,
    get_mamba_scan_stats,
    log_mamba_backend,
    reset_mamba_scan_stats,
)
from model.hybrid.model import HybridForCausalLM
from utils.dataset import (
    MmapShardDataset,
    TokenizedShardProducer,
    verify_tokenizer_vocab,
)

torch.set_float32_matmul_precision("high")

CHECKPOINT_FILENAME = "model_ckpt.pth"
CONFIG_FILENAME = "config.json"


# ---------------------------------------------------------------------------
# Seeding & logging
# ---------------------------------------------------------------------------


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)


def setup_logging(run_dir: Path, *, level: int = logging.INFO) -> logging.Logger:
    """Console + rotating run log under ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Model & optimizers
# ---------------------------------------------------------------------------


def build_training_config(vocab_size: int) -> HybridMambaMoEConfig:
    """Default ~80–120M Hybrid config aligned with the prior Mixtral-scale run."""
    hidden_size = 512
    num_heads = 8
    head_dim = 64
    return HybridMambaMoEConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=8,
        num_heads=num_heads,
        num_kv_heads=8,
        head_dim=head_dim,
        intermediate_size=512,
        window_size=512,
        num_experts=8,
        top_k=2,
        dropout=0.0,
        capacity_factor=None,
        router_aux_loss_coef=0.02,
        router_z_loss_coef=5e-3,
        max_position_embeddings=4096,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        mamba_state_size=16,
        mamba_conv_kernel=4,
        mamba_expand=2,
        use_dual_memory=True,
        memory_size=48,
        memory_num_heads=8,
        memory_chunk_size=512,
        stream_chunked_ce_loss=True,
        return_logits=False,
        use_auxiliary_losses=True,
        use_fused_mamba_scan=True,
    )


# Token / slot embeddings and classifier head stay on AdamW (Moonshot + Keller).
# Router expert-selection matrices are 2D hidden weights and stay on Muon
# (Moonlight SVD analysis includes routers under Muon).
_ADAMW_NAME_SUBSTRINGS = (
    "embed_tokens",
    "lm_head",
    "init_memory",
    "summary_query",
)


def _is_adamw_parameter(name: str, param: nn.Parameter) -> bool:
    """True when Muon must not own this parameter.

    Rules from arXiv:2502.16982 + torch.optim.Muon docs:
      - Muon only accepts 2D matrices (hidden-layer weights).
      - Embeddings, LM head, RMSNorm / bias / other non-matrix params -> AdamW.
      - MoE router matrices are 2D and should use Muon (not AdamW).
      - Mamba Conv1d weights are 3D -> AdamW.
      - Dual-memory slot banks (init_memory / summary_query) are embedding-like -> AdamW.
    """
    if param.ndim != 2:
        return True
    return any(key in name for key in _ADAMW_NAME_SUBSTRINGS)


def split_muon_adam_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], dict[str, list[str]]]:
    """Split parameters into AdamW vs Muon groups with name inventories."""
    adam_params: list[nn.Parameter] = []
    muon_params: list[nn.Parameter] = []
    inventory: dict[str, list[str]] = {"adamw": [], "muon": []}
    seen: set[int] = set()

    for name, param in model.named_parameters():
        # Tied embeddings / lm_head share storage; optimize once.
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)

        if _is_adamw_parameter(name, param):
            adam_params.append(param)
            inventory["adamw"].append(f"{name}{tuple(param.shape)}")
        else:
            muon_params.append(param)
            inventory["muon"].append(f"{name}{tuple(param.shape)}")

    return adam_params, muon_params, inventory


def build_optimizers(
    model: nn.Module,
    *,
    lr: float,
    muon_lr: float | None,
    adam_lr: float | None,
    weight_decay: float,
    muon_momentum: float,
    muon_nesterov: bool,
    muon_ns_steps: int,
    muon_adjust_lr_fn: str,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    use_muon: bool,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[list[optim.Optimizer], bool, dict[str, Any]]:
    """Create Muon+AdamW (paper hybrid) or AdamW-only fallback.

    With ``adjust_lr_fn='match_rms_adamw'``, Moonshot scales each Muon update by
    ``0.2 * sqrt(max(A, B))`` so a *shared* learning rate / weight decay can be
    reused for both optimizers (arXiv:2502.16982 §2.2).
    """
    adam_params, muon_params, inventory = split_muon_adam_params(model)
    total_params = sum(p.numel() for p in model.parameters())
    adam_count = sum(p.numel() for p in adam_params)
    muon_count = sum(p.numel() for p in muon_params)

    muon_available = hasattr(optim, "Muon")
    enable_muon = use_muon and bool(muon_params) and muon_available
    if use_muon and not muon_available:
        logger.warning(
            "torch.optim.Muon unavailable in this PyTorch build; "
            "falling back to AdamW for all parameters"
        )
        enable_muon = False

    shared_lr = lr
    resolved_muon_lr = muon_lr if muon_lr is not None else shared_lr
    resolved_adam_lr = adam_lr if adam_lr is not None else shared_lr

    meta: dict[str, Any] = {
        "enable_muon": enable_muon,
        "adam_param_count": adam_count,
        "muon_param_count": muon_count if enable_muon else 0,
        "adam_pct": 100.0 * adam_count / max(total_params, 1),
        "muon_pct": 100.0 * (muon_count if enable_muon else 0) / max(total_params, 1),
        "total_params": total_params,
        "muon_lr": resolved_muon_lr if enable_muon else None,
        "adam_lr": resolved_adam_lr,
        "weight_decay": weight_decay,
        "muon_momentum": muon_momentum if enable_muon else None,
        "muon_adjust_lr_fn": muon_adjust_lr_fn if enable_muon else None,
        "inventory": inventory,
    }

    logger.info(
        "optimizer split: adamw=%.2f%% (%d tensors, %s params) "
        "muon=%.2f%% (%d tensors, %s params) total=%.3fB",
        meta["adam_pct"],
        len(inventory["adamw"]),
        f"{adam_count:,}",
        meta["muon_pct"],
        len(inventory["muon"]) if enable_muon else 0,
        f"{meta['muon_param_count']:,}",
        total_params / 1e9,
    )
    logger.debug("AdamW params: %s", inventory["adamw"])
    logger.debug("Muon params: %s", inventory["muon"] if enable_muon else [])

    fused_adam = device.type == "cuda"
    if enable_muon:
        muon_kwargs: dict[str, Any] = {
            "lr": resolved_muon_lr,
            "weight_decay": weight_decay,
            "momentum": muon_momentum,
            "nesterov": muon_nesterov,
            "ns_steps": muon_ns_steps,
        }
        # PyTorch >= 2.9 exposes Moonshot RMS matching; older builds may not.
        try:
            muon_optim: optim.Optimizer = optim.Muon(
                muon_params,
                adjust_lr_fn=muon_adjust_lr_fn,
                **muon_kwargs,
            )
        except TypeError:
            logger.warning(
                "torch.optim.Muon does not accept adjust_lr_fn=%r; "
                "constructing Muon without it (upgrade PyTorch for Moonshot RMS matching)",
                muon_adjust_lr_fn,
            )
            muon_optim = optim.Muon(muon_params, **muon_kwargs)
            meta["muon_adjust_lr_fn"] = None

        adam_optim = optim.AdamW(
            adam_params,
            lr=resolved_adam_lr,
            betas=(adam_beta1, adam_beta2),
            eps=adam_eps,
            weight_decay=weight_decay,
            fused=fused_adam,
        )
        logger.info(
            "Muon(lr=%.3e, wd=%.3g, momentum=%.3g, nesterov=%s, ns_steps=%d, "
            "adjust_lr_fn=%s) + AdamW(lr=%.3e, betas=(%.2f, %.2f), wd=%.3g)",
            resolved_muon_lr,
            weight_decay,
            muon_momentum,
            muon_nesterov,
            muon_ns_steps,
            meta["muon_adjust_lr_fn"],
            resolved_adam_lr,
            adam_beta1,
            adam_beta2,
            weight_decay,
        )
        return [muon_optim, adam_optim], True, meta

    adam_optim = optim.AdamW(
        list(model.parameters()),
        lr=resolved_adam_lr,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
        weight_decay=weight_decay,
        fused=fused_adam,
    )
    logger.info(
        "AdamW-only (lr=%.3e, betas=(%.2f, %.2f), wd=%.3g)",
        resolved_adam_lr,
        adam_beta1,
        adam_beta2,
        weight_decay,
    )
    return [adam_optim], False, meta


def align_tokens_per_shard(tokens_per_shard: int, seq_len: int) -> int:
    """Round down so shard boundaries never drop partial sequences."""
    chunk = seq_len + 1  # MmapShardDataset uses seq_len+1 for input/label pairs
    aligned = (tokens_per_shard // chunk) * chunk
    aligned = max(aligned, chunk)
    return aligned


def validate_token_batch(
    input_ids: torch.Tensor,
    vocab_size: int,
    labels: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> None:
    max_id = int(input_ids.max().item())
    min_id = int(input_ids.min().item())
    if min_id < 0 or max_id >= vocab_size:
        raise ValueError(
            f"input_ids out of range [{min_id}, {max_id}] for vocab_size={vocab_size}"
        )
    if labels is None:
        return
    active = labels != ignore_index
    if not active.any():
        return
    label_max = int(labels[active].max().item())
    label_min = int(labels[active].min().item())
    if label_min < 0 or label_max >= vocab_size:
        raise ValueError(
            f"labels out of range [{label_min}, {label_max}] for vocab_size={vocab_size}"
        )


def _resolve_warmup_steps(warmup_steps: int, total_steps: int) -> int:
    if warmup_steps > 0:
        return min(warmup_steps, max(1, total_steps - 1))
    return max(1, int(total_steps * 0.026))


def _build_lr_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float) -> Any:
    decay_steps = max(1, total_steps - warmup_steps)
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


class RollingAverage:
    def __init__(self, window: int) -> None:
        self._values: deque[float] = deque(maxlen=max(1, window))

    def update(self, value: float) -> None:
        self._values.append(value)

    @property
    def mean(self) -> float | None:
        if not self._values:
            return None
        return sum(self._values) / len(self._values)


def _weighted_terms(
    model: HybridForCausalLM, out: Any, step: int, max_steps: int
) -> dict[str, float]:
    cfg = model.config
    aux = out.auxiliary_losses
    assert aux is not None
    assoc_scale = _aux_loss_schedule(step, max_steps, cfg.assoc_warmup_fraction)
    expert_scale = _expert_loss_schedule(step, max_steps, cfg.expert_warmup_fraction)
    return {
        "recon_w": float((cfg.lambda_recon * aux.recon).item()),
        "assoc_w": float((cfg.lambda_assoc * assoc_scale * aux.assoc).item()),
        "gate_w": float((cfg.lambda_gate * aux.gate).item()),
        "read_w": float((cfg.lambda_read * aux.read).item()),
        "fusion_w": float((cfg.lambda_fusion * aux.fusion).item()),
        "expert_w": float((cfg.lambda_expert * expert_scale * aux.expert).item()),
        "ssm_w": float((cfg.lambda_ssm * aux.ssm).item()),
        "slot_w": float((cfg.lambda_slot * aux.slot).item()),
        "assoc_scale": assoc_scale,
        "expert_scale": expert_scale,
    }


def _scalar_label(value: torch.Tensor | None) -> str:
    if value is None:
        return "none"
    scalar = float(value.detach().float().item())
    if math.isnan(scalar):
        return "nan"
    if math.isinf(scalar):
        return "inf"
    return f"{scalar:.8g}"


def _non_finite_diagnosis(
    model: HybridForCausalLM,
    out: Any,
    step: int,
    max_steps: int,
) -> dict[str, Any]:
    aux = out.auxiliary_losses
    weighted = _weighted_terms(model, out, step, max_steps)
    raw: dict[str, torch.Tensor | None] = {
        "loss": out.loss,
        "ce_loss": out.ce_loss,
        "router_aux": out.router_aux_loss,
        "router_z": out.router_z_loss,
    }
    if aux is not None:
        for name in (
            "recon",
            "assoc",
            "gate",
            "read",
            "fusion",
            "expert",
            "ssm",
            "slot",
        ):
            raw[name] = getattr(aux, name)
    non_finite = [
        name
        for name, tensor in raw.items()
        if tensor is not None and not torch.isfinite(tensor).all().item()
    ]
    values = {name: _scalar_label(tensor) for name, tensor in raw.items()}
    return {
        "non_finite_terms": non_finite,
        "values": values,
        "assoc_scale": weighted["assoc_scale"],
        "expert_scale": weighted["expert_scale"],
    }


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def _rng_state_dict() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _load_rng_state_dict(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_opt_sched(
    optimizers: list[optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    use_muon: bool,
) -> tuple[list[optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
    """Normalize to [muon_or_adam, adam] layout expected by save/load_checkpoint."""
    if use_muon:
        return optimizers, schedulers
    return [optimizers[0], optimizers[0]], [schedulers[0], schedulers[0]]


def save_checkpoint(
    *,
    model: HybridForCausalLM,
    optimizers: list[optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    global_step: int,
    current_shard_idx: int,
    checkpoint_dir: Path,
    logger: logging.Logger,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / CHECKPOINT_FILENAME
    tmp_path = ckpt_path.with_suffix(".pth.tmp")

    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(model.config),
        "global_step": global_step,
        "current_shard_idx": current_shard_idx,
        "rng_state": _rng_state_dict(),
        "memory_nan_fix_id": MEMORY_NAN_FIX_ID,
    }
    if len(optimizers) >= 1:
        payload["muon_optimizer_state_dict"] = optimizers[0].state_dict()
    if len(optimizers) >= 2:
        payload["adam_optimizer_state_dict"] = optimizers[1].state_dict()
    if len(schedulers) >= 1:
        payload["muon_scheduler_state_dict"] = schedulers[0].state_dict()
    if len(schedulers) >= 2:
        payload["adam_scheduler_state_dict"] = schedulers[1].state_dict()

    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)

    config_path = checkpoint_dir / CONFIG_FILENAME
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(model.config), f, indent=2)

    logger.info(
        "Checkpoint saved step=%d shard=%d path=%s",
        global_step,
        current_shard_idx,
        ckpt_path,
    )


def load_checkpoint(
    *,
    model: HybridForCausalLM,
    optimizers: list[optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    checkpoint_dir: Path,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[int, int]:
    ckpt_path = checkpoint_dir / CHECKPOINT_FILENAME
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if len(optimizers) >= 1 and "muon_optimizer_state_dict" in checkpoint:
        optimizers[0].load_state_dict(checkpoint["muon_optimizer_state_dict"])
    if len(optimizers) >= 2 and "adam_optimizer_state_dict" in checkpoint:
        optimizers[1].load_state_dict(checkpoint["adam_optimizer_state_dict"])
    if len(schedulers) >= 1 and "muon_scheduler_state_dict" in checkpoint:
        schedulers[0].load_state_dict(checkpoint["muon_scheduler_state_dict"])
    if len(schedulers) >= 2 and "adam_scheduler_state_dict" in checkpoint:
        schedulers[1].load_state_dict(checkpoint["adam_scheduler_state_dict"])

    _load_rng_state_dict(checkpoint.get("rng_state"))

    global_step = int(checkpoint.get("global_step", 0))
    current_shard_idx = int(checkpoint.get("current_shard_idx", 0))
    logger.info(
        "Resumed from %s | step=%d shard=%d fix_id=%s",
        ckpt_path,
        global_step,
        current_shard_idx,
        checkpoint.get("memory_nan_fix_id", "unknown"),
    )
    return global_step, current_shard_idx


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _format_log_line(step: int, max_steps: int, record: dict[str, Any]) -> str:
    assoc_tag = "warm" if record.get("assoc_scale") == 0.0 else "on"
    expert_tag = "warm" if record.get("expert_scale") == 0.0 else "on"
    smooth = record.get("ce_smooth")
    smooth_str = f"{smooth:.6f}" if smooth is not None else "n/a"
    return (
        f"step={step}/{max_steps} "
        f"shard={record.get('shard_idx', 0)} "
        f"loss={record['loss']:.6f} ce={record['ce_loss']:.6f} "
        f"ce_smooth={smooth_str} "
        f"router_aux={record['router_aux_loss']:.6f} "
        f"router_z={record['router_z_loss']:.6f} "
        f"recon={record['recon']:.6f} assoc={record['assoc']:.6f}({assoc_tag}) "
        f"expert={record['expert']:.6f}({expert_tag}) "
        f"grad_norm={record['grad_norm']:.4f} "
        f"muon_lr={record['muon_lr']:.2e} adam_lr={record['adam_lr']:.2e}"
    )


def train(args: argparse.Namespace, logger: logging.Logger) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(args.log_jsonl) if args.log_jsonl else None
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    tokens_per_shard = align_tokens_per_shard(args.tokens_per_shard, args.seq_len)
    if tokens_per_shard != args.tokens_per_shard:
        logger.warning(
            "Adjusted tokens_per_shard %d -> %d (multiple of seq_len+1=%d)",
            args.tokens_per_shard,
            tokens_per_shard,
            args.seq_len + 1,
        )

    producer = TokenizedShardProducer(
        cache_dir=args.cache_dir,
        tokenizer_name=args.tokenizer_name,
        tokens_per_shard=tokens_per_shard,
        max_buffered_files=args.max_buffered_files,
    )
    verify_tokenizer_vocab(producer.tokenizer, args.vocab_size)

    cfg = build_training_config(vocab_size=args.vocab_size)
    cfg.bos_token_id = producer.tokenizer.bos_token_id or cfg.bos_token_id
    cfg.eos_token_id = producer.tokenizer.eos_token_id or cfg.eos_token_id
    if args.compile:
        cfg.use_torch_compile = True
        cfg.torch_compile_mode = args.compile_mode

    logger.info(log_mamba_backend(cfg))
    logger.info(
        "device=%s amp=%s dtype=%s fused_mamba=%s memory_nan_fix=%s",
        device,
        use_amp,
        amp_dtype if use_amp else "fp32",
        fused_mamba_scan_available(),
        MEMORY_NAN_FIX_ID,
    )

    model = HybridForCausalLM(cfg).to(device)
    n_params = count_trainable_params(model)
    logger.info("trainable_params=%s (%.3fB)", f"{n_params:,}", n_params / 1e9)

    optimizers, use_muon, _opt_meta = build_optimizers(
        model,
        lr=args.lr,
        muon_lr=args.muon_lr,
        adam_lr=args.adam_lr,
        weight_decay=args.weight_decay,
        muon_momentum=args.muon_momentum,
        muon_nesterov=not args.no_muon_nesterov,
        muon_ns_steps=args.muon_ns_steps,
        muon_adjust_lr_fn=args.muon_adjust_lr_fn,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        use_muon=not args.no_muon,
        device=device,
        logger=logger,
    )

    warmup_steps = _resolve_warmup_steps(args.warmup_steps, args.max_steps)
    lr_lambda = _build_lr_lambda(warmup_steps, args.max_steps, args.min_lr_ratio)
    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = [
        torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        for opt in optimizers
    ]

    global_step = 0
    current_shard_idx = 0
    ckpt_dir = Path(args.ckpt_dir)

    if args.resume:
        ckpt_optimizers, ckpt_schedulers = _checkpoint_opt_sched(
            optimizers, schedulers, use_muon
        )
        global_step, current_shard_idx = load_checkpoint(
            model=model,
            optimizers=ckpt_optimizers,
            schedulers=ckpt_schedulers,
            checkpoint_dir=ckpt_dir,
            device=device,
            logger=logger,
        )

    stop_event = threading.Event()
    producer_thread: threading.Thread | None = None
    ce_smooth = RollingAverage(args.smooth_window)
    reset_mamba_scan_stats()

    try:
        if args.resume and os.path.exists(args.dataset_ckpt_path):
            producer.load_checkpoint(args.dataset_ckpt_path)
            current_shard_idx = producer.current_shard_idx

        producer_thread = threading.Thread(
            target=producer.start_streaming,
            kwargs={
                "stop_event": stop_event,
                "checkpoint_path": args.dataset_ckpt_path,
            },
            daemon=True,
        )
        producer_thread.start()
        logger.info("Shard producer started | cache_dir=%s", args.cache_dir)

        dl_generator = torch.Generator()
        dl_generator.manual_seed(args.seed)

        while global_step < args.max_steps:
            shard_name = f"shard_{current_shard_idx:06d}.bin"
            bin_path = os.path.join(args.cache_dir, shard_name)
            wait_start = time.time()
            timed_out = False

            while not os.path.exists(bin_path):
                if time.time() - wait_start > args.producer_wait_timeout:
                    logger.error(
                        "Timeout waiting for %s after %ds",
                        shard_name,
                        args.producer_wait_timeout,
                    )
                    timed_out = True
                    break
                time.sleep(1.0)

            if timed_out:
                break

            dataset = MmapShardDataset(bin_path=bin_path, seq_len=args.seq_len + 1)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=device.type == "cuda",
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
                prefetch_factor=2 if args.num_workers > 0 else None,
                worker_init_fn=seed_worker if args.num_workers > 0 else None,
                generator=dl_generator,
            )

            model.train()
            logger.info(
                "Shard %d | sequences=%d | step %d/%d",
                current_shard_idx,
                len(dataset),
                global_step,
                args.max_steps,
            )

            for input_ids, labels in dataloader:
                if global_step >= args.max_steps:
                    break

                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                validate_token_batch(
                    input_ids,
                    cfg.vocab_size,
                    labels=labels,
                    ignore_index=cfg.label_ignore_index,
                )

                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    outputs = model(
                        input_ids=input_ids,
                        labels=labels,
                        training_step=global_step,
                        max_training_steps=args.max_steps,
                    )
                assert outputs.loss is not None

                if not torch.isfinite(outputs.loss):
                    diagnosis = _non_finite_diagnosis(
                        model, outputs, global_step, args.max_steps
                    )
                    logger.error(
                        "Non-finite loss at step=%d: %s", global_step, diagnosis
                    )
                    if jsonl_path is not None:
                        with jsonl_path.open("a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "event": "non_finite_loss",
                                        "step": global_step,
                                        "shard_idx": current_shard_idx,
                                        **diagnosis,
                                    }
                                )
                                + "\n"
                            )
                    global_step += 1
                    continue

                outputs.loss.backward()
                grad_norm = float(
                    clip_grad_norm_(model.parameters(), args.max_grad_norm).item()
                )

                for opt in optimizers:
                    opt.step()
                for sched in schedulers:
                    sched.step()

                step_ce = (
                    float(outputs.ce_loss.item())
                    if outputs.ce_loss is not None
                    else 0.0
                )
                ce_smooth.update(step_ce)

                aux = outputs.auxiliary_losses
                assert aux is not None
                weighted = _weighted_terms(model, outputs, global_step, args.max_steps)
                muon_lr = float(schedulers[0].get_last_lr()[0])
                adam_lr = float(schedulers[-1].get_last_lr()[0])

                record: dict[str, Any] = {
                    "step": global_step,
                    "shard_idx": current_shard_idx,
                    "loss": float(outputs.loss.item()),
                    "ce_loss": step_ce,
                    "ce_smooth": ce_smooth.mean,
                    "router_aux_loss": float(outputs.router_aux_loss.item())
                    if outputs.router_aux_loss is not None
                    else 0.0,
                    "router_z_loss": float(outputs.router_z_loss.item())
                    if outputs.router_z_loss is not None
                    else 0.0,
                    "recon": float(aux.recon.item()),
                    "assoc": float(aux.assoc.item()),
                    "gate": float(aux.gate.item()),
                    "read": float(aux.read.item()),
                    "fusion": float(aux.fusion.item()),
                    "expert": float(aux.expert.item()),
                    "ssm": float(aux.ssm.item()),
                    "slot": float(aux.slot.item()),
                    "grad_norm": grad_norm,
                    "muon_lr": muon_lr,
                    "adam_lr": adam_lr,
                    **weighted,
                    "gate_stats": {
                        k: float(v.item())
                        for k, v in (outputs.gate_stats or {}).items()
                    },
                }

                if jsonl_path is not None:
                    with jsonl_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")

                if global_step % args.log_interval == 0:
                    logger.info(_format_log_line(global_step, args.max_steps, record))

                if global_step % args.save_interval == 0 and global_step > 0:
                    ckpt_optimizers, ckpt_schedulers = _checkpoint_opt_sched(
                        optimizers, schedulers, use_muon
                    )
                    save_checkpoint(
                        model=model,
                        optimizers=ckpt_optimizers,
                        schedulers=ckpt_schedulers,
                        global_step=global_step,
                        current_shard_idx=current_shard_idx,
                        checkpoint_dir=ckpt_dir,
                        logger=logger,
                    )
                    producer.save_checkpoint(args.dataset_ckpt_path)

                global_step += 1

            done_path = os.path.join(
                args.cache_dir, f"shard_{current_shard_idx:06d}.done"
            )
            with open(done_path, "w", encoding="utf-8") as f:
                f.write("done\n")
            logger.info("Marked shard %d complete", current_shard_idx)
            current_shard_idx += 1

        if global_step > 0 and global_step % args.save_interval != 0:
            ckpt_optimizers, ckpt_schedulers = _checkpoint_opt_sched(
                optimizers, schedulers, use_muon
            )
            save_checkpoint(
                model=model,
                optimizers=ckpt_optimizers,
                schedulers=ckpt_schedulers,
                global_step=global_step,
                current_shard_idx=current_shard_idx,
                checkpoint_dir=ckpt_dir,
                logger=logger,
            )
            producer.save_checkpoint(args.dataset_ckpt_path)

        scan_stats = get_mamba_scan_stats()
        logger.info(
            "Training finished steps=%d mamba_scan=%s",
            global_step,
            scan_stats,
        )

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — saving checkpoint before exit")
        if global_step > 0:
            ckpt_optimizers, ckpt_schedulers = _checkpoint_opt_sched(
                optimizers, schedulers, use_muon
            )
            save_checkpoint(
                model=model,
                optimizers=ckpt_optimizers,
                schedulers=ckpt_schedulers,
                global_step=global_step,
                current_shard_idx=current_shard_idx,
                checkpoint_dir=ckpt_dir,
                logger=logger,
            )
            producer.save_checkpoint(args.dataset_ckpt_path)
        raise

    except Exception:
        logger.exception("Training failed")
        raise

    finally:
        stop_event.set()
        if producer_thread is not None:
            producer_thread.join(timeout=30.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Hybrid Mamba-MoE on streaming tokenized shards.",
    )
    parser.add_argument(
        "--run-dir", type=str, default="./runs/train", help="Logs and artifacts root."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume model + dataset checkpoints."
    )
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Prefer reproducible CUDA kernels (slower).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed-precision autocast on CUDA.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bf16", "fp16"),
        default="bf16",
        help="Autocast dtype when AMP is enabled.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile decoder layers."
    )
    parser.add_argument("--compile-mode", type=str, default="default")

    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument(
        "--warmup-steps", type=int, default=0, help="0 = auto (~2.6%% of max_steps)."
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="Cosine floor as fraction of peak LR.",
    )
    # Moonshot Muon (arXiv:2502.16982): with match_rms_adamw, share η and λ across Muon+AdamW.
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Shared peak LR for Muon and AdamW when --muon-lr/--adam-lr are unset "
        "(Moonlight-style after update-RMS matching).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Decoupled weight decay for Muon and AdamW (Moonlight uses 0.1).",
    )
    parser.add_argument(
        "--muon-lr",
        type=float,
        default=None,
        help="Optional Muon LR override (default: --lr).",
    )
    parser.add_argument(
        "--adam-lr",
        type=float,
        default=None,
        help="Optional AdamW LR override (default: --lr).",
    )
    parser.add_argument(
        "--muon-momentum",
        type=float,
        default=0.95,
        help="Muon momentum μ (paper / Keller default: 0.95).",
    )
    parser.add_argument(
        "--no-muon-nesterov",
        action="store_true",
        help="Disable Nesterov momentum in Muon (enabled by default).",
    )
    parser.add_argument(
        "--muon-ns-steps",
        type=int,
        default=5,
        help="Newton-Schulz orthogonalization steps (paper: 5; 10 is not better).",
    )
    parser.add_argument(
        "--muon-adjust-lr-fn",
        type=str,
        default="match_rms_adamw",
        choices=("match_rms_adamw", "original"),
        help="Muon per-matrix LR scale. match_rms_adamw = Moonshot 0.2*sqrt(max(A,B)).",
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument(
        "--no-muon", action="store_true", help="Use AdamW for all parameters."
    )
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument(
        "--smooth-window", type=int, default=50, help="Rolling CE mean window."
    )
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--ckpt-dir", type=str, default="./model_ckpt")
    parser.add_argument(
        "--tokenizer-name", type=str, default="UIC-AI-lab/llama2-tokenizer"
    )
    parser.add_argument("--cache-dir", type=str, default="./data_cache")
    parser.add_argument(
        "--dataset-ckpt-path", type=str, default="./dataset_checkpoint.json"
    )
    parser.add_argument("--tokens-per-shard", type=int, default=5_000_000)
    parser.add_argument("--max-buffered-files", type=int, default=10)
    parser.add_argument("--producer-wait-timeout", type=int, default=600)
    parser.add_argument(
        "--log-jsonl",
        type=str,
        default="",
        help="Optional JSONL metrics path (default: <run-dir>/metrics.jsonl).",
    )
    args = parser.parse_args()
    if not args.log_jsonl:
        args.log_jsonl = str(Path(args.run_dir) / "metrics.jsonl")
    return args


def main() -> None:
    args = parse_args()
    logger = setup_logging(Path(args.run_dir))
    logger.info("Starting training with args: %s", vars(args))
    train(args, logger)


if __name__ == "__main__":
    main()
