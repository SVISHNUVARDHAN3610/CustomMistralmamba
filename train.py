"""Production training for Hybrid Mamba-MoE with streaming tokenized shards.

Consumes binary shards produced by ``utils.dataset.TokenizedShardProducer`` via
``MmapShardDataset``. Optional cyclic validation on ``Salesforce/wikitext``
(validation split) via ``utils.validation.WikiTextCyclicValidator``.
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
from model.core.optim import (
    _is_adamw_no_decay,
    split_muon_adam_params,
)
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
from utils.validation import WikiTextCyclicValidator

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
    """Default production Hybrid config (~148M trainable params, measured)."""
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


# Parameter-grouping helpers live in model/core/optim.py (dependency-free so
# tests can exercise them without this module's dataset imports). The names
# are re-imported above so existing references keep working.


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

        # Muon LR audit (Issue-3): the observed run logged muon_lr ≈ 1.2e-3
        # against a configured 1e-3. In the installed torch (2.13) Muon does
        # NOT touch param_groups['lr'] at construction — the Moonshot
        # per-shape RMS matching (0.2*sqrt(max(A,B)) for match_rms_adamw) is
        # applied INSIDE _single_tensor_muon at optimizer-step time, per
        # parameter. param_groups['lr'] therefore holds the configured base
        # LR here, and the effective step LR is base_lr * adjusted_ratio
        # (shape-dependent, so it cannot be summarized by one number). Older
        # torch builds baked the adjustment into group LR at init; the log
        # below makes either behavior visible at training start: if
        # group_lr != resolved_muon_lr, this build inflates the base at init
        # and the scheduler multiplies on top of the INFLATED value.
        group_lrs = [float(g["lr"]) for g in muon_optim.param_groups]
        for gi, group_lr in enumerate(group_lrs):
            if abs(group_lr - resolved_muon_lr) > 1e-12:
                logger.warning(
                    "Muon param_groups[%d].lr=%.6e != configured muon_lr=%.6e: "
                    "this torch build pre-scales the Muon LR at construction; "
                    "the scheduler warmup/cosine will apply on the INFLATED "
                    "base (effective per-step LR is further scaled per-param "
                    "by adjust_lr_fn=%r).",
                    gi,
                    group_lr,
                    resolved_muon_lr,
                    muon_adjust_lr_fn,
                )
            else:
                logger.info(
                    "Muon param_groups[%d].lr=%.6e matches configured muon_lr "
                    "(adjust_lr_fn=%r scaling is applied per-param at step "
                    "time, not baked into the group LR).",
                    gi,
                    group_lr,
                    muon_adjust_lr_fn,
                )

        # Split AdamW params so biases / norm gains / _no_weight_decay params
        # (Mamba A_log, D) get zero weight decay; embeddings and lm_head keep
        # the configured decay.
        adam_decay = [p for p in adam_params if not _is_adamw_no_decay(p)]
        adam_no_decay = [p for p in adam_params if _is_adamw_no_decay(p)]
        adam_groups = [
            {"params": adam_decay, "weight_decay": weight_decay},
            {"params": adam_no_decay, "weight_decay": 0.0},
        ]
        adam_optim = optim.AdamW(
            adam_groups,
            lr=resolved_adam_lr,
            betas=(adam_beta1, adam_beta2),
            eps=adam_eps,
            fused=fused_adam,
        )
        logger.info(
            "Muon(lr=%.3e, wd=%.3g, momentum=%.3g, nesterov=%s, ns_steps=%d, "
            "adjust_lr_fn=%s) + AdamW(lr=%.3e, betas=(%.2f, %.2f), "
            "wd=%.3g on %d params / wd=0 on %d params)",
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
            len(adam_decay),
            len(adam_no_decay),
        )
        return [muon_optim, adam_optim], True, meta

    adam_only_params = list(model.parameters())
    only_decay = [p for p in adam_only_params if not _is_adamw_no_decay(p)]
    only_no_decay = [p for p in adam_only_params if _is_adamw_no_decay(p)]
    adam_optim = optim.AdamW(
        [
            {"params": only_decay, "weight_decay": weight_decay},
            {"params": only_no_decay, "weight_decay": 0.0},
        ],
        lr=resolved_adam_lr,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
        fused=fused_adam,
    )
    logger.info(
        "AdamW-only (lr=%.3e, betas=(%.2f, %.2f), wd=%.3g on %d params / "
        "wd=0 on %d params)",
        resolved_adam_lr,
        adam_beta1,
        adam_beta2,
        weight_decay,
        len(only_decay),
        len(only_no_decay),
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


class EMABaseline:
    """Exponential moving average of recent *finite* values, on-device.

    Used by the magnitude-based spike guard: a value is "huge but finite"
    when it exceeds ``multiplier * ema``. The EMA itself only ingests values
    below the spike threshold, so one enormous step cannot ratchet the
    baseline up and hide the next spike.
    """

    def __init__(
        self,
        n_metrics: int,
        device: torch.device,
        multiplier: float,
        momentum: float = 0.99,
        min_history: int = 5,
    ) -> None:
        self.ema = torch.zeros(n_metrics, device=device, dtype=torch.float32)
        self.count = torch.zeros(n_metrics, device=device, dtype=torch.float32)
        self.multiplier = multiplier
        self.momentum = momentum
        self.min_history = min_history

    def update(self, values: torch.Tensor, finite: torch.Tensor) -> torch.Tensor:
        """Ingest a step's metric vector; return the spike mask (True = spike).

        ``finite`` is the isfinite mask for this step's values. Entries are
        only admitted into the EMA once ``min_history`` finite observations
        have accumulated, and only when the value itself is not a spike —
        early steps legitimately start large (loss ~10 vs EMA 0), so the
        warm-up avoids flagging the first steps as spikes forever.
        """
        spike = torch.zeros_like(self.ema, dtype=torch.bool)
        if self.multiplier <= 0.0:
            return spike
        # A value is a spike when its MAGNITUDE exceeds multiplier x the
        # magnitude of the baseline — sign-symmetric, because the vector
        # contains negative metrics (the gate-entropy loss is -entropy, in
        # [-ln 2, 0]); a sign-blind ``values > ema * multiplier`` turns the
        # bound into a large NEGATIVE number for those and flags every
        # in-range observation forever (the EMA then never ingests them).
        # ``trusted`` additionally requires a nonzero baseline: a metric that
        # is exactly 0 while warm (e.g. the expert loss before its 10%
        # switch-on) would otherwise arm on a zero baseline and flag its
        # first real value, which is then excluded, keeping the baseline at
        # zero — flagging forever. Transitioning off an all-zero history
        # re-arms the min_history warmup instead.
        trusted = (self.count >= self.min_history) & (self.ema != 0.0)
        bound = self.ema.abs() * self.multiplier
        first_value = ~trusted & finite & (values != 0.0) & (self.ema == 0.0)
        self.count = torch.where(first_value, torch.zeros_like(self.count), self.count)
        spike = trusted & finite & (values.abs() > bound)
        # Feed the EMA with non-spike finite values (spikes never ratchet it).
        admissible = finite & ~spike
        blended = self.momentum * self.ema + (1.0 - self.momentum) * values
        self.ema = torch.where(admissible, blended, self.ema)
        self.count = self.count + finite.to(self.count.dtype)
        return spike

    def reset(self) -> None:
        self.ema.zero_()
        self.count.zero_()


def _weighted_term_tensors(
    model: HybridForCausalLM, out: Any, step: int, max_steps: int
) -> dict[str, Any]:
    """Weighted auxiliary-loss terms, kept on-device.

    The hot training loop folds these straight into its single batched
    metric transfer; each ``.item()`` here would serialize its own GPU→CPU
    round-trip against everything already enqueued on the stream.
    """
    cfg = model.config
    aux = out.auxiliary_losses
    assert aux is not None
    assoc_scale = _aux_loss_schedule(step, max_steps, cfg.assoc_warmup_fraction)
    expert_scale = _expert_loss_schedule(step, max_steps, cfg.expert_warmup_fraction)
    return {
        "recon_w": cfg.lambda_recon * aux.recon,
        "assoc_w": cfg.lambda_assoc * assoc_scale * aux.assoc,
        "assoc_norm_w": cfg.lambda_assoc_norm * aux.assoc_norm,
        "gate_w": cfg.lambda_gate * aux.gate,
        "read_w": cfg.lambda_read * aux.read,
        "fusion_w": cfg.lambda_fusion * aux.fusion,
        "expert_w": cfg.lambda_expert * expert_scale * aux.expert,
        "ssm_w": cfg.lambda_ssm * aux.ssm,
        "slot_w": cfg.lambda_slot * aux.slot,
        # Schedule scales are plain Python floats — no device transfer.
        "assoc_scale": assoc_scale,
        "expert_scale": expert_scale,
    }


def _as_flush_scalar(value: Any, device: torch.device) -> torch.Tensor:
    """Coerce a logged metric to a 0-dim fp32 tensor on ``device``.

    Tensors are detached and cast (no sync); Python floats take an
    asynchronous H2D copy when stacked with their CUDA siblings.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=torch.float32).reshape(())
    return torch.tensor(value, device=device, dtype=torch.float32)


def _weighted_terms(
    model: HybridForCausalLM, out: Any, step: int, max_steps: int
) -> dict[str, float]:
    """Float view of :func:`_weighted_term_tensors`.

    Failure-path only (non-finite diagnosis); never called per-step.
    """
    return {
        name: float(value.item()) if isinstance(value, torch.Tensor) else value
        for name, value in _weighted_term_tensors(model, out, step, max_steps).items()
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
            "assoc_norm",
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
    """RNG states serialized weights_only-safely (primitives + tensors only).

    The NumPy Mersenne-Twister state contains an ndarray, which older
    ``weights_only=True`` unpicklers reject — it is stored as a plain int list.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),  # tuple of ints — pickle-safe
        "torch": torch.get_rng_state(),  # uint8 tensor
    }
    np_state = np.random.get_state()
    state["numpy"] = {
        "bit_generator": str(np_state[0]),
        "key": np.asarray(np_state[1]).astype(np.int64).tolist(),
        "pos": int(np_state[2]),
        "has_gauss": int(np_state[3]),
        "cached_gaussian": float(np_state[4]),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _load_rng_state_dict(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(_as_python_rng_state(state["python"]))
    if "numpy" in state:
        np.random.set_state(_as_numpy_rng_state(state["numpy"]))
    if "torch" in state:
        torch.set_rng_state(_as_torch_rng_state(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_torch_rng_state(s) for s in state["cuda"]])


def _as_python_rng_state(value: Any) -> Any:
    """Accept both legacy pickled tuples and primitive-only encodings."""
    if isinstance(value, dict):
        return (
            int(value["version"]),
            tuple(int(k) for k in value["keys"]),
            value.get("gauss_next"),
        )
    return value


def _as_numpy_rng_state(value: Any) -> Any:
    if isinstance(value, dict):
        key = np.asarray(value["key"], dtype=np.int64).astype(np.uint32)
        return (
            str(value["bit_generator"]),
            key,
            int(value["pos"]),
            int(value["has_gauss"]),
            float(value["cached_gaussian"]),
        )
    # Legacy checkpoints hold the raw ('MT19937', ndarray, ...) tuple.
    return value


def _as_torch_rng_state(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.uint8)
    return torch.tensor(value, dtype=torch.uint8)


def save_checkpoint(
    *,
    model: HybridForCausalLM,
    optimizers: list[optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    global_step: int,
    current_shard_idx: int,
    checkpoint_dir: Path,
    logger: logging.Logger,
    validator: WikiTextCyclicValidator | None = None,
    use_muon: bool = True,
    extra_payload: dict[str, Any] | None = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / CHECKPOINT_FILENAME
    tmp_path = ckpt_path.with_suffix(".pth.tmp")

    # In Muon mode optimizers=[muon, adam]; AdamW-only mode stores its single
    # optimizer ONLY under the adam_* keys (previously the same state was
    # duplicated under muon_* too, making cross-mode resume fail opaquely).
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": asdict(model.config),
        "global_step": global_step,
        "current_shard_idx": current_shard_idx,
        "rng_state": _rng_state_dict(),
        "memory_nan_fix_id": MEMORY_NAN_FIX_ID,
        "use_muon": use_muon,
    }
    if validator is not None:
        payload["validator_state_dict"] = validator.state_dict
    if use_muon:
        payload["muon_optimizer_state_dict"] = optimizers[0].state_dict()
        payload["muon_scheduler_state_dict"] = schedulers[0].state_dict()
        payload["adam_optimizer_state_dict"] = optimizers[1].state_dict()
        payload["adam_scheduler_state_dict"] = schedulers[1].state_dict()
    else:
        payload["adam_optimizer_state_dict"] = optimizers[0].state_dict()
        payload["adam_scheduler_state_dict"] = schedulers[0].state_dict()
    if extra_payload:
        payload.update(extra_payload)

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
    validator: WikiTextCyclicValidator | None = None,
    use_muon: bool | None = None,
    dl_generator: torch.Generator | None = None,
) -> tuple[int, int]:
    ckpt_path = checkpoint_dir / CHECKPOINT_FILENAME
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        # Checkpoints contain tensors + primitives only (see _rng_state_dict);
        # loading without pickle execution closes the arbitrary-code hole for
        # untrusted/shared checkpoint files.
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception as exc:  # noqa: BLE001 - legacy payloads need full pickle
        logger.warning(
            "weights_only load failed (%s); retrying with weights_only=False. "
            "Only resume from checkpoints you trust: pickle deserialization "
            "executes arbitrary code.",
            exc,
        )
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Optimizer-mode consistency: a clear error beats an opaque param-group
    # shape mismatch halfway through state restoration.
    ckpt_use_muon = bool(
        checkpoint.get("use_muon", "muon_optimizer_state_dict" in checkpoint)
    )
    has_explicit_flag = "use_muon" in checkpoint
    if use_muon is not None and has_explicit_flag and ckpt_use_muon != use_muon:
        raise RuntimeError(
            f"Checkpoint was saved with use_muon={ckpt_use_muon} but this run "
            f"uses use_muon={use_muon}; optimizer states are incompatible. "
            f"Start fresh or rerun with the matching --no-muon setting."
        )

    if ckpt_use_muon and "muon_optimizer_state_dict" in checkpoint:
        optimizers[0].load_state_dict(checkpoint["muon_optimizer_state_dict"])
        optimizers[-1].load_state_dict(checkpoint["adam_optimizer_state_dict"])
        schedulers[0].load_state_dict(checkpoint["muon_scheduler_state_dict"])
        schedulers[-1].load_state_dict(checkpoint["adam_scheduler_state_dict"])
    else:
        optimizers[0].load_state_dict(checkpoint["adam_optimizer_state_dict"])
        schedulers[0].load_state_dict(checkpoint["adam_scheduler_state_dict"])

    _load_rng_state_dict(checkpoint.get("rng_state"))

    if dl_generator is not None:
        gen_state = checkpoint.get("dl_generator_state")
        if gen_state is not None:
            dl_generator.set_state(_as_torch_rng_state(gen_state))

    if validator is not None and "validator_state_dict" in checkpoint:
        validator.load_state_dict(checkpoint["validator_state_dict"])

    # Scalar-config drift detection: resume rebuilds the config from CLI +
    # current code defaults, so any coefficient/knob that changed since the
    # checkpoint would silently continue training a different objective.
    saved_cfg = checkpoint.get("config")
    if isinstance(saved_cfg, dict):
        current_cfg = asdict(model.config)
        drifted = {
            k: {"checkpoint": saved_cfg[k], "current": current_cfg[k]}
            for k in current_cfg
            if k in saved_cfg and saved_cfg[k] != current_cfg[k]
        }
        if drifted:
            logger.warning(
                "Resumed config differs from the checkpoint payload in %d field(s): %s",
                len(drifted),
                drifted,
            )

    ckpt_fix_id = checkpoint.get("memory_nan_fix_id")
    if ckpt_fix_id != MEMORY_NAN_FIX_ID:
        logger.warning(
            "Checkpoint memory_nan_fix_id=%r differs from current %r — "
            "NaN-guard behavior changed between the two revisions.",
            ckpt_fix_id,
            MEMORY_NAN_FIX_ID,
        )

    global_step = int(checkpoint.get("global_step", 0))
    current_shard_idx = int(checkpoint.get("current_shard_idx", 0))
    logger.info(
        "Resumed from %s | step=%d shard=%d fix_id=%s use_muon=%s",
        ckpt_path,
        global_step,
        current_shard_idx,
        ckpt_fix_id if ckpt_fix_id is not None else "unknown",
        ckpt_use_muon,
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
    val_ce = record.get("val_ce_loss")
    val_str = f"{val_ce:.6f}" if val_ce is not None else "n/a"
    return (
        f"step={step}/{max_steps} "
        f"shard={record.get('shard_idx', 0)} "
        f"loss={record['loss']:.6f} ce={record['ce_loss']:.6f} "
        f"ce_smooth={smooth_str} val_ce={val_str} "
        f"router_aux={record['router_aux_loss']:.6f} "
        f"router_z={record['router_z_loss']:.6f} "
        f"recon={record['recon']:.6f} assoc={record['assoc']:.6f}({assoc_tag}) "
        f"expert={record['expert']:.6f}({expert_tag}) "
        f"grad_norm={record['grad_norm']:.4f} "
        f"muon_lr={record['muon_lr']:.2e} adam_lr={record['adam_lr']:.2e}"
    )


def _format_val_log_line(val_record: dict[str, Any]) -> str:
    return (
        f"validation step={val_record['step']} "
        f"val_loss={val_record['val_loss']:.6f} "
        f"val_ce={val_record['val_ce_loss']:.6f} "
        f"val_router_aux={val_record['val_router_aux_loss']:.6f} "
        f"val_router_z={val_record['val_router_z_loss']:.6f} "
        + (
            f"windows={val_record.get('val_windows', '?')} "
            f"eval_rows={val_record.get('val_eval_rows', '?')} "
            f"batch={val_record.get('val_batch_size', '?')}"
            if val_record.get("mode", "rows") == "packed"
            else (
                f"rows={val_record.get('val_rows', '?')} "
                f"batch={val_record.get('val_batch_size', '?')} "
                f"cursor={val_record.get('val_row_start', '?')}->"
                f"{val_record.get('val_row_end', '?')} "
                f"next={val_record.get('val_cursor', '?')}"
            )
        )
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
    # fp16 autocast needs loss scaling (gradients silently underflow to zero
    # otherwise); bf16 has the same exponent range as fp32 and must NOT be
    # scaled. With enabled=False GradScaler is a pass-through, so the same
    # call sites work for both dtypes and fp32.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(use_amp and amp_dtype == torch.float16)
        )
    else:  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(
            enabled=(use_amp and amp_dtype == torch.float16)
        )
    use_fp16_scaler = scaler.is_enabled()

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
        seed=args.seed,
        log_fn=logger.info,
    )
    verify_tokenizer_vocab(producer.tokenizer, args.vocab_size)

    cfg = build_training_config(vocab_size=args.vocab_size)
    # `is not None` (not `or`): a legitimate token id of 0 must not be
    # clobbered by the config default.
    if producer.tokenizer.bos_token_id is not None:
        cfg.bos_token_id = producer.tokenizer.bos_token_id
    if producer.tokenizer.eos_token_id is not None:
        cfg.eos_token_id = producer.tokenizer.eos_token_id
    if args.compile:
        cfg.use_torch_compile = True
        cfg.torch_compile_mode = args.compile_mode
    if args.gradient_checkpointing:
        # NOTE: mutually exclusive with --compile (HybridModel.__init__ warns
        # and keeps compile, dropping checkpointing). Checkpointing wraps the
        # FULL HybridDecoderLayer forward (memory R/W, GQA, Mamba, fusion, MoE)
        # via use_reentrant=False and suppresses the Mamba internal scan
        # checkpoint to avoid double checkpointing. Mathematically neutral:
        # outputs/losses/gradients are identical modulo CUDA kernel
        # non-determinism (index_add atomics) that already exists without it.
        cfg.gradient_checkpointing = True

    logger.info(log_mamba_backend(cfg))
    logger.info(
        "device=%s amp=%s dtype=%s fused_mamba=%s memory_nan_fix=%s",
        device,
        use_amp,
        amp_dtype if use_amp else "fp32",
        fused_mamba_scan_available(),
        MEMORY_NAN_FIX_ID,
    )
    logger.info(
        "gradient_checkpointing=%s mamba_internal_checkpoint=%s "
        "chunked_ce=%s return_logits=%s memory_debug=%s",
        cfg.gradient_checkpointing,
        cfg.mamba_internal_checkpoint,
        cfg.stream_chunked_ce_loss,
        cfg.return_logits,
        args.memory_debug,
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

    validator: WikiTextCyclicValidator | None = None
    if not args.no_validation:
        try:
            validator = WikiTextCyclicValidator(
                producer.tokenizer,
                seq_len=args.seq_len,
                num_rows=args.val_rows,
                batch_size=args.val_batch_size,
                dataset_config=args.val_dataset_config,
                bos_id=cfg.bos_token_id,
                eos_id=cfg.eos_token_id,
                pad_token_id=cfg.pad_token_id,
                mode=args.val_mode,
                eval_rows=args.val_eval_rows,
            )
            logger.info(
                "Cyclic validation enabled | dataset=Salesforce/wikitext/%s "
                "mode=%s rows=%d batch=%d interval=%d%s",
                args.val_dataset_config,
                args.val_mode,
                args.val_rows,
                args.val_batch_size,
                args.val_interval,
                (
                    ""
                    if args.val_mode == "rows"
                    else f" eval_rows={args.val_eval_rows} (fixed, non-rotating)"
                ),
            )
        except Exception:
            logger.exception(
                "Failed to initialize WikiTextCyclicValidator; disabling validation"
            )
            validator = None

    warmup_steps = _resolve_warmup_steps(args.warmup_steps, args.max_steps)
    lr_lambda = _build_lr_lambda(warmup_steps, args.max_steps, args.min_lr_ratio)
    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = [
        torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        for opt in optimizers
    ]

    global_step = 0
    current_shard_idx = 0
    ckpt_dir = Path(args.ckpt_dir)

    # Shuffle generator for the DataLoader; persisted in the checkpoint so
    # intra-shard permutations replay identically after --resume.
    dl_generator = torch.Generator()
    dl_generator.manual_seed(args.seed)

    if args.resume:
        global_step, current_shard_idx = load_checkpoint(
            model=model,
            optimizers=optimizers,
            schedulers=schedulers,
            checkpoint_dir=ckpt_dir,
            device=device,
            logger=logger,
            validator=validator,
            use_muon=use_muon,
            dl_generator=dl_generator,
        )
        # LambdaLR.load_state_dict restores `_last_lr` but not
        # param_group['lr'], and opt.step() precedes sched.step() — without
        # this sync the first post-resume step would run at constructor LR.
        for opt, sched in zip(optimizers, schedulers):
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lr_lambda(max(sched.last_epoch, 0))

    stop_event = threading.Event()
    producer_thread: threading.Thread | None = None
    ce_smooth = RollingAverage(args.smooth_window)
    # GPU-side metric accumulators: per-step scalars are summed on-device and
    # pulled to the host with ONE transfer every --log-interval steps instead
    # of ~25 blocking .item() round-trips per step (each would synchronize
    # against the whole enqueued stream).
    metric_sum: torch.Tensor | None = None
    metric_fin_cnt: torch.Tensor | None = None
    metric_bad_cnt: torch.Tensor | None = None
    metric_count = 0
    assoc_scale_sum = 0.0
    expert_scale_sum = 0.0
    metric_main_names: list[str] | None = None
    metric_gate_names: list[str] | None = None
    spike_ema: EMABaseline | None = None
    reset_mamba_scan_stats()

    try:
        # NOTE: the trainer's *consume* cursor (`current_shard_idx`, restored
        # above from the model checkpoint) stays authoritative. Do NOT copy
        # `producer.current_shard_idx` (the production/write cursor) over it:
        # the producer runs up to max_buffered_files shards ahead, so doing so
        # silently skipped every buffered-but-untrained shard on resume.
        if args.resume and os.path.exists(args.dataset_ckpt_path):
            producer.load_checkpoint(args.dataset_ckpt_path)

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

        while global_step < args.max_steps:
            shard_name = f"shard_{current_shard_idx:06d}.bin"
            bin_path = os.path.join(args.cache_dir, shard_name)
            # The producer writes shard.bin atomically (tmp+replace), then
            # publishes shard.json. Gating on BOTH closes the torn-shard race:
            # a bare existing .bin could once be caught mid-write.
            sidecar_path = os.path.join(
                args.cache_dir, shard_name.replace(".bin", ".json")
            )
            wait_start = time.time()

            while not (os.path.exists(bin_path) and os.path.exists(sidecar_path)):
                # Surface producer-thread failures instead of spinning out to
                # a misleading success exit.
                producer_error = getattr(producer, "error", None)
                if producer_error is not None:
                    raise RuntimeError(
                        f"Shard producer thread failed while waiting for "
                        f"{shard_name}: {producer_error!r}"
                    ) from (
                        producer_error
                        if isinstance(producer_error, BaseException)
                        else None
                    )
                if time.time() - wait_start > args.producer_wait_timeout:
                    raise RuntimeError(
                        f"Timed out waiting for {shard_name} after "
                        f"{args.producer_wait_timeout}s (producer stalled or "
                        f"starved) — aborting with a non-zero status instead "
                        f"of reporting success."
                    )
                time.sleep(1.0)

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

            shard_fully_consumed = True
            for batches_in_shard, (input_ids, labels) in enumerate(dataloader):
                if global_step >= args.max_steps:
                    # Hitting max-steps mid-shard: leave the shard unmarked
                    # and the cursor where it is so a later --max-steps
                    # extension resumes from this exact batch instead of
                    # silently dropping the unconsumed tail.
                    shard_fully_consumed = False
                    break

                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # The min/max reductions force a device sync each; shards are
                # immutable memmaps, so checking shard starts plus a periodic
                # sample catches corrupt files / vocab drift at ~1/256th of
                # the pipeline cost. --validate-token-interval 1 restores
                # every-batch checking.
                if batches_in_shard == 0 or (
                    args.validate_token_interval > 0
                    and global_step % args.validate_token_interval == 0
                ):
                    validate_token_batch(
                        input_ids,
                        cfg.vocab_size,
                        labels=labels,
                        ignore_index=cfg.label_ignore_index,
                    )

                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)

                if args.memory_debug and device.type == "cuda":
                    logger.debug(
                        "[mem] step=%d pre-forward alloc=%.2fGB reserved=%.2fGB "
                        "peak=%.2fGB",
                        global_step,
                        torch.cuda.memory_allocated() / 2**30,
                        torch.cuda.memory_reserved() / 2**30,
                        torch.cuda.max_memory_allocated() / 2**30,
                    )

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

                if args.memory_debug and device.type == "cuda":
                    logger.debug(
                        "[mem] step=%d post-forward alloc=%.2fGB reserved=%.2fGB "
                        "peak=%.2fGB",
                        global_step,
                        torch.cuda.memory_allocated() / 2**30,
                        torch.cuda.memory_reserved() / 2**30,
                        torch.cuda.max_memory_allocated() / 2**30,
                    )

                # Non-finiteness is handled entirely ON-DEVICE (counters +
                # masking in the metric accumulation below, plus grad
                # sanitation after clipping). A NaN/Inf loss or gradient no
                # longer branches on the host mid-step: branching on a CUDA
                # scalar (.item() / __bool__ / error_if_nonfinite=True) would
                # force a stream-syncing D2H copy EVERY step. The cost is a
                # wasted backward pass on an already-failed step.
                if use_fp16_scaler:
                    scaler.scale(outputs.loss).backward()
                else:
                    outputs.loss.backward()

                if args.memory_debug and device.type == "cuda":
                    logger.debug(
                        "[mem] step=%d post-backward alloc=%.2fGB reserved=%.2fGB "
                        "peak=%.2fGB",
                        global_step,
                        torch.cuda.memory_allocated() / 2**30,
                        torch.cuda.memory_reserved() / 2**30,
                        torch.cuda.max_memory_allocated() / 2**30,
                    )

                # Unscale before clipping so max_grad_norm applies to true
                # gradient magnitudes, not scaled ones.
                if use_fp16_scaler:
                    for opt in optimizers:
                        scaler.unscale_(opt)
                # No .item(): the raw norm tensor feeds the metric vector,
                # and non-finite norms surface via the flush-time counter.
                # (Exception: --grad-nan-guard=strict deliberately reads it
                # every step — see below.)
                grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
                allow_update = True
                if args.grad_nan_guard == "strict":
                    # The ONE deliberate per-step D2H sync in this mode
                    # (bool() on a CUDA scalar) buys TRUE skip semantics: a
                    # bad-grad step executes no optimizer update, no scheduler
                    # tick, and mutates no Adam/Muon moments. This is exactly
                    # what 'sanitize' approximates asynchronously — pick this
                    # mode only if the per-step sync stall is acceptable.
                    # Redundant but harmless on the fp16 GradScaler path,
                    # which already syncs inside scaler.step().
                    allow_update = bool(torch.isfinite(grad_norm))
                    if not allow_update:
                        logger.error(
                            "Non-finite gradient norm at step %d — optimizer "
                            "update SKIPPED (--grad-nan-guard=strict)",
                            global_step,
                        )
                        if jsonl_path is not None:
                            with jsonl_path.open("a", encoding="utf-8") as f:
                                f.write(
                                    json.dumps(
                                        {
                                            "event": "non_finite_grad",
                                            "step": global_step,
                                            "shard_idx": current_shard_idx,
                                            **_non_finite_diagnosis(
                                                model,
                                                outputs,
                                                global_step,
                                                args.max_steps,
                                            ),
                                        }
                                    )
                                    + "\n"
                                )
                elif args.grad_nan_guard == "sanitize" and not use_fp16_scaler:
                    # A non-finite norm makes clip's coef NaN, which spreads
                    # NaN across EVERY gradient (norm-bad <=> some grad bad);
                    # zero them so the optimizer update stays finite instead
                    # of permanently poisoning Adam/Muon moments (M10
                    # protection, now async). Cost: one read+write pass over
                    # all grads per step (~sub-ms at production scale) — the
                    # price of a hard guarantee without a host branch. The
                    # GradScaler path must NOT be pre-sanitized: scaler.step()
                    # only skips the update and shrinks the scale when it
                    # still sees inf/nan itself.
                    for p in model.parameters():
                        if p.grad is not None:
                            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

                if allow_update:
                    for i, opt in enumerate(optimizers):
                        if use_fp16_scaler:
                            # Internally no-ops when unscaled grads contain
                            # inf/nan; update() then shrinks the scale.
                            scaler.step(opt)
                        else:
                            opt.step()
                    if use_fp16_scaler:
                        scaler.update()
                    for sched in schedulers:
                        sched.step()
                # When strict skipped the update, the NaN grads are freed by
                # the zero_grad(set_to_none=True) at the top of the next
                # iteration — they never reach an optimizer or accumulate.

                if args.memory_debug and device.type == "cuda":
                    logger.debug(
                        "[mem] step=%d post-opt alloc=%.2fGB reserved=%.2fGB "
                        "peak=%.2fGB",
                        global_step,
                        torch.cuda.memory_allocated() / 2**30,
                        torch.cuda.memory_reserved() / 2**30,
                        torch.cuda.max_memory_allocated() / 2**30,
                    )
                    torch.cuda.reset_peak_memory_stats()

                aux = outputs.auxiliary_losses
                assert aux is not None

                # ---- metrics: accumulate on-device, transfer on flush ------
                # Every logged scalar (losses, aux terms, weighted terms,
                # gate stats, schedule scales) is summed into one fp32 vector
                # on the compute device. Nothing here synchronizes; the only
                # host readback happens in the flush below, once per
                # --log-interval steps.
                weighted_t = _weighted_term_tensors(
                    model, outputs, global_step, args.max_steps
                )
                scalars: dict[str, Any] = {
                    "loss": outputs.loss,
                    "ce_loss": (
                        outputs.ce_loss
                        if outputs.ce_loss is not None
                        else outputs.loss.new_zeros(())
                    ),
                    "router_aux_loss": (
                        outputs.router_aux_loss
                        if outputs.router_aux_loss is not None
                        else outputs.loss.new_zeros(())
                    ),
                    "router_z_loss": (
                        outputs.router_z_loss
                        if outputs.router_z_loss is not None
                        else outputs.loss.new_zeros(())
                    ),
                    "grad_norm": grad_norm,
                }
                # Schedule scales are pure-Python values: averaging them
                # host-side avoids two device-tensor constructions (H2D)
                # every step for numbers the GPU never needs.
                assoc_scale_sum += weighted_t["assoc_scale"]
                expert_scale_sum += weighted_t["expert_scale"]
                for name in (
                    "recon",
                    "assoc",
                    "gate",
                    "read",
                    "fusion",
                    "expert",
                    "ssm",
                    "slot",
                    "assoc_norm",
                ):
                    scalars[name] = getattr(aux, name)
                for name, value in weighted_t.items():
                    if isinstance(value, torch.Tensor):
                        scalars[name] = value

                gate_stats = outputs.gate_stats or {}
                # Schema is static for a fixed config; reuse the cached
                # name lists instead of rebuilding them every step (the
                # equality check below still guards against drift).
                main_names = (
                    metric_main_names
                    if metric_main_names is not None
                    else list(scalars)
                )
                gate_names = (
                    metric_gate_names
                    if metric_gate_names is not None
                    else list(gate_stats)
                )
                step_vec = torch.stack(
                    [
                        _as_flush_scalar(scalars[n], outputs.loss.device)
                        for n in main_names
                    ]
                    + [
                        _as_flush_scalar(gate_stats[n], outputs.loss.device)
                        for n in gate_names
                    ]
                )
                # Non-finite entries are masked out of the running sums and
                # counted per name instead: instability is recorded without
                # ever branching on a device scalar mid-step.
                finite = torch.isfinite(step_vec)
                # Magnitude-based spike guard ("huge but finite"): a finite
                # value exceeding --loss-spike-multiplier * (EMA of recent
                # finite values) is treated like non-finite — masked from the
                # window mean, counted in bad_cnt, and reported at flush.
                # Entirely on-device; no extra syncs vs the finite-only path.
                spike_mask: torch.Tensor | None = None
                if args.loss_spike_multiplier > 0.0:
                    if (
                        spike_ema is None
                        or metric_main_names != main_names
                        or (metric_gate_names != gate_names)
                    ):
                        spike_ema = EMABaseline(
                            step_vec.numel(),
                            step_vec.device,
                            args.loss_spike_multiplier,
                        )
                    step_spike = spike_ema.update(step_vec, finite)
                    finite = finite & ~step_spike
                    spike_mask = step_spike
                if (
                    metric_sum is None
                    or metric_main_names != main_names
                    or metric_gate_names != gate_names
                ):
                    # First step (or schema drift): (re)initialize.
                    metric_main_names, metric_gate_names = main_names, gate_names
                    metric_sum = torch.zeros_like(step_vec)
                    metric_fin_cnt = torch.zeros_like(step_vec)
                    metric_bad_cnt = torch.zeros_like(step_vec)
                    metric_count = 0
                    if spike_ema is not None:
                        spike_ema.reset()
                metric_sum += torch.where(finite, step_vec, 0.0)
                metric_fin_cnt += finite.to(step_vec.dtype)
                metric_bad_cnt += (~finite).to(step_vec.dtype)
                metric_count += 1

                if global_step % args.log_interval == 0:
                    # The window's single deliberate metrics GPU→CPU sync.
                    assert (
                        metric_sum is not None
                        and metric_fin_cnt is not None
                        and metric_bad_cnt is not None
                        and metric_main_names is not None
                        and metric_gate_names is not None
                    )
                    # Masked mean over steps where each entry was finite (or
                    # within the spike guard's magnitude band); means and
                    # bad-counts ride ONE host transfer.
                    means_vec = metric_sum / metric_fin_cnt.clamp(min=1.0)
                    # The spike EMA rides the same single transfer (it is
                    # host-consumed only at flush, like everything else).
                    ema_values: list[float] | None = None
                    if spike_ema is not None:
                        ema_values = spike_ema.ema.tolist()
                    flush_values = torch.cat([means_vec, metric_bad_cnt]).tolist()
                    values = flush_values[: means_vec.numel()]
                    bad_counts = flush_values[means_vec.numel() :]
                    n_main = len(metric_main_names)
                    metrics = dict(zip(metric_main_names, values[:n_main]))
                    ce_smooth.update(metrics["ce_loss"])
                    window = max(metric_count, 1)
                    record: dict[str, Any] = {
                        "step": global_step,
                        "shard_idx": current_shard_idx,
                        **metrics,
                        "assoc_scale": assoc_scale_sum / window,
                        "expert_scale": expert_scale_sum / window,
                        "ce_smooth": ce_smooth.mean,
                        "muon_lr": float(schedulers[0].get_last_lr()[0]),
                        "adam_lr": float(schedulers[-1].get_last_lr()[0]),
                        "gate_stats": dict(zip(metric_gate_names, values[n_main:])),
                    }
                    bad_names = {
                        name: int(cnt)
                        for name, cnt in list(
                            zip(metric_main_names, bad_counts[:n_main])
                        )
                        + list(zip(metric_gate_names, bad_counts[n_main:]))
                        if cnt > 0.0
                    }
                    if bad_names:
                        # Aggregated replacement for the old per-step
                        # non_finite_loss/non_finite_grad events; diagnosis
                        # reflects the last step of the window.
                        diagnosis = _non_finite_diagnosis(
                            model, outputs, global_step, args.max_steps
                        )
                        logger.error(
                            "Non-finite/spiked metrics in steps %d-%d: %s",
                            global_step - metric_count + 1,
                            global_step,
                            bad_names,
                        )
                        if jsonl_path is not None:
                            with jsonl_path.open("a", encoding="utf-8") as f:
                                f.write(
                                    json.dumps(
                                        {
                                            "event": "non_finite_metrics",
                                            "step": global_step,
                                            "shard_idx": current_shard_idx,
                                            "window_steps": metric_count,
                                            "counts": bad_names,
                                            **diagnosis,
                                        }
                                    )
                                    + "\n"
                                )
                                # Separate magnitude-spike event: names here may be
                                # huge-but-FINITE values that the non-finite
                                # diagnosis above cannot see. One event per window
                                # listing which metrics spiked and their EMA
                                # baselines at flush time.
                                if spike_mask is not None and spike_mask.any():
                                    all_names = list(metric_main_names) + list(
                                        metric_gate_names
                                    )
                                    spike_host = [bool(s) for s in spike_mask.tolist()]
                                    ema_host = ema_values or [0.0] * len(all_names)
                                    spike_names: dict[str, float] = {}
                                    baselines: dict[str, float] = {}
                                    for idx, (name, spiked) in enumerate(
                                        zip(all_names, spike_host)
                                    ):
                                        if spiked:
                                            spike_names[name] = float(bad_counts[idx])
                                            baselines[name] = float(ema_host[idx])
                                    f.write(
                                        json.dumps(
                                            {
                                                "event": "loss_spike",
                                                "step": global_step,
                                                "shard_idx": current_shard_idx,
                                                "window_steps": metric_count,
                                                "multiplier": args.loss_spike_multiplier,
                                                "spiked": spike_names,
                                                "ema_baseline": baselines,
                                                **diagnosis,
                                            }
                                        )
                                        + "\n"
                                    )
                    metric_sum = None
                    metric_fin_cnt = None
                    metric_bad_cnt = None
                    metric_count = 0
                    assoc_scale_sum = 0.0
                    expert_scale_sum = 0.0

                    # JSONL records and console lines are emitted per flush
                    # window (means over --log-interval steps), matching the
                    # single-transfer cadence rather than per-step.
                    if (
                        validator is not None
                        and args.val_interval > 0
                        and global_step % args.val_interval == 0
                    ):
                        val_record = validator.evaluate(
                            model,
                            device=device,
                            global_step=global_step,
                            max_training_steps=args.max_steps,
                            use_amp=use_amp,
                            amp_dtype=amp_dtype,
                            ignore_index=cfg.label_ignore_index,
                        )
                        record["val_loss"] = val_record["val_loss"]
                        record["val_ce_loss"] = val_record["val_ce_loss"]
                        record["val_router_aux_loss"] = val_record[
                            "val_router_aux_loss"
                        ]
                        record["val_router_z_loss"] = val_record["val_router_z_loss"]

                        if jsonl_path is not None:
                            with jsonl_path.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(val_record) + "\n")

                        logger.info(_format_val_log_line(val_record))

                    if jsonl_path is not None:
                        with jsonl_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(record) + "\n")

                    logger.info(_format_log_line(global_step, args.max_steps, record))

                if global_step % args.save_interval == 0 and global_step > 0:
                    save_checkpoint(
                        model=model,
                        optimizers=optimizers,
                        schedulers=schedulers,
                        global_step=global_step,
                        current_shard_idx=current_shard_idx,
                        checkpoint_dir=ckpt_dir,
                        logger=logger,
                        validator=validator,
                        use_muon=use_muon,
                        extra_payload={
                            "dl_generator_state": dl_generator.get_state(),
                            "muon_adjust_lr_fn": _opt_meta.get("muon_adjust_lr_fn"),
                        },
                    )
                    producer.save_checkpoint(args.dataset_ckpt_path)

                # Convention: global_step counts iterations ENTERED, so the
                # first completed update logs as step=0 ("updates completed
                # before this iteration"). Every consumer — assoc/expert
                # warmups, log/validation/save cadence, checkpoint contents,
                # resume — shares this convention consistently; do not shift
                # it without migrating checkpoints and scripts/
                # test_cloud_train.py in lockstep.
                global_step += 1

            if not shard_fully_consumed:
                break

            done_path = os.path.join(
                args.cache_dir, f"shard_{current_shard_idx:06d}.done"
            )
            with open(done_path, "w", encoding="utf-8") as f:
                f.write("done\n")
            logger.info("Marked shard %d complete", current_shard_idx)
            current_shard_idx += 1

        if global_step > 0 and global_step % args.save_interval != 0:
            save_checkpoint(
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                global_step=global_step,
                current_shard_idx=current_shard_idx,
                checkpoint_dir=ckpt_dir,
                logger=logger,
                validator=validator,
                use_muon=use_muon,
                extra_payload={
                    "dl_generator_state": dl_generator.get_state(),
                    "muon_adjust_lr_fn": _opt_meta.get("muon_adjust_lr_fn"),
                },
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
            save_checkpoint(
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                global_step=global_step,
                current_shard_idx=current_shard_idx,
                checkpoint_dir=ckpt_dir,
                logger=logger,
                validator=validator,
                use_muon=use_muon,
                extra_payload={
                    "dl_generator_state": dl_generator.get_state(),
                    "muon_adjust_lr_fn": _opt_meta.get("muon_adjust_lr_fn"),
                },
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
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Checkpoint each HybridDecoderLayer forward (use_reentrant=False): "
        "activations are recomputed in backward instead of retained, cutting "
        "peak VRAM substantially at ~+30%% step time. Mutually exclusive with "
        "--compile. Outputs/losses/gradients unchanged.",
    )
    parser.add_argument(
        "--memory-debug",
        action="store_true",
        help="Log torch.cuda memory allocated/reserved/max at step boundaries "
        "(forward, backward, optimizer). Zero effect when disabled; CPU-only "
        "runs log CPU RSS instead of CUDA stats.",
    )

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
        "--validate-token-interval",
        type=int,
        default=256,
        help="Check token-id ranges on each shard's first batch and every Nth "
        "step (1 = every batch; 0 = shard starts only). Each check forces a "
        "GPU sync for the min/max reduction, so per-batch checking stalls "
        "the CPU/GPU pipeline in the hot loop.",
    )
    parser.add_argument(
        "--grad-nan-guard",
        choices=("sanitize", "strict", "monitor"),
        default="sanitize",
        help="Non-finite-gradient handling on the bf16/fp32 path (fp16's "
        "GradScaler always has its own found_inf skip). Of {no per-step "
        "sync, true optimizer-skip, low complexity} you can pick any two: "
        "'sanitize' (default): zero NaN/Inf grads after clipping EVERY "
        "step — no sync, params can never be poisoned; the price is one "
        "read+write pass over all grads (~0.5-1%% of step time at 148M "
        "params, benchmark with scripts/bench_grad_guard.py) and a "
        "skipped-bad step still applying bounded momentum-decay/"
        "weight-decay drift instead of being a perfect no-op. 'strict': "
        "one deliberate D2H sync EVERY step in exchange for TRUE skip "
        "semantics — a bad-grad step runs no optimizer or scheduler "
        "update and mutates no optimizer state. 'monitor': zero overhead "
        "and zero protection — NaN grads reach opt.step() and poison "
        "Adam/Muon moments permanently; flush-time counters report the "
        "damage, but treat the affected run as dead-from-that-point and "
        "roll back to the last good checkpoint.",
    )
    parser.add_argument(
        "--smooth-window", type=int, default=50, help="Rolling CE mean window."
    )
    parser.add_argument(
        "--loss-spike-multiplier",
        type=float,
        default=100.0,
        help="Magnitude-based spike guard: a finite metric exceeding "
        "K * (EMA of its recent finite values) is treated like a non-finite "
        "value (masked out of window means, counted in metric_bad_cnt) and "
        "reported via a 'loss_spike' JSONL event. 0 disables. Covers loss, "
        "grad_norm, and every aux term. Purely on-device; no extra syncs.",
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

    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable cyclic Salesforce/wikitext validation.",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=200,
        help="Run cyclic validation every N training steps.",
    )
    parser.add_argument(
        "--val-rows",
        type=int,
        default=50,
        help="Number of wikitext validation rows consumed per validation call.",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=10,
        help="Batch size used when scoring validation rows.",
    )
    parser.add_argument(
        "--val-dataset-config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Salesforce/wikitext config name (e.g. wikitext-2-raw-v1, wikitext-103-raw-v1).",
    )
    parser.add_argument(
        "--val-mode",
        choices=("packed", "rows"),
        default="packed",
        help="'packed' (default): fixed, non-rotating eval slice tokenized into "
        "full seq_len windows with cross-document left context — matches "
        "training packing; deterministic across calls. 'rows' (legacy): each "
        "wikitext row scored independently (short rows inflate the loss) with "
        "a rotating num_rows cursor; kept for comparison with old metrics.",
    )
    parser.add_argument(
        "--val-eval-rows",
        type=int,
        default=500,
        help="Number of leading non-empty validation rows packed into the "
        "fixed eval set in --val-mode=packed (larger = lower sampling "
        "variance, slower per call).",
    )
    args = parser.parse_args()
    if not args.log_jsonl:
        args.log_jsonl = str(Path(args.run_dir) / "metrics.jsonl")
    return args


def main() -> None:
    args = parse_args()
    if args.deterministic:
        # Must be set before the first cuBLAS handle is created (i.e. before
        # any CUDA tensor op), so it lives here rather than in set_seed(),
        # which runs after imports may have touched CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    logger = setup_logging(Path(args.run_dir))
    logger.info("Starting training with args: %s", vars(args))
    train(args, logger)


if __name__ == "__main__":
    main()
