"""TPU SPMD distributed pre-training for Hybrid Mamba-MoE.

Targeted for cloud TPU environments (specifically Kaggle TPU v5e-8) using
PyTorch/XLA SPMD (Single Program, Multiple Data) execution.

Execution Architecture:
  * Single-process launcher: a single Python process drives the entire
    TPU mesh. No multi-process torchrun or xmp.spawn is used.
  * Runtime initialization: xr.use_spmd() enables GSPMD compiler
    transformations before any tensor/device initialization.
  * Device mesh: 1D ('data') or 2D ('data', 'model') mesh dynamically sized
    to xr.global_runtime_device_count() (e.g. 8 for TPU v5e-8).
  * Input sharding: batches are partitioned along dimension 0 across the
    'data' axis using xs.mark_sharding (and/or pl.MpDeviceLoader), ensuring
    each TPU core processes batch_size // num_devices sequences.
  * Parameter placement & sharding:
    - In default 'data_parallel' mode: model weights are replicated across
      cores; XLA compiler automatically reduces gradients across the mesh.
    - In optional 'fsdp' mode: large 2D parameter matrices (attention,
      mamba, MLP/MoE expert weights, LM head) are sharded along dim 0.
      Custom direct-math parameters (RMSNorm, Mamba state vectors A_log/D,
      dual-memory combine projections, memory banks) remain replicated to
      avoid DTensor recomputation issues.
  * Precision: Native bfloat16 via torch.autocast(device_type='xla', dtype=torch.bfloat16).
    No CUDA GradScaler or fp16 loss scaling (bfloat16 has full fp32 dynamic range).
  * Mamba selective scan: PyTorch Hillis-Steele parallel scan or blocked scan
    (fused CUDA kernels disabled since mamba-ssm is CUDA-only).
  * Optimizer & Scheduler:
    - Muon + AdamW hybrid (or AdamW-only fallback via --no-muon).
    - AdamW runs with fused=False (fused AdamW is CUDA-only).
    - Cosine LR schedule with warmup and min_lr_ratio floor.
  * Gradient accumulation & XLA execution:
    - Micro-batches accumulated on-device.
    - clip_grad_norm_ executed on XLA device.
    - --grad-nan-guard=sanitize zeroes out NaNs/Infs on device with ZERO host sync.
    - xm.mark_step() called after optimizer/scheduler update to execute step graph.
  * Metrics & Logging:
    - Host-sync-free on-device accumulation of metrics and gate stats.
    - Single D2H flush transfer every --log-interval steps.
    - Master-safe logging to console, train.log, and metrics.jsonl.
  * Checkpoint & Resume:
    - Atomic single-file checkpoints (.pth) via xm.save (transfers XLA tensors to CPU
      safely before writing).
    - Comprehensive state: model, optimizers, schedulers, global step, shard index,
      RNG states (Python, NumPy, PyTorch CPU, XLA), dl_generator, validator, config.
    - True resume contract validation.
  * Validation:
    - Cyclic WikiText and PackedWindow evaluation.
    - Validation batches partitioned across TPU mesh with drop_last=True to guarantee
      uniform tensor shapes across TPU cores.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Safe PyTorch/XLA import guard: allows static analysis, linting, CLI tests,
# and code-structure verification on non-TPU dev laptops without error.
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    from torch_xla.distributed.spmd import Mesh

    XLA_AVAILABLE = True
except ImportError:
    torch_xla = None
    xm = None
    pl = None
    xs = None
    xr = None
    Mesh = None
    XLA_AVAILABLE = False

from model.core.builders import count_trainable_params
from model.core.config import HybridMambaMoEConfig
from model.core.constants import MEMORY_NAN_FIX_ID
from model.core.optim import (
    _is_adamw_no_decay,
    split_muon_adam_params,
)
from model.hybrid.layer import HybridDecoderLayer
from model.hybrid.losses import _aux_loss_schedule, _expert_loss_schedule
from model.hybrid.mamba import (
    MambaBlock,
    fused_mamba_scan_available,
    get_mamba_scan_stats,
    log_mamba_backend,
    reset_mamba_scan_stats,
)
from model.hybrid.memory import CompressiveMemoryBank
from model.hybrid.model import HybridForCausalLM
from model.layers.moe import DroplessMoELayer
from model.layers.norm import RMSNorm
from utils.dataset import (
    MmapShardDataset,
    TokenizedShardProducer,
    verify_tokenizer_vocab,
)
from utils.training_logging import format_training_log_line as _format_log_line
from utils.validation import (
    PackedWindowValidator,
    WikiTextCyclicValidator,
    build_causal_labels,
)

CHECKPOINT_FILENAME = "model_ckpt.pth"
CONFIG_FILENAME = "config.json"
TPU_SPMD_CHECKPOINT_FAMILY = "tpu_spmd"
TPU_SPMD_OPTIMIZER_POLICY_MUON = "muon_adamw"
TPU_SPMD_OPTIMIZER_POLICY_ADAM = "adamw"


# ---------------------------------------------------------------------------
# Seeding & Logging
# ---------------------------------------------------------------------------


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, PyTorch (CPU), and PyTorch/XLA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if XLA_AVAILABLE and xm is not None:
        try:
            xm.set_rng_state(seed)
        except Exception:  # noqa: BLE001, S110
            pass

    if deterministic and hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def setup_logging(run_dir: Path, *, level: int = logging.INFO) -> logging.Logger:
    """Console + rotating run log under ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tpu_spmd_train")
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
# Model, Configuration & SPMD Sharding
# ---------------------------------------------------------------------------


def build_training_config(vocab_size: int) -> HybridMambaMoEConfig:
    """Default production Hybrid config (~511M trainable params, measured).

    use_fused_mamba_scan is set to False on TPU because mamba-ssm is a CUDA-only
    kernel. The PyTorch native Hillis-Steele parallel scan or blocked scan is
    used instead, which compiles to high-performance TPU XLA HLO.
    """
    hidden_size = 768
    num_heads = 12
    head_dim = 64
    return HybridMambaMoEConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=12,
        num_heads=num_heads,
        num_kv_heads=4,
        head_dim=head_dim,
        intermediate_size=768,
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
        use_fused_mamba_scan=False,  # Disabled on TPU: mamba-ssm is CUDA-only
    )


def init_tpu_spmd(logger: logging.Logger) -> tuple[torch.device, int]:
    """Initialize PyTorch/XLA SPMD runtime and determine device count dynamically."""
    if not XLA_AVAILABLE or xr is None or xm is None:
        raise RuntimeError(
            "PyTorch/XLA is not available in the current environment. "
            "pre-training/tpu_smpd_train.py must be executed in a TPU environment "
            "(e.g. Kaggle TPU v5e-8) where torch_xla is installed. "
            "DO NOT run TPU training locally on development machines."
        )

    # xr.use_spmd() MUST be called before any XLA tensor or device creation
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    if num_devices < 1:
        raise RuntimeError(
            f"No XLA/TPU devices detected: global_runtime_device_count={num_devices}. "
            "Check TPU accelerator configuration and libtpu installation."
        )

    device = xm.xla_device()
    logger.info(
        "PyTorch/XLA SPMD initialized | global_runtime_device_count=%d | "
        "device=%s | python=%s | torch=%s | torch_xla=%s",
        num_devices,
        device,
        platform.python_version(),
        torch.__version__,
        getattr(torch_xla, "__version__", "unknown"),
    )
    return device, num_devices


def create_device_mesh(
    num_devices: int,
    axis_name: str = "data",
    logger: logging.Logger | None = None,
) -> Any:
    """Create a 1D device mesh over all TPU cores and register as global mesh."""
    if not XLA_AVAILABLE or xs is None or Mesh is None:
        raise RuntimeError("torch_xla.distributed.spmd is not available.")

    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, (num_devices,), (axis_name,))
    xs.set_global_mesh(mesh)
    if logger:
        logger.info(
            "Created XLA SPMD Mesh: shape=(%d,) axis_names=(%r,) device_ids=%s",
            num_devices,
            axis_name,
            device_ids.tolist(),
        )
    return mesh


def apply_spmd_parameter_sharding(
    model: HybridForCausalLM,
    mesh: Any,
    axis_name: str = "data",
    logger: logging.Logger | None = None,
) -> int:
    """Shard large 2D parameter matrices along dimension 0 across the TPU mesh.

    Under FSDP-style SPMD model-state sharding, 2D weight matrices (linear
    projections in attention, Mamba, MLP/MoE experts, LM head) are partitioned
    across the mesh axis. Custom direct-math parameters (RMSNorm gains, Mamba
    A_log and D, dual-memory combine projections, CompressiveMemoryBank) remain
    replicated to prevent DTensor recomputation and mixed-dispatch failures.
    """
    if not XLA_AVAILABLE or xs is None:
        raise RuntimeError("torch_xla.distributed.spmd is not available.")

    replicated_params: set[int] = set()
    for module in model.modules():
        if isinstance(module, RMSNorm):
            for p in module.parameters(recurse=False):
                replicated_params.add(id(p))
        elif isinstance(module, HybridDecoderLayer) and module.use_dual_memory:
            for combine in (module.attn_memory_combine, module.state_memory_combine):
                for p in combine.parameters(recurse=False):
                    replicated_params.add(id(p))
        elif isinstance(module, CompressiveMemoryBank):
            for p in module.parameters(recurse=True):
                replicated_params.add(id(p))
        elif isinstance(module, MambaBlock):
            replicated_params.add(id(module.A_log))
            replicated_params.add(id(module.D))
        elif isinstance(module, DroplessMoELayer):
            # Bypass grouped GEMM to enable clean parameter sharding per expert module
            module.use_grouped_moe_dispatch = False
            module.use_grouped_gemm = False

    sharded_params = 0
    replicated_params_count = 0
    sharded_tensors = 0
    replicated_tensors = 0

    for name, param in model.named_parameters():
        if id(param) in replicated_params or param.ndim < 2:
            # Replicated across all TPU devices
            spec = (None, None) if param.ndim == 2 else (None,)
            xs.mark_sharding(param, mesh, spec)
            replicated_params_count += param.numel()
            replicated_tensors += 1
        else:
            # 2D weight matrix: sharded on dim 0 across axis_name
            xs.mark_sharding(param, mesh, (axis_name, None))
            sharded_params += param.numel()
            sharded_tensors += 1

    total = sharded_params + replicated_params_count
    if logger:
        logger.info(
            "SPMD parameter sharding complete | sharded: %d tensors, %s params (%.2f%%) | "
            "replicated: %d tensors, %s params (%.2f%%) | total=%s params",
            sharded_tensors,
            f"{sharded_params:,}",
            100.0 * sharded_params / max(total, 1),
            replicated_tensors,
            f"{replicated_params_count:,}",
            100.0 * replicated_params_count / max(total, 1),
            f"{total:,}",
        )
    return sharded_params


# ---------------------------------------------------------------------------
# Optimizers & Learning Rate Schedulers
# ---------------------------------------------------------------------------


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
    logger: logging.Logger,
) -> tuple[list[optim.Optimizer], bool, dict[str, Any]]:
    """Create Muon+AdamW hybrid or AdamW-only fallback for TPU.

    On TPU/XLA, fused_adam is False because fused AdamW is CUDA-only.
    PyTorch's native torch.optim.Muon performs matrix orthogonalization using
    Newton-Schulz iterations with standard matmuls, which compiles directly
    into TPU matrix unit operations.
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
            "falling back to AdamW for all parameters on TPU"
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

    if enable_muon and adam_lr is None and resolved_adam_lr >= 5e-4:
        logger.warning(
            "T-H2: Muon and AdamW are both running at lr=%.3e. "
            "AdamW on embeddings/LM-head at this LR with small batch sizes "
            "can cause excessive gradient variance. Consider passing "
            "--adam-lr 3e-4 to decouple AdamW from the Muon learning rate.",
            resolved_adam_lr,
        )

    # fused=False on TPU: fused AdamW is CUDA-specific
    fused_adam = False

    if enable_muon:
        muon_kwargs: dict[str, Any] = {
            "lr": resolved_muon_lr,
            "weight_decay": weight_decay,
            "momentum": muon_momentum,
            "nesterov": muon_nesterov,
            "ns_steps": muon_ns_steps,
        }
        try:
            muon_optim: optim.Optimizer = optim.Muon(
                muon_params,
                adjust_lr_fn=muon_adjust_lr_fn,
                **muon_kwargs,
            )
        except TypeError:
            logger.warning(
                "torch.optim.Muon does not accept adjust_lr_fn=%r; "
                "constructing Muon without it",
                muon_adjust_lr_fn,
            )
            muon_optim = optim.Muon(muon_params, **muon_kwargs)
            meta["muon_adjust_lr_fn"] = None

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


def audit_optimizer_lr(
    optimizers: list[optim.Optimizer],
    use_muon: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Capture actual per-group LR and Muon RMS scale factors (host-side only)."""
    audit: dict[str, Any] = {"groups": []}
    muon_base_lr: float | None = None
    adam_base_lr: float | None = None
    muon_scale_factors: list[float] = []

    for opt_idx, opt in enumerate(optimizers):
        opt_name = "muon" if (use_muon and opt_idx == 0) else "adamw"
        for grp_idx, group in enumerate(opt.param_groups):
            actual_lr = float(group["lr"])
            wd = float(group.get("weight_decay", 0.0))
            n_params = sum(p.numel() for p in group["params"])
            n_tensors = len(group["params"])

            group_info: dict[str, Any] = {
                "optimizer": opt_name,
                "group_idx": grp_idx,
                "actual_lr": actual_lr,
                "weight_decay": wd,
                "n_tensors": n_tensors,
                "n_params": n_params,
            }

            if opt_name == "muon":
                muon_base_lr = actual_lr
                scales = []
                for p in group["params"]:
                    if p.ndim == 2:
                        rows, cols = p.shape
                        scale = 0.2 * math.sqrt(max(rows, cols))
                        scales.append(scale)
                        muon_scale_factors.append(scale)
                if scales:
                    group_info["muon_rms_scale_min"] = min(scales)
                    group_info["muon_rms_scale_max"] = max(scales)
                    group_info["muon_rms_scale_mean"] = sum(scales) / len(scales)
                    group_info["muon_effective_lr_min"] = actual_lr * min(scales)
                    group_info["muon_effective_lr_max"] = actual_lr * max(scales)
            else:
                if adam_base_lr is None:
                    adam_base_lr = actual_lr

            audit["groups"].append(group_info)

    audit["muon_base_lr"] = muon_base_lr
    audit["adam_base_lr"] = adam_base_lr
    if muon_base_lr is not None and adam_base_lr is not None and adam_base_lr > 0:
        if muon_scale_factors:
            mean_scale = sum(muon_scale_factors) / len(muon_scale_factors)
            max_scale = max(muon_scale_factors)
            audit["muon_mean_effective_lr"] = muon_base_lr * mean_scale
            audit["muon_max_effective_lr"] = muon_base_lr * max_scale
            audit["effective_lr_ratio_mean"] = (
                muon_base_lr * mean_scale
            ) / adam_base_lr
            audit["effective_lr_ratio_max"] = (muon_base_lr * max_scale) / adam_base_lr
            logger.info(
                "T-6 optimizer LR audit:\n"
                "  Muon base lr=%.3e | RMS-match scale mean=%.2f max=%.2f\n"
                "  Muon effective lr: mean=%.3e  max=%.3e\n"
                "  AdamW base lr=%.3e\n"
                "  Effective LR ratio (Muon/AdamW): mean=%.2fx  max=%.2fx",
                muon_base_lr,
                mean_scale,
                max_scale,
                muon_base_lr * mean_scale,
                muon_base_lr * max_scale,
                adam_base_lr,
                audit["effective_lr_ratio_mean"],
                audit["effective_lr_ratio_max"],
            )
        else:
            logger.info(
                "T-6 optimizer LR audit: Muon base lr=%.3e | AdamW base lr=%.3e",
                muon_base_lr,
                adam_base_lr,
            )
    elif adam_base_lr is not None:
        logger.info(
            "T-6 optimizer LR audit: AdamW-only mode | base lr=%.3e",
            adam_base_lr,
        )

    return audit


def _per_group_grad_norms(
    optimizers: list[optim.Optimizer],
    use_muon: bool,
) -> dict[str, float]:
    """Compute per-optimizer-group gradient L2 norms at metrics flush window."""
    result: dict[str, float] = {}
    for opt_idx, opt in enumerate(optimizers):
        opt_name = "muon" if (use_muon and opt_idx == 0) else "adamw"
        for grp_idx, group in enumerate(opt.param_groups):
            norm_sq = 0.0
            for p in group["params"]:
                if p.grad is not None:
                    # Computed on-device; norm cast to float scalar
                    norm_sq += float(p.grad.detach().float().norm().item() ** 2)
            result[f"{opt_name}_g{grp_idx}_grad_norm"] = math.sqrt(norm_sq)
    return result


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


# ---------------------------------------------------------------------------
# Metrics, Diagnostic Guards & Watchdog
# ---------------------------------------------------------------------------


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


class StepProgressWatchdog:
    """Warn from a daemon thread when a training phase stops making progress."""

    def __init__(
        self,
        logger: logging.Logger,
        timeout_seconds: float,
    ) -> None:
        self.logger = logger
        self.timeout_seconds = float(timeout_seconds)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._active = False
        self._step = -1
        self._phase = "idle"
        self._last_progress = time.monotonic()
        self._last_warning = 0.0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.timeout_seconds <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._monitor,
            name="training-step-watchdog",
            daemon=True,
        )
        self._thread.start()

    def progress(self, step: int, phase: str, *, active: bool = True) -> None:
        with self._lock:
            self._step = int(step)
            self._phase = phase
            self._active = active
            self._last_progress = time.monotonic()
            self._last_warning = 0.0

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _monitor(self) -> None:
        poll_seconds = max(0.25, min(5.0, self.timeout_seconds / 4.0))
        while not self._stop.wait(poll_seconds):
            now = time.monotonic()
            with self._lock:
                active = self._active
                elapsed = now - self._last_progress
                since_warning = now - self._last_warning
                step = self._step
                phase = self._phase
                should_warn = (
                    active
                    and elapsed >= self.timeout_seconds
                    and (
                        self._last_warning == 0.0
                        or since_warning >= self.timeout_seconds
                    )
                )
                if should_warn:
                    self._last_warning = now
            if should_warn:
                self.logger.warning(
                    "Training watchdog: no progress for %.1fs at step=%d "
                    "phase=%s on TPU. Common causes include XLA graph recompilation, "
                    "HBM out-of-memory, or data starvation.",
                    elapsed,
                    step,
                    phase,
                )


def _weighted_term_tensors(
    model: HybridForCausalLM, out: Any, step: int, max_steps: int
) -> dict[str, Any]:
    """Weighted auxiliary-loss terms kept on the XLA device."""
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
        "assoc_scale": assoc_scale,
        "expert_scale": expert_scale,
    }


def _as_flush_scalar(value: Any, device: torch.device) -> torch.Tensor:
    """Coerce a metric scalar to a 0-dim float32 tensor on ``device`` without host sync."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=torch.float32).reshape(())
    return torch.tensor(value, device=device, dtype=torch.float32)


def _mean_metric_sums(
    sums: dict[str, torch.Tensor], count: int
) -> dict[str, torch.Tensor]:
    """Average microbatch metric sums on-device without host synchronization."""
    if count < 1:
        raise ValueError("Metric reduction requires at least one microbatch.")
    return {name: value / count for name, value in sums.items()}


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
    """Diagnostic info for non-finite failure paths (called only on error)."""
    aux = out.auxiliary_losses
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
            "assoc_norm",
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
    cfg = model.config
    assoc_scale = _aux_loss_schedule(step, max_steps, cfg.assoc_warmup_fraction)
    expert_scale = _expert_loss_schedule(step, max_steps, cfg.expert_warmup_fraction)
    return {
        "non_finite_terms": non_finite,
        "values": values,
        "assoc_scale": assoc_scale,
        "expert_scale": expert_scale,
    }


def align_tokens_per_shard(tokens_per_shard: int, seq_len: int) -> int:
    """Round down so shard boundaries never drop partial sequences."""
    chunk = seq_len + 1
    aligned = (tokens_per_shard // chunk) * chunk
    return max(aligned, chunk)


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


# ---------------------------------------------------------------------------
# Checkpointing & Resume
# ---------------------------------------------------------------------------


def configure_gradient_checkpointing(
    model: HybridForCausalLM,
    enabled: bool,
    logger: logging.Logger,
) -> None:
    """Configure reentrant-free gradient checkpointing."""
    if enabled:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.gradient_checkpointing_disable()
    logger.info(
        "gradient_checkpointing=%s use_reentrant=%s train_use_cache=%s "
        "input_require_grads_hook=%s autocast_weight_cache=%s",
        model.is_gradient_checkpointing,
        model.config.gradient_checkpointing_use_reentrant,
        model.config.use_cache,
        model._input_require_grads_hook is not None,
        not model.is_gradient_checkpointing,
    )


def checkpoint_runtime_contract(
    model: HybridForCausalLM,
    *,
    sharding_strategy: str,
    optimizer_policy: str,
) -> dict[str, Any]:
    """Persist runtime/graph contract to safeguard against resume drift."""
    return {
        "version": 2,
        "gradient_checkpointing": model.is_gradient_checkpointing,
        "gradient_checkpointing_use_reentrant": False,
        "use_cache": False
        if model.is_gradient_checkpointing
        else bool(model.config.use_cache),
        "distributed_strategy": TPU_SPMD_CHECKPOINT_FAMILY,
        "checkpoint_family": TPU_SPMD_CHECKPOINT_FAMILY,
        "sharding_strategy": sharding_strategy,
        "optimizer_policy": optimizer_policy,
        "hardware_target": "tpu_spmd",
    }


def normalized_checkpoint_config(model: HybridForCausalLM) -> dict[str, Any]:
    config = asdict(model.config)
    if model.is_gradient_checkpointing:
        config["use_cache"] = False
        config["gradient_checkpointing_use_reentrant"] = False
    return config


def _rng_state_dict() -> dict[str, Any]:
    """Capture host and XLA RNG states in a safe, serializable format."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    np_state = np.random.get_state()
    state["numpy"] = {
        "bit_generator": str(np_state[0]),
        "key": np.asarray(np_state[1]).astype(np.int64).tolist(),
        "pos": int(np_state[2]),
        "has_gauss": int(np_state[3]),
        "cached_gaussian": float(np_state[4]),
    }
    if XLA_AVAILABLE and xm is not None:
        try:
            state["xla"] = xm.get_rng_state()
        except Exception:  # noqa: BLE001, S110
            pass
    return state


def _load_rng_state_dict(state: dict[str, Any] | None) -> bool:
    if not state:
        return False
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np_info = state["numpy"]
        if isinstance(np_info, dict):
            key = np.asarray(np_info["key"], dtype=np.int64).astype(np.uint32)
            np.random.set_state(
                (
                    str(np_info["bit_generator"]),
                    key,
                    int(np_info["pos"]),
                    int(np_info["has_gauss"]),
                    float(np_info["cached_gaussian"]),
                )
            )
    if "torch" in state:
        torch_s = state["torch"]
        if not isinstance(torch_s, torch.Tensor):
            torch_s = torch.tensor(torch_s, dtype=torch.uint8)
        torch.set_rng_state(torch_s.cpu())
    if "xla" in state and XLA_AVAILABLE and xm is not None:
        try:
            xm.set_rng_state(state["xla"])
        except Exception:  # noqa: BLE001, S110
            pass
    return "torch" in state and "python" in state


def save_checkpoint(
    *,
    model: HybridForCausalLM,
    optimizers: list[optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    global_step: int,
    current_shard_idx: int,
    checkpoint_dir: Path,
    logger: logging.Logger,
    sharding_strategy: str,
    validator: WikiTextCyclicValidator | None = None,
    use_muon: bool = True,
    extra_payload: dict[str, Any] | None = None,
) -> None:
    """Save training state using atomic write and xm.save for XLA compatibility."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / CHECKPOINT_FILENAME
    tmp_path = ckpt_path.with_suffix(".pth.tmp")

    config = normalized_checkpoint_config(model)
    optimizer_policy = (
        TPU_SPMD_OPTIMIZER_POLICY_MUON if use_muon else TPU_SPMD_OPTIMIZER_POLICY_ADAM
    )

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "global_step": global_step,
        "current_shard_idx": current_shard_idx,
        "rng_state": _rng_state_dict(),
        "memory_nan_fix_id": MEMORY_NAN_FIX_ID,
        "use_muon": use_muon,
        "training_runtime": checkpoint_runtime_contract(
            model,
            sharding_strategy=sharding_strategy,
            optimizer_policy=optimizer_policy,
        ),
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

    # xm.save transfers any live XLA tensors to CPU before pickling, ensuring safe write
    if XLA_AVAILABLE and xm is not None:
        xm.save(payload, tmp_path, master_only=True)
    else:
        torch.save(payload, tmp_path)

    if os.path.exists(tmp_path):
        os.replace(tmp_path, ckpt_path)

    config_path = checkpoint_dir / CONFIG_FILENAME
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info(
        "Checkpoint saved step=%d shard=%d path=%s",
        global_step,
        current_shard_idx,
        ckpt_path,
    )


def validate_resume_runtime_contract(
    checkpoint: dict[str, Any],
    model: HybridForCausalLM,
    logger: logging.Logger,
    *,
    sharding_strategy: str,
    optimizer_policy: str,
) -> None:
    """Verify runtime compatibility on resume to detect silent divergence."""
    runtime = checkpoint.get("training_runtime")
    saved_cfg = checkpoint.get("config")
    if not isinstance(saved_cfg, dict):
        saved_cfg = {}
    if not isinstance(runtime, dict):
        runtime = {}

    saved_gc = bool(
        runtime.get(
            "gradient_checkpointing",
            saved_cfg.get("gradient_checkpointing", False),
        )
    )
    current_gc = model.is_gradient_checkpointing
    if saved_gc != current_gc:
        raise RuntimeError(
            f"Gradient-checkpointing mismatch on resume: checkpoint={saved_gc}, "
            f"current={current_gc}."
        )

    saved_family = runtime.get("checkpoint_family")
    compatible_families = (TPU_SPMD_CHECKPOINT_FAMILY, "single_process")
    if saved_family is not None and saved_family not in compatible_families:
        logger.warning(
            "Resuming from non-TPU checkpoint family %r. Model weights will be loaded, "
            "but verify optimizer convergence.",
            saved_family,
        )

    saved_optimizer_policy = runtime.get("optimizer_policy")
    if (
        saved_optimizer_policy is not None
        and saved_optimizer_policy != optimizer_policy
    ):
        raise RuntimeError(
            f"Optimizer policy mismatch on resume: checkpoint={saved_optimizer_policy!r}, "
            f"current={optimizer_policy!r}."
        )

    logger.info(
        "Resume runtime diagnostics PASS | gradient_checkpointing=%s "
        "strategy=%s optimizer_policy=%s",
        current_gc,
        runtime.get("distributed_strategy", "tpu_spmd"),
        optimizer_policy,
    )


def load_checkpoint(
    *,
    model: HybridForCausalLM,
    optimizers: list[optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    checkpoint_dir: Path,
    device: torch.device,
    logger: logging.Logger,
    sharding_strategy: str,
    validator: WikiTextCyclicValidator | None = None,
    use_muon: bool | None = None,
    dl_generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Restore model, optimizer, scheduler, and RNG states from checkpoint."""
    ckpt_path = checkpoint_dir / CHECKPOINT_FILENAME
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "weights_only load failed (%s); retrying with weights_only=False.",
            exc,
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    optimizer_policy = (
        TPU_SPMD_OPTIMIZER_POLICY_MUON
        if use_muon is not False
        else TPU_SPMD_OPTIMIZER_POLICY_ADAM
    )
    validate_resume_runtime_contract(
        checkpoint,
        model,
        logger,
        sharding_strategy=sharding_strategy,
        optimizer_policy=optimizer_policy,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    ckpt_use_muon = bool(
        checkpoint.get("use_muon", "muon_optimizer_state_dict" in checkpoint)
    )
    if use_muon is not None and "use_muon" in checkpoint and ckpt_use_muon != use_muon:
        raise RuntimeError(
            f"Checkpoint was saved with use_muon={ckpt_use_muon} but this run "
            f"uses use_muon={use_muon}; optimizer states are incompatible."
        )

    if ckpt_use_muon and "muon_optimizer_state_dict" in checkpoint:
        optimizers[0].load_state_dict(checkpoint["muon_optimizer_state_dict"])
        optimizers[-1].load_state_dict(checkpoint["adam_optimizer_state_dict"])
        schedulers[0].load_state_dict(checkpoint["muon_scheduler_state_dict"])
        schedulers[-1].load_state_dict(checkpoint["adam_scheduler_state_dict"])
    else:
        optimizers[0].load_state_dict(checkpoint["adam_optimizer_state_dict"])
        schedulers[0].load_state_dict(checkpoint["adam_scheduler_state_dict"])

    # Ensure optimizer param_groups reflect loaded scheduler learning rates
    for opt, sched in zip(optimizers, schedulers):
        saved_lrs = sched.get_last_lr()
        if len(saved_lrs) == len(opt.param_groups):
            for group, saved_lr in zip(opt.param_groups, saved_lrs):
                group["lr"] = saved_lr

    rng_restored = _load_rng_state_dict(checkpoint.get("rng_state"))
    if not rng_restored:
        logger.warning(
            "Resume RNG diagnostic FAIL: checkpoint RNG state is missing or incompatible."
        )

    if dl_generator is not None:
        gen_state = checkpoint.get("dl_generator_state")
        if gen_state is not None:
            if not isinstance(gen_state, torch.Tensor):
                gen_state = torch.tensor(gen_state, dtype=torch.uint8)
            dl_generator.set_state(gen_state.cpu())

    if validator is not None and "validator_state_dict" in checkpoint:
        validator.load_state_dict(checkpoint["validator_state_dict"])

    ckpt_fix_id = checkpoint.get("memory_nan_fix_id")
    if ckpt_fix_id != MEMORY_NAN_FIX_ID:
        logger.warning(
            "Checkpoint memory_nan_fix_id=%r differs from current %r.",
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
# Validation on TPU
# ---------------------------------------------------------------------------


def _format_val_log_line(val_record: dict[str, Any]) -> str:
    return (
        f"validation step={val_record['step']} "
        f"val_loss={val_record['val_loss']:.6f} "
        f"val_ce={val_record['val_ce_loss']:.6f} "
        f"val_router_aux={val_record['val_router_aux_loss']:.6f} "
        f"val_router_z={val_record['val_router_z_loss']:.6f} "
        f"rows={val_record['val_rows']} batch={val_record['val_batch_size']} "
        f"cursor={val_record['val_row_start']}->{val_record['val_row_end']} "
        f"next={val_record['val_cursor']}"
    )


@torch.no_grad()
def evaluate_cyclic_spmd(
    validator: WikiTextCyclicValidator,
    model: HybridForCausalLM,
    mesh: Any,
    device: torch.device,
    global_step: int,
    max_training_steps: int,
    amp_dtype: torch.dtype,
    ignore_index: int,
    num_devices: int,
) -> dict[str, Any]:
    """Cyclic validation evaluation with SPMD batch sharding."""
    texts, row_start, row_end = validator._next_texts()
    from utils.validation import _collate_wikitext_batch, _WikiTextRowDataset

    row_ds = _WikiTextRowDataset(
        texts,
        validator.tokenizer,
        seq_len=validator.seq_len,
        bos_id=validator.bos_id,
        eos_id=validator.eos_id,
    )
    loader = DataLoader(
        row_ds,
        batch_size=validator.batch_size,
        shuffle=False,
        drop_last=True,  # Mandatory under SPMD to prevent non-divisible tensor shapes
        collate_fn=lambda batch: _collate_wikitext_batch(batch, validator.pad_token_id),
    )

    was_training = model.training
    model.eval()

    totals: dict[str, float] = {
        "loss": 0.0,
        "ce_loss": 0.0,
        "router_aux_loss": 0.0,
        "router_z_loss": 0.0,
    }
    token_weight = 0
    batch_count = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = build_causal_labels(input_ids, attention_mask, ignore_index)

        # Mark batch sharding across TPU mesh
        if XLA_AVAILABLE and xs is not None:
            xs.mark_sharding(input_ids, mesh, ("data", None))
            xs.mark_sharding(attention_mask, mesh, ("data", None))
            xs.mark_sharding(labels, mesh, ("data", None))

        with torch.autocast(device_type="xla", dtype=amp_dtype):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                training_step=global_step,
                max_training_steps=max_training_steps,
            )

        active = int((labels != ignore_index).sum().item())
        if active == 0 or out.loss is None:
            continue

        ce_val = (
            float(out.ce_loss.item())
            if out.ce_loss is not None
            else float(out.loss.item())
        )
        totals["loss"] += float(out.loss.item()) * active
        totals["ce_loss"] += ce_val * active
        if out.router_aux_loss is not None:
            totals["router_aux_loss"] += float(out.router_aux_loss.item())
        if out.router_z_loss is not None:
            totals["router_z_loss"] += float(out.router_z_loss.item())
        token_weight += active
        batch_count += 1

    if was_training:
        model.train()

    denom = max(1, token_weight)
    b_denom = max(1, batch_count)
    return {
        "step": global_step,
        "val_loss": totals["loss"] / denom,
        "val_ce_loss": totals["ce_loss"] / denom,
        "val_router_aux_loss": totals["router_aux_loss"] / b_denom,
        "val_router_z_loss": totals["router_z_loss"] / b_denom,
        "val_rows": validator.num_rows,
        "val_batch_size": validator.batch_size,
        "val_row_start": row_start,
        "val_row_end": row_end,
        "val_cursor": validator.cursor,
        "val_active_tokens": token_weight,
    }


@torch.no_grad()
def evaluate_packed_spmd(
    validator: PackedWindowValidator,
    model: HybridForCausalLM,
    mesh: Any,
    device: torch.device,
    global_step: int,
    max_training_steps: int,
    amp_dtype: torch.dtype,
    ignore_index: int,
) -> dict[str, Any]:
    """Packed continuous token stream validation under SPMD."""
    loader = DataLoader(
        validator._dataset,
        batch_size=validator._batch_size,
        shuffle=False,
        drop_last=True,  # Drop incomplete final window under SPMD
    )

    was_training = model.training
    model.eval()

    total_ce = 0.0
    total_tokens = 0
    n_windows = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        if XLA_AVAILABLE and xs is not None:
            xs.mark_sharding(input_ids, mesh, ("data", None))
            xs.mark_sharding(labels, mesh, ("data", None))

        with torch.autocast(device_type="xla", dtype=amp_dtype):
            out = model(
                input_ids=input_ids,
                labels=labels,
                training_step=global_step,
                max_training_steps=max_training_steps,
            )

        if out.ce_loss is None or out.loss is None:
            continue

        n_tokens = int((labels != ignore_index).sum().item())
        if n_tokens == 0:
            continue

        total_ce += float(out.ce_loss.item()) * n_tokens
        total_tokens += n_tokens
        n_windows += input_ids.size(0)

    if was_training:
        model.train()

    return {
        "step": global_step,
        "packed_val_ce": (total_ce / total_tokens) if total_tokens > 0 else 0.0,
        "packed_val_tokens": total_tokens,
        "packed_val_windows": n_windows,
    }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace, logger: logging.Logger) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(args.log_jsonl) if args.log_jsonl else None
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, deterministic=args.deterministic)

    # 1. Initialize PyTorch/XLA SPMD Runtime
    device, num_devices = init_tpu_spmd(logger)
    mesh = create_device_mesh(num_devices, axis_name="data", logger=logger)

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float32
    logger.info(
        "TPU SPMD training config: num_devices=%d | amp_dtype=%s | sharding_strategy=%s",
        num_devices,
        amp_dtype,
        args.sharding_strategy,
    )

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
    if producer.tokenizer.bos_token_id is not None:
        cfg.bos_token_id = producer.tokenizer.bos_token_id
    if producer.tokenizer.eos_token_id is not None:
        cfg.eos_token_id = producer.tokenizer.eos_token_id

    logger.info(log_mamba_backend(cfg))
    logger.info(
        "device=%s amp_dtype=%s fused_mamba=%s memory_nan_fix=%s",
        device,
        amp_dtype,
        fused_mamba_scan_available(),
        MEMORY_NAN_FIX_ID,
    )

    # 2. Construct Model on XLA Device
    model = HybridForCausalLM(cfg).to(device)
    configure_gradient_checkpointing(model, args.gradient_checkpointing, logger)
    n_params = count_trainable_params(model)
    logger.info("trainable_params=%s (%.3fB)", f"{n_params:,}", n_params / 1e9)

    # 3. Model Parameter Sharding (if FSDP mode selected)
    if args.sharding_strategy == "fsdp":
        apply_spmd_parameter_sharding(model, mesh, axis_name="data", logger=logger)

    # 4. Build Optimizers & Schedulers
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
        logger=logger,
    )

    _lr_audit = audit_optimizer_lr(optimizers, use_muon, logger)
    if jsonl_path is not None:
        with jsonl_path.open("a", encoding="utf-8") as _f:
            _f.write(
                json.dumps({"event": "optimizer_audit", "step": 0, **_lr_audit}) + "\n"
            )

    # 5. Initialize Validators (with batch size aligned to TPU devices)
    val_batch_size = args.val_batch_size
    if val_batch_size % num_devices != 0:
        val_batch_size = max(num_devices, (val_batch_size // num_devices) * num_devices)
        logger.info(
            "Aligned val_batch_size to %d (multiple of num_devices=%d)",
            val_batch_size,
            num_devices,
        )

    validator: WikiTextCyclicValidator | None = None
    if not args.no_validation:
        try:
            validator = WikiTextCyclicValidator(
                producer.tokenizer,
                seq_len=args.seq_len,
                num_rows=args.val_rows,
                batch_size=val_batch_size,
                dataset_config=args.val_dataset_config,
                bos_id=cfg.bos_token_id,
                eos_id=cfg.eos_token_id,
                pad_token_id=cfg.pad_token_id,
            )
            logger.info(
                "Cyclic validation enabled | dataset=Salesforce/wikitext/%s "
                "rows=%d batch=%d interval=%d",
                args.val_dataset_config,
                args.val_rows,
                val_batch_size,
                args.val_interval,
            )
        except Exception:
            logger.exception(
                "Failed to initialize WikiTextCyclicValidator; disabling validation"
            )
            validator = None

    packed_validator: PackedWindowValidator | None = None
    if not args.no_validation:
        try:
            packed_validator = PackedWindowValidator(
                producer.tokenizer,
                seq_len=args.seq_len,
                num_texts=200,
                batch_size=val_batch_size,
                dataset_config=args.val_dataset_config,
                bos_id=cfg.bos_token_id,
                eos_id=cfg.eos_token_id,
            )
            logger.info(
                "PackedWindowValidator initialized | dataset=Salesforce/wikitext/%s "
                "stream_len=%d windows=%d",
                args.val_dataset_config,
                packed_validator._stream_len,
                len(packed_validator._dataset),
            )
        except Exception:
            logger.warning(
                "Failed to initialize PackedWindowValidator; packed validation disabled.",
                exc_info=True,
            )
            packed_validator = None

    warmup_steps = _resolve_warmup_steps(args.warmup_steps, args.max_steps)
    lr_lambda = _build_lr_lambda(warmup_steps, args.max_steps, args.min_lr_ratio)
    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = [
        torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        for opt in optimizers
    ]

    global_step = 0
    current_shard_idx = 0
    ckpt_dir = Path(args.ckpt_dir)

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
            sharding_strategy=args.sharding_strategy,
            validator=validator,
            use_muon=use_muon,
            dl_generator=dl_generator,
        )

    stop_event = threading.Event()
    producer_thread: threading.Thread | None = None
    ce_smooth = RollingAverage(args.smooth_window)

    # TPU on-device metric accumulators (zero host sync in hot loop)
    metric_sum: torch.Tensor | None = None
    metric_fin_cnt: torch.Tensor | None = None
    metric_bad_cnt: torch.Tensor | None = None
    metric_count = 0
    assoc_scale_sum = 0.0
    expert_scale_sum = 0.0
    step_window_started: float | None = None
    metric_main_names: list[str] | None = None
    metric_gate_names: list[str] | None = None

    reset_mamba_scan_stats()
    watchdog = StepProgressWatchdog(logger, args.step_watchdog_seconds)
    watchdog.start()

    # Verify batch size divisibility across TPU mesh
    batch_size = args.batch_size
    if batch_size % num_devices != 0:
        logger.warning(
            "args.batch_size=%d is not divisible by num_devices=%d. "
            "Under SPMD, input dimension 0 must be evenly partitioned across the mesh. "
            "Adjusting batch_size from %d to %d.",
            batch_size,
            num_devices,
            batch_size,
            max(num_devices, (batch_size // num_devices) * num_devices),
        )
        batch_size = max(num_devices, (batch_size // num_devices) * num_devices)

    try:
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
        logger.info(
            "Shard producer started in background thread | cache_dir=%s",
            args.cache_dir,
        )

        accum = max(1, args.gradient_accumulation_steps)
        if accum > 1:
            logger.info(
                "Gradient accumulation enabled: %d micro-batches per optimizer step "
                "(global micro-batch=%d seqs [%d per TPU core] | effective step batch=%d seqs = %d tokens)",
                accum,
                batch_size,
                batch_size // num_devices,
                batch_size * accum,
                batch_size * accum * args.seq_len,
            )
        else:
            logger.info(
                "Global micro-batch=%d seqs (%d seqs per TPU core) = %d tokens per step",
                batch_size,
                batch_size // num_devices,
                batch_size * args.seq_len,
            )

        while global_step < args.max_steps:
            shard_name = f"shard_{current_shard_idx:06d}.bin"
            bin_path = os.path.join(args.cache_dir, shard_name)
            sidecar_path = os.path.join(
                args.cache_dir, shard_name.replace(".bin", ".json")
            )
            wait_start = time.time()

            while not (os.path.exists(bin_path) and os.path.exists(sidecar_path)):
                producer_error = getattr(producer, "error", None)
                if producer_error is not None:
                    raise RuntimeError(
                        f"Shard producer thread failed while waiting for "
                        f"{shard_name}: {producer_error!r}"
                    )
                if time.time() - wait_start > args.producer_wait_timeout:
                    raise RuntimeError(
                        f"Timed out waiting for {shard_name} after "
                        f"{args.producer_wait_timeout}s (producer stalled)."
                    )
                time.sleep(1.0)

            dataset = MmapShardDataset(bin_path=bin_path, seq_len=args.seq_len + 1)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
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
            batches_iter = iter(dataloader)
            batches_seen = 0

            while True:
                if global_step >= args.max_steps:
                    shard_fully_consumed = False
                    break

                micro_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
                watchdog.progress(global_step, "data_loading")
                while len(micro_inputs) < accum:
                    try:
                        input_ids, labels = next(batches_iter)
                    except StopIteration:
                        break

                    # Move to XLA device
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)

                    # SPMD Input Sharding: Annotate batch dim to shard over the 'data' mesh axis
                    if XLA_AVAILABLE and xs is not None:
                        xs.mark_sharding(input_ids, mesh, ("data", None))
                        xs.mark_sharding(labels, mesh, ("data", None))

                    # Validate token bounds periodically without per-step sync
                    if batches_seen == 0 or (
                        args.validate_token_interval > 0
                        and global_step % args.validate_token_interval == 0
                    ):
                        validate_token_batch(
                            input_ids,
                            cfg.vocab_size,
                            labels=labels,
                            ignore_index=cfg.label_ignore_index,
                        )
                    micro_inputs.append((input_ids, labels))
                    batches_seen += 1

                if not micro_inputs:
                    break
                if len(micro_inputs) < accum:
                    # Drop incomplete tail micro-batches at shard boundary to prevent mis-scaled gradients
                    break

                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)

                if step_window_started is None:
                    step_window_started = time.perf_counter()

                step_metric_sums: dict[str, torch.Tensor] = {}
                step_gate_sums: dict[str, torch.Tensor] = {}
                schedule_scales: dict[str, float] = {}

                for micro_idx, (m_ids, m_labels) in enumerate(micro_inputs):
                    watchdog.progress(
                        global_step,
                        f"forward_micro_{micro_idx + 1}/{len(micro_inputs)}",
                    )
                    with torch.autocast(
                        device_type="xla",
                        dtype=amp_dtype,
                    ):
                        outputs = model(
                            input_ids=m_ids,
                            labels=m_labels,
                            training_step=global_step,
                            max_training_steps=args.max_steps,
                        )
                    assert outputs.loss is not None
                    aux = outputs.auxiliary_losses
                    assert aux is not None
                    weighted_t = _weighted_term_tensors(
                        model, outputs, global_step, args.max_steps
                    )
                    micro_scalars: dict[str, Any] = {
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
                    }
                    for name in (
                        "recon",
                        "assoc",
                        "assoc_norm",
                        "gate",
                        "read",
                        "fusion",
                        "expert",
                        "ssm",
                        "slot",
                    ):
                        micro_scalars[name] = getattr(aux, name)
                    for name, value in weighted_t.items():
                        if isinstance(value, torch.Tensor):
                            micro_scalars[name] = value
                        else:
                            schedule_scales[name] = float(value)

                    for name, value in micro_scalars.items():
                        scalar = _as_flush_scalar(value, outputs.loss.device)
                        step_metric_sums[name] = step_metric_sums.get(name, 0) + scalar
                    for name, value in (outputs.gate_stats or {}).items():
                        scalar = _as_flush_scalar(value, outputs.loss.device)
                        step_gate_sums[name] = step_gate_sums.get(name, 0) + scalar

                    watchdog.progress(
                        global_step,
                        f"backward_micro_{micro_idx + 1}/{len(micro_inputs)}",
                    )
                    micro_loss = outputs.loss / len(micro_inputs)
                    micro_loss.backward()

                micro_count = len(micro_inputs)
                scalars = _mean_metric_sums(step_metric_sums, micro_count)
                gate_stats = _mean_metric_sums(step_gate_sums, micro_count)
                assoc_scale_sum += schedule_scales.get("assoc_scale", 0.0)
                expert_scale_sum += schedule_scales.get("expert_scale", 0.0)

                watchdog.progress(global_step, "optimizer")

                # Gradient clipping on XLA device
                grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scalars["grad_norm"] = grad_norm

                allow_update = True
                if args.grad_nan_guard == "strict":
                    # Deliberate single host sync for strict finite check
                    allow_update = bool(torch.isfinite(grad_norm).item())
                    if not allow_update:
                        logger.error(
                            "Non-finite gradient norm at step %d — optimizer update SKIPPED",
                            global_step,
                        )
                elif args.grad_nan_guard == "sanitize":
                    # Zero host sync: sanitize NaNs/Infs on device directly
                    for p in model.parameters():
                        if p.grad is not None:
                            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

                if allow_update:
                    for opt in optimizers:
                        opt.step()
                    for sched in schedulers:
                        sched.step()

                # Mark XLA execution step boundary
                if XLA_AVAILABLE and xm is not None:
                    xm.mark_step()

                watchdog.progress(global_step, "metrics")

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
                finite = torch.isfinite(step_vec)
                if (
                    metric_sum is None
                    or metric_main_names != main_names
                    or metric_gate_names != gate_names
                ):
                    metric_main_names, metric_gate_names = main_names, gate_names
                    metric_sum = torch.zeros_like(step_vec)
                    metric_fin_cnt = torch.zeros_like(step_vec)
                    metric_bad_cnt = torch.zeros_like(step_vec)
                    metric_count = 0
                metric_sum += torch.where(finite, step_vec, 0.0)
                metric_fin_cnt += finite.to(step_vec.dtype)
                metric_bad_cnt += (~finite).to(step_vec.dtype)
                metric_count += 1

                # 6. Single D2H Host Transfer Flush Every log_interval Steps
                if global_step % args.log_interval == 0:
                    assert (
                        metric_sum is not None
                        and metric_fin_cnt is not None
                        and metric_bad_cnt is not None
                        and metric_main_names is not None
                        and metric_gate_names is not None
                    )
                    means_vec = metric_sum / metric_fin_cnt.clamp(min=1.0)
                    flush_values = torch.cat([means_vec, metric_bad_cnt]).tolist()
                    values = flush_values[: means_vec.numel()]
                    bad_counts = flush_values[means_vec.numel() :]
                    n_main = len(metric_main_names)
                    metrics = dict(zip(metric_main_names, values[:n_main]))
                    ce_smooth.update(metrics["ce_loss"])
                    window = max(metric_count, 1)
                    assert step_window_started is not None
                    step_time_s = (time.perf_counter() - step_window_started) / window

                    record: dict[str, Any] = {
                        "step": global_step,
                        "shard_idx": current_shard_idx,
                        **metrics,
                        "assoc_scale": assoc_scale_sum / window,
                        "expert_scale": expert_scale_sum / window,
                        "ce_smooth": ce_smooth.mean,
                        "muon_lr": float(schedulers[0].get_last_lr()[0]),
                        "adam_lr": float(schedulers[-1].get_last_lr()[0]),
                        "step_time_s": step_time_s,
                        "tokens_per_sec": (batch_size * accum * args.seq_len)
                        / max(step_time_s, 1e-6),
                        "gate_stats": dict(zip(metric_gate_names, values[n_main:])),
                    }

                    for _oi, (_opt, _sched) in enumerate(zip(optimizers, schedulers)):
                        _oname = "muon" if (use_muon and _oi == 0) else "adamw"
                        for _gi, _glr in enumerate(_sched.get_last_lr()):
                            record[f"{_oname}_g{_gi}_sched_lr"] = float(_glr)

                    record["per_group_grad_norms"] = _per_group_grad_norms(
                        optimizers, use_muon
                    )

                    bad_names = {
                        name: int(cnt)
                        for name, cnt in list(
                            zip(metric_main_names, bad_counts[:n_main])
                        )
                        + list(zip(metric_gate_names, bad_counts[n_main:]))
                        if cnt > 0.0
                    }
                    if bad_names:
                        diagnosis = _non_finite_diagnosis(
                            model, outputs, global_step, args.max_steps
                        )
                        logger.error(
                            "Non-finite metrics in steps %d-%d: %s",
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

                    metric_sum = None
                    metric_fin_cnt = None
                    metric_bad_cnt = None
                    metric_count = 0
                    assoc_scale_sum = 0.0
                    expert_scale_sum = 0.0
                    step_window_started = None

                    if (
                        args.lr_audit_interval > 0
                        and global_step % args.lr_audit_interval == 0
                        and global_step > 0
                    ):
                        _periodic_audit = audit_optimizer_lr(
                            optimizers, use_muon, logger
                        )
                        if jsonl_path is not None:
                            with jsonl_path.open("a", encoding="utf-8") as f:
                                f.write(
                                    json.dumps(
                                        {
                                            "event": "optimizer_audit",
                                            "step": global_step,
                                            **_periodic_audit,
                                        }
                                    )
                                    + "\n"
                                )

                    # Periodic Cyclic Validation
                    if (
                        validator is not None
                        and args.val_interval > 0
                        and global_step % args.val_interval == 0
                    ):
                        watchdog.progress(global_step, "validation")
                        val_record = evaluate_cyclic_spmd(
                            validator,
                            model,
                            mesh,
                            device,
                            global_step=global_step,
                            max_training_steps=args.max_steps,
                            amp_dtype=amp_dtype,
                            ignore_index=cfg.label_ignore_index,
                            num_devices=num_devices,
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
                        watchdog.progress(global_step, "metrics")

                    # Periodic Packed Window Validation
                    if (
                        packed_validator is not None
                        and args.val_interval > 0
                        and global_step % args.val_interval == 0
                    ):
                        watchdog.progress(global_step, "packed_validation")
                        packed_val_record = evaluate_packed_spmd(
                            packed_validator,
                            model,
                            mesh,
                            device,
                            global_step=global_step,
                            max_training_steps=args.max_steps,
                            amp_dtype=amp_dtype,
                            ignore_index=cfg.label_ignore_index,
                        )
                        record["packed_val_ce"] = packed_val_record["packed_val_ce"]
                        if jsonl_path is not None:
                            with jsonl_path.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(packed_val_record) + "\n")
                        logger.info(
                            "packed_val | step=%d packed_val_ce=%.4f windows=%d",
                            global_step,
                            packed_val_record["packed_val_ce"],
                            packed_val_record["packed_val_windows"],
                        )
                        watchdog.progress(global_step, "metrics")

                    if jsonl_path is not None:
                        with jsonl_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(record) + "\n")

                    logger.info(_format_log_line(global_step, args.max_steps, record))

                # 7. Checkpointing
                if global_step % args.save_interval == 0 and global_step > 0:
                    watchdog.progress(global_step, "checkpoint_save")
                    save_checkpoint(
                        model=model,
                        optimizers=optimizers,
                        schedulers=schedulers,
                        global_step=global_step,
                        current_shard_idx=current_shard_idx,
                        checkpoint_dir=ckpt_dir,
                        logger=logger,
                        sharding_strategy=args.sharding_strategy,
                        validator=validator,
                        use_muon=use_muon,
                        extra_payload={
                            "dl_generator_state": dl_generator.get_state(),
                            "muon_adjust_lr_fn": _opt_meta.get("muon_adjust_lr_fn"),
                            "num_devices": num_devices,
                        },
                    )
                    producer.save_checkpoint(args.dataset_ckpt_path)

                watchdog.progress(global_step, "complete", active=False)
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

        # Final checkpoint if not on save_interval boundary
        if global_step > 0 and global_step % args.save_interval != 0:
            watchdog.progress(global_step, "final_checkpoint_save")
            save_checkpoint(
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                global_step=global_step,
                current_shard_idx=current_shard_idx,
                checkpoint_dir=ckpt_dir,
                logger=logger,
                sharding_strategy=args.sharding_strategy,
                validator=validator,
                use_muon=use_muon,
                extra_payload={
                    "dl_generator_state": dl_generator.get_state(),
                    "muon_adjust_lr_fn": _opt_meta.get("muon_adjust_lr_fn"),
                    "num_devices": num_devices,
                },
            )
            producer.save_checkpoint(args.dataset_ckpt_path)

        scan_stats = get_mamba_scan_stats()
        logger.info(
            "TPU SPMD training finished steps=%d mamba_scan=%s",
            global_step,
            scan_stats,
        )

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — saving checkpoint before exit")
        if global_step > 0:
            watchdog.progress(global_step, "interrupt_checkpoint_save")
            save_checkpoint(
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                global_step=global_step,
                current_shard_idx=current_shard_idx,
                checkpoint_dir=ckpt_dir,
                logger=logger,
                sharding_strategy=args.sharding_strategy,
                validator=validator,
                use_muon=use_muon,
                extra_payload={
                    "dl_generator_state": dl_generator.get_state(),
                    "muon_adjust_lr_fn": _opt_meta.get("muon_adjust_lr_fn"),
                    "num_devices": num_devices,
                },
            )
            producer.save_checkpoint(args.dataset_ckpt_path)
        raise

    except Exception:
        logger.exception("Training failed")
        raise

    finally:
        watchdog.close()
        stop_event.set()
        if producer_thread is not None:
            producer_thread.join(timeout=30.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Hybrid Mamba-MoE on Cloud TPUs using PyTorch/XLA SPMD.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="./runs/tpu_train",
        help="Logs and artifacts root directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume model + dataset checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic operations where supported.",
    )
    parser.add_argument(
        "--sharding-strategy",
        choices=("data_parallel", "fsdp"),
        default="data_parallel",
        help="SPMD sharding strategy: 'data_parallel' (default: input batches sharded "
        "across mesh, model replicated) or 'fsdp' (FSDP-style 2D parameter sharding).",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="Precision on TPU: 'bf16' (native TPU bfloat16 autocast) or 'fp32'.",
    )
    checkpointing = parser.add_mutually_exclusive_group()
    checkpointing.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Checkpoint decoder layers with use_reentrant=False to reduce TPU HBM.",
    )
    checkpointing.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing (default).",
    )
    parser.set_defaults(gradient_checkpointing=False)

    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Warmup steps (0 = auto ~2.6%% of max_steps).",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="Cosine floor as fraction of peak LR.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Shared peak learning rate for Muon and AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Decoupled weight decay.",
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
        help="Muon momentum (default: 0.95).",
    )
    parser.add_argument(
        "--no-muon-nesterov",
        action="store_true",
        help="Disable Nesterov momentum in Muon.",
    )
    parser.add_argument(
        "--muon-ns-steps",
        type=int,
        default=5,
        help="Newton-Schulz orthogonalization steps (default: 5).",
    )
    parser.add_argument(
        "--muon-adjust-lr-fn",
        type=str,
        default="match_rms_adamw",
        choices=("match_rms_adamw", "original"),
        help="Muon per-matrix LR scale factor.",
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument(
        "--no-muon",
        action="store_true",
        help="Use AdamW for all parameters instead of Muon+AdamW hybrid.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=100_000, help="Maximum training steps."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Global micro-batch size across the TPU mesh (must be divisible by num_devices; "
        "e.g. 8 on TPU v5e-8 corresponds to 1 sequence per core). Default: 8.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        "--grad-accum-steps",
        dest="gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate per optimizer step. Default: 1.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Sequence context length in tokens.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient norm clipping threshold.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Steps between metrics logging & flush.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Steps between saving checkpoints.",
    )
    parser.add_argument(
        "--step-watchdog-seconds",
        type=float,
        default=300.0,
        help="Warn if no progress is made for this many seconds (0 disables).",
    )
    parser.add_argument(
        "--validate-token-interval",
        type=int,
        default=256,
        help="Steps between token-id sanity checks (0 = shard starts only).",
    )
    parser.add_argument(
        "--grad-nan-guard",
        choices=("sanitize", "strict"),
        default="sanitize",
        help="'sanitize' zeroes NaNs/Infs on device with ZERO host sync (recommended for TPU); "
        "'strict' triggers D2H sync to skip non-finite steps.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=50,
        help="Window for rolling CE loss mean.",
    )
    parser.add_argument(
        "--lr-audit-interval",
        type=int,
        default=0,
        help="Periodic optimizer LR audit interval (0 = startup only).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker processes for shard reading.",
    )

    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="./model_ckpt",
        help="Checkpoint directory.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="UIC-AI-lab/llama2-tokenizer",
        help="HuggingFace tokenizer name or path.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./data_cache",
        help="Directory containing or buffering tokenized shards.",
    )
    parser.add_argument(
        "--dataset-ckpt-path",
        type=str,
        default="./dataset_checkpoint.json",
        help="Shard producer checkpoint path.",
    )
    parser.add_argument(
        "--tokens-per-shard",
        type=int,
        default=5_000_000,
        help="Tokens per packed binary shard.",
    )
    parser.add_argument(
        "--max-buffered-files",
        type=int,
        default=10,
        help="Maximum downloaded shards ahead of training.",
    )
    parser.add_argument(
        "--producer-wait-timeout",
        type=int,
        default=600,
        help="Seconds to wait for a shard to become ready.",
    )
    parser.add_argument(
        "--log-jsonl",
        type=str,
        default="",
        help="Optional path for JSONL metrics (defaults to <run-dir>/metrics.jsonl).",
    )

    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable validation evaluation.",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=200,
        help="Training steps between validation runs.",
    )
    parser.add_argument(
        "--val-rows",
        type=int,
        default=50,
        help="Number of wikitext rows per cyclic evaluation.",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=8,
        help="Batch size for validation scoring (aligned to TPU device count).",
    )
    parser.add_argument(
        "--val-dataset-config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Salesforce/wikitext split configuration.",
    )

    args = parser.parse_args()
    if args.step_watchdog_seconds < 0:
        parser.error("--step-watchdog-seconds must be >= 0")
    if not args.log_jsonl:
        args.log_jsonl = str(Path(args.run_dir) / "metrics.jsonl")
    return args


def main() -> None:
    args = parse_args()
    logger = setup_logging(Path(args.run_dir))
    logger.info("Starting TPU SPMD pre-training with args: %s", vars(args))
    train(args, logger)


if __name__ == "__main__":
    main()
