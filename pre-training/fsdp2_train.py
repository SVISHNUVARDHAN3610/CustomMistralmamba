"""FSDP2 + Muon/AdamW distributed training for Hybrid Mamba-MoE.

Multi-GPU port of the root ``train.py`` trainer. Hidden 2-D weights use a
DTensor-aware Muon implementation while embeddings, heads, gains, biases,
non-matrix weights, and deliberately replicated custom-math parameters use
AdamW. Muon gathers each complete momentum matrix before Newton--Schulz since
orthogonalization is not separable over FSDP2 row shards; the gathered working
set is bounded to avoid retaining every full matrix at once.

Deliberate deviations from ``train.py`` (each required by distribution):
  * ``fully_shard`` wrapping (inner-to-outer). Parameters stay FP32 under
    FSDP2 and mixed precision is a real ``torch.autocast(bfloat16)`` around
    the forward, exactly like ``train.py`` — NOT
    ``MixedPrecisionPolicy(param_dtype=bfloat16)``: physically casting
    weights to bf16 during forward breaks ``model/hybrid/layer.py``'s
    auxiliary-loss block, which runs under ``autocast(enabled=False)`` on
    fp32-promoted activations ("recon/gate/slot paths need fp32") and hits
    fp32-activation x bf16-weight matmul errors (found on the first GPU
    smoke). The fp16 GradScaler path is removed entirely: its inf-reduction
    assumes plain gradients.
  * ``DistributedSampler(shuffle=True, seed, drop_last=True)`` replaces
    DataLoader shuffle. drop_last is mandatory — an uneven final batch would
    deadlock the collective backward. The intra-shard permutation is a pure
    function of (seed, shard_idx, rank, world_size), so resumes replay
    without persisting a sampler generator; changing world size between
    resumes changes the data order (warned at restore).
  * Shard production runs on RANK 0 only; every rank consumes the shared
    memmap shards read-only through the same bin+json sidecar wait protocol.
  * Metrics/validation totals are all-reduced to GLOBAL values before the
    single per-window host transfer; JSONL/console output happens on rank 0.
  * ``--grad-nan-guard=strict`` takes a GLOBAL min-vote over ranks before
    skipping: a per-rank skip would leave that rank's parameter shard
    permanently out of sync with the rest of the model.
  * FSDP2 checkpoints use an explicit trainer family and optimizer policy.
    Muon+AdamW and AdamW-only resumes are kept distinct instead of relying on
    unsafe positional state compatibility. RNG state is captured per rank.
  * Checkpoints persist an intra-shard batch cursor. The seeded sampler plus
    that offset resumes at the exact next per-rank batch for the same world
    size instead of replaying the beginning of the shard.
  * ``--compile`` is rejected (torch.compile x FSDP2 wrap-order pitfalls).
  * ``--batch-size`` is PER RANK (effective batch = batch-size x world x
    --grad-accum-steps); ``--device`` is derived from the launcher.

Run under torchrun::

    torchrun --nproc_per_node=8 pre-training/fsdp2_train.py \
        --cache-dir ./data_cache --run-dir ./runs/fsdp2 ...

A bare single process auto-initializes a world-of-1 group (gloo on CPU) for
smoke tests. ``--ns-self-check`` validates the Newton-Schulz math on CPU
without touching the process group or the dataset imports.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import logging
import math
import os
import platform
import random
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Light imports only above this line: model/** pulls no
# datasets/transformers, so --ns-self-check stays cheap and dependency-light
# (same rationale as model/core/optim.py).
from model.core.builders import count_trainable_params
from model.core.constants import MEMORY_NAN_FIX_ID
from model.core.optim import _is_adamw_no_decay, split_muon_adam_params
from model.hybrid.layer import HybridDecoderLayer
from model.hybrid.mamba import (
    MambaBlock,
    fused_mamba_scan_available,
    get_mamba_scan_stats,
    log_mamba_backend,
    reset_mamba_scan_stats,
)
from model.hybrid.memory import CompressiveMemoryBank
from model.layers.moe import DroplessMoELayer
from model.layers.norm import RMSNorm
from utils.fsdp2_muon import MuonDTensor
from utils.training_logging import format_training_log_line

FSDP2_CHECKPOINT_FAMILY = "fsdp2"
FSDP2_MUON_OPTIMIZER_POLICY = "fsdp2_muon_adamw"
FSDP2_ADAMW_OPTIMIZER_POLICY = "fsdp2_adamw"
FSDP2_REPLICATED_GRAD_BUCKET_CAP_MB = 25.0


def _fsdp2_optimizer_policy(use_muon: bool) -> str:
    return FSDP2_MUON_OPTIMIZER_POLICY if use_muon else FSDP2_ADAMW_OPTIMIZER_POLICY


def _runtime_environment(world_size: int) -> dict[str, Any]:
    """Resolved software/runtime versions persisted with every checkpoint."""
    cudnn_version = (
        torch.backends.cudnn.version() if torch.cuda.is_available() else None
    )
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": cudnn_version,
        "world_size": world_size,
    }


def _data_runtime_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Arguments that define the meaning of the intra-shard batch cursor."""
    return {
        "seed": args.seed,
        "batch_size_per_rank": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "seq_len": args.seq_len,
        "sampler_drop_last": True,
    }


class _OffsetSampler:
    """Resume a deterministic base sampler after ``offset`` local samples."""

    def __init__(self, base_sampler: DistributedSampler, offset: int) -> None:
        if offset < 0 or offset > len(base_sampler):
            raise ValueError(
                f"Sampler resume offset {offset} is outside [0, {len(base_sampler)}]."
            )
        self.base_sampler = base_sampler
        self.offset = offset

    def __iter__(self):
        return itertools.islice(iter(self.base_sampler), self.offset, None)

    def __len__(self) -> int:
        return len(self.base_sampler) - self.offset


def _mean_metric_sums(
    sums: dict[str, torch.Tensor], count: int
) -> dict[str, torch.Tensor]:
    """Average equal-token-count microbatch metric sums without host sync."""
    if count < 1:
        raise ValueError("Metric reduction requires at least one microbatch.")
    return {name: value / count for name, value in sums.items()}


class _FlushStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def _setup_logging(run_dir: Path, *, level: int = logging.INFO) -> logging.Logger:
    """Console logging on every rank; the run log FILE is written by rank 0
    only (shared-filesystem multi-writer appends would interleave garbage).
    Rank comes from the launcher env, readable before process-group init."""
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fsdp2_train")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = _FlushStreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if os.environ.get("RANK", "0") == "0":
        file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def _require_heavy_imports() -> None:
    """Import the pieces that pull datasets/transformers up front (fail fast
    on missing deps before any GPU/process-group work)."""
    import train as _train_mod  # noqa: F401 - pulls HybridForCausalLM et al.
    import utils.dataset
    import utils.validation  # noqa: F401


def _require_fsdp2() -> dict[str, Any]:
    """Load the public FSDP2/DTensor APIs required by the pinned baseline."""
    release = torch.__version__.split("+", maxsplit=1)[0]
    try:
        major, minor = (int(part) for part in release.split(".")[:2])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot parse PyTorch version {torch.__version__!r}."
        ) from exc
    if (major, minor) < (2, 6):
        raise RuntimeError(
            f"FSDP2 training requires torch>=2.6.0; found {torch.__version__}. "
            "Install requirements-fsdp2.txt using the matching official "
            "PyTorch CPU/CUDA wheel index."
        )
    try:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
            set_model_state_dict,
        )
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        from torch.distributed.tensor import DTensor, distribute_tensor
    except ImportError as exc:  # pragma: no cover - depends on torch build
        raise RuntimeError(
            "FSDP2 training requires torch>=2.6 with the stable "
            "torch.distributed.fsdp.fully_shard API and the "
            "torch.distributed.checkpoint.state_dict helpers. "
            f"Import failed with: {exc}"
        ) from exc
    return {
        "StateDictOptions": StateDictOptions,
        "get_model_state_dict": get_model_state_dict,
        "set_model_state_dict": set_model_state_dict,
        "MixedPrecisionPolicy": MixedPrecisionPolicy,
        "fully_shard": fully_shard,
        "DTensor": DTensor,
        "distribute_tensor": distribute_tensor,
    }


def _prepare_fsdp2_custom_math_params(
    model: torch.nn.Module,
) -> set[torch.nn.Parameter]:
    """Keep custom direct-math parameters replicated under FSDP2.

    FSDP2 temporarily represents sharded parameters as DTensors. Standard
    module calls such as ``nn.Linear`` are safe, but this model has several
    custom kernels/fast paths that read parameters directly (RMSNorm gains,
    dual-memory stacked projections, Mamba A/D vectors). Under activation
    checkpoint recompute those direct reads can see sharded DTensors, causing
    either Tensor/DTensor mixed-dispatch errors or local-shard shape mismatches.

    Mark those parameters as ignored for ``fully_shard`` and force them into
    AdamW; their gradients are averaged explicitly by
    ``_sync_replicated_param_grads`` before clipping/stepping.
    """
    ignored: set[torch.nn.Parameter] = set()
    seen: set[int] = set()

    def add(param: torch.nn.Parameter | None) -> None:
        if param is None:
            return
        param_id = id(param)
        if param_id in seen:
            return
        seen.add(param_id)
        param._fsdp2_force_adamw = True  # type: ignore[attr-defined]
        ignored.add(param)

    for module in model.modules():
        if isinstance(module, RMSNorm):
            for param in module.parameters(recurse=False):
                add(param)
        elif isinstance(module, HybridDecoderLayer) and module.use_dual_memory:
            # These projections are siblings of the memory banks, rather than
            # children of them. Checkpoint replay can otherwise enter their
            # nn.Linear calls with a local Tensor activation and a DTensor
            # weight before FSDP2 has materialized the weight.
            for combine in (
                module.attn_memory_combine,
                module.state_memory_combine,
            ):
                for param in combine.parameters(recurse=False):
                    add(param)
        elif isinstance(module, CompressiveMemoryBank):
            for param in module.parameters(recurse=True):
                add(param)
        elif isinstance(module, MambaBlock):
            add(module.A_log)
            add(module.D)

        if isinstance(module, DroplessMoELayer):
            # Grouped/stacked expert dispatch reads expert weights directly and
            # bypasses each expert module's FSDP2 hooks. The loop dispatch calls
            # the expert modules normally, so each expert can be safely sharded.
            module.use_grouped_moe_dispatch = False
            module.use_grouped_gemm = False

    return ignored


def _ordered_replicated_params(
    model: torch.nn.Module,
    ignored_params: set[torch.nn.Parameter],
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    """Return ignored parameters in stable model traversal order.

    ``fully_shard(..., ignored_params=...)`` needs identity-based set membership,
    but a set must never define distributed collective order: parameter object
    hashes depend on process-local identities and therefore differ across ranks.
    """
    ignored_ids = {id(param) for param in ignored_params}
    ordered = tuple(
        (name, param)
        for name, param in model.named_parameters()
        if id(param) in ignored_ids
    )
    ordered_ids = {id(param) for _, param in ordered}
    if ordered_ids != ignored_ids:
        raise RuntimeError(
            "Unable to build a complete deterministic order for FSDP2 "
            f"replicated parameters: expected={len(ignored_ids)} "
            f"found={len(ordered_ids)}."
        )
    return ordered


def _sync_replicated_param_grads(
    named_params: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    world_size: int,
    bucket_cap_mb: float = FSDP2_REPLICATED_GRAD_BUCKET_CAP_MB,
) -> None:
    """Average ignored-parameter gradients in rank-consistent buckets.

    Every rank first exchanges one compact presence/type vector. Parameters
    unused on every rank are skipped consistently; when only some ranks have a
    gradient, the others contribute an explicit zero gradient. The subsequent
    bucket layout depends only on stable model order, shapes, dtypes, and the
    globally agreed presence vector, so every rank launches identical
    collectives in identical order.
    """
    if world_size <= 1 or not named_params:
        return
    if bucket_cap_mb <= 0:
        raise ValueError("bucket_cap_mb must be positive")

    device = named_params[0][1].device
    status = torch.tensor(
        [
            [int(param.grad is not None) for _, param in named_params],
            [int(hasattr(param, "to_local")) for _, param in named_params],
            [
                int(param.grad is not None and hasattr(param.grad, "to_local"))
                for _, param in named_params
            ],
        ],
        device=device,
        dtype=torch.int32,
    )
    torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.MAX)
    globally_present, dtensor_params, dtensor_grads = status.tolist()

    invalid = [
        name
        for (name, _), param_is_dtensor, grad_is_dtensor in zip(
            named_params, dtensor_params, dtensor_grads
        )
        if param_is_dtensor or grad_is_dtensor
    ]
    if invalid:
        raise RuntimeError(
            "FSDP2 replicated-gradient synchronization received DTensor state "
            "for ignored parameter(s): " + ", ".join(invalid[:8])
        )

    active_grads: list[torch.Tensor] = []
    for (_, param), is_present in zip(named_params, globally_present):
        if not is_present:
            continue
        if param.grad is None:
            # Another rank used this parameter. Contribute zero locally so all
            # ranks execute the same reduction and obtain the correct average.
            param.grad = torch.zeros_like(param, memory_format=torch.preserve_format)
        elif param.grad.dtype != param.dtype:
            param.grad = param.grad.to(dtype=param.dtype)
        active_grads.append(param.grad)

    bucket_cap_bytes = max(1, int(bucket_cap_mb * 1024 * 1024))
    bucket: list[torch.Tensor] = []
    bucket_bytes = 0

    def flush_bucket() -> None:
        nonlocal bucket, bucket_bytes
        if not bucket:
            return
        if len(bucket) == 1:
            # Avoid duplicating a large gradient just to flatten one tensor.
            torch.distributed.all_reduce(bucket[0], op=torch.distributed.ReduceOp.SUM)
            bucket[0].div_(world_size)
        else:
            flat = torch.cat([grad.reshape(-1) for grad in bucket])
            torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
            flat.div_(world_size)
            offset = 0
            for grad in bucket:
                next_offset = offset + grad.numel()
                grad.copy_(flat[offset:next_offset].view_as(grad))
                offset = next_offset
        bucket = []
        bucket_bytes = 0

    with torch.no_grad():
        for grad in active_grads:
            grad_bytes = grad.numel() * grad.element_size()
            incompatible = bucket and (
                grad.device != bucket[0].device or grad.dtype != bucket[0].dtype
            )
            if incompatible or (
                bucket and bucket_bytes + grad_bytes > bucket_cap_bytes
            ):
                flush_bucket()
            bucket.append(grad)
            bucket_bytes += grad_bytes
            if bucket_bytes >= bucket_cap_bytes:
                flush_bucket()
        flush_bucket()


def _clip_grad_norm_fsdp2_mixed(
    parameters,
    max_norm: float,
    *,
    world_size: int,
) -> torch.Tensor:
    """Clip gradients for mixed FSDP2 DTensor and replicated plain params.

    ``torch.nn.utils.clip_grad_norm_`` can route through foreach/fused paths
    that assume a homogeneous tensor representation. This trainer deliberately
    has both FSDP2-managed DTensor gradients and ignored replicated Tensor
    gradients, so compute the norm and scaling explicitly.

    Replicated Tensor gradients have already been averaged across ranks by
    ``_sync_replicated_param_grads``; count one logical copy in the global norm
    by dividing their local contribution by ``world_size`` before all-reduce.
    DTensor gradients contribute their local shard, and the all-reduce sums the
    complete global parameter norm.
    """
    params = list(parameters)
    device = None
    total_sq = None

    for param in params:
        grad = param.grad
        if grad is None:
            continue
        local_grad = grad.to_local() if hasattr(grad, "to_local") else grad
        if device is None:
            device = local_grad.device
            total_sq = torch.zeros((), device=device, dtype=torch.float32)
        contrib = local_grad.detach().float().pow(2).sum()
        if not hasattr(grad, "to_local") and world_size > 1:
            contrib = contrib / world_size
        total_sq = total_sq + contrib

    if total_sq is None:
        fallback_device = params[0].device if params else torch.device("cpu")
        return torch.zeros((), device=fallback_device, dtype=torch.float32)

    if world_size > 1:
        torch.distributed.all_reduce(total_sq, op=torch.distributed.ReduceOp.SUM)
    total_norm = total_sq.sqrt()

    # If the norm is non-finite, report it and let the configured nan-guard
    # decide whether to skip or sanitize. Scaling by NaN would hide the source.
    if torch.isfinite(total_norm):
        clip_coef = max_norm / (total_norm.item() + 1e-6)
        if clip_coef < 1.0:
            for param in params:
                if param.grad is not None:
                    param.grad.mul_(clip_coef)
    return total_norm


# ---------------------------------------------------------------------------
# Process-group bootstrap
# ---------------------------------------------------------------------------


def init_distributed(dist_backend: str | None) -> tuple[int, int, torch.device]:
    """Initialize the default process group; returns (rank, world_size, device).

    Works under ``torchrun`` (env vars set) AND as a bare single process
    (world of 1 over gloo) so laptop smoke tests need no launcher.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is invalid for "
                f"{torch.cuda.device_count()} visible CUDA device(s)."
            )
        device = torch.device(f"cuda:{local_rank}")
        # Pin the process before NCCL creates any communicator. Relying on its
        # lazy initialization is fragile when an early collective is added.
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if not torch.distributed.is_initialized():
        if dist_backend is None or dist_backend == "auto":
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        else:
            backend = dist_backend
        # Fail fast when more workers than GPUs were launched: every worker
        # pins cuda:<local_rank>, so ranks beyond the device count would die
        # much later on an opaque "invalid device ordinal" from set_device.
        if (
            backend == "nccl"
            and world_size > 1
            and world_size > torch.cuda.device_count()
        ):
            n_gpus = max(torch.cuda.device_count(), 1)
            raise RuntimeError(
                f"Launched world_size={world_size} but only {n_gpus} CUDA "
                f"device(s) are visible — each worker needs its own GPU. "
                f"Relaunch with: torchrun --standalone "
                f"--nproc_per_node={n_gpus} ..."
            )
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        torch.distributed.init_process_group(
            backend=backend, rank=rank, world_size=world_size
        )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    return rank, world_size, device


# ---------------------------------------------------------------------------
# Checkpointing (consolidated, root-train.py-compatible schema)
# ---------------------------------------------------------------------------


def _local_rng_state() -> dict[str, Any]:
    """This rank's RNG state (weights_only-safe primitives + byte tensors).

    NumPy's global generator is deliberately omitted: nothing in the training
    path draws from it (DataLoader workers seed through torch's seed_worker;
    the HF stream is seeded explicitly)."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def _gather_rng_payload(rank: int, world_size: int) -> dict[str, Any]:
    """Per-rank RNG snapshot keyed by rank, so a resume at the same world
    size restores every rank's stream exactly."""
    payload: dict[str, Any] = {
        "format": "fsdp2_per_rank_v1",
        "world_size": world_size,
        "ranks": [_local_rng_state()],
    }
    if world_size > 1 and torch.distributed.is_initialized():
        gathered: list[Any] = [None] * world_size
        torch.distributed.all_gather_object(gathered, _local_rng_state())
        payload["ranks"] = gathered
    return payload


def _restore_rng_payload(state: Any, rank: int, world: int) -> bool:
    """Restore THIS rank's RNG entry; False when the payload mismatches."""
    if not isinstance(state, dict):
        return False
    if state.get("format") == "fsdp2_per_rank_v1":
        ranks = state.get("ranks") or []
        if world == 1 and len(ranks) == 1 and isinstance(ranks[0], dict):
            entry = ranks[0]
        elif len(ranks) == world and isinstance(ranks[rank], dict):
            entry = ranks[rank]
        else:
            return False
    elif world == 1 and "python" in state and "torch" in state:
        # Root-trainer checkpoint resumed by a world-of-one FSDP2 smoke run.
        entry = state
    else:
        return False
    random.setstate(entry["python"])
    torch.set_rng_state(entry["torch"].to(dtype=torch.uint8))
    if "cuda" in entry and torch.cuda.is_available():
        cuda_state = entry["cuda"]
        if isinstance(cuda_state, torch.Tensor):
            torch.cuda.set_rng_state(cuda_state.to(dtype=torch.uint8))
        elif len(cuda_state) == 1:
            torch.cuda.set_rng_state(cuda_state[0].to(dtype=torch.uint8))
        else:
            return False
    return True


def _consolidate_optimizer_state(
    opt: torch.optim.Optimizer, dtensor_cls: type
) -> dict[str, Any]:
    """``opt.state_dict()`` with DTensor values replaced by full CPU tensors.

    Preserves the exact ``{"state": {index: {...}}, "param_groups": [...]}``
    layout of a plain ``optimizer.state_dict()``. The positional layout is
    supported only inside the explicit FSDP2/AdamW checkpoint family.
    """
    sd = opt.state_dict()
    consolidated: dict[int, dict[str, Any]] = {}
    for idx, entries in sd["state"].items():
        consolidated[idx] = {
            key: (
                value.full_tensor().detach().cpu()
                if isinstance(value, dtensor_cls)
                else value.detach().cpu()
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in entries.items()
        }
    return {"state": consolidated, "param_groups": sd["param_groups"]}


def _reshard_optimizer_state(
    full_sd: dict[str, Any],
    opt: torch.optim.Optimizer,
    device: torch.device,
    distribute_tensor: Any,
) -> dict[str, Any]:
    """Inverse of :func:`_consolidate_optimizer_state`: turn full-tensor
    optimizer state back into placement-matched DTensors so stock
    ``load_state_dict`` sees values shaped like the parameters they feed."""
    # Optimizer indices are assigned across param_groups in construction order.
    index_to_param: dict[int, torch.nn.Parameter] = {}
    next_idx = 0
    for group in opt.param_groups:
        for p in group["params"]:
            index_to_param[next_idx] = p
            next_idx += 1

    state: dict[int, dict[str, Any]] = {}
    for idx, entries in full_sd["state"].items():
        p = index_to_param[int(idx)]
        restored: dict[str, Any] = {}
        for key, value in entries.items():
            if isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(p.shape):
                value = value.to(device=device, dtype=p.dtype)
                if hasattr(p, "device_mesh") and hasattr(p, "placements"):
                    restored[key] = distribute_tensor(
                        value, p.device_mesh, list(p.placements)
                    )
                else:
                    restored[key] = value
            elif isinstance(value, torch.Tensor):
                # Optimizer counters such as AdamW's scalar ``step`` are
                # replicated local state, not parameter-shaped DTensors. The
                # previous code attempted to Shard(0) these scalars and could
                # fail or stall every rank during resume.
                restored[key] = value.to(device=device)
            else:
                restored[key] = value
        state[int(idx)] = restored
    return {"state": state, "param_groups": full_sd["param_groups"]}


def save_checkpoint_fsdp2(
    *,
    model,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[Any],
    global_step: int,
    current_shard_idx: int,
    current_batch_idx: int,
    checkpoint_dir: Path,
    logger: logging.Logger,
    validator: Any | None = None,
    use_muon: bool = False,
    extra_payload: dict[str, Any] | None = None,
) -> None:
    """Consolidated checkpoint in the FSDP2 optimizer-policy family.

    Model/optimizer state is gathered to full CPU tensors via public APIs
    (every rank participates in those collectives); rank 0 alone writes the
    file atomically. At <=~200M params consolidation is sub-second; the
    sharded DCP format is the >1B-scale follow-up.
    """
    import train as train_mod  # lazy: heavy module

    api = _require_fsdp2()
    rank = torch.distributed.get_rank()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / train_mod.CHECKPOINT_FILENAME
    tmp_path = ckpt_path.with_suffix(".pth.tmp")

    config = train_mod.normalized_checkpoint_config(model)
    model_sd = api["get_model_state_dict"](
        model,
        options=api["StateDictOptions"](full_state_dict=True, cpu_offload=True),
    )

    payload: dict[str, Any] = {
        "checkpoint_schema_version": 3,
        "model_state_dict": model_sd,
        "config": config,
        "global_step": global_step,
        "current_shard_idx": current_shard_idx,
        "current_batch_idx": current_batch_idx,
        "rng_state": _gather_rng_payload(rank, torch.distributed.get_world_size()),
        "memory_nan_fix_id": MEMORY_NAN_FIX_ID,
        "use_muon": use_muon,
        "training_runtime": train_mod.checkpoint_runtime_contract(
            model,
            distributed_strategy="fsdp2",
            checkpoint_family=FSDP2_CHECKPOINT_FAMILY,
            optimizer_policy=_fsdp2_optimizer_policy(use_muon),
        ),
        "runtime_environment": _runtime_environment(torch.distributed.get_world_size()),
    }
    if validator is not None:
        # The cyclic cursor advances identically on every rank (replicated
        # params scoring the same fixed rows), so rank0's view is canonical.
        payload["validator_state_dict"] = validator.state_dict
    expected_count = 2 if use_muon else 1
    if len(optimizers) != expected_count or len(schedulers) != expected_count:
        raise RuntimeError(
            "FSDP2 checkpoint optimizer layout mismatch: "
            f"use_muon={use_muon}, optimizers={len(optimizers)}, "
            f"schedulers={len(schedulers)}, expected={expected_count}."
        )
    if use_muon:
        payload["muon_optimizer_state_dict"] = _consolidate_optimizer_state(
            optimizers[0], api["DTensor"]
        )
        payload["muon_scheduler_state_dict"] = schedulers[0].state_dict()
        adam_idx = 1
    else:
        adam_idx = 0
    payload["adam_optimizer_state_dict"] = _consolidate_optimizer_state(
        optimizers[adam_idx], api["DTensor"]
    )
    payload["adam_scheduler_state_dict"] = schedulers[adam_idx].state_dict()
    if extra_payload:
        payload.update(extra_payload)

    if rank == 0:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, ckpt_path)
        with (checkpoint_dir / train_mod.CONFIG_FILENAME).open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(config, f, indent=2)
        logger.info(
            "Checkpoint saved step=%d shard=%d batch=%d path=%s",
            global_step,
            current_shard_idx,
            current_batch_idx,
            ckpt_path,
        )
    if torch.distributed.get_world_size() > 1:
        # Keep ranks in lockstep before the next training collective.
        torch.distributed.barrier()


def load_checkpoint_fsdp2(
    *,
    model,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[Any],
    checkpoint_dir: Path,
    device: torch.device,
    logger: logging.Logger,
    data_runtime: dict[str, Any],
    validator: Any | None = None,
    use_muon: bool = False,
    dl_generator: torch.Generator | None = None,
) -> tuple[int, int, int]:
    """Resume from a policy-compatible FSDP2 consolidated checkpoint.

    Every rank reads the shared file and re-shards locally through the
    public state-dict APIs. weights_only-first loading mirrors train.py;
    the fallback warning below carries the same pickle caveat.
    """
    import train as train_mod

    api = _require_fsdp2()
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()

    ckpt_path = checkpoint_dir / "model_ckpt.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        from torch.torch_version import TorchVersion

        torch.serialization.add_safe_globals([TorchVersion])
    except (ImportError, AttributeError):
        pass  # Older PyTorch versions lack TorchVersion or add_safe_globals

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - legacy payloads need full pickle
        logger.warning(
            "weights_only load failed (%s); retrying with weights_only=False. "
            "Only resume from checkpoints you trust: pickle deserialization "
            "executes arbitrary code.",
            exc,
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    train_mod.validate_resume_runtime_contract(
        checkpoint,
        model,
        logger,
        distributed_strategy="fsdp2",
        checkpoint_family=FSDP2_CHECKPOINT_FAMILY,
        optimizer_policy=_fsdp2_optimizer_policy(use_muon),
    )

    if "current_batch_idx" not in checkpoint:
        raise RuntimeError(
            "This FSDP2 checkpoint predates exact intra-shard cursors and cannot "
            "be resumed without replaying data. Start a fresh run, or explicitly "
            "convert a checkpoint known to have been saved at a shard boundary."
        )
    saved_data_runtime = checkpoint.get("data_runtime")
    if saved_data_runtime != data_runtime:
        raise RuntimeError(
            "Exact FSDP2 resume data contract mismatch: "
            f"checkpoint={saved_data_runtime!r}, current={data_runtime!r}. "
            "Resume with the original seed, per-rank batch size, accumulation, "
            "and sequence length."
        )

    saved_environment = checkpoint.get("runtime_environment")
    if isinstance(saved_environment, dict):
        saved_world_size = saved_environment.get("world_size")
        if saved_world_size is not None and int(saved_world_size) != world:
            raise RuntimeError(
                "Exact FSDP2 resume requires the checkpoint world size: "
                f"checkpoint={saved_world_size}, current={world}. Start a fresh "
                "run when changing world size."
            )

    api["set_model_state_dict"](
        model,
        checkpoint["model_state_dict"],
        options=api["StateDictOptions"](full_state_dict=True),
    )

    ckpt_use_muon = bool(
        checkpoint.get("use_muon", "muon_optimizer_state_dict" in checkpoint)
    )
    if ckpt_use_muon != use_muon:
        raise RuntimeError(
            f"Checkpoint was saved with use_muon={ckpt_use_muon} but this run "
            f"uses use_muon={use_muon}; FSDP2 optimizer states are incompatible. "
            "Start fresh or rerun with the matching --no-muon setting."
        )

    required_state = ["adam_optimizer_state_dict", "adam_scheduler_state_dict"]
    if ckpt_use_muon:
        required_state.extend(
            ["muon_optimizer_state_dict", "muon_scheduler_state_dict"]
        )
    train_mod._require_resume_keys(checkpoint, required_state)

    expected_count = 2 if ckpt_use_muon else 1
    if len(optimizers) != expected_count or len(schedulers) != expected_count:
        raise RuntimeError(
            "FSDP2 resume optimizer layout mismatch: "
            f"use_muon={ckpt_use_muon}, optimizers={len(optimizers)}, "
            f"schedulers={len(schedulers)}, expected={expected_count}."
        )

    if ckpt_use_muon:
        optimizers[0].load_state_dict(
            _reshard_optimizer_state(
                checkpoint["muon_optimizer_state_dict"],
                optimizers[0],
                device,
                api["distribute_tensor"],
            )
        )
        schedulers[0].load_state_dict(checkpoint["muon_scheduler_state_dict"])
        adam_idx = 1
    else:
        adam_idx = 0

    optimizers[adam_idx].load_state_dict(
        _reshard_optimizer_state(
            checkpoint["adam_optimizer_state_dict"],
            optimizers[adam_idx],
            device,
            api["distribute_tensor"],
        )
    )
    schedulers[adam_idx].load_state_dict(checkpoint["adam_scheduler_state_dict"])

    optimizer_state_counts = [len(opt.state) for opt in optimizers]
    if ckpt_use_muon:
        saved_optimizer_counts = [
            len(checkpoint["muon_optimizer_state_dict"].get("state", {})),
            len(checkpoint["adam_optimizer_state_dict"].get("state", {})),
        ]
        saved_scheduler_epochs = [
            checkpoint["muon_scheduler_state_dict"].get("last_epoch"),
            checkpoint["adam_scheduler_state_dict"].get("last_epoch"),
        ]
    else:
        saved_optimizer_counts = [
            len(checkpoint["adam_optimizer_state_dict"].get("state", {}))
        ]
        saved_scheduler_epochs = [
            checkpoint["adam_scheduler_state_dict"].get("last_epoch")
        ]
    scheduler_epochs = [sched.last_epoch for sched in schedulers]
    if optimizer_state_counts != saved_optimizer_counts:
        raise RuntimeError(
            "FSDP2 optimizer state restore count mismatch: "
            f"checkpoint={saved_optimizer_counts}, loaded={optimizer_state_counts}."
        )
    if scheduler_epochs != saved_scheduler_epochs:
        raise RuntimeError(
            "FSDP2 scheduler state restore mismatch: "
            f"checkpoint={saved_scheduler_epochs}, loaded={scheduler_epochs}."
        )

    rng_restored = _restore_rng_payload(checkpoint.get("rng_state"), rank, world)
    if not rng_restored:
        saved_ws = (checkpoint.get("rng_state") or {}).get("world_size", "unknown")
        logger.warning(
            "RNG state missing or saved under a different layout/world size "
            "(%s) — continuing with fresh RNG. Intra-shard data order still "
            "replays deterministically via the seeded DistributedSampler.",
            saved_ws,
        )

    if dl_generator is not None:
        gen_state = checkpoint.get("dl_generator_state")
        if gen_state is not None:
            dl_generator.set_state(gen_state.to(dtype=torch.uint8))
        else:
            logger.warning(
                "Resume DataLoader RNG diagnostic FAIL: dl_generator_state missing."
            )

    logger.info(
        "Resume state diagnostics %s | rank=%d optimizer_entries=%s "
        "scheduler_last_epoch=%s rng_restored=%s dataloader_rng_restored=%s",
        "PASS" if rng_restored else "WARN",
        rank,
        optimizer_state_counts,
        scheduler_epochs,
        rng_restored,
        dl_generator is None or checkpoint.get("dl_generator_state") is not None,
    )

    if validator is not None and "validator_state_dict" in checkpoint:
        validator.load_state_dict(checkpoint["validator_state_dict"])

    # Scalar-config drift detection (mirrors train.load_checkpoint).
    saved_cfg = checkpoint.get("config")
    if isinstance(saved_cfg, dict):
        current_cfg = dataclasses.asdict(model.config)
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
    current_batch_idx = int(checkpoint.get("current_batch_idx", 0))
    logger.info(
        "Resumed from %s | step=%d shard=%d batch=%d fix_id=%s optimizer=%s",
        ckpt_path,
        global_step,
        current_shard_idx,
        current_batch_idx,
        ckpt_fix_id if ckpt_fix_id is not None else "unknown",
        _fsdp2_optimizer_policy(use_muon),
    )
    return global_step, current_shard_idx, current_batch_idx


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------


def build_fsdp2_optimizers(
    model,
    *,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[list[torch.optim.Optimizer], bool, dict[str, Any]]:
    """Build DTensor Muon + unfused AdamW, or an AdamW-only fallback.

    FSDP2 parameters are DTensors while deliberately ignored custom-math
    parameters are ordinary replicated tensors. Stock unfused AdamW provides
    local elementwise updates for both. :class:`MuonDTensor` owns only sharded
    hidden 2-D matrices and gathers each complete momentum matrix before
    Newton--Schulz orthogonalization.
    """
    adam_params, muon_params, inventory = split_muon_adam_params(model)
    total_params = sum(p.numel() for p in model.parameters())
    adam_count = sum(p.numel() for p in adam_params)
    muon_count = sum(p.numel() for p in muon_params)
    enable_muon = not args.no_muon and bool(muon_params)

    shared_lr = args.lr
    resolved_muon_lr = args.muon_lr if args.muon_lr is not None else shared_lr
    resolved_adam_lr = args.adam_lr if args.adam_lr is not None else shared_lr

    meta: dict[str, Any] = {
        "enable_muon": enable_muon,
        "optimizer_policy": _fsdp2_optimizer_policy(enable_muon),
        "adam_param_count": adam_count if enable_muon else total_params,
        "muon_param_count": muon_count if enable_muon else 0,
        "adam_pct": 100.0
        * (adam_count if enable_muon else total_params)
        / max(total_params, 1),
        "muon_pct": 100.0 * (muon_count if enable_muon else 0) / max(total_params, 1),
        "total_params": total_params,
        "muon_lr": resolved_muon_lr if enable_muon else None,
        "adam_lr": resolved_adam_lr,
        "weight_decay": args.weight_decay,
        "muon_momentum": args.muon_momentum if enable_muon else None,
        "muon_adjust_lr_fn": args.muon_adjust_lr_fn if enable_muon else None,
        "muon_gather_buffer_mb": (args.muon_gather_buffer_mb if enable_muon else None),
        "inventory": inventory,
    }

    logger.info(
        "optimizer split: adamw=%.2f%% (%d tensors, %s params) "
        "muon=%.2f%% (%d tensors, %s params) total=%.3fB",
        meta["adam_pct"],
        len(inventory["adamw"]) if enable_muon else len(list(model.parameters())),
        f"{meta['adam_param_count']:,}",
        meta["muon_pct"],
        len(inventory["muon"]) if enable_muon else 0,
        f"{meta['muon_param_count']:,}",
        total_params / 1e9,
    )
    logger.debug(
        "AdamW params: %s",
        inventory["adamw"]
        if enable_muon
        else [name for name, _ in model.named_parameters()],
    )
    logger.debug("Muon params: %s", inventory["muon"] if enable_muon else [])

    def _adamw(params_list: list[torch.nn.Parameter]) -> torch.optim.AdamW:
        adam_decay = [p for p in params_list if not _is_adamw_no_decay(p)]
        adam_no_decay = [p for p in params_list if _is_adamw_no_decay(p)]
        logger.info(
            "AdamW(lr=%.3e, betas=(%.2f, %.2f), wd=%.3g on %d params / "
            "wd=0 on %d params, fused=False, foreach=False)",
            resolved_adam_lr,
            args.adam_beta1,
            args.adam_beta2,
            args.weight_decay,
            len(adam_decay),
            len(adam_no_decay),
        )
        return torch.optim.AdamW(
            [
                {"params": adam_decay, "weight_decay": args.weight_decay},
                {"params": adam_no_decay, "weight_decay": 0.0},
            ],
            lr=resolved_adam_lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            fused=False,  # fused kernels are not DTensor-safe
            foreach=False,  # groups mix DTensors and replicated plain tensors
        )

    if enable_muon:
        if args.adam_lr is None and resolved_adam_lr >= 5e-4:
            logger.warning(
                "Muon and AdamW are both running at lr=%.3e. AdamW owns the "
                "embeddings and LM head; consider --adam-lr 3e-4 if this "
                "shared rate is unstable.",
                resolved_adam_lr,
            )
        muon_optim = MuonDTensor(
            muon_params,
            lr=resolved_muon_lr,
            weight_decay=args.weight_decay,
            momentum=args.muon_momentum,
            nesterov=not args.no_muon_nesterov,
            ns_steps=args.muon_ns_steps,
            adjust_lr_fn=args.muon_adjust_lr_fn,
            gather_buffer_size_mb=args.muon_gather_buffer_mb,
        )
        adam_optim = _adamw(adam_params)
        logger.info(
            "MuonDTensor(lr=%.3e, wd=%.3g, momentum=%.3g, nesterov=%s, "
            "ns_steps=%d, adjust_lr_fn=%s, gather_buffer=%.1f MiB) + AdamW",
            resolved_muon_lr,
            args.weight_decay,
            args.muon_momentum,
            not args.no_muon_nesterov,
            args.muon_ns_steps,
            args.muon_adjust_lr_fn,
            args.muon_gather_buffer_mb,
        )
        return [muon_optim, adam_optim], True, meta

    if not args.no_muon and not muon_params:
        logger.warning(
            "Muon requested, but no eligible hidden 2-D matrices were found; "
            "falling back to AdamW for all parameters."
        )

    logger.info("Muon disabled — AdamW-only fallback for all parameters")
    return [_adamw(list(model.parameters()))], False, meta


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace, logger: logging.Logger) -> None:
    _require_heavy_imports()
    from transformers import AutoTokenizer

    import train as train_mod
    from utils.dataset import (
        MmapShardDataset,
        TokenizedShardProducer,
        verify_tokenizer_vocab,
    )
    from utils.validation import WikiTextCyclicValidator

    rank, world_size, device = init_distributed(args.dist_backend)
    is_rank0 = rank == 0
    runtime_environment = _runtime_environment(world_size)
    if is_rank0:
        logger.info("FSDP2 runtime environment: %s", runtime_environment)

    run_dir = Path(args.run_dir)
    jsonl_path = Path(args.log_jsonl) if args.log_jsonl else None
    if is_rank0:
        run_dir.mkdir(parents=True, exist_ok=True)
        if jsonl_path is not None:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_jsonl(record: dict[str, Any]) -> None:
        if is_rank0 and jsonl_path is not None:
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    # Identical seed on EVERY rank => identical fp32 init => fully_shard
    # shards one coherent model. Data divergence comes from the sampler only.
    train_mod.set_seed(args.seed, deterministic=args.deterministic)

    tokens_per_shard = train_mod.align_tokens_per_shard(
        args.tokens_per_shard, args.seq_len
    )
    if tokens_per_shard != args.tokens_per_shard and is_rank0:
        logger.warning(
            "Adjusted tokens_per_shard %d -> %d (multiple of seq_len+1=%d)",
            args.tokens_per_shard,
            tokens_per_shard,
            args.seq_len + 1,
        )

    # Shard PRODUCTION is rank0-only; consumers share the files read-only.
    # Every rank still needs the tokenizer object itself (BOS/EOS ids for the
    # config + validation tokenization), so non-zero ranks load it read-only
    # from the HF cache.
    producer: TokenizedShardProducer | None = None
    if is_rank0:
        producer = TokenizedShardProducer(
            cache_dir=args.cache_dir,
            tokenizer_name=args.tokenizer_name,
            tokens_per_shard=tokens_per_shard,
            max_buffered_files=args.max_buffered_files,
            seed=args.seed,
            log_fn=logger.info,
        )
        tokenizer = producer.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    verify_tokenizer_vocab(tokenizer, args.vocab_size)

    cfg = train_mod.build_training_config(vocab_size=args.vocab_size)
    # `is not None` (not `or`): a legitimate token id of 0 must not be
    # clobbered by the config default.
    if tokenizer.bos_token_id is not None:
        cfg.bos_token_id = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        cfg.eos_token_id = tokenizer.eos_token_id

    if args.no_fused_mamba:
        # Diagnostic knob: force the PyTorch scan tiers (e.g. to isolate a
        # suspected mamba-ssm x FSDP2 interaction without uninstalling).
        cfg.use_fused_mamba_scan = False
    logger.info(log_mamba_backend(cfg))
    logger.info(
        "rank=%d/%d device=%s dist_backend=%s fused_mamba=%s memory_nan_fix=%s",
        rank,
        world_size,
        device,
        torch.distributed.get_backend(),
        fused_mamba_scan_available(),
        MEMORY_NAN_FIX_ID,
    )

    model = train_mod.HybridForCausalLM(cfg).to(device)
    fsdp2_ignored_params = _prepare_fsdp2_custom_math_params(model)
    fsdp2_replicated_params = _ordered_replicated_params(model, fsdp2_ignored_params)
    train_mod.configure_gradient_checkpointing(
        model, args.gradient_checkpointing, logger
    )
    # All ranks must construct the same autograd graph. Catch launcher/CLI
    # drift before fully_shard collectives turn it into an opaque NCCL hang.
    local_gc_contract = torch.tensor(
        [
            int(model.is_gradient_checkpointing),
            int(model.config.gradient_checkpointing_use_reentrant),
            int(model.config.use_cache),
        ],
        device=device,
        dtype=torch.int32,
    )
    contract_min = local_gc_contract.clone()
    contract_max = local_gc_contract.clone()
    if world_size > 1:
        torch.distributed.all_reduce(contract_min, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(contract_max, op=torch.distributed.ReduceOp.MAX)
    if not torch.equal(contract_min, contract_max):
        raise RuntimeError(
            "Gradient-checkpointing/use_cache settings differ across ranks; "
            "refusing to enter FSDP2 training because collective graphs can deadlock."
        )
    logger.info(
        "Distributed checkpointing diagnostics PASS | strategy=fsdp2 (not DDP; "
        "find_unused_parameters/static_graph=N/A) rank_contract=%s",
        local_gc_contract.tolist(),
    )
    # Calibration AFTER process-group init (its rank0 broadcast engages) and
    # BEFORE fully_shard (plain params/buffers, no DTensor plumbing yet). It
    # lives on the inner HybridModel, not the HybridForCausalLM wrapper.
    # Must stay before ``fully_shard``: calibration directly executes each
    # Mamba block with a dummy input, so the standard Mamba projections must
    # still be ordinary tensors rather than FSDP2-managed DTensors.
    model.model.calibrate_ssm_norm_thresholds()
    reset_mamba_scan_stats()
    n_params = count_trainable_params(model)
    if is_rank0:
        # numel() on Shard(0) DTensors reports GLOBAL counts.
        logger.info("trainable_params=%s (%.3fB)", f"{n_params:,}", n_params / 1e9)
        logger.info(
            "FSDP2 replicated custom-math params=%d (RMSNorm/memory-bank/Mamba "
            "vectors); gradients are all-reduced manually",
            len(fsdp2_ignored_params),
        )

    api = _require_fsdp2()
    # Params stay fp32; the autocast below provides bf16 compute, mirroring
    # train.py. MixedPrecisionPolicy(param_dtype=bfloat16) would swap weights
    # to bf16 inside forward and crash layer.py's aux-loss block, which runs
    # under autocast(enabled=False) on fp32-promoted activations.
    mp_policy = api["MixedPrecisionPolicy"](reduce_dtype=torch.float32)
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16  # main() rejects anything else
    fully_shard = api["fully_shard"]
    # Inner-to-outer: layers first (independent communication groups), then
    # the root gathers everything left outside the layers. With activation
    # checkpointing, the layer's backward replay executes regular nn.Linear
    # calls while FSDP2 params would otherwise be back in DTensor form after
    # the original forward. Keeping checkpointed layer params unsharded until
    # backward completes preserves the Tensor/parameter contract for replay.
    layer_reshard_after_forward = not args.gradient_checkpointing
    for layer in model.model.layers:
        layer_param_ids = {id(param) for param in layer.parameters()}
        layer_ignored_params = {
            param for param in fsdp2_ignored_params if id(param) in layer_param_ids
        }
        fully_shard(
            layer,
            mp_policy=mp_policy,
            ignored_params=layer_ignored_params,
            reshard_after_forward=layer_reshard_after_forward,
        )
    fully_shard(model, mp_policy=mp_policy, ignored_params=fsdp2_ignored_params)
    logger.info(
        "fully_shard applied: layers=%d world=%d mp=%s layer_reshard_after_forward=%s",
        len(model.model.layers),
        world_size,
        f"{'autocast-bf16' if use_amp else 'fp32'}/fp32-params/fp32-reduce",
        layer_reshard_after_forward,
    )

    optimizers, use_muon, opt_meta = build_fsdp2_optimizers(
        model, args=args, logger=logger
    )

    validator: Any | None = None
    if not args.no_validation:
        try:
            validator = WikiTextCyclicValidator(
                tokenizer,
                seq_len=args.seq_len,
                num_rows=args.val_rows,
                batch_size=args.val_batch_size,
                dataset_config=args.val_dataset_config,
                bos_id=cfg.bos_token_id,
                eos_id=cfg.eos_token_id,
                pad_token_id=cfg.pad_token_id,
            )
            if is_rank0:
                logger.info(
                    "Cyclic validation enabled | dataset=Salesforce/wikitext/%s "
                    "rows=%d batch=%d interval=%d",
                    args.val_dataset_config,
                    args.val_rows,
                    args.val_batch_size,
                    args.val_interval,
                )
        except Exception:
            logger.exception(
                "Failed to initialize WikiTextCyclicValidator; disabling validation"
            )
            validator = None

    warmup_steps = train_mod._resolve_warmup_steps(args.warmup_steps, args.max_steps)
    lr_lambda = train_mod._build_lr_lambda(
        warmup_steps, args.max_steps, args.min_lr_ratio
    )
    schedulers: list[Any] = [
        torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        for opt in optimizers
    ]
    if is_rank0:
        logger.info(
            "LR schedule: muon_peak=%s adam_peak=%g warmup=%d%s cosine_floor=%.2f",
            f"{opt_meta['muon_lr']:g}" if use_muon else "disabled",
            opt_meta["adam_lr"],
            warmup_steps,
            " (auto)" if args.warmup_steps <= 0 else "",
            args.min_lr_ratio,
        )

    global_step = 0
    current_shard_idx = 0
    current_batch_idx = 0
    ckpt_dir = Path(args.ckpt_dir)

    # Seeds DataLoader workers; intra-shard ORDER comes from the seeded
    # DistributedSampler (epoch = shard index), which needs no persistence.
    dl_generator = torch.Generator()
    dl_generator.manual_seed(args.seed)
    data_runtime = _data_runtime_contract(args)

    if args.resume:
        global_step, current_shard_idx, current_batch_idx = load_checkpoint_fsdp2(
            model=model,
            optimizers=optimizers,
            schedulers=schedulers,
            checkpoint_dir=ckpt_dir,
            device=device,
            logger=logger,
            data_runtime=data_runtime,
            validator=validator,
            use_muon=use_muon,
            dl_generator=dl_generator,
        )
        # LambdaLR.load_state_dict restores `_last_lr` but not
        # param_group['lr'], and opt.step() precedes sched.step() — without
        # this sync the first post-resume step runs at constructor LR.
        for opt, sched in zip(optimizers, schedulers):
            saved_lrs = sched.get_last_lr()
            if len(saved_lrs) != len(opt.param_groups):
                raise RuntimeError(
                    "Scheduler/optimizer param-group mismatch after resume: "
                    f"scheduler_lrs={len(saved_lrs)} groups={len(opt.param_groups)}."
                )
            for group, saved_lr in zip(opt.param_groups, saved_lrs):
                group["lr"] = saved_lr

    stop_event = threading.Event()
    producer_thread: threading.Thread | None = None
    ce_smooth = train_mod.RollingAverage(args.smooth_window)
    # Device-side metric accumulators (zero-sync design inherited from
    # train.py): summed on-device, ONE D2H transfer per --log-interval window
    # — now of all-reduced GLOBAL sums, so JSONL means describe the fleet.
    metric_sum: torch.Tensor | None = None
    metric_fin_cnt: torch.Tensor | None = None
    metric_bad_cnt: torch.Tensor | None = None
    metric_count = 0
    # Schedule scales are pure-Python values derived from (step, max_steps)
    # — identical on every rank, so host-side averaging needs no reduction.
    assoc_scale_sum = 0.0
    expert_scale_sum = 0.0
    step_window_started: float | None = None
    metric_main_names: list[str] | None = None
    metric_gate_names: list[str] | None = None

    accum = max(1, args.grad_accum_steps)
    if accum > 1 and is_rank0:
        logger.info(
            "Gradient accumulation: %d micro-batches per optimizer step "
            "(effective batch=%d)",
            accum,
            args.batch_size * world_size * accum,
        )

    watchdog = train_mod.StepProgressWatchdog(
        logger, args.step_watchdog_seconds, rank=rank
    )
    watchdog.start()

    def _save_checkpoint() -> None:
        watchdog.progress(global_step, "checkpoint_save")
        save_checkpoint_fsdp2(
            model=model,
            optimizers=optimizers,
            schedulers=schedulers,
            global_step=global_step,
            current_shard_idx=current_shard_idx,
            current_batch_idx=current_batch_idx,
            checkpoint_dir=ckpt_dir,
            logger=logger,
            validator=validator,
            use_muon=use_muon,
            extra_payload={
                "dl_generator_state": dl_generator.get_state(),
                "optimizer_policy": opt_meta.get("optimizer_policy"),
                "muon_adjust_lr_fn": opt_meta.get("muon_adjust_lr_fn"),
                "muon_gather_buffer_mb": opt_meta.get("muon_gather_buffer_mb"),
                "data_runtime": data_runtime,
            },
        )
        if is_rank0 and producer is not None:
            producer.save_checkpoint(args.dataset_ckpt_path)

    try:
        # NOTE: the consume cursor (`current_shard_idx`) stays authoritative;
        # never copy the producer's write cursor over it (train.py gotcha).
        if (
            args.resume
            and producer is not None
            and os.path.exists(args.dataset_ckpt_path)
        ):
            producer.load_checkpoint(args.dataset_ckpt_path)

        if producer is not None:
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
            # Producer writes shard.bin atomically (tmp+replace), then
            # publishes shard.json; gating on BOTH closes the torn-shard race.
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
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=True,  # uneven tails would deadlock collective bwd
            )
            sampler.set_epoch(current_shard_idx)  # replayable permutation
            resume_sample_offset = current_batch_idx * args.batch_size
            sampler_for_loader = _OffsetSampler(sampler, resume_sample_offset)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=sampler_for_loader,
                pin_memory=device.type == "cuda",
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
                prefetch_factor=2 if args.num_workers > 0 else None,
                worker_init_fn=(
                    train_mod.seed_worker if args.num_workers > 0 else None
                ),
                generator=dl_generator,
                drop_last=True,
            )

            model.train()
            if is_rank0:
                logger.info(
                    "Shard %d | sequences=%d | remaining per-rank batches=%d | "
                    "resume batch=%d | step %d/%d",
                    current_shard_idx,
                    len(dataset),
                    len(dataloader),
                    current_batch_idx,
                    global_step,
                    args.max_steps,
                )

            shard_fully_consumed = True
            batches_iter = iter(dataloader)
            shard_resume_batch_idx = current_batch_idx
            batches_seen = current_batch_idx
            while True:
                if global_step >= args.max_steps:
                    # Mid-shard stop: leave the shard unmarked and the cursor
                    # where it is so an extended --max-steps resumes from this
                    # exact position instead of dropping the unconsumed tail.
                    shard_fully_consumed = False
                    break

                # ---- collect `accum` micro-batches for one optimizer step --
                micro_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
                watchdog.progress(global_step, "data_loading")
                while len(micro_inputs) < accum:
                    try:
                        input_ids, labels = next(batches_iter)
                    except StopIteration:
                        break
                    input_ids = input_ids.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    # Shards are immutable memmaps: first batch of every shard
                    # plus the interval cadence catches corrupt files / vocab
                    # drift at ~1/Nth of the pipeline cost (each check syncs).
                    if batches_seen == shard_resume_batch_idx or (
                        args.validate_token_interval > 0
                        and global_step % args.validate_token_interval == 0
                    ):
                        train_mod.validate_token_batch(
                            input_ids,
                            cfg.vocab_size,
                            labels=labels,
                            ignore_index=cfg.label_ignore_index,
                        )
                    micro_inputs.append((input_ids, labels))
                    batches_seen += 1

                if not micro_inputs:
                    break  # shard exhausted cleanly
                if len(micro_inputs) < accum:
                    # Partial tail at shard end: drop it rather than apply a
                    # mis-scaled update (<= accum-1 batches lost per boundary).
                    break

                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                if step_window_started is None:
                    step_window_started = time.perf_counter()

                # Micro-steps run with gradient sync OFF; the last micro-step
                # re-enables it so the WHOLE accumulated gradient is
                # reduce-scattered exactly once per optimizer step.
                if accum > 1:
                    model.set_requires_gradient_sync(False, recurse=True)
                step_metric_sums: dict[str, torch.Tensor] = {}
                step_gate_sums: dict[str, torch.Tensor] = {}
                schedule_scales: dict[str, float] = {}
                for micro_idx, (m_ids, m_labels) in enumerate(micro_inputs):
                    if accum > 1 and micro_idx == len(micro_inputs) - 1:
                        model.set_requires_gradient_sync(True, recurse=True)
                    watchdog.progress(
                        global_step,
                        f"forward_micro_{micro_idx + 1}/{len(micro_inputs)}",
                    )
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=use_amp,
                        # Keep sibling checkpoint regions and their independent
                        # backward replays on the same parameter-cast path.
                        cache_enabled=not args.gradient_checkpointing,
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
                    weighted_t = train_mod._weighted_term_tensors(
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
                        scalar = train_mod._as_flush_scalar(value, outputs.loss.device)
                        step_metric_sums[name] = step_metric_sums.get(name, 0) + scalar
                    for name, value in (outputs.gate_stats or {}).items():
                        scalar = train_mod._as_flush_scalar(value, outputs.loss.device)
                        step_gate_sums[name] = step_gate_sums.get(name, 0) + scalar
                    # Mean-of-means: equal-size micro-batches (drop_last), so
                    # dividing by count averages correctly.
                    watchdog.progress(
                        global_step,
                        f"backward_micro_{micro_idx + 1}/{len(micro_inputs)}",
                    )
                    (outputs.loss / len(micro_inputs)).backward()

                micro_count = len(micro_inputs)
                scalars = _mean_metric_sums(step_metric_sums, micro_count)
                scalars.update(schedule_scales)
                gate_stats = _mean_metric_sums(step_gate_sums, micro_count)

                watchdog.progress(global_step, "optimizer_and_collectives")
                _sync_replicated_param_grads(
                    fsdp2_replicated_params, world_size=world_size
                )
                # Params are fp32 masters (see the mp_policy note above), so
                # gradients arrive fp32 and clip sees TRUE magnitudes — no
                # scaler unscale step exists on this bf16-autocast path.
                grad_norm = _clip_grad_norm_fsdp2_mixed(
                    model.parameters(),
                    args.max_grad_norm,
                    world_size=world_size,
                )
                allow_update = True
                if args.grad_nan_guard == "strict":
                    # The deliberate per-step D2H sync buys TRUE skip
                    # semantics. Under data parallelism the vote must be
                    # GLOBAL: skipping on one rank alone would leave its
                    # parameter shard permanently out of sync with the rest.
                    finite_flag = torch.isfinite(grad_norm).to(torch.int32)
                    if world_size > 1:
                        torch.distributed.all_reduce(
                            finite_flag, op=torch.distributed.ReduceOp.MIN
                        )
                    allow_update = bool(finite_flag)
                    if not allow_update and is_rank0:
                        logger.error(
                            "Non-finite gradient norm at step %d — optimizer "
                            "update SKIPPED on some rank "
                            "(--grad-nan-guard=strict)",
                            global_step,
                        )
                        _write_jsonl(
                            {
                                "event": "non_finite_grad",
                                "step": global_step,
                                "shard_idx": current_shard_idx,
                                **train_mod._non_finite_diagnosis(
                                    model, outputs, global_step, args.max_steps
                                ),
                            }
                        )
                elif args.grad_nan_guard == "sanitize":
                    # A non-finite norm makes clip's coef NaN, spreading NaN
                    # onto EVERY gradient (norm-bad <=> some grad bad). Zero
                    # them asynchronously so optimizer moments stay finite.
                    # Elementwise op => dispatches locally on DTensor shards;
                    # the union of local shards covers the whole gradient.
                    for p in model.parameters():
                        if p.grad is not None:
                            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                # 'monitor': zero protection — flush-time counters report.

                if allow_update:
                    for opt in optimizers:
                        opt.step()
                    for sched in schedulers:
                        sched.step()
                # This is the last data cursor known to correspond to a fully
                # completed training iteration. If forward/backward raises,
                # exception checkpointing retains the previous committed
                # cursor and replays the interrupted update on resume.
                current_batch_idx = batches_seen
                watchdog.progress(global_step, "metrics_and_collectives")
                # When strict skipped, the NaN grads die via zero_grad(
                # set_to_none=True) at the top of the next iteration.

                # ---- metrics: accumulate on-device, transfer on flush ------
                scalars["grad_norm"] = grad_norm
                assoc_scale_sum += schedule_scales["assoc_scale"]
                expert_scale_sum += schedule_scales["expert_scale"]
                # Schema is static for a fixed config; cache the name lists.
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
                        train_mod._as_flush_scalar(scalars[n], outputs.loss.device)
                        for n in main_names
                    ]
                    + [
                        train_mod._as_flush_scalar(gate_stats[n], outputs.loss.device)
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

                if global_step % args.log_interval == 0:
                    assert (
                        metric_sum is not None
                        and metric_fin_cnt is not None
                        and metric_bad_cnt is not None
                        and metric_main_names is not None
                        and metric_gate_names is not None
                    )
                    # Global masked means: reduce the window's sums/counts
                    # across ranks FIRST, then divide — one collective per
                    # flush window, still exactly one host transfer.
                    if world_size > 1:
                        packed = torch.stack(
                            [metric_sum, metric_fin_cnt, metric_bad_cnt]
                        )
                        torch.distributed.all_reduce(
                            packed, op=torch.distributed.ReduceOp.SUM
                        )
                        metric_sum = packed[0]
                        metric_fin_cnt = packed[1]
                        metric_bad_cnt = packed[2]
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
                        "adam_lr": float(schedulers[-1].get_last_lr()[0]),
                        "step_time_s": step_time_s,
                        "gate_stats": dict(zip(metric_gate_names, values[n_main:])),
                    }
                    if use_muon:
                        record["muon_lr"] = float(schedulers[0].get_last_lr()[0])
                    bad_names = {
                        name: int(cnt)
                        for name, cnt in list(
                            zip(metric_main_names, bad_counts[:n_main])
                        )
                        + list(zip(metric_gate_names, bad_counts[n_main:]))
                        if cnt > 0.0
                    }
                    if bad_names:
                        # Aggregated non_finite_metrics event (diagnosis from
                        # the last step of the window; rank0 emits because the
                        # counters above were already all-reduced globally).
                        diagnosis = train_mod._non_finite_diagnosis(
                            model, outputs, global_step, args.max_steps
                        )
                        if is_rank0:
                            logger.error(
                                "Non-finite metrics in steps %d-%d: %s",
                                global_step - metric_count + 1,
                                global_step,
                                bad_names,
                            )
                            _write_jsonl(
                                {
                                    "event": "non_finite_metrics",
                                    "step": global_step,
                                    "shard_idx": current_shard_idx,
                                    "window_steps": metric_count,
                                    "counts": bad_names,
                                    **diagnosis,
                                }
                            )
                    metric_sum = None
                    metric_fin_cnt = None
                    metric_bad_cnt = None
                    metric_count = 0
                    assoc_scale_sum = 0.0
                    expert_scale_sum = 0.0
                    step_window_started = None

                    if (
                        validator is not None
                        and args.val_interval > 0
                        and global_step % args.val_interval == 0
                    ):
                        watchdog.progress(global_step, "validation_collectives")
                        # Every rank evaluates the SAME replicated buffers +
                        # sharded params on the SAME cyclic rows; average the
                        # four totals for one canonical number.
                        val_record = validator.evaluate(
                            model,
                            device=device,
                            global_step=global_step,
                            max_training_steps=args.max_steps,
                            use_amp=use_amp,
                            amp_dtype=amp_dtype,
                            ignore_index=cfg.label_ignore_index,
                        )
                        if world_size > 1:
                            val_vec = torch.tensor(
                                [
                                    val_record["val_loss"],
                                    val_record["val_ce_loss"],
                                    val_record["val_router_aux_loss"],
                                    val_record["val_router_z_loss"],
                                ],
                                device=device,
                                dtype=torch.float32,
                            )
                            torch.distributed.all_reduce(
                                val_vec, op=torch.distributed.ReduceOp.SUM
                            )
                            averaged = (val_vec / world_size).tolist()
                            val_record["val_loss"] = averaged[0]
                            val_record["val_ce_loss"] = averaged[1]
                            val_record["val_router_aux_loss"] = averaged[2]
                            val_record["val_router_z_loss"] = averaged[3]
                        record["val_loss"] = val_record["val_loss"]
                        record["val_ce_loss"] = val_record["val_ce_loss"]
                        record["val_router_aux_loss"] = val_record[
                            "val_router_aux_loss"
                        ]
                        record["val_router_z_loss"] = val_record["val_router_z_loss"]
                        _write_jsonl(val_record)
                        if is_rank0:
                            logger.info(train_mod._format_val_log_line(val_record))
                        watchdog.progress(global_step, "metrics_and_collectives")

                    _write_jsonl(record)
                    if is_rank0:
                        logger.info(
                            format_training_log_line(
                                global_step, args.max_steps, record
                            )
                        )

                watchdog.progress(global_step, "complete", active=False)

                # Metrics for an update use its entry step (the first is 0).
                # The persisted cursor below is the next step to enter.
                global_step += 1
                # Checkpoints store the NEXT step to enter. Saving after the
                # increment keeps schedules and cadences identical to an
                # uninterrupted run instead of repeating a completed step.
                if global_step % args.save_interval == 0:
                    _save_checkpoint()

            if not shard_fully_consumed:
                break

            if is_rank0:
                done_path = os.path.join(
                    args.cache_dir, f"shard_{current_shard_idx:06d}.done"
                )
                with open(done_path, "w", encoding="utf-8") as f:
                    f.write("done\n")
                logger.info("Marked shard %d complete", current_shard_idx)
            current_shard_idx += 1
            current_batch_idx = 0

        if global_step > 0 and global_step % args.save_interval != 0:
            _save_checkpoint()

        scan_stats = get_mamba_scan_stats()
        logger.info("Training finished steps=%d mamba_scan=%s", global_step, scan_stats)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — saving checkpoint before exit")
        if global_step > 0:
            _save_checkpoint()
        raise

    except Exception:
        logger.exception("Training failed")
        raise

    finally:
        watchdog.close()
        stop_event.set()
        if producer_thread is not None:
            producer_thread.join(timeout=30.0)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FSDP2 multi-GPU training for Hybrid Mamba-MoE "
        "(DTensor Muon + AdamW; launch under torchrun).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="./runs/fsdp2_train",
        help="Logs and artifacts root.",
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
        "--dist-backend",
        choices=("auto", "nccl", "gloo"),
        default="auto",
        help="'auto' picks nccl on CUDA hosts, gloo otherwise.",
    )
    parser.add_argument(
        "--no-fused-mamba",
        action="store_true",
        help="Force the PyTorch scan tiers even when mamba-ssm is importable "
        "(diagnostic knob for suspected fused-kernel/FSDP2 issues).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision entirely (fp32 compute).",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bf16",),
        default="bf16",
        help="Module-level param cast dtype (fp16 unsupported under FSDP2: "
        "the GradScaler inf-check assumes plain gradients).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Accepted for CLI compatibility with train.py; REJECTED at "
        "runtime (torch.compile x FSDP2 wrap-order pitfalls).",
    )
    parser.add_argument("--compile-mode", type=str, default="default")
    checkpointing = parser.add_mutually_exclusive_group()
    checkpointing.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Checkpoint decoder layers with use_reentrant=False to reduce VRAM.",
    )
    checkpointing.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable decoder-layer gradient checkpointing (default).",
    )
    parser.set_defaults(gradient_checkpointing=False)

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
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Shared peak LR when optimizer-specific overrides are unset.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Decoupled Muon/AdamW weight decay.",
    )
    parser.add_argument(
        "--muon-lr",
        type=float,
        default=None,
        help="Optional Muon base-LR override (default: --lr).",
    )
    parser.add_argument(
        "--adam-lr", type=float, default=None, help="Optional AdamW LR override."
    )
    parser.add_argument(
        "--muon-momentum",
        type=float,
        default=0.95,
        help="Muon momentum coefficient.",
    )
    parser.add_argument(
        "--no-muon-nesterov",
        action="store_true",
        help="Disable Muon's Nesterov momentum blend.",
    )
    parser.add_argument(
        "--muon-ns-steps",
        type=int,
        default=5,
        help="Newton-Schulz orthogonalization iterations.",
    )
    parser.add_argument(
        "--muon-adjust-lr-fn",
        type=str,
        default="match_rms_adamw",
        choices=("match_rms_adamw", "original", "spectral_unclamped"),
        help="Per-matrix Muon learning-rate adjustment.",
    )
    parser.add_argument(
        "--muon-gather-buffer-mb",
        type=float,
        default=64.0,
        help="Approximate cap for simultaneously retained full bf16 Muon "
        "momentum matrices per rank; a single larger matrix may exceed it.",
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument(
        "--no-muon",
        action="store_true",
        help="Disable Muon and optimize every parameter with AdamW.",
    )
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="PER-RANK batch size (effective batch grows with world size).",
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument(
        "--grad-accum-steps",
        "--gradient-accumulation-steps",
        dest="grad_accum_steps",
        type=int,
        default=1,
        help="Micro-batches per optimizer step (FSDP2 no-sync accumulation: "
        "one reduce-scatter of the accumulated gradient per step).",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument(
        "--step-watchdog-seconds",
        type=float,
        default=300.0,
        help="Warn per rank when a forward/backward/collective/checkpoint phase "
        "makes no progress for this many seconds; 0 disables the watchdog.",
    )
    parser.add_argument(
        "--validate-token-interval",
        type=int,
        default=256,
        help="Check token-id ranges on each shard's first batch and every "
        "Nth step (1 = every batch; 0 = shard starts only).",
    )
    parser.add_argument(
        "--grad-nan-guard",
        choices=("sanitize", "strict", "monitor"),
        default="sanitize",
        help="Same trilemma as train.py: 'sanitize' zeroes NaN/Inf grads "
        "after clipping with no sync; 'strict' pays one D2H sync per step "
        "for true skip semantics (globally voted across ranks); 'monitor' "
        "protects nothing.",
    )
    parser.add_argument(
        "--smooth-window", type=int, default=50, help="Rolling CE mean window."
    )
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="./model_ckpt_fsdp2",
        help="Checkpoint directory (distinct default from train.py so the "
        "two trainers cannot clobber each other's files).",
    )
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
        help="Salesforce/wikitext config name.",
    )
    parser.add_argument(
        "--ns-self-check",
        action="store_true",
        help="Run the Newton-Schulz parity checks on CPU and exit (no "
        "process group, no dataset imports).",
    )
    args = parser.parse_args()
    if args.step_watchdog_seconds < 0:
        parser.error("--step-watchdog-seconds must be >= 0")
    for option, value in (
        ("--lr", args.lr),
        ("--muon-lr", args.muon_lr),
        ("--adam-lr", args.adam_lr),
        ("--weight-decay", args.weight_decay),
    ):
        if value is not None and (not math.isfinite(value) or value < 0.0):
            parser.error(f"{option} must be finite and >= 0")
    if not 0.0 <= args.muon_momentum < 1.0:
        parser.error("--muon-momentum must be in [0, 1)")
    if not 0 <= args.muon_ns_steps < 100:
        parser.error("--muon-ns-steps must be in [0, 100)")
    if (
        not math.isfinite(args.muon_gather_buffer_mb)
        or args.muon_gather_buffer_mb <= 0.0
    ):
        parser.error("--muon-gather-buffer-mb must be finite and > 0")
    if not args.log_jsonl:
        args.log_jsonl = str(Path(args.run_dir) / "metrics.jsonl")
    return args


def main() -> None:
    args = parse_args()
    if args.ns_self_check:
        from utils.fsdp2_muon import run_ns_self_check

        print("pre-training/fsdp2_train.py Newton-Schulz self-check (CPU)")
        raise SystemExit(0 if run_ns_self_check() else 1)
    if args.compile:
        raise SystemExit(
            "--compile is rejected under FSDP2 v1 (torch.compile x "
            "fully_shard wrap-order pitfalls)."
        )
    if args.amp_dtype != "bf16":
        raise SystemExit("Only --amp-dtype bf16 is supported under FSDP2.")
    if args.grad_accum_steps < 1:
        raise SystemExit("--grad-accum-steps must be >= 1.")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    _require_heavy_imports()  # fail fast on missing deps before GPU work
    logger = _setup_logging(Path(args.run_dir))
    logger.info("Starting FSDP2 training with args: %s", vars(args))
    train(args, logger)


if __name__ == "__main__":
    main()
