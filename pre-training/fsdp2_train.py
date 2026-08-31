"""FSDP2 + Muon distributed training for Hybrid Mamba-MoE.

Multi-GPU port of the root ``train.py`` trainer: the SAME Moonshot Muon +
AdamW hybrid (identical grouping rules, LR resolution, warmup/cosine math,
nan-guard trilemma, zero-sync logging discipline) executed over ``fully_shard``
sharded parameters. Only the execution model changes.

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
  * Checkpoints consolidate through the public DTensor state-dict APIs
    (``get_/set_model_state_dict`` + hand consolidation of optimizer state)
    into the ROOT ``train.py`` payload schema, so checkpoints stay
    interchangeable between the two trainers. RNG state is captured
    per-rank under a different key layout than root's single-process
    payload; data order still replays exactly via the seeded sampler.
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
import json
import logging
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Light imports only above this line: model/** and utils.fsdp2_muon pull no
# datasets/transformers, so --ns-self-check stays cheap and dependency-light
# (same rationale as model/core/optim.py).
from model.core.builders import count_trainable_params
from model.core.constants import MEMORY_NAN_FIX_ID
from model.core.optim import _is_adamw_no_decay, split_muon_adam_params
from model.hybrid.mamba import (
    fused_mamba_scan_available,
    get_mamba_scan_stats,
    log_mamba_backend,
    reset_mamba_scan_stats,
)
from utils.fsdp2_muon import MuonDTensor, run_ns_self_check


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
    console = logging.StreamHandler(sys.stdout)
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
    """Public FSDP2 / DTensor / DCP state-dict APIs (torch >= 2.6 recommended)."""
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


# ---------------------------------------------------------------------------
# Process-group bootstrap
# ---------------------------------------------------------------------------


def init_distributed(dist_backend: str | None) -> tuple[int, int, torch.device]:
    """Initialize the default process group; returns (rank, world_size, device).

    Works under ``torchrun`` (env vars set) AND as a bare single process
    (world of 1 over gloo) so laptop smoke tests need no launcher.
    """
    if not torch.distributed.is_initialized():
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
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
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
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
    layout of a plain ``optimizer.state_dict()``, so payloads remain loadable
    by the root single-GPU trainer. Parameter indices derive from
    construction order; split_muon_adam_params + the group assembly here are
    deterministic given the model, keeping indices stable across trainers.
    """
    sd = opt.state_dict()
    consolidated: dict[int, dict[str, Any]] = {}
    for idx, entries in sd["state"].items():
        consolidated[idx] = {
            key: (
                value.full_tensor().cpu() if isinstance(value, dtensor_cls) else value
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
                restored[key] = distribute_tensor(
                    value, p.device_mesh, list(p.placements)
                )
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
    checkpoint_dir: Path,
    logger: logging.Logger,
    validator: Any | None = None,
    use_muon: bool = True,
    extra_payload: dict[str, Any] | None = None,
) -> None:
    """Consolidated checkpoint in the ROOT ``train.py`` payload schema.

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

    # In Muon mode optimizers=[muon, adam]; AdamW-only mode stores its single
    # optimizer ONLY under the adam_* keys (same convention as train.py).
    payload: dict[str, Any] = {
        "model_state_dict": model_sd,
        "config": config,
        "global_step": global_step,
        "current_shard_idx": current_shard_idx,
        "rng_state": _gather_rng_payload(rank, torch.distributed.get_world_size()),
        "memory_nan_fix_id": MEMORY_NAN_FIX_ID,
        "use_muon": use_muon,
        "training_runtime": train_mod.checkpoint_runtime_contract(
            model, distributed_strategy="fsdp2"
        ),
    }
    if validator is not None:
        # The cyclic cursor advances identically on every rank (replicated
        # params scoring the same fixed rows), so rank0's view is canonical.
        payload["validator_state_dict"] = validator.state_dict
    if use_muon:
        payload["muon_optimizer_state_dict"] = _consolidate_optimizer_state(
            optimizers[0], api["DTensor"]
        )
        payload["muon_scheduler_state_dict"] = schedulers[0].state_dict()
        payload["adam_optimizer_state_dict"] = _consolidate_optimizer_state(
            optimizers[-1], api["DTensor"]
        )
        payload["adam_scheduler_state_dict"] = schedulers[-1].state_dict()
    else:
        payload["adam_optimizer_state_dict"] = _consolidate_optimizer_state(
            optimizers[0], api["DTensor"]
        )
        payload["adam_scheduler_state_dict"] = schedulers[0].state_dict()
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
            "Checkpoint saved step=%d shard=%d path=%s",
            global_step,
            current_shard_idx,
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
    validator: Any | None = None,
    use_muon: bool | None = None,
    dl_generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Resume from a consolidated checkpoint (fsdp2- OR root-written).

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
    )

    api["set_model_state_dict"](
        model,
        checkpoint["model_state_dict"],
        options=api["StateDictOptions"](full_state_dict=True),
    )

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

    required_state = ["adam_optimizer_state_dict", "adam_scheduler_state_dict"]
    if ckpt_use_muon:
        required_state.extend(
            ["muon_optimizer_state_dict", "muon_scheduler_state_dict"]
        )
    train_mod._require_resume_keys(checkpoint, required_state)

    if ckpt_use_muon and "muon_optimizer_state_dict" in checkpoint:
        optimizers[0].load_state_dict(
            _reshard_optimizer_state(
                checkpoint["muon_optimizer_state_dict"],
                optimizers[0],
                device,
                api["distribute_tensor"],
            )
        )
        optimizers[-1].load_state_dict(
            _reshard_optimizer_state(
                checkpoint["adam_optimizer_state_dict"],
                optimizers[-1],
                device,
                api["distribute_tensor"],
            )
        )
        schedulers[0].load_state_dict(checkpoint["muon_scheduler_state_dict"])
        schedulers[-1].load_state_dict(checkpoint["adam_scheduler_state_dict"])
    elif "adam_optimizer_state_dict" in checkpoint:
        optimizers[0].load_state_dict(
            _reshard_optimizer_state(
                checkpoint["adam_optimizer_state_dict"],
                optimizers[0],
                device,
                api["distribute_tensor"],
            )
        )
        schedulers[0].load_state_dict(checkpoint["adam_scheduler_state_dict"])

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
# Optimizer construction (semantics identical to train.build_optimizers)
# ---------------------------------------------------------------------------


def build_fsdp2_optimizers(
    model,
    *,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[list[torch.optim.Optimizer], bool, dict[str, Any]]:
    """MuonDTensor + AdamW over FSDP2-sharded DTensor parameters.

    Grouping rules, LR resolution, decay subgrouping, hyperparameters, and
    log wording mirror ``train.build_optimizers``; only ``torch.optim.Muon``
    is swapped for the DTensor-native ``MuonDTensor``, and AdamW runs
    unfused (fused kernels' DTensor support is not guaranteed).
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
        "muon_adjust_lr_fn": args.muon_adjust_lr_fn if enable_muon else None,
        "inventory": inventory,
    }

    logger.info(
        "optimizer split: adamw=%.2f%% (%d tensors, %s params) "
        "muon=%.2f%% (%d tensors, %s params) total=%.3fB",
        100.0 * adam_count / max(total_params, 1),
        len(inventory["adamw"]),
        f"{adam_count:,}",
        100.0 * (muon_count if enable_muon else 0) / max(total_params, 1),
        len(inventory["muon"]) if enable_muon else 0,
        f"{muon_count if enable_muon else 0:,}",
        total_params / 1e9,
    )
    logger.debug("AdamW params: %s", inventory["adamw"])
    logger.debug("Muon params: %s", inventory["muon"] if enable_muon else [])

    def _adamw(params_list: list[torch.nn.Parameter]) -> torch.optim.AdamW:
        adam_decay = [p for p in params_list if not _is_adamw_no_decay(p)]
        adam_no_decay = [p for p in params_list if _is_adamw_no_decay(p)]
        logger.info(
            "AdamW(lr=%.3e, betas=(%.2f, %.2f), wd=%.3g on %d params / "
            "wd=0 on %d params, fused=False)",
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
        )

    if enable_muon:
        muon_opt = MuonDTensor(
            muon_params,
            lr=resolved_muon_lr,
            momentum=args.muon_momentum,
            nesterov=not args.no_muon_nesterov,
            weight_decay=args.weight_decay,
            ns_steps=args.muon_ns_steps,
            adjust_lr_fn=(
                args.muon_adjust_lr_fn
                if args.muon_adjust_lr_fn in ("match_rms_adamw", "original")
                else None
            ),
        )
        logger.info(
            "Muon(lr=%.3e, wd=%.3g, momentum=%.3g, nesterov=%s, ns_steps=%d, "
            "adjust_lr_fn=%s)",
            resolved_muon_lr,
            args.weight_decay,
            args.muon_momentum,
            not args.no_muon_nesterov,
            args.muon_ns_steps,
            args.muon_adjust_lr_fn,
        )
        return [muon_opt, _adamw(adam_params)], True, meta

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
    model.model.calibrate_ssm_norm_thresholds()
    reset_mamba_scan_stats()
    n_params = count_trainable_params(model)
    if is_rank0:
        # numel() on Shard(0) DTensors reports GLOBAL counts.
        logger.info("trainable_params=%s (%.3fB)", f"{n_params:,}", n_params / 1e9)

    api = _require_fsdp2()
    # Params stay fp32; the autocast below provides bf16 compute, mirroring
    # train.py. MixedPrecisionPolicy(param_dtype=bfloat16) would swap weights
    # to bf16 inside forward and crash layer.py's aux-loss block, which runs
    # under autocast(enabled=False) on fp32-promoted activations.
    mp_policy = api["MixedPrecisionPolicy"](reduce_dtype=torch.float32)
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16  # main() rejects anything else
    fully_shard = api["fully_shard"]
    # Inner-to-outer: layers first (independent reshard-after-forward /
    # prefetch), then the root gathers everything left outside the layers.
    for layer in model.model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)
    logger.info(
        "fully_shard applied: layers=%d world=%d mp=%s",
        len(model.model.layers),
        world_size,
        f"{'autocast-bf16' if use_amp else 'fp32'}/fp32-params/fp32-reduce",
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
            "LR schedule: peak=%g warmup=%d%s cosine_floor=%.2f",
            args.lr,
            warmup_steps,
            " (auto)" if args.warmup_steps <= 0 else "",
            args.min_lr_ratio,
        )

    global_step = 0
    current_shard_idx = 0
    ckpt_dir = Path(args.ckpt_dir)

    # Seeds DataLoader workers; intra-shard ORDER comes from the seeded
    # DistributedSampler (epoch = shard index), which needs no persistence.
    dl_generator = torch.Generator()
    dl_generator.manual_seed(args.seed)

    if args.resume:
        global_step, current_shard_idx = load_checkpoint_fsdp2(
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
            checkpoint_dir=ckpt_dir,
            logger=logger,
            validator=validator,
            use_muon=use_muon,
            extra_payload={
                "dl_generator_state": dl_generator.get_state(),
                "muon_adjust_lr_fn": opt_meta.get("muon_adjust_lr_fn"),
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
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=sampler,
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
                    "Shard %d | sequences=%d | per-rank batches=%d | step %d/%d",
                    current_shard_idx,
                    len(dataset),
                    len(dataloader),
                    global_step,
                    args.max_steps,
                )

            shard_fully_consumed = True
            batches_iter = iter(dataloader)
            batches_seen = 0
            while True:
                if global_step >= args.max_steps:
                    # Mid-shard stop: leave the shard unmarked and the cursor
                    # where it is so an extended --max-steps resumes from this
                    # exact position instead of dropping the unconsumed tail.
                    shard_fully_consumed = False
                    break

                # ---- collect `accum` micro-batches for one optimizer step --
                micro_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
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
                    if batches_seen == 0 or (
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
                loss_for_metrics: torch.Tensor | None = None
                for micro_idx, (m_ids, m_labels) in enumerate(micro_inputs):
                    if accum > 1 and micro_idx == len(micro_inputs) - 1:
                        model.set_requires_gradient_sync(True, recurse=True)
                    watchdog.progress(
                        global_step, f"forward_micro_{micro_idx + 1}/{len(micro_inputs)}"
                    )
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=use_amp,
                    ):
                        outputs = model(
                            input_ids=m_ids,
                            labels=m_labels,
                            training_step=global_step,
                            max_training_steps=args.max_steps,
                        )
                    assert outputs.loss is not None
                    # Mean-of-means: equal-size micro-batches (drop_last), so
                    # dividing by count averages correctly.
                    watchdog.progress(
                        global_step,
                        f"backward_micro_{micro_idx + 1}/{len(micro_inputs)}",
                    )
                    (outputs.loss / len(micro_inputs)).backward()
                    if micro_idx == len(micro_inputs) - 1:
                        loss_for_metrics = outputs.loss.detach()

                assert loss_for_metrics is not None
                outputs.loss = loss_for_metrics

                watchdog.progress(global_step, "optimizer_and_collectives")
                # Params are fp32 masters (see the mp_policy note above), so
                # gradients arrive fp32 and clip sees TRUE magnitudes — no
                # scaler unscale step exists on this bf16-autocast path.
                grad_norm_dt = clip_grad_norm_(model.parameters(), args.max_grad_norm)
                grad_norm = (
                    grad_norm_dt.full_tensor()
                    if hasattr(grad_norm_dt, "full_tensor")
                    else grad_norm_dt
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
                watchdog.progress(global_step, "metrics_and_collectives")
                # When strict skipped, the NaN grads die via zero_grad(
                # set_to_none=True) at the top of the next iteration.

                aux = outputs.auxiliary_losses
                assert aux is not None

                # ---- metrics: accumulate on-device, transfer on flush ------
                weighted_t = train_mod._weighted_term_tensors(
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
                ):
                    scalars[name] = getattr(aux, name)
                for name, value in weighted_t.items():
                    if isinstance(value, torch.Tensor):
                        scalars[name] = value

                gate_stats = outputs.gate_stats or {}
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
                    step_time_s = (
                        time.perf_counter() - step_window_started
                    ) / window
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
                            train_mod._format_log_line(
                                global_step, args.max_steps, record
                            )
                        )

                if global_step % args.save_interval == 0 and global_step > 0:
                    _save_checkpoint()
                watchdog.progress(global_step, "complete", active=False)

                # Convention: global_step counts iterations ENTERED, so the
                # first completed update logs as step=0 — identical to
                # train.py (assoc/expert warmups, cadences, checkpoints and
                # resume all share it).
                global_step += 1

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
        "(Muon + AdamW hybrid; launch under torchrun).",
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
    # Moonshot Muon (arXiv:2502.16982): with match_rms_adamw, share eta and
    # weight decay across Muon + AdamW.
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Shared peak LR for Muon and AdamW when --muon-lr/--adam-lr are "
        "unset (Moonlight-style after update-RMS matching).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Decoupled weight decay for Muon and AdamW (Moonlight uses 0.1).",
    )
    parser.add_argument(
        "--muon-lr", type=float, default=None, help="Optional Muon LR override."
    )
    parser.add_argument(
        "--adam-lr", type=float, default=None, help="Optional AdamW LR override."
    )
    parser.add_argument(
        "--muon-momentum",
        type=float,
        default=0.95,
        help="Muon momentum mu (paper / Keller default: 0.95).",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="PER-RANK batch size (effective batch grows with world size).",
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument(
        "--grad-accum-steps",
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
    if not args.log_jsonl:
        args.log_jsonl = str(Path(args.run_dir) / "metrics.jsonl")
    return args


def main() -> None:
    args = parse_args()
    if args.ns_self_check:
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
