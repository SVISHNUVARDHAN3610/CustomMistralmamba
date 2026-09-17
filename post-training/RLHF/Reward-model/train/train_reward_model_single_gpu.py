"""Reward-stage training on a single GPU (or CPU for testing/smoke).

Reference: post-training/RLHF/train_reward_model.py
This implementation runs on a single GPU without torchrun or FSDP2 process groups,
while preserving:
- Pure-Mamba Reward Model architecture (352M parameters in production, tiny in --smoke)
- Bradley-Terry logistic pairwise loss
- Gradient accumulation weighted by pair count
- AdamW optimizer with no-decay for 1D parameters, norm gains, A_log, and D
- Warmup + cosine learning rate scheduler
- Bounded PreferenceFeed streaming and PreferenceShardProducer lifecycle
- Fixed diagnostic evaluation and full evaluation (UF, HH-RLHF, HelpSteer2)
- DCP checkpoint format (family: pure_mamba_reward_dcp_v1) for seamless resume and
  compatibility with evaluate_reward_model.py and train_rlhf.py
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import random
import sys
import threading
import time
import uuid
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
REWARD_MODEL_DIR = ROOT / "post-training" / "RLHF" / "Reward-model"
for directory in (ROOT, ROOT / "post-training", REWARD_MODEL_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
os.environ["USE_JAX"] = "0"

import sft_post_train as sft
import torch
from config import RewardConfig, smoke_config
from evaluation_metrics import RewardStatistics, isolated_evaluation_rng
from reward_model import RewardModel, pairwise_loss, parameter_counts
from rlhf_dataset import (
    PreferenceCollator,
    PreferenceShardDataset,
    PreferenceShardProducer,
    SmokeTokenizer,
    accounting_summary,
    smoke_stream,
)
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers import AutoTokenizer

import train as pretrain
from model.core.optim import _is_adamw_no_decay
from model.hybrid.mamba import MambaBlock, fused_mamba_scan_available

FAMILY = "pure_mamba_reward_dcp_v1"


class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_logging(run_dir: Path, *, level: int = logging.INFO) -> logging.Logger:
    """Set up run logger with stdout and train.log handlers."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("reward_single_gpu")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = FlushStreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def resolve_precision(precision: str, *, bf16_supported: bool) -> str:
    if precision not in ("auto", "bf16", "fp32"):
        raise ValueError("RLHF supports BF16/FP32; FP16 is unsupported")
    if precision == "auto":
        return "bf16" if bf16_supported else "fp32"
    if precision == "bf16" and not bf16_supported:
        raise RuntimeError("BF16 requested on unsupported hardware; choose fp32")
    return precision


def require_mamba_kernels(
    model: torch.nn.Module, required: bool, device: torch.device
) -> None:
    """Fail before training and propagate kernel errors instead of falling back."""
    if required and (device.type != "cuda" or not fused_mamba_scan_available()):
        raise RuntimeError(
            "require_fused_mamba needs CUDA and working mamba-ssm kernels"
        )
    for module in model.modules():
        if isinstance(module, MambaBlock):
            if required and (not module.use_fused_scan or module.use_parallel_scan):
                raise ValueError("Production Mamba config disables the fused scan")
            module.require_fused_scan = required


def _local_rng_state() -> dict[str, Any]:
    """Single process RNG state snapshot."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def _gather_rng_payload(rank: int, world_size: int) -> dict[str, Any]:
    """Gather RNG payload matching repository checkpoint schema."""
    return {
        "format": "fsdp2_per_rank_v1",
        "world_size": world_size,
        "ranks": [_local_rng_state()],
    }


def _restore_rng_payload(state: Any, rank: int, world: int) -> bool:
    """Restore RNG payload from checkpoint."""
    if not isinstance(state, dict) or state.get("format") != "fsdp2_per_rank_v1":
        return False
    if state.get("world_size") != world:
        return False
    ranks = state.get("ranks")
    if not isinstance(ranks, list) or len(ranks) <= rank:
        return False
    local = ranks[rank]
    if not isinstance(local, dict):
        return False
    try:
        random.setstate(local["python"])
        torch.set_rng_state(local["torch"])
        if (
            "cuda" in local
            and torch.cuda.is_available()
            and torch.cuda.is_initialized()
        ):
            torch.cuda.set_rng_state(local["cuda"])
    except (KeyError, RuntimeError, TypeError, ValueError):
        return False
    return True


class SingleGPURewardBackend(sft.SingleGPUBackend):
    """Reward-specific precision, device, optimizer and gradient helpers on single GPU."""

    def __init__(
        self,
        cfg: RewardConfig,
        logger: logging.Logger,
        smoke: bool = False,
        device: torch.device | None = None,
    ):
        self.logger = logger
        self.smoke = smoke
        self.rank, self.world = 0, 1

        if device is not None:
            self.device = device
        elif smoke:
            self.device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Single-GPU reward training requires CUDA; specify --device cpu or use --smoke for local CPU tests"
                )
            self.device = torch.device("cuda:0")

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        bf16_supported = (
            not smoke and self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        )
        precision = resolve_precision(
            cfg.system.precision, bf16_supported=bf16_supported
        )
        self.dtype = {"fp32": None, "bf16": torch.bfloat16}[precision]
        self.precision = precision
        self.loss_scale = 1.0
        self.scale_good_steps = 0

    def wrap(self, model: torch.nn.Module, cfg: RewardConfig) -> torch.nn.Module:
        """Place model on device, configure activation checkpointing and Mamba optimizer flags."""
        model = model.to(self.device)
        model.activation_checkpointing = cfg.system.activation_checkpointing
        for name, parameter in model.named_parameters():
            if name.endswith((".A_log", ".D")):
                parameter._no_weight_decay = True
        return model

    def sum(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def broadcast(self, value: Any) -> Any:
        return value

    def barrier(self) -> None:
        pass

    def sync_backward(self, model: torch.nn.Module, enabled: bool) -> None:
        pass

    def autocast(self):
        return (
            nullcontext()
            if self.dtype is None
            else torch.autocast(self.device.type, dtype=self.dtype, cache_enabled=False)
        )

    def optimizer_step(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, max_norm: float
    ) -> tuple[float, bool]:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite RLHF gradient norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.scale_good_steps += 1
        return float(norm), True


class PreferenceFeed(sft.ShardFeed):
    """Reuse SFT's thread lifecycle, producer wait and bounded queue management."""

    def __init__(
        self,
        cfg: RewardConfig,
        tokenizer: Any,
        backend: SingleGPURewardBackend,
        purpose: str,
        cursor: dict | None = None,
        smoke: bool = False,
        producer_class: type = PreferenceShardProducer,
    ):
        self.backend = backend
        token = uuid.uuid4().hex
        cache = Path(cfg.system.output_dir) / "data_cache" / token
        self.args = SimpleNamespace(
            cache_dir=str(cache),
            max_buffered_files=cfg.data.buffer_size,
            shard_timeout=cfg.data.shard_timeout,
        )
        self.producer = None
        self.thread = None
        self.stop = threading.Event()
        self.producer = producer_class(
            str(cache),
            tokenizer,
            cfg.data,
            cfg.model.vocab_size,
            cfg.seed + (cursor or {}).get("epoch", 0),
            purpose,
            backend.logger.info,
            cursor,
            smoke_stream(purpose) if smoke else None,
        )
        self.thread = threading.Thread(
            target=self.producer.start_streaming, args=(self.stop,), daemon=True
        )
        self.thread.start()

    def wait(self, index: int) -> str | None:
        """Wait for shard readiness with timeout and error checking."""
        path = Path(self.args.cache_dir) / f"shard_{index:06d}.bin"
        deadline = time.monotonic() + self.args.shard_timeout
        while True:
            if self.producer.error is not None:
                raise RuntimeError(str(self.producer.error))
            if path.exists():
                return str(path)
            if self.producer.finished:
                later = any(p.name > path.name for p in path.parent.glob("shard_*.bin"))
                if later:
                    raise RuntimeError(f"Missing reward shard {path}")
                return None
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for reward shard {path}")
            time.sleep(0.1)

    def consume(self, path: str) -> None:
        Path(path).with_suffix(".done").touch()

    def close(self) -> None:
        super().close()
        if self.producer is not None and not self.thread.is_alive():
            for path in Path(self.args.cache_dir).glob("shard_*.bin"):
                path.with_suffix(".done").touch()
            self.producer._cleanup_consumed_shards()
            self.backend.logger.info(
                "[Reward Producer final] %s filters=%s",
                dict(self.producer.stats),
                dict(self.producer.reasons),
            )


def loader_for_shard(
    path: str,
    cfg: RewardConfig,
    tokenizer: Any,
    backend: SingleGPURewardBackend,
    shard: int,
    offset: int = 0,
    training: bool = True,
) -> tuple[PreferenceShardDataset, DataLoader]:
    dataset = PreferenceShardDataset(path)
    if training:
        sampler = DistributedSampler(
            dataset,
            num_replicas=backend.world,
            rank=backend.rank,
            shuffle=True,
            seed=cfg.seed,
            drop_last=True,
        )
        sampler.set_epoch(shard)
        indices = list(sampler)[offset * cfg.training.batch_size :]
    else:
        indices = range(len(dataset))
    kwargs: dict[str, Any] = {
        "batch_size": cfg.training.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": backend.device.type == "cuda",
        "drop_last": False,
        "collate_fn": PreferenceCollator(tokenizer.pad_token_id),
        "generator": torch.Generator().manual_seed(cfg.seed + shard),
    }
    if cfg.data.num_workers:
        kwargs["prefetch_factor"] = cfg.data.prefetch_factor
    return dataset, DataLoader(Subset(dataset, indices), **kwargs)


def score_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    backend: SingleGPURewardBackend,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single forward pass for chosen and rejected branches."""
    ids = torch.cat([batch["chosen_input_ids"], batch["rejected_input_ids"]]).to(
        backend.device, non_blocking=True
    )
    mask = torch.cat(
        [batch["chosen_attention_mask"], batch["rejected_attention_mask"]]
    ).to(backend.device, non_blocking=True)
    with backend.autocast():
        rewards = model(input_ids=ids, attention_mask=mask)
    return rewards.float().chunk(2, dim=0)


def metric_sums(chosen: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    margin = chosen.detach() - rejected.detach()
    return torch.stack(
        (
            -torch.nn.functional.logsigmoid(margin).sum(),
            chosen.detach().sum(),
            rejected.detach().sum(),
            margin.sum(),
            (margin > 0).float().sum(),
            torch.tensor(margin.numel(), device=margin.device),
        )
    ).double()


def metric_means(totals: torch.Tensor) -> dict[str, float]:
    values = totals.cpu().tolist()
    if values[-1] == 0:
        raise ValueError("No valid preference pairs were scored")
    names = (
        "loss",
        "chosen_reward",
        "rejected_reward",
        "reward_margin",
        "pairwise_accuracy",
    )
    return {name: value / values[-1] for name, value in zip(names, values)}


def evaluation_batches(
    cfg: RewardConfig,
    tokenizer: Any,
    backend: SingleGPURewardBackend,
    purpose: str,
    smoke: bool = False,
    max_batches: int | None = None,
):
    """Bounded evaluation feed independent of training feed cursor."""
    feed = PreferenceFeed(cfg, tokenizer, backend, purpose, smoke=smoke)
    batches, shard = 0, 0
    try:
        while max_batches is None or batches < max_batches:
            path = feed.wait(shard)
            if path is None:
                break
            dataset, loader = loader_for_shard(
                path, cfg, tokenizer, backend, shard, training=False
            )
            try:
                for batch in loader:
                    yield batch
                    batches += 1
                    if max_batches is not None and batches >= max_batches:
                        break
            finally:
                dataset.close()
            feed.consume(path)
            shard += 1
    finally:
        feed.close()


def build_diagnostic_set(
    cfg: RewardConfig,
    tokenizer: Any,
    backend: SingleGPURewardBackend,
    smoke: bool = False,
) -> dict[str, list[dict]]:
    """Cache fixed source prefixes: ceil(N/2) UF, floor(N/2) HH pairs."""
    result = {}
    with isolated_evaluation_rng():
        for purpose, count in zip(
            ("ultrafeedback_test", "hh_rlhf_test"),
            (
                (cfg.system.diagnostic_eval_pairs + 1) // 2,
                cfg.system.diagnostic_eval_pairs // 2,
            ),
        ):
            rows = []
            iterator = evaluation_batches(cfg, tokenizer, backend, purpose, smoke)
            try:
                for batch in iterator:
                    for i in range(batch["chosen_input_ids"].size(0)):
                        row = {
                            key: batch[key][i]
                            for key in ("category", "source", "original_split")
                        }
                        for side in ("chosen", "rejected"):
                            row[side] = batch[f"{side}_input_ids"][i][
                                batch[f"{side}_attention_mask"][i]
                            ].tolist()
                        rows.append(row)
                        if len(rows) == count:
                            break
                    if len(rows) == count:
                        break
            finally:
                if hasattr(iterator, "close"):
                    iterator.close()
            if len(rows) != count:
                raise ValueError(
                    f"{purpose}: requested {count} diagnostic pairs, only {len(rows)} survived filtering"
                )
            random.Random(cfg.seed).shuffle(rows)
            result[purpose] = rows
    return result


@torch.no_grad()
def evaluate_batches(
    model: torch.nn.Module,
    batches: Any,
    backend: SingleGPURewardBackend,
    *,
    percentile_strategy: str = "exact",
    percentile_sample_size: int = 100000,
) -> dict[str, Any]:
    """Compute RewardStatistics moments, percentiles and scale diagnostics."""
    was_training = model.training
    groups: dict[str, RewardStatistics] = {}
    with isolated_evaluation_rng():
        model.eval()
        try:
            for batch in batches:
                chosen, rejected = score_batch(model, batch, backend)
                chosen, rejected = chosen.detach().cpu(), rejected.detach().cpu()
                if "overall" not in groups:
                    groups["overall"] = RewardStatistics(
                        percentile_strategy, percentile_sample_size
                    )
                groups["overall"].update(chosen, rejected)
                for field, prefix in (
                    ("category", "category/"),
                    ("original_split", "original_"),
                ):
                    for value in sorted(set(batch[field])):
                        if value in ("overall", "unknown"):
                            continue
                        indices = [i for i, v in enumerate(batch[field]) if v == value]
                        key = prefix + value
                        if key not in groups:
                            groups[key] = RewardStatistics(
                                percentile_strategy, percentile_sample_size
                            )
                        groups[key].update(chosen[indices], rejected[indices])
            if "overall" not in groups:
                raise ValueError("No valid preference pairs were scored")
            result = groups["overall"].result()
            result["categories"] = {
                k.removeprefix("category/"): v.result()
                for k, v in sorted(groups.items())
                if k.startswith("category/")
            }
            result["subsets"] = {
                k: v.result()
                for k, v in sorted(groups.items())
                if k.startswith("original_")
            }
            return result
        finally:
            if hasattr(batches, "close"):
                batches.close()
            for group in groups.values():
                group.close()
            model.train(was_training)


def evaluate_diagnostic(
    model: torch.nn.Module,
    diagnostic: dict[str, list[dict]],
    cfg: RewardConfig,
    tokenizer: Any,
    backend: SingleGPURewardBackend,
) -> dict[str, Any]:
    result = {}
    for purpose, rows in diagnostic.items():
        collate = PreferenceCollator(tokenizer.pad_token_id)
        batches = (
            collate(rows[i : i + cfg.training.batch_size])
            for i in range(0, len(rows), cfg.training.batch_size)
        )
        result[purpose] = evaluate_batches(model, batches, backend)
        result[purpose]["subset_sha256"] = hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()
        ).hexdigest()
        result[purpose]["evaluation_scope"] = "fixed_diagnostic"
    return result


def evaluate(
    model: torch.nn.Module,
    cfg: RewardConfig,
    tokenizer: Any,
    backend: SingleGPURewardBackend,
    purpose: str = "validation",
    smoke: bool = False,
    max_batches: int | None = None,
) -> dict[str, Any]:
    if purpose == "validation":
        return {
            name: evaluate(model, cfg, tokenizer, backend, name, smoke, max_batches)
            for name in ("ultrafeedback_test", "hh_rlhf_test")
        }
    result = evaluate_batches(
        model,
        evaluation_batches(cfg, tokenizer, backend, purpose, smoke, max_batches),
        backend,
        percentile_strategy=cfg.system.percentile_strategy,
        percentile_sample_size=cfg.system.percentile_sample_size,
    )
    result["evaluation_scope"] = (
        "synthetic_fixture"
        if smoke
        else ("bounded_sample" if max_batches is not None else "full_split")
    )
    return result


def evaluate_full(
    model: torch.nn.Module,
    cfg: RewardConfig,
    tokenizer: Any,
    backend: SingleGPURewardBackend,
    smoke: bool = False,
    max_batches: int | None = None,
) -> dict[str, Any]:
    return {
        name: evaluate(model, cfg, tokenizer, backend, name, smoke, max_batches)
        for name in ("ultrafeedback_test", "hh_rlhf_test", "helpsteer2")
    }


def write_evaluation(
    result: dict[str, Any],
    cfg: RewardConfig,
    backend: SingleGPURewardBackend,
    kind: str,
    step: int | str,
) -> None:
    headline = {
        name: {
            "pairs": values["pairs"],
            "pairwise_accuracy": values["pairwise_accuracy"],
            "subsets": {
                key: {
                    "pairs": group["pairs"],
                    "pairwise_accuracy": group["pairwise_accuracy"],
                }
                for key, group in values["subsets"].items()
            },
        }
        for name, values in result.items()
    }
    backend.logger.info("%s Evaluation step=%s %s", kind, step, json.dumps(headline))
    backend.logger.info(
        "Reward Scale Diagnostics %s",
        json.dumps({k: v["reward_scale_diagnostics"] for k, v in result.items()}),
    )
    path = Path(cfg.system.output_dir)
    (path / f"{kind.lower()}_evaluation_{step}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (path / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"step": step, "evaluation": kind.lower(), "datasets": result})
            + "\n"
        )


def checkpoint_metadata(path: str | Path, family: str = FAMILY) -> dict[str, Any]:
    path = Path(path)
    if not (path / "complete").is_file():
        raise ValueError(f"Incomplete reward checkpoint: {path}")
    metadata = torch.load(path / "trainer.pt", map_location="cpu", weights_only=True)
    if metadata.get("family") != family:
        raise ValueError(f"Incompatible checkpoint: expected {family}")
    return metadata


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: RewardConfig,
    backend: SingleGPURewardBackend,
    cursor: dict[str, Any],
    step: int,
    *,
    family: str = FAMILY,
) -> None:
    """Save model and optimizer state via DCP; write trainer metadata and completion marker."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite reward checkpoint {path}")

    model_state, optimizer_state = get_state_dict(
        model,
        optimizer,
        options=StateDictOptions(full_state_dict=False, cpu_offload=True),
    )
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=path / "state",
    )
    rng = _gather_rng_payload(backend.rank, backend.world)
    payload = {
        "family": family,
        "config": cfg.to_dict(),
        "scheduler": scheduler.state_dict(),
        "cursor": cursor,
        "step": step,
        "world_size": backend.world,
        "rng": rng,
        "precision": backend.precision,
        "loss_scale": backend.loss_scale,
        "scale_good_steps": backend.scale_good_steps,
        "torch_version": str(torch.__version__),
    }
    torch.save(payload, path / "trainer.pt.tmp")
    os.replace(path / "trainer.pt.tmp", path / "trainer.pt")
    cfg.save_pretrained(str(path / "config.json"))
    (path / "complete").touch()


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    cfg: RewardConfig | None = None,
    backend: SingleGPURewardBackend | None = None,
    *,
    family: str = FAMILY,
) -> dict[str, Any]:
    """Load model and optimizer state from DCP checkpoint."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_state_dict,
        set_model_state_dict,
        set_state_dict,
    )

    metadata = checkpoint_metadata(path, family)
    if cfg is not None and metadata["config"]["model"] != cfg.to_dict()["model"]:
        raise ValueError("Checkpoint reward architecture differs from configuration")
    options = StateDictOptions(full_state_dict=False)
    if optimizer is None:
        state = {"model": get_model_state_dict(model, options=options)}
        dcp.load(state, checkpoint_id=Path(path) / "state")
        set_model_state_dict(model, state["model"], options=options)
    else:
        for key in ("seed", "model", "data", "training", "rlhf"):
            if (
                RewardConfig.from_dict(metadata["config"]).to_dict()[key]
                != cfg.to_dict()[key]
            ):
                raise ValueError(f"Resume configuration mismatch: {key}")
        if (
            metadata["world_size"] != backend.world
            or metadata["precision"] != backend.precision
        ):
            raise ValueError("Exact resume requires the same world size and precision")
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        state = {"model": model_state, "optimizer": optimizer_state}
        dcp.load(state, checkpoint_id=Path(path) / "state")
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
            options=options,
        )
        scheduler.load_state_dict(metadata["scheduler"])
        for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
            group["lr"] = lr
        backend.loss_scale = metadata["loss_scale"]
        backend.scale_good_steps = metadata["scale_good_steps"]
        if not _restore_rng_payload(metadata["rng"], backend.rank, backend.world):
            raise ValueError("Checkpoint RNG state cannot be restored")
    return metadata


def initialize_tokenizer(cfg: RewardConfig, smoke: bool = False) -> Any:
    from utils.dataset import verify_tokenizer_vocab

    tokenizer = (
        SmokeTokenizer()
        if smoke
        else AutoTokenizer.from_pretrained(
            cfg.data.tokenizer_name, revision=cfg.data.tokenizer_revision, use_fast=True
        )
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Reward tokenizer requires an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    verify_tokenizer_vocab(tokenizer, cfg.model.vocab_size)
    return tokenizer


def pin_revisions(cfg: RewardConfig) -> None:
    """Resolve moving Hub refs once; checkpoints preserve reproducible sources."""
    from huggingface_hub import HfApi

    try:
        api = HfApi()
        result = {
            field: api.dataset_info(path, revision=getattr(cfg.data, field)).sha
            for field, path in (
                ("ultrafeedback_revision", "HuggingFaceH4/ultrafeedback_binarized"),
                ("hh_rlhf_revision", "Anthropic/hh-rlhf"),
                ("helpsteer2_revision", "nvidia/HelpSteer2"),
            )
        }
        result["tokenizer_revision"] = api.model_info(
            cfg.data.tokenizer_name, revision=cfg.data.tokenizer_revision
        ).sha
    except Exception as exc:
        raise RuntimeError(f"Unable to resolve Hub revisions: {exc}") from exc
    for key, value in result.items():
        setattr(cfg.data, key, value)


def run_training(
    cfg: RewardConfig,
    smoke: bool = False,
    stop_after: int | None = None,
    preflight: bool = False,
    device: torch.device | None = None,
) -> str:
    cfg.validate()
    logger = setup_logging(Path(cfg.system.output_dir))
    feed = None
    try:
        backend = SingleGPURewardBackend(cfg, logger, smoke=smoke, device=device)
        if preflight and (
            smoke
            or backend.world != 1
            or backend.precision != "bf16"
            or not cfg.system.require_fused_mamba
        ):
            raise ValueError("--preflight requires a single BF16 GPU and fused Mamba")
        if cfg.system.require_fused_mamba and not fused_mamba_scan_available():
            raise RuntimeError(
                "require_fused_mamba: install a compatible mamba-ssm CUDA build"
            )
        if not smoke:
            pin_revisions(cfg)
        if cfg.data.pairs_per_shard < backend.world * cfg.training.batch_size:
            raise ValueError("pairs_per_shard must fit at least one global batch")

        tokenizer = initialize_tokenizer(cfg, smoke)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        with torch.device("meta"):
            counts = parameter_counts(RewardModel(cfg.model), enforce_size=not smoke)
        logger.info("reward architecture=%s parameters=%s", vars(cfg.model), counts)
        logger.info(
            "mixture probabilities=%s precision=%s device=%s",
            cfg.data.probabilities(),
            backend.precision,
            backend.device,
        )

        model = RewardModel(cfg.model, cfg.system.activation_checkpointing)
        require_mamba_kernels(model, cfg.system.require_fused_mamba, backend.device)
        model = backend.wrap(model, cfg)

        if not smoke and not fused_mamba_scan_available():
            logger.warning(
                "mamba-ssm unavailable: single-GPU training will use slow PyTorch scans"
            )

        adam_decay = [p for p in model.parameters() if not _is_adamw_no_decay(p)]
        adam_no_decay = [p for p in model.parameters() if _is_adamw_no_decay(p)]
        logger.info(
            "AdamW(lr=%.3e, betas=(0.90, 0.95), wd=%.3g on %d params / wd=0 on %d params)",
            cfg.training.learning_rate,
            cfg.training.weight_decay,
            len(adam_decay),
            len(adam_no_decay),
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": adam_decay, "weight_decay": cfg.training.weight_decay},
                {"params": adam_no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.training.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=False,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            pretrain._build_lr_lambda(
                int(cfg.training.max_steps * cfg.training.warmup_ratio),
                cfg.training.max_steps,
                0.1,
            ),
        )
        step, cursor = (
            0,
            {"epoch": 0, "shard": 0, "batch": 0, "raw_start": 0, "native_start": None},
        )
        if cfg.system.resume_from_checkpoint:
            metadata = load_checkpoint(
                cfg.system.resume_from_checkpoint,
                model,
                optimizer,
                scheduler,
                cfg,
                backend,
            )
            step, cursor = metadata["step"], metadata["cursor"]
        elif cfg.system.initial_checkpoint:
            load_checkpoint(cfg.system.initial_checkpoint, model, cfg=cfg)

        optimizer.zero_grad(set_to_none=True)
        target = (
            min(cfg.training.max_steps, stop_after)
            if stop_after
            else cfg.training.max_steps
        )
        last_saved = None
        last_full_eval = None
        diagnostic = (
            build_diagnostic_set(cfg, tokenizer, backend, smoke)
            if cfg.system.diagnostic_eval_enabled
            else None
        )
        source_counts = {s: Counter() for s in ("ultrafeedback", "hh_rlhf")}
        consumed = Counter()
        sequence_startup_logged = False
        metrics_path = Path(cfg.system.output_dir) / "metrics.jsonl"

        while cursor["epoch"] < cfg.training.num_epochs and step < target:
            feed = PreferenceFeed(cfg, tokenizer, backend, "train", cursor, smoke)
            epoch_updates = 0
            try:
                while step < target:
                    path = feed.wait(cursor["shard"])
                    if path is None:
                        if not epoch_updates and step == 0:
                            raise ValueError(
                                "No trainable preference batches; inspect filtering statistics"
                            )
                        cursor = {
                            "epoch": cursor["epoch"] + 1,
                            "shard": 0,
                            "batch": 0,
                            "raw_start": 0,
                            "native_start": None,
                        }
                        break
                    dataset, loader = loader_for_shard(
                        path,
                        cfg,
                        tokenizer,
                        backend,
                        cursor["shard"] + cursor["epoch"] * 1000000,
                        cursor["batch"],
                    )
                    cursor.update(
                        raw_start=dataset.metadata["raw_start"],
                        native_start=dataset.metadata["native_start"],
                    )
                    if not sequence_startup_logged:
                        logger.info(
                            "RLHF Sequence Statistics (bounded startup prefetch, not full dataset) %s lengths=%s",
                            json.dumps(
                                accounting_summary(
                                    cfg.data, feed.producer.accounting_snapshot(), {}
                                )
                            ),
                            json.dumps(feed.producer.sequence_summary()),
                        )
                        sequence_startup_logged = True
                    iterator = iter(loader)
                    try:
                        while step < target:
                            batches = list(
                                itertools.islice(
                                    iterator, cfg.training.gradient_accumulation_steps
                                )
                            )
                            if not batches:
                                break
                            start = time.monotonic()
                            totals = torch.zeros(
                                6, device=backend.device, dtype=torch.float64
                            )
                            count = sum(b["chosen_input_ids"].size(0) for b in batches)
                            model.train()
                            for i, batch in enumerate(batches):
                                backend.sync_backward(model, i == len(batches) - 1)
                                chosen, rejected = score_batch(model, batch, backend)
                                loss = pairwise_loss(chosen, rejected) * (
                                    chosen.size(0) / count
                                )
                                (loss * backend.loss_scale).backward()
                                consumed.update(batch["source"])
                                totals += metric_sums(chosen, rejected)
                            grad_norm, updated = backend.optimizer_step(
                                model, optimizer, cfg.training.gradient_clip_norm
                            )
                            cursor["batch"] += len(batches)
                            if not updated:
                                logger.warning(
                                    "Update skipped: loss_scale=%g",
                                    backend.loss_scale,
                                )
                                continue
                            scheduler.step()
                            step += 1
                            epoch_updates += 1
                            if step % cfg.system.log_interval == 0:
                                record = metric_means(backend.sum(totals))
                                record.update(
                                    step=step,
                                    epoch=cursor["epoch"],
                                    learning_rate=optimizer.param_groups[0]["lr"],
                                    grad_norm=float(grad_norm),
                                    pairs_per_second=count
                                    * backend.world
                                    / (time.monotonic() - start),
                                    loss_scale=backend.loss_scale,
                                    scheduler_step=scheduler.last_epoch,
                                    optimizer_state_tensors=len(optimizer.state),
                                    gpu_memory_bytes=torch.cuda.max_memory_allocated(
                                        backend.device
                                    )
                                    if backend.device.type == "cuda"
                                    else 0,
                                )
                                logger.info("reward %s", json.dumps(record))
                                with metrics_path.open("a", encoding="utf-8") as handle:
                                    handle.write(json.dumps(record) + "\n")
                            if (
                                diagnostic is not None
                                and step % cfg.system.diagnostic_eval_interval == 0
                            ):
                                result = evaluate_diagnostic(
                                    model, diagnostic, cfg, tokenizer, backend
                                )
                                write_evaluation(
                                    result, cfg, backend, "Diagnostic", step
                                )
                            if step % cfg.system.save_interval == 0:
                                last_saved = (
                                    Path(cfg.system.output_dir)
                                    / f"checkpoint-{step:08d}"
                                )
                                save_checkpoint(
                                    last_saved,
                                    model,
                                    optimizer,
                                    scheduler,
                                    cfg,
                                    backend,
                                    cursor,
                                    step,
                                )
                                if (
                                    cfg.system.full_eval_enabled
                                    and cfg.system.full_eval_at_checkpoints
                                ):
                                    write_evaluation(
                                        evaluate_full(
                                            model, cfg, tokenizer, backend, smoke
                                        ),
                                        cfg,
                                        backend,
                                        "Full",
                                        step,
                                    )
                                    last_full_eval = step
                    finally:
                        del iterator, loader
                        dataset.close()
                    if step < target:
                        cursor.update(
                            shard=cursor["shard"] + 1,
                            batch=0,
                            raw_start=dataset.metadata["raw_end"],
                            native_start=dataset.metadata["native_end"],
                        )
                        feed.consume(path)
            finally:
                feed.close()
                for name, counts in feed.producer.accounting_snapshot().items():
                    if name in source_counts:
                        source_counts[name].update(counts)
                logger.info(
                    "RLHF Sequence Statistics (producer invocation) %s",
                    json.dumps(feed.producer.sequence_summary()),
                )
                feed = None

        final_path = Path(cfg.system.output_dir) / f"checkpoint-{step:08d}"
        if step and final_path != last_saved and not final_path.exists():
            save_checkpoint(
                final_path, model, optimizer, scheduler, cfg, backend, cursor, step
            )

        if (
            cfg.system.full_eval_enabled
            and last_full_eval != step
            and (
                stop_after is None
                or step >= cfg.training.max_steps
                or cursor["epoch"] >= cfg.training.num_epochs
            )
        ):
            write_evaluation(
                evaluate_full(model, cfg, tokenizer, backend, smoke),
                cfg,
                backend,
                "Full",
                step,
            )

        consumed_global = (
            backend.sum(
                torch.tensor(
                    [consumed[s] for s in source_counts],
                    dtype=torch.int64,
                    device=backend.device,
                )
            )
            .cpu()
            .tolist()
        )
        report = accounting_summary(
            cfg.data, source_counts, dict(zip(source_counts, consumed_global))
        )
        logger.info("Dataset Accounting %s", json.dumps(report))
        Path(cfg.system.output_dir, "dataset_accounting.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return str(final_path)
    finally:
        if feed is not None:
            feed.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def parse_device_arg(device_str: str | None) -> torch.device | None:
    if device_str is None:
        return None
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{device_str}' but CUDA is not available")
    return device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", help="Nested dataclass JSON; omitted uses production defaults"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument(
        "--device",
        help="Device to use ('cuda', 'cuda:0', 'cpu', etc.). Defaults to 'cuda:0' if CUDA is available, else 'cpu'.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Single-GPU preflight: BF16 GPU, two updates and DCP save/resume",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Tiny offline CPU model, two updates"
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Stop at this update without changing scheduler horizon",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Meta-device parameter check, no allocation/training",
    )
    parser.add_argument("--write-config", help="Write default configuration and exit")
    args = parser.parse_args()

    if args.smoke and args.preflight:
        parser.error("--smoke and --preflight are mutually exclusive")
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("--stop-after must be a positive update number")

    device = parse_device_arg(args.device)

    if args.resume:
        cfg = RewardConfig.from_dict(checkpoint_metadata(args.resume)["config"])
        cfg.system.resume_from_checkpoint = args.resume
    else:
        cfg = (
            RewardConfig.from_pretrained(args.config) if args.config else RewardConfig()
        )
        if args.smoke:
            cfg = smoke_config(args.output_dir or "runs/reward_smoke")

    if args.output_dir:
        cfg.system.output_dir = args.output_dir

    if args.preflight:
        cfg.system.precision = "bf16"
        cfg.system.require_fused_mamba = True
        cfg.training.max_steps = 2
        cfg.training.batch_size = 1
        cfg.training.gradient_accumulation_steps = 2
        cfg.system.save_interval = cfg.system.diagnostic_eval_interval = 1
        cfg.system.diagnostic_eval_pairs = 4
        cfg.system.full_eval_enabled = False
        cfg.data.buffer_size = 1
        cfg.data.pairs_per_shard = 8

    if args.write_config:
        Path(args.write_config).parent.mkdir(parents=True, exist_ok=True)
        cfg.save_pretrained(args.write_config)
        return

    if args.count_only:
        with torch.device("meta"):
            print(
                json.dumps(
                    parameter_counts(RewardModel(cfg.model), not args.smoke), indent=2
                )
            )
        return

    if args.smoke and (cfg.model.d_model > 32 or cfg.training.max_steps > 4):
        parser.error("--smoke cannot run a production configuration")

    run_training(
        cfg,
        smoke=args.smoke,
        stop_after=args.stop_after,
        preflight=args.preflight,
        device=device,
    )


if __name__ == "__main__":
    main()
