"""Reward-stage training: torchrun for cloud FSDP2, --smoke for tiny CPU tests."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import threading
import time
import uuid
from collections import Counter
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "post-training"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
os.environ["USE_JAX"] = "0"

import sft_post_train as sft
import torch
from sft_fsdp2_post_train import load_pretraining_fsdp2
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers import AutoTokenizer

from model.hybrid.mamba import fused_mamba_scan_available
from model.hybrid.model import _checkpoint_autocast_contexts
from RLHF.config import RewardConfig, smoke_config
from RLHF.evaluation_metrics import RewardStatistics, isolated_evaluation_rng
from RLHF.reward_model import RewardModel, pairwise_loss, parameter_counts
from RLHF.rlhf_dataset import (
    PreferenceCollator,
    PreferenceShardDataset,
    PreferenceShardProducer,
    SmokeTokenizer,
    accounting_summary,
    smoke_stream,
)

FAMILY = "pure_mamba_reward_dcp_v1"


class RewardBackend(sft.SingleGPUBackend):
    """Reward-specific precision/sharding on the repository distributed helpers."""

    def __init__(self, cfg: RewardConfig, logger, smoke: bool = False):
        self.logger, self.smoke = logger, smoke
        self.base = load_pretraining_fsdp2()
        self.rank, self.world, self.device = 0, 1, torch.device("cpu")
        if smoke:
            if int(os.environ.get("WORLD_SIZE", "1")) != 1:
                raise RuntimeError("--smoke must run as a single CPU process")
        else:
            if not torch.cuda.is_available() or "LOCAL_RANK" not in os.environ:
                raise RuntimeError(
                    "Cloud reward training requires CUDA and torchrun; use --smoke locally"
                )
            self.api = self.base._require_fsdp2()
            self.rank, self.world, self.device = self.base.init_distributed("nccl")
        precision = cfg.system.precision
        if precision == "auto":
            precision = (
                "fp32"
                if smoke
                else ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
            )
        if (
            precision == "bf16"
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("BF16 requested on unsupported GPU; choose auto or fp16")
        self.dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[
            precision
        ]
        self.precision = precision
        self.loss_scale = 65536.0 if precision == "fp16" else 1.0
        self.scale_good_steps = 0

    def wrap(self, model: RewardModel, cfg: RewardConfig):
        if self.smoke:
            return model.to(self.device)
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            checkpoint_wrapper,
        )

        policy = self.api["MixedPrecisionPolicy"](reduce_dtype=torch.float32)
        # Each wrapper owns the complete layer. During backward FSDP gathers
        # before checkpoint replay reads A_log/D/norm directly. No ignored_params
        # (unsupported on 2.6) and no inner Mamba checkpoint regions are needed.
        model.activation_checkpointing = False
        for index, layer in enumerate(model.layers):
            if cfg.system.activation_checkpointing:
                layer = checkpoint_wrapper(
                    layer,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                    context_fn=partial(_checkpoint_autocast_contexts, "cuda"),
                )
                model.layers[index] = layer
            self.api["fully_shard"](layer, mp_policy=policy, reshard_after_forward=True)
        self.api["fully_shard"](model, mp_policy=policy, reshard_after_forward=True)
        # FSDP replaces Parameter objects; preserve Mamba optimizer metadata.
        for name, parameter in model.named_parameters():
            if name.endswith((".A_log", ".D")):
                parameter._no_weight_decay = True
        return model

    def sum(self, tensor):
        if not self.smoke:
            torch.distributed.all_reduce(tensor)
        return tensor

    def broadcast(self, value):
        if not self.smoke:
            values = [value]
            torch.distributed.broadcast_object_list(values, src=0)
            return values[0]
        return value

    def barrier(self):
        if not self.smoke:
            torch.distributed.barrier()

    def sync_backward(self, model, enabled):
        if not self.smoke:
            model.set_requires_gradient_sync(enabled, recurse=True)

    def autocast(self):
        return (
            nullcontext()
            if self.dtype is None
            else torch.autocast(self.device.type, dtype=self.dtype, cache_enabled=False)
        )

    def optimizer_step(self, model, optimizer, max_norm):
        # DTensor-local unscale avoids GradScaler's mixed Tensor/DTensor foreach
        # assumptions. The existing global norm collective supplies one shared
        # overflow decision so ranks always step/skip together.
        for param in model.parameters():
            if param.grad is not None:
                grad = (
                    param.grad.to_local()
                    if hasattr(param.grad, "to_local")
                    else param.grad
                )
                grad.div_(self.loss_scale)
        norm = self.base._clip_grad_norm_fsdp2_mixed(
            model.parameters(), max_norm, world_size=self.world
        )
        if not torch.isfinite(norm):
            if self.precision != "fp16":
                raise FloatingPointError("Nonfinite reward gradient norm")
            self.loss_scale /= 2
            self.scale_good_steps = 0
            if self.loss_scale < 1e-8:
                raise FloatingPointError(
                    "Persistent FP16 overflow; use FP32 or inspect data"
                )
            optimizer.zero_grad(set_to_none=True)
            return norm, False
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.scale_good_steps += 1
        if self.precision == "fp16" and self.scale_good_steps % 2000 == 0:
            self.loss_scale = min(self.loss_scale * 2, 2**24)
        return norm, True


class PreferenceFeed(sft.ShardFeed):
    """Reuse SFT's thread lifecycle, rank-zero wait/broadcast, and error handling."""

    def __init__(self, cfg, tokenizer, backend, purpose, cursor=None, smoke=False):
        self.backend = backend
        # Per-invocation queue; checkpoints carry the consumer cursor and pinned
        # native HF boundary, so no stale/prefetched files need to survive resume.
        token = backend.broadcast(uuid.uuid4().hex if backend.rank == 0 else None)
        cache = Path(cfg.system.output_dir) / "data_cache" / token
        self.args = SimpleNamespace(
            cache_dir=str(cache),
            max_buffered_files=cfg.data.buffer_size,
            shard_timeout=cfg.data.shard_timeout,
        )
        self.producer = self.thread = None
        self.stop = threading.Event()
        if backend.rank == 0:
            self.producer = PreferenceShardProducer(
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

    def wait(self, index):
        """SFT readiness protocol adapted to a released, fixed-size shard queue.

        The unchanged SFT feed retains consumed shards and grows its file limit
        with the cursor. Reward checkpoints use source cursors, so its wait must
        keep the configured bound constant instead.
        """
        status = None
        if self.backend.rank == 0:
            path = Path(self.args.cache_dir) / f"shard_{index:06d}.bin"
            deadline = time.monotonic() + self.args.shard_timeout
            while True:
                if self.producer.error is not None:
                    status = ("error", str(self.producer.error))
                    break
                if path.exists():  # producer publishes bin only after metadata
                    status = ("ready", str(path))
                    break
                if self.producer.finished:
                    later = any(
                        p.name > path.name for p in path.parent.glob("shard_*.bin")
                    )
                    status = (
                        ("error", f"Missing reward shard {path}")
                        if later
                        else ("end", "")
                    )
                    break
                if time.monotonic() >= deadline:
                    status = ("error", f"Timed out waiting for reward shard {path}")
                    break
                time.sleep(0.1)
        status = self.backend.broadcast(status)
        if status[0] == "error":
            raise RuntimeError(status[1])
        return None if status[0] == "end" else status[1]

    def consume(self, path):
        self.backend.barrier()
        if self.backend.rank == 0:
            Path(path).with_suffix(".done").touch()

    def close(self):
        super().close()
        if self.producer is not None and not self.thread.is_alive():
            # This invocation owns the UUID directory. Checkpoints resume from
            # source cursors, so even unconsumed prefetch files can be released.
            for path in Path(self.args.cache_dir).glob("shard_*.bin"):
                path.with_suffix(".done").touch()
            self.producer._cleanup_consumed_shards()
            self.backend.logger.info(
                "[Reward Producer final] %s filters=%s",
                dict(self.producer.stats),
                dict(self.producer.reasons),
            )


def loader_for_shard(path, cfg, tokenizer, backend, shard, offset=0, training=True):
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
        # Every rank executes the same number of scoring calls. Score each eval
        # batch on all ranks; metrics divide out replication (no padding bias).
        indices = range(len(dataset))
    kwargs = {
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


def score_batch(model, batch, backend):
    """One model call for both branches, with one live parameter set."""
    ids = torch.cat([batch["chosen_input_ids"], batch["rejected_input_ids"]]).to(
        backend.device, non_blocking=True
    )
    mask = torch.cat(
        [batch["chosen_attention_mask"], batch["rejected_attention_mask"]]
    ).to(backend.device, non_blocking=True)
    with backend.autocast():
        rewards = model(input_ids=ids, attention_mask=mask)
    return rewards.float().chunk(2, dim=0)


def metric_sums(chosen, rejected):
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


def metric_means(totals):
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


def evaluation_batches(cfg, tokenizer, backend, purpose, smoke=False, max_batches=None):
    """Separate bounded source feed; never uses a training feed or its cursor."""
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


def build_diagnostic_set(cfg, tokenizer, backend, smoke=False):
    """Cache fixed source prefixes once: ceil(N/2) UF, floor(N/2) HH pairs.

    Pinned revisions and fixed source order determine membership. A private RNG
    seeded with cfg.seed orders each tiny subset. Cache identity is content based
    and independent of training epoch/cursor and checkpoint progress.
    """
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
                iterator.close()
            if len(rows) != count:
                raise ValueError(
                    f"{purpose}: requested {count} diagnostic pairs, only {len(rows)} survived filtering"
                )
            random.Random(cfg.seed).shuffle(rows)
            result[purpose] = rows
    return result


@torch.no_grad()
def evaluate_batches(model, batches, backend):
    """Global statistics for the existing replicated FSDP evaluation protocol.

    Every rank scores the same rows and executes identical forward collectives.
    Rank zero owns each observation exactly once, including exact percentiles;
    only the final small report is broadcast. No rank-local means/percentiles are
    averaged and no reward arrays are gathered or replicated for statistics.
    """
    was_training = model.training
    groups = {}
    with isolated_evaluation_rng():
        model.eval()
        try:
            for batch in batches:
                chosen, rejected = score_batch(model, batch, backend)
                status = None
                if backend.rank == 0:
                    try:
                        chosen, rejected = (
                            chosen.detach().cpu(),
                            rejected.detach().cpu(),
                        )
                        if "overall" not in groups:
                            groups["overall"] = RewardStatistics()
                        groups["overall"].update(chosen, rejected)
                        for field, prefix in (
                            ("category", "category/"),
                            ("original_split", "original_"),
                        ):
                            for value in sorted(set(batch[field])):
                                if value in ("overall", "unknown"):
                                    continue
                                indices = [
                                    i for i, v in enumerate(batch[field]) if v == value
                                ]
                                key = prefix + value
                                if key not in groups:
                                    groups[key] = RewardStatistics()
                                groups[key].update(chosen[indices], rejected[indices])
                    except Exception as exc:  # noqa: BLE001 -- broadcast rank-zero failures
                        status = str(exc)
                status = backend.broadcast(status)
                if status is not None:
                    raise RuntimeError(f"Reward evaluation statistics failed: {status}")
            payload = None
            if backend.rank == 0:
                try:
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
                    payload = {"result": result}
                except Exception as exc:  # noqa: BLE001 -- broadcast rank-zero failures
                    payload = {"error": str(exc)}
            payload = backend.broadcast(payload)
            if "error" in payload:
                raise RuntimeError(f"Reward evaluation failed: {payload['error']}")
            return payload["result"]
        finally:
            if hasattr(batches, "close"):
                batches.close()
            for group in groups.values():
                group.close()
            model.train(was_training)


def evaluate_diagnostic(model, diagnostic, cfg, tokenizer, backend):
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
    model, cfg, tokenizer, backend, purpose="validation", smoke=False, max_batches=None
):
    """Stream a complete source, or a clearly bounded development sample."""
    if purpose == "validation":
        return {
            name: evaluate(model, cfg, tokenizer, backend, name, smoke, max_batches)
            for name in ("ultrafeedback_test", "hh_rlhf_test")
        }
    result = evaluate_batches(
        model,
        evaluation_batches(cfg, tokenizer, backend, purpose, smoke, max_batches),
        backend,
    )
    result["evaluation_scope"] = (
        "synthetic_fixture"
        if smoke
        else ("bounded_sample" if max_batches is not None else "full_split")
    )
    return result


def evaluate_full(model, cfg, tokenizer, backend, smoke=False, max_batches=None):
    return {
        name: evaluate(model, cfg, tokenizer, backend, name, smoke, max_batches)
        for name in ("ultrafeedback_test", "hh_rlhf_test", "helpsteer2")
    }


def write_evaluation(result, cfg, backend, kind, step):
    if backend.rank == 0:
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
        backend.logger.info(
            "%s Evaluation step=%s %s", kind, step, json.dumps(headline)
        )
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
                json.dumps(
                    {"step": step, "evaluation": kind.lower(), "datasets": result}
                )
                + "\n"
            )


def checkpoint_metadata(path):
    path = Path(path)
    if not (path / "complete").is_file():
        raise ValueError(f"Incomplete reward checkpoint: {path}")
    metadata = torch.load(path / "trainer.pt", map_location="cpu", weights_only=True)
    if metadata.get("family") != FAMILY:
        raise ValueError("Incompatible checkpoint: expected pure Mamba reward model")
    return metadata


def save_checkpoint(path, model, optimizer, scheduler, cfg, backend, cursor, step):
    """DCP stores sharded model/optimizer; small trainer metadata commits last."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    path = Path(path)
    exists = backend.broadcast(path.exists() if backend.rank == 0 else None)
    if exists:
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
    rng = backend.base._gather_rng_payload(backend.rank, backend.world)
    if backend.rank == 0:
        payload = {
            "family": FAMILY,
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
    backend.barrier()


def load_checkpoint(
    path, model, optimizer=None, scheduler=None, cfg=None, backend=None
):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_state_dict,
        set_model_state_dict,
        set_state_dict,
    )

    metadata = checkpoint_metadata(path)
    if cfg is not None and metadata["config"]["model"] != cfg.to_dict()["model"]:
        raise ValueError("Checkpoint reward architecture differs from configuration")
    options = StateDictOptions(full_state_dict=False)
    if optimizer is None:
        state = {"model": get_model_state_dict(model, options=options)}
        dcp.load(state, checkpoint_id=Path(path) / "state")
        set_model_state_dict(model, state["model"], options=options)
    else:
        for key in ("seed", "model", "data", "training"):
            # Fill new observability/default strategy fields for old checkpoints.
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
        if not backend.base._restore_rng_payload(
            metadata["rng"], backend.rank, backend.world
        ):
            raise ValueError("Checkpoint RNG state cannot be restored")
    return metadata


def initialize_tokenizer(cfg, smoke=False):
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


def pin_revisions(cfg, backend):
    """Resolve moving Hub refs once; checkpoints preserve reproducible sources."""
    from huggingface_hub import HfApi

    result = None
    if backend.rank == 0:
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
        except Exception as exc:  # noqa: BLE001 -- broadcast failure so peers do not hang
            result = {"error": str(exc)}
    result = backend.broadcast(result)
    if "error" in result:
        raise RuntimeError(f"Unable to resolve Hub revisions: {result['error']}")
    for key, value in result.items():
        setattr(cfg.data, key, value)


def run_training(cfg: RewardConfig, smoke=False, stop_after=None):
    cfg.validate()
    base = load_pretraining_fsdp2()
    logger = base._setup_logging(Path(cfg.system.output_dir))
    feed = None
    try:
        backend = RewardBackend(cfg, logger, smoke)
        if not smoke:
            pin_revisions(cfg, backend)
        if cfg.data.pairs_per_shard < backend.world * cfg.training.batch_size:
            raise ValueError("pairs_per_shard must fit at least one global batch")
        tokenizer = initialize_tokenizer(cfg, smoke)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        with torch.device("meta"):
            counts = parameter_counts(RewardModel(cfg.model), enforce_size=not smoke)
        if backend.rank == 0:
            logger.info("reward architecture=%s parameters=%s", vars(cfg.model), counts)
            logger.info(
                "mixture probabilities=%s precision=%s",
                cfg.data.probabilities(),
                backend.precision,
            )
        model = RewardModel(cfg.model, cfg.system.activation_checkpointing)
        model = backend.wrap(model, cfg)
        if not smoke and not fused_mamba_scan_available():
            logger.warning(
                "mamba-ssm unavailable: cloud training will use slow PyTorch scans"
            )
        args = SimpleNamespace(
            no_muon=True,
            lr=cfg.training.learning_rate,
            adam_lr=None,
            muon_lr=None,
            weight_decay=cfg.training.weight_decay,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
        )
        optimizers, _, _ = base.build_fsdp2_optimizers(model, args=args, logger=logger)
        optimizer = optimizers[0]
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            sft.pretrain._build_lr_lambda(
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
                    if not sequence_startup_logged and backend.rank == 0:
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
                                    "FP16 overflow: skipped update; loss_scale=%g",
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
                                if backend.rank == 0:
                                    logger.info("reward %s", json.dumps(record))
                                    with metrics_path.open(
                                        "a", encoding="utf-8"
                                    ) as handle:
                                        handle.write(json.dumps(record) + "\n")
                            if (
                                diagnostic is not None
                                and step
                                % (
                                    cfg.system.diagnostic_eval_interval
                                    or cfg.system.eval_interval
                                )
                                == 0
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
                if backend.rank == 0:
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
        # A deliberate early stop is an interruption, not end-of-training evaluation.
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
        if backend.rank == 0:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", help="Nested dataclass JSON; omitted uses production defaults"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
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
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("--stop-after must be a positive update number")
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
    if args.write_config:
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
    run_training(cfg, args.smoke, args.stop_after)


if __name__ == "__main__":
    main()
