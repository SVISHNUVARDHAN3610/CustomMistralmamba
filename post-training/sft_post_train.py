"""Single-GPU SFT and shared post-training loop for Hybrid Mamba-MoE.

    python post-training/sft_post_train.py --pretrained-checkpoint model_ckpt \
        --run-dir runs/sft --cache-dir data_cache/sft --max-steps 1000
    python post-training/sft_post_train.py --resume runs/sft/model_ckpt.pth \
        --run-dir runs/sft --cache-dir data_cache/sft --max-steps 1000

Warm starts load the saved architecture and model weights only. SFT resumes also
restore optimizer/scheduler/RNG state and the exact next batch. --seq-len is the
number of INPUT tokens; SFT storage windows have seq_len+1 tokens. Whole examples
must fit this context (oversized examples fail explicitly in the data producer).
Use --dataset-config with a JSON list of utils.sft_dataset source configurations
to curate a mixture for a shorter-context model. --offline-shards consumes an
already prepared, contiguous shard cache without starting Hugging Face streams.

Uses train.py's Muon/AdamW grouping, LR schedule, gradient checkpointing and RNG
helpers. BF16 autocast keeps FP32 master parameters; --device cpu --no-amp is
available for smoke tests. CE is normalized by assistant tokens across the whole
accumulation window, while auxiliary regularizers are averaged over microbatches.
Regularizers retain their saved settings; --no-auxiliary-losses disables the
memory/SSM auxiliary terms while keeping the router regularizers.

Rank zero produces SFT shards in a background thread with bounded read-ahead.
Cache shards are retained for checkpoint replay: use a dedicated cache per run
and archive/delete it only after the run is no longer needed. Optional validation
reads a SEPARATE directory of held-out SFT shards, scoring assistant tokens only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as pretrain
from model.core.config import HybridMambaMoEConfig
from model.hybrid.model import HybridForCausalLM
from utils.sft_dataset import (
    DATASET_CONFIGS,
    IGNORE_INDEX,
    MmapShardDataset,
    TokenizedShardProducer,
    verify_tokenizer_vocab,
)


def read_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        path /= pretrain.CHECKPOINT_FILENAME
    from torch.torch_version import TorchVersion

    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Expected a repository checkpoint with config and model_state_dict"
        )
    return checkpoint


class SingleGPUBackend:
    family = "sft_single_gpu_v1"
    rank = 0
    world = 1

    def __init__(self, args, logger):
        self.device = torch.device(args.device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.logger = logger

    def wrap(self, model, args):
        return model.to(self.device)

    def build_optimizers(self, model, args):
        return pretrain.build_optimizers(
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
            device=self.device,
            logger=self.logger,
        )

    def sum(self, tensor):
        return tensor

    def broadcast(self, value):
        return value

    def sync_backward(self, model, enabled):
        pass

    def clip(self, model, max_norm):
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    def model_state(self, model):
        return {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        }

    def optimizer_state(self, optimizer):
        return optimizer.state_dict()

    def restore_optimizer(self, optimizer, state):
        optimizer.load_state_dict(state)

    def rng_state(self):
        return pretrain._rng_state_dict()

    def restore_rng(self, state):
        if not pretrain._load_rng_state_dict(state):
            raise ValueError("Cannot restore SFT RNG state")


class RetainedShardProducer(TokenizedShardProducer):
    """Keep immutable shards so consumer checkpoints can lag producer checkpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_path = str(Path(self.cache_dir) / "producer_state.json")

    def _cleanup_consumed_shards(self):
        # No .done protocol: the trainer advances the read-ahead limit instead.
        pass

    def _write_shard(self, count):
        with self._lock:
            base = Path(self.cache_dir) / f"shard_{self.current_shard_idx:06d}"
            if base.with_suffix(".bin").exists():
                # Crash after atomic shard publication but before producer-state
                # publication: deterministic replay must match, never overwrite.
                tokens = np.asarray(self.token_buffer[:count], dtype="<u4").tobytes()
                mask = bytes(self.loss_mask_buffer[:count])
                if (
                    base.with_suffix(".bin").read_bytes() != tokens
                    or base.with_suffix(".mask").read_bytes() != mask
                ):
                    raise ValueError(
                        f"Cached shard differs from deterministic replay: {base}"
                    )
                reader = MmapShardDataset(str(base.with_suffix(".bin")), self.seq_len)
                reader.close()
                del self.token_buffer[:count]
                del self.loss_mask_buffer[:count]
                self.current_shard_idx += 1
            else:
                # Only incomplete sidecars may remain before the .bin commit.
                for suffix in (".mask", ".json"):
                    base.with_suffix(suffix).unlink(missing_ok=True)
                super()._write_shard(count)
            self.save_checkpoint(self.state_path)


class ShardFeed:
    def __init__(self, args, cfg, tokenizer, sources, backend, cursor, logger):
        self.args, self.backend = args, backend
        self.producer = None
        self.thread = None
        self.stop = threading.Event()
        if backend.rank == 0 and not args.offline_shards:
            self.producer = RetainedShardProducer(
                args.cache_dir,
                tokenizer_name=args.tokenizer_name,
                tokenizer=tokenizer,
                seq_len=args.seq_len + 1,
                tokens_per_shard=args.tokens_per_shard,
                max_buffered_files=cursor + args.max_buffered_files,
                seed=args.seed,
                dataset_configs=sources,
                expected_vocab_size=cfg.vocab_size,
                log_fn=logger.info,
            )
            state_path = self.producer.state_path
            if Path(state_path).exists():
                self.producer.load_checkpoint(state_path)
            elif list(Path(args.cache_dir).glob("shard_*.bin")):
                raise ValueError(
                    "Existing streaming cache has no producer checkpoint; use --offline-shards or a fresh cache"
                )
            self.thread = threading.Thread(
                target=self.producer.start_streaming,
                args=(self.stop, state_path),
                daemon=True,
            )
            self.thread.start()

    def wait(self, index):
        status = None
        if self.backend.rank == 0:
            path = Path(self.args.cache_dir) / f"shard_{index:06d}.bin"
            deadline = time.monotonic() + self.args.shard_timeout
            if self.producer is not None:
                self.producer.max_buffered_files = index + self.args.max_buffered_files
            while True:
                if self.producer is not None and self.producer.error is not None:
                    status = ("error", str(self.producer.error))
                    break
                if path.exists():
                    status = ("ready", str(path))
                    break
                if self.producer is None or self.producer.finished:
                    later = list(path.parent.glob("shard_*.bin"))
                    status = (
                        ("error", f"Missing shard in cache: {path}")
                        if any(p.name > path.name for p in later)
                        else ("end", "")
                    )
                    break
                if time.monotonic() >= deadline:
                    status = ("error", f"Timed out waiting for {path}")
                    break
                time.sleep(0.1)
        status = self.backend.broadcast(status)
        if status[0] == "error":
            raise RuntimeError(status[1])
        return None if status[0] == "end" else status[1]

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                self.backend.logger.warning(
                    "Producer still blocked in a source read; latest published shard checkpoint is durable"
                )


def runtime_contract(args, cfg, sources, backend, use_muon, tokenizer):
    vocab_hash = hashlib.sha256(
        json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
    ).hexdigest()
    return {
        "family": backend.family,
        "world_size": backend.world,
        "config": cfg.to_dict(),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "use_muon": use_muon,
        "lr": args.lr,
        "muon_lr": args.muon_lr,
        "adam_lr": args.adam_lr,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "muon_momentum": args.muon_momentum,
        "muon_ns_steps": args.muon_ns_steps,
        "muon_adjust_lr_fn": args.muon_adjust_lr_fn,
        "no_muon_nesterov": args.no_muon_nesterov,
        "adam_betas": [args.adam_beta1, args.adam_beta2],
        "adam_eps": args.adam_eps,
        "amp": not args.no_amp,
        "tokenizer": args.tokenizer_name,
        "vocab_hash": vocab_hash,
        "sources": sources,
        "tokens_per_shard": args.tokens_per_shard,
        "cache_dir": str(Path(args.cache_dir).resolve()),
    }


def restore_training_state(checkpoint, contract, optimizers, schedulers, backend):
    if checkpoint.get("sft_runtime") != contract:
        raise ValueError(
            "SFT resume contract mismatch (model, data, optimizer, schedule or world size); use --pretrained-checkpoint for a fresh weights-only run"
        )
    if len(checkpoint["optimizers"]) != len(optimizers) or len(
        checkpoint["schedulers"]
    ) != len(schedulers):
        raise ValueError("SFT optimizer/scheduler layout mismatch")
    for optimizer, state in zip(optimizers, checkpoint["optimizers"]):
        backend.restore_optimizer(optimizer, state)
    for optimizer, scheduler, state in zip(
        optimizers, schedulers, checkpoint["schedulers"]
    ):
        scheduler.load_state_dict(state)
        for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
            group["lr"] = lr
    backend.restore_rng(checkpoint["rng_state"])
    return (
        checkpoint["global_step"],
        checkpoint["current_shard_idx"],
        checkpoint["current_batch_idx"],
    )


def save_checkpoint(
    model, optimizers, schedulers, step, shard, batch, args, contract, backend
):
    payload = {
        "model_state_dict": backend.model_state(model),
        "config": model.config.to_dict(),
        "optimizers": [backend.optimizer_state(opt) for opt in optimizers],
        "schedulers": [s.state_dict() for s in schedulers],
        "rng_state": backend.rng_state(),
        "global_step": step,
        "current_shard_idx": shard,
        "current_batch_idx": batch,
        "sft_runtime": contract,
    }
    result = None
    if backend.rank == 0:
        try:
            destination = Path(args.run_dir) / pretrain.CHECKPOINT_FILENAME
            tmp = destination.with_suffix(".pth.tmp")
            torch.save(payload, tmp)
            os.replace(tmp, destination)
            TokenizedShardProducer._atomic_json(
                str(destination.parent / "config.json"), payload["config"]
            )
            result = (True, "")
        except OSError as exc:
            result = (False, str(exc))
    success, message = backend.broadcast(result)
    if not success:
        raise OSError(message)
    backend.logger.info(
        "SFT checkpoint step=%d shard=%d next_batch=%d", step, shard, batch
    )


def shard_loader(path, args, backend, index, offset):
    dataset = MmapShardDataset(path, args.seq_len + 1)
    sampler = DistributedSampler(
        dataset,
        num_replicas=backend.world,
        rank=backend.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=backend.world > 1,
    )
    sampler.set_epoch(index)
    indices = list(sampler)
    batches = math.ceil(len(indices) / args.batch_size)
    if not 0 <= offset <= batches:
        dataset.close()
        raise ValueError("Checkpoint batch cursor is outside this shard")
    indices = indices[offset * args.batch_size :]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=backend.device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed + index),
    )
    return dataset, loader


def validate_batch(inputs, labels, vocab_size):
    valid = labels != IGNORE_INDEX
    if inputs.numel() == 0 or inputs.min() < 0 or inputs.max() >= vocab_size:
        raise ValueError("Input token ID outside model vocabulary")
    if not valid.any() or labels[valid].min() < 0 or labels[valid].max() >= vocab_size:
        raise ValueError("SFT batch has invalid or missing assistant targets")
    return int(valid.sum())


def accumulated_loss(output, count, global_count, world, microbatches):
    # FSDP2 averages gradients over ranks; world compensates so CE is the
    # global assistant-token mean, not the mean of unequal per-rank means.
    return (
        output.ce_loss * (count * world / global_count)
        + (output.loss - output.ce_loss) / microbatches
    )


def train_window(model, batches, optimizers, args, backend, step):
    counts = [validate_batch(x, y, model.config.vocab_size) for x, y in batches]
    total = backend.sum(
        torch.tensor(sum(counts), device=backend.device, dtype=torch.float64)
    )
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    metrics = torch.zeros(2, device=backend.device, dtype=torch.float64)
    for i, ((inputs, labels), count) in enumerate(zip(batches, counts)):
        backend.sync_backward(model, i == len(batches) - 1)
        inputs, labels = inputs.to(backend.device), labels.to(backend.device)
        with torch.autocast(
            device_type=backend.device.type,
            dtype=torch.bfloat16,
            enabled=backend.device.type == "cuda" and not args.no_amp,
            cache_enabled=not args.gradient_checkpointing,
        ):
            output = model(
                input_ids=inputs,
                labels=labels,
                use_cache=False,
                training_step=step,
                max_training_steps=args.max_steps,
            )
            loss = accumulated_loss(output, count, total, backend.world, len(batches))
        # All ranks agree before entering backward; a local skip would desync.
        finite = backend.sum(torch.isfinite(loss).to(torch.int32))
        if int(finite) != backend.world:
            raise FloatingPointError("Non-finite SFT loss; optimizer step aborted")
        loss.backward()
        metrics[0] += output.ce_loss.detach().double() * count
        metrics[1] += count
    norm = backend.clip(model, args.max_grad_norm)
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite SFT gradient norm; optimizer step aborted")
    for optimizer in optimizers:
        optimizer.step()
    metrics = backend.sum(metrics)
    return float(metrics[0] / metrics[1]), int(metrics[1]), float(norm)


@torch.no_grad()
def evaluate(model, args, backend):
    paths = sorted(Path(args.validation_dir).glob("shard_*.bin"))
    if not paths:
        raise ValueError("Validation directory contains no SFT shards")
    was_training = model.training
    model.eval()
    totals = torch.zeros(2, device=backend.device, dtype=torch.float64)
    try:
        # Identical evaluation batches on all ranks keep FSDP forward
        # collectives aligned even for very small held-out sets.
        used = 0
        for path in paths:
            dataset = MmapShardDataset(str(path), args.seq_len + 1)
            try:
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    generator=torch.Generator().manual_seed(args.seed),
                )
                for inputs, labels in loader:
                    count = validate_batch(inputs, labels, model.config.vocab_size)
                    with torch.autocast(
                        device_type=backend.device.type,
                        dtype=torch.bfloat16,
                        enabled=backend.device.type == "cuda" and not args.no_amp,
                    ):
                        output = model(
                            input_ids=inputs.to(backend.device),
                            labels=labels.to(backend.device),
                            use_cache=False,
                        )
                    totals[0] += output.ce_loss.double() * count
                    totals[1] += count
                    used += 1
                    if used >= args.val_batches:
                        return float(totals[0] / totals[1])
            finally:
                dataset.close()
        return float(totals[0] / totals[1])
    finally:
        model.train(was_training)


def run_training(args, backend, logger):
    pretrain.set_seed(args.seed)
    checkpoint = read_checkpoint(args.resume or args.pretrained_checkpoint)
    cfg = HybridMambaMoEConfig.from_dict(checkpoint["config"])
    if cfg.label_ignore_index != IGNORE_INDEX:
        raise ValueError("SFT shards require config.label_ignore_index=-100")
    args.seq_len = args.seq_len or cfg.max_position_embeddings
    if args.seq_len > cfg.max_position_embeddings:
        raise ValueError("--seq-len exceeds the pretrained model's supported context")
    args.tokens_per_shard = args.tokens_per_shard or (args.seq_len + 1) * max(
        32, args.batch_size * backend.world
    )
    if args.tokens_per_shard % (args.seq_len + 1):
        raise ValueError("--tokens-per-shard must be a multiple of --seq-len + 1")
    cfg.return_logits = False
    cfg.use_cache = False
    cfg.use_torch_compile = False
    if args.auxiliary_losses is not None:
        cfg.use_auxiliary_losses = args.auxiliary_losses
    if args.no_fused_mamba:
        cfg.use_fused_mamba_scan = False
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    verify_tokenizer_vocab(tokenizer, cfg.vocab_size)
    for token in ("bos_token_id", "eos_token_id"):
        if getattr(tokenizer, token) != getattr(cfg, token):
            raise ValueError(
                f"Tokenizer {token} does not match the pretrained checkpoint"
            )
    sources = DATASET_CONFIGS
    if args.dataset_config:
        sources = json.loads(Path(args.dataset_config).read_text(encoding="utf-8"))
    model = HybridForCausalLM(cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    pretrain.configure_gradient_checkpointing(
        model, args.gradient_checkpointing, logger
    )
    model = backend.wrap(model, args)
    model.train()
    optimizers, use_muon, _ = backend.build_optimizers(model, args)
    lr_fn = pretrain._build_lr_lambda(
        args.warmup_steps, args.max_steps, args.min_lr_ratio
    )
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_fn) for opt in optimizers]
    contract = runtime_contract(args, cfg, sources, backend, use_muon, tokenizer)
    step, shard, offset = 0, 0, 0
    if args.resume:
        step, shard, offset = restore_training_state(
            checkpoint, contract, optimizers, schedulers, backend
        )
    del checkpoint
    if backend.rank == 0:
        TokenizedShardProducer._atomic_json(
            str(Path(args.run_dir) / "sft_config.json"), contract
        )
        logger.info(
            "SFT %s | seq_len=%d effective_batch=%d | pretrained weights loaded",
            backend.family,
            args.seq_len,
            args.batch_size * args.grad_accum_steps * backend.world,
        )
    feed = ShardFeed(args, cfg, tokenizer, sources, backend, shard, logger)
    try:
        while step < args.max_steps:
            path = feed.wait(shard)
            if path is None:
                logger.info("SFT source exhausted at step=%d", step)
                break
            dataset, loader = shard_loader(path, args, backend, shard, offset)
            try:
                iterator = iter(loader)
                while step < args.max_steps:
                    batches = list(itertools.islice(iterator, args.grad_accum_steps))
                    if not batches:
                        break
                    ce, tokens, norm = train_window(
                        model, batches, optimizers, args, backend, step
                    )
                    for scheduler in schedulers:
                        scheduler.step()
                    step += 1
                    offset += len(batches)
                    if backend.rank == 0 and (
                        step == 1 or step % args.log_interval == 0
                    ):
                        record = {
                            "step": step,
                            "ce_loss": ce,
                            "assistant_tokens": tokens,
                            "grad_norm": norm,
                            "lr": [s.get_last_lr() for s in schedulers],
                        }
                        logger.info(
                            "SFT step=%d ce=%.5f assistant_tokens=%d grad_norm=%.4f",
                            step,
                            ce,
                            tokens,
                            norm,
                        )
                        with (Path(args.run_dir) / "metrics.jsonl").open(
                            "a", encoding="utf-8"
                        ) as handle:
                            handle.write(json.dumps(record) + "\n")
                    if args.validation_dir and step % args.val_interval == 0:
                        logger.info(
                            "SFT validation step=%d assistant_ce=%.5f",
                            step,
                            evaluate(model, args, backend),
                        )
                    if step % args.save_interval == 0:
                        save_checkpoint(
                            model,
                            optimizers,
                            schedulers,
                            step,
                            shard,
                            offset,
                            args,
                            contract,
                            backend,
                        )
            finally:
                dataset.close()
            if step < args.max_steps:
                shard += 1
                offset = 0
        if step == 0:
            raise ValueError(
                "No trainable SFT batches were found; check the cache and per-rank shard size"
            )
        save_checkpoint(
            model, optimizers, schedulers, step, shard, offset, args, contract, backend
        )
    finally:
        feed.close()


def parse_args(argv=None, *, distributed=False, description=None):
    parser = argparse.ArgumentParser(
        description=description or __doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pretrained-checkpoint",
        help="Repository .pth checkpoint or directory; load weights only",
    )
    source.add_argument(
        "--resume", help="SFT .pth checkpoint or directory; restore full training state"
    )
    parser.add_argument(
        "--run-dir", default="runs/sft_fsdp2" if distributed else "runs/sft"
    )
    parser.add_argument(
        "--cache-dir",
        default="data_cache/sft_fsdp2" if distributed else "data_cache/sft",
    )
    parser.add_argument("--tokenizer-name", default="UIC-AI-lab/llama2-tokenizer")
    parser.add_argument(
        "--dataset-config", help="JSON list of weighted SFT source configs"
    )
    parser.add_argument("--offline-shards", action="store_true")
    parser.add_argument(
        "--seq-len",
        type=int,
        help="Input context length (default: saved model context)",
    )
    parser.add_argument("--tokens-per-shard", type=int)
    parser.add_argument("--max-buffered-files", type=int, default=3)
    parser.add_argument("--shard-timeout", type=float, default=1800)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Per-rank microbatch size"
    )
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--muon-lr", type=float)
    parser.add_argument("--adam-lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--no-muon", action="store_true")
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--no-muon-nesterov", action="store_true")
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument(
        "--muon-adjust-lr-fn",
        default="match_rms_adamw",
        choices=["match_rms_adamw", "original"],
    )
    parser.add_argument("--muon-gather-buffer-mb", type=float, default=64)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--auxiliary-losses",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override saved memory/SSM auxiliary-loss setting",
    )
    parser.add_argument("--no-fused-mamba", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--validation-dir", help="Separate held-out SFT shard directory"
    )
    parser.add_argument("--val-interval", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=20)
    if distributed:
        parser.add_argument("--dist-backend", choices=["nccl", "gloo"], default="nccl")
    else:
        parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args(argv)
    for name in (
        "max_steps",
        "batch_size",
        "grad_accum_steps",
        "save_interval",
        "log_interval",
        "val_interval",
        "val_batches",
        "max_buffered_files",
        "shard_timeout",
        "max_grad_norm",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seq_len is not None and args.seq_len < 1:
        parser.error("--seq-len must be positive")
    if args.tokens_per_shard is not None and args.tokens_per_shard < 2:
        parser.error("--tokens-per-shard must be >=2")
    if args.warmup_steps < 0 or not 0 <= args.min_lr_ratio <= 1:
        parser.error("Invalid warmup or min-lr-ratio")
    if (
        args.validation_dir
        and Path(args.validation_dir).resolve() == Path(args.cache_dir).resolve()
    ):
        parser.error("Validation shards must be separate from the training cache")
    output = Path(args.run_dir) / pretrain.CHECKPOINT_FILENAME
    if not args.resume and output.exists():
        parser.error(
            "Run directory already contains a checkpoint; use --resume or a fresh --run-dir"
        )
    return args


def main():
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Use sft_fsdp2_post_train.py for multi-process training")
    logger = pretrain.setup_logging(Path(args.run_dir))
    run_training(args, SingleGPUBackend(args, logger), logger)


if __name__ == "__main__":
    main()
