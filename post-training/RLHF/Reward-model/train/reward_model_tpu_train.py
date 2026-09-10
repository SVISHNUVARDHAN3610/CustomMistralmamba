"""Single-host PJRT/SPMD hybrid reward training, targeting Kaggle TPU v5e-8.

Run once with Python, not torchrun/xmp.spawn. See README.md for the Dataset
factory contract, configuration, checkpoint compatibility and Kaggle commands.
XLA imports are lazy so --help and the bounded --smoke CPU test work without XLA.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REWARD_MODEL_DIR = ROOT / "post-training" / "RLHF" / "Reward-model"
for directory in (ROOT, ROOT / "post-training", REWARD_MODEL_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import torch
import torch.nn.functional as F
from reward_model import pairwise_loss
from torch import Tensor, nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    Sampler,
)

from model.core.config import HybridMambaMoEConfig
from model.hybrid.model import HybridModel
from model.layers.moe import DroplessMoELayer

LOGGER = logging.getLogger("reward_tpu")
CHECKPOINT_FAMILY = "hybrid_reward_xla_spmd_v1"


class StaticDroplessMoE(DroplessMoELayer):
    """Same top-k mixture, using fixed shapes instead of nonzero/index dispatch.

    Every expert processes every token; only selected expert outputs contribute.
    This trades extra FLOPs for XLA-compatible static shapes without dropping
    tokens, changing routing probabilities, or adding capacity constraints.
    """

    def forward(self, x, compute_expert_loss=False, expert_var_beta=0.5):
        if compute_expert_loss or self.capacity_factor is not None:
            raise ValueError("TPU static dispatch requires dropless, reward-only MoE")
        flat = x.reshape(-1, x.shape[-1])
        weights, indices, aux, z_loss, _ = self.router(flat)
        output = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            selected_weight = (weights * (indices == index)).sum(-1, keepdim=True)
            output = output + expert(flat) * selected_weight
        return output.reshape_as(x), aux, z_loss, x.new_zeros(())


def validate_model_config(config: HybridMambaMoEConfig, sequence_length: int) -> None:
    """Reject unsupported stateful/auxiliary paths before allocating a model."""
    if config.use_dual_memory or config.use_auxiliary_losses:
        raise ValueError(
            "TPU reward training requires use_dual_memory=false and "
            "use_auxiliary_losses=false; persistent memory training is not supported"
        )
    if config.capacity_factor is not None or config.use_torch_compile:
        raise ValueError("Use capacity_factor=null and use_torch_compile=false for XLA")
    dims = (
        config.vocab_size,
        config.hidden_size,
        config.num_layers,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        config.intermediate_size,
        config.num_experts,
        config.top_k,
        config.mamba_state_size,
        config.mamba_conv_kernel,
        config.mamba_expand,
        sequence_length,
    )
    if any(d <= 0 for d in dims):
        raise ValueError("Model dimensions and sequence length must be positive")
    if (
        config.hidden_size != config.num_heads * config.head_dim
        or config.num_heads % config.num_kv_heads
        or config.head_dim % 2
        or config.top_k > config.num_experts
    ):
        raise ValueError("Invalid attention dimensions or MoE top_k")
    if sequence_length > min(config.max_position_embeddings, config.window_size):
        raise ValueError(
            "Sequence length must fit both the RoPE table and attention window"
        )
    if not 0 <= config.pad_token_id < config.vocab_size:
        raise ValueError("pad_token_id must be inside the vocabulary")


class TPURewardModel(nn.Module):
    """Repository hybrid backbone plus an unrestricted Linear(hidden_size, 1)."""

    def __init__(self, config: HybridMambaMoEConfig):
        super().__init__()
        validate_model_config(config, 1)
        self.config = config
        self.backbone = HybridModel(config)
        for layer in self.backbone.layers:
            original = layer.moe_block
            layer.moe_block = StaticDroplessMoE(original.router, original.experts)
            layer.mamba_block.use_fused_scan = False
        self.reward_head = nn.Linear(config.hidden_size, 1)
        nn.init.normal_(self.reward_head.weight, std=config.init_range)
        nn.init.zeros_(self.reward_head.bias)
        self.checkpoint_fn = None
        self.shard_activation = None

    @staticmethod
    def _layer_forward(layer, hidden, mask):
        # Always take the mask-aware path, including unpadded inputs. Avoid
        # HybridModel.forward's .item() checks and all persistent cache state.
        return layer(
            hidden,
            attention_mask=mask,
            use_cache=False,
            batch_has_padding=True,
            layer_checkpointing_active=True,
        )[0]

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError("Expected matching [B,T] token IDs and masks")
        hidden = self.backbone.embed_tokens(input_ids)
        for layer in self.backbone.layers:
            forward = partial(self._layer_forward, layer)
            if self.training and self.checkpoint_fn is not None:
                hidden = self.checkpoint_fn(forward, hidden, attention_mask)
            else:
                hidden = forward(hidden, attention_mask)
            if self.shard_activation is not None:
                self.shard_activation(hidden)
        hidden = self.backbone.norm(hidden)
        # CPU collator validates nonempty right-padding before device upload.
        indices = attention_mask.long().sum(dim=1) - 1
        final = hidden.gather(1, indices[:, None, None].expand(-1, 1, hidden.size(-1)))
        return self.reward_head(final.squeeze(1))


class FixedPreferenceCollator:
    """Reuse preference collation, then pad to one TPU compilation shape."""

    def __init__(self, length: int, vocab_size: int, pad_token_id: int):
        self.length, self.vocab_size, self.pad_token_id = (
            length,
            vocab_size,
            pad_token_id,
        )

    def __call__(self, records: list[dict]) -> dict[str, Tensor]:
        from rlhf_dataset import PreferenceCollator

        for record in records:
            for side in ("chosen", "rejected"):
                tokens = torch.as_tensor(record[side])
                if (
                    tokens.device.type != "cpu"
                    or tokens.ndim != 1
                    or not 0 < tokens.numel() <= self.length
                ):
                    raise ValueError(
                        "Provide nonempty CPU token sequences <= sequence_length"
                    )
                if tokens.dtype not in (torch.int32, torch.int64):
                    raise ValueError("Preprocessed token IDs must be integers")
                if tokens.min() < 0 or tokens.max() >= self.vocab_size:
                    raise ValueError(
                        "Preprocessed token ID is outside the model vocabulary"
                    )
        batch = PreferenceCollator(self.pad_token_id)(records)
        result = {}
        for side in ("chosen", "rejected"):
            for suffix in ("input_ids", "attention_mask"):
                key = f"{side}_{suffix}"
                tensor = batch[key]
                pad_value = self.pad_token_id if suffix == "input_ids" else 0
                result[key] = F.pad(
                    tensor, (0, self.length - tensor.size(1)), value=pad_value
                )
        return result


class EpochSampler(Sampler):
    """Deterministic map-style ordering with a resume offset, no data replay."""

    def __init__(self, dataset, seed: int, offset: int = 0):
        self.dataset, self.seed, self.offset = dataset, seed, offset

    def __iter__(self):
        sampler = RandomSampler(
            self.dataset, generator=torch.Generator().manual_seed(self.seed)
        )
        return itertools.islice(iter(sampler), self.offset, None)

    def __len__(self):
        return max(0, len(self.dataset) - self.offset)


def parameter_partition(shape, devices: int) -> tuple:
    """Shard the largest divisible parameter dimension; replicate small tensors."""
    spec = [None] * len(shape)
    candidates = [
        i for i, size in enumerate(shape) if size >= devices and size % devices == 0
    ]
    if candidates:
        spec[max(candidates, key=lambda i: shape[i])] = "fsdp"
    return tuple(spec)


class XlaRuntime:
    """One host process controls all eight physical devices through SPMD."""

    def __init__(self, seed: int):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("Launch one Python process, not torchrun or xmp.spawn")
        for key in ("XLA_USE_BF16", "XLA_DOWNCAST_BF16", "XLA_TPU_ENABLE_XRT"):
            if os.environ.get(key, "0") != "0":
                raise RuntimeError(
                    f"Unset {key}; this entry uses PJRT and BF16 autocast"
                )
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.spmd as xs
            import torch_xla.distributed.xla_backend  # register xla:// rendezvous
            import torch_xla.runtime as xr
        except ImportError as exc:
            raise RuntimeError(
                "Install matching torch/torch_xla TPU wheels on Kaggle"
            ) from exc
        if torch.__version__.split(".")[:2] != torch_xla.__version__.split(".")[:2]:
            raise RuntimeError("torch and torch_xla major/minor versions must match")
        if tuple(int(part) for part in torch_xla.__version__.split(".")[:2]) < (2, 6):
            raise RuntimeError("This SPMD entry requires torch/torch_xla 2.6 or newer")
        xr.use_spmd()  # Must precede creation of ANY XLA tensor/device.
        self.device = xm.xla_device()
        count = xr.global_runtime_device_count()
        if (
            xr.device_type() != "TPU"
            or count != 8
            or xr.addressable_runtime_device_count() != 8
        ):
            raise RuntimeError(
                "This entry requires a single TPU host with 8 addressable devices"
            )
        # xla_dist is torch.distributed, not the legacy TPU VM launch utility.
        # Gloo only coordinates the host; XLA inserts TPU gradient collectives.
        import torch.distributed as xla_dist

        if xla_dist.is_initialized():
            raise RuntimeError(
                "Run in a fresh process without an existing process group"
            )
        xla_dist.init_process_group("gloo", init_method="xla://")
        self.xm, self.xs, self.dist, self.devices = xm, xs, xla_dist, count
        self.mesh = xs.Mesh(list(range(count)), (count,), ("fsdp",))
        xm.set_rng_state(seed, self.device)
        LOGGER.info(
            "PJRT SPMD: torch=%s xla=%s devices=%d mesh=%s",
            torch.__version__,
            torch_xla.__version__,
            count,
            self.mesh,
        )

    def shard(self, model, optimizer):
        for parameter in model.parameters():
            spec = parameter_partition(parameter.shape, self.devices)
            self.xs.mark_sharding(parameter, self.mesh, spec)
            # Adam states are lazily created on the first step. Annotate them
            # before executing that graph and again after checkpoint restore.
            for value in optimizer.state.get(parameter, {}).values():
                if isinstance(value, Tensor) and value.shape == parameter.shape:
                    self.xs.mark_sharding(value, self.mesh, spec)

    def mark_step(self):
        self.xm.mark_step()

    def close(self):
        self.xm.wait_device_ops()
        self.dist.destroy_process_group()


def save_checkpoint(
    path, model, optimizer, config, contract, step, epoch, batch, runtime=None
):
    """Atomic single-host CPU checkpoint; distinct from the GPU DCP family."""
    payload = {
        "family": CHECKPOINT_FAMILY,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "contract": contract,
        "step": step,
        "epoch": epoch,
        "batch": batch,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "xla_rng": runtime.xm.get_rng_state(runtime.device) if runtime else None,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if runtime:
        runtime.mark_step()
        # xm.save gathers sharded tensors to this sole host, never every core.
        runtime.xm.save(payload, str(temporary), master_only=True)
    else:
        torch.save(payload, temporary)
    os.replace(temporary, path)
    LOGGER.info("Saved %s at step=%d epoch=%d next_batch=%d", path, step, epoch, batch)


def load_checkpoint(path, config, contract):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("family") != CHECKPOINT_FAMILY
        or checkpoint.get("config") != config.to_dict()
        or checkpoint.get("contract") != contract
    ):
        raise ValueError(
            "Incompatible TPU checkpoint configuration or Dataset/training contract"
        )
    return checkpoint


def train(dataset: Dataset, config: HybridMambaMoEConfig, args) -> Path:
    """Train already-tokenized preference pairs; no HF preprocessing or sampling mix."""
    validate_model_config(config, args.sequence_length)
    if isinstance(dataset, IterableDataset) or not hasattr(dataset, "__len__"):
        raise ValueError("Provide a deterministic map-style preprocessed Dataset")
    batches_per_epoch = len(dataset) // args.batch_size
    if not batches_per_epoch:
        raise ValueError("Dataset must contain at least one full global batch")
    contract = {
        key: getattr(args, key)
        for key in (
            "seed",
            "batch_size",
            "accumulation_steps",
            "sequence_length",
            "learning_rate",
            "weight_decay",
            "gradient_clip",
            "precision",
            "activation_checkpointing",
            "dataset_factory",
            "dataset_kwargs",
            "smoke",
        )
    }
    contract["dataset_length"] = len(dataset)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = load_checkpoint(args.resume, config, contract) if args.resume else None
    runtime = None if args.smoke else XlaRuntime(args.seed)
    device = torch.device("cpu") if runtime is None else runtime.device
    try:
        model = TPURewardModel(config)
        if args.backbone_weights:
            model.backbone.load_state_dict(
                torch.load(args.backbone_weights, map_location="cpu", weights_only=True)
            )
        if checkpoint:
            model.load_state_dict(checkpoint["model"])
        LOGGER.info(
            "Hybrid reward parameters=%s trainable=%s (no vocabulary head)",
            f"{sum(p.numel() for p in model.parameters()):,}",
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        )
        model.to(device)
        groups = [
            {
                "params": [
                    p
                    for p in model.parameters()
                    if not getattr(p, "_no_weight_decay", False)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for p in model.parameters()
                    if getattr(p, "_no_weight_decay", False)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            groups, lr=args.learning_rate, betas=(0.9, 0.95), foreach=False
        )
        if checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            torch.set_rng_state(checkpoint["torch_rng"])
            random.setstate(checkpoint["python_rng"])
            if runtime:
                runtime.xm.set_rng_state(checkpoint["xla_rng"], device)
        if args.activation_checkpointing:
            if runtime:
                from torch_xla.utils.checkpoint import checkpoint as checkpoint_fn
            else:
                from torch.utils.checkpoint import checkpoint as torch_checkpoint

                checkpoint_fn = partial(torch_checkpoint, use_reentrant=False)
            model.checkpoint_fn = checkpoint_fn
        if runtime:
            runtime.shard(model, optimizer)
            model.shard_activation = lambda hidden: runtime.xs.mark_sharding(
                hidden, runtime.mesh, ("fsdp", None, None)
            )
        step = checkpoint["step"] if checkpoint else 0
        start_epoch = checkpoint["epoch"] if checkpoint else 0
        start_batch = checkpoint["batch"] if checkpoint else 0
        destination = Path(args.output_dir) / "latest.pth"
        if step >= args.max_steps or start_epoch >= args.epochs:
            raise ValueError(
                "Checkpoint already reached max_steps/epochs; increase the limits"
            )
        optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()
        for epoch in range(start_epoch, args.epochs):
            offset = start_batch if epoch == start_epoch else 0
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                drop_last=True,
                sampler=EpochSampler(
                    dataset, args.seed + epoch, offset * args.batch_size
                ),
                collate_fn=FixedPreferenceCollator(
                    args.sequence_length, config.vocab_size, config.pad_token_id
                ),
                num_workers=args.num_workers,
                generator=torch.Generator().manual_seed(args.seed + epoch),
                **(
                    {"multiprocessing_context": "spawn", "prefetch_factor": 2}
                    if args.num_workers
                    else {}
                ),
            )
            parallel = None
            if runtime:
                from torch_xla.distributed.parallel_loader import ParallelLoader

                parallel = ParallelLoader(
                    loader,
                    [device],
                    input_sharding=runtime.xs.ShardingSpec(
                        runtime.mesh, ("fsdp", None)
                    ),
                    loader_prefetch_size=2,
                    device_prefetch_size=1,
                )
                batches = parallel.per_device_loader(device)
            else:
                batches = iter(loader)
            try:
                for batch_index, batch in enumerate(batches, start=offset):
                    within_window = (batch_index - offset) % args.accumulation_steps
                    if within_window == 0:
                        window_size = min(
                            args.accumulation_steps, batches_per_epoch - batch_index
                        )
                        metrics = torch.zeros(4, device=device)
                    context = (
                        torch.autocast("xla", dtype=torch.bfloat16)
                        if runtime and args.precision == "bf16"
                        else nullcontext()
                    )
                    with context:
                        chosen = model(
                            batch["chosen_input_ids"], batch["chosen_attention_mask"]
                        )
                        rejected = model(
                            batch["rejected_input_ids"],
                            batch["rejected_attention_mask"],
                        )
                        loss = pairwise_loss(chosen, rejected)
                    (loss / window_size).backward()
                    metrics += (
                        torch.stack(
                            (
                                loss.detach(),
                                (chosen > rejected).float().mean(),
                                chosen.float().mean(),
                                rejected.float().mean(),
                            )
                        ).detach()
                        / window_size
                    )
                    if within_window + 1 != window_size:
                        if runtime:
                            runtime.mark_step()
                        continue
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.gradient_clip, foreach=False
                    )
                    optimizer.step()  # SPMD compiler inserts global gradient synchronization.
                    if runtime:
                        runtime.shard(model, optimizer)
                        runtime.mark_step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    if step % args.log_interval == 0:
                        loss_value, accuracy, chosen_value, rejected_value = (
                            metrics.cpu().tolist()
                        )
                        LOGGER.info(
                            "step=%d epoch=%d loss=%.6f pairwise_accuracy=%.4f chosen_reward=%.4f rejected_reward=%.4f reward_margin=%.4f grad_norm=%.4f lr=%g elapsed_s=%.1f",
                            step,
                            epoch,
                            loss_value,
                            accuracy,
                            chosen_value,
                            rejected_value,
                            chosen_value - rejected_value,
                            float(norm.cpu()),
                            optimizer.param_groups[0]["lr"],
                            time.monotonic() - started,
                        )
                    next_epoch, next_batch = epoch, batch_index + 1
                    if next_batch == batches_per_epoch:
                        next_epoch, next_batch = epoch + 1, 0
                    final = step >= args.max_steps or next_epoch >= args.epochs
                    if final or step % args.save_interval == 0:
                        save_checkpoint(
                            destination,
                            model,
                            optimizer,
                            config,
                            contract,
                            step,
                            next_epoch,
                            next_batch,
                            runtime,
                        )
                    if final:
                        return destination
            finally:
                if parallel is not None:
                    parallel.close()
        return destination
    finally:
        if runtime is not None:
            runtime.close()


def smoke_model_config() -> HybridMambaMoEConfig:
    return HybridMambaMoEConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        intermediate_size=24,
        num_experts=2,
        top_k=2,
        dropout=0.0,
        mamba_state_size=4,
        max_position_embeddings=16,
        window_size=16,
        use_dual_memory=False,
        use_auxiliary_losses=False,
        use_fused_mamba_scan=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config", help="HybridMambaMoEConfig JSON (required on TPU)"
    )
    parser.add_argument(
        "--dataset-factory", help="module:callable returning a preprocessed Dataset"
    )
    parser.add_argument(
        "--dataset-kwargs", default="{}", help="JSON keyword arguments for factory"
    )
    parser.add_argument(
        "--backbone-weights",
        help="Bare HybridModel CPU state_dict (optional initialization)",
    )
    parser.add_argument("--output-dir", default="runs/reward_tpu")
    parser.add_argument("--resume")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Global preference pairs, divisible by 8",
    )
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="At most two tiny CPU steps; never initializes XLA",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.smoke:
        if args.model_config or args.dataset_factory or args.backbone_weights:
            parser.error(
                "--smoke only accepts the built-in tiny model and synthetic data"
            )
        torch.set_num_threads(1)
        args.batch_size, args.sequence_length, args.accumulation_steps = 2, 8, 2
        args.max_steps = min(args.max_steps, 2)
        args.epochs, args.log_interval, args.save_interval = 1, 1, 1
        config = smoke_model_config()
        dataset = [
            {"chosen": [1, 3 + i, 2], "rejected": [1, 20 + i, 5, 2]} for i in range(8)
        ]
    else:
        if not args.model_config or not args.dataset_factory:
            parser.error("TPU training requires --model-config and --dataset-factory")
        if args.batch_size % 8:
            parser.error("Global --batch-size must be divisible by 8")
        config = HybridMambaMoEConfig.from_pretrained(args.model_config)
        module, name = args.dataset_factory.split(":", 1)
        dataset = getattr(importlib.import_module(module), name)(
            **json.loads(args.dataset_kwargs)
        )
    if args.resume and args.backbone_weights:
        parser.error("--resume and --backbone-weights are mutually exclusive")
    if (
        not all(
            math.isfinite(x)
            for x in (args.learning_rate, args.gradient_clip, args.weight_decay)
        )
        or min(
            args.batch_size,
            args.sequence_length,
            args.accumulation_steps,
            args.max_steps,
            args.epochs,
            args.log_interval,
            args.save_interval,
            args.learning_rate,
            args.gradient_clip,
        )
        <= 0
        or args.num_workers < 0
        or args.weight_decay < 0
    ):
        parser.error(
            "Training sizes/rates must be positive; workers/weight_decay nonnegative"
        )
    return train(dataset, config, args)


if __name__ == "__main__":
    main()
