"""FSDP2 + MuonDTensor/AdamW supervised post-training.

    torchrun --standalone --nproc_per_node=4 post-training/sft_fsdp2_post_train.py \
        --pretrained-checkpoint model_ckpt --run-dir runs/sft_fsdp2 \
        --cache-dir data_cache/sft_fsdp2 --batch-size 1 --grad-accum-steps 8

Run/cache directories must be shared across all nodes. This entry point reuses
the pretraining FSDP2 custom-math exclusions, replicated-gradient reductions,
global gradient clipping, full-matrix Muon and consolidated checkpoint helpers.
The shared SFT loop lives in sft_post_train.py; --help documents its data options
including --oversized-behavior (filter/truncate/error) and --exclude-topics.

As in pre-training/fsdp2_train.py, master parameters and reductions stay FP32;
BF16 is enabled through autocast. Layers are sharded before the root, and the
optimizer is built afterward. Checkpointed layers retain gathered parameters
through backward. Full model initialization and consolidated checkpointing have
the same per-rank memory requirements as the pretraining implementation.

SFT resumes require the same world size and data/optimizer settings. To change
topology, --pretrained-checkpoint loads weights with a fresh SFT optimizer/cursor.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import timedelta
from pathlib import Path

# Disable JAX in Hugging Face datasets to prevent background worker circular imports
os.environ["USE_JAX"] = "0"

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in (ROOT, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import sft_post_train as sft


def load_pretraining_fsdp2():
    name = "_sft_pretraining_fsdp2"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "pre-training" / "fsdp2_train.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load pre-training/fsdp2_train.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class FSDP2Backend(sft.SingleGPUBackend):
    family = "sft_fsdp2_muon_v1"

    def __init__(self, args, logger):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FSDP2 post-training requires CUDA; use sft_post_train.py --device cpu for CPU smoke tests"
            )
        if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
            raise RuntimeError("Launch FSDP2 SFT with torchrun")
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(self.device)
        self.logger = logger
        self.base = load_pretraining_fsdp2()
        self.api = self.base._require_fsdp2()
        torch.distributed.init_process_group(
            args.dist_backend, timeout=timedelta(seconds=args.shard_timeout + 300)
        )
        self.rank = torch.distributed.get_rank()
        self.world = torch.distributed.get_world_size()

    def wrap(self, model, args):
        model.to(self.device)
        ignored = self.base._prepare_fsdp2_custom_math_params(model)
        self.replicated = self.base._ordered_replicated_params(model, ignored)
        # Saved thresholds are preserved; calibrate only if the source model
        # never did so, before FSDP hooks can encounter direct calibration calls.
        model.model.calibrate_ssm_norm_thresholds()
        policy = self.api["MixedPrecisionPolicy"](reduce_dtype=torch.float32)
        for layer in model.model.layers:
            layer_ids = {id(p) for p in layer.parameters()}
            self.api["fully_shard"](
                layer,
                mp_policy=policy,
                ignored_params={p for p in ignored if id(p) in layer_ids},
                reshard_after_forward=not args.gradient_checkpointing,
            )
        self.api["fully_shard"](model, mp_policy=policy, ignored_params=ignored)
        return model

    def build_optimizers(self, model, args):
        return self.base.build_fsdp2_optimizers(model, args=args, logger=self.logger)

    def sum(self, tensor):
        torch.distributed.all_reduce(tensor)
        return tensor

    def broadcast(self, value):
        values = [value]
        torch.distributed.broadcast_object_list(values, src=0)
        return values[0]

    def sync_backward(self, model, enabled):
        model.set_requires_gradient_sync(enabled, recurse=True)

    def clip(self, model, max_norm):
        self.base._sync_replicated_param_grads(self.replicated, world_size=self.world)
        return self.base._clip_grad_norm_fsdp2_mixed(
            model.parameters(), max_norm, world_size=self.world
        )

    def model_state(self, model):
        return self.api["get_model_state_dict"](
            model,
            options=self.api["StateDictOptions"](
                full_state_dict=True, cpu_offload=True
            ),
        )

    def optimizer_state(self, optimizer):
        return self.base._consolidate_optimizer_state(optimizer, self.api["DTensor"])

    def restore_optimizer(self, optimizer, state):
        optimizer.load_state_dict(
            self.base._reshard_optimizer_state(
                state, optimizer, self.device, self.api["distribute_tensor"]
            )
        )

    def rng_state(self):
        return self.base._gather_rng_payload(self.rank, self.world)

    def restore_rng(self, state):
        if not self.base._restore_rng_payload(state, self.rank, self.world):
            raise ValueError("Cannot restore this rank's SFT RNG state")


def main():
    args = sft.parse_args(distributed=True, description=__doc__)
    base = load_pretraining_fsdp2()
    logger = base._setup_logging(Path(args.run_dir))
    try:
        backend = FSDP2Backend(args, logger)
        sft.run_training(args, backend, logger)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
