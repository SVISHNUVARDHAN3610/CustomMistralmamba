# Reward Model Training Runners

This directory contains the reward-model trainers for different hardware targets. The single-GPU and FSDP2 trainers train the pure-Mamba Reward Model; the TPU trainer trains a hybrid-backbone variant. Full design notes, dataset schemas, and diagnostics are in the [RLHF guide](../../README.md).

## Available Runners

| Script | Hardware Target | Distributed Engine | Entry Method | Model |
| :--- | :--- | :--- | :--- | :--- |
| [`train_reward_model_single_gpu.py`](train_reward_model_single_gpu.py) | Single GPU | Single process (no communication backend) | `python` | Pure-Mamba, 352,798,721 params |
| [`train_reward_model.py`](train_reward_model.py) | Multi-GPU node (NVIDIA) | PyTorch FSDP2 (`fully_shard`) | `torchrun` | Pure-Mamba, 352,798,721 params |
| [`reward_model_tpu_train.py`](reward_model_tpu_train.py) | Kaggle TPU v5e-8 | `torch_xla` (PJRT runtime, SPMD mesh) | `python` | Hybrid backbone from your `HybridMambaMoEConfig` JSON |

---

## Shared Architecture & Objectives (GPU trainers)

* **Backbone**: 48 residual Mamba layers (`d_model=1024`, `d_state=16`, `d_conv=4`, `expand=2`).
* **Head**: The last non-padding hidden state goes through `Linear(1024, 1)`.
* **Total Parameters**: Exactly **352,798,721** trainable parameters. Production configurations outside 300M–400M raise.
* **Loss Function**: Bradley–Terry pairwise logistic preference loss:
  $$\mathcal{L} = -\log \sigma(r_{\text{chosen}} - r_{\text{rejected}})$$
* **Checkpoints**: Sharded PyTorch Distributed Checkpoint (DCP) directories with a completion marker, family `pure_mamba_reward_dcp_v1`, written as `<output_dir>/checkpoint-%08d/`. These are consumed by `evaluate_reward_model.py` and by PPO. Exact resume requires the same world size and precision.

---

## 1. Single-GPU Trainer (`train_reward_model_single_gpu.py`)

A trainer for one GPU that needs neither `torchrun` nor a process group.

### Features
* Gradient accumulation, gradient clipping, and a cosine schedule with linear warmup.
* Activation checkpointing per Mamba layer.
* `system.precision`: `auto`, `bf16`, or `fp32` (FP16 is rejected).
* Single-process DCP checkpoints with an atomic completion marker.

### Usage

```bash
# Inspect parameter count on the meta device
python post-training/RLHF/Reward-model/train/train_reward_model_single_gpu.py --count-only

# Dump the default configuration template
python post-training/RLHF/Reward-model/train/train_reward_model_single_gpu.py --write-config reward_config.json

# Launch training (cloud GPU)
python post-training/RLHF/Reward-model/train/train_reward_model_single_gpu.py \
    --config reward_config.json --device cuda:0 --output-dir runs/reward_model

# Resume from a saved checkpoint
python post-training/RLHF/Reward-model/train/train_reward_model_single_gpu.py \
    --resume runs/reward_model/checkpoint-00000500
```

---

## 2. Multi-GPU FSDP2 Trainer (`train_reward_model.py`)

Trainer built on PyTorch FSDP2 (`torch.distributed.fsdp.fully_shard`).

### Features
* Bottom-up `fully_shard` over complete residual layers, then the root; the optimizer is built after sharding.
* Non-reentrant activation checkpoint wrappers own whole layers.
* Streaming preference shards from a rank-0 producer with `DistributedSampler` per shard.
* Periodic diagnostic evaluation on fixed UltraFeedback / HH-RLHF subsets; full evaluation at completion.
* DCP save/load with a completion marker; checkpoint directories are never overwritten.

### Usage

```bash
# Launch on 4 GPUs on a single node
torchrun --standalone --nproc_per_node=4 post-training/RLHF/Reward-model/train/train_reward_model.py \
    --config reward_config.json

# Resume (the saved config supplies the original scheduler and data settings)
torchrun --standalone --nproc_per_node=4 post-training/RLHF/Reward-model/train/train_reward_model.py \
    --resume runs/reward_model/checkpoint-00000500

# Cloud preflight before a long run (two updates, then resume)
torchrun --standalone --nproc_per_node=2 post-training/RLHF/Reward-model/train/train_reward_model.py \
    --config reward_config.json --preflight --stop-after 1 --output-dir runs/rm_preflight
```

---

## 3. TPU Trainer (`reward_model_tpu_train.py`)

For a single Kaggle TPU v5e-8 host running PyTorch/XLA. It scores with a **hybrid** Mamba–Attention–MoE backbone (`use_dual_memory=false`, `use_auxiliary_losses=false`) built from your `HybridMambaMoEConfig` JSON, not the pure-Mamba model.

### Features
* SPMD sharding over an eight-way `fsdp` mesh with the PJRT runtime.
* Fixed-shape TPU MoE adapter that preserves dropless top-k routing.
* XLA `ParallelLoader` prefetching; BF16 autocast with FP32 weights and optimizer state.
* Atomic `latest.pth` checkpoints. These are separate from the GPU DCP checkpoint family and cannot be loaded by `evaluate_reward_model.py`.

### Usage

```bash
python post-training/RLHF/Reward-model/train/reward_model_tpu_train.py \
    --model-config reward_tpu_model.json \
    --dataset-factory my_data:make_train_dataset --dataset-kwargs '{}' \
    --output-dir /kaggle/working/reward_tpu

# Resume with the same model, Dataset, and training settings
python post-training/RLHF/Reward-model/train/reward_model_tpu_train.py \
    --model-config reward_tpu_model.json \
    --dataset-factory my_data:make_train_dataset --dataset-kwargs '{}' \
    --output-dir /kaggle/working/reward_tpu --resume /kaggle/working/reward_tpu/latest.pth
```

---

## Optimizer & Schedule Defaults (GPU trainers)

* **AdamW**: LR `1e-5`, betas `(0.9, 0.95)`, epsilon `1e-8`.
* **Weight decay** `0.01` on 2D weight matrices. Biases, normalization gains, and the Mamba `A_log` / `D` parameters use no decay (repository `_no_weight_decay` rules).
* **Gradient clipping**: max norm `1.0`.
* **Scheduler**: repository cosine schedule with 3% linear warmup and a 10% minimum LR.
* **Batching**: per-rank batch size 1, accumulation 16, one epoch capped at 10,000 updates.

---

## Downstream Consumption

GPU checkpoints are written under:
```text
<output_dir>/checkpoint-00000500/
├── .metadata
├── __0_0.distcp
└── ...
```

They are consumed by:
* Multi-dataset evaluation: [`evaluate_reward_model.py`](../evaluate_reward_model.py)
* PPO policy optimization: [`train_rlhf.py`](../../train_rlhf.py) (set `rlhf.reward_model_checkpoint`)
