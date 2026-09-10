# Reward Model Training Runners

This directory contains the training implementations for the pure-Mamba Reward Model across different hardware targets and distributed execution strategies.

## Available Runners

| Script | Hardware Target | Distributed Engine | Entry Method | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| [`train_reward_model_single_gpu.py`](file:///D:/Working_Repo/CustomMistralmamba/post-training/RLHF/Reward-model/train/train_reward_model_single_gpu.py) | Single GPU / Local workstation | Single-process (no communication backend needed) | `python` | Local testing, single-GPU fine-tuning, debugging |
| [`train_reward_model.py`](file:///D:/Working_Repo/CustomMistralmamba/post-training/RLHF/Reward-model/train/train_reward_model.py) | Multi-GPU cluster (NVIDIA) | PyTorch FSDP2 (`fully_shard`) + DeviceMesh | `torchrun` | Production large-scale training across multi-GPU nodes |
| [`reward_model_tpu_train.py`](file:///D:/Working_Repo/CustomMistralmamba/post-training/RLHF/Reward-model/train/reward_model_tpu_train.py) | Google Cloud / Kaggle TPU v5e-8 | `torch_xla` (PJRT runtime / SPMD mesh) | `python` | Scaled training on Google Cloud / Kaggle TPU pods |

---

## Shared Architecture & Objectives

All runners train the exact same pure-Mamba reward model architecture:

* **Backbone**: 48 Mamba layers (`d_model=1024`, `d_state=16`, `d_conv=4`, `expand=2`).
* **Head**: Sequence pooling via the last non-padding token followed by a single linear projection `Linear(1024, 1)`.
* **Total Parameters**: Exactly **352,798,721** trainable parameters.
* **Loss Function**: Bradley–Terry pairwise logistic preference loss:
  $$\mathcal{L} = -\log \sigma(r_{\text{chosen}} - r_{\text{rejected}})$$
* **Checkpoint Standard**: All scripts save and resume checkpoints using PyTorch Distributed Checkpoint (DCP) under the family tag `pure_mamba_reward_dcp_v1`. Checkpoints trained on single-GPU, multi-GPU FSDP2, or TPU can be evaluated or loaded into downstream PPO interchangeably.

---

## 1. Single-GPU Trainer (`train_reward_model_single_gpu.py`)

A standalone trainer designed for single-GPU setups (or local smoke verification on CPU) that eliminates the need for `torchrun` or multi-process communication groups.

### Features
* Native PyTorch execution via standard `python`.
* Supports gradient accumulation, gradient clipping, cosine learning rate scheduling with linear warmup.
* Activation checkpointing per Mamba layer.
* Automatic mixed precision (`bf16` or `fp16`).
* Single-process Distributed Checkpoint (`torch.distributed.checkpoint`) compatible with multi-GPU runs.
* Atomic checkpoint writing using temporary swap directories to prevent corrupted state on interruption.

### Usage

**1. Inspect parameter count on meta device:**
```bash
python post-training/RLHF/Reward-model/train/train_reward_model_single_gpu.py --count-only
```

**2. Dump default configuration template:**
```bash
python post-training/RLHF/Reward-model/train/train_reward_model_single_gpu.py --write-config config_single_gpu.json
```

**3. Launch training:**
```bash
python post-training/RLHF/Reward-model/train/train_reward_model_single_gpu.py \
    --config config_single_gpu.json \
    --device cuda:0 \
    --precision bf16
```

**4. Resume from a saved checkpoint:**
```bash
python post-training/RLHF/Reward-model/train/train_reward_model_single_gpu.py \
    --config config_single_gpu.json \
    --resume-from checkpoints/reward_model/step_1000
```

---

## 2. Multi-GPU FSDP2 Trainer (`train_reward_model.py`)

Production-grade trainer leveraging PyTorch Fully Sharded Data Parallel 2 (FSDP2) via `torch.distributed._composable.fsdp.fully_shard`.

### Features
* Shards model weights, gradients, and optimizer states across ranks using a 1D `DeviceMesh`.
* Layer-by-layer activation checkpointing using `torch.distributed.algorithms._checkpoint.checkpoint_wrapper`.
* Streaming multi-shard dataset consumption with per-rank deterministic offset skipping.
* Periodic evaluation on validation splits (`UltraFeedback` test preferences and `Anthropic HH-RLHF` test split).
* Distributed Checkpoint (DCP) saving with rank 0 orchestration and atomic rename.

### Usage

**Launch on 4 GPUs on a single node:**
```bash
torchrun --standalone --nproc_per_node=4 post-training/RLHF/Reward-model/train/train_reward_model.py \
    --config post-training/RLHF/Reward-model/train/fsdp2_config.json
```

**Inspect sharded parameter allocation:**
```bash
torchrun --standalone --nproc_per_node=4 post-training/RLHF/Reward-model/train/train_reward_model.py \
    --count-only
```

---

## 3. TPU Trainer (`reward_model_tpu_train.py`)

Optimized for Google Cloud TPU v5e-8 and Kaggle TPU environments running PyTorch/XLA.

### Features
* Native SPMD (Single Program, Multiple Data) sharding via `torch_xla.distributed.spmd`.
* PJRT runtime initialization.
* XLA-compatible data loader prefetching and mark-step synchronization points.
* Checkpoint preservation compatible with DCP loading.

### Usage

```bash
python post-training/RLHF/Reward-model/train/reward_model_tpu_train.py \
    --config tpu_config.json
```

---

## Optimizer & Parameter Group Configuration

All training scripts maintain identical optimizer grouping contracts:

* **Decayed Group** (`weight_decay = 0.1`):
  * All 2D weight matrices (e.g. `in_proj.weight`, `out_proj.weight`, `x_proj.weight`, `score_head.weight`).
* **Non-Decayed Group** (`weight_decay = 0.0`):
  * 1D bias vectors (`*.bias`).
  * Normalization gains (`*.norm.weight`, `RMSNorm.weight`).
  * State space parameter vectors: `A_log` and `D`.
* **Gradient Clipping**: Maximum gradient norm of `1.0`.
* **Scheduler**: Linear warmup over the initial warmup steps followed by cosine annealing down to `min_lr` ($1 \times 10^{-6}$).

---

## Downstream Consumption

Checkpoints saved by any runner in this directory are saved under:
```text
<checkpoint_dir>/step_<step_number>/
├── .metadata
├── __0_0.distcp
└── ...
```

These checkpoints can be directly consumed by:
* Multi-dataset evaluation: [`evaluate_reward_model.py`](file:///D:/Working_Repo/CustomMistralmamba/post-training/RLHF/Reward-model/evaluate_reward_model.py)
* PPO Actor-Critic alignment: [`train_rlhf.py`](file:///D:/Working_Repo/CustomMistralmamba/post-training/RLHF/train_rlhf.py)
