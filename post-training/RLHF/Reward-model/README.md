# Reward Model Package

This directory contains the reward modelling infrastructure for reinforcement learning from human feedback (RLHF) built upon the pure-Mamba architecture.

## Overview

The reward model scores conversational preference pairs by predicting an unconstrained scalar reward $r(x, y)$ for prompt $x$ and response $y$. The objective uses the Bradley–Terry logistic preference formulation:

$$\mathcal{L} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(r(x, y_w) - r(x, y_l)\right)\right]$$

where $y_w$ is the chosen response and $y_l$ is the rejected response.

## Directory Structure

```text
post-training/RLHF/Reward-model/
├── README.md                      # This documentation
├── __init__.py                    # Package exports and path setup
├── config.py                      # Dataclass configurations (ModelConfig, DataConfig, RewardConfig)
├── reward_model.py                # Pure-Mamba RewardModel architecture and pairwise_loss
├── rlhf_dataset.py                # PreferenceShardProducer, streaming, and tokenization adapters
├── evaluation_metrics.py          # RewardStatistics: exact moments and disk-backed percentiles
├── evaluate_reward_model.py       # Standalone multi-split evaluation script
├── benchmark_reward_shards.py     # Shard throughput and pipeline latency benchmark
└── train/                         # Training execution entry points
    ├── README.md                  # Training guides and runner comparisons
    ├── __init__.py                # Training subpackage exports
    ├── train_reward_model_single_gpu.py  # Standalone single-GPU trainer (no torchrun)
    ├── train_reward_model.py      # Multi-GPU FSDP2 cloud trainer (torchrun)
    └── reward_model_tpu_train.py  # Single-host Kaggle TPU v5e-8 trainer (PJRT/SPMD)
```

## Architecture

* **Backbone**: 48 residual layers consisting of `RMSNorm` followed by `MambaBlock` (`d_model=1024`, `d_state=16`, `d_conv=4`, `expand=2`).
* **Head**: Final `RMSNorm` followed by a single linear projection `Linear(1024, 1)` gathering the final non-padding token. No LM vocabulary head is allocated.
* **Parameters**: 352,798,721 total trainable parameters (verified exactly on meta-device before training).
* **Activation Checkpointing**: Layer-level non-reentrant activation checkpointing preserving memory on long contexts up to 2,048 tokens.

## Dataset Adapters & Sources

* **UltraFeedback Binarized** (`HuggingFaceH4/ultrafeedback_binarized`): `train_prefs` split for training, `test_prefs` for validation/evaluation.
* **Anthropic HH-RLHF** (`Anthropic/hh-rlhf`): `train` split for training, `test` for validation/evaluation.
* **HelpSteer2** (`nvidia/HelpSteer2`): Direct preference strength annotations held out strictly for evaluation.
* **Sampling Mixture**: Configured 35:75 relative weights between UltraFeedback and HH-RLHF (normalized to ~31.8% and ~68.2%).

## Evaluation & Diagnostics

* **Metrics**: Pairwise accuracy ($\mathbb{I}[r_w > r_l]$), chosen reward mean/std, rejected reward mean/std, and reward margin mean/std/p50/p95/p99.
* **Exact vs Approximate Percentiles**: `RewardStatistics` uses float64 Chan/Welford moments and memory-mapped disk columns for exact percentiles, with an optional uniform reservoir sample for approximate evaluation on huge datasets.

## Checkpoint Compatibility

Checkpoints adhere to Distributed Checkpoint (DCP) family `pure_mamba_reward_dcp_v1`. Checkpoints produced by either the single-GPU trainer or the FSDP2 multi-GPU trainer are interchangeable and can be directly loaded by `evaluate_reward_model.py` and the downstream PPO policy trainer `train_rlhf.py`.
