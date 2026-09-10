# ruff: noqa: N999 -- user-requested RLHF and Reward-model directory names
"""Reward-model package containing architecture, dataset adapters, configs, and evaluation."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRAIN_DIR = HERE / "train"

if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

for directory in (ROOT / "post-training", HERE, TRAIN_DIR):
    dir_str = str(directory)
    if dir_str not in sys.path:
        sys.path.append(dir_str)

from .config import (
    DataConfig,
    ModelConfig,
    PPOConfig,
    RewardConfig,
    SystemConfig,
    TrainingConfig,
    smoke_config,
)
from .evaluation_metrics import RewardStatistics, isolated_evaluation_rng
from .reward_model import (
    RewardLayer,
    RewardModel,
    last_valid_hidden,
    pairwise_loss,
    parameter_counts,
)
from .rlhf_dataset import (
    PreferenceCollator,
    PreferenceShardDataset,
    PreferenceShardProducer,
    SmokeTokenizer,
    accounting_summary,
    smoke_stream,
)

__all__ = [
    "DataConfig",
    "ModelConfig",
    "PPOConfig",
    "PreferenceCollator",
    "PreferenceShardDataset",
    "PreferenceShardProducer",
    "RewardConfig",
    "RewardLayer",
    "RewardModel",
    "RewardStatistics",
    "SmokeTokenizer",
    "SystemConfig",
    "TrainingConfig",
    "accounting_summary",
    "isolated_evaluation_rng",
    "last_valid_hidden",
    "pairwise_loss",
    "parameter_counts",
    "smoke_config",
    "smoke_stream",
]
