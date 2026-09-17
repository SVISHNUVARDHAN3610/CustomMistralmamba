# ruff: noqa: N999 -- user-requested RLHF directory name
"""Pure Mamba reward modelling and hybrid-policy PPO post-training."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REWARD_MODEL_DIR = HERE / "Reward-model"
TRAIN_DIR = REWARD_MODEL_DIR / "train"

if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

for directory in (ROOT / "post-training", HERE, REWARD_MODEL_DIR, TRAIN_DIR):
    dir_str = str(directory)
    if dir_str not in sys.path:
        sys.path.append(dir_str)

import benchmark_reward_shards
import config
import evaluate_reward_model
import evaluation_metrics
import reward_model
import reward_model_tpu_train
import rlhf_dataset
import train_reward_model
import train_reward_model_single_gpu
from config import RewardConfig
from reward_model import RewardModel, pairwise_loss

# Register submodule aliases for backward compatibility with existing tests/scripts
sys.modules.setdefault("RLHF.config", config)
sys.modules.setdefault("RLHF.reward_model", reward_model)
sys.modules.setdefault("RLHF.rlhf_dataset", rlhf_dataset)
sys.modules.setdefault("RLHF.evaluation_metrics", evaluation_metrics)
sys.modules.setdefault("RLHF.evaluate_reward_model", evaluate_reward_model)
sys.modules.setdefault("RLHF.benchmark_reward_shards", benchmark_reward_shards)
sys.modules.setdefault("RLHF.train_reward_model", train_reward_model)
sys.modules.setdefault(
    "RLHF.train_reward_model_single_gpu", train_reward_model_single_gpu
)
sys.modules.setdefault("RLHF.reward_model_tpu_train", reward_model_tpu_train)

__all__ = ["RewardConfig", "RewardModel", "pairwise_loss"]
