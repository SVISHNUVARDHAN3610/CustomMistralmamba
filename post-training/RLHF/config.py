"""Dataclass/JSON configuration, matching model.core.config conventions."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 1024
    num_layers: int = 48
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    max_length: int = 2048


@dataclass
class DataConfig:
    tokenizer_name: str = "UIC-AI-lab/llama2-tokenizer"
    tokenizer_revision: str = "main"
    ultrafeedback_weight: float = 35
    hh_rlhf_weight: float = 75
    max_length: int = 2048
    min_response_tokens: int = 128
    response_truncation_strategy: str = "head_tail"
    num_workers: int = 0
    buffer_size: int = 3  # maximum unconsumed disk shards
    pairs_per_shard: int = 64
    prefetch_factor: int = 2
    shard_timeout: int = 600
    ultrafeedback_revision: str = "main"
    hh_rlhf_revision: str = "main"
    helpsteer2_revision: str = "main"

    def probabilities(self) -> list[float]:
        weights = [self.ultrafeedback_weight, self.hh_rlhf_weight]
        if any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError(
                "Both relative dataset weights must be finite and positive"
            )
        return [w / sum(weights) for w in weights]


@dataclass
class TrainingConfig:
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    num_epochs: int = 1
    max_steps: int = 10000
    gradient_clip_norm: float = 1.0


@dataclass
class SystemConfig:
    precision: str = "auto"
    activation_checkpointing: bool = True
    output_dir: str = "runs/reward_model"
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    resume_from_checkpoint: str | None = None
    initial_checkpoint: str | None = None
    # Legacy eval_interval remains the fallback for existing configurations.
    diagnostic_eval_enabled: bool = True
    diagnostic_eval_pairs: int = 512
    diagnostic_eval_interval: int | None = None
    full_eval_enabled: bool = True
    full_eval_at_checkpoints: bool = False


@dataclass
class RewardConfig:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "RewardConfig":
        value = dict(value)
        for key, kind in (
            ("model", ModelConfig),
            ("data", DataConfig),
            ("training", TrainingConfig),
            ("system", SystemConfig),
        ):
            options = dict(value.get(key, {}))
            if key == "system":
                # Old checkpoints remain readable; pair counts replace this cap.
                options.pop("eval_batches", None)
            value[key] = kind(**options)
        result = cls(**value)
        result.validate()
        return result

    @classmethod
    def from_pretrained(cls, path: str) -> "RewardConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_pretrained(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def validate(self) -> None:
        self.data.probabilities()
        positive = [
            *asdict(self.model).values(),
            self.data.buffer_size,
            self.data.pairs_per_shard,
            self.data.prefetch_factor,
            self.data.shard_timeout,
            self.data.min_response_tokens,
            self.training.batch_size,
            self.training.gradient_accumulation_steps,
            self.training.max_steps,
            self.training.num_epochs,
            self.training.learning_rate,
            self.training.gradient_clip_norm,
            self.system.log_interval,
            self.system.eval_interval,
            self.system.save_interval,
            self.system.diagnostic_eval_pairs,
            self.system.diagnostic_eval_interval or self.system.eval_interval,
        ]
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError(
                "Model dimensions, lengths, intervals and training sizes must be positive"
            )
        if self.data.num_workers < 0 or self.training.weight_decay < 0:
            raise ValueError("num_workers and weight_decay must be nonnegative")
        if self.system.diagnostic_eval_pairs < 2:
            raise ValueError(
                "diagnostic_eval_pairs must allocate at least one pair per source"
            )
        if (
            self.system.diagnostic_eval_interval is not None
            and self.system.diagnostic_eval_interval <= 0
        ):
            raise ValueError("diagnostic_eval_interval must be positive")
        if self.data.response_truncation_strategy not in ("head", "head_tail"):
            raise ValueError("response_truncation_strategy must be head or head_tail")
        if not 0 <= self.training.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 4 <= self.data.max_length <= self.model.max_length:
            raise ValueError("Require 4 <= data.max_length <= model.max_length")
        if self.system.precision not in ("auto", "bf16", "fp16", "fp32"):
            raise ValueError("precision must be auto/bf16/fp16/fp32")


def smoke_config(output_dir: str) -> RewardConfig:
    """Explicitly tiny, CPU-only development profile; never the production size."""
    cfg = RewardConfig()
    cfg.model = ModelConfig(256, 16, 2, 4, 3, 2, 64)
    cfg.data.max_length = 64
    cfg.data.min_response_tokens = 8
    cfg.data.pairs_per_shard = 4
    cfg.training.batch_size = 2
    cfg.training.gradient_accumulation_steps = 2
    cfg.training.max_steps = 2
    cfg.system.output_dir = output_dir
    cfg.system.precision = "fp32"
    cfg.system.log_interval = cfg.system.eval_interval = cfg.system.save_interval = 1
    cfg.system.diagnostic_eval_pairs = 4
    return cfg
