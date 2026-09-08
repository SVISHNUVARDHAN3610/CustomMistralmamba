"""Pure repository Mamba blocks with a single scalar head, without an LM head."""

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from model.hybrid.mamba import MambaBlock
from model.hybrid.model import _checkpoint_autocast_contexts
from model.layers.norm import RMSNorm

from .config import ModelConfig


class RewardLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mamba = MambaBlock(cfg.d_model, cfg.d_state, cfg.d_conv, cfg.expand)

    def forward(self, hidden: Tensor, mask: Tensor, padded: bool) -> Tensor:
        update, _, _ = self.mamba(
            self.norm(hidden),
            attention_mask=mask,
            batch_has_padding=padded,
            mamba_internal_checkpoint=False,
            layer_checkpointing_active=True,
        )
        return hidden + update


def last_valid_hidden(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Gather the final nonpadding state, even for a non-prefix mask."""
    if hidden.shape[:2] != attention_mask.shape:
        raise ValueError("Hidden states and attention_mask shapes differ")
    mask = attention_mask.bool()
    if not mask.any(dim=1).all():
        raise ValueError("Cannot score an entirely padded sequence")
    positions = torch.arange(mask.size(1), device=hidden.device)
    indices = positions.expand_as(mask).masked_fill(~mask, -1).amax(dim=1)
    return hidden[torch.arange(hidden.size(0), device=hidden.device), indices]


class RewardModel(nn.Module):
    def __init__(self, config: ModelConfig, activation_checkpointing: bool = False):
        super().__init__()
        if any(v <= 0 for v in vars(config).values()):
            raise ValueError("Reward model dimensions must be positive")
        self.config = config
        self.activation_checkpointing = activation_checkpointing
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            RewardLayer(config) for _ in range(config.num_layers)
        )
        self.norm = RMSNorm(config.d_model)
        self.reward_head = nn.Linear(config.d_model, 1)
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError("Expected matching [B,T] input_ids and attention_mask")
        if input_ids.size(1) > self.config.max_length:
            raise ValueError("Sequence exceeds model.max_length")
        if not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("attention_mask must be binary")
        mask = attention_mask.bool()
        if not mask.any(dim=1).all() or (mask[:, 1:] & ~mask[:, :-1]).any():
            raise ValueError(
                "Mamba reward inputs require nonempty, right-padded sequences"
            )
        padded = not bool(mask[:, -1].all())
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            if self.training and self.activation_checkpointing:
                hidden = checkpoint(
                    layer,
                    hidden,
                    mask,
                    padded,
                    use_reentrant=False,
                    context_fn=partial(
                        _checkpoint_autocast_contexts, hidden.device.type
                    ),
                )
            else:
                hidden = layer(hidden, mask, padded)
        return self.reward_head(last_valid_hidden(self.norm(hidden), mask))


def parameter_counts(model: RewardModel, enforce_size: bool = True) -> dict[str, int]:
    """Count actual tensors (also works on meta without allocating model storage)."""
    total = sum(p.numel() for p in model.parameters())
    counts = {
        "embedding": model.embed_tokens.weight.numel(),
        "normalization": sum(
            p.numel()
            for m in model.modules()
            if isinstance(m, RMSNorm)
            for p in m.parameters()
        ),
        "reward_head": sum(p.numel() for p in model.reward_head.parameters()),
        "total": total,
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    counts["mamba"] = total - sum(
        counts[k] for k in ("embedding", "normalization", "reward_head")
    )
    if enforce_size and not 300_000_000 <= total <= 400_000_000:
        raise ValueError(f"Reward model has {total:,} parameters; require 300M–400M")
    return counts


def pairwise_loss(chosen_rewards: Tensor, rejected_rewards: Tensor) -> Tensor:
    if chosen_rewards.shape != rejected_rewards.shape:
        raise ValueError("Chosen and rejected reward shapes must match")
    return -F.logsigmoid(chosen_rewards.float() - rejected_rewards.float()).mean()
