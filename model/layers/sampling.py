"""Logit filtering helpers shared by both model families' ``generate()``."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def top_k_filter(logits: Tensor, top_k: int) -> Tensor:
    """Keep only the ``top_k`` highest-probability logits; rest become -inf."""
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(
        logits < min_values, torch.full_like(logits, float("-inf")), logits
    )


def top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    """Nucleus filtering: keep the smallest prefix with cumulative prob > top_p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    mask = torch.zeros_like(sorted_mask).scatter(1, sorted_indices, sorted_mask)
    return logits.masked_fill(mask, float("-inf"))
