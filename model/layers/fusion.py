"""Token-wise gated fusion."""

import torch
from torch import Tensor, nn


class TokenGatedFusion(nn.Module):
    """O(L) per-token fusion of attention-branch and Mamba-branch outputs."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, a: Tensor, m: Tensor) -> tuple[Tensor, Tensor]:
        g = torch.sigmoid(self.gate(torch.cat([a, m], dim=-1)))
        fused = g * a + (1.0 - g) * m
        return fused, g
