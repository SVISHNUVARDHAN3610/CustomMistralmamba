"""RMSNorm layer."""

import torch
from torch import nn

from model.core.fsdp import local_dtensor


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        norm_x = torch.rsqrt(variance + self.eps)

        return (x_fp32 * norm_x).to(input_dtype) * local_dtensor(self.weight)
