"""Dtype helpers for numerically sensitive ops under AMP."""

import torch
from torch import Tensor


def _is_low_precision(dtype: torch.dtype) -> bool:
    return dtype in (torch.float16, torch.bfloat16)


def _promote_fp32(x: Tensor) -> Tensor:
    """Promote activations to fp32 for numerically sensitive ops under AMP."""
    return x.float() if _is_low_precision(x.dtype) else x


def _restore_dtype(x: Tensor, dtype: torch.dtype) -> Tensor:
    return x.to(dtype) if x.dtype != dtype else x
