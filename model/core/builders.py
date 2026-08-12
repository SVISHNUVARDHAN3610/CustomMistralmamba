"""Model builder utilities."""

import copy

from torch import nn

from model.core.config import HybridMambaMoEConfig
from model.hybrid.model import HybridForCausalLM


def count_trainable_params(
    module: nn.Module, exclude_training_aux: bool = False
) -> int:
    total = 0
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if exclude_training_aux and (
            "recon_decoder" in name or "assoc_key" in name or "assoc_val" in name
        ):
            continue
        total += param.numel()
    return total


def build_test3_null_baseline_config(
    hybrid_config: HybridMambaMoEConfig,
    tolerance: float = 0.02,
) -> HybridMambaMoEConfig:
    """
    Test 3 null hypothesis: larger Mamba state, no explicit memory banks.
    Binary-searches mamba_state_size (then mamba_expand) to match param count.
    Raises if a match within ``tolerance`` cannot be found.
    """
    target = count_trainable_params(
        HybridForCausalLM(hybrid_config), exclude_training_aux=True
    )
    null_config = copy.deepcopy(hybrid_config)
    null_config.use_dual_memory = False

    def _count(cfg: HybridMambaMoEConfig) -> int:
        return count_trainable_params(HybridForCausalLM(cfg))

    best_state = null_config.mamba_state_size
    lo, hi = null_config.mamba_state_size, null_config.mamba_state_size * 128
    while lo <= hi:
        mid = (lo + hi) // 2
        null_config.mamba_state_size = mid
        if _count(null_config) <= target:
            best_state = mid
            lo = mid + 1
        else:
            hi = mid - 1

    null_config.mamba_state_size = best_state
    if _count(null_config) < target * (1.0 - tolerance):
        best_expand = null_config.mamba_expand
        for expand in range(null_config.mamba_expand, 17):
            null_config.mamba_expand = expand
            if _count(null_config) > target * (1.0 + tolerance):
                break
            best_expand = expand
            if _count(null_config) >= target * (1.0 - tolerance):
                break
        null_config.mamba_expand = best_expand

    matched = _count(null_config)
    ratio = matched / max(target, 1)
    if ratio < (1.0 - tolerance) or ratio > (1.0 + tolerance):
        raise ValueError(
            f"Could not match null baseline params within {tolerance:.0%}: "
            f"target={target}, null={matched}, ratio={ratio:.4f}. "
            f"Try adjusting mamba_expand / hidden_size manually."
        )

    return null_config
