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


_ADAMW_NAME_SUBSTRINGS = (
    "embed_tokens",
    "lm_head",
    "init_memory",
    "summary_query",
    "A_log",
    "dt_proj",
)


def _is_adamw_parameter(name: str, param: nn.Parameter) -> bool:
    """True when Muon must not own this parameter."""
    if getattr(param, "_no_weight_decay", False):
        return True
    if param.ndim != 2:
        return True
    return any(key in name for key in _ADAMW_NAME_SUBSTRINGS)


def split_muon_adam_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], dict[str, list[str]], dict[int, str]]:
    """Split parameters into AdamW vs Muon groups with name inventories."""
    adam_params: list[nn.Parameter] = []
    muon_params: list[nn.Parameter] = []
    inventory: dict[str, list[str]] = {"adamw": [], "muon": []}
    param_names: dict[int, str] = {}
    seen: set[int] = set()

    for name, param in model.named_parameters():
        param_id = id(param)
        param_names[param_id] = name
        if param_id in seen:
            continue
        seen.add(param_id)

        if _is_adamw_parameter(name, param):
            adam_params.append(param)
            inventory["adamw"].append(f"{name}{tuple(param.shape)}")
        else:
            muon_params.append(param)
            inventory["muon"].append(f"{name}{tuple(param.shape)}")

    return adam_params, muon_params, inventory, param_names


def build_adamw_param_groups(
    params: list[nn.Parameter],
    weight_decay: float,
    name_lookup: dict[int, str] | None = None,
) -> list[dict[str, object]]:
    """Split AdamW parameters into weight-decay and no-decay groups."""
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for param in params:
        param_name = name_lookup.get(id(param), "") if name_lookup else ""
        is_no_decay = (
            getattr(param, "_no_weight_decay", False)
            or param.ndim < 2
            or "norm" in param_name
            or "bias" in param_name
            or "embed_tokens" in param_name
            or "init_memory" in param_name
            or "summary_query" in param_name
            or "A_log" in param_name
            or "D" in param_name
        )
        if is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: list[dict[str, object]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups
