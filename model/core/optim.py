"""Pure parameter-grouping helpers for the Muon + AdamW hybrid optimizer.

Kept dependency-free (torch.nn only) so unit tests can exercise the split
without importing ``train.py`` (which pulls in datasets/transformers).
"""

from __future__ import annotations

from torch import nn

# Token / slot embeddings and classifier head stay on AdamW (Moonshot + Keller).
# Router expert-selection matrices are 2D hidden weights and stay on Muon
# (Moonlight SVD analysis includes routers under Muon).
_ADAMW_NAME_SUBSTRINGS = (
    "embed_tokens",
    "lm_head",
    "init_memory",
    "summary_query",
)


def _is_adamw_parameter(name: str, param: nn.Parameter) -> bool:
    """True when Muon must not own this parameter.

    Rules from arXiv:2502.16982 + torch.optim.Muon docs:
      - Explicit metadata wins: params flagged ``_no_weight_decay`` (Mamba
        A_log / D) must never sit in a decayed Muon group, so they route to
        AdamW (in its no-decay subgroup).
      - Muon only accepts 2D matrices (hidden-layer weights).
      - Embeddings, LM head, RMSNorm / bias / other non-matrix params -> AdamW.
      - MoE router matrices are 2D and should use Muon (not AdamW).
      - Mamba Conv1d weights are 3D -> AdamW.
      - Dual-memory slot banks (init_memory / summary_query) are embedding-like -> AdamW.
    """
    if getattr(param, "_no_weight_decay", False):
        return True
    if param.ndim != 2:
        return True
    return any(key in name for key in _ADAMW_NAME_SUBSTRINGS)


def _is_adamw_no_decay(param: nn.Parameter) -> bool:
    """True when AdamW must apply zero weight decay to this parameter.

    Standard LLM recipe: biases and norm gains (ndim < 2) plus any param
    flagged ``_no_weight_decay`` (decaying Mamba's A_log pulls exp(A_log)
    toward identity and destroys the learned state-decay timescales).
    Embeddings / lm_head stay decayed (Moonshot + Keller recipe).
    """
    return param.ndim != 2 or getattr(param, "_no_weight_decay", False)


def split_muon_adam_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], dict[str, list[str]]]:
    """Split parameters into AdamW vs Muon groups with name inventories."""
    adam_params: list[nn.Parameter] = []
    muon_params: list[nn.Parameter] = []
    inventory: dict[str, list[str]] = {"adamw": [], "muon": []}
    seen: set[int] = set()

    for name, param in model.named_parameters():
        # Tied embeddings / lm_head share storage; optimize once.
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)

        if _is_adamw_parameter(name, param):
            adam_params.append(param)
            inventory["adamw"].append(f"{name}{tuple(param.shape)}")
        else:
            muon_params.append(param)
            inventory["muon"].append(f"{name}{tuple(param.shape)}")

    return adam_params, muon_params, inventory
