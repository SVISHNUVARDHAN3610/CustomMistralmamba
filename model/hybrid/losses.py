"""Hybrid auxiliary loss functions and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.core.dtype import _is_low_precision, _promote_fp32

if TYPE_CHECKING:
    from model.hybrid.memory import CompressiveMemoryBank


@dataclass
class HybridLayerAuxLosses:
    """Per-layer raw (unweighted) auxiliary loss scalars."""

    recon: Tensor
    assoc: Tensor
    gate: Tensor
    read: Tensor
    fusion: Tensor
    expert: Tensor
    ssm: Tensor
    slot: Tensor

    @staticmethod
    def zeros(device: torch.device, dtype: torch.dtype) -> HybridLayerAuxLosses:
        z = torch.tensor(0.0, device=device, dtype=dtype)
        return HybridLayerAuxLosses(
            recon=z, assoc=z, gate=z, read=z, fusion=z, expert=z, ssm=z, slot=z
        )


@dataclass
class HybridAuxiliaryLossBreakdown:
    """Model-level averaged auxiliary losses (unweighted)."""

    recon: Tensor | None = None
    assoc: Tensor | None = None
    gate: Tensor | None = None
    read: Tensor | None = None
    fusion: Tensor | None = None
    expert: Tensor | None = None
    ssm: Tensor | None = None
    slot: Tensor | None = None


def _aux_loss_schedule(
    step: int | None, max_steps: int | None, warmup_fraction: float
) -> float:
    if step is None or max_steps is None or max_steps <= 0:
        return 1.0
    warmup_steps = max(1, int(max_steps * warmup_fraction))
    return min(1.0, step / warmup_steps)


def _expert_loss_schedule(
    step: int | None, max_steps: int | None, warmup_fraction: float
) -> float:
    if step is None or max_steps is None or max_steps <= 0:
        return 1.0
    start = int(max_steps * warmup_fraction)
    return 1.0 if step >= start else 0.0


def write_gate_entropy_loss(
    gate: Tensor,
    eps: float = 1e-6,
    row_mask: Tensor | None = None,
) -> Tensor:
    """Negative binary entropy of write gates (encourage non-saturated gates).

    ``gate`` is typically ``[B, memory_size, H]``. When ``row_mask`` is provided
    (``[B]`` bool, True = include), rows with no valid write tokens are skipped
    so all-padding chunks contribute a finite zero instead of poisoning the mean.
    """
    # FP16: clamp before log to avoid NaNs when gates saturate or drift slightly.
    gate_f = _promote_fp32(gate).clamp(min=eps, max=1.0 - eps)
    ent = -(gate_f * torch.log(gate_f) + (1.0 - gate_f) * torch.log(1.0 - gate_f))
    per_row = ent.reshape(ent.size(0), -1).mean(dim=-1)
    if row_mask is not None:
        valid = row_mask.bool()
        if not valid.any():
            return torch.zeros((), device=gate.device, dtype=gate.dtype)
        return (-per_row[valid].mean()).to(dtype=gate.dtype)
    return (-per_row.mean()).to(dtype=gate.dtype)


def masked_token_mse(
    pred: Tensor,
    target: Tensor,
    valid_mask: Tensor | None,
) -> Tensor:
    """Mean squared error over valid token positions only (pad-safe)."""
    if valid_mask is None:
        return F.mse_loss(pred, target)
    valid = valid_mask.bool()
    if not valid.any():
        return torch.zeros((), device=target.device, dtype=target.dtype)
    # [B, L] reduction over hidden, then mean over valid tokens.
    per_tok = (pred - target).pow(2).mean(dim=-1)
    return per_tok[valid].mean()


def combine_read_utilization_loss(
    combine: nn.Linear, r_min: float, eps: float = 1e-6
) -> Tensor:
    weight = combine.weight
    hidden = weight.size(0)
    w_own = weight[:, :hidden]
    w_mem = weight[:, hidden:]
    r = w_mem.norm() / (w_own.norm() + w_mem.norm() + eps)
    return torch.relu(r_min - r) ** 2


def fusion_balance_loss(fusion_gate: Tensor) -> Tensor:
    g_bar = fusion_gate.mean(dim=(0, 1))
    hidden = g_bar.size(0)
    return ((g_bar - 0.5) ** 2).sum() / hidden


def memory_slot_diversity_loss(
    attn_mem: Tensor,
    state_mem: Tensor,
    margin: float,
    cross_alpha: float,
    eps: float = 1e-6,
) -> Tensor:
    out_dtype = attn_mem.dtype
    attn_f = _promote_fp32(attn_mem)
    state_f = _promote_fp32(state_mem)

    def _intra(mem: Tensor) -> Tensor:
        mem_norm = mem / mem.norm(dim=-1, keepdim=True).clamp(min=eps)
        sim = torch.matmul(mem_norm, mem_norm.transpose(-1, -2))
        m = sim.size(-1)
        mask = ~torch.eye(m, device=sim.device, dtype=torch.bool)
        excess = torch.relu(sim - margin) ** 2
        return excess.masked_select(mask.unsqueeze(0)).mean()

    intra = _intra(attn_f) + _intra(state_f)
    a_norm = attn_f / attn_f.norm(dim=-1, keepdim=True).clamp(min=eps)
    s_norm = state_f / state_f.norm(dim=-1, keepdim=True).clamp(min=eps)
    cross = (a_norm * s_norm).sum(dim=-1).abs().mean()
    return (intra + cross_alpha * cross).to(dtype=out_dtype)


def ssm_state_norm_loss(ssm_state: Tensor, gamma: Tensor) -> Tensor:
    s_bar = ssm_state.float().pow(2).mean()
    return torch.relu(s_bar - gamma)


class MemoryReconstructionDecoder(nn.Module):
    """Training-only decoder: reconstruct x from compressed summary s."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by recon_decoder_heads.")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _shape_heads(self, t: Tensor, seq_len: int) -> Tensor:
        b = t.size(0)
        return t.reshape(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor, summary: Tensor) -> Tensor:
        q_len = x.size(1)
        k_len = summary.size(1)
        q = self._shape_heads(self.q_proj(x), q_len)
        k = self._shape_heads(self.k_proj(summary), k_len)
        v = self._shape_heads(self.v_proj(summary), k_len)
        if _is_low_precision(x.dtype):
            q = _promote_fp32(q)
            k = _promote_fp32(k)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(x.size(0), q_len, -1)
        return self.out_proj(out)


def memory_reconstruction_loss(
    x: Tensor, summary: Tensor, decoder: nn.Module
) -> Tensor:
    recon = decoder(x, summary)
    return F.mse_loss(recon, x)


def associative_retrieval_loss(
    bank: CompressiveMemoryBank,
    x: Tensor,
    new_memory: Tensor,
    per_token_residual: Tensor,
    sample_count: int,
    attention_mask: Tensor | None,
) -> Tensor:
    batch_size, seq_len, hidden = x.shape
    device = x.device
    dtype = x.dtype
    if attention_mask is not None:
        valid = attention_mask.bool()
    else:
        valid = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)

    if not valid.any():
        return torch.tensor(0.0, device=device, dtype=dtype)

    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    masked_pos = torch.where(valid, positions, seq_len)
    sorted_pos, _ = masked_pos.sort(dim=1)
    valid_counts = valid.sum(dim=1)
    max_valid = int(valid_counts.max().item())
    n_sel = min(sample_count, max_valid)
    if n_sel == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    candidates = sorted_pos[:, :max_valid]
    col_idx = torch.arange(max_valid, device=device).unsqueeze(0)
    row_valid_cols = col_idx < valid_counts.unsqueeze(1)
    rand = torch.rand(batch_size, max_valid, device=device)
    rand = rand.masked_fill(~row_valid_cols, -1.0)
    _, perm = rand.sort(dim=1, descending=True)
    sel_local = perm[:, :n_sel]
    indices = candidates.gather(1, sel_local)
    sample_mask = torch.arange(n_sel, device=device).unsqueeze(
        0
    ) < valid_counts.unsqueeze(1).clamp(max=n_sel)
    # Rows with fewer than n_sel valid tokens still fill sel_local from padded
    # candidate slots (sentinel position == seq_len). Clamp for gather safety;
    # zero those slots in sample_mask so they do not affect the loss.
    valid_index = indices < seq_len
    indices = indices.clamp(max=seq_len - 1)
    sample_mask = sample_mask & valid_index

    x_sel = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, hidden))
    keys = bank.assoc_key(x_sel)
    values = bank.assoc_val(x_sel)
    retrieved = bank.read_query(keys, new_memory)
    err = (retrieved - values).pow(2).sum(dim=-1)
    surprise = per_token_residual.gather(1, indices).detach()
    if n_sel > 1:
        sigma = surprise.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
        surprise = surprise.clamp(max=3.0 * sigma)
    surprise = surprise.clamp(min=0.0)
    weighted = surprise * err
    per_row = (weighted * sample_mask).sum(dim=1) / sample_mask.sum(dim=1).clamp(min=1)
    row_mask = valid_counts > 0
    if not row_mask.any():
        return torch.tensor(0.0, device=device, dtype=dtype)
    return per_row[row_mask].mean()
