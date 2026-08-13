"""Rotary positional embeddings."""

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """
    Cache is allocated once up to `max_position_embeddings` and never
    re-registered at runtime. The previous implementation re-registered
    `cos_cached`/`sin_cached` via `register_buffer` whenever `seq_len`
    exceeded the cache, which is not guaranteed to free the old buffer and
    can bloat/corrupt FSDP state_dicts over time. If a sequence longer than
    `max_position_embeddings` is requested we fail loudly instead of quietly
    growing a mutable buffer underneath FSDP.
    """

    def __init__(
        self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Registered once, fixed size for the lifetime of the module.
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: int, position_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Requested seq_len={seq_len} exceeds RotaryEmbedding.max_position_embeddings="
                f"{self.max_position_embeddings}. Increase MixtralConfig.max_position_embeddings."
            )

        if position_ids is not None:
            cos = self.cos_cached[position_ids].to(dtype=x.dtype, device=x.device)
            sin = self.sin_cached[position_ids].to(dtype=x.dtype, device=x.device)
            return cos, sin

        return (
            self.cos_cached[:seq_len, :].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len, :].to(dtype=x.dtype, device=x.device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)  # [B, S, D] -> [B, 1, S, D]
        sin = sin.unsqueeze(1)
    elif cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)  # [S, D] -> [1, 1, S, D]
        sin = sin.unsqueeze(0).unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
