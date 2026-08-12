"""Sliding-window grouped-query attention."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.core.config import MixtralConfig
from model.layers.rope import RotaryEmbedding, apply_rotary_pos_emb


class SlidingWindowGQA(nn.Module):
    """
    Sliding-window grouped-query attention using SDPA only.

    The flex_attention experimental path was removed: it depended on a
    module-level global (`_CURRENT_PADDING_MASK`) that is unsafe with FSDP
    (concurrent forward passes across wrapped layers/ranks can race on it),
    it indexed the padding mask with block-local `kv_idx` instead of the
    global sequence index (semantically wrong), and it was unsupported on
    T4 (compute capability 7.5) in any case -- the previous code even forced
    `HAS_FLEX = False` unconditionally, so this path was already dead code.
    """

    def __init__(self, config: MixtralConfig) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.window_size = config.window_size
        self.head_dim = config.head_dim
        self.dropout = config.dropout

        assert self.num_heads % self.num_kv_heads == 0, (
            f"Configuration Error: num_heads ({self.num_heads}) must be perfectly divisible by num_kv_heads ({self.num_kv_heads})."
        )
        assert self.hidden_size == self.num_heads * self.head_dim, (
            f"Configuration Error: hidden_size ({self.hidden_size}) must precisely equal num_heads * head_dim."
        )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self._sliding_mask_cache: dict[tuple[int, int, str], Tensor] = {}
        self._sliding_mask_cache_max = 64

    def _repeat_kv(self, x: Tensor, n_rep: int) -> Tensor:
        batch_size, num_kv_heads, seq_len, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, None, :, :]
            .expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
            .reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
        active_batch_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        del active_batch_mask  # KV growth is fine; pad mask blocks finished keys.
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        query_states = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = self.rotary_emb(
            value_states, seq_len=seq_len, position_ids=position_ids
        )
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if key_states.size(2) > self.window_size:
            key_states = key_states[:, :, -self.window_size :, :]
            value_states = value_states[:, :, -self.window_size :, :]
            # Padding mask must match truncated KV length, or the `&` with
            # sliding_causal_mask raises a shape mismatch once seq_len >
            # window_size (latent during generate() with a mask present).
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask[:, -self.window_size :]
                elif attention_mask.dim() == 4:
                    attention_mask = attention_mask[:, :, :, -self.window_size :]

        # Cache only the sliding window (O(1) in L for decode), not the
        # full history — RoPE is already baked into cached K.
        present_key_value = (key_states, value_states) if use_cache else None

        kv_seq_len = key_states.size(2)

        num_queries_per_kv = self.num_heads // self.num_kv_heads
        key_states_r = self._repeat_kv(key_states, num_queries_per_kv)
        value_states_r = self._repeat_kv(value_states, num_queries_per_kv)

        mask_key = (seq_len, kv_seq_len, str(device))
        sliding_causal_mask = self._sliding_mask_cache.get(mask_key)
        if sliding_causal_mask is None:
            row_idx = torch.arange(seq_len, device=device).unsqueeze(1) + (
                kv_seq_len - seq_len
            )
            col_idx = torch.arange(kv_seq_len, device=device).unsqueeze(0)
            sliding_causal_mask = (row_idx >= col_idx) & (
                (row_idx - col_idx) < self.window_size
            )
            if len(self._sliding_mask_cache) >= self._sliding_mask_cache_max:
                self._sliding_mask_cache.clear()
            self._sliding_mask_cache[mask_key] = sliding_causal_mask

        if attention_mask is not None:
            padding_mask = (
                attention_mask.unsqueeze(1).unsqueeze(2)
                if attention_mask.dim() == 2
                else attention_mask
            )
            attn_mask = sliding_causal_mask.unsqueeze(0) & padding_mask.bool()
        else:
            attn_mask = sliding_causal_mask

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states_r,
            value_states_r,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )
        return self.out_proj(attn_output), present_key_value
