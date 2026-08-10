"""
model.py

Contains two model families:

1. MixtralConfig / MixtralForCausalLM -- sliding-window-GQA + Top-2 MoE
   baseline (control for ablations).

2. HybridMambaMoEConfig / HybridForCausalLM -- Hybrid Mamba-MoE with Dual
   Memory (research/research.md v2.0):
       - Sliding window GQA branch (reuses SlidingWindowGQA)
       - Mamba selective-SSM branch (MambaBlock) with checkpointed sequential
         scan by default (optional Hillis-Steele parallel scan) and
         incremental (conv_state, ssm_state) caching for autoregressive decode
       - Two compressive memory banks (CompressiveMemoryBank), one per
         branch: read *into* branch inputs, write *raw* branch outputs
         (research.md §3.2), O(L * m), m << L; decode/prefill write in
         chunk-sized buffers to match training
       - Token-wise gated fusion (TokenGatedFusion) -- O(L), not O(L^2)
       - Top-2 sparse MoE (shared DroplessMoELayer)

Falsification hooks (design doc §6):
    - HybridDecoderLayer returns per-layer write-gate stats
    - use_dual_memory=False for architecture-level memory-off ablation
    - HybridModel.zero_memory_states() for Test-1 zeroed-at-inference
    - build_test3_null_baseline_config() for matched-parameter SSM-only null
"""

import copy
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass
class MixtralConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 14336
    window_size: int = 4096
    rms_norm_eps: float = 1e-5
    init_range: float = 0.02
    router_aux_loss_coef: float = 0.02
    router_z_loss_coef: float = 5e-3

    num_experts: int = 8
    top_k: int = 2
    dropout: float = 0.1

    # Optional expert-capacity limiting for MoE (see DroplessMoELayer). None
    # preserves fully dropless, batch-independent behavior (default for research).
    # Set e.g. 1.25 only on memory-constrained hardware; logits then depend on
    # batch composition.
    capacity_factor: float | None = None

    # RoPE / positional configuration (previously hardcoded deep inside
    # RotaryEmbedding / SlidingWindowGQA).
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0

    # Special tokens (previously hardcoded to 1/2 inside the data pipeline
    # with no link back to the model config).
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    # Labels equal to this value are ignored by CrossEntropyLoss.
    label_ignore_index: int = -100

    # If True, lm_head.weight shares storage with embed_tokens.weight.
    tie_word_embeddings: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serializes configuration parameters to a dictionary layout."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "MixtralConfig":
        """Instantiates a configuration object from a standard dictionary."""
        return cls(**config_dict)

    def save_pretrained(self, save_path: str) -> None:
        """Saves configuration layout to a local JSON file."""
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_pretrained(cls, load_path: str) -> "MixtralConfig":
        """Loads configuration layout from a saved local JSON file."""
        with open(load_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


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

        return (x_fp32 * norm_x).to(input_dtype) * self.weight


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


class SwiGLUExpert(nn.Module):
    """Standard SwiGLU feed-forward expert. (The deeper duplicate definition
    that used to shadow this one -- adding two extra Linear+RMSNorm+SiLU
    residual blocks per expert -- has been removed; it silently inflated
    params ~30% and activation memory ~2x with no indication that was
    intended.)"""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_proj = self.w_gate(x)
        up_proj = self.w_up(x)
        activated = F.silu(gate_proj) * up_proj
        return self.w_down(activated)


class MOERouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})."
            )

        self.num_experts = num_experts
        self.top_k = top_k
        self.wg = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        if len(orig_shape) == 3:
            x = x.reshape(-1, orig_shape[-1])

        input_dtype = x.dtype
        # IMPORTANT: do NOT cast `x` to float32 before this matmul. Under
        # FSDP's MixedPrecision(param_dtype=torch.float16), `self.wg.weight`
        # is fp16 during forward; casting the *input* to fp32 first creates
        # a genuine fp32-vs-fp16 dtype mismatch that nn.Linear (correctly)
        # refuses to run ("mat1 and mat2 to have the same dtype"). This
        # used to be silently papered over by an outer
        # `torch.amp.autocast(dtype=torch.float16)` in main.py, which
        # downcast the fp32 tensor back to fp16 right before the linear op
        # -- but that autocast has been removed as redundant with FSDP's own
        # MixedPrecision policy, so the router must not depend on it.
        # Instead: run the linear at its native (weight) dtype, then upcast
        # the *output* logits to fp32 -- that's what actually needs the
        # extra precision (clamp / logsumexp / softmax stability), not the
        # matmul itself.
        logits = self.wg(x).to(torch.float32)
        # Clamp router logits to prevent FP16 softmax overflow.
        logits = torch.clamp(logits, min=-30.0, max=30.0)

        # Router z-loss: numerically stable logsumexp over the expert dim.
        logsumexp_vals = torch.logsumexp(logits, dim=-1)
        z_loss = torch.mean(logsumexp_vals**2)

        # Full softmax distribution over ALL experts is needed for the
        # Switch-Transformer-style load-balancing loss (p_i below). Top-k is
        # only used to select which experts actually process each token.
        full_probs = F.softmax(logits, dim=-1)  # [N, E]

        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1).to(input_dtype)

        # f_i: fraction of tokens for which expert i is in the top-k set.
        one_hot_indices = F.one_hot(topk_indices, num_classes=self.num_experts).float()
        f_i = one_hot_indices.sum(dim=1).mean(
            dim=0
        )  # [E], mean over tokens of "selected count"

        # p_i: mean router probability assigned to expert i across ALL
        # tokens (not just top-k weight) -- this is the standard Switch
        # Transformer formulation. Using only top-k weights (as before)
        # underweights the probability mass on non-selected experts and
        # understates load imbalance.
        p_i = full_probs.mean(dim=0)  # [E]

        aux_loss = self.num_experts * torch.sum(f_i * p_i)

        return topk_weights, topk_indices, aux_loss, z_loss


class DroplessMoELayer(nn.Module):
    """
    MoE dispatch/combine.

    `capacity_factor` (from config) optionally bounds the number of tokens
    each expert will process to `capacity_factor * num_tokens / num_experts`,
    dropping overflow tokens for that expert. This is a memory safety valve for
    T4: with imbalanced routing, a naive dispatch can spike a single expert's
    batch 2-3x versus the average, which is the difference between fitting in
    16GB and OOM. Set `capacity_factor=None` to restore the original fully
    "dropless" (no token ever skipped) behavior -- note this reintroduces the
    memory-spike risk under imbalanced routing.

    When capacity drops tokens, remaining top-k weights are renormalized so
    per-token MoE magnitude is preserved. During training, overflow tokens
    are chosen via a random permutation (not always the first ``capacity``
    indices) to reduce order bias; eval uses stable first-``capacity`` order.
    """

    def __init__(
        self,
        router: nn.Module,
        experts: nn.ModuleList,
        capacity_factor: float | None = None,
    ):
        super().__init__()
        self.router = router
        self.experts = experts
        self.num_experts = len(experts)
        self.capacity_factor = capacity_factor

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        num_tokens = x_flat.size(0)

        topk_weights, topk_indices, aux_loss, z_loss = self.router(x_flat)
        moe_output = torch.zeros_like(x_flat)
        applied_weights = torch.zeros(
            num_tokens, device=x_flat.device, dtype=x_flat.dtype
        )

        capacity = None
        if self.capacity_factor is not None:
            capacity = max(
                1,
                int(
                    self.capacity_factor
                    * num_tokens
                    * topk_indices.size(1)
                    / self.num_experts
                ),
            )

        for expert_idx in range(self.num_experts):
            token_mask = topk_indices == expert_idx
            if not token_mask.any():
                continue

            row_indices, k_indices = torch.where(token_mask)

            if capacity is not None and row_indices.numel() > capacity:
                if self.training:
                    perm = torch.randperm(
                        row_indices.numel(), device=row_indices.device
                    )[:capacity]
                    row_indices = row_indices[perm]
                    k_indices = k_indices[perm]
                else:
                    row_indices = row_indices[:capacity]
                    k_indices = k_indices[:capacity]

            expert_inputs = x_flat[row_indices]
            expert_outputs = self.experts[expert_idx](expert_inputs)

            gating_scale = topk_weights[row_indices, k_indices].unsqueeze(-1)
            moe_output.index_add_(0, row_indices, expert_outputs * gating_scale)
            applied_weights.index_add_(
                0, row_indices, gating_scale.squeeze(-1).to(applied_weights.dtype)
            )

        if self.capacity_factor is not None:
            target_weights = topk_weights.sum(dim=-1)
            renorm = target_weights / applied_weights.clamp(min=1e-9)
            renorm = torch.where(
                applied_weights > 0,
                renorm.clamp(max=10.0),
                torch.ones_like(renorm),
            )
            moe_output = moe_output * renorm.unsqueeze(-1)

        return moe_output.reshape(*orig_shape), aux_loss, z_loss


class MixtralDecoderLayer(nn.Module):
    def __init__(self, config: MixtralConfig) -> None:
        super().__init__()
        self.rmsnorm_attn = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_block = SlidingWindowGQA(config)
        self.rmsnorm_moe = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        router = MOERouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
        )
        experts = nn.ModuleList(
            [
                SwiGLUExpert(config.hidden_size, config.intermediate_size)
                for _ in range(config.num_experts)
            ]
        )
        self.moe_block = DroplessMoELayer(
            router, experts, capacity_factor=config.capacity_factor
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:

        attn_out, present_key_value = self.attention_block(
            self.rmsnorm_attn(x),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x_attn = x + attn_out

        moe_in = self.rmsnorm_moe(x_attn)
        if attention_mask is not None and attention_mask.dim() == 2:
            token_mask = attention_mask[:, -x.size(1) :].unsqueeze(-1).to(moe_in.dtype)
            moe_in = moe_in * token_mask
        moe_out, aux_loss, z_loss = self.moe_block(moe_in)
        x_out = x_attn + moe_out

        return x_out, aux_loss, z_loss, present_key_value


@dataclass
class MixtralTrainingOutput:
    """
    Encapsulates logits, language modeling loss, auxiliary routing losses,
    and attention state caches for unified training and inference steps.

    NOTE: this used to be defined twice in this file (an earlier, incomplete
    definition with only `loss`/`router_loss`, and this one). Only this
    single definition remains; the training loop in main.py relies on
    `ce_loss`, `router_aux_loss`, and `router_z_loss` all being present.
    """

    logits: Tensor
    loss: Tensor | None = None
    ce_loss: Tensor | None = None
    router_aux_loss: Tensor | None = None
    router_z_loss: Tensor | None = None
    past_key_values: list[tuple[Tensor, Tensor]] | None = None


class MixtralModel(nn.Module):
    def __init__(self, config: MixtralConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [MixtralDecoderLayer(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list | None]:
        hidden_states = self.embed_tokens(input_ids)
        seq_len = hidden_states.size(1)

        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)

        total_aux_loss = torch.tensor(
            0.0, device=hidden_states.device, dtype=hidden_states.dtype
        )
        total_z_loss = torch.tensor(
            0.0, device=hidden_states.device, dtype=hidden_states.dtype
        )
        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None

            hidden_states, layer_aux_loss, layer_z_loss, present_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            total_aux_loss = total_aux_loss + layer_aux_loss
            total_z_loss = total_z_loss + layer_z_loss

            if use_cache:
                present_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)
        n_layers = max(len(self.layers), 1)
        return (
            hidden_states,
            total_aux_loss / n_layers,
            total_z_loss / n_layers,
            present_key_values,
        )


class MixtralForCausalLM(nn.Module):
    def __init__(self, config: MixtralConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.router_z_loss_coef = config.router_z_loss_coef
        self.init_range = config.init_range

        self.model = MixtralModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = self.init_range / math.sqrt(2 * self.config.num_layers)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None and not getattr(
                module.bias, "_no_reinit", False
            ):
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.init_range)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: list | None = None,
        use_cache: bool = False,
        labels: Tensor | None = None,
    ) -> MixtralTrainingOutput:

        hidden_states, aux_loss, z_loss, present_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)

        # NOTE: `ce_loss` must be initialized before the conditional --
        # previously it was only assigned inside `if labels is not None`
        # but was unconditionally referenced when building the output
        # below, which raised a NameError whenever `labels` was omitted
        # (e.g. plain inference/generation calls).
        loss = None
        ce_loss = None
        if labels is not None:
            # `labels` are expected to ALREADY be the next-token shift of
            # `input_ids` (i.e. labels[i] is the target for logits[i]),
            # matching the (chunk[:-1], chunk[1:]) contract produced by
            # MmapShardDataset. Do NOT shift again here.
            if attention_mask is not None and attention_mask.dim() == 2:
                labels = labels.masked_fill(
                    attention_mask[:, -labels.size(1) :] == 0,
                    self.config.label_ignore_index,
                )
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.label_ignore_index)
            ce_loss = loss_fct(logits.view(-1, self.vocab_size), labels.reshape(-1))

            loss = (
                ce_loss
                + (self.router_aux_loss_coef * aux_loss)
                + (self.router_z_loss_coef * z_loss)
            )

        output = MixtralTrainingOutput(
            logits=logits,
            loss=loss,
            ce_loss=ce_loss,
            router_aux_loss=aux_loss,
            router_z_loss=z_loss,
            past_key_values=present_key_values,
        )
        return output


# =============================================================================
# Hybrid Mamba–MoE with Dual Memory
# =============================================================================


@dataclass
class HybridMambaMoEConfig(MixtralConfig):
    """
    Extends MixtralConfig with Mamba-branch and memory-branch settings.
    All Mixtral fields are reused for the GQA and MoE parts of each layer.
    """

    mamba_state_size: int = 16
    mamba_conv_kernel: int = 4
    mamba_expand: int = 2
    mamba_dt_rank: int | None = None

    use_dual_memory: bool = True
    memory_size: int = 64
    memory_num_heads: int = 8

    # Split long training sequences so memory write params get BPTT gradients.
    memory_chunk_size: int | None = 512
    # Decode: write memory banks every N new tokens (matches training chunking).
    memory_write_interval: int | None = None

    use_parallel_scan: bool = False
    gradient_checkpointing: bool = False


MambaCache = tuple[Tensor, Tensor]  # (conv_state, ssm_state)


class MambaBlock(nn.Module):
    """
    Selective SSM (Mamba / S6) in pure PyTorch.

    Training defaults to a checkpointed sequential associative scan (O(L)
    work, bounded activation memory). Optional Hillis-Steele parallel scan
    (`use_parallel_scan=True`) is faster on short sequences but O(L log L).
    Decode uses allocate_inference_cache() + step().
    """

    def __init__(
        self,
        hidden_size: int,
        state_size: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        use_parallel_scan: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.d_inner = expand * hidden_size
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(hidden_size / 16)
        self.use_parallel_scan = use_parallel_scan

        self.in_proj = nn.Linear(hidden_size, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=conv_kernel,
            groups=self.d_inner,
            padding=conv_kernel - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * state_size, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Softplus(dt_bias) ~ Uniform[dt_min, dt_max] at init (official Mamba).
        # Marked _no_reinit so HybridForCausalLM._init_weights does not zero it.
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # type: ignore[attr-defined]

        A = torch.arange(1, state_size + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]
        self.out_proj = nn.Linear(self.d_inner, hidden_size, bias=False)

    def allocate_inference_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> MambaCache:
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        ssm_dtype = torch.float32
        conv_state = torch.zeros(
            batch_size,
            self.d_inner,
            self.conv_kernel,
            device=device,
            dtype=conv_dtype,
        )
        ssm_state = torch.zeros(
            batch_size,
            self.d_inner,
            self.state_size,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def forward(
        self,
        x: Tensor,
        cache: MambaCache | None = None,
        use_cache: bool = False,
        attention_mask: Tensor | None = None,
        active_batch_mask: Tensor | None = None,
    ) -> tuple[Tensor, MambaCache | None]:
        """
        x: [B, L, hidden_size]
        If use_cache and L==1 and cache is provided, runs a single decode step.
        Otherwise runs full-sequence prefill (parallel scan); when use_cache,
        returns updated (conv_state, ssm_state) for subsequent steps.
        """
        _, seq_len, _ = x.shape

        if use_cache and cache is not None and seq_len == 1:
            return self.step(x, cache[0], cache[1], active_batch_mask=active_batch_mask)

        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        x_conv = x_in.transpose(1, 2)
        if use_cache:
            # Keep last conv_kernel *valid* tokens as the rolling conv buffer.
            if attention_mask is not None and attention_mask.dim() == 2:
                token_mask = attention_mask[:, -seq_len:]
                # Right-padding assumption: valid prefix length per row.
                valid_lens = token_mask.sum(dim=1)
                conv_state = torch.zeros(
                    x.size(0),
                    self.d_inner,
                    self.conv_kernel,
                    device=x.device,
                    dtype=x_in.dtype,
                )
                for b in range(x.size(0)):
                    vl = int(valid_lens[b].item())
                    if vl <= 0:
                        continue
                    take = min(self.conv_kernel, vl)
                    conv_state[b, :, -take:] = x_in[b, vl - take : vl].transpose(0, 1)
            else:
                pad = max(self.conv_kernel - seq_len, 0)
                conv_state = F.pad(x_conv, (pad, 0))[
                    :, :, -self.conv_kernel :
                ].contiguous()
        else:
            conv_state = None

        x_conv = self.conv1d(x_conv)[..., :seq_len]
        x_conv = F.silu(x_conv).transpose(1, 2)

        x_dbl = self.x_proj(x_conv)
        dt, B_param, C_param = torch.split(
            x_dbl, [self.dt_rank, self.state_size, self.state_size], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log.float())
        y, ssm_state = self._selective_scan(
            x_conv,
            dt,
            A,
            B_param,
            C_param,
            self.D,
            return_final_state=use_cache,
            use_parallel_scan=self.use_parallel_scan,
            training=self.training,
            attention_mask=attention_mask,
        )
        y = y * F.silu(z)
        out = self.out_proj(y)

        new_cache: MambaCache | None = None
        if use_cache:
            assert conv_state is not None and ssm_state is not None
            new_cache = (conv_state, ssm_state)
        return out, new_cache

    def step(
        self,
        x: Tensor,
        conv_state: Tensor,
        ssm_state: Tensor,
        active_batch_mask: Tensor | None = None,
    ) -> tuple[Tensor, MambaCache]:
        """Single-token decode. x: [B, 1, hidden_size]."""
        assert x.size(1) == 1
        dtype = x.dtype

        prev_conv = conv_state.clone()
        prev_ssm = ssm_state.clone()

        xz = self.in_proj(x.squeeze(1))
        x_in, z = xz.chunk(2, dim=-1)

        # Shift conv buffer in-place (slice copy, no full tensor roll).
        conv_state[:, :, :-1].copy_(conv_state[:, :, 1:])
        conv_state[:, :, -1] = x_in
        x_conv = torch.sum(conv_state * self.conv1d.weight.squeeze(1), dim=-1)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x_conv = F.silu(x_conv).to(dtype=dtype)

        x_dbl = self.x_proj(x_conv)
        dt, B_param, C_param = torch.split(
            x_dbl, [self.dt_rank, self.state_size, self.state_size], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log.float())
        dt_f = dt.float()
        dA = torch.exp(dt_f.unsqueeze(-1) * A)  # [B, d_inner, n]
        dB_u = (
            dt_f.unsqueeze(-1)
            * B_param.float().unsqueeze(1)
            * x_conv.float().unsqueeze(-1)
        )
        ssm_state = ssm_state.float() * dA + dB_u
        y = (ssm_state * C_param.float().unsqueeze(1)).sum(dim=-1)
        y = (y + x_conv.float() * self.D.float()).to(dtype)
        y = y * F.silu(z)
        out = self.out_proj(y).unsqueeze(1)

        if active_batch_mask is not None and (~active_batch_mask).any():
            inactive = ~active_batch_mask
            conv_state = conv_state.clone()
            ssm_state = ssm_state.clone()
            conv_state[inactive] = prev_conv[inactive]
            ssm_state[inactive] = prev_ssm[inactive]
            out = out.clone()
            out[inactive] = 0

        return out, (conv_state, ssm_state)

    @staticmethod
    def _parallel_associative_scan(delta_a: Tensor, delta_b_u: Tensor) -> Tensor:
        """
        Hillis-Steele inclusive scan: h_t = delta_a_t * h_{t-1} + delta_b_u_t.
        O(L log L) work and training memory — use only when use_parallel_scan.
        """
        seq_len = delta_a.size(1)
        a = delta_a
        b = delta_b_u
        n = 1
        while n < seq_len:
            a_prev = a[:, :-n]
            b_prev = b[:, :-n]
            a_curr = a[:, n:]
            b_curr = b[:, n:]
            a = torch.cat([a[:, :n], a_curr * a_prev], dim=1)
            b = torch.cat([b[:, :n], a_curr * b_prev + b_curr], dim=1)
            n *= 2
        return b

    @staticmethod
    def _sequential_associative_scan(delta_a: Tensor, delta_b_u: Tensor) -> Tensor:
        """O(L) work sequential scan; pair with checkpoint during training."""
        _, seq_len, _, _ = delta_a.shape
        state = torch.zeros_like(delta_b_u[:, 0])
        outputs = []
        for t in range(seq_len):
            state = delta_a[:, t] * state + delta_b_u[:, t]
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    @classmethod
    def _selective_scan(
        cls,
        u: Tensor,
        dt: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
        return_final_state: bool = False,
        use_parallel_scan: bool = False,
        training: bool = False,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        u, dt: [B, L, d_inner]; A: [d_inner, n]; B, C: [B, L, n]; D: [d_inner]
        On pad positions (attention_mask==0), apply identity state transition
        so SSM state does not decay through padding.
        """
        input_dtype = u.dtype
        u_f = u.float()
        dt_f = dt.float()
        B_f = B.float()
        C_f = C.float()

        delta_a = torch.exp(dt_f.unsqueeze(-1) * A)  # [B, L, d_inner, n]
        delta_b_u = dt_f.unsqueeze(-1) * B_f.unsqueeze(2) * u_f.unsqueeze(-1)

        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("MambaBlock expects 2D attention_mask [B, L].")
            if attention_mask.size(1) < u.size(1):
                raise ValueError(
                    f"attention_mask length {attention_mask.size(1)} < seq_len {u.size(1)}."
                )
            token_mask = attention_mask[:, -u.size(1) :].to(dtype=delta_a.dtype)
            m = token_mask.unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
            # Pad steps: h_t = 1 * h_{t-1} + 0
            delta_a = delta_a * m + (1.0 - m)
            delta_b_u = delta_b_u * m

        if use_parallel_scan:
            states = cls._parallel_associative_scan(delta_a, delta_b_u)
        elif training:
            states = checkpoint(
                cls._sequential_associative_scan,
                delta_a,
                delta_b_u,
                use_reentrant=False,
            )
        else:
            states = cls._sequential_associative_scan(delta_a, delta_b_u)

        y = (states * C_f.unsqueeze(2)).sum(dim=-1)
        y = y + u_f * D.float()
        if attention_mask is not None:
            y = y * token_mask.unsqueeze(-1)
        final_state = states[:, -1].contiguous() if return_final_state else None
        return y.to(input_dtype), final_state


class CompressiveMemoryBank(nn.Module):
    """
    Fixed-size (m slots) gated read/write memory bank.

    Multi-head scaled-dot attention over a small bank (m << L).
    """

    def __init__(
        self, hidden_size: int, memory_size: int = 64, num_heads: int = 8
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by memory_num_heads ({num_heads})."
            )
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.init_memory = nn.Parameter(torch.randn(memory_size, hidden_size) * 0.02)
        self.summary_query = nn.Parameter(torch.randn(memory_size, hidden_size) * 0.02)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.write_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.write_update = nn.Linear(hidden_size, hidden_size)

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        return (
            self.init_memory.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .to(device=device, dtype=dtype)
            .clone()
        )

    @staticmethod
    def _key_padding_mask(attention_mask: Tensor | None) -> Tensor | None:
        """Convert 1=keep / 0=pad mask to key_padding_mask (True = ignore)."""
        if attention_mask is None:
            return None
        if attention_mask.dim() != 2:
            raise ValueError(
                "CompressiveMemoryBank expects a 2D attention_mask [B, L]."
            )
        return ~attention_mask.bool()

    def _shape_heads(self, t: Tensor, seq_len: int) -> Tensor:
        # [B, L, H] -> [B, heads, L, head_dim]
        b = t.size(0)
        return t.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _attend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        query: [B, Q, H], key/value: [B, K, H]
        key_padding_mask: [B, K] True = ignore
        """
        bsz, q_len, _ = query.shape
        k_len = key.size(1)
        q = self._shape_heads(self.q_proj(query), q_len)
        k = self._shape_heads(self.k_proj(key), k_len)
        v = self._shape_heads(self.v_proj(value), k_len)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            all_masked = key_padding_mask.all(dim=-1)  # [B]
            if all_masked.any():
                # Avoid NaN softmax when a row has no valid keys.
                scores = scores.clone()
                scores[all_masked] = 0.0
        attn = F.softmax(scores, dim=-1)
        if key_padding_mask is not None and all_masked.any():
            attn = attn.clone()
            attn[all_masked] = 0.0
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.out_proj(out)

    def read(
        self,
        x: Tensor,
        memory: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        del attention_mask  # queries may be zeroed by caller for pads
        return self._attend(x, memory, memory, key_padding_mask=None)

    def write(
        self,
        x: Tensor,
        memory: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        kpm = self._key_padding_mask(attention_mask)
        batch_size = x.size(0)
        query = self.summary_query.unsqueeze(0).expand(batch_size, -1, -1)
        chunk_summary = self._attend(query, x, x, key_padding_mask=kpm)

        gate = torch.sigmoid(
            self.write_gate(torch.cat([memory, chunk_summary], dim=-1))
        )
        new_memory = gate * memory + (1.0 - gate) * self.write_update(chunk_summary)
        return new_memory, gate


class TokenGatedFusion(nn.Module):
    """O(L) per-token fusion of attention-branch and Mamba-branch outputs."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, a: Tensor, m: Tensor) -> tuple[Tensor, Tensor]:
        g = torch.sigmoid(self.gate(torch.cat([a, m], dim=-1)))
        fused = g * a + (1.0 - g) * m
        return fused, g


HybridMemoryState = tuple[Tensor, Tensor]
# Accumulated raw branch outputs awaiting a chunked memory write.
MemoryWriteBuffer = tuple[Tensor, Tensor]


def _hybrid_layer_forward(
    layer: "HybridDecoderLayer",
    hidden_states: Tensor,
    memory_state: HybridMemoryState | None,
    attention_mask: Tensor | None,
    position_ids: Tensor | None,
    past_key_value: tuple[Tensor, Tensor] | None,
    mamba_cache: MambaCache | None,
    use_cache: bool,
    skip_memory_write: bool,
    write_buffer: MemoryWriteBuffer | None,
    active_batch_mask: Tensor | None,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    tuple[Tensor, Tensor] | None,
    HybridMemoryState | None,
    MambaCache | None,
    dict[str, Tensor],
    MemoryWriteBuffer | None,
]:
    return layer(
        hidden_states,
        memory_state=memory_state,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        mamba_cache=mamba_cache,
        use_cache=use_cache,
        skip_memory_write=skip_memory_write,
        write_buffer=write_buffer,
        active_batch_mask=active_batch_mask,
    )


class HybridDecoderLayer(nn.Module):
    """
    RMSNorm -> memory-conditioned {GQA, Mamba} in parallel -> write raw
    branch outputs to memory banks -> TokenGatedFusion -> residual ->
    RMSNorm -> Top-2 MoE -> residual.

    Matches research.md §3.2: banks are read *into* each branch (via an
    input combine on the shared normed states), and raw branch outputs
    write back — not the memory-augmented tensors.
    """

    def __init__(self, config: HybridMambaMoEConfig) -> None:
        super().__init__()
        self.use_dual_memory = config.use_dual_memory

        self.rmsnorm_in = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_block = SlidingWindowGQA(config)
        self.mamba_block = MambaBlock(
            hidden_size=config.hidden_size,
            state_size=config.mamba_state_size,
            conv_kernel=config.mamba_conv_kernel,
            expand=config.mamba_expand,
            dt_rank=config.mamba_dt_rank,
            use_parallel_scan=config.use_parallel_scan,
        )

        if self.use_dual_memory:
            self.attn_memory_bank = CompressiveMemoryBank(
                config.hidden_size, config.memory_size, config.memory_num_heads
            )
            self.state_memory_bank = CompressiveMemoryBank(
                config.hidden_size, config.memory_size, config.memory_num_heads
            )
            # Condition each branch's *input* with a memory read (diagram:
            # AM/SM -.read.-> GQA/Mamba), rather than mixing after the branch.
            self.attn_memory_combine = nn.Linear(
                config.hidden_size * 2, config.hidden_size
            )
            self.state_memory_combine = nn.Linear(
                config.hidden_size * 2, config.hidden_size
            )

        self.fusion = TokenGatedFusion(config.hidden_size)
        self.rmsnorm_moe = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        router = MOERouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
        )
        experts = nn.ModuleList(
            [
                SwiGLUExpert(config.hidden_size, config.intermediate_size)
                for _ in range(config.num_experts)
            ]
        )
        self.moe_block = DroplessMoELayer(
            router, experts, capacity_factor=config.capacity_factor
        )

    def init_memory_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> HybridMemoryState | None:
        if not self.use_dual_memory:
            return None
        return (
            self.attn_memory_bank.init_state(batch_size, device, dtype),
            self.state_memory_bank.init_state(batch_size, device, dtype),
        )

    def allocate_mamba_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> MambaCache:
        return self.mamba_block.allocate_inference_cache(batch_size, device, dtype)

    def zero_memory_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> HybridMemoryState | None:
        """Test-1 hook: same shapes as init_state, but all zeros."""
        state = self.init_memory_state(batch_size, device, dtype)
        if state is None:
            return None
        return tuple(torch.zeros_like(t) for t in state)  # type: ignore[return-value]

    def forward(
        self,
        x: Tensor,
        memory_state: HybridMemoryState | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        mamba_cache: MambaCache | None = None,
        use_cache: bool = False,
        skip_memory_write: bool = False,
        write_buffer: MemoryWriteBuffer | None = None,
        active_batch_mask: Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        tuple[Tensor, Tensor] | None,
        HybridMemoryState | None,
        MambaCache | None,
        dict[str, Tensor],
        MemoryWriteBuffer | None,
    ]:
        residual = x
        x_norm = self.rmsnorm_in(x)
        seq_len = x.size(1)

        # Memory R/W keys are the current chunk tokens [B, seq_len]; GQA may
        # receive a longer window-aligned padding mask during cached decode.
        token_attention_mask: Tensor | None = None
        if attention_mask is not None and attention_mask.dim() == 2:
            if attention_mask.size(1) < seq_len:
                raise ValueError(
                    f"attention_mask length {attention_mask.size(1)} < seq_len {seq_len}."
                )
            token_attention_mask = attention_mask[:, -seq_len:]

        hidden_mask: Tensor | None = None
        if token_attention_mask is not None:
            hidden_mask = token_attention_mask.unsqueeze(-1).to(x_norm.dtype)
            x_norm = x_norm * hidden_mask

        new_memory_state = memory_state
        new_write_buffer: MemoryWriteBuffer | None = write_buffer
        gate_stats: dict[str, Tensor] = {}
        attn_input = x_norm
        mamba_input = x_norm

        if self.use_dual_memory:
            if memory_state is None:
                memory_state = self.init_memory_state(x.size(0), x.device, x.dtype)
            a_mem, s_mem = memory_state

            # Read memory *into* each branch (condition inputs before GQA/Mamba).
            a_read = self.attn_memory_bank.read(
                x_norm, a_mem, attention_mask=token_attention_mask
            )
            attn_input = self.attn_memory_combine(torch.cat([x_norm, a_read], dim=-1))

            s_read = self.state_memory_bank.read(
                x_norm, s_mem, attention_mask=token_attention_mask
            )
            mamba_input = self.state_memory_combine(torch.cat([x_norm, s_read], dim=-1))

        if hidden_mask is not None:
            attn_input = attn_input * hidden_mask
            mamba_input = mamba_input * hidden_mask

        attn_out, present_key_value = self.attention_block(
            attn_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            active_batch_mask=active_batch_mask,
        )
        mamba_token_mask = token_attention_mask
        mamba_out, new_mamba_cache = self.mamba_block(
            mamba_input,
            cache=mamba_cache,
            use_cache=use_cache,
            attention_mask=mamba_token_mask,
            active_batch_mask=active_batch_mask,
        )

        if self.use_dual_memory:
            assert memory_state is not None
            a_mem, s_mem = memory_state
            if hidden_mask is not None:
                attn_out = attn_out * hidden_mask
                mamba_out = mamba_out * hidden_mask

            # Accumulate raw branch outputs for chunk-aligned memory writes.
            buf_attn = attn_out
            buf_mamba = mamba_out
            if active_batch_mask is not None:
                active = active_batch_mask.to(dtype=buf_attn.dtype).view(-1, 1, 1)
                buf_attn = buf_attn * active
                buf_mamba = buf_mamba * active
            if write_buffer is not None:
                buf_attn = torch.cat([write_buffer[0], buf_attn], dim=1)
                buf_mamba = torch.cat([write_buffer[1], buf_mamba], dim=1)

            if skip_memory_write:
                new_memory_state = memory_state
                new_write_buffer = (buf_attn, buf_mamba)
            else:
                # Build mask for buffered tokens: prior buffer assumed valid,
                # current tokens use token_attention_mask when present.
                buf_len = buf_attn.size(1)
                cur_len = attn_out.size(1)
                if token_attention_mask is not None:
                    if buf_len > cur_len:
                        prior = torch.ones(
                            token_attention_mask.size(0),
                            buf_len - cur_len,
                            device=token_attention_mask.device,
                            dtype=token_attention_mask.dtype,
                        )
                        write_mask = torch.cat([prior, token_attention_mask], dim=1)
                    else:
                        write_mask = token_attention_mask
                else:
                    write_mask = None
                if active_batch_mask is not None and write_mask is not None:
                    write_mask = write_mask * active_batch_mask.unsqueeze(-1).to(
                        dtype=write_mask.dtype
                    )

                new_a_mem, a_write_gate = self.attn_memory_bank.write(
                    buf_attn, a_mem, attention_mask=write_mask
                )
                new_s_mem, s_write_gate = self.state_memory_bank.write(
                    buf_mamba, s_mem, attention_mask=write_mask
                )
                new_memory_state = (new_a_mem, new_s_mem)
                new_write_buffer = None
                gate_stats = {
                    "attn_write_gate_mean": a_write_gate.detach().mean(),
                    "state_write_gate_mean": s_write_gate.detach().mean(),
                }

        fused, _fusion_gate = self.fusion(attn_out, mamba_out)
        if hidden_mask is not None:
            fused = fused * hidden_mask
        x = residual + fused

        moe_in = self.rmsnorm_moe(x)
        if hidden_mask is not None:
            moe_in = moe_in * hidden_mask
        moe_out, aux_loss, z_loss = self.moe_block(moe_in)
        x_out = x + moe_out

        return (
            x_out,
            aux_loss,
            z_loss,
            present_key_value,
            new_memory_state,
            new_mamba_cache,
            gate_stats,
            new_write_buffer,
        )


def _top_k_filter(logits: Tensor, top_k: int) -> Tensor:
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(
        logits < min_values, torch.full_like(logits, float("-inf")), logits
    )


def _top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    mask = torch.zeros_like(sorted_mask).scatter(1, sorted_indices, sorted_mask)
    return logits.masked_fill(mask, float("-inf"))


@dataclass
class HybridTrainingOutput:
    logits: Tensor
    loss: Tensor | None = None
    ce_loss: Tensor | None = None
    router_aux_loss: Tensor | None = None
    router_z_loss: Tensor | None = None
    past_key_values: list[tuple[Tensor, Tensor]] | None = None
    memory_states: list[HybridMemoryState | None] | None = None
    mamba_caches: list[MambaCache | None] | None = None
    gate_stats: dict[str, Tensor] | None = None
    write_buffers: list[MemoryWriteBuffer | None] | None = None


class HybridModel(nn.Module):
    def __init__(self, config: HybridMambaMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [HybridDecoderLayer(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def allocate_mamba_caches(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> list[MambaCache]:
        return [
            layer.allocate_mamba_cache(batch_size, device, dtype)
            for layer in self.layers
        ]

    def init_memory_states(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> list[HybridMemoryState | None]:
        return [
            layer.init_memory_state(batch_size, device, dtype) for layer in self.layers
        ]

    def zero_memory_states(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> list[HybridMemoryState | None]:
        """Test-1: keep memory modules, but start from all-zero banks."""
        return [
            layer.zero_memory_state(batch_size, device, dtype) for layer in self.layers
        ]

    def forward(
        self,
        input_ids: Tensor,
        memory_states: list[HybridMemoryState | None] | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: list | None = None,
        mamba_caches: list[MambaCache | None] | None = None,
        past_seen_tokens: int | None = None,
        use_cache: bool = False,
        skip_memory_write: bool = False,
        write_buffers: list[MemoryWriteBuffer | None] | None = None,
        active_batch_mask: Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        list | None,
        list[HybridMemoryState | None],
        list[MambaCache | None] | None,
        dict[str, Tensor],
        list[MemoryWriteBuffer | None] | None,
    ]:
        hidden_states = self.embed_tokens(input_ids)
        batch_size, seq_len = hidden_states.shape[:2]

        if position_ids is None:
            # Absolute positions must track tokens *seen*, not truncated KV length.
            if past_seen_tokens is None:
                past_seen_tokens = 0
            position_ids = (
                (
                    torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
                    + past_seen_tokens
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        max_pos = int(position_ids.max().item()) if position_ids.numel() else -1
        if max_pos >= self.config.max_position_embeddings:
            raise ValueError(
                f"position_ids max={max_pos} exceeds "
                f"max_position_embeddings={self.config.max_position_embeddings}."
            )

        total_aux_loss = torch.tensor(
            0.0, device=hidden_states.device, dtype=hidden_states.dtype
        )
        total_z_loss = torch.tensor(
            0.0, device=hidden_states.device, dtype=hidden_states.dtype
        )
        present_key_values = [] if use_cache else None
        new_memory_states: list[HybridMemoryState | None] = []
        new_mamba_caches: list[MambaCache | None] | None = [] if use_cache else None
        new_write_buffers: list[MemoryWriteBuffer | None] | None = (
            [] if self.config.use_dual_memory else None
        )
        all_gate_stats: dict[str, Tensor] = {}

        for i, layer in enumerate(self.layers):
            layer_past_kv = past_key_values[i] if past_key_values is not None else None
            layer_memory = memory_states[i] if memory_states is not None else None
            layer_mamba = mamba_caches[i] if mamba_caches is not None else None
            layer_buf = write_buffers[i] if write_buffers is not None else None

            layer_fn = _hybrid_layer_forward
            if self.config.gradient_checkpointing and self.training and not use_cache:
                (
                    hidden_states,
                    layer_aux_loss,
                    layer_z_loss,
                    present_kv,
                    layer_new_memory,
                    layer_new_mamba,
                    layer_gate_stats,
                    layer_new_buf,
                ) = checkpoint(
                    layer_fn,
                    layer,
                    hidden_states,
                    layer_memory,
                    attention_mask,
                    position_ids,
                    layer_past_kv,
                    layer_mamba,
                    use_cache,
                    skip_memory_write,
                    layer_buf,
                    active_batch_mask,
                    use_reentrant=False,
                )
            else:
                (
                    hidden_states,
                    layer_aux_loss,
                    layer_z_loss,
                    present_kv,
                    layer_new_memory,
                    layer_new_mamba,
                    layer_gate_stats,
                    layer_new_buf,
                ) = layer_fn(
                    layer,
                    hidden_states,
                    layer_memory,
                    attention_mask,
                    position_ids,
                    layer_past_kv,
                    layer_mamba,
                    use_cache,
                    skip_memory_write,
                    layer_buf,
                    active_batch_mask,
                )
            total_aux_loss = total_aux_loss + layer_aux_loss
            total_z_loss = total_z_loss + layer_z_loss
            new_memory_states.append(layer_new_memory)
            if new_write_buffers is not None:
                new_write_buffers.append(layer_new_buf)
            for k, v in layer_gate_stats.items():
                all_gate_stats[f"layer_{i}_{k}"] = v
            if use_cache:
                present_key_values.append(present_kv)
                new_mamba_caches.append(layer_new_mamba)

        hidden_states = self.norm(hidden_states)
        n_layers = max(len(self.layers), 1)
        return (
            hidden_states,
            total_aux_loss / n_layers,
            total_z_loss / n_layers,
            present_key_values,
            new_memory_states,
            new_mamba_caches,
            all_gate_stats,
            new_write_buffers,
        )


class HybridForCausalLM(nn.Module):
    def __init__(self, config: HybridMambaMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.router_z_loss_coef = config.router_z_loss_coef
        self.init_range = config.init_range

        self.model = HybridModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = self.init_range / math.sqrt(2 * self.config.num_layers)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None and not getattr(
                module.bias, "_no_reinit", False
            ):
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.init_range)

    def _memory_write_interval(self) -> int:
        interval = self.config.memory_write_interval
        if interval is not None:
            return max(1, interval)
        chunk = self.config.memory_chunk_size
        return max(1, chunk if chunk is not None else 512)

    def _should_chunk_training(
        self, seq_len: int, use_cache: bool, memory_states: list | None
    ) -> bool:
        chunk_size = self.config.memory_chunk_size
        return (
            self.config.use_dual_memory
            and chunk_size is not None
            and seq_len > chunk_size
            and not use_cache
            and memory_states is None
        )

    def _apply_label_ignore(
        self, labels: Tensor, attention_mask: Tensor | None
    ) -> Tensor:
        if attention_mask is not None and attention_mask.dim() == 2:
            labels = labels.masked_fill(
                attention_mask[:, -labels.size(1) :] == 0,
                self.config.label_ignore_index,
            )
        return labels

    def forward(
        self,
        input_ids: Tensor,
        memory_states: list[HybridMemoryState | None] | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: list | None = None,
        mamba_caches: list[MambaCache | None] | None = None,
        past_seen_tokens: int | None = None,
        use_cache: bool = False,
        labels: Tensor | None = None,
        skip_memory_write: bool = False,
        write_buffers: list[MemoryWriteBuffer | None] | None = None,
        active_batch_mask: Tensor | None = None,
    ) -> HybridTrainingOutput:
        seq_len = input_ids.size(1)
        if self._should_chunk_training(seq_len, use_cache, memory_states):
            return self._forward_chunked(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
            )

        (
            hidden_states,
            aux_loss,
            z_loss,
            present_key_values,
            new_memory_states,
            new_mamba_caches,
            gate_stats,
            new_write_buffers,
        ) = self.model(
            input_ids=input_ids,
            memory_states=memory_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            mamba_caches=mamba_caches,
            past_seen_tokens=past_seen_tokens,
            use_cache=use_cache,
            skip_memory_write=skip_memory_write,
            write_buffers=write_buffers,
            active_batch_mask=active_batch_mask,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        ce_loss = None
        if labels is not None:
            labels = self._apply_label_ignore(labels, attention_mask)
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.label_ignore_index)
            ce_loss = loss_fct(logits.view(-1, self.vocab_size), labels.reshape(-1))
            loss = (
                ce_loss
                + (self.router_aux_loss_coef * aux_loss)
                + (self.router_z_loss_coef * z_loss)
            )

        return HybridTrainingOutput(
            logits=logits,
            loss=loss,
            ce_loss=ce_loss,
            router_aux_loss=aux_loss,
            router_z_loss=z_loss,
            past_key_values=present_key_values,
            memory_states=new_memory_states,
            mamba_caches=new_mamba_caches,
            gate_stats=gate_stats,
            write_buffers=new_write_buffers,
        )

    def _forward_chunked(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        position_ids: Tensor | None,
        labels: Tensor | None,
    ) -> HybridTrainingOutput:
        """BPTT through memory banks within one backward pass."""
        chunk_size = self.config.memory_chunk_size
        assert chunk_size is not None
        seq_len = input_ids.size(1)
        batch_size = input_ids.size(0)
        device = input_ids.device

        memory_states: list[HybridMemoryState | None] | None = None
        logits_chunks: list[Tensor] = []
        total_aux = torch.tensor(0.0, device=device)
        total_z = torch.tensor(0.0, device=device)
        all_gate_stats: dict[str, Tensor] = {}
        token_weight = 0

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk_len = end - start
            chunk_ids = input_ids[:, start:end]
            chunk_mask = (
                attention_mask[:, start:end] if attention_mask is not None else None
            )
            if position_ids is not None:
                chunk_pos = position_ids[:, start:end]
            else:
                chunk_pos = (
                    torch.arange(start, end, dtype=torch.long, device=device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )

            (
                hidden_states,
                aux_loss,
                z_loss,
                _,
                memory_states,
                _,
                gate_stats,
                _,
            ) = self.model(
                input_ids=chunk_ids,
                memory_states=memory_states,
                attention_mask=chunk_mask,
                position_ids=chunk_pos,
                use_cache=False,
            )
            logits_chunks.append(self.lm_head(hidden_states))
            total_aux = total_aux + aux_loss * chunk_len
            total_z = total_z + z_loss * chunk_len
            token_weight += chunk_len
            all_gate_stats.update(gate_stats)

        logits = torch.cat(logits_chunks, dim=1)
        aux_loss = total_aux / max(token_weight, 1)
        z_loss = total_z / max(token_weight, 1)

        loss = None
        ce_loss = None
        if labels is not None:
            labels = self._apply_label_ignore(labels, attention_mask)
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.label_ignore_index)
            ce_loss = loss_fct(logits.view(-1, self.vocab_size), labels.reshape(-1))
            loss = (
                ce_loss
                + (self.router_aux_loss_coef * aux_loss)
                + (self.router_z_loss_coef * z_loss)
            )

        return HybridTrainingOutput(
            logits=logits,
            loss=loss,
            ce_loss=ce_loss,
            router_aux_loss=aux_loss,
            router_z_loss=z_loss,
            past_key_values=None,
            memory_states=memory_states,
            mamba_caches=None,
            gate_stats=all_gate_stats,
            write_buffers=None,
        )

    def _flush_memory_write_buffers(
        self,
        memory_states: list[HybridMemoryState | None] | None,
        write_buffers: list[MemoryWriteBuffer | None] | None,
    ) -> tuple[
        list[HybridMemoryState | None] | None,
        list[MemoryWriteBuffer | None] | None,
    ]:
        """Write any pending buffered branch outputs into memory banks."""
        if (
            not self.config.use_dual_memory
            or memory_states is None
            or write_buffers is None
        ):
            return memory_states, write_buffers

        new_states: list[HybridMemoryState | None] = []
        for layer, mem, buf in zip(self.model.layers, memory_states, write_buffers):
            if mem is None or buf is None or not layer.use_dual_memory:
                new_states.append(mem)
                continue
            a_mem, s_mem = mem
            new_a, _ = layer.attn_memory_bank.write(buf[0], a_mem)
            new_s, _ = layer.state_memory_bank.write(buf[1], s_mem)
            new_states.append((new_a, new_s))
        return new_states, None

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        eos_token_id: int | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Autoregressive generation with incremental KV + Mamba + memory caches.
        Prefill and decode write memory in chunks of ``memory_write_interval``
        (buffered branch outputs), matching training ``memory_chunk_size``.
        """
        was_training = self.training
        self.eval()

        device = input_ids.device
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )

        generated = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(generated)

        finished = torch.zeros(generated.size(0), dtype=torch.bool, device=device)
        batch_size = generated.size(0)

        prompt_len = generated.size(1)
        if prompt_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Prompt length {prompt_len} exceeds "
                f"max_position_embeddings={self.config.max_position_embeddings}."
            )
        if prompt_len + max_new_tokens > self.config.max_position_embeddings:
            raise ValueError(
                f"prompt_len+max_new_tokens="
                f"{prompt_len + max_new_tokens} exceeds "
                f"max_position_embeddings={self.config.max_position_embeddings}."
            )

        write_interval = self._memory_write_interval()
        past_key_values = None
        memory_states = None
        mamba_caches = None
        write_buffers: list[MemoryWriteBuffer | None] | None = None
        past_seen_tokens = 0
        tokens_in_write_buffer = 0
        out = None

        try:
            # Chunked prefill so memory writes match training chunk size.
            for start in range(0, prompt_len, write_interval):
                end = min(start + write_interval, prompt_len)
                chunk = generated[:, start:end]
                chunk_mask = attention_mask[:, :end]
                if chunk_mask.size(1) > self.config.window_size:
                    chunk_mask = chunk_mask[:, -self.config.window_size :]
                chunk_pos = (
                    torch.arange(start, end, dtype=torch.long, device=device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                # Flush write at end of each prefill chunk.
                out = self.forward(
                    input_ids=chunk,
                    attention_mask=chunk_mask,
                    position_ids=chunk_pos,
                    past_key_values=past_key_values,
                    memory_states=memory_states,
                    mamba_caches=mamba_caches,
                    write_buffers=write_buffers,
                    past_seen_tokens=past_seen_tokens,
                    use_cache=True,
                    skip_memory_write=False,
                )
                past_key_values = out.past_key_values
                memory_states = out.memory_states
                mamba_caches = out.mamba_caches
                write_buffers = out.write_buffers
                past_seen_tokens = end
                tokens_in_write_buffer = 0

            assert out is not None

            for _step in range(max_new_tokens):
                logits = out.logits[:, -1, :]
                if do_sample:
                    next_token_logits = logits / max(temperature, 1e-8)
                    if top_k is not None:
                        next_token_logits = _top_k_filter(next_token_logits, top_k)
                    if top_p is not None:
                        next_token_logits = _top_p_filter(next_token_logits, top_p)
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )

                generated = torch.cat([generated, next_token], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, (~finished).long().unsqueeze(-1)], dim=1
                )
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if finished.all():
                    break

                active = ~finished
                if not active.any():
                    break

                step_position_ids = torch.full(
                    (batch_size, 1),
                    past_seen_tokens,
                    dtype=torch.long,
                    device=device,
                )
                step_attn_mask = attention_mask
                if step_attn_mask.size(1) > self.config.window_size:
                    step_attn_mask = step_attn_mask[:, -self.config.window_size :]

                step_input = next_token.clone()
                step_input[~active] = self.config.pad_token_id
                tokens_in_write_buffer += 1
                do_memory_write = tokens_in_write_buffer >= write_interval

                out = self.forward(
                    input_ids=step_input,
                    attention_mask=step_attn_mask,
                    position_ids=step_position_ids,
                    past_key_values=past_key_values,
                    memory_states=memory_states,
                    mamba_caches=mamba_caches,
                    write_buffers=write_buffers,
                    past_seen_tokens=past_seen_tokens,
                    use_cache=True,
                    skip_memory_write=not do_memory_write,
                    active_batch_mask=active,
                )
                past_key_values = out.past_key_values
                memory_states = out.memory_states
                mamba_caches = out.mamba_caches
                write_buffers = out.write_buffers
                past_seen_tokens += 1
                if do_memory_write:
                    tokens_in_write_buffer = 0

            # Flush any partial decode write buffer so pending tokens are stored.
            if write_buffers is not None and any(b is not None for b in write_buffers):
                memory_states, write_buffers = self._flush_memory_write_buffers(
                    memory_states, write_buffers
                )
        finally:
            self.train(was_training)

        return generated


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_test3_null_baseline_config(
    hybrid_config: HybridMambaMoEConfig,
    tolerance: float = 0.02,
) -> HybridMambaMoEConfig:
    """
    Test 3 null hypothesis: larger Mamba state, no explicit memory banks.
    Binary-searches mamba_state_size (then mamba_expand) to match param count.
    Raises if a match within ``tolerance`` cannot be found.
    """
    target = count_trainable_params(HybridForCausalLM(hybrid_config))
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
