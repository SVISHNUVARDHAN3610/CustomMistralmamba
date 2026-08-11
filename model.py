"""
model.py

Contains two model families:

1. MixtralConfig / MixtralForCausalLM -- sliding-window-GQA + Top-2 MoE
   baseline (control for ablations).

2. HybridMambaMoEConfig / HybridForCausalLM -- Hybrid Mamba-MoE with Dual
   Memory (research/research.md v2.0):
       - Sliding window GQA branch (reuses SlidingWindowGQA)
       - Mamba selective-SSM branch (MambaBlock) with fused CUDA selective
         scan when `mamba-ssm` is installed (falls back to checkpointed
         sequential PyTorch scan), optional Hillis-Steele parallel scan, and
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
import warnings
from collections.abc import Callable
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

        return topk_weights, topk_indices, aux_loss, z_loss, logits


def expert_specialization_loss(
    expert_out: Tensor,
    logits: Tensor,
    var_beta: float,
) -> Tensor:
    """Orthogonality + routing variance loss (Guo et al., NeurIPS 2025)."""
    top_k = expert_out.size(1)
    cos_terms: list[Tensor] = []
    for i in range(top_k):
        for j in range(i + 1, top_k):
            cos_terms.append(
                F.cosine_similarity(expert_out[:, i], expert_out[:, j], dim=-1).abs()
            )
    if cos_terms:
        l_ortho = torch.stack(cos_terms, dim=0).mean()
    else:
        l_ortho = torch.tensor(0.0, device=expert_out.device, dtype=expert_out.dtype)
    full_probs = F.softmax(logits, dim=-1)
    var_loss = -full_probs.var(dim=-1, unbiased=False).mean()
    return l_ortho + var_beta * var_loss


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

    `use_grouped_moe_dispatch=True` sorts token assignments by expert and
    uses stacked weight tensors for fewer kernel launches (same math).
    """

    def __init__(
        self,
        router: nn.Module,
        experts: nn.ModuleList,
        capacity_factor: float | None = None,
        use_grouped_moe_dispatch: bool = True,
        use_grouped_gemm: bool = False,
    ):
        super().__init__()
        self.router = router
        self.experts = experts
        self.num_experts = len(experts)
        self.capacity_factor = capacity_factor
        self.use_grouped_moe_dispatch = use_grouped_moe_dispatch
        self.use_grouped_gemm = use_grouped_gemm

    @staticmethod
    def _swiglu_forward(
        x: Tensor, w_gate: Tensor, w_up: Tensor, w_down: Tensor
    ) -> Tensor:
        gate = F.linear(x, w_gate)
        up = F.linear(x, w_up)
        return F.linear(F.silu(gate) * up, w_down)

    def _stack_expert_weights(
        self,
    ) -> tuple[Tensor, Tensor, Tensor]:
        w_gate = torch.stack([e.w_gate.weight for e in self.experts], dim=0)
        w_up = torch.stack([e.w_up.weight for e in self.experts], dim=0)
        w_down = torch.stack([e.w_down.weight for e in self.experts], dim=0)
        return w_gate, w_up, w_down

    def _apply_capacity(
        self,
        row_indices: Tensor,
        k_indices: Tensor,
        capacity: int | None,
    ) -> tuple[Tensor, Tensor]:
        if capacity is None or row_indices.numel() <= capacity:
            return row_indices, k_indices
        if self.training:
            perm = torch.randperm(row_indices.numel(), device=row_indices.device)[
                :capacity
            ]
            return row_indices[perm], k_indices[perm]
        return row_indices[:capacity], k_indices[:capacity]

    def _forward_grouped(
        self,
        x_flat: Tensor,
        topk_weights: Tensor,
        topk_indices: Tensor,
        capacity: int | None,
        compute_expert_loss: bool,
        expert_var_beta: float,
        logits: Tensor,
    ) -> tuple[Tensor, Tensor]:
        num_tokens = x_flat.size(0)
        top_k = topk_indices.size(1)
        moe_output = torch.zeros_like(x_flat)
        applied_weights = torch.zeros(
            num_tokens, device=x_flat.device, dtype=x_flat.dtype
        )
        expert_out: Tensor | None = None
        if compute_expert_loss and self.training:
            expert_out = torch.zeros(
                num_tokens,
                top_k,
                x_flat.size(-1),
                device=x_flat.device,
                dtype=x_flat.dtype,
            )

        flat_expert = topk_indices.reshape(-1)
        flat_token = torch.arange(num_tokens, device=x_flat.device).repeat_interleave(
            top_k
        )
        flat_k = torch.arange(top_k, device=x_flat.device).repeat(num_tokens)

        sort_order = flat_expert.argsort()
        sorted_expert = flat_expert[sort_order]
        sorted_token = flat_token[sort_order]
        sorted_k = flat_k[sort_order]

        w_gate, w_up, w_down = self._stack_expert_weights()

        for expert_idx in range(self.num_experts):
            mask = sorted_expert == expert_idx
            if not mask.any():
                continue
            idx = torch.where(mask)[0]
            row_indices = sorted_token[idx]
            k_indices = sorted_k[idx]
            row_indices, k_indices = self._apply_capacity(
                row_indices, k_indices, capacity
            )
            expert_inputs = x_flat[row_indices]
            expert_outputs = self._swiglu_forward(
                expert_inputs, w_gate[expert_idx], w_up[expert_idx], w_down[expert_idx]
            )
            if expert_out is not None:
                expert_out[row_indices, k_indices] = expert_outputs
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

        expert_loss = torch.tensor(0.0, device=x_flat.device, dtype=x_flat.dtype)
        if expert_out is not None:
            expert_loss = expert_specialization_loss(
                expert_out, logits, var_beta=expert_var_beta
            )
        return moe_output, expert_loss

    @staticmethod
    def _grouped_mm_available() -> bool:
        return hasattr(torch, "_grouped_mm")

    def _forward_grouped_gemm(
        self,
        x_flat: Tensor,
        topk_weights: Tensor,
        topk_indices: Tensor,
        capacity: int | None,
        compute_expert_loss: bool,
        expert_var_beta: float,
        logits: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Grouped-GEMM MoE dispatch when torch._grouped_mm is available."""
        if not self._grouped_mm_available() or capacity is not None:
            return self._forward_grouped(
                x_flat,
                topk_weights,
                topk_indices,
                capacity,
                compute_expert_loss,
                expert_var_beta,
                logits,
            )

        num_tokens = x_flat.size(0)
        top_k = topk_indices.size(1)
        moe_output = torch.zeros_like(x_flat)
        applied_weights = torch.zeros(
            num_tokens, device=x_flat.device, dtype=x_flat.dtype
        )
        expert_out: Tensor | None = None
        if compute_expert_loss and self.training:
            expert_out = torch.zeros(
                num_tokens,
                top_k,
                x_flat.size(-1),
                device=x_flat.device,
                dtype=x_flat.dtype,
            )

        flat_expert = topk_indices.reshape(-1)
        flat_token = torch.arange(num_tokens, device=x_flat.device).repeat_interleave(
            top_k
        )
        flat_k = torch.arange(top_k, device=x_flat.device).repeat(num_tokens)
        sort_order = flat_expert.argsort()
        sorted_expert = flat_expert[sort_order]
        sorted_token = flat_token[sort_order]
        sorted_k = flat_k[sort_order]

        if sorted_token.numel() == 0:
            expert_loss = torch.tensor(0.0, device=x_flat.device, dtype=x_flat.dtype)
            return moe_output, expert_loss

        sorted_inputs = x_flat[sorted_token]
        counts = torch.bincount(sorted_expert, minlength=self.num_experts)
        offs = torch.zeros(self.num_experts + 1, dtype=torch.long, device=x_flat.device)
        offs[1:] = counts.cumsum(0)

        w_gate, w_up, w_down = self._stack_expert_weights()
        grouped_mm = torch._grouped_mm
        try:
            gate = grouped_mm(sorted_inputs, w_gate.transpose(1, 2), offs)
            up = grouped_mm(sorted_inputs, w_up.transpose(1, 2), offs)
            hidden = F.silu(gate) * up
            expert_outputs = grouped_mm(hidden, w_down.transpose(1, 2), offs)
        except (RuntimeError, TypeError):
            return self._forward_grouped(
                x_flat,
                topk_weights,
                topk_indices,
                capacity,
                compute_expert_loss,
                expert_var_beta,
                logits,
            )

        if expert_out is not None:
            expert_out[sorted_token, sorted_k] = expert_outputs
        gating_scale = topk_weights[sorted_token, sorted_k].unsqueeze(-1)
        moe_output.index_add_(0, sorted_token, expert_outputs * gating_scale)
        applied_weights.index_add_(
            0, sorted_token, gating_scale.squeeze(-1).to(applied_weights.dtype)
        )

        expert_loss = torch.tensor(0.0, device=x_flat.device, dtype=x_flat.dtype)
        if expert_out is not None:
            expert_loss = expert_specialization_loss(
                expert_out, logits, var_beta=expert_var_beta
            )
        return moe_output, expert_loss

    def _forward_loop(
        self,
        x_flat: Tensor,
        topk_weights: Tensor,
        topk_indices: Tensor,
        capacity: int | None,
        compute_expert_loss: bool,
        expert_var_beta: float,
        logits: Tensor,
    ) -> tuple[Tensor, Tensor]:
        num_tokens = x_flat.size(0)
        top_k = topk_indices.size(1)
        moe_output = torch.zeros_like(x_flat)
        applied_weights = torch.zeros(
            num_tokens, device=x_flat.device, dtype=x_flat.dtype
        )
        expert_out: Tensor | None = None
        if compute_expert_loss and self.training:
            expert_out = torch.zeros(
                num_tokens,
                top_k,
                x_flat.size(-1),
                device=x_flat.device,
                dtype=x_flat.dtype,
            )

        for expert_idx in range(self.num_experts):
            token_mask = topk_indices == expert_idx
            if not token_mask.any():
                continue

            row_indices, k_indices = torch.where(token_mask)
            row_indices, k_indices = self._apply_capacity(
                row_indices, k_indices, capacity
            )

            expert_inputs = x_flat[row_indices]
            expert_outputs = self.experts[expert_idx](expert_inputs)
            if expert_out is not None:
                expert_out[row_indices, k_indices] = expert_outputs

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

        expert_loss = torch.tensor(0.0, device=x_flat.device, dtype=x_flat.dtype)
        if expert_out is not None:
            expert_loss = expert_specialization_loss(
                expert_out, logits, var_beta=expert_var_beta
            )
        return moe_output, expert_loss

    def forward(
        self,
        x: torch.Tensor,
        compute_expert_loss: bool = False,
        expert_var_beta: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        num_tokens = x_flat.size(0)

        topk_weights, topk_indices, aux_loss, z_loss, logits = self.router(x_flat)

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

        dispatch_fn = (
            self._forward_grouped_gemm
            if self.use_grouped_gemm
            else self._forward_grouped
            if self.use_grouped_moe_dispatch
            else self._forward_loop
        )
        moe_output, expert_loss = dispatch_fn(
            x_flat,
            topk_weights,
            topk_indices,
            capacity,
            compute_expert_loss,
            expert_var_beta,
            logits,
        )

        return moe_output.reshape(*orig_shape), aux_loss, z_loss, expert_loss


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
        moe_out, aux_loss, z_loss, _ = self.moe_block(moe_in)
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
    # Use mamba-ssm fused CUDA selective_scan when available (CUDA, no padding).
    use_fused_mamba_scan: bool = True
    # PyTorch fallback scan dispatch (when fused path unavailable).
    parallel_scan_fallback_max_len: int = 4096
    blocked_scan_chunk_size: int = 256
    blocked_scan_min_len: int = 4096
    sequential_scan_min_len: int = 65536
    gradient_checkpointing: bool = False
    mamba_internal_checkpoint: bool = True
    debug_state_checks: bool = False
    use_grouped_moe_dispatch: bool = True
    use_grouped_gemm: bool = False
    decode_write_fast_threshold: int = 4
    use_torch_compile: bool = False
    torch_compile_mode: str = "default"
    use_cuda_graph: bool = False

    # Chunked training: stream CE per chunk to avoid materializing [B, L, V].
    stream_chunked_ce_loss: bool = True
    return_logits: bool = True

    # Auxiliary training losses (see loss-definitions.md). Training-only.
    use_auxiliary_losses: bool = True
    lambda_recon: float = 0.08
    lambda_assoc: float = 1.2e-4
    assoc_warmup_fraction: float = 0.05
    assoc_sample_count: int = 24
    lambda_gate: float = 1e-3
    gate_entropy_eps: float = 1e-6
    lambda_read: float = 5e-3
    read_util_min_fraction: float = 0.15
    lambda_fusion: float = 8e-3
    lambda_expert: float = 2e-3
    expert_warmup_fraction: float = 0.10
    expert_var_beta: float = 0.5
    lambda_ssm: float = 1e-5
    lambda_slot: float = 3e-3
    slot_similarity_margin: float = 0.3
    slot_cross_bank_alpha: float = 0.1
    recon_decoder_heads: int = 2


MambaCache = tuple[Tensor, Tensor]  # (conv_state, ssm_state)


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
    def zeros(device: torch.device, dtype: torch.dtype) -> "HybridLayerAuxLosses":
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


def write_gate_entropy_loss(gate: Tensor, eps: float = 1e-6) -> Tensor:
    ent = -(gate * torch.log(gate + eps) + (1.0 - gate) * torch.log(1.0 - gate + eps))
    return -ent.mean()


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
    def _intra(mem: Tensor) -> Tensor:
        mem_norm = mem / mem.norm(dim=-1, keepdim=True).clamp(min=eps)
        sim = torch.matmul(mem_norm, mem_norm.transpose(-1, -2))
        m = sim.size(-1)
        mask = ~torch.eye(m, device=sim.device, dtype=torch.bool)
        excess = torch.relu(sim - margin) ** 2
        return excess.masked_select(mask.unsqueeze(0)).mean()

    intra = _intra(attn_mem) + _intra(state_mem)
    a_norm = attn_mem / attn_mem.norm(dim=-1, keepdim=True).clamp(min=eps)
    s_norm = state_mem / state_mem.norm(dim=-1, keepdim=True).clamp(min=eps)
    cross = (a_norm * s_norm).sum(dim=-1).abs().mean()
    return intra + cross_alpha * cross


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
    bank: "CompressiveMemoryBank",
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


def _assert_right_padded_attention_mask(
    attention_mask: Tensor, debug_state_checks: bool
) -> None:
    """Valid tokens must form a left prefix (right-padding), not interior holes."""
    if not debug_state_checks:
        return
    valid = attention_mask.bool()
    for b in range(valid.size(0)):
        row = valid[b]
        n_valid = int(row.sum().item())
        if n_valid == 0:
            continue
        prefix = row[:n_valid]
        if not prefix.all() or row[n_valid:].any():
            raise ValueError(
                "Mamba prefill cache init requires right-padded attention_mask "
                "(valid tokens form a left prefix)."
            )


def _materialize_write_buffer(
    write_buffer: "MemoryWriteBuffer | None",
) -> tuple[Tensor | None, Tensor | None]:
    """Materialize pre-allocated write buffer into tensors for memory write."""
    if write_buffer is None or write_buffer.filled == 0:
        return None, None
    return write_buffer.materialize()


def _write_buffer_token_len(
    write_buffer: "MemoryWriteBuffer | None",
) -> int:
    if write_buffer is None:
        return 0
    return write_buffer.token_len()


class MemoryWriteBuffer:
    """Pre-allocated buffer for chunked memory writes (amortized O(k) append)."""

    __slots__ = (
        "attn_buf",
        "batch_size",
        "capacity",
        "filled",
        "hidden_size",
        "mamba_buf",
    )

    def __init__(
        self,
        batch_size: int,
        hidden_size: int,
        capacity: int = 512,
    ) -> None:
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.capacity = max(1, capacity)
        self.filled = 0
        self.attn_buf: Tensor | None = None
        self.mamba_buf: Tensor | None = None

    def _ensure_capacity(
        self, add_tokens: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        needed = self.filled + add_tokens
        if self.attn_buf is None:
            cap = max(self.capacity, needed)
            self.attn_buf = torch.zeros(
                self.batch_size, cap, self.hidden_size, device=device, dtype=dtype
            )
            self.mamba_buf = torch.zeros(
                self.batch_size, cap, self.hidden_size, device=device, dtype=dtype
            )
            self.capacity = cap
            return
        if needed > self.capacity:
            new_cap = max(self.capacity * 2, needed)
            assert self.mamba_buf is not None
            new_attn = torch.zeros(
                self.batch_size, new_cap, self.hidden_size, device=device, dtype=dtype
            )
            new_mamba = torch.zeros(
                self.batch_size, new_cap, self.hidden_size, device=device, dtype=dtype
            )
            new_attn[:, : self.filled] = self.attn_buf[:, : self.filled]
            new_mamba[:, : self.filled] = self.mamba_buf[:, : self.filled]
            self.attn_buf = new_attn
            self.mamba_buf = new_mamba
            self.capacity = new_cap

    def append(self, attn: Tensor, mamba: Tensor) -> None:
        add = attn.size(1)
        self._ensure_capacity(add, attn.device, attn.dtype)
        assert self.attn_buf is not None and self.mamba_buf is not None
        self.attn_buf[:, self.filled : self.filled + add] = attn
        self.mamba_buf[:, self.filled : self.filled + add] = mamba
        self.filled += add

    def append_single_token(self, attn: Tensor, mamba: Tensor) -> None:
        """Fast path for decode: append one token without realloc after warm-up."""
        if self.attn_buf is None or self.filled >= self.capacity:
            self._ensure_capacity(1, attn.device, attn.dtype)
        assert self.attn_buf is not None and self.mamba_buf is not None
        self.attn_buf[:, self.filled : self.filled + 1] = attn
        self.mamba_buf[:, self.filled : self.filled + 1] = mamba
        self.filled += 1

    def materialize(self) -> tuple[Tensor, Tensor]:
        assert self.attn_buf is not None and self.mamba_buf is not None
        return self.attn_buf[:, : self.filled], self.mamba_buf[:, : self.filled]

    def token_len(self) -> int:
        return self.filled


def _validate_hybrid_cache_states(
    config: HybridMambaMoEConfig,
    num_layers: int,
    batch_size: int,
    memory_states: list | None,
    mamba_caches: list | None,
    write_buffers: list | None,
    past_key_values: list | None,
    active_batch_mask: Tensor | None,
) -> None:
    if not config.debug_state_checks:
        return
    if memory_states is not None:
        assert len(memory_states) == num_layers
    if mamba_caches is not None:
        assert len(mamba_caches) == num_layers
    if write_buffers is not None:
        assert len(write_buffers) == num_layers
        for buf in write_buffers:
            if buf is not None:
                assert buf.filled >= 0
                if buf.attn_buf is not None:
                    assert buf.attn_buf.size(0) == batch_size
    if past_key_values is not None:
        assert len(past_key_values) == num_layers
    if active_batch_mask is not None:
        assert active_batch_mask.dtype == torch.bool
        assert active_batch_mask.size(0) == batch_size


_SELECTIVE_SCAN_FN: Callable[..., Tensor] | None = None
_SELECTIVE_SCAN_PROBE_DONE = False
_FUSED_SCAN_WARNED = False


def fused_mamba_scan_available() -> bool:
    """True when mamba-ssm fused selective_scan CUDA kernels can be imported."""
    return _load_selective_scan_fn() is not None


def log_mamba_backend(config: HybridMambaMoEConfig | None = None) -> str:
    """Log which Mamba selective-scan backend will be used. Returns summary string."""
    fused = fused_mamba_scan_available()
    if config is None:
        config = HybridMambaMoEConfig()
    if fused and config.use_fused_mamba_scan:
        msg = "Mamba backend: fused CUDA selective_scan (mamba-ssm)"
    elif config.use_parallel_scan:
        msg = "Mamba backend: Hillis-Steele parallel scan (explicit use_parallel_scan=True)"
    else:
        msg = (
            "Mamba backend: PyTorch fallback "
            f"(parallel L<={config.parallel_scan_fallback_max_len}, "
            f"blocked {config.blocked_scan_min_len}<L<={config.sequential_scan_min_len}, "
            f"sequential L>{config.sequential_scan_min_len})"
        )
        if torch.cuda.is_available() and not fused:
            warnings.warn(
                "CUDA is available but mamba-ssm is not installed; training will use "
                "slow PyTorch scan fallbacks. Install mamba-ssm for production runs.",
                stacklevel=2,
            )
    return msg


def probe_mamba_scan_timing(
    config: HybridMambaMoEConfig | None = None,
    batch_size: int = 2,
    seq_len: int = 512,
    device: torch.device | None = None,
) -> str:
    """One-step timing probe: fused vs PyTorch fallback selective scan."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config is None:
        config = HybridMambaMoEConfig(
            hidden_size=128,
            mamba_state_size=8,
            mamba_expand=2,
        )
    if not torch.cuda.is_available():
        return "mamba_scan_probe: skipped (CPU only)"

    import time

    block = MambaBlock(
        hidden_size=config.hidden_size,
        state_size=config.mamba_state_size,
        expand=config.mamba_expand,
        use_fused_scan=False,
    ).to(device)
    x = torch.randn(batch_size, seq_len, config.hidden_size, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        block(x)
    torch.cuda.synchronize()
    fallback_ms = (time.perf_counter() - t0) * 1000

    if fused_mamba_scan_available():
        block_fused = MambaBlock(
            hidden_size=config.hidden_size,
            state_size=config.mamba_state_size,
            expand=config.mamba_expand,
            use_fused_scan=True,
        ).to(device)
        block_fused.load_state_dict(block.state_dict())
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            block_fused(x)
        torch.cuda.synchronize()
        fused_ms = (time.perf_counter() - t0) * 1000
        return (
            f"mamba_scan_probe: fused={fused_ms:.1f}ms fallback={fallback_ms:.1f}ms "
            f"speedup={fallback_ms / max(fused_ms, 1e-6):.2f}x"
        )
    return f"mamba_scan_probe: fallback={fallback_ms:.1f}ms (mamba-ssm not installed)"


def _compute_batch_has_padding(attention_mask: Tensor | None, seq_len: int) -> bool:
    """Single sync point per forward for padding detection."""
    if attention_mask is None:
        return False
    if attention_mask.dim() != 2:
        return True
    if attention_mask.size(1) < seq_len:
        return True
    return not attention_mask[:, -seq_len:].all().item()


def _load_selective_scan_fn() -> Callable[..., Tensor] | None:
    """Lazy import of mamba-ssm selective_scan_fn (optional dependency)."""
    global _SELECTIVE_SCAN_FN, _SELECTIVE_SCAN_PROBE_DONE
    if _SELECTIVE_SCAN_PROBE_DONE:
        return _SELECTIVE_SCAN_FN
    _SELECTIVE_SCAN_PROBE_DONE = True
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        _SELECTIVE_SCAN_FN = selective_scan_fn
    except ImportError:
        _SELECTIVE_SCAN_FN = None
    return _SELECTIVE_SCAN_FN


def _attention_mask_has_padding(attention_mask: Tensor | None, seq_len: int) -> bool:
    if attention_mask is None:
        return False
    if attention_mask.dim() != 2:
        return True
    if attention_mask.size(1) < seq_len:
        return True
    return not attention_mask[:, -seq_len:].all().item()


class MambaBlock(nn.Module):
    """
    Selective SSM (Mamba / S6).

    Prefill/training uses fused CUDA selective_scan from `mamba-ssm` when
    available (`use_fused_scan=True`, CUDA, no padding). Otherwise falls back
    to length-aware PyTorch scans: parallel (short L), blocked (medium L),
    sequential+checkpoint (very long L). Optional Hillis-Steele parallel scan
    (`use_parallel_scan=True`) bypasses fused CUDA.
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
        use_fused_scan: bool = True,
        parallel_scan_fallback_max_len: int = 4096,
        blocked_scan_chunk_size: int = 256,
        blocked_scan_min_len: int = 4096,
        sequential_scan_min_len: int = 65536,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.d_inner = expand * hidden_size
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(hidden_size / 16)
        self.use_parallel_scan = use_parallel_scan
        self.use_fused_scan = use_fused_scan
        self.parallel_scan_fallback_max_len = parallel_scan_fallback_max_len
        self.blocked_scan_chunk_size = blocked_scan_chunk_size
        self.blocked_scan_min_len = blocked_scan_min_len
        self.sequential_scan_min_len = sequential_scan_min_len

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
        debug_state_checks: bool = False,
        batch_has_padding: bool | None = None,
        mamba_internal_checkpoint: bool = True,
        layer_checkpointing_active: bool = False,
    ) -> tuple[Tensor, MambaCache | None, Tensor | None]:
        """
        x: [B, L, hidden_size]
        If use_cache and L==1 and cache is provided, runs a single decode step.
        Otherwise runs full-sequence prefill (parallel scan); when use_cache,
        returns updated (conv_state, ssm_state) for subsequent steps.
        """
        _, seq_len, _ = x.shape

        if use_cache and cache is not None and seq_len == 1:
            out, cache_out = self.step(
                x, cache[0], cache[1], active_batch_mask=active_batch_mask
            )
            return out, cache_out, cache[1]

        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        x_conv = x_in.transpose(1, 2)
        if use_cache:
            # Keep last conv_kernel *valid* tokens as the rolling conv buffer.
            if attention_mask is not None and attention_mask.dim() == 2:
                token_mask = attention_mask[:, -seq_len:]
                _assert_right_padded_attention_mask(token_mask, debug_state_checks)
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
            return_final_state=True,
            use_parallel_scan=self.use_parallel_scan,
            use_fused_scan=self.use_fused_scan,
            training=self.training,
            attention_mask=attention_mask,
            batch_has_padding=batch_has_padding,
            parallel_scan_fallback_max_len=self.parallel_scan_fallback_max_len,
            blocked_scan_chunk_size=self.blocked_scan_chunk_size,
            blocked_scan_min_len=self.blocked_scan_min_len,
            sequential_scan_min_len=self.sequential_scan_min_len,
            mamba_internal_checkpoint=mamba_internal_checkpoint,
            layer_checkpointing_active=layer_checkpointing_active,
        )
        y = y * F.silu(z)
        out = self.out_proj(y)

        new_cache: MambaCache | None = None
        if use_cache:
            assert conv_state is not None and ssm_state is not None
            new_cache = (conv_state, ssm_state)
        return out, new_cache, ssm_state

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

    @staticmethod
    def _blocked_associative_scan(
        delta_a: Tensor, delta_b_u: Tensor, block_size: int
    ) -> Tensor:
        """Vectorized scan within blocks; carry state between blocks."""
        _, seq_len, _, _ = delta_a.shape
        state = torch.zeros_like(delta_b_u[:, 0])
        outputs: list[Tensor] = []
        for start in range(0, seq_len, block_size):
            end = min(start + block_size, seq_len)
            block_a = delta_a[:, start:end]
            block_b = delta_b_u[:, start:end]
            if start > 0:
                block_b = block_b.clone()
                block_b[:, 0] = block_a[:, 0] * state + block_b[:, 0]
            block_out = MambaBlock._parallel_associative_scan(block_a, block_b)
            outputs.append(block_out)
            state = block_out[:, -1]
        return torch.cat(outputs, dim=1)

    @classmethod
    def _run_associative_scan(
        cls,
        delta_a: Tensor,
        delta_b_u: Tensor,
        *,
        use_parallel_scan: bool,
        training: bool,
        seq_len: int,
        parallel_scan_fallback_max_len: int,
        blocked_scan_chunk_size: int,
        blocked_scan_min_len: int,
        sequential_scan_min_len: int,
        mamba_internal_checkpoint: bool = True,
        layer_checkpointing_active: bool = False,
    ) -> Tensor:
        use_scan_checkpoint = (
            training and mamba_internal_checkpoint and not layer_checkpointing_active
        )
        if use_parallel_scan or seq_len <= parallel_scan_fallback_max_len:
            return cls._parallel_associative_scan(delta_a, delta_b_u)
        if seq_len <= sequential_scan_min_len:
            scan_fn = lambda a, b: cls._blocked_associative_scan(
                a, b, blocked_scan_chunk_size
            )
            if use_scan_checkpoint:
                return checkpoint(scan_fn, delta_a, delta_b_u, use_reentrant=False)
            return scan_fn(delta_a, delta_b_u)
        if use_scan_checkpoint:
            return checkpoint(
                cls._sequential_associative_scan,
                delta_a,
                delta_b_u,
                use_reentrant=False,
            )
        return cls._sequential_associative_scan(delta_a, delta_b_u)

    @staticmethod
    def _fused_selective_scan(
        u: Tensor,
        dt: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
        return_final_state: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """
        mamba-ssm fused CUDA selective scan.
        u, dt: [B, L, d_inner]; A: [d_inner, n]; B, C: [B, L, n]; D: [d_inner]
        """
        selective_scan_fn = _load_selective_scan_fn()
        if selective_scan_fn is None:
            raise RuntimeError("mamba-ssm selective_scan_fn is not available.")

        input_dtype = u.dtype
        u_t = u.transpose(1, 2).contiguous()
        dt_t = dt.transpose(1, 2).contiguous()
        b_t = B.transpose(1, 2).contiguous()
        c_t = C.transpose(1, 2).contiguous()

        result = selective_scan_fn(
            u_t.float(),
            dt_t.float(),
            A.float(),
            b_t.float(),
            c_t.float(),
            D.float(),
            delta_bias=None,
            delta_softplus=False,
            return_last_state=return_final_state,
        )
        if return_final_state:
            y_t, final_state = result
            return y_t.transpose(1, 2).to(input_dtype), final_state
        y_t = result
        return y_t.transpose(1, 2).to(input_dtype), None

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
        use_fused_scan: bool = True,
        training: bool = False,
        attention_mask: Tensor | None = None,
        batch_has_padding: bool | None = None,
        parallel_scan_fallback_max_len: int = 4096,
        blocked_scan_chunk_size: int = 256,
        blocked_scan_min_len: int = 4096,
        sequential_scan_min_len: int = 65536,
        mamba_internal_checkpoint: bool = True,
        layer_checkpointing_active: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """
        u, dt: [B, L, d_inner]; A: [d_inner, n]; B, C: [B, L, n]; D: [d_inner]
        On pad positions (attention_mask==0), apply identity state transition
        so SSM state does not decay through padding.
        """
        global _FUSED_SCAN_WARNED
        seq_len = u.size(1)
        has_padding = (
            batch_has_padding
            if batch_has_padding is not None
            else _attention_mask_has_padding(attention_mask, seq_len)
        )
        can_use_fused = (
            use_fused_scan
            and not use_parallel_scan
            and u.is_cuda
            and not has_padding
            and _load_selective_scan_fn() is not None
        )
        if can_use_fused:
            try:
                return cls._fused_selective_scan(
                    u,
                    dt,
                    A,
                    B,
                    C,
                    D,
                    return_final_state=return_final_state,
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                if not _FUSED_SCAN_WARNED:
                    warnings.warn(
                        f"mamba-ssm fused selective_scan failed ({type(exc).__name__}: "
                        f"{exc}); falling back to PyTorch scan.",
                        stacklevel=2,
                    )
                    _FUSED_SCAN_WARNED = True

        input_dtype = u.dtype
        u_f = u.float()
        dt_f = dt.float()
        B_f = B.float()
        C_f = C.float()

        delta_a = torch.exp(dt_f.unsqueeze(-1) * A)  # [B, L, d_inner, n]
        delta_b_u = dt_f.unsqueeze(-1) * B_f.unsqueeze(2) * u_f.unsqueeze(-1)

        token_mask: Tensor | None = None
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("MambaBlock expects 2D attention_mask [B, L].")
            if attention_mask.size(1) < seq_len:
                raise ValueError(
                    f"attention_mask length {attention_mask.size(1)} < seq_len {seq_len}."
                )
            token_mask = attention_mask[:, -seq_len:].to(dtype=delta_a.dtype)
            m = token_mask.unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
            # Pad steps: h_t = 1 * h_{t-1} + 0
            delta_a = delta_a * m + (1.0 - m)
            delta_b_u = delta_b_u * m

        states = cls._run_associative_scan(
            delta_a,
            delta_b_u,
            use_parallel_scan=use_parallel_scan,
            training=training,
            seq_len=seq_len,
            parallel_scan_fallback_max_len=parallel_scan_fallback_max_len,
            blocked_scan_chunk_size=blocked_scan_chunk_size,
            blocked_scan_min_len=blocked_scan_min_len,
            sequential_scan_min_len=sequential_scan_min_len,
            mamba_internal_checkpoint=mamba_internal_checkpoint,
            layer_checkpointing_active=layer_checkpointing_active,
        )

        y = (states * C_f.unsqueeze(2)).sum(dim=-1)
        y = y + u_f * D.float()
        if token_mask is not None:
            y = y * token_mask.unsqueeze(-1)
        final_state = states[:, -1].contiguous() if return_final_state else None
        return y.to(input_dtype), final_state


class CompressiveMemoryBank(nn.Module):
    """
    Fixed-size (m slots) gated read/write memory bank.

    Multi-head scaled-dot attention over a small bank (m << L).
    """

    def __init__(
        self,
        hidden_size: int,
        memory_size: int = 64,
        num_heads: int = 8,
        recon_decoder_heads: int = 2,
        enable_aux_modules: bool = True,
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

        if enable_aux_modules:
            self.recon_decoder = MemoryReconstructionDecoder(
                hidden_size, recon_decoder_heads
            )
            self.assoc_key = nn.Linear(hidden_size, hidden_size, bias=False)
            self.assoc_val = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.recon_decoder = None
            self.assoc_key = None
            self.assoc_val = None

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
        return t.reshape(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _attend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        fast_path: bool = False,
    ) -> Tensor:
        """
        query: [B, Q, H], key/value: [B, K, H]
        key_padding_mask: [B, K] True = ignore
        """
        bsz, q_len, _ = query.shape
        k_len = key.size(1)
        if fast_path and q_len * k_len <= 256:
            q = self.q_proj(query).view(bsz, q_len, self.num_heads, self.head_dim)
            k = self.k_proj(key).view(bsz, k_len, self.num_heads, self.head_dim)
            v = self.v_proj(value).view(bsz, k_len, self.num_heads, self.head_dim)
            scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * self.scale
            if key_padding_mask is not None:
                scores = scores.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )
            attn = F.softmax(scores, dim=-1)
            out = torch.einsum("bhqk,bkhd->bqhd", attn, v)
            out = out.reshape(bsz, q_len, self.hidden_size)
            return self.out_proj(out)

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

    def read_query(self, query: Tensor, memory: Tensor) -> Tensor:
        return self._attend(query, memory, memory, key_padding_mask=None)

    def write(
        self,
        x: Tensor,
        memory: Tensor,
        attention_mask: Tensor | None = None,
        fast_path: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        kpm = self._key_padding_mask(attention_mask)
        batch_size = x.size(0)
        query = self.summary_query.unsqueeze(0).expand(batch_size, -1, -1)
        chunk_summary = self._attend(
            query, x, x, key_padding_mask=kpm, fast_path=fast_path
        )

        gate = torch.sigmoid(
            self.write_gate(torch.cat([memory, chunk_summary], dim=-1))
        )
        new_memory = gate * memory + (1.0 - gate) * self.write_update(chunk_summary)
        if kpm is not None:
            all_masked_rows = kpm.all(dim=-1)
            if all_masked_rows.any():
                new_memory = new_memory.clone()
                new_memory[all_masked_rows] = memory[all_masked_rows]
        return new_memory, gate, chunk_summary


def batched_dual_memory_read(
    bank_a: CompressiveMemoryBank,
    bank_b: CompressiveMemoryBank,
    x: Tensor,
    mem_a: Tensor,
    mem_b: Tensor,
) -> tuple[Tensor, Tensor]:
    """Single batched attention for both memory banks (stacked projections)."""
    bsz, seq_len, hidden = x.shape
    m_len = mem_a.size(1)
    num_heads = bank_a.num_heads
    head_dim = bank_a.head_dim
    scale = bank_a.scale

    queries = torch.cat([x, x], dim=0)
    keys = torch.cat([mem_a, mem_b], dim=0)
    values = keys
    bank_idx = torch.arange(2 * bsz, device=x.device) // bsz

    q_w = torch.stack([bank_a.q_proj.weight, bank_b.q_proj.weight], dim=0)
    k_w = torch.stack([bank_a.k_proj.weight, bank_b.k_proj.weight], dim=0)
    v_w = torch.stack([bank_a.v_proj.weight, bank_b.v_proj.weight], dim=0)
    out_w = torch.stack([bank_a.out_proj.weight, bank_b.out_proj.weight], dim=0)

    q = torch.bmm(queries, q_w[bank_idx].transpose(1, 2))
    k = torch.bmm(keys, k_w[bank_idx].transpose(1, 2))
    v = torch.bmm(values, v_w[bank_idx].transpose(1, 2))

    q = q.view(2 * bsz, seq_len, num_heads, head_dim)
    k = k.view(2 * bsz, m_len, num_heads, head_dim)
    v = v.view(2 * bsz, m_len, num_heads, head_dim)

    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    attn = F.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", attn, v)
    out = out.reshape(2 * bsz, seq_len, hidden)
    out = torch.bmm(out, out_w[bank_idx].transpose(1, 2))

    return out[:bsz], out[bsz:]


def _batched_memory_summarize(
    bank_a: CompressiveMemoryBank,
    bank_b: CompressiveMemoryBank,
    buf_a: Tensor,
    buf_b: Tensor,
    key_padding_mask: Tensor | None,
    fast_path: bool,
) -> tuple[Tensor, Tensor]:
    """Batched summary_query attention for dual-bank writes."""
    bsz = buf_a.size(0)
    m_len = bank_a.memory_size
    num_heads = bank_a.num_heads
    head_dim = bank_a.head_dim
    scale = bank_a.scale
    buf_len = buf_a.size(1)

    queries = torch.cat(
        [
            bank_a.summary_query.unsqueeze(0).expand(bsz, -1, -1),
            bank_b.summary_query.unsqueeze(0).expand(bsz, -1, -1),
        ],
        dim=0,
    )
    keys = torch.cat([buf_a, buf_b], dim=0)
    values = keys
    bank_idx = torch.arange(2 * bsz, device=buf_a.device) // bsz

    q_w = torch.stack([bank_a.q_proj.weight, bank_b.q_proj.weight], dim=0)
    k_w = torch.stack([bank_a.k_proj.weight, bank_b.k_proj.weight], dim=0)
    v_w = torch.stack([bank_a.v_proj.weight, bank_b.v_proj.weight], dim=0)
    out_w = torch.stack([bank_a.out_proj.weight, bank_b.out_proj.weight], dim=0)

    q = torch.bmm(queries, q_w[bank_idx].transpose(1, 2))
    k = torch.bmm(keys, k_w[bank_idx].transpose(1, 2))
    v = torch.bmm(values, v_w[bank_idx].transpose(1, 2))
    q = q.view(2 * bsz, m_len, num_heads, head_dim)
    k = k.view(2 * bsz, buf_len, num_heads, head_dim)
    v = v.view(2 * bsz, buf_len, num_heads, head_dim)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    if key_padding_mask is not None:
        kpm = torch.cat([key_padding_mask, key_padding_mask], dim=0)
        scores = scores.masked_fill(kpm.unsqueeze(1).unsqueeze(2), float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", attn, v)
    out = out.reshape(2 * bsz, m_len, bank_a.hidden_size)
    out = torch.bmm(out, out_w[bank_idx].transpose(1, 2))

    return out[:bsz], out[bsz:]


def batched_dual_memory_write(
    bank_a: CompressiveMemoryBank,
    bank_b: CompressiveMemoryBank,
    buf_attn: Tensor,
    buf_mamba: Tensor,
    mem_a: Tensor,
    mem_b: Tensor,
    attention_mask: Tensor | None = None,
    fast_path: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fused write for both memory banks (stacked projections)."""
    kpm: Tensor | None = None
    if attention_mask is not None:
        kpm = ~attention_mask.bool()

    a_summary, s_summary = _batched_memory_summarize(
        bank_a, bank_b, buf_attn, buf_mamba, kpm, fast_path
    )

    gate_w = torch.stack([bank_a.write_gate.weight, bank_b.write_gate.weight], dim=0)
    gate_b = torch.stack(
        [
            bank_a.write_gate.bias
            if bank_a.write_gate.bias is not None
            else torch.zeros(
                bank_a.hidden_size, device=buf_attn.device, dtype=buf_attn.dtype
            ),
            bank_b.write_gate.bias
            if bank_b.write_gate.bias is not None
            else torch.zeros(
                bank_b.hidden_size, device=buf_attn.device, dtype=buf_attn.dtype
            ),
        ],
        dim=0,
    )
    update_w = torch.stack(
        [bank_a.write_update.weight, bank_b.write_update.weight], dim=0
    )
    bsz = buf_attn.size(0)
    bank_idx = torch.arange(2 * bsz, device=buf_attn.device) // bsz

    mem_stacked = torch.cat([mem_a, mem_b], dim=0)
    summary_stacked = torch.cat([a_summary, s_summary], dim=0)
    gate_in = torch.cat([mem_stacked, summary_stacked], dim=-1)
    gate_logits = torch.bmm(gate_in, gate_w[bank_idx].transpose(1, 2)) + gate_b[
        bank_idx
    ].unsqueeze(1)
    gate = torch.sigmoid(gate_logits)
    updates = torch.bmm(summary_stacked, update_w[bank_idx].transpose(1, 2))
    new_mem_stacked = gate * mem_stacked + (1.0 - gate) * updates
    new_a = new_mem_stacked[:bsz]
    new_s = new_mem_stacked[bsz:]

    if kpm is not None:
        all_masked_rows = kpm.all(dim=-1)
        if all_masked_rows.any():
            new_a = new_a.clone()
            new_s = new_s.clone()
            new_a[all_masked_rows] = mem_a[all_masked_rows]
            new_s[all_masked_rows] = mem_b[all_masked_rows]

    a_gate = gate[:bsz]
    s_gate = gate[bsz:]
    return new_a, a_gate, a_summary, new_s, s_gate, s_summary


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
    training_step: int | None = None,
    max_training_steps: int | None = None,
    batch_has_padding: bool | None = None,
    layer_checkpointing_active: bool = False,
    decode_accumulate_only: bool = False,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    tuple[Tensor, Tensor] | None,
    HybridMemoryState | None,
    MambaCache | None,
    dict[str, Tensor],
    MemoryWriteBuffer | None,
    HybridLayerAuxLosses,
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
        training_step=training_step,
        max_training_steps=max_training_steps,
        batch_has_padding=batch_has_padding,
        layer_checkpointing_active=layer_checkpointing_active,
        decode_accumulate_only=decode_accumulate_only,
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
        self.config = config
        self.use_dual_memory = config.use_dual_memory
        self.use_auxiliary_losses = config.use_auxiliary_losses

        self.rmsnorm_in = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_block = SlidingWindowGQA(config)
        self.mamba_block = MambaBlock(
            hidden_size=config.hidden_size,
            state_size=config.mamba_state_size,
            conv_kernel=config.mamba_conv_kernel,
            expand=config.mamba_expand,
            dt_rank=config.mamba_dt_rank,
            use_parallel_scan=config.use_parallel_scan,
            use_fused_scan=config.use_fused_mamba_scan,
            parallel_scan_fallback_max_len=config.parallel_scan_fallback_max_len,
            blocked_scan_chunk_size=config.blocked_scan_chunk_size,
            blocked_scan_min_len=config.blocked_scan_min_len,
            sequential_scan_min_len=config.sequential_scan_min_len,
        )

        if self.use_dual_memory:
            enable_aux = config.use_auxiliary_losses
            self.attn_memory_bank = CompressiveMemoryBank(
                config.hidden_size,
                config.memory_size,
                config.memory_num_heads,
                recon_decoder_heads=config.recon_decoder_heads,
                enable_aux_modules=enable_aux,
            )
            self.state_memory_bank = CompressiveMemoryBank(
                config.hidden_size,
                config.memory_size,
                config.memory_num_heads,
                recon_decoder_heads=config.recon_decoder_heads,
                enable_aux_modules=enable_aux,
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
            router,
            experts,
            capacity_factor=config.capacity_factor,
            use_grouped_moe_dispatch=config.use_grouped_moe_dispatch,
            use_grouped_gemm=config.use_grouped_gemm,
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
        training_step: int | None = None,
        max_training_steps: int | None = None,
        batch_has_padding: bool | None = None,
        layer_checkpointing_active: bool = False,
        decode_accumulate_only: bool = False,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        tuple[Tensor, Tensor] | None,
        HybridMemoryState | None,
        MambaCache | None,
        dict[str, Tensor],
        MemoryWriteBuffer | None,
        HybridLayerAuxLosses,
    ]:
        residual = x
        x_norm = self.rmsnorm_in(x)
        seq_len = x.size(1)
        cfg = self.config
        layer_aux = HybridLayerAuxLosses.zeros(x.device, x.dtype)

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

            # Batched read into both branches (single stacked attention pass).
            a_read, s_read = batched_dual_memory_read(
                self.attn_memory_bank, self.state_memory_bank, x_norm, a_mem, s_mem
            )
            attn_input = self.attn_memory_combine(torch.cat([x_norm, a_read], dim=-1))
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
        mamba_out, new_mamba_cache, ssm_state = self.mamba_block(
            mamba_input,
            cache=mamba_cache,
            use_cache=use_cache,
            attention_mask=mamba_token_mask,
            active_batch_mask=active_batch_mask,
            debug_state_checks=cfg.debug_state_checks,
            batch_has_padding=batch_has_padding,
            mamba_internal_checkpoint=cfg.mamba_internal_checkpoint,
            layer_checkpointing_active=layer_checkpointing_active,
        )

        if self.use_dual_memory:
            assert memory_state is not None
            a_mem, s_mem = memory_state
            prev_a_mem = a_mem
            prev_s_mem = s_mem
            if hidden_mask is not None:
                attn_out = attn_out * hidden_mask
                mamba_out = mamba_out * hidden_mask

            # Accumulate raw branch outputs for chunk-aligned memory writes.
            buf_attn = attn_out
            buf_mamba = mamba_out
            skip_active_mask = (
                decode_accumulate_only
                and active_batch_mask is not None
                and active_batch_mask.all()
            )
            if active_batch_mask is not None and not skip_active_mask:
                active = active_batch_mask.to(dtype=buf_attn.dtype).view(-1, 1, 1)
                buf_attn = buf_attn * active
                buf_mamba = buf_mamba * active

            write_cap = self.config.memory_write_interval
            if write_cap is None:
                write_cap = self.config.memory_chunk_size
            if write_cap is None:
                write_cap = 512

            if write_buffer is None:
                new_buf = MemoryWriteBuffer(
                    x.size(0), cfg.hidden_size, capacity=write_cap
                )
            else:
                new_buf = write_buffer

            if decode_accumulate_only and seq_len == 1:
                new_buf.append_single_token(buf_attn, buf_mamba)
            else:
                new_buf.append(buf_attn, buf_mamba)

            if skip_memory_write:
                new_memory_state = memory_state
                new_write_buffer = new_buf
            else:
                buf_attn_cat, buf_mamba_cat = new_buf.materialize()
                # Build mask for buffered tokens: prior buffer assumed valid,
                # current tokens use token_attention_mask when present.
                buf_len = buf_attn_cat.size(1)
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

                write_fast = buf_len <= cfg.decode_write_fast_threshold
                (
                    new_a_mem,
                    a_write_gate,
                    a_summary,
                    new_s_mem,
                    s_write_gate,
                    s_summary,
                ) = batched_dual_memory_write(
                    self.attn_memory_bank,
                    self.state_memory_bank,
                    buf_attn_cat,
                    buf_mamba_cat,
                    a_mem,
                    s_mem,
                    attention_mask=write_mask,
                    fast_path=write_fast,
                )
                if active_batch_mask is not None and (~active_batch_mask).any():
                    inactive = ~active_batch_mask
                    new_a_mem = new_a_mem.clone()
                    new_s_mem = new_s_mem.clone()
                    new_a_mem[inactive] = prev_a_mem[inactive]
                    new_s_mem[inactive] = prev_s_mem[inactive]
                new_memory_state = (new_a_mem, new_s_mem)
                new_write_buffer = None
                gate_stats = {
                    "attn_write_gate_mean": a_write_gate.detach().mean(),
                    "state_write_gate_mean": s_write_gate.detach().mean(),
                }

                if self.training and self.use_auxiliary_losses:
                    attn_recon_out = self.attn_memory_bank.recon_decoder(
                        buf_attn_cat, a_summary
                    )
                    mamba_recon_out = self.state_memory_bank.recon_decoder(
                        buf_mamba_cat, s_summary
                    )
                    attn_recon = F.mse_loss(attn_recon_out, buf_attn_cat)
                    mamba_recon = F.mse_loss(mamba_recon_out, buf_mamba_cat)
                    attn_recon_tok = (
                        (buf_attn_cat - attn_recon_out).pow(2).mean(dim=-1).sqrt()
                    )
                    mamba_recon_tok = (
                        (buf_mamba_cat - mamba_recon_out).pow(2).mean(dim=-1).sqrt()
                    )
                    attn_assoc = associative_retrieval_loss(
                        self.attn_memory_bank,
                        buf_attn_cat,
                        new_a_mem,
                        attn_recon_tok,
                        cfg.assoc_sample_count,
                        write_mask,
                    )
                    mamba_assoc = associative_retrieval_loss(
                        self.state_memory_bank,
                        buf_mamba_cat,
                        new_s_mem,
                        mamba_recon_tok,
                        cfg.assoc_sample_count,
                        write_mask,
                    )
                    gate_loss = write_gate_entropy_loss(
                        a_write_gate, cfg.gate_entropy_eps
                    ) + write_gate_entropy_loss(s_write_gate, cfg.gate_entropy_eps)
                    slot_loss = memory_slot_diversity_loss(
                        new_a_mem,
                        new_s_mem,
                        cfg.slot_similarity_margin,
                        cfg.slot_cross_bank_alpha,
                    )
                    layer_aux = HybridLayerAuxLosses(
                        recon=(attn_recon + mamba_recon) / 2.0,
                        assoc=(attn_assoc + mamba_assoc) / 2.0,
                        gate=gate_loss / 2.0,
                        read=layer_aux.read,
                        fusion=layer_aux.fusion,
                        expert=layer_aux.expert,
                        ssm=layer_aux.ssm,
                        slot=slot_loss,
                    )

        fused, fusion_gate = self.fusion(attn_out, mamba_out)
        if hidden_mask is not None:
            fused = fused * hidden_mask
        x = residual + fused

        moe_in = self.rmsnorm_moe(x)
        if hidden_mask is not None:
            moe_in = moe_in * hidden_mask
        expert_scale = _expert_loss_schedule(
            training_step, max_training_steps, cfg.expert_warmup_fraction
        )
        moe_out, aux_loss, z_loss, expert_loss = self.moe_block(
            moe_in,
            compute_expert_loss=(
                self.training and self.use_auxiliary_losses and expert_scale > 0.0
            ),
            expert_var_beta=cfg.expert_var_beta,
        )
        x_out = x + moe_out

        if self.training and self.use_auxiliary_losses:
            read_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            if self.use_dual_memory:
                read_loss = combine_read_utilization_loss(
                    self.attn_memory_combine, cfg.read_util_min_fraction
                ) + combine_read_utilization_loss(
                    self.state_memory_combine, cfg.read_util_min_fraction
                )
            fusion_loss = fusion_balance_loss(fusion_gate)
            ssm_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            if ssm_state is not None:
                gamma = getattr(self, "ssm_norm_gamma", None)
                if gamma is not None:
                    ssm_loss = ssm_state_norm_loss(ssm_state, gamma)
            layer_aux = HybridLayerAuxLosses(
                recon=layer_aux.recon,
                assoc=layer_aux.assoc,
                gate=layer_aux.gate,
                read=read_loss / 2.0 if self.use_dual_memory else read_loss,
                fusion=fusion_loss,
                expert=expert_loss,
                ssm=ssm_loss,
                slot=layer_aux.slot,
            )

        return (
            x_out,
            aux_loss,
            z_loss,
            present_key_value,
            new_memory_state,
            new_mamba_cache,
            gate_stats,
            new_write_buffer,
            layer_aux,
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
    logits: Tensor | None
    loss: Tensor | None = None
    ce_loss: Tensor | None = None
    router_aux_loss: Tensor | None = None
    router_z_loss: Tensor | None = None
    past_key_values: list[tuple[Tensor, Tensor]] | None = None
    memory_states: list[HybridMemoryState | None] | None = None
    mamba_caches: list[MambaCache | None] | None = None
    gate_stats: dict[str, Tensor] | None = None
    write_buffers: list[MemoryWriteBuffer | None] | None = None
    auxiliary_losses: HybridAuxiliaryLossBreakdown | None = None


class HybridModel(nn.Module):
    def __init__(self, config: HybridMambaMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        layers: list[HybridDecoderLayer] = [
            HybridDecoderLayer(config) for _ in range(config.num_layers)
        ]
        if config.use_torch_compile:
            if config.gradient_checkpointing:
                warnings.warn(
                    "use_torch_compile and gradient_checkpointing are mutually "
                    "exclusive; disabling gradient_checkpointing for compile.",
                    stacklevel=2,
                )
                config.gradient_checkpointing = False
            compile_backend = "inductor" if torch.cuda.is_available() else "aot_eager"
            layers = [
                torch.compile(
                    layer,
                    mode=config.torch_compile_mode,
                    backend=compile_backend,
                )  # type: ignore[assignment]
                for layer in layers
            ]
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer(
            "ssm_norm_gammas",
            torch.zeros(config.num_layers),
            persistent=True,
        )
        self.register_buffer(
            "ssm_gammas_calibrated",
            torch.tensor(0.0),
            persistent=True,
        )

    def _ssm_calibration_done(self) -> bool:
        return bool(self.ssm_gammas_calibrated.item() > 0) or bool(
            self.ssm_norm_gammas.any().item()
        )

    @torch.no_grad()
    def calibrate_ssm_norm_thresholds(
        self, batch_size: int = 1, seq_len: int = 8
    ) -> None:
        if self._ssm_calibration_done():
            for i, layer in enumerate(self.layers):
                layer.ssm_norm_gamma = self.ssm_norm_gammas[i]
            return

        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype
        dummy = torch.randn(
            batch_size, seq_len, self.config.hidden_size, device=device, dtype=dtype
        )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(dummy, src=0)

        for i, layer in enumerate(self.layers):
            _, _, ssm_state = layer.mamba_block(
                dummy,
                use_cache=False,
                mamba_internal_checkpoint=False,
                layer_checkpointing_active=self.config.gradient_checkpointing,
            )
            assert ssm_state is not None
            norms = ssm_state.float().pow(2).mean(dim=(1, 2))
            self.ssm_norm_gammas[i] = torch.quantile(norms, 0.9)
            layer.ssm_norm_gamma = self.ssm_norm_gammas[i]

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(self.ssm_norm_gammas, src=0)
            for i, layer in enumerate(self.layers):
                layer.ssm_norm_gamma = self.ssm_norm_gammas[i]

        self.ssm_gammas_calibrated.fill_(1.0)

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
        training_step: int | None = None,
        max_training_steps: int | None = None,
        decode_accumulate_only: bool = False,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        list | None,
        list[HybridMemoryState | None],
        list[MambaCache | None] | None,
        dict[str, Tensor],
        list[MemoryWriteBuffer | None] | None,
        HybridAuxiliaryLossBreakdown,
    ]:
        hidden_states = self.embed_tokens(input_ids)
        batch_size, seq_len = hidden_states.shape[:2]

        batch_has_padding = _compute_batch_has_padding(attention_mask, seq_len)

        if (
            self.training
            and self.config.use_auxiliary_losses
            and not self._ssm_calibration_done()
        ):
            # Use a short dummy sequence; full-seq calibration is wasteful and
            # can trigger Mamba scan checkpoints on the first training step.
            self.calibrate_ssm_norm_thresholds(batch_size=batch_size)

        _validate_hybrid_cache_states(
            self.config,
            len(self.layers),
            batch_size,
            memory_states,
            mamba_caches,
            write_buffers,
            past_key_values,
            active_batch_mask,
        )

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
        aux_sums = HybridLayerAuxLosses.zeros(hidden_states.device, hidden_states.dtype)

        layer_checkpointing_active = (
            self.config.gradient_checkpointing and self.training and not use_cache
        )

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
                    layer_aux,
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
                    training_step,
                    max_training_steps,
                    batch_has_padding,
                    layer_checkpointing_active,
                    decode_accumulate_only,
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
                    layer_aux,
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
                    training_step,
                    max_training_steps,
                    batch_has_padding,
                    layer_checkpointing_active,
                    decode_accumulate_only,
                )
            total_aux_loss = total_aux_loss + layer_aux_loss
            total_z_loss = total_z_loss + layer_z_loss
            new_memory_states.append(layer_new_memory)
            if new_write_buffers is not None:
                new_write_buffers.append(layer_new_buf)
            for k, v in layer_gate_stats.items():
                all_gate_stats[f"layer_{i}_{k}"] = v
            aux_sums = HybridLayerAuxLosses(
                recon=aux_sums.recon + layer_aux.recon,
                assoc=aux_sums.assoc + layer_aux.assoc,
                gate=aux_sums.gate + layer_aux.gate,
                read=aux_sums.read + layer_aux.read,
                fusion=aux_sums.fusion + layer_aux.fusion,
                expert=aux_sums.expert + layer_aux.expert,
                ssm=aux_sums.ssm + layer_aux.ssm,
                slot=aux_sums.slot + layer_aux.slot,
            )
            if use_cache:
                present_key_values.append(present_kv)
                new_mamba_caches.append(layer_new_mamba)

        hidden_states = self.norm(hidden_states)
        n_layers = max(len(self.layers), 1)
        aux_avg = HybridLayerAuxLosses(
            recon=aux_sums.recon / n_layers,
            assoc=aux_sums.assoc / n_layers,
            gate=aux_sums.gate / n_layers,
            read=aux_sums.read / n_layers,
            fusion=aux_sums.fusion / n_layers,
            expert=aux_sums.expert / n_layers,
            ssm=aux_sums.ssm / n_layers,
            slot=aux_sums.slot / n_layers,
        )
        aux_breakdown = HybridAuxiliaryLossBreakdown(
            recon=aux_avg.recon,
            assoc=aux_avg.assoc,
            gate=aux_avg.gate,
            read=aux_avg.read,
            fusion=aux_avg.fusion,
            expert=aux_avg.expert,
            ssm=aux_avg.ssm,
            slot=aux_avg.slot,
        )
        return (
            hidden_states,
            total_aux_loss / n_layers,
            total_z_loss / n_layers,
            present_key_values,
            new_memory_states,
            new_mamba_caches,
            all_gate_stats,
            new_write_buffers,
            aux_breakdown,
        )


@dataclass
class _CudaDecodeGraphRunner:
    """CUDA graph replay for fixed-shape single-token decode steps."""

    model: "HybridForCausalLM"
    graph: torch.cuda.CUDAGraph | None = None
    static_input_ids: Tensor | None = None
    static_attention_mask: Tensor | None = None
    static_position_ids: Tensor | None = None
    static_out: HybridTrainingOutput | None = None
    mask_width: int = 0

    def capture(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        past_key_values: list | None,
        memory_states: list | None,
        mamba_caches: list | None,
        write_buffers: list | None,
        past_seen_tokens: int,
        active_batch_mask: Tensor,
    ) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            self.mask_width = attention_mask.size(1)
            self.static_input_ids = input_ids.clone()
            self.static_attention_mask = attention_mask.clone()
            self.static_position_ids = position_ids.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.static_out = self.model.forward(
                        input_ids=self.static_input_ids,
                        attention_mask=self.static_attention_mask,
                        position_ids=self.static_position_ids,
                        past_key_values=past_key_values,
                        mamba_caches=mamba_caches,
                        memory_states=memory_states,
                        write_buffers=write_buffers,
                        past_seen_tokens=past_seen_tokens,
                        use_cache=True,
                        skip_memory_write=True,
                        active_batch_mask=active_batch_mask,
                        decode_accumulate_only=True,
                    )
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_out = self.model.forward(
                    input_ids=self.static_input_ids,
                    attention_mask=self.static_attention_mask,
                    position_ids=self.static_position_ids,
                    past_key_values=past_key_values,
                    mamba_caches=mamba_caches,
                    memory_states=memory_states,
                    write_buffers=write_buffers,
                    past_seen_tokens=past_seen_tokens,
                    use_cache=True,
                    skip_memory_write=True,
                    active_batch_mask=active_batch_mask,
                    decode_accumulate_only=True,
                )
            return True
        except (RuntimeError, ValueError):
            self.graph = None
            self.static_out = None
            self.static_attention_mask = None
            self.static_position_ids = None
            return False

    def replay(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
    ) -> HybridTrainingOutput:
        assert self.graph is not None and self.static_input_ids is not None
        assert self.static_out is not None
        assert self.static_attention_mask is not None
        assert self.static_position_ids is not None
        self.static_input_ids.copy_(input_ids)
        self.static_attention_mask.copy_(attention_mask)
        self.static_position_ids.copy_(position_ids)
        self.graph.replay()
        return self.static_out


class HybridForCausalLM(nn.Module):
    def __init__(self, config: HybridMambaMoEConfig) -> None:
        super().__init__()
        self.config = config
        if config.use_dual_memory and not config.use_auxiliary_losses:
            msg = (
                "use_dual_memory=True with use_auxiliary_losses=False leaves memory "
                "write-path parameters without gradients on short chunks. Enable "
                "auxiliary losses for correct dual-memory training."
            )
            if config.debug_state_checks:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)
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

    def _weighted_auxiliary_loss(
        self,
        aux: HybridAuxiliaryLossBreakdown | None,
        device: torch.device,
        dtype: torch.dtype,
        training_step: int | None = None,
        max_training_steps: int | None = None,
    ) -> Tensor:
        if not self.config.use_auxiliary_losses or aux is None:
            return torch.tensor(0.0, device=device, dtype=dtype)
        cfg = self.config
        assoc_scale = _aux_loss_schedule(
            training_step, max_training_steps, cfg.assoc_warmup_fraction
        )
        expert_scale = _expert_loss_schedule(
            training_step, max_training_steps, cfg.expert_warmup_fraction
        )
        return (
            cfg.lambda_recon * aux.recon
            + cfg.lambda_assoc * assoc_scale * aux.assoc
            + cfg.lambda_gate * aux.gate
            + cfg.lambda_read * aux.read
            + cfg.lambda_fusion * aux.fusion
            + cfg.lambda_expert * expert_scale * aux.expert
            + cfg.lambda_ssm * aux.ssm
            + cfg.lambda_slot * aux.slot
        )

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
        training_step: int | None = None,
        max_training_steps: int | None = None,
        decode_accumulate_only: bool = False,
    ) -> HybridTrainingOutput:
        seq_len = input_ids.size(1)
        if self._should_chunk_training(seq_len, use_cache, memory_states):
            return self._forward_chunked(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                training_step=training_step,
                max_training_steps=max_training_steps,
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
            auxiliary_losses,
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
            training_step=training_step,
            max_training_steps=max_training_steps,
            decode_accumulate_only=decode_accumulate_only,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        ce_loss = None
        if labels is not None:
            labels = self._apply_label_ignore(labels, attention_mask)
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.label_ignore_index)
            assert logits is not None
            ce_loss = loss_fct(logits.view(-1, self.vocab_size), labels.reshape(-1))
            aux_total = self._weighted_auxiliary_loss(
                auxiliary_losses,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                training_step=training_step,
                max_training_steps=max_training_steps,
            )
            loss = (
                ce_loss
                + (self.router_aux_loss_coef * aux_loss)
                + (self.router_z_loss_coef * z_loss)
                + aux_total
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
            auxiliary_losses=auxiliary_losses,
        )

    def _forward_chunked(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        position_ids: Tensor | None,
        labels: Tensor | None,
        training_step: int | None = None,
        max_training_steps: int | None = None,
    ) -> HybridTrainingOutput:
        """BPTT through memory banks within one backward pass."""
        chunk_size = self.config.memory_chunk_size
        assert chunk_size is not None
        seq_len = input_ids.size(1)
        batch_size = input_ids.size(0)
        device = input_ids.device
        stream_ce = self.config.stream_chunked_ce_loss and labels is not None
        materialize_logits = self.config.return_logits or not stream_ce

        memory_states: list[HybridMemoryState | None] | None = None
        logits_chunks: list[Tensor] = []
        # MOERouter aux/z are per-token means; HybridModel layer-averages them;
        # here we token-weight across internal chunks.
        total_aux = torch.tensor(0.0, device=device)
        total_z = torch.tensor(0.0, device=device)
        gate_stat_sums: dict[str, Tensor] = {}
        gate_stat_counts: dict[str, int] = {}
        token_weight = 0
        ce_loss_sum = torch.tensor(0.0, device=device)
        loss_fct = (
            nn.CrossEntropyLoss(ignore_index=self.config.label_ignore_index)
            if labels is not None
            else None
        )
        aux_weighted = HybridLayerAuxLosses.zeros(
            device, self.model.embed_tokens.weight.dtype
        )

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
                chunk_aux,
            ) = self.model(
                input_ids=chunk_ids,
                memory_states=memory_states,
                attention_mask=chunk_mask,
                position_ids=chunk_pos,
                use_cache=False,
                training_step=training_step,
                max_training_steps=max_training_steps,
            )
            chunk_logits = self.lm_head(hidden_states)
            if materialize_logits:
                logits_chunks.append(chunk_logits)
            if labels is not None and loss_fct is not None:
                chunk_labels = labels[:, start:end]
                chunk_labels = self._apply_label_ignore(chunk_labels, chunk_mask)
                chunk_ce = loss_fct(
                    chunk_logits.view(-1, self.vocab_size), chunk_labels.reshape(-1)
                )
                ce_loss_sum = ce_loss_sum + chunk_ce * chunk_len
            del chunk_logits
            total_aux = total_aux + aux_loss * chunk_len
            total_z = total_z + z_loss * chunk_len
            token_weight += chunk_len
            aux_weighted = HybridLayerAuxLosses(
                recon=aux_weighted.recon + chunk_aux.recon * chunk_len,
                assoc=aux_weighted.assoc + chunk_aux.assoc * chunk_len,
                gate=aux_weighted.gate + chunk_aux.gate * chunk_len,
                read=aux_weighted.read + chunk_aux.read * chunk_len,
                fusion=aux_weighted.fusion + chunk_aux.fusion * chunk_len,
                expert=aux_weighted.expert + chunk_aux.expert * chunk_len,
                ssm=aux_weighted.ssm + chunk_aux.ssm * chunk_len,
                slot=aux_weighted.slot + chunk_aux.slot * chunk_len,
            )
            for key, val in gate_stats.items():
                if key not in gate_stat_sums:
                    gate_stat_sums[key] = val.clone()
                    gate_stat_counts[key] = 1
                else:
                    gate_stat_sums[key] = gate_stat_sums[key] + val
                    gate_stat_counts[key] += 1

        logits = torch.cat(logits_chunks, dim=1) if materialize_logits else None
        aux_loss = total_aux / max(token_weight, 1)
        z_loss = total_z / max(token_weight, 1)
        all_gate_stats = {
            k: gate_stat_sums[k] / gate_stat_counts[k] for k in gate_stat_sums
        }
        tw = max(token_weight, 1)
        auxiliary_losses = HybridAuxiliaryLossBreakdown(
            recon=aux_weighted.recon / tw,
            assoc=aux_weighted.assoc / tw,
            gate=aux_weighted.gate / tw,
            read=aux_weighted.read / tw,
            fusion=aux_weighted.fusion / tw,
            expert=aux_weighted.expert / tw,
            ssm=aux_weighted.ssm / tw,
            slot=aux_weighted.slot / tw,
        )

        loss = None
        ce_loss = None
        if labels is not None:
            ce_loss = ce_loss_sum / max(token_weight, 1)
            aux_total = self._weighted_auxiliary_loss(
                auxiliary_losses,
                device=device,
                dtype=self.model.embed_tokens.weight.dtype,
                training_step=training_step,
                max_training_steps=max_training_steps,
            )
            loss = (
                ce_loss
                + (self.router_aux_loss_coef * aux_loss)
                + (self.router_z_loss_coef * z_loss)
                + aux_total
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
            auxiliary_losses=auxiliary_losses,
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
            buf_attn, buf_mamba = _materialize_write_buffer(buf)
            assert buf_attn is not None and buf_mamba is not None
            new_a, _, _, new_s, _, _ = batched_dual_memory_write(
                layer.attn_memory_bank,
                layer.state_memory_bank,
                buf_attn,
                buf_mamba,
                a_mem,
                s_mem,
            )
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

        total_len = prompt_len + max_new_tokens
        generated_buf = torch.full(
            (batch_size, total_len),
            self.config.pad_token_id,
            dtype=input_ids.dtype,
            device=device,
        )
        generated_buf[:, :prompt_len] = generated
        attn_buf = torch.zeros(
            batch_size, total_len, dtype=attention_mask.dtype, device=device
        )
        attn_buf[:, :prompt_len] = attention_mask
        cur_len = prompt_len

        write_interval = self._memory_write_interval()
        past_key_values = None
        memory_states = None
        mamba_caches = None
        write_buffers: list[MemoryWriteBuffer | None] | None = None
        past_seen_tokens = 0
        tokens_in_write_buffer = 0
        out = None
        cuda_runner: _CudaDecodeGraphRunner | None = None

        try:
            # Chunked prefill so memory writes match training chunk size.
            for start in range(0, prompt_len, write_interval):
                end = min(start + write_interval, prompt_len)
                chunk = generated_buf[:, start:end]
                chunk_mask = attn_buf[:, :end]
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

                generated_buf[:, cur_len : cur_len + 1] = next_token
                attn_buf[:, cur_len : cur_len + 1] = (~finished).long().unsqueeze(-1)
                cur_len += 1
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
                step_attn_mask = attn_buf[:, :cur_len]
                if step_attn_mask.size(1) > self.config.window_size:
                    step_attn_mask = step_attn_mask[:, -self.config.window_size :]

                step_input = next_token.clone()
                step_input[~active] = self.config.pad_token_id
                tokens_in_write_buffer += 1
                do_memory_write = tokens_in_write_buffer >= write_interval
                decode_accumulate = not do_memory_write

                use_graph = (
                    self.config.use_cuda_graph
                    and not do_sample
                    and decode_accumulate
                    and active.all()
                    and torch.cuda.is_available()
                )

                if use_graph:
                    if (
                        cuda_runner is not None
                        and cuda_runner.mask_width != step_attn_mask.size(1)
                    ):
                        cuda_runner = None
                    if cuda_runner is None:
                        cuda_runner = _CudaDecodeGraphRunner(self)
                        if not cuda_runner.capture(
                            step_input,
                            step_attn_mask,
                            step_position_ids,
                            past_key_values,
                            memory_states,
                            mamba_caches,
                            write_buffers,
                            past_seen_tokens,
                            active,
                        ):
                            cuda_runner = None
                    if cuda_runner is not None:
                        out = cuda_runner.replay(
                            step_input, step_attn_mask, step_position_ids
                        )
                    else:
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
                            skip_memory_write=True,
                            active_batch_mask=active,
                            decode_accumulate_only=True,
                        )
                else:
                    cuda_runner = None
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
                        decode_accumulate_only=decode_accumulate,
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

        return generated_buf[:, :cur_len]


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
