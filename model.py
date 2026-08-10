"""
model.py

Contains two model families:

1. MixtralConfig / MixtralForCausalLM -- sliding-window-GQA + Top-2 MoE
   baseline (control for ablations).

2. HybridMambaMoEConfig / HybridForCausalLM -- Hybrid Mamba-MoE with Dual
   Memory (research/research.md v2.0):
       - Sliding window GQA branch (reuses SlidingWindowGQA)
       - Mamba selective-SSM branch (MambaBlock) with parallel associative
         scan for prefill/training and incremental (conv_state, ssm_state)
         caching for autoregressive decode
       - Two compressive memory banks (CompressiveMemoryBank), one per
         branch: read *into* branch inputs, write *raw* branch outputs
         (research.md §3.2), O(L * m), m << L
       - Token-wise gated fusion (TokenGatedFusion) -- O(L), not O(L^2)
       - Top-2 sparse MoE (shared DroplessMoELayer)

Falsification hooks (design doc §6):
    - HybridDecoderLayer returns per-layer write-gate stats
    - use_dual_memory=False for architecture-level memory-off ablation
    - HybridModel.zero_memory_states() for Test-1 zeroed-at-inference
    - build_test3_null_baseline_config() for matched-parameter SSM-only null
"""

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
    # preserves the original fully dropless behavior.
    capacity_factor: float | None = 1.25

    # RoPE / positional configuration (previously hardcoded deep inside
    # RotaryEmbedding / SlidingWindowGQA).
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0

    # Special tokens (previously hardcoded to 1/2 inside the data pipeline
    # with no link back to the model config).
    bos_token_id: int = 1
    eos_token_id: int = 2

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
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
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
            if attention_mask is not None and attention_mask.dim() == 2:
                attention_mask = attention_mask[:, -self.window_size :]

        # Cache only the sliding window (O(1) in L for decode), not the
        # full history — RoPE is already baked into cached K.
        present_key_value = (key_states, value_states) if use_cache else None

        kv_seq_len = key_states.size(2)

        num_queries_per_kv = self.num_heads // self.num_kv_heads
        key_states_r = self._repeat_kv(key_states, num_queries_per_kv)
        value_states_r = self._repeat_kv(value_states, num_queries_per_kv)

        row_idx = torch.arange(seq_len, device=device).unsqueeze(1) + (
            kv_seq_len - seq_len
        )
        col_idx = torch.arange(kv_seq_len, device=device).unsqueeze(0)
        sliding_causal_mask = (row_idx >= col_idx) & (
            (row_idx - col_idx) < self.window_size
        )

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
    dropping (zeroing) overflow tokens for that expert. This is a memory
    safety valve for T4: with imbalanced routing, a naive dispatch can spike
    a single expert's batch 2-3x versus the average, which is the difference
    between fitting in 16GB and OOM. Set `capacity_factor=None` to restore
    the original fully "dropless" (no token ever skipped) behavior -- note
    this reintroduces the memory-spike risk under imbalanced routing.
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
        x_flat = x.view(-1, orig_shape[-1])
        num_tokens = x_flat.size(0)

        topk_weights, topk_indices, aux_loss, z_loss = self.router(x_flat)
        moe_output = torch.zeros_like(x_flat)

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
                # Keep first `capacity` tokens in original order; drop the
                # rest (they simply get no contribution from this expert).
                row_indices = row_indices[:capacity]
                k_indices = k_indices[:capacity]

            expert_inputs = x_flat[row_indices]
            expert_outputs = self.experts[expert_idx](expert_inputs)

            gating_scale = topk_weights[row_indices, k_indices].unsqueeze(-1)
            moe_output.index_add_(0, row_indices, expert_outputs * gating_scale)

        return moe_output.view(*orig_shape), aux_loss, z_loss


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

        moe_out, aux_loss, z_loss = self.moe_block(self.rmsnorm_moe(x_attn))
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
        return hidden_states, total_aux_loss, total_z_loss, present_key_values


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
            loss_fct = nn.CrossEntropyLoss()
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


MambaCache = tuple[Tensor, Tensor]  # (conv_state, ssm_state)


class MambaBlock(nn.Module):
    """
    Selective SSM (Mamba / S6) in pure PyTorch.

    Prefill/training uses a parallel associative scan (no per-token Python
    loop). Autoregressive decode uses allocate_inference_cache() + step()
    with (conv_state, ssm_state), matching the official mamba_ssm API shape.
    No custom CUDA dependency — runs on any SDPA-capable device.
    """

    def __init__(
        self,
        hidden_size: int,
        state_size: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.d_inner = expand * hidden_size
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(hidden_size / 16)

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
    ) -> tuple[Tensor, MambaCache | None]:
        """
        x: [B, L, hidden_size]
        If use_cache and L==1 and cache is provided, runs a single decode step.
        Otherwise runs full-sequence prefill (parallel scan); when use_cache,
        returns updated (conv_state, ssm_state) for subsequent steps.
        """
        _, seq_len, _ = x.shape

        if use_cache and cache is not None and seq_len == 1:
            return self.step(x, cache[0], cache[1])

        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        x_conv = x_in.transpose(1, 2)
        if use_cache:
            # Keep last conv_kernel tokens as the rolling conv buffer.
            pad = max(self.conv_kernel - seq_len, 0)
            conv_state = F.pad(x_conv, (pad, 0))[:, :, -self.conv_kernel :].contiguous()
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
            x_conv, dt, A, B_param, C_param, self.D, return_final_state=use_cache
        )
        y = y * F.silu(z)
        out = self.out_proj(y)

        new_cache: MambaCache | None = None
        if use_cache:
            assert conv_state is not None and ssm_state is not None
            new_cache = (conv_state, ssm_state)
        return out, new_cache

    def step(
        self, x: Tensor, conv_state: Tensor, ssm_state: Tensor
    ) -> tuple[Tensor, MambaCache]:
        """Single-token decode. x: [B, 1, hidden_size]."""
        assert x.size(1) == 1
        dtype = x.dtype

        xz = self.in_proj(x.squeeze(1))
        x_in, z = xz.chunk(2, dim=-1)

        # Roll causal-conv buffer and insert the new token.
        conv_state = conv_state.roll(shifts=-1, dims=-1)
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
        return out, (conv_state, ssm_state)

    @staticmethod
    def _parallel_associative_scan(delta_a: Tensor, delta_b_u: Tensor) -> Tensor:
        """
        Inclusive parallel scan for h_t = delta_a_t * h_{t-1} + delta_b_u_t
        with h_{-1} = 0.

        delta_a, delta_b_u: [B, L, D, N] -> h: [B, L, D, N]

        Uses doubling (Blelloch-style) with out-of-place updates so autograd
        stays correct — no per-token Python loop over L.
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
    ) -> tuple[Tensor, Tensor | None]:
        """
        u, dt: [B, L, d_inner]; A: [d_inner, n]; B, C: [B, L, n]; D: [d_inner]
        """
        input_dtype = u.dtype
        u_f = u.float()
        dt_f = dt.float()
        B_f = B.float()
        C_f = C.float()

        delta_a = torch.exp(dt_f.unsqueeze(-1) * A)  # [B, L, d_inner, n]
        delta_b_u = dt_f.unsqueeze(-1) * B_f.unsqueeze(2) * u_f.unsqueeze(-1)

        states = cls._parallel_associative_scan(delta_a, delta_b_u)
        y = (states * C_f.unsqueeze(2)).sum(dim=-1)
        y = y + u_f * D.float()
        final_state = states[:, -1].contiguous() if return_final_state else None
        return y.to(input_dtype), final_state


class CompressiveMemoryBank(nn.Module):
    """
    Fixed-size (m slots) gated read/write memory bank.

    - read: tokens attend to memory as K/V — O(L * m)
    - write: learned summary queries compress the chunk, GRU-style gate
      blends into existing memory — O(L * m)

    Memory is threaded through forward by the caller (not a module buffer).
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

        self.init_memory = nn.Parameter(torch.randn(memory_size, hidden_size) * 0.02)
        self.read_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.summary_query = nn.Parameter(torch.randn(memory_size, hidden_size) * 0.02)
        self.write_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
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
        """Convert 1=keep / 0=pad mask to MultiheadAttention key_padding_mask."""
        if attention_mask is None:
            return None
        if attention_mask.dim() != 2:
            raise ValueError(
                "CompressiveMemoryBank expects a 2D attention_mask [B, L]."
            )
        # True = ignore that key position.
        return ~attention_mask.bool()

    def read(
        self,
        x: Tensor,
        memory: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        # Queries are sequence tokens; memory slots are never padded.
        del attention_mask  # read attends to fixed memory slots, not sequence keys
        out, _ = self.read_attn(x, memory, memory, need_weights=False)
        return out

    def write(
        self,
        x: Tensor,
        memory: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = x.size(0)
        query = self.summary_query.unsqueeze(0).expand(batch_size, -1, -1)
        kpm = self._key_padding_mask(attention_mask)
        chunk_summary, _ = self.write_attn(
            query, x, x, key_padding_mask=kpm, need_weights=False
        )

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
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        tuple[Tensor, Tensor] | None,
        HybridMemoryState | None,
        MambaCache | None,
        dict[str, Tensor],
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

        new_memory_state = memory_state
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

        attn_out, present_key_value = self.attention_block(
            attn_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        mamba_out, new_mamba_cache = self.mamba_block(
            mamba_input, cache=mamba_cache, use_cache=use_cache
        )

        if self.use_dual_memory:
            assert memory_state is not None
            a_mem, s_mem = memory_state
            # Write *raw* branch outputs (research E/F), not memory-mixed tensors.
            new_a_mem, a_write_gate = self.attn_memory_bank.write(
                attn_out, a_mem, attention_mask=token_attention_mask
            )
            new_s_mem, s_write_gate = self.state_memory_bank.write(
                mamba_out, s_mem, attention_mask=token_attention_mask
            )
            new_memory_state = (new_a_mem, new_s_mem)
            gate_stats = {
                "attn_write_gate_mean": a_write_gate.detach().mean(),
                "state_write_gate_mean": s_write_gate.detach().mean(),
            }

        fused, _fusion_gate = self.fusion(attn_out, mamba_out)
        x = residual + fused

        moe_out, aux_loss, z_loss = self.moe_block(self.rmsnorm_moe(x))
        x_out = x + moe_out

        return (
            x_out,
            aux_loss,
            z_loss,
            present_key_value,
            new_memory_state,
            new_mamba_cache,
            gate_stats,
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
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        list | None,
        list[HybridMemoryState | None],
        list[MambaCache | None] | None,
        dict[str, Tensor],
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
        all_gate_stats: dict[str, Tensor] = {}

        for i, layer in enumerate(self.layers):
            layer_past_kv = past_key_values[i] if past_key_values is not None else None
            layer_memory = memory_states[i] if memory_states is not None else None
            layer_mamba = mamba_caches[i] if mamba_caches is not None else None

            (
                hidden_states,
                layer_aux_loss,
                layer_z_loss,
                present_kv,
                layer_new_memory,
                layer_new_mamba,
                layer_gate_stats,
            ) = layer(
                hidden_states,
                memory_state=layer_memory,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past_kv,
                mamba_cache=layer_mamba,
                use_cache=use_cache,
            )
            total_aux_loss = total_aux_loss + layer_aux_loss
            total_z_loss = total_z_loss + layer_z_loss
            new_memory_states.append(layer_new_memory)
            for k, v in layer_gate_stats.items():
                all_gate_stats[f"layer_{i}_{k}"] = v
            if use_cache:
                present_key_values.append(present_kv)
                new_mamba_caches.append(layer_new_mamba)

        hidden_states = self.norm(hidden_states)
        return (
            hidden_states,
            total_aux_loss,
            total_z_loss,
            present_key_values,
            new_memory_states,
            new_mamba_caches,
            all_gate_stats,
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
        memory_states: list[HybridMemoryState | None] | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: list | None = None,
        mamba_caches: list[MambaCache | None] | None = None,
        past_seen_tokens: int | None = None,
        use_cache: bool = False,
        labels: Tensor | None = None,
    ) -> HybridTrainingOutput:
        (
            hidden_states,
            aux_loss,
            z_loss,
            present_key_values,
            new_memory_states,
            new_mamba_caches,
            gate_stats,
        ) = self.model(
            input_ids=input_ids,
            memory_states=memory_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            mamba_caches=mamba_caches,
            past_seen_tokens=past_seen_tokens,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        ce_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
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
        )

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
        Total cost is O(L) in the final sequence length (one prefill + L_new
        single-token steps), not O(L^2) full re-forwards.
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

        position_ids = (
            torch.arange(prompt_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        out = self.forward(
            input_ids=generated,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_seen_tokens=0,
            use_cache=True,
        )
        past_key_values = out.past_key_values
        memory_states = out.memory_states
        mamba_caches = out.mamba_caches
        past_seen_tokens = prompt_len

        try:
            for step in range(max_new_tokens):
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

                step_position_ids = torch.full(
                    (batch_size, 1),
                    past_seen_tokens,
                    dtype=torch.long,
                    device=device,
                )
                step_attn_mask = attention_mask
                if step_attn_mask.size(1) > self.config.window_size:
                    step_attn_mask = step_attn_mask[:, -self.config.window_size :]

                out = self.forward(
                    input_ids=next_token,
                    attention_mask=step_attn_mask,
                    position_ids=step_position_ids,
                    past_key_values=past_key_values,
                    memory_states=memory_states,
                    mamba_caches=mamba_caches,
                    past_seen_tokens=past_seen_tokens,
                    use_cache=True,
                )
                past_key_values = out.past_key_values
                memory_states = out.memory_states
                mamba_caches = out.mamba_caches
                past_seen_tokens += 1
        finally:
            self.train(was_training)

        return generated


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_test3_null_baseline_config(
    hybrid_config: HybridMambaMoEConfig,
) -> HybridMambaMoEConfig:
    """
    Test 3 null hypothesis: larger Mamba state, no explicit memory banks.
    Heuristic bump — verify with count_trainable_params and adjust.
    """
    import copy

    null_config = copy.deepcopy(hybrid_config)
    null_config.use_dual_memory = False
    null_config.mamba_state_size = hybrid_config.mamba_state_size * 4
    return null_config
