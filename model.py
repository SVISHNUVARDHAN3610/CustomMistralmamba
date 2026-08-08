"""
model.py

Contains two model families:

1. MixtralConfig / MixtralForCausalLM  -- the original sliding-window-GQA +
   Top-2 MoE baseline (unchanged from the working version). Keep this around
   as the control model for ablations -- in particular it doubles as the
   "no explicit memory, no SSM branch" reference point.

2. HybridMambaMoEConfig / HybridForCausalLM -- the proposed v2.0 architecture
   (Hybrid Mamba-MoE with Dual Memory, see design doc v2.0):
       - Sliding window GQA branch (reuses SlidingWindowGQA below)
       - Mamba (selective SSM) branch (MambaBlock)
       - Two compressive memory banks (CompressiveMemoryBank), one per
         branch, with explicit bounded-size, gated read/write -- NOT a
         static nn.Parameter. This is the actual mechanism the v1.0 doc's
         "Aₗ, Mₗ" diagram implied but never implemented.
       - Token-wise gated fusion (TokenGatedFusion) instead of the
         bidirectional cross-attention from v1.0 that reintroduced O(L^2)
         cost. Fusion here is O(L); memory read/write is O(L * m) with a
         constant memory size m << L.

Both models share: RMSNorm, RotaryEmbedding, SlidingWindowGQA, SwiGLUExpert,
MOERouter, DroplessMoELayer.

For the falsification tests from the design doc (rare-fact recall,
write-gate activity monitoring, matched-parameter bigger-Mamba-state null
hypothesis) see the bottom of this file for the relevant hooks:
    - HybridDecoderLayer.forward returns per-layer write-gate stats
    - HybridMambaMoEConfig.use_dual_memory=False gives you the "memory off"
      ablation for Test 1
    - Bumping HybridMambaMoEConfig.mamba_state_size while setting
      use_dual_memory=False gives you the Test 3 null-hypothesis baseline
      (bigger SSM state, no explicit memory) at roughly matched parameter
      count -- see the parameter-count helper at the bottom.
"""

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# =============================================================================
# Shared config (Mixtral baseline)
# =============================================================================


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


# =============================================================================
# Shared building blocks
# =============================================================================


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
    re-registered at runtime. If a sequence longer than
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
    """Sliding-window grouped-query attention using SDPA only."""

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

        present_key_value = (key_states, value_states) if use_cache else None

        if key_states.size(2) > self.window_size:
            key_states = key_states[:, :, -self.window_size :, :]
            value_states = value_states[:, :, -self.window_size :, :]
            # BUGFIX: attention_mask (padding mask over the same key/value
            # sequence) must be truncated the same way, or it ends up a
            # different length than key_states/value_states below and the
            # `&` with sliding_causal_mask raises a shape mismatch. This
            # only shows up once seq_len > window_size with a mask passed
            # in -- e.g. during generate() on a sequence longer than one
            # window -- so it was latent in the original code.
            if attention_mask is not None and attention_mask.dim() == 2:
                attention_mask = attention_mask[:, -self.window_size :]

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
    """Standard SwiGLU feed-forward expert."""

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
        logits = self.wg(x).to(torch.float32)
        logits = torch.clamp(logits, min=-30.0, max=30.0)

        logsumexp_vals = torch.logsumexp(logits, dim=-1)
        z_loss = torch.mean(logsumexp_vals**2)

        full_probs = F.softmax(logits, dim=-1)  # [N, E]

        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1).to(input_dtype)

        one_hot_indices = F.one_hot(topk_indices, num_classes=self.num_experts).float()
        f_i = one_hot_indices.sum(dim=1).mean(dim=0)  # [E]

        p_i = full_probs.mean(dim=0)  # [E]

        aux_loss = self.num_experts * torch.sum(f_i * p_i)

        return topk_weights, topk_indices, aux_loss, z_loss


class DroplessMoELayer(nn.Module):
    """MoE dispatch/combine with optional capacity-limited expert batches."""

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
                row_indices = row_indices[:capacity]
                k_indices = k_indices[:capacity]

            expert_inputs = x_flat[row_indices]
            expert_outputs = self.experts[expert_idx](expert_inputs)

            gating_scale = topk_weights[row_indices, k_indices].unsqueeze(-1)
            moe_output.index_add_(0, row_indices, expert_outputs * gating_scale)

        return moe_output.view(*orig_shape), aux_loss, z_loss


# =============================================================================
# Mixtral baseline: decoder layer / model / causal LM
# =============================================================================


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
            if module.bias is not None:
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
# NEW: Hybrid Mamba-MoE with Dual Memory (v2.0)
# =============================================================================


@dataclass
class HybridMambaMoEConfig(MixtralConfig):
    """
    Extends MixtralConfig with Mamba-branch and memory-branch settings.
    All Mixtral fields (hidden_size, num_heads, window_size, num_experts,
    top_k, etc.) are reused unchanged for the GQA and MoE parts of each
    layer.
    """

    # --- Mamba (selective SSM) branch ---
    mamba_state_size: int = 16  # n: SSM state dimension per inner channel
    mamba_conv_kernel: int = 4  # causal depthwise conv kernel width
    mamba_expand: int = 2  # d_inner = mamba_expand * hidden_size
    mamba_dt_rank: int | None = None  # defaults to ceil(hidden_size / 16) if None

    # --- Dual memory branch ---
    use_dual_memory: bool = (
        True  # set False for the "memory off" / "SSM-only" ablations
    )
    memory_size: int = 64  # m: fixed number of memory slots (bounded, != L)
    memory_num_heads: int = 8  # heads used for memory read/write cross-attention

    def __post_init__(self):
        # dataclass inheritance note: MixtralConfig has no __post_init__, so
        # this is safe to define here without calling super().
        pass


class MambaBlock(nn.Module):
    """
    Selective State Space Model block (Mamba / S6), reference sequential-scan
    implementation. This trades throughput for correctness/portability: no
    custom CUDA kernel dependency, works on any device SDPA works on
    (including T4). Swap in an optimized selective-scan kernel later without
    changing the surrounding architecture.

    NOTE on caching: this implementation does not yet expose an incremental
    (conv_state, ssm_state) cache for step-by-step autoregressive generation
    the way SlidingWindowGQA exposes past_key_value -- it always scans the
    full sequence passed to forward(). Fine for training and for the
    falsification-test evaluations (which process fixed chunks), not yet
    wired for token-by-token generation.
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

        # Produces per-token, input-dependent (dt, B, C) -- the "selective"
        # part of selective SSM: the model can choose, per token, how much
        # to write into / read from state.
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * state_size, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is initialized structured (HiPPO-like: -1..-state_size per
        # channel) and kept negative via log-parameterization + negation,
        # standard Mamba init.
        A = torch.arange(1, state_size + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, hidden_size]
        _, seq_len, _ = x.shape

        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_in, z = xz.chunk(2, dim=-1)

        x_in = x_in.transpose(1, 2)  # [B, d_inner, L]
        x_in = self.conv1d(x_in)[..., :seq_len]  # causal conv, trim padding tail
        x_in = F.silu(x_in)
        x_in = x_in.transpose(1, 2)  # [B, L, d_inner]

        x_dbl = self.x_proj(x_in)  # [B, L, dt_rank + 2*state_size]
        dt, B_param, C_param = torch.split(
            x_dbl, [self.dt_rank, self.state_size, self.state_size], dim=-1
        )
        dt = F.softplus(
            self.dt_proj(dt)
        )  # [B, L, d_inner], strictly positive step size

        A = -torch.exp(
            self.A_log.float()
        )  # [d_inner, state_size], strictly negative -> stable

        y = self._selective_scan(x_in, dt, A, B_param, C_param, self.D)

        y = y * F.silu(z)
        return self.out_proj(y)

    @staticmethod
    def _selective_scan(
        u: torch.Tensor,  # [B, L, d_inner]
        dt: torch.Tensor,  # [B, L, d_inner]
        A: torch.Tensor,  # [d_inner, n]
        B: torch.Tensor,  # [B, L, n]
        C: torch.Tensor,  # [B, L, n]
        D: torch.Tensor,  # [d_inner]
    ) -> torch.Tensor:
        """
        Naive sequential scan, O(L) time / O(1) extra memory per step (state
        is [B, d_inner, n], reused in place across the loop). This is the
        textbook-correct but unfused implementation -- prioritizes being
        obviously correct over throughput. For long sequences on a single
        T4 this will be the training-speed bottleneck; that's a known,
        expected tradeoff of skipping the custom CUDA kernel, not a bug.
        """
        batch_size, seq_len, d_inner = u.shape
        n = A.shape[1]

        # Discretize: deltaA = exp(dt * A), per (batch, time, channel, state)
        deltaA = torch.exp(dt.unsqueeze(-1) * A)  # [B, L, d_inner, n]
        deltaB_u = (
            dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        )  # [B, L, d_inner, n]

        state = torch.zeros(batch_size, d_inner, n, device=u.device, dtype=deltaA.dtype)
        ys = []
        for t in range(seq_len):
            state = deltaA[:, t] * state + deltaB_u[:, t]
            y_t = (state * C[:, t].unsqueeze(1)).sum(dim=-1)  # [B, d_inner]
            ys.append(y_t)

        y = torch.stack(ys, dim=1).to(u.dtype)  # [B, L, d_inner]
        y = y + u * D
        return y


class CompressiveMemoryBank(nn.Module):
    """
    Fixed-size (m slots), gated read/write memory bank -- see design doc
    v2.0, "Compressive Memory Banks" section. This is the actual mechanism;
    v1.0's `attention_memory` / `state_memory` were static nn.Parameter
    vectors with no read/write dynamics at all.

    - read(x, memory): x (length L) attends to memory (length m) as K/V.
      Cost O(L * m), linear in L since m is fixed.
    - write(x, memory): a fixed set of m learned summary queries attend to x
      (length L) as K/V to compress it down to m vectors, then a GRU-style
      gate blends that summary into the existing memory. Cost O(L * m).

    Memory state is NOT a module buffer -- it's threaded explicitly through
    forward() calls by the caller (HybridModel), the same way KV-cache is
    threaded through past_key_value. This makes it trivial to: reset memory
    between independent sequences, persist it across chunks of one long
    sequence, or zero it out for the Test 1 "memory off" falsification run.
    """

    def __init__(
        self, hidden_size: int, memory_size: int = 64, num_heads: int = 8
    ) -> None:
        super().__init__()
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
    ) -> torch.Tensor:
        return (
            self.init_memory.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .to(device=device, dtype=dtype)
            .clone()
        )

    def read(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d] as query, memory: [B, m, d] as key/value -> O(L*m)
        out, _ = self.read_attn(x, memory, memory, need_weights=False)
        return out

    def write(
        self, x: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compress x (length L) down to m summary vectors via learned queries.
        batch_size = x.size(0)
        query = self.summary_query.unsqueeze(0).expand(batch_size, -1, -1)
        chunk_summary, _ = self.write_attn(query, x, x, need_weights=False)  # [B, m, d]

        gate = torch.sigmoid(
            self.write_gate(torch.cat([memory, chunk_summary], dim=-1))
        )
        new_memory = gate * memory + (1.0 - gate) * self.write_update(chunk_summary)

        # gate returned for Test 2 (write-gate activity monitoring): values
        # saturating near 0 or 1 across training indicate the memory is
        # either never updating or never persisting.
        return new_memory, gate


class TokenGatedFusion(nn.Module):
    """
    O(L) per-token fusion of the attention-branch and Mamba-branch outputs.
    Replaces v1.0's bidirectional nn.MultiheadAttention cross-fusion, which
    cost O(L^2) and was run twice (redundant weights) for no benefit: both
    branches already produce per-position, aligned outputs, so mixing them
    is a local operation and doesn't need cross-sequence attention.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self, a: torch.Tensor, m: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.sigmoid(self.gate(torch.cat([a, m], dim=-1)))
        fused = g * a + (1.0 - g) * m
        return fused, g


HybridMemoryState = tuple[
    torch.Tensor, torch.Tensor
]  # (attn_memory, state_memory), each [B, m, d]


class HybridDecoderLayer(nn.Module):
    """
    One layer of the v2.0 architecture:
        RMSNorm -> {SlidingWindowGQA, MambaBlock} (parallel)
                -> optional memory read/write per branch
                -> TokenGatedFusion
                -> residual add -> RMSNorm -> Top-2 MoE -> residual add
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
            # Combine branch output with its memory read (concat + project,
            # rather than overwrite) so the branch can learn how much to
            # rely on memory vs. its own local computation.
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

    def forward(
        self,
        x: torch.Tensor,
        memory_state: HybridMemoryState | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | None,
        HybridMemoryState | None,
        dict[str, torch.Tensor],
    ]:
        residual = x
        x_norm = self.rmsnorm_in(x)

        attn_out, present_key_value = self.attention_block(
            x_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        mamba_out = self.mamba_block(x_norm)

        new_memory_state = memory_state
        gate_stats: dict[str, torch.Tensor] = {}

        if self.use_dual_memory:
            if memory_state is None:
                memory_state = self.init_memory_state(x.size(0), x.device, x.dtype)
            a_mem, s_mem = memory_state

            a_read = self.attn_memory_bank.read(attn_out, a_mem)
            attn_out = self.attn_memory_combine(torch.cat([attn_out, a_read], dim=-1))

            s_read = self.state_memory_bank.read(mamba_out, s_mem)
            mamba_out = self.state_memory_combine(
                torch.cat([mamba_out, s_read], dim=-1)
            )

            new_a_mem, a_write_gate = self.attn_memory_bank.write(attn_out, a_mem)
            new_s_mem, s_write_gate = self.state_memory_bank.write(mamba_out, s_mem)
            new_memory_state = (new_a_mem, new_s_mem)

            # Detached scalars for logging (Test 2) -- mean write-gate value
            # per bank. Gate saturating near 0 or 1 across training is the
            # signal memory isn't functioning as intended.
            gate_stats = {
                "attn_write_gate_mean": a_write_gate.detach().mean(),
                "state_write_gate_mean": s_write_gate.detach().mean(),
            }

        fused, _fusion_gate = self.fusion(attn_out, mamba_out)
        x = residual + fused

        moe_out, aux_loss, z_loss = self.moe_block(self.rmsnorm_moe(x))
        x_out = x + moe_out

        return x_out, aux_loss, z_loss, present_key_value, new_memory_state, gate_stats


def _top_k_filter(logits: Tensor, top_k: int) -> Tensor:
    """Keep only the top_k highest-logit tokens per row, -inf out the rest."""
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(
        logits < min_values, torch.full_like(logits, float("-inf")), logits
    )


def _top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    """Nucleus filtering: keep smallest set of tokens whose cumulative prob >= top_p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_mask = cumulative_probs > top_p
    # shift right so we always keep at least the single highest-prob token
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
    # Threaded memory state, one (attn_mem, state_mem) tuple per layer (or
    # None per layer if that layer has use_dual_memory=False). Pass this
    # back in as `memory_states` on the next forward() call to persist
    # memory across chunks of a long sequence -- this is what makes the
    # memory bank behave like a KV-cache-style running state rather than
    # something reset every forward pass.
    memory_states: list[HybridMemoryState | None] | None = None
    # Per-layer write-gate means, for Test 2 monitoring. Keys are
    # "layer_{i}_attn_write_gate_mean" / "layer_{i}_state_write_gate_mean".
    gate_stats: dict[str, Tensor] | None = None


class HybridModel(nn.Module):
    def __init__(self, config: HybridMambaMoEConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [HybridDecoderLayer(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_states: list[HybridMemoryState | None] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list | None,
        list[HybridMemoryState | None],
        dict[str, torch.Tensor],
    ]:
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
        new_memory_states: list[HybridMemoryState | None] = []
        all_gate_stats: dict[str, torch.Tensor] = {}

        for i, layer in enumerate(self.layers):
            layer_past_kv = past_key_values[i] if past_key_values is not None else None
            layer_memory_state = memory_states[i] if memory_states is not None else None

            (
                hidden_states,
                layer_aux_loss,
                layer_z_loss,
                present_kv,
                layer_new_memory,
                layer_gate_stats,
            ) = layer(
                hidden_states,
                memory_state=layer_memory_state,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past_kv,
                use_cache=use_cache,
            )
            total_aux_loss = total_aux_loss + layer_aux_loss
            total_z_loss = total_z_loss + layer_z_loss
            new_memory_states.append(layer_new_memory)

            for k, v in layer_gate_stats.items():
                all_gate_stats[f"layer_{i}_{k}"] = v

            if use_cache:
                present_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)
        return (
            hidden_states,
            total_aux_loss,
            total_z_loss,
            present_key_values,
            new_memory_states,
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
            if module.bias is not None:
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
        use_cache: bool = False,
        labels: Tensor | None = None,
    ) -> HybridTrainingOutput:

        (
            hidden_states,
            aux_loss,
            z_loss,
            present_key_values,
            new_memory_states,
            gate_stats,
        ) = self.model(
            input_ids=input_ids,
            memory_states=memory_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        ce_loss = None
        if labels is not None:
            # labels are expected to already be the next-token shift of
            # input_ids, matching the (chunk[:-1], chunk[1:]) contract from
            # the Mixtral baseline's data pipeline. Do NOT shift again here.
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
        Simple autoregressive generation for interactive use (chat interface,
        qualitative eval, etc.).

        CACHING LIMITATION (read before using this for anything
        latency-sensitive): MambaBlock does not yet expose an incremental
        (conv_state, ssm_state) cache -- see its docstring. Because of that,
        this method re-runs the FULL forward pass over the entire sequence
        generated so far at every single step, instead of reusing
        past_key_value / memory_states incrementally the way a
        transformer-only model normally would during generation. That makes
        total generation cost O(L^2) in the final sequence length rather than
        O(L). It is correct, just not efficient -- fine for research-loop
        generation checks and demos, not for production serving. If you need
        fast interactive generation, the next step is adding incremental
        Mamba state caching (conv buffer + SSM state per layer) and threading
        it through the same way `past_key_value` is threaded through
        SlidingWindowGQA today.

        Args:
            input_ids: [B, L_prompt] prompt token ids.
            max_new_tokens: number of tokens to generate after the prompt.
            temperature: softmax temperature; lower = more deterministic.
            top_k: if set, restrict sampling to the top_k highest-logit tokens.
            top_p: if set, restrict sampling to the smallest nucleus with
                cumulative probability >= top_p (applied after top_k, if both given).
            do_sample: if False, use greedy argmax decoding (temperature/top_k/
                top_p are ignored).
            eos_token_id: stop generating (per-sequence) once this token is
                produced. Defaults to config.eos_token_id.
            attention_mask: [B, L_prompt] optional padding mask matching input_ids.

        Returns:
            Tensor [B, L_prompt + up_to_max_new_tokens] of generated token ids.
            Sequences that hit eos early are padded with eos_token_id for the
            remaining steps so the batch stays a single rectangular tensor.
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

        try:
            for _ in range(max_new_tokens):
                seq_len = generated.size(1)
                if seq_len > self.config.max_position_embeddings:
                    # Keep only the most recent window so RotaryEmbedding
                    # doesn't raise on sequences longer than it was built for.
                    window = self.config.max_position_embeddings
                    model_input_ids = generated[:, -window:]
                    model_attention_mask = attention_mask[:, -window:]
                else:
                    model_input_ids = generated
                    model_attention_mask = attention_mask

                out = self.forward(
                    input_ids=model_input_ids,
                    attention_mask=model_attention_mask,
                    use_cache=False,
                )
                next_token_logits = out.logits[:, -1, :] / max(temperature, 1e-8)

                if do_sample:
                    if top_k is not None:
                        next_token_logits = _top_k_filter(next_token_logits, top_k)
                    if top_p is not None:
                        next_token_logits = _top_p_filter(next_token_logits, top_p)
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Sequences that already hit eos just keep emitting eos so the
                # batch stays rectangular.
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
        finally:
            self.train(was_training)

        return generated


# =============================================================================
# Falsification-test helpers
# =============================================================================


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_test3_null_baseline_config(
    hybrid_config: HybridMambaMoEConfig,
) -> HybridMambaMoEConfig:
    """
    Test 3 (design doc v2.0): the null hypothesis for "do we need explicit
    memory at all" is a Mamba branch with a larger state dimension instead
    of a second memory subsystem. This builds a config with:
        - use_dual_memory = False (no CompressiveMemoryBank params)
        - mamba_state_size increased to roughly absorb the parameter budget
          that the memory banks would otherwise have used.

    The state-size bump below is a rough heuristic, not an exact parameter
    match -- after constructing both models, call count_trainable_params()
    on each and fine-tune mamba_state_size until they're within a few
    percent of each other before running the comparison.
    """
    import copy

    null_config = copy.deepcopy(hybrid_config)
    null_config.use_dual_memory = False
    # Heuristic bump; verify against count_trainable_params and adjust.
    null_config.mamba_state_size = hybrid_config.mamba_state_size * 4
    return null_config
