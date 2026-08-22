"""Model configuration dataclasses."""

import json
from dataclasses import asdict, dataclass
from typing import Any

from torch import Tensor


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

    # QK-Normalization (applies RMSNorm per head to query and key states).
    use_qk_norm: bool = False

    # Number of initial token slots permanently retained in sliding-window KV cache (StreamingLLM attention sinks).
    attention_sink_size: int = 0

    # Optional z-loss on final vocabulary logits (PaLM / OLMo-2 style stability regularizer).
    final_logit_z_loss_coef: float = 0.0

    # Optional layer topology pattern (e.g. ["hybrid", "mamba_only", ...]). None defaults to all hybrid/standard.
    layer_types: list[str] | None = None

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
