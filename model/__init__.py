"""Hybrid Mamba-MoE model package.

Contains two model families:

1. MixtralConfig / MixtralForCausalLM -- sliding-window-GQA + Top-2 MoE
   baseline (control for ablations).

2. HybridMambaMoEConfig / HybridForCausalLM -- Hybrid Mamba-MoE with Dual
   Memory (research/research.md v2.0).
"""

from model.core import (
    MEMORY_NAN_FIX_ID,
    HybridMambaMoEConfig,
    MambaCache,
    MixtralConfig,
    build_adamw_param_groups,
    build_test3_null_baseline_config,
    count_trainable_params,
    split_muon_adam_params,
)
from model.hybrid import (
    CompressiveMemoryBank,
    HybridAuxiliaryLossBreakdown,
    HybridDecoderLayer,
    HybridForCausalLM,
    HybridLayerAuxLosses,
    HybridMemoryState,
    HybridModel,
    HybridTrainingOutput,
    MambaBlock,
    MemoryWriteBuffer,
    _aux_loss_schedule,
    _batched_memory_summarize,
    _compute_batch_has_padding,
    _expert_loss_schedule,
    _hybrid_layer_forward,
    _write_buffer_token_len,
    associative_retrieval_loss,
    batched_dual_memory_read,
    batched_dual_memory_write,
    fused_mamba_scan_available,
    get_mamba_scan_stats,
    log_mamba_backend,
    probe_mamba_scan_timing,
    reset_mamba_scan_stats,
)
from model.layers import DroplessMoELayer, MOERouter, SwiGLUExpert
from model.mixtral import MixtralForCausalLM, MixtralModel, MixtralTrainingOutput

__all__ = [
    "MEMORY_NAN_FIX_ID",
    "CompressiveMemoryBank",
    "DroplessMoELayer",
    "HybridAuxiliaryLossBreakdown",
    "HybridDecoderLayer",
    "HybridForCausalLM",
    "HybridLayerAuxLosses",
    "HybridMambaMoEConfig",
    "HybridMemoryState",
    "HybridModel",
    "HybridTrainingOutput",
    "MOERouter",
    "MambaBlock",
    "MambaCache",
    "MemoryWriteBuffer",
    "MixtralConfig",
    "MixtralForCausalLM",
    "MixtralModel",
    "MixtralTrainingOutput",
    "SwiGLUExpert",
    "_aux_loss_schedule",
    "_batched_memory_summarize",
    "_compute_batch_has_padding",
    "_expert_loss_schedule",
    "_hybrid_layer_forward",
    "_write_buffer_token_len",
    "associative_retrieval_loss",
    "batched_dual_memory_read",
    "batched_dual_memory_write",
    "build_adamw_param_groups",
    "build_test3_null_baseline_config",
    "count_trainable_params",
    "fused_mamba_scan_available",
    "get_mamba_scan_stats",
    "log_mamba_backend",
    "probe_mamba_scan_timing",
    "reset_mamba_scan_stats",
    "split_muon_adam_params",
]
