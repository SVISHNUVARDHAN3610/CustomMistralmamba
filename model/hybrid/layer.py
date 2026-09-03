"""Hybrid decoder layer."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from model.core.config import HybridMambaMoEConfig, MambaCache
from model.core.dtype import _promote_fp32, _restore_dtype
from model.hybrid.losses import (
    HybridLayerAuxLosses,
    _expert_loss_schedule,
    assoc_state_norm_loss,
    associative_retrieval_loss,
    combine_read_utilization_loss,
    fusion_balance_loss,
    masked_token_mse,
    memory_slot_diversity_loss,
    ssm_state_norm_loss,
    write_gate_entropy_loss,
)
from model.hybrid.mamba import MambaBlock
from model.hybrid.memory import (
    CompressiveMemoryBank,
    HybridMemoryState,
    MemoryWriteBuffer,
    batched_dual_memory_read,
    batched_dual_memory_write,
)
from model.layers.attention import SlidingWindowGQA
from model.layers.fusion import TokenGatedFusion
from model.layers.moe import DroplessMoELayer, MOERouter, SwiGLUExpert
from model.layers.norm import RMSNorm


def _gate_monitoring_stats(
    prefix: str, gate: Tensor, saturation_threshold: float
) -> dict[str, Tensor]:
    """Return scalar gate-distribution telemetry without retaining autograd state."""
    gate_f = gate.detach().float()
    low = gate_f <= saturation_threshold
    high = gate_f >= 1.0 - saturation_threshold
    return {
        f"{prefix}_mean": gate_f.mean(),
        # Moments and bucket fractions are mergeable across FSDP ranks by the
        # trainers' existing sum/count reduction. Per-rank extrema/quantiles
        # are deliberately avoided because averaging them is not a global
        # extremum/quantile.
        f"{prefix}_mean_square": gate_f.square().mean(),
        f"{prefix}_overwrite_saturated_fraction": low.float().mean(),
        f"{prefix}_interior_fraction": (~(low | high)).float().mean(),
        f"{prefix}_retention_saturated_fraction": high.float().mean(),
    }


def _memory_state_monitoring_stats(
    prefix: str, state: Tensor, gamma: Tensor | None
) -> dict[str, Tensor]:
    """Return mergeable per-slot state-norm telemetry used by T-7."""
    per_slot_mse = state.detach().float().pow(2).mean(dim=-1).reshape(-1)
    stats = {
        f"{prefix}_norm": per_slot_mse.mean(),
        f"{prefix}_norm_mean_square": per_slot_mse.square().mean(),
    }
    if gamma is not None:
        gamma_f = gamma.detach().float().to(device=per_slot_mse.device)
        half_gamma = 0.5 * gamma_f
        twice_gamma = 2.0 * gamma_f
        below_half = per_slot_mse <= half_gamma
        near_below = (per_slot_mse > half_gamma) & (per_slot_mse <= gamma_f)
        near_above = (per_slot_mse > gamma_f) & (per_slot_mse <= twice_gamma)
        far_above = per_slot_mse > twice_gamma
        stats.update(
            {
                f"{prefix}_norm_below_half_gamma_fraction": below_half.float().mean(),
                f"{prefix}_norm_half_to_one_gamma_fraction": near_below.float().mean(),
                f"{prefix}_norm_one_to_two_gamma_fraction": near_above.float().mean(),
                f"{prefix}_norm_above_two_gamma_fraction": far_above.float().mean(),
                f"{prefix}_norm_above_gamma_fraction": (near_above | far_above)
                .float()
                .mean(),
            }
        )
    return stats


def _hybrid_layer_forward(
    layer: HybridDecoderLayer,
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

        # Zero pad positions' raw branch outputs BEFORE fusion so the fusion
        # gate sees identical inputs in both the dual-memory arm and the
        # Jamba-like ablation arm (use_dual_memory=False). The residual output
        # is masked again after fusion either way; this keeps `fusion_gate`
        # stats/loss comparable across ablation arms on padded batches.
        if hidden_mask is not None:
            attn_out = attn_out * hidden_mask
            mamba_out = mamba_out * hidden_mask

        if self.use_dual_memory:
            assert memory_state is not None
            a_mem, s_mem = memory_state
            prev_a_mem = a_mem
            prev_s_mem = s_mem

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
                new_buf.append_single_token(buf_attn, buf_mamba, token_attention_mask)
            else:
                new_buf.append(buf_attn, buf_mamba, token_attention_mask)

            if skip_memory_write:
                new_memory_state = memory_state
                new_write_buffer = new_buf
            else:
                buf_attn_cat, buf_mamba_cat, buf_valid = new_buf.materialize()
                # Exact accumulated validity (never reconstruct prior pads as ones).
                write_mask: Tensor | None = None
                if buf_valid is not None:
                    write_mask = buf_valid.to(
                        dtype=(
                            token_attention_mask.dtype
                            if token_attention_mask is not None
                            else torch.long
                        )
                    )
                if active_batch_mask is not None and write_mask is not None:
                    write_mask = write_mask * active_batch_mask.unsqueeze(-1).to(
                        dtype=write_mask.dtype
                    )

                buf_len = buf_attn_cat.size(1)
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
                    **_gate_monitoring_stats(
                        "attn_write_gate",
                        a_write_gate,
                        cfg.gate_saturation_threshold,
                    ),
                    **_gate_monitoring_stats(
                        "state_write_gate",
                        s_write_gate,
                        cfg.gate_saturation_threshold,
                    ),
                    **_memory_state_monitoring_stats(
                        "attn_mem",
                        new_a_mem,
                        getattr(self, "assoc_norm_gamma", None),
                    ),
                    **_memory_state_monitoring_stats(
                        "state_mem",
                        new_s_mem,
                        getattr(self, "assoc_norm_gamma", None),
                    ),
                }

                if self.training and self.use_auxiliary_losses:
                    # FP16 AMP: recon/gate/slot paths need fp32 attention + logs.
                    with torch.autocast(
                        device_type=buf_attn_cat.device.type, enabled=False
                    ):
                        buf_a = _promote_fp32(buf_attn_cat)
                        buf_m = _promote_fp32(buf_mamba_cat)
                        sum_a = _promote_fp32(a_summary)
                        sum_s = _promote_fp32(s_summary)
                        write_valid = (
                            write_mask.bool() if write_mask is not None else None
                        )
                        row_has_valid = (
                            write_valid.any(dim=-1) if write_valid is not None else None
                        )
                        attn_recon_out = self.attn_memory_bank.recon_decoder(
                            buf_a, sum_a
                        )
                        mamba_recon_out = self.state_memory_bank.recon_decoder(
                            buf_m, sum_s
                        )
                        attn_recon_tok = (
                            (buf_a - attn_recon_out).pow(2).mean(dim=-1).sqrt()
                        )
                        mamba_recon_tok = (
                            (buf_m - mamba_recon_out).pow(2).mean(dim=-1).sqrt()
                        )
                        attn_recon = masked_token_mse(
                            attn_recon_out, buf_a, write_valid
                        )
                        mamba_recon = masked_token_mse(
                            mamba_recon_out, buf_m, write_valid
                        )
                        del attn_recon_out, mamba_recon_out
                        attn_assoc = associative_retrieval_loss(
                            self.attn_memory_bank,
                            buf_a,
                            _promote_fp32(new_a_mem),
                            attn_recon_tok,
                            cfg.assoc_sample_count,
                            write_mask,
                            err_clip=cfg.assoc_err_clip,
                        )
                        mamba_assoc = associative_retrieval_loss(
                            self.state_memory_bank,
                            buf_m,
                            _promote_fp32(new_s_mem),
                            mamba_recon_tok,
                            cfg.assoc_sample_count,
                            write_mask,
                            err_clip=cfg.assoc_err_clip,
                        )
                        gate_loss = write_gate_entropy_loss(
                            a_write_gate,
                            cfg.gate_entropy_eps,
                            row_mask=row_has_valid,
                            saturation_threshold=cfg.gate_saturation_threshold,
                            saturation_penalty_weight=cfg.gate_saturation_penalty_weight,
                        ) + write_gate_entropy_loss(
                            s_write_gate,
                            cfg.gate_entropy_eps,
                            row_mask=row_has_valid,
                            saturation_threshold=cfg.gate_saturation_threshold,
                            saturation_penalty_weight=cfg.gate_saturation_penalty_weight,
                        )
                        slot_loss = memory_slot_diversity_loss(
                            new_a_mem,
                            new_s_mem,
                            cfg.slot_similarity_margin,
                            cfg.slot_cross_bank_alpha,
                        )
                        # Bound the post-write bank state (ssm_state_norm_loss
                        # analog, T-7). Zero state contributes nothing (hinge at 0),
                        # so a Jamba-like zero-bank start never adds a bias.
                        assoc_norm_gamma = getattr(self, "assoc_norm_gamma", None)
                        if assoc_norm_gamma is not None:
                            assoc_norm_loss = assoc_state_norm_loss(
                                new_a_mem, assoc_norm_gamma
                            ) + assoc_state_norm_loss(new_s_mem, assoc_norm_gamma)
                        else:
                            assoc_norm_loss = torch.tensor(
                                0.0, device=x.device, dtype=x.dtype
                            )
                    layer_aux = HybridLayerAuxLosses(
                        recon=_restore_dtype((attn_recon + mamba_recon) / 2.0, x.dtype),
                        assoc=_restore_dtype((attn_assoc + mamba_assoc) / 2.0, x.dtype),
                        gate=_restore_dtype(gate_loss / 2.0, x.dtype),
                        read=layer_aux.read,
                        fusion=layer_aux.fusion,
                        expert=layer_aux.expert,
                        ssm=layer_aux.ssm,
                        slot=_restore_dtype(slot_loss, x.dtype),
                        assoc_norm=_restore_dtype(assoc_norm_loss / 2.0, x.dtype),
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
                self.training
                and self.use_auxiliary_losses
                and cfg.lambda_expert > 0.0
                and expert_scale > 0.0
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
            fusion_loss = fusion_balance_loss(
                fusion_gate, target=cfg.fusion_balance_target
            )
            ssm_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            if ssm_state is not None and cfg.lambda_ssm > 0.0:
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
                assoc_norm=layer_aux.assoc_norm,
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
