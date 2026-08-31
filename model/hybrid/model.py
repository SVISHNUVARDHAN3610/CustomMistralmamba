"""Hybrid Mamba-MoE model and causal LM head."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from model.core.config import HybridMambaMoEConfig, MambaCache
from model.hybrid.layer import (
    HybridDecoderLayer,
    HybridMemoryState,
    _hybrid_layer_forward,
)
from model.hybrid.losses import (
    HybridAuxiliaryLossBreakdown,
    HybridLayerAuxLosses,
    _aux_loss_schedule,
    _expert_loss_schedule,
)
from model.hybrid.mamba import _compute_batch_has_padding, _validate_hybrid_cache_states
from model.hybrid.memory import (
    MemoryWriteBuffer,
    _materialize_write_buffer,
    batched_dual_memory_write,
)
from model.layers.norm import RMSNorm
from model.layers.rope import RotaryEmbedding
from model.layers.sampling import top_k_filter as _top_k_filter
from model.layers.sampling import top_p_filter as _top_p_filter


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
        # One RoPE table shared by every attention layer (identical head_dim /
        # theta everywhere): avoids num_layers duplicate cos/sin caches. The
        # buffers are non-persistent, so state_dicts are unchanged.
        shared_rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        layers: list[HybridDecoderLayer] = []
        for _ in range(config.num_layers):
            layer = HybridDecoderLayer(config)
            layer.attention_block.rotary_emb = shared_rotary_emb
            layers.append(layer)
        if config.use_torch_compile:
            if config.gradient_checkpointing:
                raise ValueError(
                    "use_torch_compile and gradient_checkpointing are mutually "
                    "exclusive."
                )
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


def _per_row_position_ids(attention_mask: Tensor) -> Tensor:
    """Absolute RoPE positions for right-padded rows.

    Each token's position is its index among the row's *valid* tokens, so
    shorter rows in a padded batch are not offset by their pad count. Pad
    slots get their left neighbor's position (clamped at 0); they are masked
    out of attention anyway.
    """
    return (attention_mask.cumsum(dim=-1) - 1).clamp(min=0).to(torch.long)


# NOTE on `config.use_cuda_graph`: the previous CUDA-graph single-token decode
# runner was removed (audit finding H2). Warm-up passes ran against the live
# conv/ssm/KV caches (MambaBlock.step() shifts conv_state in place), and graph
# replay froze KV/SSM/memory state at capture-time addresses whenever an eager
# flush step interleaved — silently producing garbage continuations. A correct
# implementation needs persistent captured buffers (in-place copies into
# static caches + ring-buffer KV). Until that exists the flag stays as a
# documented no-op: generate() warns once and always decodes eagerly.


class HybridForCausalLM(nn.Module):
    def __init__(self, config: HybridMambaMoEConfig) -> None:
        super().__init__()
        self.config = config
        self._input_require_grads_hook: torch.utils.hooks.RemovableHandle | None = None
        if config.gradient_checkpointing_use_reentrant:
            raise ValueError(
                "HybridForCausalLM supports only non-reentrant gradient "
                "checkpointing; set gradient_checkpointing_use_reentrant=False."
            )
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
        if config.gradient_checkpointing:
            self.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    @property
    def is_gradient_checkpointing(self) -> bool:
        return bool(self.config.gradient_checkpointing)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def enable_input_require_grads(self) -> None:
        """Require gradients on embedding *outputs* without unfreezing weights.

        This mirrors the Hugging Face checkpointing contract: if input embedding
        weights are frozen, the first checkpointed layer still receives an input
        participating in autograd. The hook is harmless when embeddings are
        already trainable and is installed at most once.
        """
        if self._input_require_grads_hook is not None:
            return

        def _require_grads(_module: nn.Module, _inputs: tuple, output: Tensor) -> None:
            output.requires_grad_(True)

        self._input_require_grads_hook = self.get_input_embeddings().register_forward_hook(
            _require_grads
        )

    def disable_input_require_grads(self) -> None:
        if self._input_require_grads_hook is not None:
            self._input_require_grads_hook.remove()
            self._input_require_grads_hook = None

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: dict[str, object] | None = None,
    ) -> None:
        """Enable layer recomputation with the required non-reentrant engine."""
        kwargs = gradient_checkpointing_kwargs or {"use_reentrant": False}
        if kwargs.get("use_reentrant", False) is not False:
            raise ValueError(
                "Only use_reentrant=False is supported for gradient checkpointing."
            )
        if self.config.use_torch_compile:
            raise ValueError(
                "torch.compile and gradient checkpointing are mutually exclusive."
            )
        self.config.gradient_checkpointing = True
        self.config.gradient_checkpointing_use_reentrant = False
        self.config.use_cache = False
        self.enable_input_require_grads()

    def gradient_checkpointing_disable(self) -> None:
        self.config.gradient_checkpointing = False
        self.config.gradient_checkpointing_use_reentrant = False
        self.config.use_cache = True
        self.disable_input_require_grads()

    def train(self, mode: bool = True) -> HybridForCausalLM:
        super().train(mode)
        # Cache materialization and layer recomputation are incompatible during
        # training. Eval/generation explicitly re-enable the config preference;
        # forward still accepts an explicit use_cache argument as before.
        self.config.use_cache = not (mode and self.is_gradient_checkpointing)
        return self

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

    def _count_valid_label_tokens(
        self, labels: Tensor, attention_mask: Tensor | None = None
    ) -> int:
        labels = self._apply_label_ignore(labels, attention_mask)
        return int((labels != self.config.label_ignore_index).sum().item())

    def _vocab_z_loss(self, logits: Tensor) -> Tensor:
        """PaLM/OLMo-style z-loss on final logits: mean(logsumexp^2)."""
        return torch.logsumexp(logits.float(), dim=-1).pow(2).mean()

    def _compute_ce_loss(
        self,
        hidden_states: Tensor,
        labels: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Token-weighted CE mean over non-ignored labels plus vocab z-loss.

        When batches contain padding, lm_head runs only on valid label positions
        so peak activation memory is [N_valid, V] instead of [B*L, V].

        Returns (ce_loss, vocab_z_loss); z-loss is 0 when disabled.
        """
        labels = self._apply_label_ignore(labels, attention_mask)
        ignore_index = self.config.label_ignore_index
        zero = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        valid = labels != ignore_index
        if not valid.any():
            return zero, zero
        hidden_valid = hidden_states[valid]
        logits_valid = self.lm_head(hidden_valid)
        ce_loss = F.cross_entropy(logits_valid, labels[valid], reduction="mean")
        z_loss = (
            self._vocab_z_loss(logits_valid)
            if self.config.vocab_z_loss_coef > 0.0
            else zero
        )
        return ce_loss, z_loss

    def _stream_chunk_ce_loss(
        self,
        hidden_states: Tensor,
        labels: Tensor,
        attention_mask: Tensor | None,
        materialize_logits: bool,
    ) -> tuple[Tensor | None, Tensor, int, Tensor]:
        """
        Per-chunk CE without retaining full [B, L, V] logits unless requested.

        Returns (optional_logits_chunk, ce_mean_over_valid, valid_token_count,
        vocab_z_loss_for_chunk).
        """
        labels = self._apply_label_ignore(labels, attention_mask)
        ignore_index = self.config.label_ignore_index
        valid = labels != ignore_index
        n_valid = int(valid.sum().item())
        zero = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        if n_valid == 0:
            return None, zero, 0, zero

        if materialize_logits:
            chunk_logits = self.lm_head(hidden_states)
            ce_loss = F.cross_entropy(
                chunk_logits.view(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=ignore_index,
            )
            z_loss = (
                self._vocab_z_loss(chunk_logits[valid])
                if self.config.vocab_z_loss_coef > 0.0
                else zero
            )
            return chunk_logits, ce_loss, n_valid, z_loss

        # VRAM: lm_head only on supervised tokens — avoids [B*L, V] peak logits.
        hidden_valid = hidden_states[valid]
        logits_valid = self.lm_head(hidden_valid)
        ce_loss = F.cross_entropy(logits_valid, labels[valid], reduction="mean")
        z_loss = (
            self._vocab_z_loss(logits_valid)
            if self.config.vocab_z_loss_coef > 0.0
            else zero
        )
        return None, ce_loss, n_valid, z_loss

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
        if self.training and self.is_gradient_checkpointing and use_cache:
            raise ValueError(
                "use_cache=True is incompatible with gradient-checkpointed "
                "training. Call eval() for cached inference or omit use_cache."
            )
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
        logits = self.lm_head(hidden_states) if self.config.return_logits else None

        loss = None
        ce_loss = None
        vocab_z_total = torch.tensor(
            0.0, device=hidden_states.device, dtype=hidden_states.dtype
        )
        if labels is not None:
            if self.config.return_logits:
                labels = self._apply_label_ignore(labels, attention_mask)
                loss_fct = nn.CrossEntropyLoss(
                    ignore_index=self.config.label_ignore_index
                )
                assert logits is not None
                ce_loss = loss_fct(logits.view(-1, self.vocab_size), labels.reshape(-1))
                if self.config.vocab_z_loss_coef > 0.0 and logits is not None:
                    valid_labels = labels != self.config.label_ignore_index
                    if valid_labels.any():
                        vocab_z_total = self._vocab_z_loss(logits[valid_labels])
            else:
                ce_loss, vocab_z_total = self._compute_ce_loss(
                    hidden_states, labels, attention_mask
                )
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
                + (self.config.vocab_z_loss_coef * vocab_z_total)
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
        """BPTT through memory banks within one backward pass.

        VRAM: each chunk streams CE via _stream_chunk_ce_loss so peak logits are
        [N_valid, V] (not [B, chunk, V]) when return_logits=False; chunk logits
        are not concatenated unless return_logits=True.
        """
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
        # here we weight across internal chunks by chunk length and divide by
        # the summed chunk lengths (NOT the valid-token count: the eight layer
        # aux terms and router aux/z are means over all chunk positions, so
        # mixing a valid-token denominator with chunk-length weights skews
        # their scale under right-padding). CE and vocab-z ARE valid-token
        # quantities and keep the token_weight normalization.
        total_aux = torch.tensor(0.0, device=device)
        total_z = torch.tensor(0.0, device=device)
        gate_stat_sums: dict[str, Tensor] = {}
        gate_stat_counts: dict[str, int] = {}
        token_weight = 0
        total_chunk_len = 0
        ce_loss_sum = torch.tensor(0.0, device=device)
        vocab_z_sum = torch.tensor(0.0, device=device)
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
            chunk_logits: Tensor | None = None
            if labels is not None:
                chunk_labels = labels[:, start:end]
                chunk_logits, chunk_ce, n_valid, chunk_z = self._stream_chunk_ce_loss(
                    hidden_states,
                    chunk_labels,
                    chunk_mask,
                    materialize_logits,
                )
                if n_valid > 0:
                    ce_loss_sum = ce_loss_sum + chunk_ce * n_valid
                    vocab_z_sum = vocab_z_sum + chunk_z * n_valid
                    token_weight += n_valid
                if materialize_logits and chunk_logits is not None:
                    logits_chunks.append(chunk_logits)
                del chunk_logits
            elif materialize_logits:
                chunk_logits = self.lm_head(hidden_states)
                logits_chunks.append(chunk_logits)
                del chunk_logits
            total_aux = total_aux + aux_loss * chunk_len
            total_z = total_z + z_loss * chunk_len
            total_chunk_len += chunk_len
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
        chunk_tw = max(total_chunk_len, 1)
        aux_loss = total_aux / chunk_tw
        z_loss = total_z / chunk_tw
        all_gate_stats = {
            k: gate_stat_sums[k] / gate_stat_counts[k] for k in gate_stat_sums
        }
        auxiliary_losses = HybridAuxiliaryLossBreakdown(
            recon=aux_weighted.recon / chunk_tw,
            assoc=aux_weighted.assoc / chunk_tw,
            gate=aux_weighted.gate / chunk_tw,
            read=aux_weighted.read / chunk_tw,
            fusion=aux_weighted.fusion / chunk_tw,
            expert=aux_weighted.expert / chunk_tw,
            ssm=aux_weighted.ssm / chunk_tw,
            slot=aux_weighted.slot / chunk_tw,
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
                + (self.config.vocab_z_loss_coef * vocab_z_sum / max(token_weight, 1))
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
            buf_attn, buf_mamba, buf_mask = _materialize_write_buffer(buf)
            assert buf_attn is not None and buf_mamba is not None
            write_mask = buf_mask.to(dtype=torch.long) if buf_mask is not None else None
            new_a, _, _, new_s, _, _ = batched_dual_memory_write(
                layer.attn_memory_bank,
                layer.state_memory_bank,
                buf_attn,
                buf_mamba,
                a_mem,
                s_mem,
                attention_mask=write_mask,
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

        ``eos_token_id=None`` falls back to ``config.eos_token_id``; any
        negative value (e.g. ``-1``) *disables* EOS early-stopping entirely.

        Works regardless of ``config.return_logits``: the flag is forced on
        for the duration of generation (as-trained checkpoints ship with it
        off for VRAM). Note for right-padded batches: per-row RoPE positions
        honor the padding mask, but pad K/V entries still occupy sliding-window
        cache slots, so very long pads relative to the window can crowd out
        real context at the tail.
        """
        was_training = self.training
        self.eval()

        device = input_ids.device
        if input_ids.numel() == 0 or input_ids.size(1) == 0:
            raise ValueError("generate() requires a non-empty prompt.")
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )
        # Negative sentinel = caller explicitly wants no EOS early-stop.
        eos_stopping_enabled = eos_token_id is None or eos_token_id >= 0

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
        out: HybridTrainingOutput | None = None

        # As-trained configs set return_logits=False (saves VRAM during
        # training); generation needs last-step logits, so force materialization
        # for the duration and restore afterwards. Configs defaulting to True
        # are unaffected.
        prev_return_logits = self.config.return_logits
        self.config.return_logits = True
        if self.config.use_cuda_graph:
            warnings.warn(
                "config.use_cuda_graph=True is a no-op: the CUDA-graph decode "
                "path was removed because it silently corrupted recurrent "
                "state (capture-time cache addresses frozen across replay). "
                "Decoding eagerly.",
                stacklevel=2,
            )

        try:
            # Chunked prefill so memory writes match training chunk size. The
            # FULL prefix mask is passed (not window-truncated): the attention
            # layer truncates mask columns in lockstep with KV eviction, which
            # keeps sink slots aligned with absolute positions 0..K-1 even
            # when interior pads exist.
            for start in range(0, prompt_len, write_interval):
                end = min(start + write_interval, prompt_len)
                chunk = generated_buf[:, start:end]
                chunk_mask = attn_buf[:, :end]
                chunk_pos = _per_row_position_ids(chunk_mask)[:, start:end]
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
            assert out.logits is not None

            for _step in range(max_new_tokens):
                logits = out.logits[:, -1, :]
                if do_sample:
                    # fp32 sampling: softmax/multinomial in bf16/fp16 lose
                    # low-probability tails and can zero them out entirely.
                    next_token_logits = logits.float() / max(temperature, 1e-8)
                    if top_k is not None:
                        next_token_logits = _top_k_filter(next_token_logits, top_k)
                    if top_p is not None:
                        next_token_logits = _top_p_filter(next_token_logits, top_p)
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits.float(), dim=-1, keepdim=True)

                if eos_stopping_enabled:
                    next_token = torch.where(
                        finished.unsqueeze(-1),
                        torch.full_like(next_token, int(eos_token_id)),
                        next_token,
                    )

                generated_buf[:, cur_len : cur_len + 1] = next_token
                attn_buf[:, cur_len : cur_len + 1] = (~finished).long().unsqueeze(-1)
                cur_len += 1
                if eos_stopping_enabled:
                    finished = finished | (next_token.squeeze(-1) == eos_token_id)
                    if finished.all():
                        break

                active = ~finished
                if not active.any():
                    break

                # Per-row decode position: index among this row's valid tokens
                # (padded prompts must not inflate shorter rows' positions).
                step_position_ids = (
                    attn_buf[:, :cur_len].sum(dim=1, keepdim=True).long() - 1
                ).clamp(min=0)
                step_attn_mask = attn_buf[:, :cur_len]

                step_input = next_token.clone()
                step_input[~active] = self.config.pad_token_id
                tokens_in_write_buffer += 1
                do_memory_write = tokens_in_write_buffer >= write_interval
                decode_accumulate = not do_memory_write

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
            self.config.return_logits = prev_return_logits
            self.train(was_training)

        return generated_buf[:, :cur_len]
