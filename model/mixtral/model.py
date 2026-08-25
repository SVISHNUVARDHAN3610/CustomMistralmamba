"""Mixtral baseline model."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.core.config import MixtralConfig
from model.layers.attention import SlidingWindowGQA
from model.layers.moe import DroplessMoELayer, MOERouter, SwiGLUExpert
from model.layers.norm import RMSNorm
from model.layers.rope import RotaryEmbedding
from model.layers.sampling import top_k_filter as _top_k_filter
from model.layers.sampling import top_p_filter as _top_p_filter


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
        layers = []
        for _ in range(config.num_layers):
            layer = MixtralDecoderLayer(config)
            layer.attention_block.rotary_emb = shared_rotary_emb
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list | None = None,
        use_cache: bool = False,
        past_seen_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list | None]:
        hidden_states = self.embed_tokens(input_ids)
        seq_len = hidden_states.size(1)

        if position_ids is None:
            # Incremental decoding must continue RoPE at the cached offset,
            # not restart at 0 (which silently scrambles relative distances).
            offset = past_seen_tokens if past_seen_tokens is not None else 0
            position_ids = torch.arange(
                offset,
                offset + seq_len,
                dtype=torch.long,
                device=hidden_states.device,
            ).unsqueeze(0)
            if position_ids.size(0) == 1 and hidden_states.size(0) > 1:
                position_ids = position_ids.expand(hidden_states.size(0), -1)

        if int(position_ids.max().item()) >= self.config.max_position_embeddings:
            raise ValueError(
                f"position_ids exceed max_position_embeddings="
                f"{self.config.max_position_embeddings}; fixed caches cannot grow."
            )

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
        past_seen_tokens: int | None = None,
    ) -> MixtralTrainingOutput:

        hidden_states, aux_loss, z_loss, present_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            past_seen_tokens=past_seen_tokens,
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
            if self.config.vocab_z_loss_coef > 0.0:
                valid = labels != self.config.label_ignore_index
                if valid.any():
                    loss = (
                        loss
                        + self.config.vocab_z_loss_coef
                        * torch.logsumexp(logits[valid].float(), dim=-1).pow(2).mean()
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
        Autoregressive generation with an incremental KV cache, mirroring the
        hybrid family's ``generate()`` (minus the memory/Mamba machinery).

        ``eos_token_id=None`` falls back to ``config.eos_token_id``; any
        negative value (e.g. ``-1``) *disables* EOS early-stopping entirely.
        Per-row RoPE positions honor right-padding; pad K/V entries still
        occupy sliding-window cache slots.
        """
        if input_ids.numel() == 0 or input_ids.size(1) == 0:
            raise ValueError("generate() requires a non-empty prompt.")
        device = input_ids.device
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )
        eos_stopping_enabled = eos_token_id is None or eos_token_id >= 0

        generated = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(generated)

        batch_size = generated.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        prompt_len = generated.size(1)
        max_positions = self.config.max_position_embeddings
        if prompt_len > max_positions:
            raise ValueError(
                f"Prompt length {prompt_len} exceeds "
                f"max_position_embeddings={max_positions}."
            )
        if prompt_len + max_new_tokens > max_positions:
            raise ValueError(
                f"prompt_len+max_new_tokens={prompt_len + max_new_tokens} "
                f"exceeds max_position_embeddings={max_positions}."
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

        past_key_values: list | None = None
        past_seen_tokens = 0
        out: MixtralTrainingOutput | None = None

        # Prefill in one pass (no recurrent state to chunk around); pass
        # the FULL prefix mask so the attention layer's sink-eviction mask
        # truncation stays aligned with KV slots.
        prefill_pos = (attn_buf[:, :prompt_len].cumsum(dim=-1) - 1).clamp(min=0)
        out = self.forward(
            input_ids=generated_buf[:, :prompt_len],
            attention_mask=attn_buf[:, :prompt_len],
            position_ids=prefill_pos,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = out.past_key_values
        past_seen_tokens = prompt_len

        for _step in range(max_new_tokens):
            logits = out.logits[:, -1, :]
            if do_sample:
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

            step_input = next_token.clone()
            step_input[~active] = self.config.pad_token_id
            step_position_ids = (
                attn_buf[:, :cur_len].sum(dim=1, keepdim=True).long() - 1
            ).clamp(min=0)

            out = self.forward(
                input_ids=step_input,
                attention_mask=attn_buf[:, :cur_len],
                position_ids=step_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                past_seen_tokens=past_seen_tokens,
            )
            past_key_values = out.past_key_values
            past_seen_tokens += 1

        return generated_buf[:, :cur_len]
