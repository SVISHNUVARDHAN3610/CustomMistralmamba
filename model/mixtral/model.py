"""Mixtral baseline model."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from model.core.config import MixtralConfig
from model.layers.attention import SlidingWindowGQA
from model.layers.moe import DroplessMoELayer, MOERouter, SwiGLUExpert
from model.layers.norm import RMSNorm
from model.layers.rope import RotaryEmbedding


class MixtralDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MixtralConfig,
        rotary_emb: RotaryEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.rmsnorm_attn = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_block = SlidingWindowGQA(config, rotary_emb=rotary_emb)
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
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.layers = nn.ModuleList(
            [
                MixtralDecoderLayer(config, rotary_emb=self.rotary_emb)
                for _ in range(config.num_layers)
            ]
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
