"""Response-token PPO math and a critic on the repository's hybrid policy.

Rollouts and replay share the SFT chunked-memory computation. All forwards go
through Module.__call__; no inference cache or old-policy model is retained.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.core.config import HybridMambaMoEConfig
from model.hybrid.model import HybridForCausalLM
from model.layers.sampling import top_k_filter, top_p_filter

from .config import PPOConfig


class PolicyValueModel(HybridForCausalLM):
    """Original hybrid weights/LM head plus optional per-state scalar critic.

    Episodes start with empty memory. Replay uses fixed SFT memory chunks, with
    differentiable memory between chunks and no cross-episode state. Evaluation
    mode disables dropout/auxiliary losses during BOTH rollout and optimization.
    """

    def __init__(self, config: HybridMambaMoEConfig, with_value: bool = True):
        super().__init__(config)
        if config.capacity_factor is not None:
            raise ValueError("PPO requires batch-independent dropless MoE routing")
        self.value_head = nn.Linear(config.hidden_size, 1) if with_value else None
        if self.value_head is not None:
            nn.init.zeros_(self.value_head.weight)
            nn.init.zeros_(self.value_head.bias)
        # Outer layer checkpoint wrappers own recomputation under FSDP2.
        config.gradient_checkpointing = False
        config.use_torch_compile = False
        self.eval()

    @property
    def layers(self) -> nn.ModuleList:
        return self.model.layers

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        positions: Tensor,
        tokens: Tensor | None = None,
        temperature: float = 1.0,
    ):
        memory = None
        hidden_chunks = []
        chunk_size = (
            self.config.memory_chunk_size or input_ids.size(1)
            if self.config.use_dual_memory
            else input_ids.size(1)
        )
        for start in range(0, input_ids.size(1), chunk_size):
            end = min(start + chunk_size, input_ids.size(1))
            pos = torch.arange(start, end, device=input_ids.device).expand(
                input_ids.size(0), -1
            )
            output = self.model(
                input_ids[:, start:end],
                memory_states=memory,
                attention_mask=attention_mask[:, start:end],
                position_ids=pos,
                use_cache=False,
            )
            hidden_chunks.append(output[0])
            memory = output[4]
        hidden = torch.cat(hidden_chunks, dim=1)
        selected = hidden.gather(
            1, positions[..., None].expand(-1, -1, hidden.size(-1))
        )
        logits = self.lm_head(selected).float() / temperature
        values = (
            self.value_head(selected).squeeze(-1).float()
            if self.value_head is not None
            else None
        )
        if tokens is None:
            return logits, values
        log_distribution = F.log_softmax(logits, dim=-1)
        logprobs = log_distribution.gather(-1, tokens[..., None]).squeeze(-1)
        entropy = -(log_distribution.exp() * log_distribution).sum(-1)
        return logprobs, values, entropy


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return values.masked_fill(~mask.bool(), 0).sum() / mask.sum().clamp_min(1)


def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    mask: Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Finite response episodes: EOS and max-token cutoff both terminate (V=0)."""
    rewards, values = rewards.float(), values.float()
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[:, 0])
    for t in range(rewards.size(1) - 1, -1, -1):
        next_mask = (
            mask[:, t + 1] if t + 1 < mask.size(1) else torch.zeros_like(mask[:, t])
        )
        next_value = (
            values[:, t + 1] if t + 1 < values.size(1) else torch.zeros_like(carry)
        )
        delta = rewards[:, t] + gamma * next_value * next_mask - values[:, t]
        carry = (delta + gamma * gae_lambda * next_mask * carry) * mask[:, t]
        advantages[:, t] = carry
    return advantages, (advantages + values).masked_fill(~mask.bool(), 0)


def ppo_objective(
    new_logprobs,
    old_logprobs,
    values,
    old_values,
    advantages,
    returns,
    entropy,
    mask,
    config: PPOConfig,
):
    """Clipped policy and clipped value errors; losses are token means."""
    delta = (new_logprobs - old_logprobs).masked_fill(~mask.bool(), 0)
    ratio = delta.exp()
    surrogate = torch.minimum(
        ratio * advantages,
        ratio.clamp(1 - config.clip_range, 1 + config.clip_range) * advantages,
    )
    clipped_values = old_values + (values - old_values).clamp(
        -config.value_clip_range, config.value_clip_range
    )
    value_error = torch.maximum(
        (values - returns).square(), (clipped_values - returns).square()
    )
    policy_loss = -masked_mean(surrogate, mask)
    value_loss = 0.5 * masked_mean(value_error, mask)
    entropy_mean = masked_mean(entropy, mask)
    loss = (
        policy_loss
        + config.value_loss_coef * value_loss
        - config.entropy_coef * entropy_mean
    )
    return loss, {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy_mean.detach(),
        "approx_kl": masked_mean((ratio - 1) - delta, mask).detach(),
        "clip_fraction": masked_mean(
            ((ratio - 1).abs() > config.clip_range).float(), mask
        ).detach(),
    }


def sample_tokens(logits, *, temperature=1.0, top_p=1.0, top_k=None, do_sample=True):
    """Reuse native filters; non-full-support decoding is evaluation-only."""
    logits = logits.float() / temperature
    if not do_sample:
        return logits.argmax(-1)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p < 1:
        logits = top_p_filter(logits, top_p)
    return torch.multinomial(logits.softmax(-1), 1).squeeze(-1)


@dataclass
class Rollout:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    positions: torch.Tensor  # state BEFORE each generated action
    tokens: torch.Tensor
    response_mask: torch.Tensor
    old_logprobs: torch.Tensor
    reference_logprobs: torch.Tensor
    old_values: torch.Tensor
    raw_rewards: torch.Tensor
    token_rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    eos: torch.Tensor


@torch.no_grad()
def collect_rollout(
    policy, reference, reward_model, prompts, config, backend, pad_id, eos_id
):
    """Genuine autoregressive rollout. Same number of calls on every rank.

    Fixed rectangular context avoids rank-dependent memory-chunk collective
    counts. New tokens replace right-padding directly; no interior pad holes.
    EOS is an action; subsequent slots are masked. Prefix replay trades speed
    for exact agreement with the SFT memory computation during PPO updates.
    """
    policy.eval()
    reference.eval()
    reward_model.eval()
    b, r = len(prompts), config.max_new_tokens
    width = config.max_prompt_tokens + r
    ids = torch.full((b, width), pad_id, dtype=torch.long, device=backend.device)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    lengths = torch.tensor([len(p) for p in prompts], device=backend.device)
    for i, prompt in enumerate(prompts):
        if not 0 < len(prompt) <= config.max_prompt_tokens:
            raise ValueError("Prompt exceeds PPO context budget")
        ids[i, : len(prompt)] = torch.as_tensor(prompt, device=backend.device)
        attention[i, : len(prompt)] = True
    positions = lengths[:, None] + torch.arange(r, device=backend.device)[None, :] - 1
    actions = torch.full((b, r), pad_id, dtype=torch.long, device=backend.device)
    mask = torch.zeros_like(actions, dtype=torch.bool)
    old = torch.zeros((b, r), device=backend.device)
    values = torch.zeros_like(old)
    finished = torch.zeros(b, dtype=torch.bool, device=backend.device)
    rows = torch.arange(b, device=backend.device)
    for t in range(r):
        with backend.autocast():
            logits, value = policy(
                ids, attention, positions[:, t : t + 1], temperature=config.temperature
            )
        token = sample_tokens(
            logits[:, 0],
            top_p=config.top_p,
            top_k=config.top_k,
            do_sample=config.do_sample,
        )
        active = ~finished
        token = torch.where(active, token, torch.full_like(token, pad_id))
        mask[:, t], actions[:, t] = active, token
        old[:, t] = F.log_softmax(logits[:, 0], -1).gather(1, token[:, None]).squeeze(1)
        values[:, t] = value[:, 0]
        ids[rows, lengths + t] = token
        attention[rows, lengths + t] = active
        finished |= active & (token == eos_id)
    with backend.autocast():
        reference_logprobs, _, _ = reference(
            ids, attention, positions, tokens=actions, temperature=config.temperature
        )
        raw = reward_model(ids, attention).float().squeeze(-1)
    kl = (old - reference_logprobs).masked_fill(~mask, 0)
    token_rewards = -config.kl_beta * kl
    token_rewards[rows, mask.sum(-1) - 1] += raw
    advantages, returns = generalized_advantage_estimate(
        token_rewards, values, mask, config.gamma, config.gae_lambda
    )
    return Rollout(
        ids,
        attention,
        positions,
        actions,
        mask,
        old,
        reference_logprobs,
        values,
        raw,
        token_rewards,
        advantages,
        returns,
        finished,
    )


def assert_frozen(model):
    if model.training or any(
        p.requires_grad or p.grad is not None for p in model.parameters()
    ):
        raise AssertionError("Reference/reward model must remain frozen in eval mode")
