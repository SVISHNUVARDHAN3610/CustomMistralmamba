"""Mixture-of-Experts components."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
    # VRAM: accumulate pairwise cosines with a running sum instead of stacking
    # all pair tensors (same mean, lower peak intermediate memory).
    l_ortho = torch.tensor(0.0, device=expert_out.device, dtype=expert_out.dtype)
    n_pairs = 0
    for i in range(top_k):
        for j in range(i + 1, top_k):
            l_ortho = (
                l_ortho
                + F.cosine_similarity(expert_out[:, i], expert_out[:, j], dim=-1)
                .abs()
                .mean()
            )
            n_pairs += 1
    if n_pairs > 0:
        l_ortho = l_ortho / n_pairs
    # Routing variance uses router logits only — no expert_out retention needed.
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
            # VRAM: [num_tokens, top_k, H] only while expert specialization loss
            # is computed; released when dispatch returns.
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
                # AMP: expert_outputs may be fp16 while expert_out is fp32.
                expert_out[row_indices, k_indices] = expert_outputs.to(
                    dtype=expert_out.dtype
                )
            gating_scale = topk_weights[row_indices, k_indices].unsqueeze(-1)
            moe_output.index_add_(
                0,
                row_indices,
                (expert_outputs * gating_scale).to(dtype=moe_output.dtype),
            )
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
            # VRAM: [num_tokens, top_k, H] only while expert specialization loss
            # is computed; released when dispatch returns.
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
            expert_out[sorted_token, sorted_k] = expert_outputs.to(
                dtype=expert_out.dtype
            )
        gating_scale = topk_weights[sorted_token, sorted_k].unsqueeze(-1)
        moe_output.index_add_(
            0,
            sorted_token,
            (expert_outputs * gating_scale).to(dtype=moe_output.dtype),
        )
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
            # VRAM: [num_tokens, top_k, H] only while expert specialization loss
            # is computed; released when dispatch returns.
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
                expert_out[row_indices, k_indices] = expert_outputs.to(
                    dtype=expert_out.dtype
                )

            gating_scale = topk_weights[row_indices, k_indices].unsqueeze(-1)
            moe_output.index_add_(
                0,
                row_indices,
                (expert_outputs * gating_scale).to(dtype=moe_output.dtype),
            )
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
