"""Distributed Muon for PyTorch FSDP2 ``Shard(0)`` DTensor parameters.

The math is the official Muon recipe (Keller Jordan,
https://github.com/KellerJordan/Muon): SGD-Nesterov momentum, then a quintic
Newton-Schulz iteration that orthogonalizes each 2-D gradient/momentum matrix
(singular values compressed toward ~1), then the Moonlight RMS-matching scale
``0.2 * sqrt(max(A, B))`` (arXiv:2502.16982 §2.2) and decoupled
``lr``-scaled weight decay — exactly the semantics ``train.py::build_optimizers``
gets from ``torch.optim.Muon(adjust_lr_fn='match_rms_adamw')``.

Why this module exists: Newton-Schulz is defined over the COMPLETE 2-D matrix.
FSDP2 hands optimizers ``DTensor`` parameters sharded on dim 0, and the
operation is NOT row-block-separable — ``A = X @ X.mT`` couples every row
block across ranks — so running NS independently per local shard is wrong.
The fix here is the simple exact scheme: one all-gather per parameter per
step (the momentum view, cast to bf16 first to halve comm volume), batched
redundant NS on the full matrices, then a local slice back onto each rank's
shard. NorMuon-style owner round-robin / Dion-style all-to-all are strictly
scalability refinements of the same communication pattern and slot in behind
:func:`batched_zeropower_via_newtonschulz5` later.

This file is intentionally dependency-light (torch only, no
datasets/transformers) so parity checks can run on CPU without the training
entry point's heavy imports — same rationale as ``model/core/optim.py``.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import torch
from torch import optim

try:  # torch >= 2.5 exposes these public DTensor APIs
    from torch.distributed.tensor import DTensor, Shard
except ImportError as _exc:  # pragma: no cover - depends on torch build
    raise ImportError(
        "utils.fsdp2_muon requires public DTensor APIs from torch>=2.5 "
        "(torch.distributed.tensor). FSDP2 training additionally wants "
        "torch>=2.6 for the stable fully_shard path."
    ) from _exc

__all__ = [
    "MuonDTensor",
    "adjust_lr_factor",
    "batched_zeropower_via_newtonschulz5",
    "run_ns_self_check",
    "zeropower_via_newtonschulz5",
]

# Quintic Newton-Schulz coefficients (Keller Jordan's tuned values; see
# https://kellerjordan.github.io/posts/muon/). One iteration applies
# x -> a*x + b*x^3 + c*x^5 to the singular values; chosen so iterated
# singular values land in roughly [0.7, 1.3].
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315

_VALID_ADJUST_LR_FNS = ("match_rms_adamw", "original")


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate ``UVᵀ`` of the SVD of ``G`` via quintic Newton-Schulz.

    Verbatim-faithful port of the official implementation: bf16 compute,
    Frobenius pre-normalization (Ortho(cG) = Ortho(G)), transpose tall
    matrices so rows <= cols, ``steps`` iterations of
    ``X <- aX + (bA + cA²)X`` with ``A = XXᵀ``, orientation restored.
    Returns bf16; the result's singular values lie roughly in [0.5, 1.5].
    """
    a, b, c = _NS_A, _NS_B, _NS_C
    X = G.bfloat16()
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True
    # Normalize before the loop so the spectral norm starts at ~1.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def batched_zeropower_via_newtonschulz5(
    mats: list[torch.Tensor], steps: int = 5
) -> list[torch.Tensor]:
    """NS over many full matrices, batching identical shapes into bmm calls.

    Mathematically EXACT per matrix — NS acts independently along leading
    batch dims — so this is purely a kernel-launch fusion. Matrices are
    grouped by shape; every group runs as one stacked ``[B, m, n]`` batch.
    Inputs must be full (unsharded) 2-D tensors; outputs preserve order.
    """
    if not mats:
        return []
    order: dict[tuple[int, int], list[int]] = {}
    for i, mat in enumerate(mats):
        order.setdefault((mat.size(-2), mat.size(-1)), []).append(i)

    out: list[torch.Tensor | None] = [None] * len(mats)
    a, b, c = _NS_A, _NS_B, _NS_C
    for idxs in order.values():
        X = torch.stack([mats[i] for i in idxs], dim=0).bfloat16()
        transposed = X.size(-2) > X.size(-1)
        if transposed:
            X = X.transpose(-2, -1)
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(steps):
            A = X @ X.transpose(-2, -1)
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        if transposed:
            X = X.transpose(-2, -1)
        X = X.to(mats[idxs[0]].dtype)
        for rank_in_batch, global_idx in enumerate(idxs):
            out[global_idx] = X[rank_in_batch]
    return out  # type: ignore[return-value]


def adjust_lr_factor(shape: tuple[int, ...], fn: str | None) -> float:
    """Per-matrix LR scale matching ``torch.optim.Muon(adjust_lr_fn=...)``.

    ``'match_rms_adamw'``: Moonshot RMS matching ``0.2 * sqrt(max(A, B))``
    so Muon and AdamW can share one peak LR / weight decay.
    ``'original'``: Keller's ``max(1, rows/cols)**0.5``.
    ``None``: no scaling.
    """
    if fn is None:
        return 1.0
    m, n = shape[-2], shape[-1]
    if fn == "match_rms_adamw":
        return 0.2 * math.sqrt(max(m, n))
    if fn == "original":
        return max(1.0, m / n) ** 0.5
    raise ValueError(
        f"Unknown adjust_lr_fn={fn!r}; expected one of {_VALID_ADJUST_LR_FNS} or None"
    )


def _rank_and_world() -> tuple[int, int]:
    """(rank, world_size) of the default process group; (0, 1) if unset."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


class MuonDTensor(optim.Optimizer):
    """Muon over FSDP2-sharded DTensor matrices (``Shard(0)`` placement).

    Owns ONLY the hidden 2-D matrices routed by
    ``model.core.optim.split_muon_adam_params``; embeddings / lm_head /
    norms / biases stay on a stock ``torch.optim.AdamW`` alongside (the
    two-optimizer ``[muon, adam]`` structure of ``train.py`` is preserved,
    including one ``LambdaLR`` per optimizer).

    Per step, per parameter:
      1. LOCAL elementwise (no comm): momentum EMA into a shard-matched
         DTensor buffer, Nesterov blend ``u = g.lerp(buf, μ)``.
      2. ONE collective: ``u.bfloat16().full_tensor()`` — all-gather of the
         momentum view (bf16 halves the wire bytes; NS casts to bf16 anyway).
      3. Redundant batched NS on the full matrices (every rank computes the
         same result deterministically from identical gathered inputs).
      4. Scale by :func:`adjust_lr_factor`, slice off THIS rank's
         ``torch.chunk(dim=0)`` row block (mirrors ``Shard(0)`` splitting),
         apply decoupled weight decay and the update locally.

    Momentum state lives as DTensors with the param's placement, so
    ``get_optimizer_state_dict(full_state_dict=True)`` consolidates it for
    checkpointing without special cases. No ``.item()`` / ``.cpu()`` ever
    runs in :meth:`step`; the only collective is the all-gather in (2).
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = "match_rms_adamw",
    ) -> None:
        if adjust_lr_fn not in _VALID_ADJUST_LR_FNS and adjust_lr_fn is not None:
            raise ValueError(
                f"adjust_lr_fn={adjust_lr_fn!r} unsupported; "
                f"expected one of {_VALID_ADJUST_LR_FNS} or None"
            )
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)
        self._validate_params()

    def _validate_params(self) -> None:
        """Fail loudly on anything NS-over-full-matrices cannot own."""
        for group in self.param_groups:
            for p in group["params"]:
                if not isinstance(p, DTensor):
                    raise TypeError(
                        "MuonDTensor requires FSDP2-wrapped DTensor parameters; "
                        f"got plain {type(p).__name__} with shape "
                        f"{tuple(p.shape)}. Wrap the model with fully_shard "
                        "BEFORE constructing optimizers."
                    )
                placements = tuple(p.placements)
                if placements != (Shard(0),):
                    raise ValueError(
                        "MuonDTensor expects Shard(0) parameters (FSDP2 "
                        f"default); got placements {placements} for shape "
                        f"{tuple(p.shape)}."
                    )
                if p.ndim != 2:
                    raise ValueError(
                        "Muon orthogonalization is defined on 2-D matrices; "
                        f"got ndim={p.ndim} shape {tuple(p.shape)}. Route "
                        "conv/1-D/embedding params to AdamW instead (see "
                        "model/core/optim.py::split_muon_adam_params)."
                    )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        rank, world_size = _rank_and_world()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            mu = group["momentum"]
            nesterov = group["nesterov"]

            # Phase A (local): momentum EMA + Nesterov blend; gather views.
            pending: list[tuple[Any, torch.Tensor]] = []  # (param, u_full bf16)
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    # Keller's reference zero-fills missing grads so the
                    # momentum buffer keeps decaying instead of freezing.
                    p.grad = torch.zeros_like(p)
                    grad = p.grad
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(p)  # DTensor, Shard(0), fp32
                    state["momentum_buffer"] = buf
                buf.lerp_(grad, 1 - mu)
                u = grad.lerp(buf, mu) if nesterov else buf
                # The single collective of the step: gather the bf16 view.
                u_full = u.to(torch.bfloat16).full_tensor()
                pending.append((p, u_full))

            # Phase B (redundant, comm-free): batched NS on full matrices.
            orth = batched_zeropower_via_newtonschulz5(
                [u_full for _, u_full in pending], steps=group["ns_steps"]
            )

            # Phase C (local): scale, slice to this rank's shard, apply.
            for (p, _), u_orth in zip(pending, orth):
                factor = adjust_lr_factor(tuple(p.shape), group["adjust_lr_fn"])
                update_full = u_orth.to(dtype=p.dtype) * factor
                # torch.chunk(dim=0) is exactly how Shard(0) splits; every
                # rank holds an IDENTICAL update_full (deterministic kernels
                # on bit-identical gathered inputs), so local slicing
                # reconstructs the sharded update with zero extra comms.
                update_local = update_full.chunk(world_size, dim=0)[rank]
                p_local = p.to_local()
                if wd != 0.0:
                    p_local.mul_(1 - lr * wd)
                p_local.add_(update_local, alpha=-lr)

        return loss


def _ns_reference_fp32(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Independent straight-from-the-paper NS written in fp32 (no bf16 cast,
    no transpose shortcut) — used by :func:`run_ns_self_check` to pin the
    production implementation's coefficients/orientation/normalization."""
    X = G.float() / (G.float().norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = _NS_B * A + _NS_C * (A @ A)
        X = _NS_A * X + B @ X
    return X


def run_ns_self_check() -> bool:
    """CPU parity gate for the NS math — no process group required.

    Calibrated to what the ALGORITHM guarantees, not to exact UVᵀ equality
    (NS returns an orthogonal matrix whose singular values sit in a BAND,
    so flattened-direction cosine plateaus around 0.96-0.99 by design):

      1. implementation fidelity vs an independent fp32 reference NS
         (catches wrong coefficients / transpose / normalization bugs);
      2. directional sanity vs the true polar factor UVᵀ (cos >= 0.95);
      3. conditioning: the update's singular-value spread collapses toward
         the known approximation band;
      4. the batched path equals the single-matrix path (fusion only);
      5. tall/wide transpose handling is symmetric;
      6. adjust_lr_factor matches the documented formulas.
    """
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= cond
        print(
            f"[{'PASS' if cond else 'FAIL'}] {name}"
            + (f" — {detail}" if detail else "")
        )

    torch.manual_seed(0)
    shapes = [(64, 64), (256, 96), (96, 256), (512, 128), (7, 3)]
    for m, n in shapes:
        g = torch.randn(m, n)
        ns = zeropower_via_newtonschulz5(g, steps=5).float()
        ref_paper = _ns_reference_fp32(g, steps=5)
        impl_err = float((ns - ref_paper).abs().max())
        check(
            f"matches paper algorithm ({m}x{n})",
            impl_err < 5e-2,
            f"maxdiff_vs_fp32_ref={impl_err:.2e}",
        )
        uv_t = torch.linalg.svd(g.float(), full_matrices=False)
        uv_t = uv_t.U @ uv_t.Vh
        cos = torch.nn.functional.cosine_similarity(
            ns.flatten(), uv_t.flatten(), dim=0
        ).item()
        check(
            f"polar-factor direction ({m}x{n}) cos>=0.95", cos >= 0.95, f"cos={cos:.5f}"
        )
        sv = torch.linalg.svdvals(ns)
        sv_in = torch.linalg.svdvals(g.float())
        ratio_after = (sv.max() / sv.min()).item()
        ratio_before = (sv_in.max() / sv_in.min()).item()
        check(
            f"spectral compression ({m}x{n})",
            bool(
                sv.max() <= 3.0
                and sv.min() >= 0.01
                and ratio_after <= max(30.0, ratio_before)
            ),
            f"sv=[{sv.min():.3f},{sv.max():.3f}] cond {ratio_before:.1f}->{ratio_after:.1f}",
        )

    # Batched == unbatched (identical math, fused launches only).
    mats = [torch.randn(s[0], s[1]) for s in [(64, 64), (64, 64), (96, 32)]]
    solo = [zeropower_via_newtonschulz5(x, 5) for x in mats]
    batched = batched_zeropower_via_newtonschulz5(mats, 5)
    max_diff = max(float((a - b).abs().max()) for a, b in zip(solo, batched))
    check(
        "batched == single within bf16 tol", max_diff < 1e-2, f"maxdiff={max_diff:.2e}"
    )

    # Transpose symmetry: NS(G) == NS(Gᵀ)ᵀ up to bf16 noise.
    g = torch.randn(200, 60)
    direct = zeropower_via_newtonschulz5(g, 5)
    flipped = zeropower_via_newtonschulz5(g.mT, 5).mT
    sym_diff = float((direct - flipped).abs().max())
    check("transpose symmetry", sym_diff < 1e-2, f"maxdiff={sym_diff:.2e}")

    check(
        "adjust_lr_factor formulas",
        abs(adjust_lr_factor((512, 512), "match_rms_adamw") - 0.2 * math.sqrt(512))
        < 1e-12
        and abs(adjust_lr_factor((96, 256), "original") - 1.0) < 1e-12
        and abs(adjust_lr_factor((256, 96), "original") - math.sqrt(256 / 96)) < 1e-12
        and adjust_lr_factor((8, 8), None) == 1.0,
    )
    return ok


if __name__ == "__main__":
    print("utils/fsdp2_muon.py Newton-Schulz self-check (CPU)")
    sys.exit(0 if run_ns_self_check() else 1)
