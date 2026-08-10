# Hybrid Mamba–MoE with Dual Memory: A Sub-Quadratic Architecture for Long-Context Language Modeling

**Author:** Vishnu Vardhan
**Status:** Research Proposal — Design Finalized (v2.0), Reference Implementation Complete
**Date:** August 2026

---

## 1. Executive Summary

Standard Transformer decoders scale quadratically with sequence length, which
caps how much context a model can practically use at inference time. This
project proposes a hybrid architecture — sliding-window attention, a Mamba
state-space branch, an explicit bounded-size memory, and sparse Top-2
Mixture-of-Experts — that keeps per-layer cost linear in sequence length while
retaining a mechanism for long-range, non-recurring information that both
attention and the SSM branch tend to lose over long contexts.

A reference PyTorch implementation exists (`model.py`), verified with
forward/backward passes and an autoregressive `generate()` method. This
document lays out the problem, the design, what's built, what's still
unproven, and the evaluation plan that determines whether the memory
component earns its place in the architecture.

---

## 2. Problem Statement

Transformer-based LLMs have three structural limitations that compound at
long context lengths:

| Limitation | Cause | Consequence |
|---|---|---|
| **Quadratic compute** | Full self-attention is O(L²) in sequence length L | Long-context inference (100K+ tokens) is expensive or infeasible at scale |
| **No persistent, addressable memory** | Everything the model "knows" about earlier tokens lives implicitly in KV cache or SSM state | Rare, one-off facts stated early in a long context get diluted or fall out of the attention window entirely |
| **Uniform per-token compute** | Every token is processed by the same dense feed-forward stack | No mechanism to spend more compute on tokens that need it and less on tokens that don't |

Existing approaches address these individually but not together:

- **Mamba / SSMs** fix the compute problem (linear time) but have a known
  weakness — state is continuously overwritten, so recall of specific,
  rarely-repeated information degrades over long sequences (recency bias).
- **Sliding-window attention** is cheap and precise locally, but has zero
  visibility beyond its window.
- **Mixtral-style Top-2 MoE** fixes the uniform-compute problem via sparse
  routing, but doesn't address context length or memory at all.
- **Jamba** combines Mamba + attention + MoE, which is the closest prior
  work, but has no explicit, queryable memory — it's still just Mamba's
  decaying state plus a local window.

**Our problem statement:** *Can we keep the linear-time, sparse-compute
benefits of a Mamba+MoE hybrid while adding a bounded-cost memory mechanism
that measurably improves recall of rare, long-range information — without
reintroducing quadratic cost to do it?*

---

## 3. Proposed Architecture

### 3.1 Design Goals

1. Linear-time (O(L)) per-layer cost, matching Mamba/Jamba-class efficiency.
2. Explicit, bounded-size memory that is genuinely read from and written to
   — not a static learned bias.
3. Cross-branch fusion (attention output ↔ SSM output) that does not
   reintroduce quadratic cost.
4. Sparse, conditional compute via Top-2 MoE, unchanged from Mixtral.

### 3.2 Architecture Diagram

```mermaid
flowchart TD
    subgraph Layer["Hybrid Decoder Layer"]
        A[Input Hidden States] --> B[RMSNorm]
        B --> C[Sliding Window GQA]
        B --> D[Mamba Block - Selective SSM]

        AM[(Attention Memory Bank, m slots)] -. read .-> C
        SM[(State Memory Bank, m slots)] -. read .-> D

        C --> E[Attention branch output]
        D --> F[SSM branch output]

        E -. write .-> AM
        F -. write .-> SM

        E --> G[Token-wise Gated Fusion - O(L)]
        F --> G
        G --> H[Residual Add + RMSNorm]
        H --> I[Top-2 Sparse MoE]
        I --> J[Residual Add]
        J --> K[Layer Output]
    end

    AM ==> AM2[Carried to next chunk / layer]
    SM ==> SM2[Carried to next chunk / layer]
```

Memory banks persist across sequence chunks (like a KV cache) and are
threaded explicitly through the forward pass rather than reset every call.

### 3.3 Component Summary

| Component | Role | Cost |
|---|---|---|
| **Sliding Window GQA** | Precise local context, grouped-query attention (Mistral-style) | O(L·w), w = window size |
| **Mamba Block (Selective SSM)** | Linear-time global context via continuously-updated state | O(L·d·n), n = state dim |
| **Compressive Memory Bank** (×2, one per branch) | Bounded-size (m slots) explicit memory with **gated read and write** | O(L·m), m constant, m ≪ L |
| **Token-wise Gated Fusion** | Combines attention-branch and SSM-branch outputs per position | O(L·d²) |
| **Top-2 MoE** | Sparse, conditional feed-forward compute (Mixtral-style) | O(L·2·d_ff·d) |

**Why memory is bounded and gated, not a static parameter:** the memory bank
is read via cross-attention (chunk tokens as queries, m memory slots as
keys/values — O(L·m), linear because m is fixed) and written via a
GRU-style gate that blends a compressed summary of the current chunk into
existing memory. This is what makes it a genuine memory system rather than a
learned bias term added to the input.

**Why fusion is per-token, not cross-attention:** the attention branch and
SSM branch both already produce per-position, aligned outputs. An earlier
version of this design used bidirectional sequence-to-sequence cross-attention
to fuse them, which silently reintroduced O(L²) cost — the single biggest
efficiency regression caught during design review. Per-token gating achieves
the same fusion at O(L).

### 3.4 Comparison to Existing Architectures

| Feature | Transformer | Mamba | Jamba (Mamba+MoE) | This Architecture |
|---|---|---|---|---|
| Local attention | ✅ Full | ❌ | ✅ GQA | ✅ Sliding Window GQA |
| State space model | ❌ | ✅ | ✅ | ✅ Mamba Block |
| Explicit bounded memory | ❌ | ❌ | ❌ | ✅ Gated read/write banks |
| Sparse MoE | ❌ | ❌ | ✅ Top-2 | ✅ Top-2 |
| Time complexity | O(L²) | O(L) | O(L) | O(L) |
| Long-range recall of rare facts | Limited by window | Decays (recency bias) | Decays (recency bias) | Targeted by memory bank — **unproven, see §5** |

---

## 4. Compute & Parameter Accounting

Removing the O(L²) cross-attention fusion (see §3.3) and replacing it with
O(L·m) memory reads + O(L·d²) token fusion means the added machinery is
linear throughout. At L = 100K tokens (a target use case), the old
cross-attention term would have dominated total layer FLOPs by orders of
magnitude; the current design removes that term entirely.

Extra parameters over a Jamba-style baseline (Mamba + GQA + MoE, no explicit
memory): roughly **6d² per layer** (memory read/write projections + gates),
which is comparable to what the removed bidirectional cross-attention module
cost (~4d²) — the redesign removes the quadratic runtime cost without adding
net extra parameters.

---

## 5. Implementation Status

A full reference implementation exists in `model.py`:

- ✅ `MixtralConfig` / `MixtralForCausalLM` — baseline (sliding-window GQA +
  Top-2 MoE), used as the control model for all comparisons below.
- ✅ `MambaBlock` — selective SSM with a **pure-PyTorch parallel associative
  scan** for prefill/training (no per-token Python loop) and incremental
  `(conv_state, ssm_state)` step cache for decode. No custom CUDA kernel
  dependency; runs on any SDPA-compatible device. Peak activation memory for
  the scan is still O(L·d·n) because the unfused implementation materializes
  the expanded state — acceptable on large training GPUs, a known limit vs
  fused selective-scan kernels.
- ✅ `CompressiveMemoryBank` — gated read/write memory. Banks are **read into**
  each branch (condition GQA/Mamba inputs) and **written from raw branch
  outputs**, matching §3.2. Padding masks are applied on write so pad tokens
  do not pollute summaries. State is threaded across chunks.
- ✅ `TokenGatedFusion` — O(L) branch fusion.
- ✅ `HybridForCausalLM` — full model wiring; forward/backward verified;
  memory-on, memory-off (`use_dual_memory=False`), and memory-zeroed-at-
  inference (`zero_memory_states`) hooks for §6 tests.
- ✅ `generate()` — autoregressive decoding (greedy + temperature/top-k/top-p,
  per-sequence EOS) with incremental KV + Mamba + memory caches — O(L) total
  cost. Absolute positions use `past_seen_tokens` (not truncated KV length).

**Caching correctness:** `tests/test_model.py` checks incremental vs full-forward
last-logit cosine similarity on a small config. Re-run before latency-sensitive
use at larger scales.

**Bug found and fixed during testing:** the original `SlidingWindowGQA`
truncated key/value tensors to the trailing attention window once the
sequence exceeded `window_size`, but did not truncate the padding
`attention_mask` to match — a latent shape-mismatch bug that only surfaces
once a sequence longer than one window is generated with a mask present.
Fixed; affects both the baseline and hybrid models.

---

## 6. Open Question: Is the Memory Actually Necessary?

This is the central unresolved question and the main risk to the project's
core contribution. Mamba's hidden state is, by construction, already a
compressed summary of everything before the current token. Before scaling
this architecture up, we need to show the explicit memory bank captures
something that state genuinely can't — otherwise it's added complexity with
no benefit.

### Falsification Plan

| Test | Method | Pass condition |
|---|---|---|
| **1. Rare-fact recall** | Inject a single, non-recurring fact early in a long sequence; query for exact recall late. Compare memory-on vs. memory-zeroed at inference. | Recall degrades sharply when memory is zeroed |
| **2. Write-gate activity monitoring** | Log gate values from the memory write rule across training | Gates show non-degenerate activity (not saturating near 0 or 1) |
| **3. Matched-parameter null hypothesis** | Train a baseline with a larger Mamba state dimension instead of the memory subsystem, at roughly matched parameter count | Memory-bank model beats the bigger-state baseline on rare-fact recall |

All three should point the same direction before this architecture is
trained at scale. If any fail, the right response is to simplify — drop the
explicit memory or redesign what it specifically targets — rather than add
more machinery on top of a mechanism that hasn't earned its place.

---

## 7. Other Open Challenges

| Challenge | Notes |
|---|---|
| Chunk size vs. memory size tradeoff | Smaller chunks → more frequent, more responsive memory writes but more overhead; needs empirical sweep |
| Write-gate saturation | Same failure mode as vanilla RNNs over very long sequences; may need periodic reset or regularization |
| Router collapse (MoE) | Standard Mixtral-class issue; load-balancing auxiliary loss already implemented |
| Hardware efficiency | Selective scan is parallelized in pure PyTorch but still **unfused** (materializes O(L·d·n) state); memory read/write is small-shaped and may not be GPU-efficient without custom kernels |
| Incremental generation | ✅ Done — `MambaBlock.step` + KV/memory threading; `generate()` is O(L). Remaining work is fused CUDA kernels for wall-clock, not correctness |

---

## 8. Roadmap

1. Run the three falsification tests (§6) at small scale before any larger
   training run (`use_dual_memory=False`, `zero_memory_states`, and
   `build_test3_null_baseline_config` are the hooks in `model.py`).
2. Incremental Mamba/KV/memory caching for `generate()` is implemented;
   next efficiency step is an optional fused selective-scan CUDA kernel if
   wall-clock on long contexts becomes the bottleneck.
3. If memory proves redundant: simplify to a Mamba+GQA+MoE baseline (Jamba-
   equivalent) and redirect effort toward the bigger-state approach instead.
4. Either path: benchmark wall-clock and memory footprint against the
   Mixtral baseline at matched parameter count, on progressively longer
   context lengths, to confirm the linear-time claim holds in practice and
   not just in FLOP accounting.

---

## 9. Related Work

- **Mamba** — Selective State Spaces for linear-time sequence modeling.
- **Jamba** — Hybrid Mamba-Transformer-MoE; closest prior architecture,
  lacks explicit memory.
- **Mixtral** — Top-2 sparse MoE routing, reused here unchanged.
- **Compressive Transformer** — precedent for fixed-size, gated-write memory,
  the direct basis for the memory bank design here.
- **Perceiver / Perceiver IO** — precedent for linear-in-input-size attention
  via a small set of latent slots, the same trick applied to the memory read.

---

## 10. Conclusion

The efficiency side of this design is solid: the architecture is linear in
sequence length throughout, and the redesign that got it there (bounded
memory + per-token fusion, replacing sequence-to-sequence cross-attention)
removed a real quadratic-cost regression without adding net parameters. What
remains unproven — and what the next round of experiments needs to
establish before further investment — is whether the memory component is
*indispensable*, not just cheap. The falsification tests in §6 are designed
to answer that directly and should gate any decision to scale this up.