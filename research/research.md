# Hybrid Mamba–MoE with Dual Memory: A Sub-Quadratic Architecture for Long-Context Language Modeling

**Author:** Vishnu Vardhan
**Status:** Research Proposal — Design Finalized (v2.1), Reference Implementation Complete
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

A reference PyTorch implementation exists (modularized under `model/`), verified
with 73 unit tests in `tests/test_model.py`, a 50-step mixed CPU training test
in `tests/test_mixed_cpu_training.py`, forward/backward passes, chunked
long-context training, and an autoregressive `generate()` method. A cloud
training smoke script (`scripts/test_cloud_train.py`) exercises the full
training objective on IMDB at ~150M parameters, and `scripts/eval_recall.py`
evaluates synthetic associative recall and falsification deltas. This document lays out the
problem, the design, what's built, what's still unproven, and the evaluation
plan that determines whether the memory component earns its place in the
architecture.

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

**How memory conditions each branch (implementation detail):** reads from
`attn_memory_bank` / `state_memory_bank` are concatenated with the shared
RMSNorm'd hidden states and projected through `attn_memory_combine` /
`state_memory_combine` before entering GQA and Mamba respectively. Raw branch
outputs (`attn_out`, `mamba_out`) — not the memory-augmented tensors — are
written back to the banks. This matches §3.2: banks are read *into* each
branch and written *from* branch outputs.

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
| Long-range recall of rare facts | Limited by window | Decays (recency bias) | Decays (recency bias) | Targeted by memory bank — **unproven, see §6** |

---

## 4. Compute & Parameter Accounting

Removing the O(L²) cross-attention fusion (see §3.3) and replacing it with
O(L·m) memory reads + O(L·d²) token fusion means the added machinery is
linear throughout. At L = 100K tokens (a target use case), the old
cross-attention term would have dominated total layer FLOPs by orders of
magnitude; the current design removes that term entirely.

Extra parameters over a Jamba-style baseline (Mamba + GQA + MoE, no explicit
memory): roughly **6d² per layer** (memory read/write projections + gates +
combine layers), which is comparable to what the removed bidirectional
cross-attention module cost (~4d²) — the redesign removes the quadratic
runtime cost without adding net extra parameters.

### 4.1 Training Objective

`HybridForCausalLM` optimizes a primary language-modeling loss plus router
stabilizers and eight memory-specific auxiliary losses (full definitions in
`loss-definitions.md`):

```
loss = ce_loss
     + router_aux_loss_coef · aux_loss      # default 0.02
     + router_z_loss_coef · z_loss          # default 5e-3
     + Σ λᵢ · Lᵢ                            # eight auxiliary terms
```

| Auxiliary loss | Symbol | Default λ | Purpose |
|---|---|---|---|
| Compressive reconstruction | `L_recon` | 0.08 | Train write path to retain chunk information |
| Associative retrieval | `L_assoc` | 1.2e-4 | Post-write key→value recall (Test 1 proxy) |
| Write-gate entropy | `L_gate` | 1e-3 | Prevent gate saturation (Test 2) |
| Read utilization | `L_read` | 5e-3 | Ensure memory reads affect branch inputs |
| Fusion balance | `L_fusion` | 8e-3 | Keep fusion gate near 0.5 per dimension |
| Expert specialization | `L_expert` | 2e-3 | Encourage expert output diversity |
| SSM state norm | `L_ssm` | 1e-5 | Penalize runaway SSM state magnitude |
| Slot diversity | `L_slot` | 3e-3 | Prevent memory slot collapse |

Auxiliary losses are **training-only** (`use_auxiliary_losses=True` by
default). `L_assoc` and `L_expert` use warmup schedules; reconstruction
decoders and assoc projections are omitted from inference when aux is
disabled. `HybridForCausalLM` warns (or raises in `debug_state_checks` mode)
if `use_dual_memory=True` with `use_auxiliary_losses=False`, because write
parameters may receive no gradient on short single-chunk forwards.

**Write-path gradient problem:** even with BPTT across `memory_chunk_size`
chunks, the last chunk's write receives no `ce_loss` gradient, and writes
within a chunk are not re-read in the same chunk. The auxiliary losses exist
specifically to give the write path a direct training signal — see
`loss-definitions.md` §3 for the structural analysis.

---

## 5. Implementation Status

A full reference implementation exists in `model.py`. Two model families
share the GQA and MoE stacks:

| Class | Role |
|---|---|
| `MixtralConfig` / `MixtralForCausalLM` | Baseline: sliding-window GQA + Top-2 MoE (control for ablations) |
| `HybridMambaMoEConfig` / `HybridForCausalLM` | Full hybrid with dual memory |

### 5.1 Core Components

- ✅ **`MambaBlock`** — selective SSM with **four-tier scan dispatch** when the
  fused CUDA kernel is unavailable: (1) optional `mamba-ssm` fused
  `selective_scan` on CUDA (unpadded batches, or per-row unpadded scan for
  padded batches); (2) Hillis-Steele parallel scan for
  `L ≤ parallel_scan_fallback_max_len` (default 4096); (3) blocked vectorized
  scan for `blocked_scan_min_len < L ≤ sequential_scan_min_len` (defaults
  4096–65536); (4) checkpointed sequential scan for very long `L`. Explicit
  `use_parallel_scan=True` forces tier-2. Incremental `(conv_state, ssm_state)`
  step cache for decode. `log_mamba_backend()`, `probe_mamba_scan_timing()`, and
  `get_mamba_scan_stats()` report the active backend at startup.

- ✅ **`CompressiveMemoryBank`** — multi-head attention read/write over m fixed
  slots. GRU-style write gate blends chunk summary into memory. Padding-safe:
  all-masked rows skip writes and use finite softmax guards (see
  `MEMORY_NAN_FIX_ID`). Optional training-only `MemoryReconstructionDecoder`
  and assoc key/value projections.

- ✅ **`TokenGatedFusion`** — O(L) per-token sigmoid gate over concatenated
  branch outputs.

- ✅ **`HybridDecoderLayer`** — full layer wiring per §3.2/§3.3. Returns
  per-layer write-gate stats and auxiliary loss scalars.

- ✅ **`HybridModel` / `HybridForCausalLM`** — stacks layers, threads memory
  across chunks, computes full training loss. Key config fields on
  `HybridMambaMoEConfig`:

  | Field | Default | Notes |
  |---|---|---|
  | `mamba_state_size` | 16 | SSM state dimension per inner channel |
  | `memory_size` | 64 | Slots per bank (m) |
  | `memory_chunk_size` | 512 | Training BPTT chunk size; `None` disables chunking |
  | `memory_write_interval` | `None` → chunk size | Decode write cadence |
  | `use_dual_memory` | `True` | `False` = Jamba-like (no banks) |
  | `use_auxiliary_losses` | `True` | Eight training aux losses |
  | `stream_chunked_ce_loss` | `True` | VRAM: CE on valid tokens only, no full `[B,L,V]` logits |
  | `return_logits` | `True` | Set `False` in long-context training for memory savings |
  | `gradient_checkpointing` | `False` | Per-layer activation checkpointing |
  | `use_fused_mamba_scan` | `True` | Requires `mamba-ssm` + CUDA |
  | `use_grouped_moe_dispatch` | `True` | Sort-by-expert token dispatch |
  | `use_cuda_graph` | `False` | Optional CUDA-graph single-token decode |
  | `capacity_factor` | `None` | Fully dropless MoE (batch-independent) |

- ✅ **`MemoryWriteBuffer`** — pre-allocated decode/prefill buffer that
  accumulates raw branch outputs with per-token validity masks; flushes to
  memory banks every `memory_write_interval` tokens. Matches training
  chunking semantics at inference.

- ✅ **`batched_dual_memory_read` / `batched_dual_memory_write`** — fused
  dual-bank attention for throughput.

- ✅ **`generate()`** — incremental KV + Mamba + memory caches; memory writes
  batched via `MemoryWriteBuffer`; finished rows frozen via `active_batch_mask`;
  optional CUDA-graph fast path for single-token decode.

### 5.2 Falsification Hooks (§6)

| Hook | Location | Use |
|---|---|---|
| `use_dual_memory=False` | `HybridMambaMoEConfig` | Architecture-level memory-off (Jamba-like) |
| `zero_memory_states()` | `HybridModel` | Test 1: zero banks at inference, keep modules |
| `build_test3_null_baseline_config()` | `model.py` | Test 3: binary-search `mamba_state_size` / `mamba_expand` for param-matched SSM-only null |
| `gate_stats` | `HybridTrainingOutput` | Test 2: per-layer write-gate means (logging) |
| `skip_memory_write=True` | forward / generate | Ablation: read without write |

### 5.3 Training & Test Infrastructure

- ✅ **`tests/test_model.py`** — 73 unit tests covering forward/backward, QK-norm,
  attention sinks, shared RoPE, layer routing, optimizer parameter grouping,
  chunked training parity, incremental vs full-forward logit cosine similarity,
  memory persistence across chunks, padding/NaN edge cases, fused Mamba parity,
  auxiliary loss gradients, MoE dispatch, CUDA-graph decode parity, and
  falsification hooks.

- ✅ **`tests/test_mixed_cpu_training.py`** — 50-step mixed CPU training test
  (1M-5M parameters) asserting strictly finite losses and gradients with zero NaNs.

- ✅ **`scripts/eval_recall.py`** — synthetic associative recall & scientific
  falsification harness evaluating Condition 1 (Memory-On), Condition 2 (Test 1 Zeroed),
  and Condition 3 (Test 3 Null baseline).

- ✅ **`scripts/test_cloud_train.py`** — one-epoch IMDB smoke test (~150M
  params, `memory_chunk_size=256`, `stream_chunked_ce_loss=True`,
  `return_logits=False`) with validation CE, cosine LR, and per-loss-term
  logging. Intended for GPU cloud hosts (Colab, Kaggle, etc.).

- ✅ **`loss-definitions.md`** — full specification of all eight auxiliary
  losses plus output logit z-loss, hyperparameter ranges, and tuning signals.

### 5.4 Bugs Found and Fixed During Implementation

| Issue | Fix |
|---|---|
| `SlidingWindowGQA` truncated KV but not `attention_mask` past `window_size` | Mask truncated to match; affects baseline and hybrid |
| All-padding rows in memory write/attend produced NaN softmax | Finite guards in `_attend`, `_batched_memory_summarize`, and write-buffer masking (`MEMORY_NAN_FIX_ID`) |
| `RotaryEmbedding` re-registered growing cos/sin buffers | Fixed-size cache up to `max_position_embeddings`; fail loudly if exceeded |
| Write-buffer validity lost across multi-append decode steps | Per-token `mask_buf` stored with branch outputs |

**Caching correctness:** `tests/test_model.py` checks incremental vs full-forward
last-logit cosine similarity on a small config. Re-run before latency-sensitive
use at larger scales.

---

## 6. Open Question: Is the Memory Actually Necessary?

This is the central unresolved question and the main risk to the project's
core contribution. Mamba's hidden state is, by construction, already a
compressed summary of everything before the current token. Before scaling
this architecture up, we need to show the explicit memory bank captures
something that state genuinely can't — otherwise it's added complexity with
no benefit.

The auxiliary losses (§4.1) are now implemented and give the write path a
direct training signal, but they do not by themselves prove memory is
*indispensable* — only that it can be trained. The falsification tests below
are still required before scaling.

### Falsification Plan

| Test | Method | Pass condition |
|---|---|---|
| **1. Rare-fact recall** | Inject a single, non-recurring fact early in a long sequence; query for exact recall late. Compare memory-on vs. `zero_memory_states()` at inference. | Recall degrades sharply when memory is zeroed |
| **2. Write-gate activity monitoring** | Log gate values and histograms from the memory write rule across training (`gate_stats` in `HybridTrainingOutput`) | Gates show non-degenerate activity (not saturating near 0 or 1). Requires `memory_chunk_size` chunking (or explicit cross-chunk memory threading) so write params receive gradients |
| **3. Matched-parameter null hypothesis** | Train baseline via `build_test3_null_baseline_config()` — larger Mamba state, no memory banks, at ~matched param count | Memory-bank model beats the bigger-state baseline on rare-fact recall |

All three should point the same direction before this architecture is
trained at scale. If any fail, the right response is to simplify — drop the
explicit memory or redesign what it specifically targets — rather than add
more machinery on top of a mechanism that hasn't earned its place.

---

## 7. Other Open Challenges

| Challenge | Notes |
|---|---|
| Chunk size vs. memory size tradeoff | `memory_chunk_size` (training) and `memory_write_interval` (decode) should be matched; sweep empirically |
| Write-gate saturation | Same failure mode as vanilla RNNs over very long sequences; `L_gate` entropy loss mitigates but may need periodic reset or stronger regularization |
| Router collapse (MoE) | Standard Mixtral-class issue; load-balancing aux loss + z-loss implemented; `L_expert` adds specialization pressure after warmup |
| Hardware efficiency | Pure PyTorch scan is unfused (materializes O(L·d·n) state); install `mamba-ssm` for production. Memory read/write is small-shaped — batched dual-bank paths help but custom kernels may still be needed at scale |
| Incremental generation | ✅ Done — `MambaBlock.step` + KV/memory threading + `MemoryWriteBuffer`; `generate()` is O(L) per step. Optional `use_cuda_graph` for wall-clock. Remaining work is fused CUDA kernels, not correctness |
| Padded-batch training stability | All-masked-row NaN guards are in place; cloud runs should log `MEMORY_NAN_FIX_ID` to confirm the correct `model.py` revision is loaded |
| Auxiliary loss tuning | Eight λ coefficients with warmup schedules; no large-scale sweep yet — defaults are starting points from `loss-definitions.md` |
| Scientific evidence | Falsification harness (needle/rare-fact eval) not yet built — see `Improvement-suggestions.md` §B |

---

## 8. Roadmap

1. **Run the three falsification tests (§6)** at small scale before any larger
   training run. Hooks are ready: `use_dual_memory=False`,
   `zero_memory_states()`, `build_test3_null_baseline_config()`.
2. **Cloud smoke → short training runs** using `scripts/test_cloud_train.py`
   as the starting point; monitor per-loss-term breakdown and `gate_stats`.
3. **Build needle/rare-fact eval harness** with fixed seeds and controlled
   distance (see `Improvement-suggestions.md` §B).
4. If memory proves redundant: simplify to `use_dual_memory=False` (Jamba-
   equivalent) and redirect effort toward the bigger-state null baseline.
5. Either path: benchmark wall-clock and memory footprint against the
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
  the direct basis for the memory bank design and `L_recon` auxiliary loss.
- **Perceiver / Perceiver IO** — precedent for linear-in-input-size attention
  via a small set of latent slots, the same trick applied to the memory read.

---

## 10. Conclusion

The efficiency side of this design is solid: the architecture is linear in
sequence length throughout, and the redesign that got it there (bounded
memory + per-token fusion, replacing sequence-to-sequence cross-attention)
removed a real quadratic-cost regression without adding net parameters. The
reference implementation is feature-complete for training and inference —
including auxiliary losses that address the write-path gradient problem,
chunked long-context training with VRAM-efficient CE streaming, and 66 unit
tests — but **what remains unproven is whether the memory component is
indispensable**, not just trainable. The falsification tests in §6 are
designed to answer that directly and should gate any decision to scale this
up.
