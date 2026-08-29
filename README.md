# CustomMistralMamba: Hybrid Mamba–MoE with Dual Compressive Memory

**A sub-quadratic decoder architecture for long-context language modeling, combining sliding-window attention, a selective state-space branch, explicit gated memory, and sparse Mixture-of-Experts.**

| | |
|---|---|
| **Author** | Vishnu Vardhan |
| **Status** | Research prototype — architecture finalized (v2.1), reference implementation complete |
| **Language / stack** | Python, PyTorch (>=2.1) |
| **Entry point** | `from model import HybridForCausalLM, HybridMambaMoEConfig` |
| **Design document** | [`research/research.md`](research/research.md) |
| **Loss specification** | [`research/loss-definitions.md`](research/loss-definitions.md) |
| **Package documentation** | [`model/README.md`](model/README.md) — full architecture reference, 24 sections |
| **Unit tests** | [`tests/test_model.py`](tests/test_model.py) |

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Problem Statement](#2-problem-statement)
3. [Proposed Solution](#3-proposed-solution)
4. [Architecture Overview](#4-architecture-overview)
5. [What Is Implemented](#5-what-is-implemented)
6. [Repository Structure](#6-repository-structure)
7. [Installation](#7-installation)
8. [Quickstart](#8-quickstart)
9. [Training](#9-training)
10. [Testing](#10-testing)
11. [Scientific Status — What's Proven vs. Unproven](#11-scientific-status--whats-proven-vs-unproven)
12. [Falsification Plan](#12-falsification-plan)
13. [Design Principles](#13-design-principles)
14. [Comparison to Prior Architectures](#14-comparison-to-prior-architectures)
15. [Related Work](#15-related-work)
16. [Roadmap](#16-roadmap)
17. [Limitations](#17-limitations)
18. [Citation](#18-citation)

---

## 1. Motivation

Long-context language modeling exposes a persistent tension in decoder-only architectures: attention gives precise, addressable recall but costs O(L²); recurrent and state-space alternatives give linear-time compute but degrade rare, long-range recall as their fixed-size state is continuously overwritten. This project asks whether that trade-off is fundamental, or whether a *small, explicit, addressable memory* — bounded in size, gated in its updates, and cheap to read/write — can recover some of what attention loses at range, without paying attention's asymptotic cost. `CustomMistralMamba` is the reference implementation built to test that question.

## 2. Problem Statement

Transformer decoders accumulate three structural weaknesses as context length grows:

| Limitation | Mechanism | Consequence |
|---|---|---|
| Quadratic compute | Full self-attention is O(L²) | 100K+ token inference is expensive or infeasible |
| No persistent addressable memory | Knowledge lives only in the KV cache or recurrent state | Rare, one-off facts stated early get diluted or fall outside the window |
| Uniform per-token compute | Every token passes through the same dense FFN | No mechanism to spend extra compute where it's needed |

Prior work addresses these piecemeal: **Mamba** fixes compute but exhibits recency bias since its state is continuously overwritten; **sliding-window attention** is cheap and precise locally but blind beyond the window; **Mixtral**-style MoE fixes uniform compute but says nothing about context length; **Jamba** combines Mamba, attention, and MoE — the closest prior art — but still has no explicit, queryable memory beyond the SSM's own decaying state.

**Research question:** *Can a Mamba+MoE hybrid retain linear-time, sparse-compute efficiency while adding a bounded-cost memory mechanism that measurably improves recall of rare, long-range facts — without reintroducing quadratic fusion or attention?*

## 3. Proposed Solution

Four mechanisms are composed in a single decoder stack:

1. **Sliding-window grouped-query attention (GQA)** — precise local context at O(L·w) cost, Mistral-style.
2. **Mamba selective state-space model (SSM)** — linear-time global context via a continuously updated hidden state, in the selective-scan (S6) formulation.
3. **Dual compressive memory banks** — one bounded-size (`m` slots), gated read/write memory bank per branch (`attn_memory_bank`, `state_memory_bank`), so attention and the SSM each get their own explicit, addressable store.
4. **Top-2 sparse Mixture-of-Experts (MoE)** — conditional compute in the feed-forward path, reused from Mixtral.

The design keeps every component sub-quadratic: memory reads/writes are bounded to O(L·m) with `m ≪ L`; the two branches are fused via **per-token gating**, O(L·d²), rather than cross-attention between branches, O(L²); and the FFN path stays sparse via dropless Top-2 MoE. The result is an architecture that is linear per layer throughout, with an added component whose entire job is to be tested — not assumed — to help.

## 4. Architecture Overview

Each `HybridDecoderLayer` runs two branches in parallel, each conditioned on its own memory bank, then fuses and routes through MoE:

```
x → RMSNorm
  ├─ [read attn_memory_bank] → SlidingWindowGQA ──┐
  └─ [read state_memory_bank] → MambaBlock ───────┤
                                                     ├─→ TokenGatedFusion → residual
  attn_out → write attn_memory_bank                 │
  mamba_out → write state_memory_bank                
                                                     ↓
                                    RMSNorm → Top-2 Sparse MoE → residual → layer output
```

Memory is genuinely read-before-branch and written-from-raw-branch-output — not a static learned bias folded into the residual stream. Two complete model families are provided so the memory contribution can be ablated cleanly:

| Family | Config | Class | Role |
|---|---|---|---|
| **Baseline** | `MixtralConfig` | `MixtralForCausalLM` | Ablation control — GQA + MoE only, no Mamba, no memory |
| **Hybrid** | `HybridMambaMoEConfig` | `HybridForCausalLM` | Full architecture — GQA + Mamba + dual memory + MoE |

Full per-layer diagrams, compute-cost tables, and a component-by-component reference live in [`model/README.md`](model/README.md) (sections 5–9).

## 5. What Is Implemented

The reference implementation is feature-complete for training and autoregressive inference:

- Both model families (`MixtralForCausalLM`, `HybridForCausalLM`) with matched GQA/MoE building blocks for apples-to-apples ablation.
- `CompressiveMemoryBank` with single-sigmoid gated EMA writes, padding-safe batched read/write, and a training-only reconstruction/associative auxiliary path.
- `MambaBlock` with a four-tier scan dispatch (fused CUDA kernel when `mamba-ssm` is available, Hillis-Steele parallel scan, blocked vectorized scan, sequential scan with checkpointing) so behavior is correct with or without GPU fused kernels.
- **Eight auxiliary losses** (reconstruction, associative recall, gate regularization, slot utilization, read/fusion/SSM-calibration/expert-routing terms) with warmup schedules, documented formula-by-formula in [`research/loss-definitions.md`](research/loss-definitions.md).
- Chunked, truncated-BPTT training for long sequences, with memory and Mamba state threaded across chunks.
- Incremental KV / Mamba / memory caching for `generate()`, including a fast single-token decode path (`MambaBlock.step()`, `MemoryWriteBuffer.append_single_token()`).
- A parameter-matched **null baseline builder** (`build_test3_null_baseline_config`) that expands the Mamba state size to compensate for a disabled memory bank, so memory's contribution can be isolated from raw parameter count.
- Production training script (`train.py`) consuming memory-mapped tokenized shards, with cyclic WikiText validation, mixed-precision-safe FP32 promotion for numerically sensitive ops, and full checkpoint/resume support.
- A cloud training smoke test (`scripts/test_cloud_train.py`) exercising the complete training objective at ~200M parameters on IMDB.
- 83 unit tests covering forward/backward correctness, shape invariants, caching, memory falsification hooks, and numerical stability (`MEMORY_NAN_FIX_ID` guards against NaNs on the memory path), plus a separate CPU smoke module (`tests.test_toy_train_smoke`) that runs real chunked-BPTT training steps.

## 6. Repository Structure

```
CustomMistralmamba/
├── README.md                     # This file
├── requirements.txt               # torch, ruff, pre-commit
├── pyproject.toml                 # ruff lint config
│
├── model/                         # Core architecture package
│   ├── README.md                  # Full architecture reference (24 sections)
│   ├── core/                      # Config dataclasses, dtype helpers, param builders
│   ├── layers/                    # Shared blocks: RMSNorm, RoPE, GQA, MoE, fusion gate
│   ├── mixtral/                   # Baseline ablation model
│   └── hybrid/                    # Memory bank, Mamba block, aux losses, hybrid layer/model
│
├── research/
│   ├── research.md                # Full research proposal, problem framing, evaluation plan
│   ├── loss-definitions.md        # Formula-level spec for all eight auxiliary losses
│   └── Improvement-suggestions.md # Deferred research-grade backlog (post-review)
│
├── scripts/
│   ├── toy_train.py               # ~5M-param smoke test, single file, no dataset needed
│   ├── test_cloud_train.py        # ~200M-param IMDB training smoke test
│   └── verify_model_package.py    # Import / API surface sanity check
│
├── utils/
│   ├── dataset.py                 # TokenizedShardProducer, MmapShardDataset (streaming shards)
│   └── validation.py              # WikiTextCyclicValidator for periodic held-out eval
│
├── train.py                       # Production training loop (streaming shards + checkpointing)
└── tests/
    ├── test_model.py              # 83 unit tests over the model package
    └── test_toy_train_smoke.py    # Smoke test for scripts/toy_train.py
```

## 7. Installation

```bash
git clone https://github.com/SVISHNUVARDHAN3610/CustomMistralmamba.git
cd CustomMistralmamba
pip install -r requirements.txt
```

The only hard runtime dependency is `torch>=2.1.0`. GPU acceleration for the Mamba branch via fused CUDA kernels is optional (`mamba-ssm>=2.2.0`, commented out by default in `requirements.txt`) — the implementation auto-falls-back to pure-PyTorch scan tiers if it is not installed, at the cost of throughput, not correctness.

## 8. Quickstart

```python
import torch
from model import HybridForCausalLM, HybridMambaMoEConfig

cfg = HybridMambaMoEConfig(
    vocab_size=3200,
    hidden_size=256,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    head_dim=64,
    intermediate_size=512,
    window_size=16,
    num_experts=4,
    memory_size=16,
    use_dual_memory=True,
)

model = HybridForCausalLM(cfg).train()
ids = torch.randint(0, cfg.vocab_size, (2, 128))
labels = ids.roll(shifts=-1, dims=1)

out = model(input_ids=ids, labels=labels, training_step=0, max_training_steps=1000)
out.loss.backward()
```

For autoregressive generation, a Mixtral-only ablation baseline, and the parameter-matched null baseline used in Test 3, see section 18 of [`model/README.md`](model/README.md).

For an even smaller, dependency-free smoke test:

```bash
python scripts/toy_train.py
```

## 9. Training

`train.py` runs production training against pre-tokenized, memory-mapped binary shards:

```bash
python train.py \
  --cache-dir /path/to/shards \
  --run-dir runs/hybrid-150m \
  --ckpt-dir ./model_ckpt
```

It handles seeding (with an optional fully-deterministic mode), rotating run logs, streaming shard consumption via `MmapShardDataset`, periodic cyclic validation against `Salesforce/wikitext`, gradient clipping, checkpoint save/resume (`model_ckpt.pth` + `config.json`), and warmup schedules for the auxiliary and expert-routing losses. `scripts/test_cloud_train.py` provides a smaller, self-contained IMDB-based smoke test for verifying the full objective end-to-end on cloud hardware before a long run.

### Optimizer: Muon + AdamW (Moonshot-style)

Training uses the Moonlight hybrid optimizer (arXiv:2502.16982): 2D hidden matrices are updated by Muon, everything else (embeddings, `lm_head`, 1D norms/biases, `_no_weight_decay` params) by AdamW. A shared peak LR (`--lr`) is used for both — with `adjust_lr_fn='match_rms_adamw'`, Muon internally scales each update by `0.2·sqrt(max(A,B))` so its update RMS matches AdamW's at the same nominal LR.

**LR visibility note:** in current PyTorch builds the Muon RMS matching is applied *per parameter at optimizer-step time*; `param_groups[i]['lr']` keeps the configured base LR, and the scheduler's warmup/cosine multiplies that base. Some torch versions instead bake the shape-dependent scaling into the group LR at construction, which inflates the base the scheduler acts on (observed as `muon_lr ≈ 1.2×` the configured value in an earlier run's logs). Training start now logs every Muon group LR and warns if it differs from the configured `--muon-lr`, so whichever behavior your torch build has is visible in the run log.

### Validation methodology

Two validation modes are available (`--val-mode`):

- **`packed` (default)** — a **fixed, non-rotating** slice of the first `--val-eval-rows` (default 500) non-empty wikitext validation rows is tokenized into one contiguous stream of `[BOS] doc [EOS]` documents and sliced into full `seq_len` windows with pre-shifted labels — identical packing to training. Every validation call scores the same windows, so val loss curves are directly comparable across steps (no rotating-cursor sampling noise), and cross-document windows see left context, so short rows are no longer penalized.
- **`rows` (legacy)** — the original behavior: each wikitext row scored independently with per-row BOS/EOS and right-padding, over a rotating 50-row cursor. Kept only for comparison with historical metrics; it systematically reports *higher and noisier* val loss than packed mode (short rows have no left context, and each call samples a different row slice).

The same validation event includes `"mode"` in `metrics.jsonl`, so packed and legacy numbers are never confused in analysis.

## 10. Testing

```bash
python -m unittest tests.test_model -v
python -m unittest tests.test_toy_train_smoke -v
# Single test:
python -m unittest tests.test_model.TestHybridModel.test_forward_backward -v
```

(The project standard is `unittest`; pytest is not a dependency.)

`tests/test_model.py` covers both model families across forward/backward correctness, shape and dtype invariants under AMP, KV/Mamba/memory cache threading through `generate()`, padding-mask correctness in the memory write path, the four Mamba scan-backend tiers, and the memory-zeroing falsification hook (`HybridModel.zero_memory_states()`). `tests/test_toy_train_smoke.py` verifies the minimal training loop runs end-to-end without dataset dependencies.

## 11. Scientific Status — What's Proven vs. Unproven

**Implemented and verified:** the architecture runs correctly forward and backward at multiple scales, from a ~5M-parameter toy config to a ~200M-parameter cloud smoke run; gradients flow through both branches and both memory banks; caching is correct across incremental decode; the four Mamba scan tiers produce numerically consistent results with and without fused CUDA kernels.

**Not yet established:** whether the dual memory banks are *necessary* rather than *redundant* with the Mamba branch's own hidden state, which is already a compressed summary of everything seen so far. This is the central open question the codebase is built to answer, not a claim it currently makes.

## 12. Falsification Plan

Three experiments, documented in full in [`research/research.md`](research/research.md) §6, must pass before the memory component is scaled up:

1. **Ablation-at-inference:** rare-fact recall should measurably degrade when memory is zeroed via `zero_memory_states()` at test time, relative to the same model with memory intact.
2. **Non-degenerate gate activity:** write gates must show real, non-saturated activity during training — not settle at a trivial always-write or always-ignore value.
3. **Beats a matched null baseline:** the hybrid model with memory must outperform `build_test3_null_baseline_config` — a parameter-matched control with `use_dual_memory=False` and an enlarged Mamba state — on long-range recall tasks.

If the memory component fails these tests, the documented next step (see [`research/Improvement-suggestions.md`](research/Improvement-suggestions.md)) is to simplify to `use_dual_memory=False` and redirect effort toward a leaner, Jamba-style baseline rather than defend an unproductive component.

## 13. Design Principles

- **Everything sub-quadratic.** No component in the hybrid stack reintroduces O(L²) cost; branch fusion happens through per-token gating, not cross-branch attention.
- **Memory as a genuine read/write system, not a bias.** The memory-augmented tensor feeds the branch as *input*; the branch's *raw* output is what gets written back — read-into-input and write-from-output are deliberately decoupled so the bank behaves like memory rather than a learned constant.
- **Ablatable by construction.** `use_dual_memory=False`, the Mixtral baseline, and the parameter-matched null baseline all exist specifically so any claimed benefit of memory can be isolated and falsified, not just asserted.
- **Numerically defensive.** FP32 promotion under autocast for scan and router math, router logit clamping, NaN-safe gated writes on all-padded rows — the kind of guardrails that matter once training runs get long and unattended.
- **Backend-portable by default.** The Mamba branch works correctly (if more slowly) with no CUDA kernel installed at all, so the reference implementation isn't gated on a specific hardware/library stack to be usable or testable.

## 14. Comparison to Prior Architectures

| Feature | Transformer | Mamba | Jamba | Mixtral (baseline, this repo) | Hybrid (this repo) |
|---|---|---|---|---|---|
| Local attention | Full O(L²) | None | GQA | Sliding GQA | Sliding GQA |
| SSM branch | No | Yes | Yes | No | Yes |
| Explicit memory | No | No | No | No | Dual gated banks |
| Sparse MoE | No | No | Yes | Yes | Yes |
| Time complexity | O(L²) | O(L) | O(L) | O(L) | O(L) |
| Long-range rare-fact recall | Window-limited | Recency bias | Recency bias | Window-limited | Targeted — unproven |

## 15. Related Work

- **Mamba** (Gu & Dao, 2023) — selective state-space models for linear-time sequence modeling; basis for the SSM branch.
- **Jamba** (Lieber et al., 2024) — hybrid Mamba–Transformer–MoE architecture; the closest prior work, lacking explicit queryable memory.
- **Mixtral** (Jiang et al., 2024) — Top-2 sparse MoE with load-balancing and z-loss regularization, reused directly for the FFN path.
- **Compressive Transformer** (Rae et al., ICLR 2020) — gated compressive memory; conceptual basis for `CompressiveMemoryBank` and the reconstruction loss.
- **Titans** (Behrouz et al., NeurIPS 2025) — associative-memory loss with surprise weighting; basis for the associative-recall auxiliary loss.
- **Perceiver / Perceiver IO** — attention over a small fixed latent set as a way to get O(L·m) cost instead of O(L²); the same trick underlies memory-bank reads here.

## 16. Roadmap

1. Run the falsification suite (§12) at small scale — hooks are already implemented, evaluation harness is not.
2. Build a needle-in-a-haystack / rare-fact recall evaluation harness with fixed seeds and controlled retrieval distance.
3. Match training tokens and peak activation memory, not just parameter count, in the null-baseline comparison.
4. Add a Jamba-style control (`use_dual_memory=False`) as the primary comparison point, not only the Mixtral baseline.
5. If memory proves redundant: simplify the architecture and redirect effort toward efficiency (fused scan, grouped MoE dispatch) rather than memory.
6. If memory proves useful: scale along a 10M → 100M → 1B ladder with a consistent evaluation protocol before making any long-context claims.

A longer, prioritized backlog — gate regularizers, content-addressable slots, FLOPs-matched controls, mixed-precision recipe, FSDP profiling — is tracked in [`research/Improvement-suggestions.md`](research/Improvement-suggestions.md).

## 17. Limitations

- No large-scale hyperparameter sweep has been run over the eight auxiliary-loss coefficients; current defaults are informed starting points from `loss-definitions.md`, not tuned values.
- Fused CUDA scan kernels are optional and untested at the largest sequence-length tiers on every hardware target; wall-clock linear-time claims are FLOP-true but not yet wall-clock-verified at 100K+ tokens.
- All current training runs are smoke-scale (toy config, ~200M-parameter IMDB run); no long-context or large-scale training run has been completed.
- The central scientific claim — that dual memory improves rare-fact recall beyond what a larger Mamba state alone provides — is explicitly unproven pending the falsification suite in §12.

## 18. Citation

If you use this codebase or build on the architecture, please cite the design document:

```
Vishnu Vardhan. "Hybrid Mamba–MoE with Dual Memory: A Sub-Quadratic Architecture
for Long-Context Language Modeling." Research Proposal, August 2026.
https://github.com/SVISHNUVARDHAN3610/CustomMistralmamba
```

For the complete architectural specification, see [`model/README.md`](model/README.md). For the full research proposal and evaluation plan, see [`research/research.md`](research/research.md). For loss formulas and tuning guidance, see [`research/loss-definitions.md`](research/loss-definitions.md).