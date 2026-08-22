# CustomMistralMamba: AI Agent Knowledge Base & Repository Guide

This document serves as the primary technical specification, architectural reference, and operational playbook for any AI agent working on the `CustomMistralMamba` codebase.

---

## 1. Project Overview & Scientific Motivation

### 1.1 Problem Statement
Modern Large Language Models (LLMs) built on decoder-only Transformers suffer from three fundamental bottlenecks as context length $L$ scales:
1. **Quadratic Compute Complexity ($O(L^2)$):** Standard full self-attention materializes $L \times L$ attention matrices, making 100K+ token inference and training computationally prohibitive.
2. **Lack of Explicit, Addressable Memory:** Contextual knowledge resides implicitly in key-value (KV) caches or continuously overwritten recurrent states. In long contexts, rare, one-off facts stated early in the prompt suffer from recency bias or fall outside sliding attention windows.
3. **Uniform Per-Token Compute:** Standard Transformer decoders route every token through the same dense Feed-Forward Network (FFN), expending identical compute regardless of token difficulty.

### 1.2 The Research Hypothesis
`CustomMistralMamba` investigates whether an explicit, bounded-size, gated read/write memory system can recover rare long-range information that both sliding-window attention and state-space models lose over extended contexts—**without paying quadratic compute or memory costs**.

### 1.3 Scientific Status & Falsification Protocol
The dual-memory mechanism is a **research hypothesis under test**, not an established theorem. Mamba's hidden state is already a continuous compression of past context. Before claiming that explicit memory is necessary, three falsification tests must be satisfied (`research/research.md` §6):
1. **Test 1 (Ablation-at-Inference):** Rare-fact recall must degrade when memory states are zeroed at inference (`HybridModel.zero_memory_states()`) compared to when memory is intact.
2. **Test 2 (Non-Degenerate Gate Activity):** Memory write gates (`gate_stats`) must exhibit active, non-saturated dynamics during training (prevented from collapsing to 0 or 1 via $L_{gate}$).
3. **Test 3 (Matched-Parameter Null Hypothesis):** The hybrid model with memory must outperform a parameter-matched control with `use_dual_memory=False` and an expanded Mamba state (`build_test3_null_baseline_config()`).

If memory fails these tests, the documented plan is to simplify the architecture to `use_dual_memory=False` (a lean Jamba-style hybrid) rather than adding unearned complexity.

---

## 2. Architecture Specification

The architecture comprises two model families sharing core neural primitives:
- **Baseline Family (`MixtralForCausalLM` / `MixtralConfig`):** Sliding-Window GQA + Top-2 MoE (control ablation).
- **Hybrid Family (`HybridForCausalLM` / `HybridMambaMoEConfig`):** Dual-Branch GQA + Mamba-SSM conditioned on Dual Compressive Memory Banks, fused via per-token gating, followed by Top-2 Sparse MoE.

### 2.1 Per-Layer Architecture & Data Flow

```
Input x (hidden_size d)
  │
  ├──────────────────────────────────────────────────────────┐ (Residual 1)
  ▼                                                          │
RMSNorm                                                      │
  │                                                          │
  ├─► [batched_dual_memory_read from attn_bank]  ──► attn_combine([x_norm; a_read])  ──► SlidingWindowGQA ──► attn_out
  │                                                                                                             │
  └─► [batched_dual_memory_read from state_bank] ──► state_combine([x_norm; s_read]) ──► MambaBlock       ──► mamba_out
                                                                                                                │
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Write Path:                                                                                                 │
  │ Append raw (attn_out, mamba_out) to MemoryWriteBuffer (with validity mask)                                  │
  │ If write_interval reached: batched_dual_memory_write -> update (attn_bank, state_bank)                       │
  │ (In training + aux mode: computes L_recon, L_assoc, L_gate, L_slot)                                         │
  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
                                                                                                                │
  attn_out & mamba_out ────────────────────────────────────────────────────────► TokenGatedFusion ─────────────┘
                                                                                        │
                                                                                        ▼
                                                                                   fused + Residual 1
                                                                                        │
  ┌─────────────────────────────────────────────────────────────────────────────────────┤ (Residual 2)
  ▼                                                                                     │
RMSNorm                                                                                 │
  │                                                                                     │
Top-2 Sparse Dropless MoE (SwiGLU Experts)                                              │
  │                                                                                     │
  ▼                                                                                     │
moe_out ────────────────────────────────────────────────────────────────────────────────┴──► Layer Output x_out
```

### 2.2 Critical Component Breakdown

#### 1. Sliding-Window Grouped-Query Attention (`SlidingWindowGQA`)
- **File:** `model/layers/attention.py`
- **Complexity:** $O(L \cdot w)$, where $w = \text{window\_size}$ (default 4096).
- **Mechanics:** Grouped-Query Attention with $N_q$ query heads and $N_{kv}$ key/value heads ($N_q \pmod{N_{kv}} = 0$). Uses `RotaryEmbedding` (`model/layers/rope.py`) with base $\theta=10000.0$ and pre-allocated non-growing buffers up to `max_position_embeddings`.
- **KV Truncation & Mask Alignment:** KV cache is bounded to the trailing `window_size` tokens. **Crucial Invariant:** When KV is truncated, `attention_mask` is truncated to match to prevent dimension mismatch crashes during generation.
- **SDPA:** Relies exclusively on `torch.nn.functional.scaled_dot_product_attention`.

#### 2. Mamba Selective State-Space Branch (`MambaBlock`)
- **File:** `model/hybrid/mamba.py`
- **Complexity:** $O(L \cdot d \cdot n)$, where $n = \text{mamba\_state\_size}$ (default 16).
- **Scan Dispatch Tiers:**
  1. **Tier 1 (Fused CUDA):** `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` on unpadded CUDA batches.
  2. **Tier 1b (Unpadded Fused CUDA):** Runs fused scan per row on valid prefixes, using non-in-place `cat/pad/stack` to preserve autograd graphs without NaNing training.
  3. **Tier 2 (Parallel Scan):** Hillis-Steele associative scan ($O(L \log L)$) when $L \le 4096$ or `use_parallel_scan=True`.
  4. **Tier 3 (Blocked Scan):** Blocked vectorized associative scan for $4096 < L \le 65536$.
  5. **Tier 4 (Sequential Scan):** Checkpointed sequential scan for $L > 65536$.
- **Decode Fast Path:** `MambaBlock.step()` performs single-token state recurrence $h_t = \bar{A} h_{t-1} + \bar{B} x_t$ with in-place rolling conv buffer updates.

#### 3. Dual Compressive Memory Banks (`CompressiveMemoryBank` & `MemoryWriteBuffer`)
- **File:** `model/hybrid/memory.py`
- **Complexity:** $O(L \cdot m)$ per layer, where $m = \text{memory\_size}$ (default 64 slots).
- **Memory Conditioning Semantics:**
  - **Read:** Multi-head attention where token representations are queries and memory slots are keys/values. The read vector is concatenated with normalized input and projected through `*_memory_combine` to condition branch inputs.
  - **Write:** A learned `summary_query` attends over the chunk tokens. A GRU-style gate blends the summary into memory: $\text{gate} = \sigma(W[\text{mem}; \text{summary}])$, $\text{new\_mem} = \text{gate} \odot \text{mem} + (1 - \text{gate}) \odot \text{update}(\text{summary})$.
  - **Input/Output Decoupling:** Memory conditions branch *inputs*, but writes are produced from raw branch *outputs* (`attn_out`, `mamba_out`).
- **Batched Operations:** `batched_dual_memory_read` and `batched_dual_memory_write` stack parameters and states across both banks to execute in a single batched kernel.
- **Write Buffer:** `MemoryWriteBuffer` holds raw branch outputs with explicit boolean validity masks (`mask_buf`) across decoding steps until `memory_write_interval` tokens accumulate.

#### 4. Token-Wise Gated Fusion (`TokenGatedFusion`)
- **File:** `model/layers/fusion.py`
- **Complexity:** $O(L \cdot d^2)$ — linear in sequence length.
- **Mechanism:** $g = \sigma(W_{fuse} [a; m])$, $\text{fused} = g \odot a + (1 - g) \odot m$. Completely avoids quadratic cross-attention between branches.

#### 5. Dropless Top-2 Sparse MoE (`DroplessMoELayer`, `MOERouter`, `SwiGLUExpert`)
- **File:** `model/layers/moe.py`
- **Router:** Computes top-$k$ (default 2) routing over $E$ (default 8) experts. Logits are clamped to $[-30.0, 30.0]$ for FP16 stability.
- **Dispatch Modes:**
  - `use_grouped_gemm=True`: Uses `torch._grouped_mm` when supported.
  - `use_grouped_moe_dispatch=True`: Sorts tokens by expert index and runs stacked weights for minimal kernel launches.
  - Loop fallback: Iterates over experts sequentially.
- **Capacity:** Dropless by default (`capacity_factor=None`) for deterministic research reproducibility.

---

## 3. Training Objective & Auxiliary Loss System

### 3.1 Total Loss Formulation
The overall training objective combines the primary causal language modeling loss, MoE routing stabilizers, and eight memory/branch auxiliary losses:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha_{\text{aux}} \mathcal{L}_{\text{router\_aux}} + \alpha_{z} \mathcal{L}_{\text{router\_z}} + \sum_{i=1}^{8} \lambda_i \mathcal{L}_i$$

| Term | Coefficient / Schedule | Description & Purpose |
|---|---|---|
| **$\mathcal{L}_{\text{CE}}$** | $1.0$ (implicit) | Cross-entropy on valid next-token predictions. |
| **$\mathcal{L}_{\text{router\_aux}}$** | $\alpha_{\text{aux}} = 0.02$ | Switch-Transformer load-balancing across experts ($E \sum f_i p_i$). |
| **$\mathcal{L}_{\text{router\_z}}$** | $\alpha_{z} = 0.005$ | ST-MoE router logit $z$-loss ($\text{mean}(\text{logsumexp}(logits)^2)$). |
| **$\mathcal{L}_{\text{recon}}$ (Loss 1)** | $\lambda_{\text{recon}} = 0.08$ | **Compressive Reconstruction:** Lightweight cross-attention decoder reconstructs chunk tokens from write summary. Provides direct write-path gradients. |
| **$\mathcal{L}_{\text{assoc}}$ (Loss 2)** | $\lambda_{\text{assoc}} = 1.2\times 10^{-4}$ (5% warmup) | **Associative Retrieval:** Titans-style surprise-weighted loss verifying post-write key $\to$ value retrieval from updated memory slots. |
| **$\mathcal{L}_{\text{gate}}$ (Loss 3)** | $\lambda_{\text{gate}} = 1\times 10^{-3}$ | **Write-Gate Entropy:** Maximizes binary entropy $-\text{mean}(H(g))$ to prevent write gates from saturating at 0 or 1. |
| **$\mathcal{L}_{\text{read}}$ (Loss 4)** | $\lambda_{\text{read}} = 5\times 10^{-3}$ | **Read Utilization:** Hinge penalty on combine weight norms to prevent bypassing memory reads ($r_{\text{min}} = 0.15$). |
| **$\mathcal{L}_{\text{fusion}}$ (Loss 5)** | $\lambda_{\text{fusion}} = 8\times 10^{-3}$ | **Fusion Balance:** Pulls batch-average fusion gate $\bar{g}$ toward 0.5 to prevent branch starvation. |
| **$\mathcal{L}_{\text{expert}}$ (Loss 6)** | $\lambda_{\text{expert}} = 2\times 10^{-3}$ (10% step-on) | **Expert Specialization:** Penalizes pairwise cosine similarity between top-2 expert outputs and encourages router variance. |
| **$\mathcal{L}_{\text{ssm}}$ (Loss 7)** | $\lambda_{\text{ssm}} = 1\times 10^{-5}$ | **SSM Norm Regularization:** Hinge penalty on SSM state norm against init-calibrated threshold $\gamma$. |
| **$\mathcal{L}_{\text{slot}}$ (Loss 8)** | $\lambda_{\text{slot}} = 3\times 10^{-3}$ | **Slot Diversity:** Penalizes intra-bank slot cosine similarity above margin $\tau=0.3$ and cross-bank slot alignment ($\alpha=0.1$). |

### 3.2 The Write-Path Gradient Problem
In single-chunk forwards or within standard BPTT, the write-path parameters ($\text{write\_gate}$, $\text{write\_update}$, $\text{summary\_query}$) receive zero direct gradients from $\mathcal{L}_{\text{CE}}$ because written memory is not re-read in the same chunk. $\mathcal{L}_{\text{recon}}$ and $\mathcal{L}_{\text{assoc}}$ solve this structural limitation. `HybridForCausalLM` enforces `use_auxiliary_losses=True` whenever `use_dual_memory=True`.

---

## 4. Execution Pipelines & Memory Lifecycle

### 4.1 Chunked Long-Context Training (`_forward_chunked`)
When sequence length $L > \text{memory\_chunk\_size}$ (default 512):
1. Input sequences are partitioned into contiguous chunks of length `memory_chunk_size`.
2. `memory_states` $(M_{\text{attn}}, M_{\text{state}})$ are threaded sequentially across chunks inside **one backward pass** (truncated BPTT).
3. **VRAM Optimization (`stream_chunked_ce_loss=True`):** Cross-entropy is computed per chunk exclusively on non-ignored label tokens. The full $[B, L, V]$ logits tensor is never materialized when `return_logits=False`.

### 4.2 Incremental Decode & Generation (`generate`)
1. **Chunked Prefill:** Prompt is evaluated in `memory_write_interval` slices; memory writes are flushed at chunk boundaries.
2. **Decode Step:** Appends single token representations to `MemoryWriteBuffer` via `append_single_token()`.
3. **Write Flush:** When the buffer reaches `memory_write_interval`, `batched_dual_memory_write` updates memory banks atomically.
4. **Finished Row Freezing:** Sequences hitting `eos_token_id` have their KV, Mamba, and memory updates masked out via `active_batch_mask`.
5. **CUDA Graph Decode:** Optional `use_cuda_graph=True` captures fixed-shape single-token decode for rapid inference on CUDA.

---

## 5. Training & Data Infrastructure

### 5.1 Dataset Pipeline (`utils/dataset.py`)
- **`TokenizedShardProducer`:** Streams multi-corpus datasets from HuggingFace (`FineWeb`, `FineWeb-Edu`, `The Stack v2`, `PG-19`, `Open-Web-Math`, `NuminaMath-CoT`, `ELI5`, `UltraChat-200k`), tokenizes with BOS/EOS wrapping, and serializes into binary `uint16` shards (`.bin` + `.json`). Supports fast native `IterableDataset` resumption.
- **`MmapShardDataset`:** Zero-copy memory-mapped consumer dataset yielding paired input/target chunks $(x_{0:L}, x_{1:L+1})$.
- **Vocab Safety:** `verify_tokenizer_vocab()` asserts strict equality between tokenizer size and `vocab_size` before training starts.

### 5.2 Cyclic Validation (`utils/validation.py`)
- **`WikiTextCyclicValidator`:** Runs periodic evaluation over sliding windows of `Salesforce/wikitext` validation split, computing exact validation CE, perplexity, and router auxiliary metrics.

### 5.3 Production Training Loop (`train.py`)
- **Optimizer Split:**
  - **Muon:** 2D hidden layer weight matrices and MoE router projections (scaled by Moonshot RMS adjustment $\propto 0.2 \sqrt{\max(A,B)}$).
  - **AdamW:** Embeddings (`embed_tokens`, `init_memory`, `summary_query`), LM head (`lm_head`), Conv1D, RMSNorm gains, and biases.
- **Learning Rate Schedule:** Cosine decay with linear warmup (`_build_lr_lambda`).
- **Mixed Precision & Autocast:** FP32 promotion for sensitive ops (`RMSNorm`, router logits, scan fallback recurrence, memory entropy/slot math).

---

## 6. Repository Layout

```
CustomMistralmamba/
├── README.md                     # High-level project summary and quickstart
├── pyproject.toml                 # Ruff configuration
├── requirements.txt               # Dependencies (torch, ruff, pre-commit)
├── train.py                       # Production training script (Muon+AdamW, streaming shards)
│
├── model/                         # Core Architecture Subpackage
│   ├── __init__.py                # Public API exports
│   ├── README.md                  # Detailed architectural reference manual (24 sections)
│   ├── core/                      # Configuration, dtype utilities, builders
│   │   ├── builders.py            # count_trainable_params, build_test3_null_baseline_config
│   │   ├── config.py              # MixtralConfig, HybridMambaMoEConfig, MambaCache
│   │   ├── constants.py           # MEMORY_NAN_FIX_ID revision marker
│   │   └── dtype.py               # FP32 promotion/restoration helpers
│   ├── layers/                    # Shared Neural Primitives
│   │   ├── attention.py           # SlidingWindowGQA (grouped-query sliding window)
│   │   ├── fusion.py              # TokenGatedFusion (O(L) per-token gating)
│   │   ├── moe.py                 # DroplessMoELayer, MOERouter, SwiGLUExpert
│   │   ├── norm.py                # RMSNorm (with FP32 precision promotion)
│   │   └── rope.py                # RotaryEmbedding (fixed-size buffer cache)
│   ├── mixtral/                   # Baseline Model Family
│   │   └── model.py               # MixtralDecoderLayer, MixtralModel, MixtralForCausalLM
│   └── hybrid/                    # Research Architecture
│       ├── layer.py               # HybridDecoderLayer (dual branch + dual memory wiring)
│       ├── losses.py              # Eight auxiliary losses & schedule functions
│       ├── mamba.py               # MambaBlock (4-tier scan dispatch, incremental decode)
│       ├── memory.py              # CompressiveMemoryBank, MemoryWriteBuffer, batched ops
│       └── model.py               # HybridModel, HybridForCausalLM, chunked BPTT, generate()
│
├── research/
│   ├── research.md                # Research proposal, compute analysis, falsification plan
│   ├── loss-definitions.md        # Mathematical specifications for all auxiliary losses
│   └── Improvement-suggestions.md # Backlog of deferred performance & scaling proposals
│
├── scripts/
│   ├── toy_train.py               # Minimal 5M-param standalone training smoke test
│   ├── test_cloud_train.py        # 150M-param IMDB training smoke test for cloud GPUs
│   └── verify_model_package.py    # Import and API surface integrity check
│
├── utils/
│   ├── dataset.py                 # TokenizedShardProducer, MmapShardDataset
│   └── validation.py              # WikiTextCyclicValidator for periodic eval
│
└── tests/
    ├── test_model.py              # 66 comprehensive unit tests covering all modules
    └── test_toy_train_smoke.py    # End-to-end smoke test for scripts/toy_train.py
```

---

## 7. Mandatory Rules for AI Agents Working on This Repository

### Rule 1: Local Environment vs. Cloud Execution
- **This local machine is strictly for development, testing, linting, and debugging.**
- Full-scale training runs (e.g., `train.py` with multi-gigabyte shards, large hidden sizes, or 100M+ parameters) are designed for cloud GPU clusters (A100/H100/T4).
- **Local Testing & Debugging Constraint:** When running local tests, scripts, or validation, use ONLY small configurations of **5M–10M parameters**, batch size $1\text{--}2$, short sequence lengths ($L \le 256\text{--}512$), and minimal steps ($5\text{--}10$).
- Use `scripts/toy_train.py` (`build_toy_config()`) or `_small_hybrid_config()` in `tests/test_model.py` as templates for local execution.

### Rule 2: Sanctity of Production Architecture & Configurations
- **NEVER modify the production architecture, default dataclass configurations in `model/core/config.py`, or default arguments in `train.py` just to make them fit on the local laptop.**
- If you need a smaller model for local testing, pass explicit override arguments or use dedicated test/toy configs.

### Rule 3: Git & File Hygiene
- **Do not read, analyze, modify, or commit `.venv` or any pattern listed in `.gitignore`** (`.env.local`, `config.yaml`, `__pycache__`, `reviews/`, `*.log`, `Improvments/`).
- Preserve all existing comments, docstrings, and revision markers (`MEMORY_NAN_FIX_ID`).

### Rule 4: Sub-Quadratic Complexity Invariant
- Every modification to the attention, Mamba, memory, fusion, or MoE pathways must strictly maintain $O(L)$ or $O(L \cdot w)$ / $O(L \cdot m)$ asymptotic complexity per layer.
- **NEVER introduce bidirectional sequence-to-sequence cross-attention across the full sequence length** between branches or memory banks.

### Rule 5: Memory Conditioning Contract
- Gated memory reads condition branch **inputs** via `*_memory_combine`.
- Memory writes are updated from **raw branch outputs** (`attn_out`, `mamba_out`), never from post-fusion or memory-augmented states.
- Writes must remain padding-safe and pass `MEMORY_NAN_FIX_ID` guards.

### Rule 6: Verification Protocol Before Committing Changes
Always use the virtual environment Python executable (`.\.venv\Scripts\python.exe`) for running scripts, testing, and linting. Verify correctness in this sequence:
1. Run package verification:
   ```bash
   .\.venv\Scripts\python.exe scripts/verify_model_package.py
   ```
2. Run unit tests using the virtual environment:
   ```bash
   .\.venv\Scripts\python.exe -m unittest discover tests
   ```
3. Run a quick 10-step toy training smoke run:
   ```bash
   .\.venv\Scripts\python.exe scripts/toy_train.py --steps 10 --batch-size 2 --seq-len 256
   ```
4. Run Ruff lint check:
   ```bash
   .\.venv\Scripts\ruff.exe check .
   ```
