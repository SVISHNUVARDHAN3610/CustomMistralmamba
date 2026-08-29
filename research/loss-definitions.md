# Loss Function Definitions — Hybrid Mamba–MoE with Dual Memory

**Model:** `HybridForCausalLM` / `HybridMambaMoEConfig` (`model.py`)  
**Design reference:** `research/research.md`  
**Status:** Implemented in `model.py` (enabled via `use_auxiliary_losses=True` by default).

---

## 1. Purpose

This file defines the **current** training objective and **eight proposed auxiliary losses** intended to:

1. Give the memory **write** pathway a reliable training signal under chunked long-context training.
2. Align optimization with the falsification tests in `research/research.md` (rare-fact recall, write-gate health, parameter-matched null baseline).
3. Stabilize routing and fusion without replacing the primary language-modeling objective.

Terminology here is aligned with the actual code (`CompressiveMemoryBank`, `TokenGatedFusion`, `MOERouter`, `_forward_chunked`, etc.).

---

## 2. Current training objective (implemented)

`HybridForCausalLM` currently optimizes **three terms**:

```python
loss = (
    ce_loss
    + router_aux_loss_coef * aux_loss  # default 0.02
    + router_z_loss_coef * z_loss  # default 5e-3
)
```

| Term | Symbol | Where computed | What it trains | Default coef |
|------|--------|----------------|----------------|--------------|
| Cross-entropy | `ce_loss` | `HybridForCausalLM.forward` | All parameters upstream of `lm_head` | 1.0 (implicit) |
| Router load-balancing | `aux_loss` | `MOERouter.forward` | `MOERouter.wg` — Switch Transformer style token/expert balance (Fedus et al., 2021) | 0.02 |
| Router z-loss | `z_loss` | `MOERouter.forward` | `MOERouter.wg` — penalizes large `logsumexp` of router logits (ST-MoE stabilizer, Zoph et al., 2022) | 5e-3 |

**Not in the loss today:**

- Memory read/write parameters (`CompressiveMemoryBank`)
- Fusion gate (`TokenGatedFusion`)
- Mamba SSM state (beyond indirect `ce_loss`)
- Expert specialization beyond load balancing

`gate_stats` (per-layer write-gate means) are computed and returned for **logging only** — they are `.detach()`-ed and never added to `loss`.

`label_ignore_index` (default `-100`) is applied via `CrossEntropyLoss(ignore_index=...)` and `_apply_label_ignore()` for padded label positions.

---

## 3. Why additional losses are proposed

### 3.1 The write-path gradient problem (structural, not just detachment)

An older report cited explicit `detach_memory_states()` between chunks. **That call does not exist in the current `model.py`.** `_forward_chunked()` threads `memory_states` across chunks inside one backward pass (truncated BPTT).

However, the **structural** gradient issue remains:

| Path | Within one chunk | Across chunks |
|------|------------------|---------------|
| **Read** (`read` → `*_memory_combine` → branch → fusion → MoE → `lm_head`) | Receives `ce_loss` gradient | Receives `ce_loss` gradient from future chunks |
| **Write** (`write` → updated memory) | Output is **not** read again in the same chunk | Receives `ce_loss` only if a **later** chunk reads the written memory |

Consequences:

- In a **single-chunk** forward (`seq_len ≤ memory_chunk_size`), write parameters may receive **no** gradient from `ce_loss`.
- In multi-chunk training, the **last chunk's write** receives no `ce_loss` gradient.
- Write parameters are trained only **indirectly** and **sparsely** through future reads — unlike read parameters, which are on the main logits path every chunk.

This matches the motivation in Compressive Transformers (Rae et al., ICLR 2020): local auxiliary losses help when full BPTT through memory is expensive or incomplete.

### 3.2 Link to falsification tests

| Falsification test (`research/research.md` §6) | Losses that support it |
|-----------------------------------------------|------------------------|
| **Test 1** — rare-fact recall (memory-on vs memory-zeroed) | #1 Reconstruction, #2 Associative retrieval |
| **Test 2** — write-gate activity (not saturated) | #3 Write-gate entropy |
| **Test 3** — parameter-matched null baseline | No new loss required; uses `build_test3_null_baseline_config()` |

Losses #4–#8 are protective / secondary refinements once the memory subsystem is actually trainable.

---

## 4. Proposed auxiliary losses (8)

Losses are ranked by priority. All are **training-only** — none affect `generate()` or inference.

---

### Loss 1 — Compressive Memory Reconstruction (`L_recon`)

| Field | Detail |
|-------|--------|
| **Priority** | Very high — correctness fix for write-path training |
| **Goal** | Train `write_attn`, `write_gate`, `write_update` to produce summaries that retain chunk information, even when `ce_loss` does not reach the write path |
| **Applies to** | Both `attn_memory_bank` and `state_memory_bank` (separate decoders or shared architecture per bank) |

**Intuition:** After `write()` compresses branch outputs `x` into a summary `s` (via attention pooling inside `write()`), a lightweight decoder `g` must reconstruct `x` from `s`. If reconstruction succeeds, the compressed representation is informative — the training-time analogue of Test 1, but active during training.

**Formula:**

Given chunk branch output `x ∈ ℝ^{B×L×d}` and write summary `s ∈ ℝ^{B×m×d}` (the attention-pooled summary already computed inside `write()`):

```
L_recon = (1 / (B·L·d)) · ‖ x − g(s) ‖²₂
```

- `g`: single cross-attention layer — token positions of `x` as queries, `s` as keys/values; **no FFN** (deliberately weak decoder).
- `x`: `attn_out` or `mamba_out` **before** the write gate blends into memory (the content being compressed).

**Hyperparameters:**

| Param | Default | Range | Notes |
|-------|---------|-------|-------|
| `lambda_recon` | `0.08` | `[0.05, 0.15]` | Larger than router aux — may be the primary write-path signal |
| Decoder heads | `2` | `[1, 4]` | Training-only scaffold; droppable at inference |
| Decoder depth | 1 cross-attn layer | — | Underpowered on purpose |

**Tuning signals:** `L_recon` plateaus high → `memory_size` too small or `lambda_recon` too low; drops to ~0 immediately → decoder overfitting, reduce `lambda_recon`.

**References:**

- Rae et al., *Compressive Transformers for Long-Range Sequence Modelling*, ICLR 2020 ([arXiv:1911.05507](https://arxiv.org/abs/1911.05507)) — auto-encoding and attention-reconstruction auxiliary losses for compressive memory.
- Martins et al., *∞-former: Infinite Memory Transformer*, ACL 2022 ([arXiv:2109.00301](https://arxiv.org/abs/2109.00301)) — reconstruction-style auxiliaries in compressive-memory models.

**Implementation note:** The Compressive Transformer paper sometimes trains the compression network with a **separate** objective from the main LM loss. This proposal **adds** `L_recon` to the total loss with coefficient `lambda_recon` — a deliberate design choice to be validated empirically.

---

### Loss 2 — Associative Key–Value Retrieval (`L_assoc`)

| Field | Detail |
|-------|--------|
| **Priority** | Very high — direct training-time proxy for Test 1 |
| **Goal** | After `write()`, verify that reading the updated memory with a key recovers the stored value |
| **Depends on** | Loss #1 (memory should contain something meaningful before retrieval is scored) |

**Intuition:** Simplified Titans-style associative memory objective. Sample token positions, project branch outputs to `(key, value)` pairs, read from **post-write** memory, and penalize retrieval error. Weight positions by a cheap "surprise" proxy so poorly reconstructed tokens (often rare facts) contribute more.

**Formula:**

After `write()` produces `M_new`, sample `T` token positions. Let `k_t, v_t` be linear projections of branch output at position `t`. Let `v̂_t = read(query=k_t, memory=M_new)`.

```
s_t = stopgrad( ‖ x_t − g(s) ‖₂ )          # per-token reconstruction residual from Loss #1

v̄_t = v_t / ‖v_t‖₂ ,  v̂̄_t = v̂_t / ‖v̂_t‖₂   # L2-normalize (stability, see below)

L_assoc = (1/T) · Σ_t  clip(s_t, 0, 3σ) · mean_d( (v̂̄_t − v̄_t)² )
```

- `T`: sampled positions per chunk (default 24; constant cost vs `L`).
- `s_t`: surprise weight — **proxy** for Titans' gradient-of-loss surprise (cheaper at this scale).
- `clip(s_t, 0, 3σ)`: per-chunk, prevents one outlier from dominating.
- **Stability (v4 revision):** the original form penalized `‖v̂_t − v_t‖²₂` — an
  *unnormalized* `.sum(dim=-1)` over `d` hidden dims of raw linear outputs,
  which is unbounded in both `d` and projection magnitude. A single
  large-magnitude chunk drove `L_assoc` to 1e8–1e23 in the 22.9k-step run
  (steps 2800/5400/10200/10600 in `metrics.jsonl`), poisoning the total loss
  and gradients. The revision (a) L2-normalizes `v̂_t` and `v_t` so the
  per-sample error is bounded in `[0, 4]`, (b) takes the **mean** over `d`
  instead of the sum, and (c) clamps per-sample error at `assoc_err_clip`
  (default 25, a backstop; normalization already bounds it below this).

**Hyperparameters:**

| Param | Default | Range | Notes |
|-------|---------|-------|-------|
| `lambda_assoc` | `0.0614` | `[2e-2, 2e-1]` | Ramp from 0 over first 5% of training steps; **rescaled for the normalized/mean formulation** — `0.0614 ≈ 1.2e-4 · 512` preserves the original effective weighting for `hidden_size=512` |
| `assoc_err_clip` | `25.0` | `[4, 100]` | Hard per-sample error clamp (backstop; normalized err ≤ 4) |
| `T` | `24` | `[16, 32]` | Fixed sample count per chunk |
| Warm-up | 0 → full over steps 0–5% | — | Memory needs basic content before retrieval scoring |

**References:**

- Behrouz et al., *Titans: Learning to Memorize at Test Time*, NeurIPS 2025 ([arXiv:2501.00663](https://arxiv.org/abs/2501.00663)) — associative memory loss `‖M(k) − v‖²` with surprise-weighted updates.
- Nelson et al., *Needle in the Haystack for Memory Based Large Language Models (Larimar)*, 2024 ([arXiv:2407.01437](https://arxiv.org/abs/2407.01437)) — KV-binding objectives improve needle retrieval.

---

### Loss 3 — Memory Write-Gate Entropy (`L_gate`)

| Field | Detail |
|-------|--------|
| **Priority** | High — cheap stabilization |
| **Goal** | Prevent write gates from saturating at 0 or 1 (always-overwrite / never-update failure mode) |
| **Applies to** | `a_write_gate`, `s_write_gate` already returned by `CompressiveMemoryBank.write()` |

**Formula:**

For gate tensor `g ∈ (0,1)^{B×m×d}`:

```
H(g) = −[ g·log(g+ε) + (1−g)·log(1−g+ε) ]     (per element)

L_gate = −mean(H(g))    # maximize entropy → push away from {0, 1}
```

Sum/average over both banks and all layers.

**Hyperparameters:**

| Param | Default | Range |
|-------|---------|-------|
| `lambda_gate` | `1e-3` | `[1e-4, 5e-3]` |
| `ε` | `1e-6` | fixed |

**Validation:** `gate_stats["*_write_gate_mean"]` should move off saturated extremes after this term is active (Test 2 monitoring metric becomes a training signal).

**References:** Rae et al. 2020 (gated memory needs aux signal beyond truncated BPTT); Titans ablations on forgetting/gating mechanisms.

---

### Loss 4 — Memory Read Utilization (`L_read`)

| Field | Detail |
|-------|--------|
| **Priority** | High — closes the read-side bypass gap |
| **Goal** | Prevent `attn_memory_combine` / `state_memory_combine` from learning to ignore the memory-read half of their input |
| **Cost** | Negligible — operates on weight norms only |

**Formula:**

For combine layer weight `W = [W_own | W_mem] ∈ ℝ^{d×2d}`:

```
r = ‖W_mem‖_F / (‖W_own‖_F + ‖W_mem‖_F + ε)

L_read = max(0, r_min − r)²
```

Hinge: zero penalty once memory-read fraction exceeds floor `r_min`.

**Hyperparameters:**

| Param | Default | Range |
|-------|---------|-------|
| `lambda_read` | `5e-3` | `[1e-3, 1e-2]` |
| `r_min` | `0.15` | `[0.10, 0.25]` |

Applied per layer to both `attn_memory_combine` and `state_memory_combine`.

---

### Loss 5 — Fusion-Gate Balance (`L_fusion`)

| Field | Detail |
|-------|--------|
| **Priority** | Medium–high — protects Mamba / state-memory investment |
| **Goal** | Prevent `TokenGatedFusion` from collapsing to "always attention branch" |
| **Analogue** | Load-balancing for the attention-vs-Mamba routing decision (like `aux_loss` for MoE) |

**Formula:**

For fusion gate `g ∈ [0,1]^{B×L×d}` from `TokenGatedFusion.forward()`:

```
ḡ = mean_{B,L}(g) ∈ ℝ^d

L_fusion = ‖ ḡ − 0.5 ‖²₂ / d
```

Only the **batch mean** is pulled toward 0.5 — individual tokens remain free.

**Hyperparameters:**

| Param | Default | Range |
|-------|---------|-------|
| `lambda_fusion` | `8e-3` | `[5e-3, 1e-2]` |
| Target | `0.5` | fixed |

**Implementation note:** `HybridDecoderLayer` currently discards the fusion gate (`fused, _fusion_gate = self.fusion(...)`). Implementation must retain `g` for this loss.

---

### Loss 6 — Expert Specialization (`L_expert`)

| Field | Detail |
|-------|--------|
| **Priority** | Medium — orthogonal to the memory research question |
| **Goal** | Encourage top-2 selected experts to produce different outputs and more discriminative routing |
| **Complements** | `aux_loss` / `z_loss` (balance *which* expert; this shapes *what* experts learn) |

**Formula:**

For token `t` routed to top-2 experts `{i, j}` with outputs `e_i(t), e_j(t) ∈ ℝ^d`:

```
L_ortho = (1/|T|) · Σ_t | cos_sim(e_i(t), e_j(t)) |

L_var   = −(1/|T|) · Σ_t Var_e[ softmax(router_logits(t)) ]    # per Guo et al. 2025

L_expert = L_ortho + β · L_var
```

**Hyperparameters:**

| Param | Default | Range | Notes |
|-------|---------|-------|-------|
| `lambda_expert` | `2e-3` | `[1e-3, 5e-3]` | Below `router_aux_loss_coef` (0.02) |
| `β` | `0.5` | `[0.3, 1.0]` | Ortho vs variance balance |
| Warm-up | Step on at ~10% training | — | Let router balance first via `aux_loss` |

**References:**

- Guo et al., *Advancing Expert Specialization for Better MoE*, NeurIPS 2025 Oral ([arXiv:2505.22323](https://arxiv.org/abs/2505.22323)).
- Fedus et al., *Switch Transformers*, JMLR 2022 — `aux_loss` baseline this complements.

---

### Loss 7 — SSM State Norm Regularization (`L_ssm`)

| Field | Detail |
|-------|--------|
| **Priority** | Medium — numerical / representational health |
| **Goal** | Keep Mamba SSM state norms in a healthy range at long context |
| **Framing** | **Not** overflow prevention (`deltaA ∈ (0,1)` already bounds divergence) — addresses slow-decay / precision drift when state norm grows large relative to recent-token signal |

**Formula:**

For final per-chunk SSM state `s` (or mean over scan steps):

```
s̄ = mean_t ‖s_t‖²₂

L_ssm = max(0, s̄ − γ)
```

`γ`: 90th percentile of `‖s_t‖²` measured once at initialization (config-dependent; not a hardcoded constant).

**Hyperparameters:**

| Param | Default | Range |
|-------|---------|-------|
| `lambda_ssm` | `1e-5` | `[1e-6, 1e-4]` |
| `γ` | measured at init | — |

Aggregated as mean over layers.

---

### Loss 8 — Memory Slot Diversity (`L_slot`)

| Field | Detail |
|-------|--------|
| **Priority** | Medium–low — capacity utilization |
| **Goal** | Prevent `m=64` slots from collapsing to near-identical content within a bank |
| **Cost** | `O(m²·d)`, trivial for `m=64` |

**Formula:**

For memory `M ∈ ℝ^{B×m×d}`, row-normalize `M̂_p = M_p / ‖M_p‖`:

```
L_slot_intra = (1/m²) · Σ_{p≠q} max(0, cos_sim(M̂_p, M̂_q) − τ)²

L_slot_cross = (1/m) · Σ_p |cos_sim(M̂_p^attn, M̂_p^state)|    # weak decorrelation (abs prevents anti-aligned slots)

L_slot = L_slot_intra + α · L_slot_cross
```

**Hyperparameters:**

| Param | Default | Range |
|-------|---------|-------|
| `lambda_slot` | `3e-3` | `[1e-3, 8e-3]` |
| `τ` (intra margin) | `0.3` | `[0.2, 0.5]` |
| `α` (cross-bank) | `0.1` | `[0.05, 0.2]` |

**References:** Locatello et al., *Object-Centric Learning with Slot Attention*, NeurIPS 2020; van den Oord et al., *VQ-VAE*, NeurIPS 2017 (codebook/slot collapse).

---

### Loss 9 — Associative Memory State Norm (`L_assoc_norm`) *(v4 addition)*

| Field | Detail |
|-------|--------|
| **Priority** | Low–medium — numerical health of the recurrent bank state |
| **Goal** | Keep the post-write compressive-memory state bounded (`ssm_state_norm_loss` analogue) |
| **Why** | The recurrent update `M ← g·M + (1−g)·write_update(summary)` receives little/no CE gradient within a chunk; `write_update` outputs are unbounded linear projections, so the bank state can drift large over many chunks (late-run gate saturation >0.9 in layers 5–7 correlates with this drift). |

**Formula:**

For post-write bank state `M ∈ ℝ^{B×m×d}`:

```
M̄ = mean_{B,m,d}( M² )

L_assoc_norm = max(0, M̄ − γ)
```

`γ`: max of the 90th-percentile post-write state norms of both banks, measured
once from a dummy write at initialization (`assoc_norm_gammas` buffer; fallback
`assoc_norm_gamma_init = 1e-3` before calibration). Summed over both banks per
layer and averaged over layers, like every other aux loss.

**Hyperparameters:**

| Param | Default | Range | Notes |
|-------|---------|-------|-------|
| `lambda_assoc_norm` | `1e-3` | `[1e-4, 1e-2]` | Same order as `lambda_gate` — a soft leash, not a hard clamp |
| `γ` quantile | `0.9` | `[0.8, 0.99]` | Calibration percentile (`assoc_norm_gamma_quantile`) |

---

## 5. Combined objective (proposed)

**Keep unchanged:** `ce_loss`, `aux_loss`, `z_loss` — same coefficients, same implementations.

```python
L_total = (
    ce_loss
    + router_aux_loss_coef * aux_loss  # 0.02
    + router_z_loss_coef * z_loss  # 5e-3
    + lambda_recon * L_recon  # 0.08
    + assoc_weight(step)
    * L_assoc  # → 0.0614 (5% warm-up; normalized vectors, mean over hidden)
    + lambda_gate * L_gate  # 1e-3
    + lambda_read * L_read  # 5e-3
    + lambda_fusion * L_fusion  # 8e-3
    + expert_weight(step) * L_expert  # → 2e-3 (on at 10%)
    + lambda_ssm * L_ssm  # 1e-5
    + lambda_slot * L_slot  # 3e-3
    + lambda_assoc_norm * L_assoc_norm  # 1e-3 (v4: memory-state bound)
)
```

`assoc_weight(step)` and `expert_weight(step)` are schedule functions (linear ramp / step-on), not fixed constants.

### Coefficient rationale (summary)

| Group | Losses | Magnitude | Role |
|-------|--------|-----------|------|
| Primary | `ce_loss` | 1.0 | Language modeling |
| Router | `aux_loss`, `z_loss` | ~1e-2 | MoE balance + logit stability |
| Memory correctness | `L_recon`, `L_assoc` | ~1e-2 | Write-path training + retrieval |
| Stabilization | `L_gate`, `L_read`, `L_fusion`, `L_ssm`, `L_slot` | ~1e-5 – 8e-3 | Prevent known failure modes |
| MoE refinement | `L_expert` | 2e-3 | Specialization (secondary) |

**Monitoring recommendation:** Log each term's **raw** (unweighted) value and weighted contribution for the first few hundred steps. Re-tune any coefficient whose weighted gradient norm exceeds ~5% of `ce_loss`.

---

## 6. Comparative summary

| Rank | Loss | Primary target | Expected impact | Compute | Impl. difficulty | Default λ |
|------|------|----------------|-----------------|---------|-------------------|-----------|
| 1 | `L_recon` | Write-path gradient starvation | Very high (correctness) | Low–moderate | Moderate | 0.08 |
| 2 | `L_assoc` | Rare-fact recall (Test 1) | High | Low | Moderate–high | 0.03 |
| 3 | `L_gate` | Write-gate saturation (Test 2) | Moderate | Negligible | Low | 1e-3 |
| 4 | `L_read` | Combine-layer memory bypass | Moderate | Negligible | Low | 5e-3 |
| 5 | `L_fusion` | Attention-branch collapse | Moderate (protective) | Negligible | Low | 8e-3 |
| 6 | `L_expert` | Expert redundancy | Small–moderate | Low | Low–moderate | 2e-3 |
| 7 | `L_ssm` | Long-context SSM precision | Small–moderate | Negligible | Moderate | 1e-5 |
| 8 | `L_slot` | Slot collapse | Moderate (capacity) | Low | Moderate | 3e-3 |
| 9 | `L_assoc_norm` | Memory-state drift (v4) | Small | Negligible | Low | 1e-3 |

---

## 7. Expected training dynamics

| Phase | What happens |
|-------|--------------|
| **Steps 0–5%** | Effectively current 3-loss setup + `L_recon`, `L_gate`, `L_read`, `L_fusion`, `L_ssm`, `L_slot`. `L_assoc` ramping in. |
| **Steps 5–10%** | `L_assoc` at full weight; memory write/read pathway actively shaped. Watch `L_recon` ↓ and `gate_stats` move off init. |
| **Steps 10%+** | `L_expert` activates; expert outputs should diverge without fighting settled load balancing. |

---

## 8. Trade-offs and open decisions

| Topic | Notes |
|-------|-------|
| **Auxiliary-loss budget** | Nine small terms still add up — gradient-norm audit early in training is mandatory. |
| **Loss #1 decoder** | Training-only parameters; can be omitted from inference checkpoints. |
| **Mixed vs separate optimizers** | Compressive Transformer sometimes used separate compression training; this design uses one combined loss — validate that write path does not collapse activations to ease reconstruction. |
| **Deferred 9th candidate** | Contrastive memory-key structuring (Focused-Transformer-style) — deferred until Tests 1–3 confirm memory is worth keeping. |
| **Scale sensitivity** | Default λ values assume `HybridMambaMoEConfig` defaults; larger `hidden_size` / `num_layers` may need re-scaling. |

---

## 9. What this document does **not** cover

- Hyperparameter sweeps or ablation protocol
- Changes to `ce_loss`, `aux_loss`, or `z_loss` formulations
- Inference-time behavior (`generate()`, falsification eval harnesses)

---

## 10. References (consolidated)

| Short name | Citation |
|------------|----------|
| Compressive Transformer | Rae et al., ICLR 2020, [arXiv:1911.05507](https://arxiv.org/abs/1911.05507) |
| ∞-former | Martins et al., ACL 2022, [arXiv:2109.00301](https://arxiv.org/abs/2109.00301) |
| Titans | Behrouz et al., NeurIPS 2025, [arXiv:2501.00663](https://arxiv.org/abs/2501.00663) |
| Larimar | Nelson et al., 2024, [arXiv:2407.01437](https://arxiv.org/abs/2407.01437) |
| Switch Transformer | Fedus et al., JMLR 2022 |
| ST-MoE / z-loss | Zoph et al., 2022 |
| Expert Specialization MoE | Guo et al., NeurIPS 2025 Oral, [arXiv:2505.22323](https://arxiv.org/abs/2505.22323) |
| Slot Attention | Locatello et al., NeurIPS 2020 |
| VQ-VAE | van den Oord et al., NeurIPS 2017 |

---

*Coefficients and schedules are defined in `HybridMambaMoEConfig` and combined in `HybridForCausalLM._weighted_auxiliary_loss()`.*
