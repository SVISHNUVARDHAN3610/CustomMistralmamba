# Codebase Audit — Risks & Findings

**Date:** 2026-08-25 · **Audit target:** commit `f22bf03` (+ uncommitted `.gitignore` edit) · **Branch:** `vsenapathi/test-ox-alpha`

**Method:** three independent read-only audit passes (model internals; training pipeline; inference/scripts/infra/security) plus an end-to-end runtime validation. Every finding below was re-verified against source by the orchestrator before recording; items that could be exercised locally were reproduced at runtime. Findings observed independently by two or more auditors are marked. No repository files were modified; the only artifact created is this file and the temporary fixture under `.claude/tmp_audit/` (both gitignored).

---

## Summary

| Severity | Count |
|---|---|
| High | 5 |
| Medium | 14 |
| Low | 14 |
| Potential risks (unconfirmed) | 17 |

No Critical-severity issue was found. The core architecture contracts (pre-shifted labels, right-padding validity, sub-quadratic compute, memory read/write decoupling, ablation hooks, NaN guards, AMP promotion points, fixed caches) were traced and **hold** — see "Verified clean".

---

## Confirmed issues

### HIGH

**H1. `generate()` crashes on every checkpoint produced by `train.py`**
- **Where:** `model/hybrid/model.py:723` (`logits = ... if self.config.return_logits else None`) vs `:1081` (`logits = out.logits[:, -1, :]`); configs: `train.py:148`, `scripts/toy_train.py:48`.
- **Description:** training configs set `return_logits=False`; `generate()`'s prefill forward does not force logit materialization, so `out.logits` is `None` and the first decode step raises `TypeError: 'NoneType' object is not subscriptable`.
- **Impact:** the primary inference surface is unusable on any as-trained model without hand-editing the loaded config.
- **Evidence:** runtime-reproduced twice (audit fixture and independent agent probe).
- **Remediation:** materialize last-step logits inside `generate()` (run `lm_head` on the final hidden state there, or temporarily force the flag around prefill), minimally raise a descriptive error.
- **Verified:** runtime repro + source trace (found independently by two auditors and the E2E run).

**H2. CUDA-graph decode corrupts recurrent state (latent; `use_cuda_graph=True` only, default False)**
- **Where:** `model/hybrid/model.py:411-471` (`_CudaDecodeGraphRunner.capture/replay`) + `model/hybrid/mamba.py:412-414`; decode integration `model/hybrid/model.py:1126-1157`.
- **Description:** `capture()` runs two side-stream warm-up forwards **against the live caches**; `MambaBlock.step()` shifts `conv_state` in place, so the current token is consumed twice before capture (probe: double-applying one token perturbs conv_state by ~30× its output scale). `replay()` copies back only ids/mask/positions — never KV/SSM/memory state — so replayed graphs read capture-time addresses: the attention window and memory states freeze at capture content and the SSM recurrence never chains across steps. Python-side `MemoryWriteBuffer.filled` also advances during warm-up/capture bookkeeping but not on replay.
- **Impact:** enabling the README-recommended fast path (`model/README.md` §16.2) silently produces garbage continuations from the first captured step. Mitigated today only by the default-off flag and a parity test that self-skips on CPU-only CI.
- **Evidence:** source-level proof of in-place vs out-of-place update semantics and missing copy-back; not GPU-reproducible locally (no CUDA).
- **Remediation:** rewrite around persistent captured buffers (in-place `copy_` into static caches, ring-buffer KV) or snapshot/restore caches around warm-up; until then disable/remove the feature and stop advertising it. Found independently by two auditors.
- **Verified:** code-order trace by orchestrator + both auditors.

**H3. Resume replaces the trainer's consume cursor with the producer's write cursor — silently skips up to `max_buffered_files` shards per restart**
- **Where:** `train.py:753-755` (overwrite) with `utils/dataset.py:132-168` (producer checkpoint = write cursor), `:352-359` (backpressure keeps ≤10 shards ahead, `--max-buffered-files` default `train.py:1149`), `:170-185` (cleanup deletes only `.done`-marked bins).
- **Description:** on `--resume`, `current_shard_idx` (trainer consume position) is overwritten by `producer.current_shard_idx` (production position, which runs ahead under backpressure). Shards in `[consume, produce)` are never trained; having no `.done` sentinel they are also never deleted — permanently occupying backpressure slots (starvation → spurious 600 s timeouts) and leaking disk.
- **Impact:** every resume silently discards 0–10 shards (~up to 50M tokens ≈ 6% of the default data budget) and can spiral into producer timeouts.
- **Remediation:** persist the trainer's consume index as the authoritative resume cursor and have the producer delete/truncate bins below it.
- **Verified:** orchestrator source check of both checkpoint payloads and the overwrite line.

**H4. Trainer can memmap a half-written shard (existence-only readiness gate, non-atomic shard write)** *(found independently by two auditors)*
- **Where:** `utils/dataset.py:395` (`shard_tokens.tofile(bin_path)` directly to final path; the `.json` sidecar written after at `:396-402` is never consulted) vs `train.py:777-791` (`while not os.path.exists(bin_path)` then immediate `np.memmap`, fixing `num_sequences` from whatever bytes exist, `dataset.py:414-415`).
- **Description:** unlike model checkpoints (tmp + `os.replace`), shards have no completion barrier. A poll landing inside the write window trains on a truncated/misaligned shard silently (the alignment warning fires once per process). A fresh start on a dirty cache-dir deterministically consumes a torn leftover shard (producer thread joined with only 30 s timeout, `train.py:1024-1025`).
- **Impact:** silent data corruption on slow/network storage or after an unclean kill.
- **Remediation:** write `shard_NNNNNN.bin.tmp` → `os.replace`, or have the trainer gate on size == `tokens_per_shard*2` bytes / the `.json` sidecar.
- **Verified:** orchestrator source check on both sides of the handoff.

**H5. Native-bf16 model crashes outside autocast: mixed-dtype matmul in dual-memory read**
- **Where:** `model/hybrid/memory.py:405-410` (same defect in `_batched_memory_summarize`, `:459-476`).
- **Description:** `batched_dual_memory_read` promotes only `q`/`k` to fp32; `attn` (fp32) then einsums against `v` (still bf16): `RuntimeError: expected m1 and m2 to have the same dtype`. Works under autocast (which casts both operands), crashes for any caller running the model at native bf16 — contradicting the repo's own stated policy (`model/layers/moe.py:48-61`: ops must work at native weight dtype without relying on outer autocast).
- **Impact:** any bf16 eval harness, native-bf16 deployment, or FSDP `MixedPrecision(param_dtype=bf16)` forward dies at the first memory read; current autocast-wrapped training masks it.
- **Evidence:** runtime traceback through the public API plus minimal standalone repro.
- **Remediation:** promote `v` alongside `k` (and keep the final `out_proj` bmm dtype-consistent), or compute the whole stacked attention fp32 and cast the result once, mirroring `_attend`.
- **Verified:** orchestrator read of `memory.py:392-414` confirming promotion sites.

### MEDIUM

**M1. Batched right-padded generation mis-positions shorter rows** *(runtime-reproduced)*
- `model/hybrid/model.py:1053-1057` (shared `torch.arange` prefill positions — fine for right-pad) and `:1110-1115` (`step_position_ids = past_seen_tokens` where `past_seen_tokens` = padded prompt width).
- Rows with shorter prompts get RoPE decode positions inflated by their pad count, changing relative distances to their real tokens; pad K/V entries also consume sliding-window budget. Probe: identical prompt yields different first decoded token alone (`50`) vs batched with right-padding (`61`). Single-sequence generation is correct. No test covers mixed-length batched `generate()`.
- **Remediation:** derive per-row positions from `attention_mask.cumsum(-1)-1`; exclude pads from cache/window accounting.

**M2. Chunked path normalizes router-aux/z (and all eight aux losses) by valid tokens while summing over all chunk positions**
- `model/hybrid/model.py:868-869` (`total_aux += aux_loss * chunk_len`) vs `:889-904` (`/ max(token_weight, 1)` where `token_weight` counts non-ignored labels only).
- Under padding, effective `router_aux_loss_coef`/`router_z_loss_coef`/every `λᵢ` silently rescale with pad fraction, and logged aux metrics become path-dependent (probe: chunked 1.3353 vs unchunked 1.9990 on identical padded input, CE identical). Production long-context training (seq_len 1024 > chunk 512) hits exactly this regime.
- **Remediation:** divide by `Σ chunk_len` (matching the unchunked mean-of-means) or weight each chunk by its own valid count consistently.

**M3. `--amp-dtype fp16` offered with no GradScaler**
- `train.py:637,829-839,865` — forward+backward with zero scaler anywhere (grep confirms; cloud script has one at `scripts/test_cloud_train.py:614`). fp16 grads silently underflow to zero: training looks healthy but barely learns. Default bf16 unaffected.
- **Remediation:** add a scaler for fp16, refuse the flag, or document bf16-only.

**M4. Strict vocab-equality check ignores the documented Llama id==vocab_size quirk that the cloud script handles**
- `utils/dataset.py:55-62` (`len(tokenizer) != expected → raise`) vs `scripts/test_cloud_train.py:46-67` (`resolve_tokenizer_vocab_size`). A tokenizer reporting 32000 while emitting id 32000 passes the gate, then `validate_token_batch` (`train.py:330-335`) hard-crashes hours into a run.
- **Remediation:** use `resolve_tokenizer_vocab_size` in the producer/train.py.

**M5. `MEMORY_NAN_FIX_ID` mismatch on resume is logged, never compared**
- `train.py:578-584` interpolates `checkpoint.get("memory_nan_fix_id")` into an info line; nothing warns/errors when it differs from the current constant — defeating the ID's stated purpose (proving which guard revision produced a run).
- **Remediation:** `logger.warning` (or hard-fail behind a flag) on mismatch.

**M6. Checkpoint config payload is saved but never restored or validated on resume**
- `train.py:514,533-535` save `asdict(config)`; `load_checkpoint` (`:545-585`) reads neither payload nor `config.json`. Resume rebuilds purely from CLI + current code defaults (`:656-658`), so scalar-field drift (coefficients, chunk sizes, warmups, new knobs) silently continues training a different objective on restored weights. Shape mismatches catch some cases loudly; scalars load fine.
- **Remediation:** reconstruct config from `checkpoint["config"]` (explicit override allow-list) or diff-and-warn.

**M7. Skip-based fallback resume miscounts blank samples → duplicate docs after restart**
- `utils/dataset.py:361-369` blanks `continue` *before* `cumulative_samples += 1`, but fallback resumption does `hf_stream.skip(self.cumulative_samples)` (`:314-328`), which counts blanks. Post-restart stream position drifts backward by the number of blanks consumed. Also triggered silently whenever native `state_dict()` capture fails mid-run (`:371-380` demotes to fallback).
- **Remediation:** increment the skip counter before the blank check; keep separate doc counter for logging.

**M8. Streaming interleave order is not reproducible despite `--seed`** *(static analysis; `datasets` not installable locally)*
- `utils/dataset.py:269-271` calls `interleave_datasets(..., probabilities=...)` with no `seed=`; HF draws from unseeded entropy that `set_seed` does not control. Identical seeds see different data permutations — undermining the bitwise-resume/comparison experiments the repo exists for.
- **Remediation:** pass `seed=` through and record resolved per-stream state.

**M9. Producer failure/starvation exits with success status; diagnostics bypass the run log**
- Timeout path `break`s to normal completion (`train.py:778-797`), process exit 0 — indistinguishable from finishing. The producer thread is daemonized and reports exclusively via `print()` (`utils/dataset.py:146,165,337,...`), so root causes never reach `runs/*/train.log`.
- **Remediation:** surface producer exceptions (event/queue → raise in trainer), route prints through the logger, exit non-zero on wait-timeout.

**M10. No gradient-finiteness gate before `opt.step()`**
- Only loss finiteness is checked (`train.py:842`); finite loss with NaN/Inf grads passes `clip_grad_norm_` (returned norm `nan` merely logged, `:866-868`), poisons weights via `opt.step()` (`:870-871`), and the poison is captured by the next periodic checkpoint (`:950-963`) — after which the non-finite-loss skip path stalls learning forever on corrupted weights.
- **Remediation:** skip+log when the clip result is non-finite (mirror the existing loss-gate pattern).

**M11. Documented `--log-jsonl ""` disable flag crashes the cloud script**
- `scripts/test_cloud_train.py:516` help says "set empty to disable", but `:622` `log_path = args.log_jsonl if str(args.log_jsonl) else None` — `str(Path('')) == '.'` is truthy → `IsADirectoryError` on first append. The sibling script guards correctly (`toy_train.py:116-121`).
- **Remediation:** mirror toy_train's guard.

**M12. `torch.load(..., weights_only=False)` on checkpoints (arbitrary-code execution surface)** *(found independently by two auditors)*
- `train.py:559` — the only `torch.load` in the repo; `--ckpt-dir` is CLI-provided and checkpoints are shared artifacts. Any planted/modified `model_ckpt.pth` executes arbitrary pickle code on load.
- **Remediation:** `weights_only=True` (store RNG/numpy states as tensors/primitives) or document the trust boundary explicitly.

**M13. Mixtral control baseline cannot do correct cached decode**
- `model/mixtral/model.py:111-125` — no `past_seen_tokens` anywhere; default `position_ids = torch.arange(seq_len)` restarts RoPE at 0 each incremental call (silently wrong; the attention sink path at least raises). No `generate()` exists in the family despite README §10 claiming coverage "across both model families … through `generate()`".
- **Impact:** manual cached decoding of the control model is silently wrong, weakening ablation comparisons.
- **Remediation:** add position offset plumbing + a small `generate()` mirroring the hybrid one.

**M14. CI never runs `tests/test_toy_train_smoke.py` — the only chunked-BPTT backward gate is dark**
- `.github/workflows/ci.yaml:62` runs exactly `tests.test_model`; the smoke module (10 real chunked-BPTT training steps) is excluded, so a regression in the *gradient* path of `_forward_chunked` (forward parity is tested; backward-through-chunking is not) ships green.
- **Remediation:** `python -m unittest discover -s tests -v` (or a second invocation line).

### LOW

**L1.** Reaching `--max-steps` mid-shard still marks the shard `.done` and advances the cursor (`train.py:813-815,968-974`) — extending max-steps later silently loses the unconsumed tail (compounds H3).
**L2.** First optimizer step after resume runs at constructor LR, not the resumed schedule LR — `LambdaLR.load_state_dict` restores `_last_lr` but not `param_group['lr']`; `opt.step()` precedes `sched.step()` (`train.py:724-727,870-873`). Agent-demonstrated probe; negligible magnitude, conservative direction.
**L3.** `dl_generator` shuffle position not checkpointed (`train.py:768-769`) — intra-shard permutations differ after resume; breaks bitwise-resume experiments only.
**L4.** Falsy-zero token-id override: `cfg.bos_token_id = tokenizer.bos_token_id or cfg.bos_token_id` (`train.py:657-658`) clobbers a legitimate id-0 BOS/EOS. Latent (llama2 uses 1/2).
**L5.** AdamW-only checkpoints store the same optimizer/scheduler state under both `muon_*` and `adam_*` keys (`train.py:485-493,521-528`); cross-mode resume fails with an opaque param-group mismatch instead of a clear error.
**L6.** Validation averaging weights batch-mean aux/router terms by active-token counts (`utils/validation.py:194-200`) — `val_loss` biased under unequal padding; `val_ce_loss` remains clean.
**L7.** `blocked_scan_min_len` is a dead dispatch knob (`config.py:116` threaded through `mamba.py` but never read in a condition; README §tier table holds only because defaults coincide). Setting it is silently ignored.
**L8.** Pad-masking of raw branch outputs applied only in the dual-memory arm (`layer.py:277-284`) — with `use_dual_memory=False`, unmasked pads reach the fusion gate, so `fusion_gate` stats/loss differ between ablation arms on padded batches (residual output stays correct).
**L9.** Aux warmup schedules return full weight when `training_step`/`max_training_steps` are omitted (`losses.py:56-57,65-68`) — secondary callers/tests silently get `L_assoc`/`L_expert` always-on instead of the spec'd 5%/10% ramps.
**L10.** Docs rot cluster (each individually verified): `build_training_config` measures ≈**147.8M** trainable (excl-aux) vs docstring "~80–120M"; cloud config ≈**200.2M** vs "~150M" *(counts measured by auditor probe, not re-run locally to respect the no-production-scale-instantiation rule)*; README `pytest tests/ -v` (pytest absent; project standard is unittest); "66 unit tests" (actual 78+2 smoke); "GRU-style gated writes" (code is single-sigmoid gated EMA).
**L11.** Compiled bytecode tracked in git: `.github/scripts/__pycache__/validate_pr_description.cpython-313.pyc` (Python 3.13 bytecode in a 3.11 project).
**L12.** `.gitignore` lacks patterns for `runs/`, `*.pth`, `*.bin` shards, `data_cache/`, `.ruff_cache/` — currently clean, but one `git add .` after local training would stage multi-GB binaries.
**L13.** Non-reproducible lint gates: `.pre-commit-config.yaml` pins `rev: stable`; CI installs ruff unpinned (`ci.yaml:40`) vs `ruff>=0.6.0` floor in requirements.
**L14.** Undeclared runtime deps for the training entry point: `numpy`, `datasets`, `transformers` are imported (directly/transitively) by `train.py`/`utils/` but appear nowhere in `requirements.txt` — fresh install + `python train.py --help` fails with ImportError. (Documented in CLAUDE.md as intentional for CI, but an extras group would remove the trap.)

---

## Potential risks / suspicions (unconfirmed)

| # | Risk | Where | What would settle it |
|---|---|---|---|
| P1 | All-masked query rows in SDPA lack explicit NaN guard (memory paths guard; GQA relies on backend). torch 2.13 CPU returned zeros; older/CUDA backends historically NaN, surviving `fused*hidden_mask`. Not covered by `MEMORY_NAN_FIX_ID` | `attention.py:265-272` | Zero-length-row batch on target CUDA build; if guarded, bump `MEMORY_NAN_FIX_ID` per CLAUDE.md |
| P2 | Sink-path mask re-tags `mask[:, :K]` of the *already-windowed* mask as sinks after eviction — corresponds to evicted mid-history, not absolute sinks 0..K−1; benign for today's single-token decode + all-ones history, wrong for multi-token cached forwards with interior padding | `attention.py:147-180` + `generate():1117-1118` | Cached two-token forward post-eviction vs monolithic reference with interior pads |
| P3 | `within_window` arithmetic assumes contiguous cache columns; wrong distances for post-eviction multi-token query chunks (recent block uncompensated) | `attention.py:220-226` | Same test as P2 |
| P4 | `torch._grouped_mm` offsets built as int64; some CUDA builds require int32 — failure swallowed by broad except → permanent silent fallback to loop dispatch + per-call exception overhead | `moe.py:340-359` | Counter-probe that `_forward_grouped` isn't entered when `use_grouped_gemm=True` on target build |
| P5 | SSM γ calibrated on synthetic `randn` (seq 8) rather than embedding-scale activations — if real ‖s_t‖ quantiles sit far away, `lambda_ssm` is effectively always-on/off | `model.py:136-153` | Compare γ to the 90th percentile on one real batch |
| P6 | Calibration mutates layer attributes mid-training-forward; may graph-break under compile / be lost under FSDP wrapping | `model.py:217-224` | One compiled/FSDP step from uncalibrated state |
| P7 | Dense bool sliding mask is O(L²) bytes per layer (~16 MB @ L=4096 × layers) — tension with the "no O(L²) anywhere" contract at long context (compute stays O(L·W)) | `attention.py:236-246` | Allocator high-water at L=8192 |
| P8 | `generate()` with prompt > window_size AND sinks together untested (mask truncation + sink re-tag interplay) | `model.py:1047-1052` | Windowed+sinks generate vs visible-column reference |
| P9 | Sampling numerics untested: softmax/multinomial in logits dtype, top-k tie handling keeps extras | `model.py:1083-1091` | Seeded sampling-distribution test |
| P10 | Windows: mmap handles held by `persistent_workers` can delay shard deletion (`except OSError: pass`) worsening backpressure | `dataset.py:178-185` | Cross-process delete-while-mmapped test |
| P11 | `CUBLAS_WORKSPACE_CONFIG` set possibly after CUDA init → `--deterministic` may not fully pin cuBLAS (`warn_only=True` hides residue) | `train.py:74` | GPU A/B of two identical-seed runs |
| P12 | Native `IterableDataset.state_dict()` fidelity mid-interleave unknown; per-sample capture cost unknown | `dataset.py:290-380` | Unit test vs installed `datasets` asserting re-streamed token equality |
| P13 | Producer checkpoint serializes the whole `token_buffer` as JSON — potentially huge/slow saves | `dataset.py:137` | Profile one save late in a run |
| P14 | Older torch builds silently drop Moonshot RMS matching (`adjust_lr_fn` TypeError fallback) — shared-LR premise broken across versions; nothing records which mode a checkpoint trained under | `train.py:240-253` | Persist `meta["muon_adjust_lr_fn"]` in checkpoint and validate on resume |
| P15 | Train/val CE populations differ (packed docs give cross-document EOS→BOS targets; validation caps rows and masks ends) — small systematic offset between `ce_loss` and `val_ce_loss` even for a perfect model | `dataset.py:365-368` vs `validation.py:64-67` | Score one training shard through the validation labeler |
| P16 | Empty-prompt `generate()` dies with bare `AssertionError` (`assert out is not None`) instead of a clear error | `model.py:1078` | Guard clause |
| P17 | `eos_token_id=None` means "fall back to config" — EOS stopping cannot actually be disabled via the documented parameter | `model.py:997-999` | Sentinel value distinct from None |

---

## Verified clean (condensed)

The following were traced and hold; listed so future audits know coverage:

- **Contract: pre-shifted labels** — dataset returns `(chunk[:-1], chunk[1:])`; hybrid/mixtral CE paths never re-shift; validation rolls labels via `build_causal_labels` with correct triple masking.
- **Contract: right-padding** — Mamba identity transitions on pads (all scan tiers + unpadded fused path + decode inactive-row restore); write-buffer masks carried explicitly; vectorized conv-state gather matches reference incl. `vl<K`/`vl=0` edges; inactive rows' memory states restored post-write.
- **Contract: sub-quadratic** — memory ops O(L·m), fusion O(L·d), assoc sampling fixed-T, MoE dispatch O(N·E), scans linear/log-linear (dense-mask caveat → P7).
- **Contract: read/write decoupling + chunk alignment** — reads condition inputs, raw outputs buffered; decode/prefill/training flush cadences correct; partial decode buffer flushed before return; `skip_memory_write` accumulate→flush probed.
- **Contract: ablation hooks** — `zero_memory_states()`, `gate_stats` threading, `use_dual_memory=False` (Jamba mode trains, states all-None), `build_test3_null_baseline_config` param-ratio 0.9995.
- **Contract: NaN guards** — memory `_attend`/summarize/write all-masked handling consistent with `MEMORY_NAN_FIX_ID` (gap → P1); recon/assoc/gate/slot guarded.
- **Contract: AMP numerics** — router fp32 + ±30 clamp, scans promoted fp32, aux island disables autocast, RMSNorm fp32, `dt_proj.bias _no_reinit`, `A_log/D _no_weight_decay`.
- **Contract: fixed caches/loud limits** — RoPE raises oversize; position bound checked; generate preflights length; no runtime buffer re-registration.
- **Scan consistency** — Hillis-Steele ≡ blocked ≡ sequential (1e-5); incremental `step()` chain ≡ prefill (≈5e-8); recurrence algebra matches across tiers.
- **Aux losses** — all eight formulas match `research/loss-definitions.md`; combination order/coefficients/warmups match spec §5.
- **CE/vocab-z accounting** — four CE variants are valid-token means; z-loss gated on coef, valid-rows only, correctly weighted in chunked path (aux normalization gap → M2).
- **MoE** — top-2 softmax weights, dropless combine, Switch aux with full softmax, z-loss formula, capacity renorm clamp — grouped dispatch proven equal to loop by existing test.
- **Optimizer routing** — metadata-first split, tied-weight dedup, decay/no-decay subgroups (matches `model/core/optim.py` contract).
- **Checkpoint atomicity (model)** — tmp+`os.replace`; torn tmp never loaded; KeyboardInterrupt preserves previous good checkpoint; RNG/python/numpy/torch/cuda + validator cursor persisted.
- **Serialization surface** — single `torch.load`; no pickle/subprocess/os.system/`trust_remote_code`/raw requests in source; no secrets/binaries tracked (except L11).
- **Non-contamination** — memory/Mamba/write-buffer state is call-threaded and freshly cloned from learned init when absent; only persistent buffers are the ssm γ tables; no cross-batch leakage in training or validation.
- **Non-finite-loss skip leaves gradients clean** (zero_grad precedes every forward).
- **API surface** — `verify_model_package.py` PASS (17 modules / 11 symbols, non-vacuous).
- **Falsification hooks present** per CLAUDE.md (Test 1/2/3 infrastructure).

---

## Test coverage gaps (ranked)

1. **CUDA-graph decode parity** — test exists but `skipTest`s on CPU-only CI; combined with H2 this is the most dangerous dark path.
2. **Batched mixed-length/right-padded generation** — none; M1 lives here.
3. **Checkpoint save/load/resume roundtrip** — zero unit tests (this audit covered it once, manually, via the E2E fixture); M2/M5/M6/L2/L3 all live here.
4. **Data pipeline** — nothing imports `utils.dataset` in tests; producer BOS/EOS wrapping, uint16 packing, checkpoint roundtrip, `MmapShardDataset` alignment untested.
5. **AMP fp32-promotion points under actual autocast** — the promotions exist for a dtype environment no test creates.
6. **Chunked-BPTT gradient equivalence** (and CI doesn't even run the smoke module — M14).
7. **Sampling parameters** (`do_sample=True` never tested).
8. **Mixtral functional coverage** — import-level only; M13 would be caught instantly by a decode-parity test.
9. **`build_causal_labels`** — duplicated 3× across files, zero tests pinning semantics.
10. **MoE edge shapes** (`num_experts=1`, capacity overflow-renorm), **memory-flush boundary conditions**, **validation cursor wraparound/mismatch reset**, **cloud-script pure helpers**.

---

## Tests performed in this audit

**Environment:** Windows 10, Python 3.13.14, torch 2.13.0+cu126 (CPU-only; NVIDIA driver too old for this build), numpy present, `datasets`/`transformers`/`mamba_ssm` absent.

1. **Static audit** — three parallel read-only auditors over `model/`, `train.py`, `utils/`, `scripts/`, tests, CI, deps, docs; every recorded finding re-verified against source lines by the orchestrator; duplicates merged into single entries; no auditor claim was dropped — all survived verification (some with severity calibration or scope caveats noted inline).
2. **Unit suite:** `python -m unittest tests.test_model` → **Ran 78 tests, OK** (8 skipped — all pre-existing CUDA/fused-scan skips).
3. **Package check:** `scripts/verify_model_package.py` → **PASS** (17 modules, 11 public symbols).
4. **End-to-end runtime validation** (temporary fixture `.claude/tmp_audit/e2e_validate.py`, gitignored; **no repository files created or modified**):
   - **Config:** laptop-safe toy hybrid — **5,114,880 trainable params**, **batch_size=2**, seq_len 128, CPU fp32, 10 total steps across 2 synthetic packed-uint16 shards (48 sequences each, BOS/EOS sentinels).
   - Real production functions exercised by importing `train.py` with in-process stubs only for the missing optional deps (`datasets`/`transformers` — never instantiated): `set_seed`, `build_optimizers` (real Muon+AdamW split: 95.3%/4.65%, wd=0.1 on 22 params / wd=0 on 71), `_resolve_warmup_steps`/`_build_lr_lambda`, `validate_token_batch`, `save_checkpoint`/`load_checkpoint`.
   - **Result: 26/26 checks pass** — shard packing; param budget; optimizer split; warmup start LR; 8-step training phase (finite loss/grads, pre-shifted-labels honored); LR progression (1e-3 → 2.05e-4 along cosine); checkpoint file + payload keys; `config.json` roundtrip incl. newest knobs; resume counters (step=8, shard=1); **weights bit-identical after roundtrip (0 mismatches across the whole state_dict)**; **optimizer state structurally restored** (Muon momentum + AdamW moments/steps); scheduler positions restored ([8, 8]); **torch RNG state restored bit-exact**; resumed training progressed to step 10; metrics JSONL parses line-complete (10 records, finite losses); greedy generation shape/determinism/valid-id range; sampled generation with temperature/top-k/top-p; batched generation; cache-bound overrun raises loudly.
   - One check is a **deliberate bug reproduction** (H1): as-trained config crashes `generate()` — confirming the finding, then flipping the flag at runtime (fixture-only) to validate the rest of the inference machinery.

**Limitations / checks not performable here:**
- **CPU-only:** no CUDA execution — H2 established by source-level proof, not GPU replay; fused mamba-ssm kernels, FP16 GradScaler path (cloud script), FSDP/torch.compile behaviors untested.
- **No `datasets`/`transformers`:** producer streaming, native-state resume fidelity, wikitext validator, and the entire cloud script are static-analysis conclusions only (flagged where relevant: M8, P12).
- **Production-scale configs never instantiated** per local-dev rules; the corrected parameter counts in L10 are auditor probe measurements, not re-verified.
- DataLoader `num_workers>0` path, wall-clock performance, and dynamic security testing not exercised.

---

## Resolution log (2026-08-25 remediation pass)

All work done as **uncommitted working-tree edits on `vsenapathi/test-ox-alpha`**. Laptop-safe configs only (`build_toy_config()` ~5M / `_small_hybrid_config()` ~10M); production defaults untouched except where a finding itself required changing production code paths. ✅ = fixed in code · 📄 = documented/deliberate · ⏸ = open (needs hardware or uninstalled deps).

### HIGH — all fixed ✅

| # | Disposition |
|---|---|
| H1 | `generate()` now forces `config.return_logits=True` for its duration (restored in `finally` alongside `self.train(was_training)`), so as-trained checkpoints work. Regression test `test_generate_return_logits_false_works`; E2E H1 repro converted into a fix-verification check. |
| H2 | `_CudaDecodeGraphRunner` **deleted**; `generate()` decodes eagerly. `use_cuda_graph` kept on the config as a deprecated no-op (old checkpoints/configs still load) with a warn-once; README no longer advertises the fast path. |
| H3 | Producer-write-cursor overwrite removed; the trainer's consume cursor is the authoritative resume position (NOTE comment marks the contract). Producer checkpoint/cleanup operate strictly below it. |
| H4 | Shard writes are atomic: `shard_NNNNNN.bin.tmp` → fsync → `os.replace`; `.json` sidecar published last with `num_tokens` as its final field; trainer waits for **both** bin and sidecar before memmap. |
| H5 | `batched_dual_memory_read` / `_batched_memory_summarize` promote `v` (and gate/update weight math) to fp32 under low precision and restore activation dtype before the stacked `out_proj` bmm — native-bf16 forward works outside autocast. |

### MEDIUM — all fixed ✅

| # | Disposition |
|---|---|
| M1 | Per-row RoPE positions: prefill `(mask.cumsum(-1)-1).clamp(min=0)`, decode `valid_count−1`. Regression test pins exact prefill/decode positions for a mixed-length right-padded batch. |
| M2 | Chunked path normalizes router-aux/router-z/all-eight aux losses by `Σ chunk_len` (`total_chunk_len` accumulator); CE/vocab-z remain valid-token-weighted. Path-dependence probe now agrees between chunked/unchunked. |
| M3 | `torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype==fp16))` wired scale→backward→`unscale_`→clip→`scaler.step`→`update`; bf16/fp32 keep the pass-through scaler. |
| M4 | `resolve_tokenizer_vocab_size()` probes real encodes + special-token ids and returns `max(reported, max_id+1)`; `verify_tokenizer_vocab` raises only when the *emitted* range exceeds `vocab_size`. |
| M5 | `load_checkpoint` warns on `memory_nan_fix_id` mismatch against the current constant. |
| M6 | `load_checkpoint` diffs the saved config payload vs the live one and warns per drifted scalar field. |
| M7 | `cumulative_samples += 1` moved **before** the blank-sample check; separate doc counter for logging — skip-based fallback resume no longer drifts. |
| M8 | `seed=` passed to `interleave_datasets` (older `datasets` versions raising `TypeError` fall back with a warning). |
| M9 | Producer starvation timeout raises `RuntimeError` (non-zero exit); trainer polls `producer.error` and re-raises; producer logs route through `log_fn` into the run log instead of bare `print`. |
| M10 | Non-finite clip-norm result gates `opt.step()` (skip + JSONL `"non_finite_grad"` event), mirroring the loss-gate pattern. |
| M11 | Cloud script mirrors toy_train's guard: `log_jsonl` equal to `""`/`"."` disables JSONL instead of crashing. |
| M12 | Checkpoint loads try `weights_only=True` first (broad-except fallback to `False` with an explicit trust-boundary warning). RNG states serialized pickle-free: python RNG raw tuple, numpy MT state as dict of int64 lists, torch RNG as uint8 tensors. |
| M13 | Mixtral baseline gained `past_seen_tokens` plumbing (default arange offset, loud bound raise) and a full `generate()` mirroring hybrid semantics. |
| M14 | CI now also runs `python -m unittest tests.test_toy_train_smoke -v`. |

### LOW — 13 fixed ✅, 1 documented 📄

| # | Disposition |
|---|---|
| L1 | `shard_fully_consumed` flag: `.done`/cursor advance only when the shard was fully consumed (extends H3). |
| L2 | Post-resume LR sync: `group["lr"] = group["initial_lr"] * lr_lambda(max(sched.last_epoch,0))` so the first step after resume uses schedule LR. |
| L3 | `dl_generator` state persisted via `extra_payload` and restored on resume (bitwise-resume experiments preserved). |
| L4 | Token-id overrides use `is not None` instead of truthiness (id-0 BOS/EOS respected). |
| L5 | New checkpoint schema: `muon_*` keys stored only when Muon active, `adam_*` always, explicit `use_muon` flag; cross-mode resume raises a clear `RuntimeError`; legacy checkpoints still resume via key-presence inference. |
| L6 | Validation accumulates router-aux/z **unweighted** and divides by `max(batch_count, 1)`; `val_loss`/`val_ce_loss` stay token-weighted. |
| L7 | Scan dispatch computes `parallel_limit = max(parallel_scan_fallback_max_len, blocked_scan_min_len − 1)` — `blocked_scan_min_len` is now a live knob. |
| L8 | Raw branch outputs are pad-masked **before fusion in both arms**, so `fusion_gate` stats are comparable across the dual-memory/Jamba ablation on padded batches. |
| L9 | 📄 Left as-is deliberately: secondary callers/tests intentionally get always-on aux weights; docstrings note it and unit tests pin the default schedule values. Changing it would alter pinned test expectations for no training-path benefit (train.py always passes both steps). |
| L10 | Docs rot fixed: measured param counts (~148M train / ~200M cloud), unittest commands replace pytest, 83-test count, "single-sigmoid gated EMA" wording, CUDA-graph removal notes across README/model-README/research. |
| L11 | Tracked bytecode removed via `git rm --cached`; `__pycache__/` ignored. |
| L12 | `.gitignore` expanded (`runs/`, `data_cache/`, `model_ckpt/`, `*.pth(.tmp)`, shard bins/tmp/`.done`, `*.jsonl`, `.ruff_cache/`) after verifying zero tracked-file collisions. |
| L13 | `ruff==0.16.2` pinned identically in requirements.txt, CI, and `.pre-commit-config.yaml` (`rev: v0.16.2`). |
| L14 | requirements.txt declares `numpy>=1.24`, `datasets>=2.16`, `transformers>=4.38` under a "Training entry points" section (CI installs torch only, unchanged). |

### Potential risks — 12 addressed ✅, 5 open ⏸

| # | Disposition |
|---|---|
| P1 | ✅ SDPA all-masked-row guard added; `MEMORY_NAN_FIX_ID` bumped to `-v3` per CLAUDE.md rule. |
| P2/P3 | ✅ `generate()` feeds **full-prefix** masks into cached forwards (layer truncates window/sinks consistently); new parity regression test drives a chunked 16+8 cached forward vs monolithic reference with interior pads, sinks=2, window=16 — cosine > 0.99. |
| P4 | ✅ Grouped-GEMM offsets built as int32 on CUDA; unsupported builds get a warn-once permanent fallback instead of per-call swallowed exceptions. |
| P5 | ⏸ Needs one real-batch ‖s_t‖ percentile measurement (GPU run); blind recalibration rejected. |
| P6 | ⏸ Needs a compiled/FSDP step from uncalibrated state (no CUDA locally); code paths left untouched. |
| P7 | ⏸ Needs allocator high-water profiling at L=8192; compute is already sub-quadratic, this is a memory-measurement question. |
| P8 | ✅ Covered by the same windowed+sinks+pads parity test as P2/P3 (prompt longer than window, interior padding, sink retention). |
| P9 | ✅ Sampling promoted to fp32 (`logits.float()/temperature`, argmax on float); `do_sample=True` now exercised end-to-end — which immediately caught a real bug (missing `import torch` in `model/layers/sampling.py`, since fixed). |
| P10 | ✅ Shard cleanup retries 3× with backoff while keeping the `.done` sentinel, then sweeps stale `.bin.tmp` files. |
| P11 | ◐ `CUBLAS_WORKSPACE_CONFIG` set at the top of `main()` before any CUDA init; full GPU A/B determinism proof still needs hardware. |
| P12 | ⏸ Requires `datasets` installed to assert native-state resume fidelity; static analysis stands. |
| P13 | ✅ Producer checkpoint stores `token_buffer` as base64-packed uint16 bytes with a length header (compact + corruption-detecting) instead of a JSON number list. |
| P14 | ✅ Every checkpoint records `muon_adjust_lr_fn` in its payload, so the shared-LR premise a run trained under is recoverable. |
| P15 | ⏸ Train/val CE-population offset is a measurement task (score one training shard through the validation labeler); no code change justified yet. |
| P16 | ✅ Empty-prompt `generate()` raises a descriptive `ValueError`; regression test added. |
| P17 | ✅ Negative `eos_token_id` is a sentinel disabling EOS stopping (documented); regression test asserts generation continues to full length. |

### Verification after remediation

1. `py_compile` over all 14 touched modules: clean.
2. Unit suite: **83/83 OK** (8 pre-existing CUDA skips) — includes **5 new regression tests** (H1, P16, P17, M1, P2/P3).
3. Smoke module (`tests.test_toy_train_smoke`): 2/2 OK; `scripts/verify_model_package.py`: PASS.
4. E2E pipeline fixture (toy ~5M, CPU): **27/27 checks pass**, including bit-exact weight/optimizer/RNG roundtrip, resume continuation, and the H1-fix verification; `weights_only=True` load verified (M12).
5. `ruff check .` → **All checks passed!**; `ruff format .` applied to the 3 stragglers → `40 files already formatted`; unit suite re-run green after formatting.
6. Production-scale configs were never instantiated (local-dev rules honored throughout).
