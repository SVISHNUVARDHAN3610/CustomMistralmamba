# Improvement Suggestions

Deferred research-grade ideas from the `model.py` code review.
**Do not treat these as open bugs** — correctness fixes live in `model.py`;
this file is a backlog for pushing the project toward paper-quality evidence
and wall-clock efficiency.

Work these later, after Critical/High/Medium review fixes are in.

---

## A. Make memory scientifically trustworthy

1. **Write auxiliary loss (optional)** — e.g. predict next-chunk bag-of-tokens /
   retrieval probe from memory so the write path has a direct objective if CE
   alone under-trains gates.
2. **Gate regularizers** — entropy / saturation penalty so Test 2 isn’t just
   “mean ≈ 0.5.” Log histograms, not only means.
3. **Content-addressable slots** — slot keys separate from values; optional
   LRU / usage counters (Infini-attention / Memorizing Transformer–style) if
   pure GRU blend saturates.
4. **Detach policy ablations** — compare full BPTT vs truncated BPTT (detach
   every K chunks) for stability at long L.

## B. Falsification suite that can kill the idea (`research.md` §6)

5. **Needle / rare-fact harness** — fixed seeds, controlled distance, report
   exact-match + calibrated confidence; memory-on vs `zero_memory_states` vs
   null baseline.
6. **Matched FLOPs, not only params** — Test 3 today matches parameters; also
   match train tokens and peak activation memory.
7. **Jamba-like control** — GQA+Mamba+MoE without dual banks
   (`use_dual_memory=False`) as primary peer, not only Mixtral.
8. **Perturbation tests** — shuffle memory slots / freeze write / freeze read;
   show which path carries the gain.
9. **Scale ladder** — 10M → 100M → 1B with the same eval; don’t claim
   long-context wins from 10M smoke only.

## C. Efficiency (honest long-context claims)

10. **Fused selective scan** — `mamba_ssm` / Triton kernel; keep pure PyTorch
    as fallback. Without this, “linear-time” is FLOP-true, wall-clock-false
    at 100K.
11. **Chunked parallel scan + checkpoint** — middle ground if CUDA isn’t ready.
12. **Grouped MoE dispatch** — sort tokens by expert, one grouped GEMM; drop
    Python `for e in experts` (current loop is correct but a throughput cliff).
13. **Shared RoPE cache across layers** — today every `SlidingWindowGQA` owns
    full cos/sin buffers.
14. **Flash / SDPA hygiene** — ensure mask dtype/layout hits efficient kernels;
    consider `is_causal` + window where possible.
15. **Decode packing** — compact active sequences; don’t step finished rows
    through the model at all (stronger than masking).

## D. Training stack

16. **Activation checkpointing default on for long L.**
17. **Mixed precision recipe** — keep scan/router in fp32; rest bf16; document it.
18. **FSDP / activation memory profile** at target L before claiming feasibility.
19. **LR / WD exclusions** — honor `A_log` / `D` `_no_weight_decay` in the real
    optimizer (flags exist in the model; wiring must live in training code).

## E. Architecture peaks (if §6 passes)

20. **Layer pattern ablations** — attention every N layers vs every layer.
21. **Memory only on attention branch or only on Mamba** — 2× banks may be
    redundant; ablate.
22. **Cross-layer shared memory** — one bank per stack vs per layer.
23. **Fusion alternatives** — learned scalar per layer, softmax over
    {attn, mamba, memory-read}, or low-rank gates to cut `2d²`.
24. **Depth-width trade under fixed params** — memory may help more with fewer
    experts / narrower FF.

## F. Reporting bar for a paper

25. Publish **wall-clock vs L** and **peak VRAM vs L** next to FLOP curves.
26. Report **gate trajectories** and **memory slot usage** over training.
27. Release config + seeds + exact needle protocol so the dual-memory claim is
    falsifiable by others.

---

## Suggested next three (when returning to this file)

1. §6 needle harness + param/FLOP-matched null baseline runs.
2. Fused selective scan (or chunked parallel scan) for honest long-L timing.
3. Grouped MoE dispatch for training throughput.
