# Distributed Pre-Training Directory (`pre-training/`)

This directory contains distributed pre-training implementations for the **Hybrid Mamba–MoE with Dual Compressive Memory** language model architecture. Each trainer adapts the model's forward and loss semantics, auxiliary losses, tokenized shard data pipeline, checkpointing, and validation for a specific distributed accelerator environment.

---

## Available Trainers

| Script | Target Hardware | Execution Paradigm | Sharding Mechanism | Optimizer |
| :--- | :--- | :--- | :--- | :--- |
| [`tpu_smpd_train.py`](file:///D:/Working_Repo/CustomMistralmamba/pre-training/tpu_smpd_train.py) | **Cloud TPU** (e.g., Kaggle TPU v5e-8) | **Single-process SPMD** via PyTorch/XLA | GSPMD `Mesh` + `mark_sharding` (Data Parallel or FSDP) | **Muon + AdamW** hybrid (or AdamW-only) |
| [`fsdp2_train.py`](file:///D:/Working_Repo/CustomMistralmamba/pre-training/fsdp2_train.py) | **Multi-GPU Clusters** (e.g., 8× H100/A100) | **Process-per-GPU** via `torchrun` | PyTorch FSDP2 (`torch.distributed.fsdp.fully_shard`) | **AdamW** (decoupled weight decay) |

---

## 1. TPU SPMD Trainer (`tpu_smpd_train.py`)

Optimized specifically for Cloud TPU environments, including **Kaggle TPU v5e-8** and Google Cloud TPU v5e pod slices.

### Architecture & Key Mechanisms

1. **Single-Process SPMD Model:**
   * Unlike multi-process DDP or FSDP where separate Python processes run per accelerator rank, PyTorch/XLA SPMD runs as a **single Python process** controlling the entire TPU mesh.
   * `xr.use_spmd()` enables GSPMD compiler transformations before any tensors or devices are allocated.
   * Device count is dynamically detected via `xr.global_runtime_device_count()` (supports TPU v5e-8, v5e-4, v5e-1, etc.).

2. **Mesh & Sharding Strategy:**
   * A logical device mesh is established across all TPU cores along a `'data'` axis.
   * **Input Data Sharding:** Incoming input batches `(input_ids, labels)` of shape `(batch_size, seq_len)` are annotated with `xs.mark_sharding(tensor, mesh, ('data', None))`. Each TPU core processes `batch_size // num_devices` sequences per microstep.
   * **Sharding Strategies (`--sharding-strategy`):**
     * `data_parallel` *(Default)*: Model weights remain replicated across TPU cores. The XLA compiler automatically handles local forward computation and AllReduces gradients across cores during backward pass. Recommended for models up to ~511M on TPU v5e-8 (128 GB aggregate HBM).
     * `fsdp`: 2D weight matrices (attention projections, Mamba projections, MoE expert layers, LM head) are sharded along dimension 0 (`('data', None)`). Custom direct-math parameters (RMSNorm gains, Mamba `A_log`/`D` vectors, dual-memory combine projections, `CompressiveMemoryBank` buffers) remain replicated to ensure DTensor stability.

3. **TPU Precision:**
   * Native `bfloat16` execution via `torch.autocast(device_type="xla", dtype=torch.bfloat16)`.
   * No CUDA `GradScaler` or loss scaling is used (bfloat16 matches the dynamic range of fp32 and must not be artificially scaled).

4. **Mamba Selective Scan on TPU:**
   * Fused CUDA kernels (`mamba-ssm`) are disabled (`cfg.use_fused_mamba_scan = False`).
   * The model automatically uses PyTorch's native Hillis-Steele parallel scan or blocked vectorized scan, which compiles directly to TPU XLA HLO.

5. **Optimizers & Schedulers:**
   * **Muon + AdamW Hybrid:** 2D matrices are optimized with `torch.optim.Muon` (Newton-Schulz iterations compiled on TPU), and non-2D / embeddings / norms / head are optimized with `torch.optim.AdamW` (`fused=False`).
   * Decoupled cosine learning rate schedule with linear warmup.
   * On-device gradient clipping via `clip_grad_norm_`.
   * Host-sync-free NaN handling via `--grad-nan-guard sanitize` (`torch.nan_to_num_` in-place on device).

6. **XLA Step Boundaries & Host-Sync-Free Metrics:**
   * `xm.mark_step()` is invoked strictly after the optimizer and scheduler update steps, allowing XLA to fuse the forward, backward, accumulation, and optimizer graphs.
   * Metric scalars are accumulated as on-device tensors during training. A single device-to-host transfer occurs every `--log-interval` steps, preventing TPU pipeline stalls.

7. **Atomic Checkpointing & True Resume:**
   * Serialized with `xm.save(payload, tmp_path)` to ensure XLA tensors are cleanly transferred to CPU before writing, followed by atomic file replacement (`model_ckpt.pth` + `config.json`).
   * Captures full model state, optimizer states, scheduler states, global step, shard progress, RNG states (Python, NumPy, PyTorch CPU, XLA RNG), and runtime contract metadata.

### Example Launch on Kaggle TPU v5e-8

```bash
python pre-training/tpu_smpd_train.py \
    --cache-dir ./data_cache \
    --run-dir ./runs/tpu_train \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --seq-len 1024 \
    --lr 1e-3 \
    --adam-lr 3e-4 \
    --amp-dtype bf16 \
    --sharding-strategy data_parallel \
    --log-interval 10 \
    --save-interval 1000
```

---

## 2. Multi-GPU FSDP2 Trainer (`fsdp2_train.py`)

Multi-GPU port of the root `train.py` using PyTorch FSDP2 (`torch.distributed.fsdp.fully_shard`) across NVIDIA GPUs.

### Architecture & Key Mechanisms

1. **Process-per-GPU Architecture:**
   * Launched via `torchrun --nproc_per_node=N`.
   * Communicates across ranks via NCCL collective operations.
   * Uses `DistributedSampler(shuffle=True, seed, drop_last=True)` to partition shard sequences across ranks.

2. **FSDP2 Wrapping (`fully_shard`):**
   * Model decoder layers are wrapped inner-to-outer with `fully_shard`.
   * Master weights remain in FP32 while forward computations run under `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`.
   * Custom math parameters (RMSNorm, Mamba state vectors, dual-memory combine, memory banks) are explicitly ignored by `fully_shard` via `_prepare_fsdp2_custom_math_params` and synchronized across ranks.

3. **Optimizer Policy:**
   * Uses AdamW across all parameter groups. (Muon is excluded under multi-GPU FSDP2 to avoid expensive full-matrix all-gathers on every step).

4. **Intra-Shard Resumption:**
   * Persists an intra-shard batch cursor so resumes replay at the exact next per-rank batch for the same world size.

### Example Launch with torchrun

```bash
torchrun --nproc_per_node=8 pre-training/fsdp2_train.py \
    --cache-dir ./data_cache \
    --run-dir ./runs/fsdp2_train \
    --batch-size 4 \
    --grad-accum-steps 2 \
    --seq-len 1024 \
    --lr 3e-4 \
    --log-interval 10 \
    --save-interval 1000
```

---

## Common Features Across Both Trainers

* **Data Ingestion:** Both scripts read packed binary shards (`shard_NNNNNN.bin` + `shard_NNNNNN.json`) produced by `utils.dataset.TokenizedShardProducer` using `utils.dataset.MmapShardDataset`.
* **Objective:** Standard Causal Language Modeling cross-entropy loss plus router auxiliary loss, router z-loss, and the 8 memory/SSM auxiliary losses (recon, assoc, assoc_norm, gate, read, fusion, expert, ssm, slot).
* **Validation:** Both trainers support cyclic evaluation on Salesforce/wikitext (`WikiTextCyclicValidator`) and continuous stream packed-window evaluation (`PackedWindowValidator`).
* **Memory NaN Fix Tracking:** Both serialize and verify `MEMORY_NAN_FIX_ID` across checkpoints to prevent behavior drift.

---

## Local Development Rules

> [!IMPORTANT]
> **This repository follows strict development environment rules:**
> * Local laptops and development machines are **development-only**.
> * **Do NOT execute `tpu_smpd_train.py` or `fsdp2_train.py` on local machines.**
> * Local validation should be limited to static analysis, syntax checking (`python -m py_compile`), linting (`ruff check`), formatting (`ruff format`), and lightweight hardware-independent unit tests (`tests/test_toy_train_smoke.py`, `scripts/verify_model_package.py`).
> * Full production training runs take place in the target cloud environments: Kaggle TPU v5e-8 for `tpu_smpd_train.py` and cloud GPU clusters for `fsdp2_train.py`.
