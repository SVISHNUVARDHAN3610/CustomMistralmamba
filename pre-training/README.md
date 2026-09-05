# Pre-Training Directory (`pre-training`)

The `pre-training` directory contains the multi-GPU distributed pre-training engine for the **Hybrid Mamba–MoE with Dual Compressive Memory** architecture, powered by **PyTorch Fully Sharded Data Parallel 2 (FSDP2)** and the **Muon + AdamW hybrid optimizer**.

---

## Architecture & Distributed Design

The primary training entry point is [`fsdp2_train.py`](file:///d:/Working_Repo/CustomMistralmamba/pre-training/fsdp2_train.py). It scales the hybrid model across multi-GPU and multi-node clusters while adhering to strict mathematical and numerical contracts:

### 1. FSDP2 Bottom-Up Parameter Sharding
* Submodules (individual `HybridBlock` / `TransformerBlock` layers) are wrapped via `torch.distributed.fsdp.fully_shard` **before** the root model is wrapped.
* Non-matrix parameters, biases, norms, gains, and deliberately replicated custom-math parameters (such as selective scan $A, B, C$ projections and memory compression projections) are managed with custom exclusions:
  * Replicated parameters are registered via [`_prepare_fsdp2_custom_math_params`](file:///d:/Working_Repo/CustomMistralmamba/pre-training/fsdp2_train.py#L380) and their gradients are explicitly synchronized via [`_sync_replicated_param_grads`](file:///d:/Working_Repo/CustomMistralmamba/pre-training/fsdp2_train.py#L420) across all ranks.
  * 2D hidden weight matrices are fully sharded across ranks as `DTensor`s.

### 2. Mixed Precision Policy
* **FP32 Master Parameters**: All model parameters and optimizer state are maintained in `float32`.
* **Forward Autocast**: Mixed precision is applied via `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` around forward blocks.
* **Auxiliary Loss Precision**: Critical memory bank read/write auxiliary losses, router logits, and SSM state calculations are promoted to `float32` activations, avoiding the NaN and underflow pitfalls common with naive bfloat16 parameter casting.

### 3. Distributed Muon + AdamW Optimization
* [`build_fsdp2_optimizers`](file:///d:/Working_Repo/CustomMistralmamba/pre-training/fsdp2_train.py#L650) splits model parameters into two optimizer groups:
  1. **`MuonDTensor`**: Handles 2D hidden linear weights. Gathers full momentum matrices across ranks into bounded buffers (`--muon-gather-buffer-mb`), computes 5th-order Newton–Schulz orthogonalization, and redistributes sharded updates.
  2. **`AdamW`**: Handles 1D parameters, embeddings, LM head, biases, normalization gains, and replicated custom-math tensors.

### 4. Consolidated Checkpoints & Resumption
* Checkpoints save full model weights and sharded optimizer states using consolidated DTensor state dict options:
  * Model weights are saved in a clean FP32 PyTorch state dict compatible with downstream single-GPU SFT or inference.
  * Checkpoints persist the intra-shard batch cursor, seeded sampler state, and per-rank RNG payloads for deterministic resumption.

---

## Launch Commands

### Single-Node Multi-GPU Training (e.g. 8 GPUs)
```bash
torchrun --standalone --nproc_per_node=8 pre-training/fsdp2_train.py \
    --cache-dir ./data_cache \
    --run-dir ./runs/pretrain_fsdp2 \
    --batch-size 2 \
    --grad-accum-steps 4 \
    --lr 2e-4 \
    --seq-len 4096 \
    --max-steps 50000 \
    --save-interval 500 \
    --log-interval 10
```

### Multi-Node Cluster Launch
On multi-node clusters, launch via `torchrun` with cluster rendezvous arguments:
```bash
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    pre-training/fsdp2_train.py \
    --cache-dir /shared/data_cache \
    --run-dir /shared/runs/pretrain_fsdp2 \
    --batch-size 2 \
    --grad-accum-steps 4 \
    --lr 2e-4
```

### Resuming Distributed Pretraining
Resuming requires matching world size, seed, and model architecture:
```bash
torchrun --standalone --nproc_per_node=8 pre-training/fsdp2_train.py \
    --resume ./runs/pretrain_fsdp2/model_ckpt.pth \
    --cache-dir ./data_cache \
    --run-dir ./runs/pretrain_fsdp2
```

---

## Key CLI Arguments

* `--nproc_per_node`: Number of GPUs per node (matches available GPUs).
* `--cache-dir`: Shared directory where `TokenizedShardProducer` publishes `.bin` and `.json` shards.
* `--run-dir`: Output directory for checkpoints, metrics, and logs.
* `--batch-size`: Microbatch size **per rank**.  
  $$\text{Effective Batch Size} = \text{batch\_size} \times \text{world\_size} \times \text{grad\_accum\_steps}$$
* `--gradient-checkpointing`: Enabled by default (`--no-gradient-checkpointing` to disable).
* `--dist-backend`: `nccl` (default on Linux CUDA) or `gloo`.
