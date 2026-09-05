# Scripts Directory (`scripts`)

The `scripts` directory contains tools for verification, benchmarking, cloud smoke testing, and local development of the **Hybrid Mamba–MoE with Dual Compressive Memory** architecture.

---

## Script Index

| Script | Category | Purpose | Typical Execution |
| :--- | :--- | :--- | :--- |
| [`toy_train.py`](file:///d:/Working_Repo/CustomMistralmamba/scripts/toy_train.py) | Development / Smoke Test | Laptop-safe CPU training smoke test (~5M parameters) without external dataset downloads | `python scripts/toy_train.py --steps 20 --batch-size 2` |
| [`test_cloud_train.py`](file:///d:/Working_Repo/CustomMistralmamba/scripts/test_cloud_train.py) | Cloud GPU Verification | Cloud GPU smoke test (~200M config) validating CUDA kernels, autocast, selective scan & AMP | `python scripts/test_cloud_train.py --steps 10` |
| [`verify_model_package.py`](file:///d:/Working_Repo/CustomMistralmamba/scripts/verify_model_package.py) | Packaging & API Integrity | Verifies the public API surface in `model/__init__.py` and enforces clean imports | `python scripts/verify_model_package.py` |
| [`bench_grad_guard.py`](file:///d:/Working_Repo/CustomMistralmamba/scripts/bench_grad_guard.py) | Benchmarking | Benchmarks gradient clipping and NaN guards under mixed precision (BF16/FP32) | `python scripts/bench_grad_guard.py` |
| [`check_gradient_checkpointing.py`](file:///d:/Working_Repo/CustomMistralmamba/scripts/check_gradient_checkpointing.py) | Memory Profiling | Verifies non-reentrant gradient checkpointing memory reduction and numerical parity | `python scripts/check_gradient_checkpointing.py` |

---

## Detailed Script Guides

### 1. [`toy_train.py`](file:///d:/Working_Repo/CustomMistralmamba/scripts/toy_train.py)
* **Goal**: Provides a lightweight end-to-end training loop that runs safely on a developer's CPU laptop in seconds.
* **Architecture**: Instantiates a scaled-down 5M-parameter `HybridForCausalLM` model (2 layers, 128 hidden dim, 2 heads, 2 experts) and runs forward/backward passes on synthetic data.
* **Verifications**: Checks that cross-entropy loss, all 8 auxiliary regularizers, gradient clipping, AdamW/Muon optimizer steps, and cosine learning rate schedules function together without CUDA dependencies.
* **Usage**:
  ```bash
  python scripts/toy_train.py --steps 20 --batch-size 2 --device cpu
  ```

### 2. [`test_cloud_train.py`](file:///d:/Working_Repo/CustomMistralmamba/scripts/test_cloud_train.py)
* **Goal**: Pre-flight verification script for cloud GPU environments (e.g., A100/H100/L4).
* **Architecture**: Builds a 200M-parameter config with full context window and tests:
  1. Fused Mamba scan (`mamba_ssm.ops.selective_scan_fn`) vs. parallel scan fallbacks.
  2. Bfloat16 autocast forward with FP32 promotional casting in auxiliary loss modules.
  3. Memory consumption with non-reentrant gradient checkpointing enabled.
* **Usage**:
  ```bash
  python scripts/test_cloud_train.py --steps 10 --batch-size 4 --amp
  ```

### 3. [`verify_model_package.py`](file:///d:/Working_Repo/CustomMistralmamba/scripts/verify_model_package.py)
* **Goal**: CI contract enforcer verifying that `model` can be imported as a clean standalone package and that `model.__all__` exports all declared public symbols.
* **Usage**:
  ```bash
  python scripts/verify_model_package.py
  ```

### 4. [`bench_grad_guard.py`](file:///d:/Working_Repo/CustomMistralmamba/scripts/bench_grad_guard.py)
* **Goal**: Benchmarks gradient reduction and NaN guard execution. Verifies that `MEMORY_NAN_FIX_ID` guards prevent non-finite gradient corruption without imposing prohibitive overhead on backward passes.
* **Usage**:
  ```bash
  python scripts/bench_grad_guard.py
  ```

### 5. [`check_gradient_checkpointing.py`](file:///d:/Working_Repo/CustomMistralmamba/scripts/check_gradient_checkpointing.py)
* **Goal**: Compares memory usage and output gradients between `gradient_checkpointing=True` and `False` across hybrid attention/SSM blocks to ensure zero gradient divergence.
* **Usage**:
  ```bash
  python scripts/check_gradient_checkpointing.py
  ```
