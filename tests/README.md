# Tests Directory (`tests`)

The `tests` directory contains the unit, integration, and distributed contract tests for the **Hybrid Mamba–MoE with Dual Compressive Memory** repository.

All tests are designed to execute safely on CPU in continuous integration (CI) environments without requiring GPUs or active internet downloads.

---

## Test Suite Index

| Test Module | Coverage | Key Test Classes |
| :--- | :--- | :--- |
| [`test_model.py`](file:///d:/Working_Repo/CustomMistralmamba/tests/test_model.py) | Full architecture verification (83 unit tests): sliding-window GQA, Mamba scans, dual memory, Top-2 MoE, 8 auxiliary losses, ablation hooks | `TestHybridModel`, `TestMambaScanParity`, `TestDualMemory`, `TestMoERouting`, `TestAuxiliaryLosses` |
| [`test_toy_train_smoke.py`](file:///d:/Working_Repo/CustomMistralmamba/tests/test_toy_train_smoke.py) | End-to-end CPU training smoke test | `TestToyTrainSmoke` |
| [`test_sft_dataset.py`](file:///d:/Working_Repo/CustomMistralmamba/tests/test_sft_dataset.py) | SFT dataset adapters, tokenization, binary shard generation, window packing, and oversized conversation handling | `TestSFTAdapters`, `TestSFTShards` |
| [`test_sft_post_train.py`](file:///d:/Working_Repo/CustomMistralmamba/tests/test_sft_post_train.py) | SFT training loop, CPU warm start, resume contract parity, token-weighted CE loss, and distributed sampler cursor | `TestSFTTraining` |
| [`test_fsdp2_contracts.py`](file:///d:/Working_Repo/CustomMistralmamba/tests/test_fsdp2_contracts.py) | Distributed contracts: Newton–Schulz math, MuonDTensor steps, replicated gradient sync, and structured telemetry | `TestFSDP2Contracts`, `TestNewtonSchulzMath` |

---

## Detailed Test Module Descriptions

### 1. [`test_model.py`](file:///d:/Working_Repo/CustomMistralmamba/tests/test_model.py)
* **Scan Parity Across 4 Tiers**: Verifies numerical consistency between sequential scan, blocked vectorized scan, and parallel scan algorithms.
* **Dual Compressive Memory**: Validates read/write decoupling, chunk-aligned memory writes, memory bank reset mechanisms, and gradient flow into memory combination layers.
* **Top-2 MoE & Auxiliary Losses**: Tests expert routing, capacity factor enforcement, router z-loss, load-balancing loss, and the full suite of eight auxiliary losses.
* **Ablation Hooks**: Tests `use_dual_memory=False`, `zero_memory_states()`, and parameter-matched null baseline configurations.

### 2. [`test_sft_dataset.py`](file:///d:/Working_Repo/CustomMistralmamba/tests/test_sft_dataset.py)
* **Message Tokenization & Masking**: Verifies that user prompts, system messages, and padding have loss mask `0`, while assistant responses have loss mask `1`.
* **Window Packing**: Confirms that complete conversations fit into `seq_len` windows without cross-window splitting.
* **Oversized Conversation Handling**:
  * Tests that `oversized_behavior="filter"` cleanly skips conversations longer than the context window and increments `skipped_oversized_samples`.
  * Tests `oversized_behavior="truncate"` and `oversized_behavior="error"`.
* **Topic Filtering**: Validates [`get_dataset_configs(exclude_topics=...)`](file:///d:/Working_Repo/CustomMistralmamba/utils/sft_dataset.py#L144) topic exclusion and sampling weight re-normalization.

### 3. [`test_sft_post_train.py`](file:///d:/Working_Repo/CustomMistralmamba/tests/test_sft_post_train.py)
* **Training Integrity**: Verifies that warm-starting from a pretrained checkpoint updates weights on CPU and advances global step and batch offsets.
* **Interrupted Resumption Parity**: Validates that simulating an interruption and resuming from a checkpoint produces identical weights and states compared to uninterrupted execution.
* **Loss Weighting**: Ensures cross-entropy loss is scaled by global assistant tokens across accumulation windows.

### 4. [`test_fsdp2_contracts.py`](file:///d:/Working_Repo/CustomMistralmamba/tests/test_fsdp2_contracts.py)
* **Newton–Schulz Math**: Validates that 5th-order Newton–Schulz orthogonalization matches true SVD polar decomposition within tight numerical tolerances.
* **Replicated Gradient Synchronization**: Tests multi-process Gloo synchronization for non-sharded custom-math parameters.

---

## Running the Tests

Run all unit tests:
```bash
python -m unittest discover tests -v
```

Run specific test modules:
```bash
# Model architecture tests (83 tests):
python -m unittest tests.test_model -v

# SFT dataset tests:
python -m unittest tests.test_sft_dataset -v

# SFT post-training loop tests:
python -m unittest tests.test_sft_post_train -v

# FSDP2 contracts:
python -m unittest tests.test_fsdp2_contracts -v

# Quick toy train smoke test:
python -m unittest tests.test_toy_train_smoke -v
```
