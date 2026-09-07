# Utilities Module (`utils`)

The `utils` module provides shared pipelines, optimizers, dataset loaders, streaming consumers, validation runners, and telemetry utilities used across pre-training and supervised fine-tuning (SFT) in the **Hybrid Mamba–MoE with Dual Compressive Memory** repository.

---

## Module Overview

| File | Purpose | Key Symbols |
| :--- | :--- | :--- |
| [`dataset.py`](file:///d:/Working_Repo/CustomMistralmamba/utils/dataset.py) | Streaming pretraining dataset producer & reader | [`TokenizedShardProducer`](file:///d:/Working_Repo/CustomMistralmamba/utils/dataset.py#L83), [`MmapShardDataset`](file:///d:/Working_Repo/CustomMistralmamba/utils/dataset.py#L427), [`verify_tokenizer_vocab`](file:///d:/Working_Repo/CustomMistralmamba/utils/dataset.py#L83), [`resolve_tokenizer_vocab_size`](file:///d:/Working_Repo/CustomMistralmamba/utils/dataset.py#L46) |
| [`sft_dataset.py`](file:///d:/Working_Repo/CustomMistralmamba/utils/sft_dataset.py) | SFT dialogue mixture streaming, window packing & shard generation | [`TokenizedShardProducer`](file:///d:/Working_Repo/CustomMistralmamba/utils/sft_dataset.py#L377), [`MmapShardDataset`](file:///d:/Working_Repo/CustomMistralmamba/utils/sft_dataset.py#L671), [`extract_messages_from_sample`](file:///d:/Working_Repo/CustomMistralmamba/utils/sft_dataset.py#L202), [`tokenize_messages`](file:///d:/Working_Repo/CustomMistralmamba/utils/sft_dataset.py#L356), [`get_dataset_configs`](file:///d:/Working_Repo/CustomMistralmamba/utils/sft_dataset.py#L144) |
| [`fsdp2_muon.py`](file:///d:/Working_Repo/CustomMistralmamba/utils/fsdp2_muon.py) | Distributed DTensor-aware Muon optimizer for PyTorch FSDP2 | [`MuonDTensor`](file:///d:/Working_Repo/CustomMistralmamba/utils/fsdp2_muon.py#L182), [`zeropower_via_newtonschulz5`](file:///d:/Working_Repo/CustomMistralmamba/utils/fsdp2_muon.py#L15), [`adjust_lr_factor`](file:///d:/Working_Repo/CustomMistralmamba/utils/fsdp2_muon.py#L82) |
| [`validation.py`](file:///d:/Working_Repo/CustomMistralmamba/utils/validation.py) | Validation dataset pipeline, token perplexity evaluation | [`CyclicValidationDataset`](file:///d:/Working_Repo/CustomMistralmamba/utils/validation.py#L42), [`evaluate_validation_loss`](file:///d:/Working_Repo/CustomMistralmamba/utils/validation.py#L125), [`build_causal_labels`](file:///d:/Working_Repo/CustomMistralmamba/utils/validation.py#L31) |
| [`training_logging.py`](file:///d:/Working_Repo/CustomMistralmamba/utils/training_logging.py) | Standardized telemetry formatting and structured JSONL logging | [`format_training_log_line`](file:///d:/Working_Repo/CustomMistralmamba/utils/training_logging.py#L11), [`append_jsonl_record`](file:///d:/Working_Repo/CustomMistralmamba/utils/training_logging.py#L33) |

---

## Detailed Component Documentation

### 1. Pretraining Streaming Shards ([`dataset.py`](file:///d:/Working_Repo/CustomMistralmamba/utils/dataset.py))

* **`TokenizedShardProducer`**: Runs in a background thread on Rank 0, streaming interleaved Hugging Face datasets (e.g. FineWeb, Stack-v2, Math, OpenWebMath). It tokenizes documents, bounds memory with a buffered shard queue, and writes immutable binary files (`shard_NNNNNN.bin`) with sidecar JSON metadata.
* **`MmapShardDataset`**: Reads binary token shards via `np.memmap` for zero-overhead multi-worker data loading. Yields pre-shifted `(input_ids, labels)` pairs of sequence length `seq_len`.
* **Vocabulary Verification**:
  * [`resolve_tokenizer_vocab_size(tokenizer)`](file:///d:/Working_Repo/CustomMistralmamba/utils/dataset.py#L46): Probes tokenizers to determine the actual maximum token ID that could be emitted, preventing out-of-range CUDA embedding index errors.
  * [`verify_tokenizer_vocab(tokenizer, vocab_size)`](file:///d:/Working_Repo/CustomMistralmamba/utils/dataset.py#L83): Enforces strict equality between tokenizer output range and model embedding dimensions before training starts.

### 2. Supervised Fine-Tuning Pipeline ([`sft_dataset.py`](file:///d:/Working_Repo/CustomMistralmamba/utils/sft_dataset.py))

* **Data Formatting & Roles**:
  * Serializes conversations using explicit role-text convention: `\n<role>:\n<content>`, with `system`, `developer`, `user`, `assistant`, and `tool` (normalizing synonyms like `human`, `prompter`, `gpt`, `model`, `bot`).
  * Only assistant content and its trailing EOS token are supervised (loss mask = 1). All user prompts, system instructions, tool outputs, and padding have loss label `-100` (`IGNORE_INDEX`).
* **Conversation Packing**:
  * Whole conversations are packed into windows of size `seq_len + 1` with shared causal attention.
  * Conversations are **never split** across windows; any remainder at the end of a window is padded with `pad_id` and mask `0`.
* **Oversized Conversation Handling (`--oversized-behavior`)**:
  * `filter` (default): Gracefully skips conversations exceeding the window length, increments `skipped_oversized_samples`, and continues streaming without crashing training.
  * `truncate`: Truncates tokens and loss mask to `seq_len` (skipping the sample if no supervised assistant tokens remain after truncation).
  * `error`: Raises an explicit `ValueError` (legacy mode).
* **Topic Mixture & Customization**:
  * Default mixture contains 8 topics (`general`, `coding`, `math`, `science`, `agents`, `long_context`, `structured`, `safety`).
  * [`get_dataset_configs(exclude_topics=...)`](file:///d:/Working_Repo/CustomMistralmamba/utils/sft_dataset.py#L144): Allows filtering out topics (e.g. `exclude_topics=["long_context"]` for 4096-context training) while automatically re-normalizing sampling weights to sum to 1.0.

### 3. FSDP2 Muon Distributed Optimizer ([`fsdp2_muon.py`](file:///d:/Working_Repo/CustomMistralmamba/utils/fsdp2_muon.py))

* **`MuonDTensor`**:
  * Adapts the Muon optimizer to work seamlessly with PyTorch FSDP2 parameter sharding.
  * Because Newton–Schulz matrix orthogonalization is non-separable across row shards, `MuonDTensor` gathers full parameter momentum matrices into bounded memory buffers (`--muon-gather-buffer-mb`), computes the orthogonalized update via [`zeropower_via_newtonschulz5`](file:///d:/Working_Repo/CustomMistralmamba/utils/fsdp2_muon.py#L15), and shards the update back into DTensors.
  * Supports `match_rms_adamw` and `original` scaling modes via [`adjust_lr_factor`](file:///d:/Working_Repo/CustomMistralmamba/utils/fsdp2_muon.py#L82).

### 4. Validation & Metrics ([`validation.py`](file:///d:/Working_Repo/CustomMistralmamba/utils/validation.py), [`training_logging.py`](file:///d:/Working_Repo/CustomMistralmamba/utils/training_logging.py))

* **`CyclicValidationDataset`**: Wraps validation shards to provide deterministically cycled evaluation batches.
* **`evaluate_validation_loss`**: Computes perplexity and cross-entropy loss exclusively on valid tokens, ignoring padding and prompt tokens.
* **`training_logging`**: Formats console log lines and writes structured JSONL telemetry for downstream analysis.

---

## Testing Utilities

Run the unit tests covering `utils`:
```bash
python -m unittest tests.test_sft_dataset -v
```
