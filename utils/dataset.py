"""
Streaming HF dataset -> tokenized binary shard producer, and a memory-mapped
consumer Dataset for training.

Fixes applied vs. the reviewed version:
  - `verify_tokenizer_vocab()` added: assert the tokenizer's vocab size
    matches `MixtralConfig.vocab_size` before training starts, instead of
    silently letting a mismatch corrupt embeddings/logits.
  - `TokenizedShardProducer` now attempts to use `datasets`' native
    `IterableDataset.state_dict()` / `load_state_dict()` checkpointing
    (available in recent `datasets` versions) so resumption does not have to
    restart the stream from position zero and re-skip `cumulative_samples`
    elements (which was O(N) dead time on long runs). If the installed
    `datasets` version does not support this (older versions raise
    `AttributeError`/`NotImplementedError`), we fall back to the previous
    skip-based approach and log a clear warning so this limitation is
    visible rather than silent.
  - BOS/EOS ids are still read from the tokenizer first and only default to
    1/2 if the tokenizer doesn't define them -- this was already correct in
    the reviewed version, kept as-is (see review.md 2.5; the fallback only
    triggers when `tokenizer.bos_token_id`/`eos_token_id` is None, it does
    not clobber a tokenizer that defines different ids).
  - `MmapShardDataset` gained an optional `strict_alignment` check that
    warns (once) if `total_tokens % seq_len != 0`, so silent token loss at
    shard boundaries is visible. The actual fix for token loss is on the
    producer side in main.py: `TOKENS_PER_SHARD` is chosen to be an exact
    multiple of `seq_len` so no tokens are dropped at shard boundaries.
"""

import base64
import glob
import json
import os
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from datasets import interleave_datasets, load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def resolve_tokenizer_vocab_size(
    tokenizer: PreTrainedTokenizerBase,
    probe_texts: tuple[str, ...] = (
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n):\n    a, b = 0, 1\n    return a",
        "1234567890 !@#$%^&*() []{};':\",./<>?",
        "\n\t éü中文 한국어 Русский",
    ),
) -> int:
    """Resolve the vocab size this tokenizer can *actually emit*.

    Llama-family tokenizers frequently report ``len(tokenizer) == V`` while
    still encoding some inputs to ``id == V`` (the reported size excludes a
    slot the encoder happily produces). A plain equality check against the
    model config passes such a tokenizer, which then crashes out-of-range
    hours into a run. Probe with real encodes to find the true ceiling.
    """
    reported = len(tokenizer)
    max_id = -1
    for text in probe_texts:
        # Probe strings are constants; a failure here is a real tokenizer bug
        # and should surface rather than be swallowed.
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            max_id = max(max_id, int(max(ids)))
    # Special tokens are legal outputs too.
    for special in (
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    ):
        if special is not None:
            max_id = max(max_id, int(special))
    return max(reported, max_id + 1)


def verify_tokenizer_vocab(
    tokenizer: PreTrainedTokenizerBase, expected_vocab_size: int
) -> None:
    """Raises if the tokenizer can emit ids the model config cannot embed.

    Only ids >= vocab_size can corrupt training (CUDA indexing errors or
    silent aliasing into unrelated embedding rows), so the check fires when
    the probed ceiling exceeds the configured size. Extra embedding rows are
    harmless dead weight. See :func:`resolve_tokenizer_vocab_size` for why
    the reported ``len(tokenizer)`` alone is not trusted.
    """
    resolved = resolve_tokenizer_vocab_size(tokenizer)
    if resolved > expected_vocab_size:
        raise ValueError(
            f"Tokenizer '{tokenizer.name_or_path}' can emit token ids up to "
            f"{resolved - 1}, but the model vocab_size={expected_vocab_size}. "
            f"Such ids crash or corrupt training mid-run. Set --vocab-size to "
            f"{resolved} (see resolve_tokenizer_vocab_size) or use a tokenizer "
            f"whose emitted range fits the model."
        )


def extract_text_from_sample(sample: dict[str, Any], text_col: Any) -> str:
    """Extracts and formats string text across diverse dataset column structures."""
    if isinstance(text_col, list):
        parts = [
            str(sample.get(col, "")).strip() for col in text_col if sample.get(col)
        ]
        return "\n\n".join(parts)
    elif text_col == "messages":
        msgs = sample.get("messages", [])
        if isinstance(msgs, list):
            formatted = []
            for msg in msgs:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    formatted.append(f"{role}: {content}")
            return "\n".join(formatted)
        return str(msgs)
    else:
        val = sample.get(text_col)
        if val is None:
            for fallback in ["text", "code", "content"]:
                if sample.get(fallback):
                    val = sample[fallback]
                    break
        return str(val) if val is not None else ""


class TokenizedShardProducer:
    """
    Handles streaming raw text from HF Hub, tokenizing with BOS/EOS tags, and
    serializing tightly packed binary arrays on CPU with checkpoint save/load
    capabilities.
    """

    def __init__(
        self,
        cache_dir: str,
        tokenizer_name: str = "UIC-AI-lab/llama2-tokenizer",
        tokens_per_shard: int = 100_000,
        max_buffered_files: int = 3,
        seed: int | None = None,
        log_fn: Callable[[str], None] | None = None,
    ):
        self.cache_dir = cache_dir
        self.tokenizer_name = tokenizer_name
        self.tokens_per_shard = tokens_per_shard
        self.max_buffered_files = max_buffered_files
        # Seeds interleave_datasets' sampling so stream order is reproducible
        # for a given --seed (HF otherwise draws from unseeded entropy).
        self.seed = seed
        # Diagnostics route through the caller's logger when provided, so
        # producer messages land in runs/*/train.log instead of only stdout.
        self.log = (
            log_fn if log_fn is not None else (lambda msg: print(msg, flush=True))
        )
        # Exception raised inside the streaming thread; the trainer polls
        # this and aborts loudly rather than timing out like a stall.
        self.error: BaseException | None = None

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token_id = 1
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = 2

        os.makedirs(cache_dir, exist_ok=True)

        # State variables for checkpointing.
        self.current_shard_idx = 0
        self.cumulative_samples = 0
        self.token_buffer = []

        # Native `datasets` IterableDataset resumption state (populated by
        # start_streaming once the stream object exists); used instead of
        # the slow skip-based fallback when available.
        self._native_ds_state: dict[str, Any] | None = None
        self._supports_native_state = False

    def save_checkpoint(self, checkpoint_path: str):
        """Saves current dataset streaming state and buffer to a JSON file.

        The token buffer is serialized as base64-packed uint16 (a JSON list of
        Python ints was both large and slow for buffers near tokens_per_shard).
        """
        buf = np.asarray(self.token_buffer, dtype=np.uint16)
        state = {
            "current_shard_idx": self.current_shard_idx,
            "cumulative_samples": self.cumulative_samples,
            "token_buffer_len": int(buf.size),
            "token_buffer_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
            "tokens_per_shard": self.tokens_per_shard,
            "tokenizer_name": self.tokenizer_name,
            "native_ds_state": self._native_ds_state,
        }
        tmp_path = f"{checkpoint_path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f)
        os.replace(tmp_path, checkpoint_path)
        self.log(f"[CPU Producer] Checkpoint saved successfully to: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Loads dataset state from checkpoint file to resume streaming seamlessly."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Dataset checkpoint not found at: {checkpoint_path}"
            )

        with open(checkpoint_path, "r") as f:
            state = json.load(f)

        self.current_shard_idx = state["current_shard_idx"]
        self.cumulative_samples = state["cumulative_samples"]
        if "token_buffer_b64" in state:
            raw = base64.b64decode(state["token_buffer_b64"])
            arr = np.frombuffer(raw, dtype=np.uint16)
            expected_len = int(state.get("token_buffer_len", arr.size))
            if arr.size != expected_len:
                raise ValueError(
                    f"Dataset checkpoint token buffer corrupt: decoded "
                    f"{arr.size} tokens, header says {expected_len}"
                )
            self.token_buffer = arr.astype(np.int64).tolist()
        else:  # legacy checkpoints serialized the buffer as a JSON int list
            self.token_buffer = list(state.get("token_buffer", []))
        self._native_ds_state = state.get("native_ds_state")
        self.log(
            f"[CPU Producer] Resumed checkpoint state from {checkpoint_path} | "
            f"Next Shard: {self.current_shard_idx:06d} | "
            f"Samples Skipped: {self.cumulative_samples}"
        )

    def _cleanup_consumed_shards(self):
        """Removes consumed binary shards marked with .done sentinels.

        On Windows, workers holding mmap handles can make unlink fail
        transiently — retry briefly; leaving the `.done` sentinel in place on
        failure means the next cleanup cycle tries again (the shard is never
        lost, just deleted late).
        """
        sentinel_files = glob.glob(os.path.join(self.cache_dir, "*.done"))
        for sentinel_path in sentinel_files:
            shard_base = sentinel_path.replace(".done", "")
            candidates = [
                f"{shard_base}.bin",
                f"{shard_base}.bin.tmp",
                f"{shard_base}.json",
            ]

            last_exc: OSError | None = None
            for attempt in range(3):
                try:
                    for path in candidates:
                        if os.path.exists(path):
                            os.remove(path)
                    os.remove(sentinel_path)
                    last_exc = None
                    break
                except OSError as exc:
                    last_exc = exc
                    time.sleep(0.5 * (attempt + 1))
            if last_exc is not None:
                self.log(
                    f"[CPU Producer] Could not delete consumed shard "
                    f"{shard_base}.* yet ({last_exc}); will retry next cycle."
                )

    def _build_stream(self):
        """Builds and returns the interleaved HF streaming dataset."""
        dataset_configs = [
            {
                "path": "HuggingFaceFW/fineweb",
                "name": "sample-100BT",
                "weight": 0.35,
                "text_col": "text",
                "split": "train",
            },
            {
                "path": "HuggingFaceFW/fineweb-edu",
                "name": None,
                "weight": 0.15,
                "text_col": "text",
                "split": "train",
            },
            {
                "path": "bigcode/the-stack-v2",
                "name": None,
                "weight": 0.15,
                "text_col": "code",
                "split": "train",
            },
            {
                "path": "emozilla/pg19",
                "name": None,
                "weight": 0.04,
                "text_col": "text",
                "split": "train",
            },
            {
                "path": "open-web-math/open-web-math",
                "name": None,
                "weight": 0.10,
                "text_col": "text",
                "split": "train",
            },
            {
                "path": "AI-MO/NuminaMath-CoT",
                "name": None,
                "weight": 0.10,
                "text_col": ["problem", "solution"],
                "split": "train",
            },
            {
                "path": "sentence-transformers/eli5",
                "name": None,
                "weight": 0.05,
                "text_col": ["question", "answer"],
                "split": "train",
            },
            {
                "path": "HuggingFaceH4/ultrachat_200k",
                "name": None,
                "weight": 0.06,
                "text_col": "messages",
                "split": "train_sft",
            },
        ]

        streams = []
        probabilities = []

        for cfg in dataset_configs:
            split_name = cfg.get("split", "train")
            kwargs = {"split": split_name, "streaming": True}
            if cfg.get("name"):
                kwargs["name"] = cfg["name"]

            ds = load_dataset(cfg["path"], **kwargs)

            col_target = cfg["text_col"]
            ds = ds.map(
                lambda sample, c=col_target: {
                    "text": extract_text_from_sample(sample, c)
                }
            ).select_columns(["text"])

            streams.append(ds)
            probabilities.append(cfg["weight"])

        if self.seed is not None:
            try:
                hf_stream = interleave_datasets(
                    streams,
                    probabilities=probabilities,
                    stopping_strategy="all_exhausted",
                    seed=self.seed,
                )
            except TypeError:
                # Older `datasets` without the seed kwarg: keep going, but the
                # stream order will not be reproducible across runs.
                warnings.warn(
                    "[CPU Producer] installed `datasets` lacks "
                    "interleave_datasets(seed=); interleaved stream order will "
                    "NOT be reproducible despite the configured seed."
                )
                hf_stream = interleave_datasets(
                    streams,
                    probabilities=probabilities,
                    stopping_strategy="all_exhausted",
                )
        else:
            hf_stream = interleave_datasets(
                streams, probabilities=probabilities, stopping_strategy="all_exhausted"
            )
        return hf_stream

    def start_streaming(self, stop_event=None, checkpoint_path: str | None = None):
        """Thread-target wrapper: records failures on ``self.error``.

        A daemon thread that dies with an exception is otherwise invisible —
        the trainer would just time out as if the producer were slow. Storing
        the exception lets the trainer raise it with context.
        """
        try:
            self._stream_loop(stop_event, checkpoint_path)
        except BaseException as exc:
            self.error = exc
            self.log(f"[CPU Producer] FAILED with: {exc!r}")
            raise

    def _stream_loop(self, stop_event=None, checkpoint_path: str | None = None):
        """Main execution engine loop for streaming, tokenizing, and adding BOS/EOS."""
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

        self.log(
            "[CPU Producer] Connecting to Hugging Face multi-stream repositories..."
        )

        hf_stream = self._build_stream()

        # Prefer native IterableDataset resumption (avoids restarting the
        # stream from scratch and re-skipping cumulative_samples elements,
        # which is O(N) dead time). Fall back to skip-based resumption if
        # the installed `datasets` version doesn't support it.
        self._supports_native_state = hasattr(hf_stream, "load_state_dict") and hasattr(
            hf_stream, "state_dict"
        )

        if self._native_ds_state is not None:
            if self._supports_native_state:
                try:
                    hf_stream.load_state_dict(self._native_ds_state)
                    self.log(
                        "[CPU Producer] Restored native IterableDataset state (no re-skip needed)."
                    )
                except (
                    AttributeError,
                    NotImplementedError,
                    TypeError,
                    OSError,
                    ValueError,
                ) as exc:  # pragma: no cover - defensive
                    warnings.warn(
                        f"[CPU Producer] Native dataset state restore failed ({exc}); "
                        f"falling back to skip-based resumption from sample {self.cumulative_samples}."
                    )
                    if self.cumulative_samples > 0:
                        hf_stream = hf_stream.skip(self.cumulative_samples)
            elif self.cumulative_samples > 0:
                warnings.warn(
                    "[CPU Producer] Installed `datasets` version does not support "
                    "IterableDataset.state_dict()/load_state_dict(); falling back to "
                    f"skip-based resumption ({self.cumulative_samples} samples, O(N) cost). "
                    "Upgrade `datasets` for fast, exact resumption."
                )
                hf_stream = hf_stream.skip(self.cumulative_samples)
        elif self.cumulative_samples > 0:
            self.log(
                f"[CPU Producer] Skipping first {self.cumulative_samples} samples to resume stream position..."
            )
            hf_stream = hf_stream.skip(self.cumulative_samples)

        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        self.log(
            f"[CPU Producer] Tokenizer initialized: {self.tokenizer_name} (Vocab size: {len(self.tokenizer)})"
        )
        self.log(
            "[CPU Producer] Pipeline streaming live. Extracting, adding BOS/EOS, and chunking tokens..."
        )

        for sample in hf_stream:
            if stop_event and stop_event.is_set():
                self.log(
                    "[CPU Producer] Shutdown signal received. Exiting thread cleanly..."
                )
                break

            self._cleanup_consumed_shards()

            while (
                len(glob.glob(os.path.join(self.cache_dir, "*.bin")))
                >= self.max_buffered_files
            ):
                if stop_event and stop_event.is_set():
                    break
                time.sleep(0.5)
                self._cleanup_consumed_shards()

            text = sample["text"]
            # Count BEFORE the blank check: skip-based fallback resume calls
            # hf_stream.skip(self.cumulative_samples), which counts blanks —
            # incrementing after the check drifted the resumed stream position
            # backward by the number of blank docs consumed (duplicated data).
            self.cumulative_samples += 1
            if not text.strip():
                continue

            raw_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            doc_tokens = [bos_id] + raw_tokens + [eos_id]

            self.token_buffer.extend(doc_tokens)

            if self._supports_native_state:
                try:
                    self._native_ds_state = hf_stream.state_dict()
                except (
                    AttributeError,
                    NotImplementedError,
                    TypeError,
                    OSError,
                ):  # pragma: no cover - defensive
                    self._native_ds_state = None

            while len(self.token_buffer) >= self.tokens_per_shard:
                shard_tokens = np.array(
                    self.token_buffer[: self.tokens_per_shard], dtype=np.uint16
                )
                self.token_buffer = self.token_buffer[self.tokens_per_shard :]

                bin_path = os.path.join(
                    self.cache_dir, f"shard_{self.current_shard_idx:06d}.bin"
                )
                json_path = os.path.join(
                    self.cache_dir, f"shard_{self.current_shard_idx:06d}.json"
                )

                # Atomic handoff: write to a tmp file, fsync, then os.replace.
                # A trainer polling for shard_NNNNNN.bin can otherwise memmap
                # a half-written file (existence != complete); after replace,
                # existence guarantees a full, durable shard. The .json sidecar
                # is published last so its presence is a second completion
                # signal for consumers that gate on it.
                bin_tmp = f"{bin_path}.tmp"
                with open(bin_tmp, "wb") as bf:
                    shard_tokens.tofile(bf)
                    bf.flush()
                    os.fsync(bf.fileno())
                os.replace(bin_tmp, bin_path)

                json_tmp = f"{json_path}.tmp"
                with open(json_tmp, "w") as jf:
                    json.dump(
                        {
                            "cumulative_samples_streamed_at_boundary": self.cumulative_samples,
                            "num_tokens": int(shard_tokens.size),
                        },
                        jf,
                    )
                    jf.flush()
                    os.fsync(jf.fileno())
                os.replace(json_tmp, json_path)

                self.current_shard_idx += 1


class MmapShardDataset(Dataset):
    """Zero-copy consumer map layer reading memory-mapped binary shards."""

    _warned_alignment = False

    def __init__(self, bin_path: str, seq_len: int):
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.num_sequences = len(self.data) // self.seq_len

        remainder = len(self.data) % self.seq_len
        if remainder != 0 and not MmapShardDataset._warned_alignment:
            # This is expected to be rare/zero if TOKENS_PER_SHARD (main.py)
            # is chosen as an exact multiple of seq_len; warn once so a
            # misconfiguration is visible instead of silently dropping
            # tokens at every shard boundary.
            warnings.warn(
                f"[MmapShardDataset] shard has {remainder} leftover tokens that don't "
                f"fill a full sequence of length {seq_len} and will be dropped. Set "
                f"TOKENS_PER_SHARD to a multiple of seq_len to avoid this."
            )
            MmapShardDataset._warned_alignment = True

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        return chunk[:-1], chunk[1:]
