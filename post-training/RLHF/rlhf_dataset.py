"""Preference record adapter over utils.dataset's bounded background shard producer.

Uses SFT's ShardFeed lifecycle and role-text tokenizer. No second queue, worker
pool, download manager, or packed language-modelling data path is introduced.
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import interleave_datasets, load_dataset
from torch.utils.data import Dataset

from utils.dataset import TokenizedShardProducer
from utils.sft_dataset import TokenizedShardProducer as SFTShardProducer
from utils.sft_dataset import tokenize_messages

from .config import DataConfig


def sources(config: DataConfig, purpose: str = "train") -> list[dict]:
    if purpose in ("train", "validation"):
        probabilities = config.probabilities()
        return [
            {
                "path": "HuggingFaceH4/ultrafeedback_binarized",
                "name": "default",
                "split": "train_prefs" if purpose == "train" else "test_prefs",
                "revision": config.ultrafeedback_revision,
                "adapter": "ultrafeedback",
                "weight": probabilities[0],
            },
            {
                "path": "Anthropic/hh-rlhf",
                "name": "default",
                "split": "train" if purpose == "train" else "test",
                "revision": config.hh_rlhf_revision,
                "adapter": "hh_rlhf",
                "weight": probabilities[1],
            },
        ]
    if purpose == "helpsteer2":
        return [
            {
                "path": "nvidia/HelpSteer2",
                "name": "default",
                "data_dir": "preference",
                "split": "train",
                "revision": config.helpsteer2_revision,
                "adapter": "helpsteer2",
                "weight": 1.0,
            }
        ]
    raise ValueError(f"Unknown dataset purpose {purpose!r}")


def _string(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing or empty text")
    return value.strip()


def _messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("Expected a nonempty message list")
    result = []
    for msg in value:
        if not isinstance(msg, dict) or msg.get("role") not in (
            "system",
            "user",
            "assistant",
        ):
            raise ValueError("Malformed conversation role")
        result.append({"role": msg["role"], "content": _string(msg.get("content"))})
    return result


def _hh_messages(text: str) -> list[dict[str, str]]:
    parts = re.split(r"(?:^|\n\n)(Human|Assistant):", text)
    if parts[0].strip() or len(parts) < 5:
        raise ValueError("Malformed HH dialogue")
    return _messages(
        [
            {
                "role": "user" if parts[i] == "Human" else "assistant",
                "content": parts[i + 1],
            }
            for i in range(1, len(parts), 2)
        ]
    )


def normalize_preference(row: dict, source: str) -> dict:
    """Normalize verified source schemas; invalid records raise ValueError.

    Canonical prompt/chosen/rejected are strings. prompt_messages preserves
    multi-turn boundaries so the existing SFT tokenizer can format them exactly.
    """
    category = "overall"
    if source in ("ultrafeedback", "hh_rlhf"):
        if source == "ultrafeedback":
            prompt = _string(row.get("prompt"))
            chosen, rejected = (
                _messages(row.get("chosen")),
                _messages(row.get("rejected")),
            )
        else:
            chosen = _hh_messages(_string(row.get("chosen")))
            rejected = _hh_messages(_string(row.get("rejected")))
        if (
            len(chosen) < 2
            or chosen[:-1] != rejected[:-1]
            or chosen[-1]["role"] != "assistant"
            or rejected[-1]["role"] != "assistant"
            or chosen[-2]["role"] != "user"
        ):
            raise ValueError("Chosen/rejected must share the same complete prompt")
        messages = chosen[:-1]
        if source == "ultrafeedback" and messages[-1]["content"] != prompt:
            raise ValueError("UltraFeedback prompt disagrees with conversation")
        chosen, rejected = chosen[-1]["content"], rejected[-1]["content"]
        prompt = "\n\n".join(m["content"] for m in messages)
    elif source == "helpsteer2":
        prompt = _string(row.get("prompt"))
        strength = row.get("preference_strength")
        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not math.isfinite(strength)
            or not 0 < abs(strength) <= 3
        ):
            raise ValueError("HelpSteer2 tie or invalid preference_strength")
        first, second = _string(row.get("response_1")), _string(row.get("response_2"))
        chosen, rejected = (first, second) if strength < 0 else (second, first)
        parts = re.split(r"<extra_id_1>(User|Assistant)\s*", prompt)
        messages = [{"role": "user", "content": _string(parts[0])}]
        messages.extend(
            {"role": parts[i].lower(), "content": _string(parts[i + 1])}
            for i in range(1, len(parts), 2)
        )
        if messages[-1]["role"] != "user":
            raise ValueError("HelpSteer2 prompt must end with a user turn")
        category = f"{row.get('split', 'unknown')}/strength_{abs(strength):g}"
    else:
        raise ValueError(f"Unsupported preference schema {source!r}")
    if chosen == rejected:
        raise ValueError("Identical chosen and rejected responses")
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "prompt_messages": messages,
        "category": category,
    }


def tokenize_preference(record: dict, tokenizer, config: DataConfig) -> dict:
    """Keep the entire shared prompt; reserve response tokens or filter the pair.

    Response overflow retains its beginning and ending, independently per side.
    Oversized prompts are filtered rather than silently removing the instruction.
    """
    messages = record.get(
        "prompt_messages", [{"role": "user", "content": record["prompt"]}]
    )
    prefix, _ = tokenize_messages(tokenizer, messages)
    prefix += tokenizer.encode("\nassistant:\n", add_special_tokens=False)
    encoded = {
        side: tokenizer.encode(record[side], add_special_tokens=False)
        for side in ("chosen", "rejected")
    }
    if not all(encoded.values()):
        raise ValueError("Empty tokenized response")
    budget = config.max_length - len(prefix) - 1
    reserve = min(config.min_response_tokens, max(map(len, encoded.values())))
    if budget < reserve:
        raise ValueError("Prompt too long to preserve response context")
    for side, ids in encoded.items():
        if len(ids) > budget:
            head = (budget + 1) // 2
            ids = ids[:head] + (ids[-(budget - head) :] if budget > head else [])
        encoded[side] = prefix + ids + [tokenizer.eos_token_id]
    if encoded["chosen"] == encoded["rejected"]:
        raise ValueError("Truncation/tokenization removed the preference difference")
    encoded["category"] = record.get("category", "overall")
    return encoded


def _envelope(row: dict, source: str) -> dict:
    return {"source": source, "payload": json.dumps(row, ensure_ascii=False)}


def load_streaming_source(config: dict):
    """Use the reference path/name/split schema with preference-specific selection."""
    kwargs = {
        key: config[key]
        for key in ("name", "revision", "data_dir", "data_files")
        if config.get(key) is not None
    }
    try:
        return load_dataset(
            config["path"], split=config.get("split", "train"), streaming=True, **kwargs
        )
    except Exception as exc:
        raise RuntimeError(
            f"Cannot open HF preference stream {config!r}: {exc}"
        ) from exc


class PreferenceShardProducer(SFTShardProducer, TokenizedShardProducer):
    """Pair adapter using unchanged pretraining/SFT producer interfaces.

    SFT supplies tokenizer injection, atomic JSON, and bounded `_flush`; the
    reference producer supplies thread failure propagation and retrying cleanup.
    Pair-aware iteration/storage live here because packed LM buffers cannot
    preserve chosen/rejected boundaries. Neither imported module is patched.
    """

    start_streaming = TokenizedShardProducer.start_streaming
    _cleanup_consumed_shards = TokenizedShardProducer._cleanup_consumed_shards

    def __init__(
        self,
        cache_dir: str,
        tokenizer,
        config: DataConfig,
        vocab_size: int,
        seed: int,
        purpose: str = "train",
        log_fn=None,
        cursor: dict | None = None,
        sample_stream=None,
    ):
        super().__init__(
            cache_dir,
            tokenizer_name=config.tokenizer_name,
            # Initialize the existing SFT plumbing with a valid window. The
            # inherited _flush counts buffer entries; ours are whole pairs.
            seq_len=config.max_length,
            tokens_per_shard=config.max_length,
            max_buffered_files=config.buffer_size,
            seed=seed,
            tokenizer=tokenizer,
            log_fn=log_fn,
            expected_vocab_size=vocab_size,
            dataset_configs=sources(config, purpose),
        )
        self.tokens_per_shard = config.pairs_per_shard
        self.config, self.vocab_size, self.purpose = config, vocab_size, purpose
        self.sample_stream = sample_stream
        self.stats = Counter(raw=0, filtered=0, valid=0)
        self.reasons = Counter()
        cursor = cursor or {}
        self.current_shard_idx = cursor.get("shard", 0)
        self.cumulative_samples = cursor.get("raw_start", 0)
        self.raw_start = self.cumulative_samples
        self._native_ds_state = cursor.get("native_start")
        self.native_start = self._native_ds_state

    def _stream_loop(self, stop_event=None, checkpoint_path=None):
        """Stream intact preference pairs through the inherited bounded flush.

        The consumer checkpoint owns source position. Native HF state and the
        reference producer's deterministic raw-row skip fallback are supported.
        """
        if checkpoint_path is not None:
            raise ValueError("Reward data state is saved with the trainer checkpoint")
        stream = self._build_stream()
        native = hasattr(stream, "state_dict") and hasattr(stream, "load_state_dict")
        restored = False
        if self._native_ds_state is not None and native:
            try:
                stream.load_state_dict(self._native_ds_state)
                restored = True
                self.log("[Reward Producer] Restored native HF stream boundary")
            except (
                AttributeError,
                NotImplementedError,
                TypeError,
                OSError,
                ValueError,
            ) as exc:
                warnings.warn(
                    f"Native reward stream restore failed ({exc}); replaying raw rows"
                )
        if not restored and self.cumulative_samples:
            self.log(
                f"[Reward Producer] Resume fallback: skip {self.cumulative_samples} raw rows (O(N))"
            )
            stream = stream.skip(self.cumulative_samples)
        self.error, self.finished = None, False
        self.log(
            f"[Reward Producer] Streaming {self.purpose}; bounded shards={self.max_buffered_files}"
        )
        iterator = iter(stream)
        while stop_event is None or not stop_event.is_set():
            try:
                sample = next(iterator)
            except StopIteration:
                if self._flush(stop_event, final=True):
                    self.finished = True
                return
            self.cumulative_samples += 1
            self.token_buffer.extend(self._encode_sample(sample))
            if native:
                try:
                    self._native_ds_state = stream.state_dict()
                except (AttributeError, NotImplementedError, TypeError, OSError):
                    self._native_ds_state = None
            if not self._flush(stop_event):
                return

    def save_checkpoint(self, checkpoint_path: str):
        raise ValueError(
            "Save reward data cursors through the reward trainer checkpoint"
        )

    def load_checkpoint(self, checkpoint_path: str):
        raise ValueError(
            "Restore reward data cursors through the reward trainer checkpoint"
        )

    def _build_stream(self):
        if self.sample_stream is not None:
            return self.sample_stream
        streams = []
        configs = sources(self.config, self.purpose)
        for cfg in configs:
            ds = load_streaming_source(cfg)
            required = (
                {"chosen", "rejected"}
                if cfg["adapter"] == "hh_rlhf"
                else (
                    {"prompt", "chosen", "rejected"}
                    if cfg["adapter"] == "ultrafeedback"
                    else {
                        "prompt",
                        "response_1",
                        "response_2",
                        "preference_strength",
                        "split",
                    }
                )
            )
            if ds.features is not None and not required.issubset(ds.features):
                raise ValueError(
                    f"Dataset schema changed for {cfg['path']}: {ds.features}"
                )
            streams.append(
                ds.map(_envelope, fn_kwargs={"source": cfg["adapter"]}).select_columns(
                    ["source", "payload"]
                )
            )
        return interleave_datasets(
            streams,
            probabilities=[c["weight"] for c in configs],
            seed=self.seed,
            stopping_strategy="all_exhausted",
        )

    def _encode_sample(self, sample: dict) -> list:
        self.stats["raw"] += 1
        try:
            record = normalize_preference(
                json.loads(sample["payload"]), sample["source"]
            )
        except (ValueError, KeyError, TypeError) as exc:
            self.stats["filtered"] += 1
            self.reasons[str(exc)] += 1
            return []
        # Tokenizer exceptions are serious failures and propagate via self.error.
        # Only our explicit content/truncation checks are filtered.
        try:
            record = tokenize_preference(record, self.tokenizer, self.config)
        except ValueError as exc:
            if str(exc) not in (
                "Empty tokenized response",
                "Prompt too long to preserve response context",
                "Truncation/tokenization removed the preference difference",
            ):
                raise RuntimeError("Reward tokenizer failed") from exc
            self.stats["filtered"] += 1
            self.reasons[str(exc)] += 1
            return []
        for side in ("chosen", "rejected"):
            if min(record[side]) < 0 or max(record[side]) >= self.vocab_size:
                raise ValueError("Tokenizer emitted an out-of-vocabulary reward token")
        self.stats["valid"] += 1
        return [record]

    def _write_shard(self, count: int) -> None:
        records = self.token_buffer[:count]
        tokens, offsets, categories = [], [0], []
        for record in records:
            for side in ("chosen", "rejected"):
                tokens.extend(record[side])
                offsets.append(len(tokens))
            categories.append(record["category"])
        self._publish_shard(
            np.asarray(tokens, dtype="<u4"),
            {
                "format": "reward-pairs-v1",
                "offsets": offsets,
                "categories": categories,
                "raw_start": self.raw_start,
                "raw_end": self.cumulative_samples,
                "native_start": self.native_start,
                "native_end": self._native_ds_state,
                "stats": dict(self.stats),
                "filter_reasons": dict(self.reasons),
            },
        )
        del self.token_buffer[:count]
        self.raw_start = self.cumulative_samples
        self.native_start = self._native_ds_state
        self.log(f"[Reward Producer] {dict(self.stats)} filters={dict(self.reasons)}")

    def _publish_shard(self, array: np.ndarray, metadata: dict) -> None:
        """Publish the binary last, matching the existing SFT ready-file contract."""
        base = Path(self.cache_dir) / f"shard_{self.current_shard_idx:06d}"
        if base.with_suffix(".bin").exists() or base.with_suffix(".json").exists():
            raise FileExistsError(f"Reward shard already exists: {base}")
        temporary = str(base) + ".bin.tmp"
        with open(temporary, "wb") as handle:
            array.tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
        self._atomic_json(str(base) + ".json", metadata)
        os.replace(temporary, str(base) + ".bin")
        self.current_shard_idx += 1


class PreferenceShardDataset(Dataset):
    """Lazy mmap pair reader. Each item is two independently sized sequences."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.metadata = json.loads(self.path.with_suffix(".json").read_text())
        self.offsets = self.metadata["offsets"]
        if (
            self.metadata.get("format") != "reward-pairs-v1"
            or len(self.offsets) < 3
            or len(self.offsets) % 2 != 1
            or self.offsets[0] != 0
            or any(a >= b for a, b in zip(self.offsets, self.offsets[1:]))
            or self.path.stat().st_size != self.offsets[-1] * 4
            or len(self.metadata["categories"]) != len(self)
        ):
            raise ValueError(f"Corrupt or incompatible preference shard {path}")
        self._data = None

    def __len__(self):
        return (len(self.offsets) - 1) // 2

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._data is None:
            self._data = np.memmap(self.path, dtype="<u4", mode="r")
        a, b, c = self.offsets[2 * index : 2 * index + 3]
        return {
            "chosen": self._data[a:b],
            "rejected": self._data[b:c],
            "category": self.metadata["categories"][index],
        }

    def close(self):
        if self._data is not None:
            self._data._mmap.close()
            self._data = None

    def __getstate__(self):
        return dict(self.__dict__, _data=None)


class PreferenceCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, records: list[dict]) -> dict:
        batch = {"category": [r.get("category", "overall") for r in records]}
        # One common dynamic width permits a single chosen+rejected model call.
        width = max(len(r[s]) for r in records for s in ("chosen", "rejected"))
        for side in ("chosen", "rejected"):
            ids = torch.full((len(records), width), self.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for i, row in enumerate(records):
                length = len(row[side])
                ids[i, :length] = torch.as_tensor(np.asarray(row[side], dtype=np.int64))
                mask[i, :length] = True
            batch[f"{side}_input_ids"] = ids
            batch[f"{side}_attention_mask"] = mask
        return batch


class SmokeTokenizer:
    """Offline deterministic test vocabulary; never used for cloud training."""

    bos_token_id, eos_token_id, pad_token_id, unk_token_id = 1, 2, 0, 3
    name_or_path = "reward-smoke-only"

    def __len__(self):
        return 256

    def encode(self, text, add_special_tokens=False):
        return [4 + (byte % 252) for byte in text.encode()]


def smoke_stream(purpose: str):
    from datasets import IterableDataset

    rows = []
    for i in range(12):
        if purpose == "helpsteer2":
            row = {
                "prompt": f"Q{i}?",
                "response_1": "Correct.",
                "response_2": "Wrong.",
                "preference_strength": -1,
                "split": "validation",
            }
            source = "helpsteer2"
        elif i % 2:
            source = "hh_rlhf"
            row = {
                "chosen": f"\n\nHuman: Q{i}?\n\nAssistant: Correct.",
                "rejected": f"\n\nHuman: Q{i}?\n\nAssistant: Wrong.",
            }
        else:
            source = "ultrafeedback"
            prompt = f"Q{i}?"
            row = {
                "prompt": prompt,
                "chosen": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "Correct."},
                ],
                "rejected": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "Wrong."},
                ],
            }
        rows.append(_envelope(row, source))
    return IterableDataset.from_generator(lambda: iter(rows))
