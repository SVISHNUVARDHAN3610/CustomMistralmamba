"""Weighted HF SFT streams -> binary shards -> memory-mapped training pairs.

Follows ``utils.dataset``: CPU producer, bounded shard queue, atomic publication,
JSON checkpoints, ``shard_NNNNNN.done`` cleanup, and shifted ``(inputs, labels)``.
Import BOTH classes from this module: SFT shards have uint32 tokens and a uint8
loss-mask companion and are incompatible with the pretraining shard reader.

Topic weights are conversation sampling probabilities, NOT token percentages.
Each topic is divided equally among its listed sources; rStar's two SFT subsets
share one source allocation. Repeated ToolACE/APIGen allocations are intentional.
Smoltalk/all, Tulu and other compilations overlap; no global deduplication or
semantic topic classification is implied by these source buckets.

Formatting deliberately follows the pretraining role-text convention, using
ordinary ``role:\n`` headers (no new vocabulary). Assistant content and its EOS
are supervised; system/user/tool content and padding have label -100. Use this
same format at inference. Complete conversations are packed into seq_len windows
with shared causal attention, as in pretraining, but never split across windows.
Oversize conversations raise instead of silently removing long-context prompts.
Set seq_len to the model's supported context plus one (one token is shifted away).

Example (run the producer in a thread alongside the existing shard queue loop)::

    from utils.sft_dataset import TokenizedShardProducer, MmapShardDataset
    producer = TokenizedShardProducer(
        "SFT_DATA/shards", seq_len=65537, tokens_per_shard=65537 * 8, seed=42,
    )
    # thread target: producer.start_streaming(stop_event, checkpoint_path)
    # inputs, labels = MmapShardDataset(shard_path, seq_len=65537)[0]

Checkpoint the producer and consumer together while paused. Resume requires the
same source revisions/settings and a matching queue snapshot. Stream resumption
replays and skips normalized conversations deterministically (O(N)); it does not
claim native HF resumption for the OASST tree and raw JSONL adapters. A checkpoint
ahead/behind the queue must be reconciled by the trainer, never overwritten here.
"""

from __future__ import annotations

import base64
import glob
import hashlib
import itertools
import json
import math
import os
import threading
import time
from collections.abc import Callable, Iterator
from copy import deepcopy
from typing import Any

# Disable JAX in Hugging Face datasets to prevent background worker circular imports
os.environ["USE_JAX"] = "0"
try:
    import datasets.config

    datasets.config.JAX_AVAILABLE = False
except (ImportError, AttributeError):
    pass

import numpy as np
import torch
from datasets import Features, IterableDataset, Value, interleave_datasets, load_dataset
from huggingface_hub import HfFileSystem
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils.dataset import resolve_tokenizer_vocab_size, verify_tokenizer_vocab

TOPIC_WEIGHTS = {
    "general": 0.45,
    "coding": 0.15,
    "math": 0.08,
    "science": 0.05,
    "agents": 0.08,
    "long_context": 0.07,
    "structured": 0.05,
    "safety": 0.07,
}


def _source(path, text_col="messages", name=None, split="train", **kwargs):
    return dict(path=path, name=name, split=split, text_col=text_col, **kwargs)


# Same path/name/split/text_col/weight schema as utils/dataset.py; topic and
# adapter identify conversions which cannot be expressed as a text-column pair.
TOPIC_SOURCES = {
    "general": [
        _source("HuggingFaceTB/smoltalk", name="all"),
        _source("teknium/OpenHermes-2.5", "conversations"),
        _source("OpenAssistant/oasst1", "text", adapter="oasst"),
        _source("HuggingFaceH4/no_robots"),
        _source("allenai/WildChat", "conversation"),
        _source("HuggingFaceH4/ultrachat_200k", split="train_sft"),
        _source("allenai/tulu-3-sft-mixture"),
    ],
    "coding": [
        _source("nvidia/OpenCodeInstruct", ["input", "output"], name="train"),
        _source("ise-uiuc/Magicoder-OSS-Instruct-75K", ["problem", "solution"]),
        _source(
            "microsoft/rStar-Coder",
            ["question", "response"],
            name="seed_sft",
            adapter="rstar",
            extra_names=["synthetic_sft"],
        ),
        _source(
            "codeparrot/apps",
            ["question", "solutions"],
            adapter="apps",
            revision="refs/convert/parquet",
            data_files="all/train/*.parquet",
        ),
    ],
    "math": [
        _source("open-r1/OpenR1-Math-220k", name="default"),
        _source(
            "open-r1/OpenThoughts-114k-math", "conversations", adapter="correct_math"
        ),
        _source("nvidia/OpenMathInstruct-2", ["problem", "generated_solution"]),
        _source("oumi-ai/MetaMathQA-R1"),
        _source("bespokelabs/Bespoke-Stratos-17k", "conversations"),
    ],
    "science": [
        _source("open-thoughts/OpenThoughts-114k", "conversations", name="default"),
        _source("allenai/SciRIFF", ["input", "output"], name="4096"),
    ],
    "agents": [
        _source("open-thoughts/OpenThoughts-Agent-SFT-100K", "conversations"),
        _source("Team-ACE/ToolACE", "conversations"),
        _source("HuggingFaceTB/smoltalk", name="apigen-80k"),
    ],
    "long_context": [_source("zai-org/LongAlign-10k")],
    "structured": [
        _source("HuggingFaceTB/smoltalk", name="apigen-80k"),
        _source("HuggingFaceTB/smoltalk", name="smol-constraints"),
        _source("Team-ACE/ToolACE", "conversations"),
    ],
    # Read individual JSONL objects: heterogeneous unused metadata currently
    # breaks Arrow schema inference on this source's normal load_dataset path.
    "safety": [
        _source(
            "nvidia/Nemotron-SFT-Safety-v2",
            adapter="jsonl",
            data_file="data/train.jsonl",
        )
    ],
}


def get_dataset_configs(
    exclude_topics: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return independently editable configs with absolute sampling weights.

    If exclude_topics is specified, those topics are excluded and the remaining
    topic weights are re-normalized so they sum to 1.0.
    """
    exclude = set(exclude_topics or [])
    filtered_weights = {k: v for k, v in TOPIC_WEIGHTS.items() if k not in exclude}
    if not filtered_weights:
        raise ValueError("Cannot exclude all topics from dataset configs")
    total_weight = sum(filtered_weights.values())
    normalized_topic_weights = {
        k: v / total_weight for k, v in filtered_weights.items()
    }

    configs = []
    for topic, sources in TOPIC_SOURCES.items():
        if topic in exclude:
            continue
        for source in sources:
            cfg = deepcopy(source)
            names = [cfg["name"]] + cfg.pop("extra_names", [])
            for name in names:
                configs.append(
                    dict(
                        cfg,
                        name=name,
                        topic=topic,
                        weight=normalized_topic_weights[topic]
                        / len(sources)
                        / len(names),
                    )
                )
    return configs


DATASET_CONFIGS = get_dataset_configs()
IGNORE_INDEX = -100
_ROLES = {
    "human": "user",
    "prompter": "user",
    "gpt": "assistant",
    "model": "assistant",
    "bot": "assistant",
    "function": "tool",
    "observation": "tool",
}
_FEATURES = Features({"conversation_json": Value("string")})


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def extract_messages_from_sample(
    sample: dict[str, Any], cfg: dict[str, Any]
) -> list[dict]:
    """Normalize a source row without ever using metadata/test cases as answers.

    Empty/incomplete rows return []; unknown roles/schema errors raise. Tool-call
    payloads, identifiers, reasoning and tool definitions survive normalization.
    """
    if cfg.get("adapter") == "correct_math" and sample.get("correct") is not True:
        return []
    col = cfg["text_col"]
    if isinstance(col, list):
        prompt, answer = (_text(sample.get(col[0])), sample.get(col[1]))
        if cfg.get("adapter") == "apps":
            solutions = (
                json.loads(answer) if isinstance(answer, str) and answer else answer
            )
            if not isinstance(solutions, list):
                raise ValueError("APPS solutions must be a JSON list")
            answer = next(
                (s for s in solutions if isinstance(s, str) and s.strip()), ""
            )
        if cfg.get("adapter") in {"apps", "rstar"} and sample.get("starter_code"):
            prompt += "\n\nStarter code:\n" + _text(sample["starter_code"])
        if not prompt.strip() or not _text(answer).strip():
            return []
        raw = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": _text(answer)},
        ]
    else:
        if col not in sample:
            raise ValueError(f"Missing conversation column {col!r}")
        raw = sample[col]
        if isinstance(raw, str):
            raw = json.loads(raw)
    if not isinstance(raw, list):
        raise TypeError("Conversation must be a list of messages")
    messages = []
    for msg in raw:
        if not isinstance(msg, dict):
            raise TypeError("Each message must be an object")
        role = msg.get("role", msg.get("from", ""))
        role = _ROLES.get(role, role)
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role {role!r}")
        content = _text(msg.get("content", msg.get("value")))
        # JSON is retained as JSON, never Python repr or executed code.
        for key in (
            "reasoning_content",
            "tool_calls",
            "function_call",
            "name",
            "tool_call_id",
        ):
            if msg.get(key) is not None:
                content += f"\n<{key}>" + _text(msg[key]) + f"</{key}>"
        if not content.strip():
            return []
        messages.append({"role": role, "content": content})
    system = sample.get("system") or sample.get("system_prompt")
    if system and (not messages or messages[0]["role"] != "system"):
        messages.insert(0, {"role": "system", "content": _text(system)})
    if sample.get("tools"):
        messages.insert(
            0, {"role": "system", "content": "Tools:\n" + _text(sample["tools"])}
        )
    # Keep trailing observations out of the target and avoid prompt-only rows.
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if not any(m["role"] == "user" for m in messages):
        return []
    return messages


def _oasst_conversations(rows) -> Iterator[list[dict]]:
    """Join the train split by ID, independent of incoming order (about 84k rows).

    This one small source is indexed in RAM. Emit one full path per assistant
    leaf; missing/deleted/rejected ancestors invalidate the entire path.
    """
    valid = {
        row["message_id"]: {
            "message_id": row["message_id"],
            "parent_id": row.get("parent_id"),
            "role": row.get("role", ""),
            "text": _text(row.get("text")),
        }
        for row in rows
        if not row.get("deleted")
        and row.get("review_result") is not False
        and _text(row.get("text")).strip()
    }
    paths = []
    ancestors = set()
    for key, row in valid.items():
        if row["role"] != "assistant":
            continue
        path, seen, current = [], set(), key
        while current is not None:
            if current in seen or current not in valid:
                path = []
                break
            seen.add(current)
            node = valid[current]
            path.append(node)
            current = node.get("parent_id")
        if path:
            path.reverse()
            paths.append((key, path))
            ancestors.update(n["message_id"] for n in path[:-1])
    for key, path in paths:
        if key not in ancestors:
            yield [
                {"role": _ROLES.get(n["role"], n["role"]), "content": n["text"]}
                for n in path
            ]


def _source_rows(cfg):
    if cfg.get("adapter") == "jsonl":
        fs = HfFileSystem()
        revision = cfg.get("revision", "main")
        path = f"datasets/{cfg['path']}@{revision}/{cfg['data_file']}"
        with fs.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        return
    kwargs = {"split": cfg["split"], "streaming": True}
    for key in ("name", "revision", "data_files"):
        if cfg.get(key) is not None:
            kwargs[key] = cfg[key]
    yield from load_dataset(cfg["path"], **kwargs)


def _normalized_rows(cfg):
    try:
        rows = _source_rows(cfg)
        if cfg.get("adapter") == "oasst":
            rows = ({"messages": messages} for messages in _oasst_conversations(rows))
            normalize_cfg = dict(cfg, text_col="messages")
        else:
            normalize_cfg = cfg
        accepted = 0
        for row in rows:
            messages = extract_messages_from_sample(row, normalize_cfg)
            if messages:
                accepted += 1
                yield {"conversation_json": json.dumps(messages, ensure_ascii=False)}
        if not accepted:
            raise ValueError("Source has no valid SFT conversations")
    except Exception as exc:
        raise RuntimeError(
            f"SFT source {cfg['path']} ({cfg.get('name')}, {cfg['split']}): {exc}"
        ) from exc


def tokenize_messages(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Encode the explicit role-text format; supervise assistant bodies and EOS."""
    ids, mask = [], []
    if tokenizer.bos_token_id is not None:
        ids.append(tokenizer.bos_token_id)
        mask.append(0)
    for index, msg in enumerate(messages):
        header = ("\n" if index else "") + msg["role"] + ":\n"
        header_ids = tokenizer.encode(header, add_special_tokens=False)
        body_ids = tokenizer.encode(msg["content"], add_special_tokens=False)
        assistant = msg["role"] == "assistant"
        ids.extend(header_ids + body_ids)
        mask.extend([0] * len(header_ids) + [int(assistant)] * len(body_ids))
        if assistant:
            ids.append(tokenizer.eos_token_id)
            mask.append(1)
    return ids, mask


class TokenizedShardProducer:
    """Bounded, resumable CPU producer with the pretraining producer interface."""

    def __init__(
        self,
        cache_dir: str,
        tokenizer_name: str = "UIC-AI-lab/llama2-tokenizer",
        tokens_per_shard: int | None = None,
        max_buffered_files: int = 3,
        seed: int | None = 0,
        log_fn: Callable[[str], None] | None = None,
        *,
        seq_len: int = 65537,
        dataset_configs: list[dict] | None = None,
        tokenizer=None,
        expected_vocab_size: int | None = None,
        oversized_behavior: str = "filter",
    ):
        if seq_len < 2 or max_buffered_files < 1:
            raise ValueError("seq_len must be >=2 and max_buffered_files >=1")
        if oversized_behavior not in {"filter", "truncate", "error"}:
            raise ValueError(
                f"Invalid oversized_behavior {oversized_behavior!r}; expected 'filter', 'truncate', or 'error'"
            )
        self.seq_len = seq_len
        self.oversized_behavior = oversized_behavior
        self.skipped_oversized_samples = 0
        self.tokens_per_shard = (
            tokens_per_shard if tokens_per_shard is not None else seq_len * 8
        )
        if self.tokens_per_shard < seq_len or self.tokens_per_shard % seq_len:
            raise ValueError("tokens_per_shard must be a positive multiple of seq_len")
        self.cache_dir = os.fspath(cache_dir)
        self.tokenizer_name = tokenizer_name
        self.max_buffered_files = max_buffered_files
        self.seed = 0 if seed is None else seed
        self.log = log_fn if log_fn is not None else lambda msg: print(msg, flush=True)
        self.dataset_configs = deepcopy(
            DATASET_CONFIGS if dataset_configs is None else dataset_configs
        )
        weights = [cfg["weight"] for cfg in self.dataset_configs]
        if not weights or any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("Every source must have a positive finite weight")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-10):
            raise ValueError("Dataset sampling weights must sum to 1")
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        )
        if self.tokenizer.eos_token_id is None:
            raise ValueError("SFT requires a tokenizer with an EOS token")
        if expected_vocab_size is not None:
            verify_tokenizer_vocab(self.tokenizer, expected_vocab_size)
        self.vocab_size = expected_vocab_size or resolve_tokenizer_vocab_size(
            self.tokenizer
        )
        if self.vocab_size > np.iinfo(np.uint32).max + 1:
            raise ValueError("Tokenizer vocabulary exceeds uint32 storage")
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        self.current_shard_idx = 0
        self.cumulative_samples = 0
        self.token_buffer: list[int] = []
        self.loss_mask_buffer: list[int] = []
        self.error: BaseException | None = None
        self.finished = False
        self._lock = threading.RLock()
        os.makedirs(self.cache_dir, exist_ok=True)

    def _settings(self):
        vocab = self.tokenizer.get_vocab()
        return {
            "format": "sft-role-text-v1",
            "seq_len": self.seq_len,
            "tokens_per_shard": self.tokens_per_shard,
            "seed": self.seed,
            "tokenizer_name": self.tokenizer_name,
            "vocab_size": self.vocab_size,
            "vocab_hash": hashlib.sha256(
                json.dumps(vocab, sort_keys=True).encode()
            ).hexdigest(),
            "bos_id": self.tokenizer.bos_token_id,
            "eos_id": self.tokenizer.eos_token_id,
            "pad_id": self.pad_id,
            "dataset_configs": self.dataset_configs,
            "oversized_behavior": self.oversized_behavior,
        }

    def save_checkpoint(self, checkpoint_path: str):
        with self._lock:
            state = {
                "settings": self._settings(),
                "current_shard_idx": self.current_shard_idx,
                "cumulative_samples": self.cumulative_samples,
                "skipped_oversized_samples": self.skipped_oversized_samples,
                "finished": self.finished,
                "token_buffer_b64": base64.b64encode(
                    np.asarray(self.token_buffer, dtype="<u4").tobytes()
                ).decode("ascii"),
                "loss_mask_b64": base64.b64encode(bytes(self.loss_mask_buffer)).decode(
                    "ascii"
                ),
            }
            self._atomic_json(checkpoint_path, state)

    def load_checkpoint(self, checkpoint_path: str):
        with self._lock:
            with open(checkpoint_path, encoding="utf-8") as handle:
                state = json.load(handle)
            loaded_settings = dict(state.get("settings", {}))
            current_settings = self._settings()
            if "oversized_behavior" not in loaded_settings:
                loaded_settings["oversized_behavior"] = self.oversized_behavior
            if loaded_settings != current_settings:
                raise ValueError("SFT checkpoint settings/tokenizer/mix do not match")
            tokens = np.frombuffer(
                base64.b64decode(state["token_buffer_b64"], validate=True), dtype="<u4"
            ).tolist()
            mask = list(base64.b64decode(state["loss_mask_b64"], validate=True))
            if len(tokens) != len(mask) or any(m not in (0, 1) for m in mask):
                raise ValueError("Corrupt SFT checkpoint mask")
            self._validate_tokens(tokens)
            self.token_buffer, self.loss_mask_buffer = tokens, mask
            self.current_shard_idx = state["current_shard_idx"]
            self.cumulative_samples = state["cumulative_samples"]
            self.skipped_oversized_samples = state.get("skipped_oversized_samples", 0)
            self.finished = state["finished"]
        self.log(
            f"[SFT Producer] Restored {self.cumulative_samples} conversations; deterministic replay required"
        )

    @staticmethod
    def _atomic_json(path, value):
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def _build_stream(self):
        streams = [
            IterableDataset.from_generator(
                _normalized_rows, gen_kwargs={"cfg": cfg}, features=_FEATURES
            )
            for cfg in self.dataset_configs
        ]
        return interleave_datasets(
            streams,
            probabilities=[c["weight"] for c in self.dataset_configs],
            seed=self.seed,
            stopping_strategy="all_exhausted",
        )

    def _cleanup_consumed_shards(self):
        for sentinel in glob.glob(os.path.join(self.cache_dir, "shard_*.done")):
            base = sentinel[:-5]
            try:
                for suffix in (
                    ".bin",
                    ".mask",
                    ".json",
                    ".bin.tmp",
                    ".mask.tmp",
                    ".json.tmp",
                ):
                    path = base + suffix
                    if os.path.exists(path):
                        os.remove(path)
                os.remove(sentinel)
            except OSError:
                # Windows mmap handles can delay removal until the next cycle.
                pass

    def _validate_tokens(self, ids):
        if ids and (min(ids) < 0 or max(ids) >= self.vocab_size):
            raise ValueError("Token ID outside the configured model vocabulary")

    def _pad_window(self):
        count = (-len(self.token_buffer)) % self.seq_len
        self.token_buffer.extend([self.pad_id] * count)
        self.loss_mask_buffer.extend([0] * count)

    def _append_conversation(self, messages):
        ids, mask = tokenize_messages(self.tokenizer, messages)
        self._validate_tokens(ids)
        if len(ids) > self.seq_len:
            if self.oversized_behavior == "error":
                raise ValueError(
                    f"SFT conversation has {len(ids)} tokens, exceeding seq_len={self.seq_len}. "
                    "Increase seq_len within the model's supported context; no silent truncation is applied."
                )
            elif self.oversized_behavior == "truncate":
                ids = ids[: self.seq_len]
                mask = mask[: self.seq_len]
                if not any(mask[1:]):
                    with self._lock:
                        self.cumulative_samples += 1
                        self.skipped_oversized_samples += 1
                    return
            elif self.oversized_behavior == "filter":
                with self._lock:
                    self.cumulative_samples += 1
                    self.skipped_oversized_samples += 1
                if (
                    self.skipped_oversized_samples <= 5
                    or self.skipped_oversized_samples % 100 == 0
                ):
                    self.log(
                        f"[SFT Producer] Skipped oversized conversation ({len(ids)} tokens > "
                        f"seq_len={self.seq_len}; total skipped: {self.skipped_oversized_samples})"
                    )
                return
            else:
                raise ValueError(
                    f"Unknown oversized_behavior: {self.oversized_behavior}"
                )
        if not any(mask[1:]):
            with self._lock:
                self.cumulative_samples += 1
            self.log(
                "[SFT Producer] Skipping conversation with no supervised next-token target"
            )
            return
        with self._lock:
            used = len(self.token_buffer) % self.seq_len
            if used and used + len(ids) > self.seq_len:
                self._pad_window()
            self.token_buffer.extend(ids)
            self.loss_mask_buffer.extend(mask)
            self.cumulative_samples += 1

    def _write_shard(self, count):
        # Publish .bin LAST, so the existing queue's *.bin polling is safe.
        with self._lock:
            base = os.path.join(self.cache_dir, f"shard_{self.current_shard_idx:06d}")
            if any(
                os.path.exists(base + suffix)
                for suffix in (".bin", ".mask", ".json", ".done")
            ):
                raise FileExistsError(
                    f"SFT shard already exists: {base}; restore a matching queue checkpoint"
                )
            for suffix, arr in (
                (".mask", np.asarray(self.loss_mask_buffer[:count], dtype=np.uint8)),
                (".bin", np.asarray(self.token_buffer[:count], dtype="<u4")),
            ):
                with open(base + suffix + ".tmp", "wb") as handle:
                    arr.tofile(handle)
                    handle.flush()
                    os.fsync(handle.fileno())
            self._atomic_json(
                base + ".json",
                {
                    "format": "sft-role-text-v1",
                    "dtype": "<u4",
                    "seq_len": self.seq_len,
                    "num_tokens": count,
                    "cumulative_samples_streamed_at_boundary": self.cumulative_samples,
                },
            )
            os.replace(base + ".mask.tmp", base + ".mask")
            os.replace(base + ".bin.tmp", base + ".bin")
            del self.token_buffer[:count]
            del self.loss_mask_buffer[:count]
            self.current_shard_idx += 1

    def _flush(self, stop_event, final=False):
        while len(self.token_buffer) >= self.tokens_per_shard or (
            final and self.token_buffer
        ):
            self._cleanup_consumed_shards()
            while (
                len(glob.glob(os.path.join(self.cache_dir, "shard_*.bin")))
                >= self.max_buffered_files
            ):
                if stop_event is not None and stop_event.is_set():
                    return False
                time.sleep(0.1)
                self._cleanup_consumed_shards()
            if stop_event is not None and stop_event.is_set():
                return False
            self._write_shard(min(len(self.token_buffer), self.tokens_per_shard))
        return True

    def start_streaming(self, stop_event=None, checkpoint_path: str | None = None):
        """Thread target; polls stop even during backpressure and exposes errors."""
        try:
            self.error = None
            if checkpoint_path and os.path.exists(checkpoint_path):
                self.load_checkpoint(checkpoint_path)
            if self.finished:
                return
            if not self._flush(stop_event):
                return
            stream = itertools.islice(
                iter(self._build_stream()), self.cumulative_samples, None
            )
            while stop_event is None or not stop_event.is_set():
                try:
                    sample = next(stream)
                except StopIteration:
                    with self._lock:
                        self._pad_window()
                    if self._flush(stop_event, final=True):
                        self.finished = True
                    break
                self._append_conversation(json.loads(sample["conversation_json"]))
                if not self._flush(stop_event):
                    break
        except BaseException as exc:
            self.error = exc
            self.log(f"[SFT Producer] FAILED: {exc!r}")
            raise
        finally:
            # Failed conversion hasn't incremented cumulative_samples: retry is
            # safe after fixing the cause, without hiding or dropping that row.
            if checkpoint_path and self.error is None:
                self.save_checkpoint(checkpoint_path)


class MmapShardDataset(Dataset):
    """SFT companion to the pretraining consumer; labels are already shifted.

    Each item has seq_len-1 positions. Pass labels directly to this repository's
    model (do not shift again). Close all readers before marking a shard .done.
    """

    def __init__(self, bin_path: str, seq_len: int):
        self.bin_path = os.fspath(bin_path)
        self.seq_len = seq_len
        base = os.path.splitext(self.bin_path)[0]
        self.mask_path = base + ".mask"
        with open(base + ".json", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if (
            metadata.get("format") != "sft-role-text-v1"
            or metadata.get("dtype") != "<u4"
        ):
            raise ValueError(
                "Not an SFT shard; use the matching pretraining/SFT reader"
            )
        if seq_len < 2 or metadata["seq_len"] != seq_len:
            raise ValueError("Consumer seq_len must equal producer seq_len")
        count = metadata["num_tokens"]
        if count <= 0 or count % seq_len or os.path.getsize(self.bin_path) != count * 4:
            raise ValueError("Truncated or unaligned SFT token shard")
        if os.path.getsize(self.mask_path) != count:
            raise ValueError("SFT mask/token size mismatch")
        self.num_sequences = count // seq_len
        self._data = None
        self._mask = None

    @property
    def data(self):
        if self._data is None:
            self._data = np.memmap(self.bin_path, dtype="<u4", mode="r")
        return self._data

    @property
    def mask(self):
        if self._mask is None:
            self._mask = np.memmap(self.mask_path, dtype=np.uint8, mode="r")
        return self._mask

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        start = idx * self.seq_len
        end = start + self.seq_len
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        target_mask = torch.from_numpy(self.mask[start + 1 : end].astype(np.bool_))
        labels = chunk[1:].clone()
        labels[~target_mask] = IGNORE_INDEX
        return chunk[:-1], labels

    def close(self):
        for name in ("_data", "_mask"):
            arr = getattr(self, name)
            if arr is not None:
                arr._mmap.close()
                setattr(self, name, None)

    def __getstate__(self):
        # Spawned DataLoader workers open their own lazy mmap handles.
        return dict(self.__dict__, _data=None, _mask=None)


SFTTokenizedShardProducer = TokenizedShardProducer
SFTMmapShardDataset = MmapShardDataset
