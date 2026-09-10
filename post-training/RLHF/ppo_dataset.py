"""Prompt-only storage adapter on the existing RLHF/SFT bounded shard pipeline."""

import json
from functools import partial

import numpy as np
from datasets import IterableDataset

from RLHF.rlhf_dataset import (
    PreferenceShardDataset,
    PreferenceShardProducer,
    load_streaming_source,
)
from RLHF.train_reward_model import PreferenceFeed
from utils.sft_dataset import tokenize_messages


class PromptShardProducer(PreferenceShardProducer):
    """Reuse stream resume, bounded flush, publish, cleanup and thread ownership."""

    def __init__(self, *args, ppo, smoke=False, **kwargs):
        self.ppo, self.smoke = ppo, smoke
        super().__init__(*args, **kwargs)

    def _build_stream(self):
        if self.smoke:
            return IterableDataset.from_generator(
                lambda: ({"prompt": f"Q{i}?"} for i in range(32))
            )
        stream = load_streaming_source(
            {
                "path": self.ppo.prompt_dataset,
                "name": self.ppo.prompt_config,
                "split": self.ppo.prompt_split
                if self.purpose == "train"
                else self.ppo.evaluation_split,
                "revision": self.ppo.prompt_revision,
            }
        )
        if (
            stream.features is not None
            and self.ppo.prompt_column not in stream.features
        ):
            raise ValueError(f"Prompt dataset has no {self.ppo.prompt_column!r} column")
        return (
            stream.take(self.ppo.prompt_limit)
            if self.ppo.prompt_limit is not None
            else stream
        )

    def _encode_sample(self, sample):
        self.stats["raw"] += 1
        prompt = sample.get(self.ppo.prompt_column)
        if not isinstance(prompt, str) or not prompt.strip():
            self.stats["filtered"] += 1
            self.reasons["missing_prompt"] += 1
            return []
        ids, _ = tokenize_messages(
            self.tokenizer, [{"role": "user", "content": prompt.strip()}]
        )
        ids += self.tokenizer.encode("\nassistant:\n", add_special_tokens=False)
        if len(ids) > self.ppo.max_prompt_tokens:
            self.stats["filtered"] += 1
            self.reasons["oversize_prompt"] += 1
            return []
        if not ids or min(ids) < 0 or max(ids) >= self.vocab_size:
            raise ValueError("Invalid PPO prompt tokenizer output")
        self.stats["valid"] += 1
        return [ids]

    def _write_shard(self, count):
        tokens, offsets = [], [0]
        for ids in self.token_buffer[:count]:
            tokens.extend(ids)
            offsets.append(len(tokens))
        self._publish_shard(
            np.asarray(tokens, dtype="<u4"),
            {
                "format": "ppo-prompts-v1",
                "offsets": offsets,
                "raw_start": self.raw_start,
                "raw_end": self.cumulative_samples,
                "native_start": self.native_start,
                "native_end": self._native_ds_state,
            },
        )
        del self.token_buffer[:count]
        self.raw_start, self.native_start = (
            self.cumulative_samples,
            self._native_ds_state,
        )


class PromptShardDataset(PreferenceShardDataset):
    """Same lazy uint32 mmap/worker cleanup, with one sequence per observation."""

    def __init__(self, path):
        from pathlib import Path

        self.path = Path(path)
        self.metadata = json.loads(self.path.with_suffix(".json").read_text())
        self.offsets = self.metadata["offsets"]
        if (
            self.metadata.get("format") != "ppo-prompts-v1"
            or len(self.offsets) < 2
            or self.offsets[0] != 0
            or any(a >= b for a, b in zip(self.offsets, self.offsets[1:]))
            or self.path.stat().st_size != self.offsets[-1] * 4
        ):
            raise ValueError(f"Corrupt PPO prompt shard: {path}")
        self._data = None

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._data is None:
            self._data = np.memmap(self.path, dtype="<u4", mode="r")
        a, b = self.offsets[index : index + 2]
        return self._data[a:b].astype(np.int64)


def prompt_feed(cfg, tokenizer, backend, cursor=None, *, diagnostic=False, smoke=False):
    return PreferenceFeed(
        cfg,
        tokenizer,
        backend,
        "validation" if diagnostic else "train",
        cursor,
        producer_class=partial(PromptShardProducer, ppo=cfg.rlhf, smoke=smoke),
    )


def prompt_collator(rows):
    return [row.tolist() for row in rows]
