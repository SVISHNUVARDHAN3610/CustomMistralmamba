"""Offline SFT data contracts; run with the training dependencies installed."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
import threading
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from utils.sft_dataset import (
    DATASET_CONFIGS,
    TOPIC_WEIGHTS,
    MmapShardDataset,
    TokenizedShardProducer,
    _normalized_rows,
    _oasst_conversations,
    extract_messages_from_sample,
    get_dataset_configs,
    tokenize_messages,
)


class TinyTokenizer:
    """Character tokenizer with one >uint16 token to catch silent overflow."""

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 3
    name_or_path = "test-tokenizer"

    def __len__(self):
        return 100_000

    def get_vocab(self):
        return {"high": 70000, "eos": 2}

    def encode(self, text, add_special_tokens=False):
        return [70000 if char == "~" else ord(char) + 10 for char in text]


def conversation(prompt="Q", answer="A"):
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]


def stream_rows(messages):
    return [{"conversation_json": json.dumps(m)} for m in messages]


class TestSFTAdapters(unittest.TestCase):
    def test_topic_and_source_weights(self):
        topics = defaultdict(float)
        paths = defaultdict(float)
        for cfg in DATASET_CONFIGS:
            topics[cfg["topic"]] += cfg["weight"]
            paths[cfg["topic"], cfg["path"]] += cfg["weight"]
            self.assertIn(cfg["split"], {"train", "train_sft"})
        self.assertEqual(set(topics), set(TOPIC_WEIGHTS))
        for topic, weight in TOPIC_WEIGHTS.items():
            self.assertAlmostEqual(topics[topic], weight)
        self.assertAlmostEqual(sum(topics.values()), 1)
        self.assertAlmostEqual(paths["coding", "microsoft/rStar-Coder"], 0.15 / 4)
        configs = get_dataset_configs()
        configs[0]["weight"] = 0
        self.assertNotEqual(DATASET_CONFIGS[0]["weight"], 0)

    def test_every_declared_adapter_has_an_offline_fixture(self):
        for cfg in DATASET_CONFIGS:
            if cfg.get("adapter") == "oasst":
                continue
            col = cfg["text_col"]
            if isinstance(col, list):
                row = dict(
                    zip(
                        col,
                        [
                            "Question",
                            '["print(1)"]'
                            if cfg.get("adapter") == "apps"
                            else "Answer",
                        ],
                    )
                )
            else:
                row = {
                    col: [
                        {"from": "human", "value": "Question"},
                        {"from": "gpt", "value": "Answer"},
                    ]
                }
            row["correct"] = True
            with self.subTest(source=cfg["path"], name=cfg["name"]):
                messages = extract_messages_from_sample(row, cfg)
                self.assertEqual(messages[0]["role"], "user")
                self.assertEqual(messages[-1]["role"], "assistant")

    def test_tool_context_and_calls_are_retained(self):
        row = {
            "system": "Use tools",
            "tools": [{"name": "lookup"}],
            "conversations": [
                {"from": "human", "value": "Find it"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call1",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": {"found": True}, "tool_call_id": "call1"},
                {"from": "gpt", "value": "Found"},
            ],
        }
        messages = extract_messages_from_sample(row, {"text_col": "conversations"})
        self.assertIn("lookup", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "Use tools")
        self.assertIn('"arguments": "{}"', messages[3]["content"])
        self.assertIn("call1", messages[4]["content"])
        self.assertEqual(messages[4]["role"], "tool")

    def test_apps_selects_solution_and_preserves_starter_without_test_answers(self):
        cfg = next(c for c in DATASET_CONFIGS if c["path"] == "codeparrot/apps")
        row = {
            "question": "Solve",
            "solutions": '["", "print(1)", "print(2)"]',
            "starter_code": "def solve():",
            "input_output": "SECRET TESTS",
        }
        messages = extract_messages_from_sample(row, cfg)
        self.assertIn("def solve():", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "print(1)")
        self.assertNotIn("SECRET", json.dumps(messages))
        row["solutions"] = "[]"
        self.assertEqual(extract_messages_from_sample(row, cfg), [])

    def test_incorrect_math_and_prompt_only_rows_are_not_targets(self):
        cfg = next(c for c in DATASET_CONFIGS if c.get("adapter") == "correct_math")
        self.assertEqual(extract_messages_from_sample({"correct": False}, cfg), [])
        self.assertEqual(
            extract_messages_from_sample(
                {"messages": conversation()[:1]}, {"text_col": "messages"}
            ),
            [],
        )
        with self.assertRaisesRegex(ValueError, "Missing conversation"):
            extract_messages_from_sample({}, {"text_col": "messages"})

    def test_oasst_handles_out_of_order_branches_and_missing_ancestors(self):
        def node(key, parent, role, text, **kwargs):
            return dict(
                message_id=key, parent_id=parent, role=role, text=text, **kwargs
            )

        rows = [
            node("a2", "u2", "assistant", "second"),
            node("bad", "missing", "assistant", "orphan"),
            node("a1", "u1", "assistant", "first"),
            node("u2", "a1", "prompter", "followup"),
            node("alt", "u1", "assistant", "alternative"),
            node("u1", None, "prompter", "question"),
            node("deleted", "u1", "assistant", "deleted", deleted=True),
        ]
        paths = list(_oasst_conversations(rows))
        self.assertEqual(len(paths), 2)
        self.assertEqual(
            [m["content"] for m in paths[0]],
            ["question", "first", "followup", "second"],
        )
        self.assertEqual(paths[1][-1]["content"], "alternative")

    def test_empty_source_fails_with_source_name(self):
        cfg = dict(DATASET_CONFIGS[0])
        with (
            patch("utils.sft_dataset._source_rows", return_value=iter([])),
            self.assertRaisesRegex(RuntimeError, "smoltalk.*no valid"),
        ):
            list(_normalized_rows(cfg))

    def test_only_assistant_bodies_and_eos_are_supervised(self):
        tokenizer = TinyTokenizer()
        messages = (
            [{"role": "system", "content": "Policy"}]
            + conversation("Question", "~")
            + [
                {"role": "tool", "content": "Observation"},
                {"role": "assistant", "content": "Done"},
            ]
        )
        ids, mask = tokenize_messages(tokenizer, messages)
        supervised = [token for token, keep in zip(ids, mask) if keep]
        self.assertEqual(supervised, [70000, 2] + tokenizer.encode("Done") + [2])
        self.assertEqual(mask[0], 0)


class TestSFTShards(unittest.TestCase):
    def producer(self, directory, **kwargs):
        return TokenizedShardProducer(
            directory,
            tokenizer=TinyTokenizer(),
            seq_len=64,
            tokens_per_shard=128,
            max_buffered_files=100,
            log_fn=lambda msg: None,
            **kwargs,
        )

    def test_binary_roundtrip_shift_mask_and_partial_final_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(directory)
            producer._build_stream = lambda: stream_rows([conversation("Q", "~")])
            producer.start_streaming()
            self.assertTrue(producer.finished)
            path = os.path.join(directory, "shard_000000.bin")
            reader = MmapShardDataset(path, 64)
            try:
                inputs, labels = reader[0]
                self.assertEqual(len(inputs), 63)
                self.assertEqual(labels[labels != -100].tolist(), [70000, 2])
                self.assertEqual(int(inputs[0]), 1)
                worker_reader = pickle.loads(pickle.dumps(reader))
                self.assertIsNone(worker_reader._data)
                self.assertEqual(worker_reader[0][1].tolist(), labels.tolist())
                worker_reader.close()
                with self.assertRaises(IndexError):
                    reader[1]
            finally:
                reader.close()
            Path(directory, "shard_000000.done").touch()
            producer._cleanup_consumed_shards()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_conversations_never_cross_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(directory)
            messages = [
                conversation("q" * 15, "a" * 10),
                conversation("x" * 15, "b" * 10),
            ]
            producer._build_stream = lambda: stream_rows(messages)
            producer.start_streaming()
            reader = MmapShardDataset(os.path.join(directory, "shard_000000.bin"), 64)
            try:
                self.assertEqual(len(reader), 2)
                for i, answer in enumerate(["a", "b"]):
                    inputs, labels = reader[i]
                    self.assertEqual(int(inputs[0]), 1)
                    self.assertEqual(
                        labels[labels != -100].tolist(),
                        TinyTokenizer().encode(answer * 10) + [2],
                    )
            finally:
                reader.close()

    def test_resume_matches_uninterrupted_bytes(self):
        messages = [conversation(str(i), "answer" * 2) for i in range(8)]
        with (
            tempfile.TemporaryDirectory() as full,
            tempfile.TemporaryDirectory() as resumed,
        ):
            reference = self.producer(full)
            reference._build_stream = lambda: stream_rows(messages)
            reference.start_streaming()
            first = self.producer(resumed)
            stop = threading.Event()

            def interrupted():
                for index, row in enumerate(stream_rows(messages)):
                    if index == 2:
                        stop.set()
                    yield row

            first._build_stream = interrupted
            checkpoint = os.path.join(resumed, "checkpoint.json")
            first.start_streaming(stop, checkpoint)
            second = self.producer(resumed)
            second._build_stream = lambda: stream_rows(messages)
            second.start_streaming(checkpoint_path=checkpoint)
            self.assertTrue(second.finished)
            self.assertEqual(second.cumulative_samples, len(messages))
            for file in Path(full).iterdir():
                self.assertEqual(
                    file.read_bytes(), Path(resumed, file.name).read_bytes(), file.name
                )

    def test_stop_during_backpressure_preserves_pending_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(directory)
            producer.max_buffered_files = 1
            Path(directory, "shard_999999.bin").touch()
            producer._append_conversation(conversation())
            producer._pad_window()
            producer._append_conversation(conversation())
            producer._pad_window()
            stop = threading.Event()
            stop.set()
            before = producer.token_buffer[:]
            self.assertFalse(producer._flush(stop))
            self.assertEqual(producer.token_buffer, before)
            self.assertEqual(producer.current_shard_idx, 0)

    def test_oversize_and_out_of_vocab_are_explicit_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(directory)
            with self.assertRaisesRegex(ValueError, "exceeding seq_len"):
                producer._append_conversation(conversation("q" * 100))
            self.assertEqual(producer.cumulative_samples, 0)
            producer.vocab_size = 1000
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                producer._append_conversation(conversation(answer="~"))

    def test_checkpoint_incompatible_settings_and_corruption_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(directory)
            producer._append_conversation(conversation(answer="~"))
            path = os.path.join(directory, "state.json")
            producer.save_checkpoint(path)
            other = self.producer(directory, seed=91)
            with self.assertRaisesRegex(ValueError, "do not match"):
                other.load_checkpoint(path)
            state = json.loads(Path(path).read_text())
            state["loss_mask_b64"] = ""
            Path(path).write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                producer.load_checkpoint(path)

    def test_reader_detects_truncated_mask_and_wrong_sequence_length(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(directory)
            producer._build_stream = lambda: stream_rows([conversation()])
            producer.start_streaming()
            path = os.path.join(directory, "shard_000000.bin")
            with self.assertRaisesRegex(ValueError, "seq_len"):
                MmapShardDataset(path, 32)
            np.asarray([0], dtype=np.uint8).tofile(
                os.path.join(directory, "shard_000000.mask")
            )
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                MmapShardDataset(path, 64)

    def test_mixture_interleaves_normalized_uniform_schema_reproducibly(self):
        cfgs = [
            dict(DATASET_CONFIGS[0], weight=0.75),
            dict(DATASET_CONFIGS[1], weight=0.25),
        ]
        rows = [{"messages": conversation(), "conversations": conversation()}] * 10
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(directory, dataset_configs=cfgs)
            with patch(
                "utils.sft_dataset._source_rows", side_effect=lambda cfg: iter(rows)
            ):
                first = list(producer._build_stream())
                second = list(producer._build_stream())
                self.assertEqual(first, second)
                self.assertGreaterEqual(len(first), 20)
                self.assertEqual(set(first[0]), {"conversation_json"})


if __name__ == "__main__":
    unittest.main()
