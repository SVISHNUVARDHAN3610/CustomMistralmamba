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

    def test_get_dataset_configs_exclude_topics(self):
        configs = get_dataset_configs(exclude_topics=["long_context"])
        self.assertFalse(any(c["topic"] == "long_context" for c in configs))
        self.assertAlmostEqual(sum(c["weight"] for c in configs), 1.0)
        with self.assertRaisesRegex(ValueError, "Cannot exclude all"):
            get_dataset_configs(exclude_topics=list(TOPIC_WEIGHTS.keys()))

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
            producer_filter = self.producer(directory, oversized_behavior="filter")
            producer_filter._append_conversation(conversation("q" * 100))
            self.assertEqual(producer_filter.skipped_oversized_samples, 1)
            self.assertEqual(producer_filter.cumulative_samples, 1)
            self.assertEqual(len(producer_filter.token_buffer), 0)

            producer_err = self.producer(directory, oversized_behavior="error")
            with self.assertRaisesRegex(ValueError, "exceeding seq_len"):
                producer_err._append_conversation(conversation("q" * 100))
            self.assertEqual(producer_err.cumulative_samples, 0)
            producer_err.vocab_size = 1000
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                producer_err._append_conversation(conversation(answer="~"))

    def test_oversize_truncate_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(directory, oversized_behavior="truncate")
            producer._append_conversation(conversation("q" * 100, "a" * 10))
            self.assertLessEqual(len(producer.token_buffer), 64)
            self.assertEqual(producer.cumulative_samples, 1)

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


class TestSFTConversationChunking(unittest.TestCase):
    def producer(self, directory, seq_len=513, **kwargs):
        defaults = {
            "seq_len": seq_len,
            "tokens_per_shard": seq_len * 4,
            "max_buffered_files": 100,
            "oversized_behavior": "chunk",
            "overlap_turns": 1,
            "max_chunks_per_conversation": 4,
            "min_assistant_tokens": 16,
            "min_chunk_tokens": 64,
            "log_fn": lambda msg: None,
        }
        defaults.update(kwargs)
        return TokenizedShardProducer(
            directory,
            tokenizer=TinyTokenizer(),
            **defaults,
        )

    def test_chunking_normal_conversation_not_oversized(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(directory, seq_len=513)
            messages = [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
            ]
            producer._append_conversation(messages)
            self.assertEqual(producer.stats["conversations_seen"], 1)
            self.assertEqual(producer.stats["conversations_kept_whole"], 1)
            self.assertEqual(producer.stats["conversations_chunked"], 0)
            self.assertEqual(producer.stats["conversations_filtered"], 0)
            self.assertEqual(producer.stats["chunks_created"], 0)
            self.assertEqual(producer.stats["chunks_emitted"], 1)
            self.assertEqual(producer.stats["chunks_dropped"], 0)
            self.assertGreater(producer.stats["tokens_retained"], 0)
            self.assertEqual(producer.stats["tokens_dropped"], 0)
            self.assertGreater(producer.stats["assistant_tokens_retained"], 0)
            self.assertLessEqual(len(producer.token_buffer), 513)

    def test_chunking_slightly_oversized_553_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(directory, seq_len=513, min_chunk_tokens=64)
            messages = [
                {"role": "user", "content": "u" * 180},
                {"role": "assistant", "content": "a" * 140},
                {"role": "user", "content": "u" * 95},
                {"role": "assistant", "content": "a" * 95},
            ]
            ids, _ = tokenize_messages(TinyTokenizer(), messages)
            self.assertGreater(len(ids), 513)
            self.assertLess(len(ids), 600)

            producer._append_conversation(messages)
            self.assertEqual(producer.stats["conversations_seen"], 1)
            self.assertEqual(producer.stats["conversations_kept_whole"], 0)
            self.assertEqual(producer.stats["conversations_chunked"], 1)
            self.assertEqual(producer.stats["chunks_created"], 2)
            self.assertEqual(producer.stats["chunks_emitted"], 2)
            self.assertEqual(producer.stats["chunks_dropped"], 0)
            self.assertGreater(producer.stats["assistant_tokens_retained"], 0)

            chunks = producer._chunk_conversation(messages)
            self.assertEqual(len(chunks), 2)
            for c_ids, c_mask in chunks:
                self.assertLessEqual(len(c_ids), 513)
                self.assertGreaterEqual(sum(c_mask), producer.min_assistant_tokens)

    def test_chunking_moderate_conversation_2733_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            logs = []
            producer = self.producer(
                directory,
                seq_len=513,
                max_chunks_per_conversation=4,
                log_fn=logs.append,
            )
            messages = []
            for i in range(10):
                messages.append(
                    {"role": "user", "content": f"User turn {i}: " + "u" * 110}
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Assistant turn {i}: " + "a" * 125,
                    }
                )

            ids, _ = tokenize_messages(TinyTokenizer(), messages)
            self.assertGreater(len(ids), 2500)
            self.assertLess(len(ids), 3000)

            producer._append_conversation(messages)
            self.assertEqual(producer.stats["conversations_chunked"], 1)
            self.assertGreater(producer.stats["chunks_created"], 4)
            self.assertEqual(producer.stats["chunks_emitted"], 4)
            self.assertEqual(
                producer.stats["chunks_dropped"],
                producer.stats["chunks_created"] - 4,
            )
            self.assertGreater(producer.stats["tokens_retained"], 0)
            self.assertGreater(producer.stats["tokens_dropped"], 0)
            self.assertGreater(producer.stats["assistant_tokens_retained"], 0)
            self.assertTrue(any("Capped chunks for conversation" in m for m in logs))
            self.assertTrue(
                any("[SFT Producer] Chunked conversation:" in m for m in logs)
            )

    def test_chunking_very_long_conversation_64545_tokens_capped(self):
        with tempfile.TemporaryDirectory() as directory:
            logs = []
            producer = self.producer(
                directory,
                seq_len=513,
                max_chunks_per_conversation=4,
                log_fn=logs.append,
            )
            messages = []
            for i in range(100):
                messages.append({"role": "user", "content": "u" * 300})
                messages.append({"role": "assistant", "content": "a" * 320})

            ids, _ = tokenize_messages(TinyTokenizer(), messages)
            self.assertGreater(len(ids), 60000)

            producer._append_conversation(messages)
            self.assertEqual(producer.stats["conversations_chunked"], 1)
            self.assertGreater(producer.stats["chunks_created"], 50)
            self.assertEqual(producer.stats["chunks_emitted"], 4)
            self.assertEqual(
                producer.stats["chunks_dropped"],
                producer.stats["chunks_created"] - 4,
            )
            self.assertTrue(any("Capped chunks for conversation" in m for m in logs))

    def test_chunking_oversized_assistant_message(self):
        with tempfile.TemporaryDirectory() as directory:
            logs = []
            producer = self.producer(
                directory,
                seq_len=513,
                min_assistant_tokens=16,
                log_fn=logs.append,
            )
            messages = [
                {"role": "user", "content": "Tell me a long story."},
                {"role": "assistant", "content": "w" * 1200},
            ]
            producer._append_conversation(messages)
            self.assertEqual(producer.stats["conversations_chunked"], 1)
            self.assertGreaterEqual(producer.stats["chunks_emitted"], 2)
            self.assertTrue(
                any(
                    "Message-level splitting required for oversized assistant message"
                    in m
                    for m in logs
                )
            )

            chunks = producer._chunk_conversation(messages)
            self.assertGreaterEqual(len(chunks), 2)
            for c_ids, c_mask in chunks:
                self.assertLessEqual(len(c_ids), 513)
                self.assertGreaterEqual(sum(c_mask), 16)
                self.assertIn(1, c_mask)

    def test_chunking_no_assistant_chunk_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(
                directory,
                seq_len=100,
                min_assistant_tokens=50,
                overlap_turns=0,
            )
            messages = [
                {"role": "user", "content": "u" * 10},
                {"role": "assistant", "content": "a" * 5},
                {"role": "user", "content": "u" * 10},
                {"role": "assistant", "content": "a" * 60},
            ]
            chunks = producer._chunk_conversation(messages)
            for _, c_mask in chunks:
                self.assertGreaterEqual(sum(c_mask), 50)
            self.assertGreaterEqual(producer.stats["chunks_dropped"], 1)

    def test_chunking_oasst1_tree_reconstruction_then_chunking(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(directory, seq_len=200)
            raw_tree = [
                {
                    "message_id": "m1",
                    "parent_id": None,
                    "role": "user",
                    "text": "Question 1: " + "q" * 80,
                },
                {
                    "message_id": "m2",
                    "parent_id": "m1",
                    "role": "assistant",
                    "text": "Answer 1: " + "a" * 80,
                },
                {
                    "message_id": "m3",
                    "parent_id": "m2",
                    "role": "user",
                    "text": "Question 2: " + "q" * 80,
                },
                {
                    "message_id": "m4",
                    "parent_id": "m3",
                    "role": "assistant",
                    "text": "Answer 2: " + "a" * 80,
                },
            ]
            reconstructed = list(_oasst_conversations(raw_tree))
            self.assertEqual(len(reconstructed), 1)
            self.assertEqual(len(reconstructed[0]), 4)
            self.assertEqual(reconstructed[0][0]["role"], "user")
            self.assertEqual(reconstructed[0][-1]["role"], "assistant")

            producer._append_conversation(reconstructed[0])
            self.assertEqual(producer.stats["conversations_chunked"], 1)
            self.assertEqual(producer.stats["chunks_emitted"], 2)
            for c_ids, c_mask in producer._chunk_conversation(reconstructed[0]):
                self.assertLessEqual(len(c_ids), 200)
                self.assertGreater(sum(c_mask), 0)

    def test_chunking_overlap_turns_boundary(self):
        messages = [
            {"role": "user", "content": "Prompt 0: " + "x" * 30},
            {"role": "assistant", "content": "Reply 0: " + "y" * 30},
            {"role": "user", "content": "Prompt 1: " + "x" * 30},
            {"role": "assistant", "content": "Reply 1: " + "y" * 30},
            {"role": "user", "content": "Prompt 2: " + "x" * 30},
            {"role": "assistant", "content": "Reply 2: " + "y" * 30},
        ]
        with tempfile.TemporaryDirectory() as directory:
            prod_overlap = self.producer(
                directory, seq_len=250, overlap_turns=1, min_chunk_tokens=30
            )
            chunks_overlap = prod_overlap._chunk_conversation(messages)
            self.assertEqual(len(chunks_overlap), 2)

            prod_no_overlap = self.producer(
                directory, seq_len=250, overlap_turns=0, min_chunk_tokens=30
            )
            chunks_no_overlap = prod_no_overlap._chunk_conversation(messages)
            self.assertEqual(len(chunks_no_overlap), 2)

            for c_ids, _ in chunks_overlap:
                self.assertLessEqual(len(c_ids), 250)
            for c_ids, _ in chunks_no_overlap:
                self.assertLessEqual(len(c_ids), 250)

            tokens_overlap = sum(len(c[0]) for c in chunks_overlap)
            tokens_no_overlap = sum(len(c[0]) for c in chunks_no_overlap)
            self.assertGreater(tokens_overlap, tokens_no_overlap)

    def test_chunking_deterministic_selection(self):
        messages = []
        for i in range(12):
            messages.append({"role": "user", "content": f"Turn {i} user " + "u" * 60})
            messages.append(
                {"role": "assistant", "content": f"Turn {i} ast " + "a" * 60}
            )

        with (
            tempfile.TemporaryDirectory() as dir1,
            tempfile.TemporaryDirectory() as dir2,
            tempfile.TemporaryDirectory() as dir3,
        ):
            p1 = self.producer(
                dir1, seq_len=200, seed=42, max_chunks_per_conversation=3
            )
            p1._append_conversation(messages)

            p2 = self.producer(
                dir2, seq_len=200, seed=42, max_chunks_per_conversation=3
            )
            p2._append_conversation(messages)

            p3 = self.producer(
                dir3, seq_len=200, seed=999, max_chunks_per_conversation=3
            )
            p3._append_conversation(messages)

            self.assertEqual(p1.token_buffer, p2.token_buffer)
            self.assertEqual(p1.loss_mask_buffer, p2.loss_mask_buffer)
            self.assertNotEqual(p1.token_buffer, p3.token_buffer)

    def test_budget_small_conversation_keep_all(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(directory, seq_len=200, keep_all_threshold=8)
            self.assertEqual(producer._calculate_chunk_budget(3), 3)
            messages = []
            for i in range(3):
                messages.append({"role": "user", "content": f"Turn {i} " + "u" * 50})
                messages.append(
                    {"role": "assistant", "content": f"Ans {i} " + "a" * 50}
                )
            producer._append_conversation(messages)
            self.assertEqual(producer.stats["conversations_chunked"], 1)
            self.assertEqual(producer.stats["conversations_keep_all"], 1)
            self.assertEqual(producer.stats["conversations_sampled"], 0)
            self.assertEqual(producer.stats["chunks_created"], 3)
            self.assertEqual(producer.stats["chunks_emitted"], 3)
            self.assertEqual(producer.stats["chunks_dropped"], 0)

    def test_budget_medium_conversation_keep_all(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(directory, seq_len=200, keep_all_threshold=8)
            self.assertEqual(producer._calculate_chunk_budget(6), 6)
            messages = []
            for i in range(6):
                messages.append({"role": "user", "content": f"Turn {i} " + "u" * 50})
                messages.append(
                    {"role": "assistant", "content": f"Ans {i} " + "a" * 50}
                )
            producer._append_conversation(messages)
            self.assertEqual(producer.stats["conversations_chunked"], 1)
            self.assertEqual(producer.stats["conversations_keep_all"], 1)
            self.assertEqual(producer.stats["conversations_sampled"], 0)
            self.assertEqual(producer.stats["chunks_created"], 6)
            self.assertEqual(producer.stats["chunks_emitted"], 6)
            self.assertEqual(producer.stats["chunks_dropped"], 0)

    def test_budget_larger_conversation_retention_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(
                directory,
                seq_len=200,
                retention_ratio=0.25,
                min_chunks_per_conversation=4,
                max_chunks_per_conversation=32,
                keep_all_threshold=8,
            )
            self.assertEqual(producer._calculate_chunk_budget(40), 10)
            messages = []
            for i in range(40):
                messages.append({"role": "user", "content": f"Turn {i} " + "u" * 50})
                messages.append(
                    {"role": "assistant", "content": f"Ans {i} " + "a" * 50}
                )
            producer._append_conversation(messages)
            self.assertEqual(producer.stats["conversations_chunked"], 1)
            self.assertEqual(producer.stats["conversations_keep_all"], 0)
            self.assertEqual(producer.stats["conversations_sampled"], 1)
            self.assertEqual(producer.stats["conversations_hit_max_cap"], 0)
            self.assertEqual(producer.stats["chunks_created"], 40)
            self.assertEqual(producer.stats["chunks_emitted"], 10)
            self.assertEqual(producer.stats["chunks_dropped"], 30)

    def test_budget_extreme_conversation_hit_max_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(
                directory,
                seq_len=200,
                retention_ratio=0.25,
                max_chunks_per_conversation=32,
                keep_all_threshold=8,
            )
            self.assertEqual(producer._calculate_chunk_budget(200), 32)
            messages = []
            for i in range(200):
                messages.append({"role": "user", "content": f"Turn {i} " + "u" * 50})
                messages.append(
                    {"role": "assistant", "content": f"Ans {i} " + "a" * 50}
                )
            producer._append_conversation(messages)
            self.assertEqual(producer.stats["conversations_chunked"], 1)
            self.assertEqual(producer.stats["conversations_keep_all"], 0)
            self.assertEqual(producer.stats["conversations_sampled"], 1)
            self.assertEqual(producer.stats["conversations_hit_max_cap"], 1)
            self.assertEqual(producer.stats["chunks_created"], 200)
            self.assertEqual(producer.stats["chunks_emitted"], 32)
            self.assertEqual(producer.stats["chunks_dropped"], 168)

    def test_stratified_selection_covers_beginning_middle_end(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(directory, seed=42)
            total_chunks = 100
            target_chunks = 8
            indices = producer._stratified_sample_indices(total_chunks, target_chunks)
            self.assertEqual(len(indices), target_chunks)
            # Explicitly protect beginning (0) and end (total_chunks - 1)
            self.assertEqual(indices[0], 0)
            self.assertEqual(indices[-1], total_chunks - 1)
            k = target_chunks - 2
            interior_total = total_chunks - 2
            for i in range(k):
                start = 1 + (i * interior_total // k)
                end = 1 + ((i + 1) * interior_total // k)
                self.assertTrue(
                    start <= indices[1 + i] < end,
                    f"Interior index {indices[1 + i]} not in stratum [{start}, {end})",
                )

    def test_token_retention_statistics_with_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(
                directory,
                seq_len=200,
                overlap_turns=1,
                min_chunk_tokens=30,
            )
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Question 1: " + "q" * 40},
                {"role": "assistant", "content": "Answer 1: " + "a" * 40},
                {"role": "user", "content": "Question 2: " + "q" * 40},
                {"role": "assistant", "content": "Answer 2: " + "a" * 40},
                {"role": "user", "content": "Question 3: " + "q" * 40},
                {"role": "assistant", "content": "Answer 3: " + "a" * 40},
            ]
            ids, _ = tokenize_messages(TinyTokenizer(), messages)
            orig_len = len(ids)
            self.assertGreater(orig_len, 200)

            producer._append_conversation(messages)
            stats = producer.stats

            self.assertEqual(stats["conversations_chunked"], 1)
            self.assertGreater(stats["chunks_emitted"], 1)
            self.assertEqual(stats["tokens_seen"], orig_len)

            # Training tokens emitted must exceed unique source tokens due to overlap
            self.assertGreater(
                stats["training_tokens_emitted"],
                stats["unique_source_tokens_retained"],
            )
            self.assertEqual(
                stats["overlap_tokens"],
                stats["training_tokens_emitted"]
                - stats["unique_source_tokens_retained"],
            )
            # Tokens dropped must equal tokens_seen minus unique_source_tokens_retained
            self.assertEqual(
                stats["tokens_dropped"],
                stats["tokens_seen"] - stats["unique_source_tokens_retained"],
            )
            # Unique source tokens retained cannot exceed tokens seen
            self.assertLessEqual(stats["unique_source_tokens_retained"], orig_len)

            summary = producer.get_stats_summary()
            self.assertIn("Original source tokens:", summary)
            self.assertIn("Unique source tokens kept:", summary)
            self.assertIn("Training tokens emitted:", summary)
            self.assertIn("Overlap tokens:", summary)
            self.assertIn("Tokens dropped:", summary)
            self.assertIn("Retention rate:", summary)

    def test_chunk_retention_ratio_parameter_and_alias(self):
        with (
            tempfile.TemporaryDirectory() as dir1,
            tempfile.TemporaryDirectory() as dir2,
        ):
            p1 = self.producer(dir1, chunk_retention_ratio=0.35)
            self.assertEqual(p1.chunk_retention_ratio, 0.35)
            self.assertEqual(p1.retention_ratio, 0.35)

            p2 = self.producer(dir2, retention_ratio=0.35)
            self.assertEqual(p2.chunk_retention_ratio, 0.35)
            self.assertEqual(p2.retention_ratio, 0.35)

    def test_min_chunk_tokens_default_is_64(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(directory)
            self.assertEqual(producer.min_chunk_tokens, 64)

    def test_stratified_selection_strictly_sorted_and_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(directory, seed=123)
            for total, target in [(10, 4), (50, 12), (100, 32), (500, 32), (6, 6)]:
                indices = producer._stratified_sample_indices(total, target)
                self.assertEqual(len(indices), target)
                self.assertEqual(indices, sorted(indices))
                self.assertEqual(len(set(indices)), len(indices))
                if target > 1:
                    for j in range(len(indices) - 1):
                        self.assertLess(indices[j], indices[j + 1])

    def test_chunking_consecutive_conversations_advance_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(directory, seed=42)
            ind1 = producer._stratified_sample_indices(100, 10)
            ind2 = producer._stratified_sample_indices(100, 10)
            self.assertNotEqual(ind1, ind2)

    def test_budget_legacy_fixed_cap_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = self.producer(
                directory,
                seq_len=200,
                budget_strategy="fixed",
                max_chunks_per_conversation=4,
            )
            self.assertEqual(producer._calculate_chunk_budget(20), 4)
            messages = []
            for i in range(20):
                messages.append({"role": "user", "content": f"Turn {i} " + "u" * 50})
                messages.append(
                    {"role": "assistant", "content": f"Ans {i} " + "a" * 50}
                )
            producer._append_conversation(messages)
            self.assertEqual(producer.stats["conversations_chunked"], 1)
            self.assertEqual(producer.stats["chunks_emitted"], 4)
            self.assertEqual(producer.stats["chunks_dropped"], 16)


if __name__ == "__main__":
    unittest.main()
