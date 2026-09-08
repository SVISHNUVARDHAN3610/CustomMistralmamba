"""Synthetic CPU evaluation/accounting tests; no distributed process group."""

import copy
import json
import logging
import random
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "post-training"))

from RLHF.config import RewardConfig, smoke_config
from RLHF.evaluation_metrics import RewardStatistics, isolated_evaluation_rng
from RLHF.rlhf_dataset import (
    LengthStatistics,
    PreferenceCollator,
    PreferenceShardProducer,
    SmokeTokenizer,
    accounting_summary,
    smoke_stream,
    sources,
    tokenize_preference,
)
from RLHF.train_reward_model import (
    PreferenceFeed,
    build_diagnostic_set,
    evaluate_batches,
    evaluate_diagnostic,
    evaluate_full,
    loader_for_shard,
)

from utils.sft_dataset import tokenize_messages


def cpu_backend():
    return SimpleNamespace(
        rank=0,
        world=1,
        device=torch.device("cpu"),
        broadcast=lambda value: value,
        sum=lambda value: value,
        barrier=lambda: None,
        autocast=nullcontext,
        logger=logging.getLogger("reward-diagnostic-tests"),
    )


class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, input_ids, attention_mask):
        self.seen.append(input_ids.tolist())
        # Deliberate incidental RNG use proves evaluation restores all RNGs.
        random.random()
        np.random.random()
        torch.rand(1)
        return (input_ids * attention_mask).double().sum(1, keepdim=True) / 100


def _evaluation_collective_worker(rank, world, init_method):
    """Repository Gloo test pattern: CPU collectives only, no FSDP/model sharding."""
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=init_method
    )
    try:
        backend = cpu_backend()
        backend.rank, backend.world = rank, world

        def broadcast(value):
            values = [value]
            torch.distributed.broadcast_object_list(values, src=0)
            return values[0]

        backend.broadcast = broadcast
        batch = {
            "chosen_input_ids": torch.tensor([[3], [4], [5]]),
            "rejected_input_ids": torch.tensor([[1], [2], [2]]),
            "chosen_attention_mask": torch.ones(3, 1, dtype=torch.bool),
            "rejected_attention_mask": torch.ones(3, 1, dtype=torch.bool),
            "category": ["overall"] * 3,
            "original_split": ["train", "validation", "train"],
        }
        result = evaluate_batches(FixtureModel(), iter([batch]), backend)
        reference = evaluate_batches(FixtureModel(), iter([batch]), cpu_backend())
        assert result == reference
        assert result["pairs"] == 3
        # The trainer sums disjoint consumed counts, not replicated producer counts.
        consumed = torch.tensor([1, 2] if rank == 0 else [3, 1], dtype=torch.int64)
        torch.distributed.all_reduce(consumed)
        assert consumed.tolist() == [4, 3]
    finally:
        torch.distributed.destroy_process_group()


class DiagnosticTests(unittest.TestCase):
    @unittest.skipUnless(torch.distributed.is_gloo_available(), "CPU Gloo unavailable")
    def test_two_process_cpu_evaluation_collectives(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _evaluation_collective_worker,
                args=(2, (Path(directory) / "store").resolve().as_uri()),
                nprocs=2,
                join=True,
            )

    def stats(self, chosen, rejected, chunks=None):
        accumulator = RewardStatistics()
        try:
            if chunks is None:
                accumulator.update(chosen, rejected)
            else:
                for a, b in chunks:
                    accumulator.update(chosen[a:b], rejected[a:b])
            return accumulator.result()
        finally:
            accumulator.close()

    def test_distribution_and_margin_statistics(self):
        chosen, rejected = np.array([3.0, 4.0, 5.0]), np.array([1.0, 2.0, 2.0])
        result = self.stats(chosen, rejected)
        self.assertEqual(result["pairs"], 3)
        self.assertEqual(result["pairwise_accuracy"], 1)
        for key, values in (
            ("chosen_reward", chosen),
            ("rejected_reward", rejected),
            ("margin", np.array([2.0, 2.0, 3.0])),
            ("reward", np.r_[chosen, rejected]),
        ):
            for suffix, expected in zip(
                ("mean", "std", "min", "max", "p50", "p95", "p99"),
                (
                    values.mean(),
                    values.std(),
                    values.min(),
                    values.max(),
                    *np.quantile(values, [0.5, 0.95, 0.99]),
                ),
            ):
                self.assertAlmostEqual(result[f"{key}_{suffix}"], expected)

    def test_ties_negative_and_single_pair(self):
        result = self.stats([3.0, 2.0, 1.0], [1.0, 2.0, 3.0])
        self.assertEqual(result["pairwise_accuracy"], 1 / 3)
        for kind in ("positive", "zero", "negative"):
            self.assertEqual(result[f"fraction_{kind}_margin"], 1 / 3)
        self.assertEqual(self.stats([1.0], [1.0])["margin_std"], 0)

    def test_chunked_global_statistics_match_unpartitioned(self):
        chosen = np.array([1e9 + i for i in (1, 4, 8, 20, 30)], dtype=np.float64)
        rejected = chosen - np.array([2, -1, 0, 3, 20])
        full = self.stats(chosen, rejected)
        chunked = self.stats(chosen, rejected, [(0, 2), (2, 3), (3, 5)])
        for key in full:
            if isinstance(full[key], (int, float)):
                self.assertAlmostEqual(chunked[key], full[key], places=6)

    def test_invalid_rewards_and_failure_rng_cleanup(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.stats([float("nan")], [0.0])
        with self.assertRaisesRegex(ValueError, "No valid"):
            self.stats([], [])
        before = torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError), isolated_evaluation_rng():
            torch.rand(10)
            raise RuntimeError("fixture failure")
        torch.testing.assert_close(before, torch.get_rng_state())

    def test_fixed_diagnostic_and_rng_and_training_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, tokenizer, backend = (
                smoke_config(directory),
                SmokeTokenizer(),
                cpu_backend(),
            )
            cursor = {
                "epoch": 0,
                "shard": 0,
                "batch": 0,
                "raw_start": 0,
                "native_start": None,
            }
            original_cursor = copy.deepcopy(cursor)
            training = PreferenceFeed(
                cfg, tokenizer, backend, "train", cursor, smoke=True
            )
            dataset = None
            try:
                path = training.wait(0)
                dataset, loader = loader_for_shard(path, cfg, tokenizer, backend, 0)
                iterator = iter(loader)
                next(iterator)
                remaining_expected = list(loader)[1]
                python_state, numpy_state, torch_state = (
                    random.getstate(),
                    np.random.get_state(),
                    torch.get_rng_state().clone(),
                )
                diagnostic = build_diagnostic_set(cfg, tokenizer, backend, smoke=True)
                self.assertEqual(
                    diagnostic,
                    build_diagnostic_set(cfg, tokenizer, backend, smoke=True),
                )
                self.assertEqual([len(rows) for rows in diagnostic.values()], [2, 2])
                model = FixtureModel().train()
                first = evaluate_diagnostic(model, diagnostic, cfg, tokenizer, backend)
                inputs = copy.deepcopy(model.seen)
                model.seen.clear()
                second = evaluate_diagnostic(model, diagnostic, cfg, tokenizer, backend)
                self.assertEqual(first, second)
                self.assertEqual(model.seen, inputs)
                self.assertEqual(random.getstate(), python_state)
                np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
                torch.testing.assert_close(torch.get_rng_state(), torch_state)
                self.assertTrue(model.training)
                self.assertEqual(cursor, original_cursor)
                remaining_actual = next(iterator)
                torch.testing.assert_close(
                    remaining_actual["chosen_input_ids"],
                    remaining_expected["chosen_input_ids"],
                )
            finally:
                if dataset is not None:
                    dataset.close()
                training.close()

    def test_full_evaluation_is_separate_and_preserves_helpsteer_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = smoke_config(directory)
            result = evaluate_full(
                FixtureModel(),
                cfg,
                SmokeTokenizer(),
                cpu_backend(),
                smoke=True,
                max_batches=1,
            )
            self.assertEqual(
                set(result), {"ultrafeedback_test", "hh_rlhf_test", "helpsteer2"}
            )
            self.assertEqual(
                set(result["helpsteer2"]["subsets"]),
                {"original_train", "original_validation"},
            )
            self.assertEqual(result["helpsteer2"]["pairs"], 2)
            for group in result["helpsteer2"]["subsets"].values():
                self.assertEqual(group["pairs"], 1)
                self.assertIn("margin_p99", group)
            self.assertNotIn("helpsteer2", [s["adapter"] for s in sources(cfg.data)])

    def test_replicated_rank_statistics_count_rows_once(self):
        row = tokenize_preference(
            {"prompt": "Q?", "chosen": "yes", "rejected": "no"},
            SmokeTokenizer(),
            smoke_config("unused").data,
        )
        batch = PreferenceCollator(0)([row])
        backend = cpu_backend()
        reference = evaluate_batches(FixtureModel(), iter([batch]), backend)
        backend.world = 4  # Simulate the existing replicated-forward protocol.
        broadcasts = []

        def broadcast(value):
            broadcasts.append(value)
            return value

        backend.broadcast = broadcast
        result = evaluate_batches(FixtureModel(), iter([batch]), backend)
        self.assertEqual(reference, result)
        backend.rank = 1
        backend.broadcast = lambda _: broadcasts.pop(0)
        with patch(
            "RLHF.train_reward_model.RewardStatistics",
            side_effect=AssertionError("non-owner counted a replicated row"),
        ):
            peer = evaluate_batches(FixtureModel(), iter([batch]), backend)
        self.assertEqual(peer, result)
        self.assertEqual(peer["pairs"], 1)

    def test_source_accounting_after_filtering_and_consumption(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = smoke_config(directory)
            producer = PreferenceShardProducer(
                directory, SmokeTokenizer(), cfg.data, 256, 42
            )
            uf, hh = list(smoke_stream("train").take(2))
            records = [
                producer._encode_sample(uf)[0],
                producer._encode_sample(hh)[0],
                producer._encode_sample(hh)[0],
            ]
            producer._encode_sample(dict(uf, payload="{}"))
            counts = producer.accounting_snapshot()
            # Only examples actually selected on disjoint ranks are consumed.
            rank0, rank1 = (
                Counter([records[0]["source"]]),
                Counter([records[2]["source"]]),
            )
            report = accounting_summary(cfg.data, counts, rank0 + rank1)
            self.assertEqual(
                report["configured_weights"], {"ultrafeedback": 35, "hh_rlhf": 75}
            )
            self.assertEqual(report["target_probabilities"]["ultrafeedback"], 35 / 110)
            uf_stats, hh_stats = (
                report["sources"]["ultrafeedback"],
                report["sources"]["hh_rlhf"],
            )
            self.assertEqual(
                (
                    uf_stats["raw"],
                    uf_stats["normalized"],
                    uf_stats["filtered"],
                    uf_stats["valid"],
                    uf_stats["tokenized"],
                ),
                (2, 1, 1, 1, 1),
            )
            self.assertEqual(uf_stats["retention_rate"], 0.5)
            self.assertEqual(uf_stats["valid_mixture"], 1 / 3)
            self.assertEqual(hh_stats["valid_mixture"], 2 / 3)
            self.assertEqual(uf_stats["consumed_mixture"], 0.5)
            self.assertEqual(hh_stats["consumed"], 1)


class TruncationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = smoke_config("unused").data
        self.cfg.min_response_tokens = 1
        self.tokenizer = SmokeTokenizer()
        self.record = {"prompt": "Q?", "chosen": "Good", "rejected": "Bad"}
        self.prefix, _ = tokenize_messages(
            self.tokenizer, [{"role": "user", "content": "Q?"}]
        )
        self.prefix += self.tokenizer.encode("\nassistant:\n")
        self.budget = self.cfg.max_length - len(self.prefix) - 1

    def test_complete_and_tiny_responses(self):
        for chosen in ("Good", "a"):
            row = tokenize_preference(
                dict(self.record, chosen=chosen), self.tokenizer, self.cfg
            )
            self.assertEqual(
                row["chosen"], self.prefix + self.tokenizer.encode(chosen) + [2]
            )

    def test_head_tail_and_independent_lengths(self):
        record = dict(self.record, chosen="abcdefghijklmnopqrstuvwxyz" * 10)
        original = self.tokenizer.encode(record["chosen"])
        for strategy in ("head", "head_tail"):
            self.cfg.response_truncation_strategy = strategy
            diagnostics = {}
            row = tokenize_preference(record, self.tokenizer, self.cfg, diagnostics)
            head = (self.budget + 1) // 2
            expected = (
                original[: self.budget]
                if strategy == "head"
                else original[:head] + original[-(self.budget - head) :]
            )
            self.assertEqual(row["chosen"], self.prefix + expected + [2])
            self.assertEqual(
                row["rejected"], self.prefix + self.tokenizer.encode("Bad") + [2]
            )
            self.assertEqual(diagnostics["chosen_truncated"], 1)
            self.assertEqual(diagnostics["rejected_truncated"], 0)
            self.assertEqual(row, tokenize_preference(record, self.tokenizer, self.cfg))

    def test_no_duplicate_terminal_eos(self):
        original_encode = self.tokenizer.encode
        for text in ("Good", "a" * 100):

            def encode(value, text=text, **kwargs):
                result = original_encode(value, **kwargs)
                return result + [2, 2] if value == text else result

            with patch.object(self.tokenizer, "encode", side_effect=encode):
                row = tokenize_preference(
                    dict(self.record, chosen=text), self.tokenizer, self.cfg
                )
            self.assertEqual(row["chosen"][-1], 2)
            self.assertNotEqual(row["chosen"][-2], 2)
            self.assertLessEqual(len(row["chosen"]), self.cfg.max_length)

    def test_prompt_budget_zero_nearly_full_and_reserve(self):
        for budget in (0, -1):
            self.cfg.max_length = len(self.prefix) + 1 + budget
            diagnostic = {}
            with self.assertRaisesRegex(ValueError, "Prompt too long"):
                tokenize_preference(self.record, self.tokenizer, self.cfg, diagnostic)
            self.assertEqual(diagnostic["insufficient_budget"], 1)
        self.cfg.max_length = len(self.prefix) + 2
        row = tokenize_preference(self.record, self.tokenizer, self.cfg)
        self.assertEqual(len(row["chosen"]), self.cfg.max_length)
        self.assertEqual(row["chosen"][:-2], self.prefix)
        self.cfg.min_response_tokens = 8
        with self.assertRaisesRegex(ValueError, "Prompt too long"):
            tokenize_preference(self.record, self.tokenizer, self.cfg)

    def test_head_cutoff_on_internal_eos(self):
        self.cfg.response_truncation_strategy = "head"
        self.cfg.max_length = len(self.prefix) + 3  # two body tokens plus EOS
        encode = self.tokenizer.encode

        def internal_eos(value, **kwargs):
            return [10, 2, 11, 12] if value == "Good" else encode(value, **kwargs)

        with patch.object(self.tokenizer, "encode", side_effect=internal_eos):
            result = tokenize_preference(self.record, self.tokenizer, self.cfg)
        self.assertEqual(result["chosen"], self.prefix + [10, 2])

    def test_truncation_collision_filtered_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            self.cfg.response_truncation_strategy = "head"
            producer = PreferenceShardProducer(
                directory, self.tokenizer, self.cfg, 256, 42
            )
            envelope = next(iter(smoke_stream("train")))
            raw = json.loads(envelope["payload"])
            for side in ("chosen", "rejected"):
                raw[side][-1]["content"] = "a" * 200 + side
            self.assertEqual(
                producer._encode_sample(dict(envelope, payload=json.dumps(raw))), []
            )
            counts = producer.accounting_snapshot()["ultrafeedback"]
            self.assertEqual(counts["both_truncated"], 1)
            self.assertEqual(counts["filtered"], 1)
            self.assertEqual(counts["valid_before_truncation"], 1)
            self.assertIn(
                "truncated/chosen_original_length",
                producer.sequence_summary()["ultrafeedback"],
            )
            self.cfg.max_length = 4
            self.assertEqual(producer._encode_sample(envelope), [])
            self.assertEqual(
                producer.accounting_snapshot()["ultrafeedback"]["insufficient_budget"],
                1,
            )

    def test_bounded_length_statistics_and_configuration(self):
        first, second = LengthStatistics(capacity=8), LengthStatistics(capacity=8)
        for value in range(100):
            first.add(value)
            second.add(value)
        self.assertEqual(first.summary(), second.summary())
        self.assertEqual(first.summary()["mean"], 49.5)
        self.assertEqual(first.summary()["max"], 99)
        self.assertEqual(len(first.sample), 8)
        cfg = RewardConfig()
        self.assertEqual(cfg.data.max_length, 2048)
        self.assertEqual(cfg.system.diagnostic_eval_pairs, 512)
        self.assertEqual(cfg.data.response_truncation_strategy, "head_tail")
        legacy = cfg.to_dict()
        legacy["system"]["eval_batches"] = 16
        self.assertNotIn(
            "eval_batches", RewardConfig.from_dict(legacy).to_dict()["system"]
        )
        cfg.data.response_truncation_strategy = "invalid"
        with self.assertRaisesRegex(ValueError, "strategy"):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()
