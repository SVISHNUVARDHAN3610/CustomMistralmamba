"""Tiny CPU reward tests; never launch FSDP2 or allocate the production model."""

import json
import math
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "post-training"))

import torch
from datasets import IterableDataset
from RLHF.config import DataConfig, ModelConfig, RewardConfig, smoke_config
from RLHF.evaluate_reward_model import main as evaluation_main
from RLHF.reward_model import (
    RewardModel,
    last_valid_hidden,
    pairwise_loss,
    parameter_counts,
)
from RLHF.rlhf_dataset import (
    PreferenceCollator,
    PreferenceShardDataset,
    PreferenceShardProducer,
    SmokeTokenizer,
    load_streaming_source,
    normalize_preference,
    smoke_stream,
    sources,
    tokenize_preference,
)
from RLHF.train_reward_model import (
    RewardBackend,
    checkpoint_metadata,
    initialize_tokenizer,
    load_checkpoint,
    run_training,
)

from utils.dataset import TokenizedShardProducer
from utils.sft_dataset import tokenize_messages


class RewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_normalization(self):
        uf, hh = list(smoke_stream("train").take(2))
        for sample in (uf, hh):
            record = normalize_preference(
                json.loads(sample["payload"]), sample["source"]
            )
            self.assertEqual(record["prompt"], "Q0?" if sample is uf else "Q1?")
            self.assertEqual(record["chosen"], "Correct.")
            self.assertEqual(record["rejected"], "Wrong.")
        hs = json.loads(next(iter(smoke_stream("helpsteer2")))["payload"])
        self.assertEqual(normalize_preference(hs, "helpsteer2")["chosen"], "Correct.")
        hs["preference_strength"] = 2
        self.assertEqual(normalize_preference(hs, "helpsteer2")["chosen"], "Wrong.")
        hs["preference_strength"] = 0
        with self.assertRaises(ValueError):
            normalize_preference(hs, "helpsteer2")
        bad = json.loads(uf["payload"])
        bad["rejected"][0]["content"] = "Different prompt"
        with self.assertRaises(ValueError):
            normalize_preference(bad, "ultrafeedback")

    def test_multiturn(self):
        row = {
            side: "\n\nHuman: First?\n\nAssistant: Earlier.\n\nHuman: Next?\n\nAssistant: "
            + answer
            for side, answer in (("chosen", "Yes."), ("rejected", "No."))
        }
        record = normalize_preference(row, "hh_rlhf")
        self.assertEqual(
            [m["role"] for m in record["prompt_messages"]],
            ["user", "assistant", "user"],
        )
        self.assertEqual(record["chosen"], "Yes.")

    def test_mixture(self):
        cfg = DataConfig()
        self.assertEqual((cfg.ultrafeedback_weight, cfg.hh_rlhf_weight), (35, 75))
        self.assertAlmostEqual(cfg.probabilities()[0], 0.3181818181818182)
        self.assertAlmostEqual(cfg.probabilities()[1], 0.6818181818181818)
        self.assertNotIn("nvidia/HelpSteer2", [s["path"] for s in sources(cfg)])
        from datasets import interleave_datasets

        def sample():
            streams = [
                IterableDataset.from_generator(
                    lambda value=value: ({"v": value} for _ in range(8000))
                )
                for value in (0, 1)
            ]
            return [
                row["v"]
                for row in interleave_datasets(
                    streams,
                    probabilities=cfg.probabilities(),
                    seed=42,
                    stopping_strategy="all_exhausted",
                ).take(5000)
            ]

        first = sample()
        self.assertEqual(first, sample())
        self.assertLess(abs(first.count(0) / len(first) - 35 / 110), 0.025)

    def test_head_and_padding(self):
        hidden = torch.arange(2 * 5 * 3).view(2, 5, 3).float()
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]])
        torch.testing.assert_close(
            last_valid_hidden(hidden, mask), torch.stack([hidden[0, 1], hidden[1, 3]])
        )
        model = RewardModel(ModelConfig(256, 16, 2, 4, 3, 2, 64)).eval()
        ids = torch.tensor([[4, 5, 6, 0, 0], [4, 5, 6, 7, 8]])
        masks = ids != 0
        with torch.no_grad():
            scores = model(ids, masks)
            self.assertEqual(scores.shape, (2, 1))
            torch.testing.assert_close(
                scores[:1], model(ids[:1, :3], masks[:1, :3]), atol=1e-6, rtol=1e-5
            )
            ids[0, 3:] = 199
            torch.testing.assert_close(scores, model(ids, masks))
        with self.assertRaises(ValueError):
            model(ids, torch.zeros_like(masks))
        with self.assertRaises(ValueError):
            model(ids, torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]))

    def test_loss(self):
        chosen = torch.tensor([[0.0]], requires_grad=True)
        rejected = torch.tensor([[0.0]], requires_grad=True)
        loss = pairwise_loss(chosen, rejected)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        self.assertLess(chosen.grad.item(), 0)
        self.assertGreater(rejected.grad.item(), 0)
        self.assertLess(pairwise_loss(chosen + 2, rejected), loss)

    def test_actual_parameter_count(self):
        with torch.device("meta"):
            model = RewardModel(ModelConfig())
        counts = parameter_counts(model)
        self.assertGreaterEqual(counts["total"], 300_000_000)
        self.assertLessEqual(counts["total"], 400_000_000)
        self.assertEqual(counts["trainable"], counts["total"])
        self.assertEqual(counts["reward_head"], 1025)
        self.assertFalse(
            any(
                "lm_head" in n or "attention" in n or "expert" in n
                for n, _ in model.named_parameters()
            )
        )

    def test_tokenization_and_truncation(self):
        tokenizer = SmokeTokenizer()
        cfg = smoke_config("unused").data
        record = normalize_preference(
            json.loads(next(iter(smoke_stream("train")))["payload"]), "ultrafeedback"
        )
        encoded = tokenize_preference(record, tokenizer, cfg)
        expected, _ = tokenize_messages(
            tokenizer,
            record["prompt_messages"]
            + [{"role": "assistant", "content": record["chosen"]}],
        )
        self.assertEqual(encoded["chosen"], expected)
        record["chosen"] = "a" * 100 + "ending"
        record["rejected"] = "Short."
        encoded = tokenize_preference(record, tokenizer, cfg)
        self.assertEqual(len(encoded["chosen"]), cfg.max_length)
        self.assertEqual(encoded["chosen"][-7:-1], tokenizer.encode("ending"))
        batch = PreferenceCollator(0)([encoded])
        self.assertEqual(
            batch["chosen_input_ids"].shape, batch["rejected_input_ids"].shape
        )
        self.assertLess(batch["rejected_attention_mask"].sum(), cfg.max_length)
        record["prompt_messages"][0]["content"] = "p" * 200
        with self.assertRaisesRegex(ValueError, "Prompt too long"):
            tokenize_preference(record, tokenizer, cfg)

    def test_background_pipeline_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = smoke_config(directory)
            cfg.data.buffer_size = 1
            producer = PreferenceShardProducer(
                directory,
                SmokeTokenizer(),
                cfg.data,
                256,
                42,
                sample_stream=smoke_stream("train"),
                log_fn=lambda _: None,
            )
            self.assertIsInstance(producer, TokenizedShardProducer)
            stop = threading.Event()
            thread = threading.Thread(target=producer.start_streaming, args=(stop,))
            thread.start()
            try:
                deadline = time.monotonic() + 10
                while (
                    not list(Path(directory).glob("*.json"))
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.02)
                self.assertEqual(len(list(Path(directory).glob("*.bin"))), 1)
                dataset = PreferenceShardDataset(
                    str(Path(directory, "shard_000000.bin"))
                )
                self.assertEqual(len(dataset), 4)
                batch = PreferenceCollator(0)([dataset[0], dataset[1]])
                self.assertEqual(batch["chosen_input_ids"].size(0), 2)
                self.assertEqual(batch["source"], ["ultrafeedback", "hh_rlhf"])
                dataset.close()
                time.sleep(0.1)
                self.assertEqual(len(list(Path(directory).glob("*.bin"))), 1)
            finally:
                stop.set()
                thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertIsNone(producer.error)
            with (
                patch.object(
                    producer,
                    "_build_stream",
                    side_effect=RuntimeError("dataset unavailable"),
                ),
                self.assertRaisesRegex(RuntimeError, "dataset unavailable"),
            ):
                producer.start_streaming()
            self.assertIsInstance(producer.error, RuntimeError)

    def test_training_checkpoint_resume_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = smoke_config(str(Path(directory, "continuous")))
            cfg.data.pairs_per_shard = 6  # stop/resume inside the first shard
            cfg.system.full_eval_at_checkpoints = True
            full = run_training(cfg, smoke=True)
            self.assertTrue(
                Path(directory, "continuous", "full_evaluation_1.json").is_file()
            )
            cfg_no_eval = smoke_config(str(Path(directory, "no_evaluation")))
            cfg_no_eval.data.pairs_per_shard = 6
            cfg_no_eval.system.diagnostic_eval_enabled = False
            cfg_no_eval.system.full_eval_enabled = False
            no_eval = run_training(cfg_no_eval, smoke=True)
            cfg = smoke_config(str(Path(directory, "resumed")))
            cfg.data.pairs_per_shard = 6
            first = run_training(cfg, smoke=True, stop_after=1)
            self.assertEqual(checkpoint_metadata(first)["step"], 1)
            cfg.system.resume_from_checkpoint = first
            final = run_training(cfg, smoke=True)
            self.assertEqual(checkpoint_metadata(final)["step"], 2)
            a, b, c = (
                RewardModel(cfg.model),
                RewardModel(cfg.model),
                RewardModel(cfg.model),
            )
            load_checkpoint(full, a, cfg=cfg)
            load_checkpoint(final, b, cfg=cfg)
            load_checkpoint(no_eval, c, cfg=cfg)
            for name, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
                torch.testing.assert_close(value, c.state_dict()[name], rtol=0, atol=0)
            first_diagnostic = json.loads(
                Path(
                    directory, "continuous", "diagnostic_evaluation_1.json"
                ).read_text()
            )
            resumed_diagnostic = json.loads(
                Path(directory, "resumed", "diagnostic_evaluation_2.json").read_text()
            )
            for source in first_diagnostic:
                self.assertEqual(
                    first_diagnostic[source]["subset_sha256"],
                    resumed_diagnostic[source]["subset_sha256"],
                )
            accounting = json.loads(
                Path(cfg.system.output_dir, "dataset_accounting.json").read_text()
            )
            self.assertEqual(
                sum(s["consumed"] for s in accounting["sources"].values()), 2
            )
            result = evaluation_main(
                [
                    "--checkpoint",
                    final,
                    "--smoke",
                    "--max-batches",
                    "1",
                    "--output-dir",
                    str(Path(directory, "evaluation")),
                ]
            )
            self.assertEqual(
                set(result), {"ultrafeedback_test", "hh_rlhf_test", "helpsteer2"}
            )
            self.assertEqual(result["helpsteer2"]["pairs"], 2)
            self.assertTrue(0 <= result["helpsteer2"]["pairwise_accuracy"] <= 1)
            self.assertTrue(Path(cfg.system.output_dir, "metrics.jsonl").is_file())

    def test_native_and_fallback_shard_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = smoke_config(directory)
            original = Path(directory, "original")
            producer = PreferenceShardProducer(
                str(original),
                SmokeTokenizer(),
                cfg.data,
                256,
                42,
                sample_stream=smoke_stream("train"),
                log_fn=lambda _: None,
            )
            producer.start_streaming()
            boundary = json.loads(Path(original, "shard_000001.json").read_text())
            self.assertEqual(boundary["raw_start"], 4)
            for native in (boundary["native_start"], None):
                target = Path(directory, "native" if native else "fallback")
                cursor = {"shard": 1, "raw_start": 4, "native_start": native}
                resumed = PreferenceShardProducer(
                    str(target),
                    SmokeTokenizer(),
                    cfg.data,
                    256,
                    42,
                    cursor=cursor,
                    sample_stream=smoke_stream("train"),
                    log_fn=lambda _: None,
                )
                resumed.start_streaming()
                self.assertEqual(
                    Path(original, "shard_000001.bin").read_bytes(),
                    Path(target, "shard_000001.bin").read_bytes(),
                )

    def test_filter_stats_and_tokenizer_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = smoke_config(directory)
            sample = next(iter(smoke_stream("train")))
            producer = PreferenceShardProducer(
                directory, SmokeTokenizer(), cfg.data, 256, 42, log_fn=lambda _: None
            )
            self.assertEqual(len(producer._encode_sample(sample)), 1)
            bad = dict(sample, payload="{}")
            self.assertEqual(producer._encode_sample(bad), [])
            self.assertEqual(
                dict(producer.stats), {"raw": 2, "valid": 1, "filtered": 1}
            )
            with (
                patch.object(
                    producer.tokenizer,
                    "encode",
                    side_effect=ValueError("broken tokenizer"),
                ),
                self.assertRaisesRegex(RuntimeError, "Reward tokenizer failed"),
            ):
                producer._encode_sample(sample)

    def test_fsdp2_wrapping_contract_without_distributed_launch(self):
        # Inspect the actual wrapper construction; mock only fully_shard, since
        # executing FSDP2 belongs on cloud GPUs. This is not a GPU validation.
        cfg = smoke_config("unused")
        model = RewardModel(cfg.model)
        backend = object.__new__(RewardBackend)
        backend.smoke = False
        calls = []

        def fully_shard(module, *, mp_policy, reshard_after_forward):
            self.assertTrue(reshard_after_forward)
            calls.append(module)

        backend.api = {
            "MixedPrecisionPolicy": lambda **kw: kw,
            "fully_shard": fully_shard,
        }
        model = backend.wrap(model, cfg)
        self.assertEqual(calls, [*model.layers, model])
        self.assertFalse(model.activation_checkpointing)
        ids = torch.tensor([[4, 5, 6]])
        model(ids, torch.ones_like(ids)).sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertTrue(
            all(
                p._no_weight_decay
                for n, p in model.named_parameters()
                if n.endswith(".A_log")
            )
        )
        plain = RewardModel(cfg.model)
        plain.load_state_dict(model.state_dict())  # wrapper-independent checkpoint keys

    @unittest.skipUnless(
        os.environ.get("RM_LIVE_SCHEMA") == "1", "Opt-in tiny live HF samples"
    )
    def test_live_hf_samples(self):
        cfg = RewardConfig()
        tokenizer = initialize_tokenizer(cfg)
        for source in sources(cfg.data) + sources(cfg.data, "helpsteer2"):
            rows = list(load_streaming_source(source).take(2))
            self.assertEqual(len(rows), 2)
            valid = []
            for row in rows:
                try:
                    valid.append(
                        tokenize_preference(
                            normalize_preference(row, source["adapter"]),
                            tokenizer,
                            cfg.data,
                        )
                    )
                except ValueError as exc:
                    print(f"Live sample filtered {source['path']}: {exc}")
            self.assertTrue(valid, source["path"])
            batch = PreferenceCollator(tokenizer.pad_token_id)(valid)
            self.assertEqual(batch["chosen_input_ids"].size(0), len(valid))
            # Reuse only these two downloaded rows: exercise the background
            # producer, native iterable state, shard format and tiny real forward.
            envelopes = [
                {"source": source["adapter"], "payload": json.dumps(row)}
                for row in rows
            ]
            with tempfile.TemporaryDirectory() as directory:
                producer = PreferenceShardProducer(
                    directory,
                    tokenizer,
                    cfg.data,
                    cfg.model.vocab_size,
                    cfg.seed,
                    purpose="helpsteer2"
                    if source["adapter"] == "helpsteer2"
                    else "train",
                    sample_stream=IterableDataset.from_generator(
                        lambda envelopes=envelopes: iter(envelopes)
                    ),
                    log_fn=lambda _: None,
                )
                thread = threading.Thread(target=producer.start_streaming)
                thread.start()
                thread.join(15)
                self.assertFalse(thread.is_alive())
                self.assertIsNone(producer.error)
                dataset = PreferenceShardDataset(
                    str(Path(directory, "shard_000000.bin"))
                )
                try:
                    batch = PreferenceCollator(tokenizer.pad_token_id)(
                        [dataset[i] for i in range(len(dataset))]
                    )
                    tiny = RewardModel(ModelConfig(32000, 16, 1, 4, 3, 2, 2048)).eval()
                    with torch.no_grad():
                        rewards = tiny(
                            batch["chosen_input_ids"], batch["chosen_attention_mask"]
                        )
                    self.assertEqual(rewards.shape, (len(valid), 1))
                finally:
                    dataset.close()


if __name__ == "__main__":
    unittest.main()
