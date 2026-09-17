"""Tiny CPU contracts for the TPU entry; no XLA runtime or hardware required."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "post-training"))

import torch
from RLHF.reward_model_tpu_train import (
    EpochSampler,
    FixedPreferenceCollator,
    StaticDroplessMoE,
    TPURewardModel,
    XlaRuntime,
    main,
    parameter_partition,
    smoke_model_config,
    validate_model_config,
)

from model.hybrid.model import HybridModel
from model.layers.moe import DroplessMoELayer


class TPURewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_static_moe_matches_outputs_and_gradients(self):
        config = smoke_model_config()
        config.top_k = 1
        original = HybridModel(config).layers[0].moe_block
        original.use_grouped_moe_dispatch = False
        adapted = StaticDroplessMoE(
            copy.deepcopy(original.router), copy.deepcopy(original.experts)
        )
        a = torch.randn(2, 5, 16, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        expected, *_ = original(a)
        actual, *_ = adapted(b)
        torch.testing.assert_close(actual, expected)
        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(a.grad, b.grad)
        for source, target in zip(original.parameters(), adapted.parameters()):
            # Static execution may produce explicit zeros for unrouted experts.
            reference = (
                source.grad if source.grad is not None else torch.zeros_like(source)
            )
            torch.testing.assert_close(target.grad, reference)

    def test_backbone_equivalence_and_padding(self):
        config = smoke_model_config()
        model = TPURewardModel(config).eval()
        reference = HybridModel(config).eval()
        reference.load_state_dict(model.backbone.state_dict())
        batch = FixedPreferenceCollator(8, 32, 0)(
            [
                {"chosen": [1, 3, 2], "rejected": [1, 4, 5, 2]},
                {"chosen": [1, 6, 7, 2], "rejected": [1, 8, 2]},
            ]
        )
        ids, mask = batch["chosen_input_ids"], batch["chosen_attention_mask"]
        with torch.no_grad():
            hidden = reference(ids, attention_mask=mask, use_cache=False)[0]
            expected = model.reward_head(hidden[torch.arange(2), mask.sum(1) - 1])
            actual = model(ids, mask)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(actual.shape, (2, 1))
            torch.testing.assert_close(actual[:1], model(ids[:1, :3], mask[:1, :3]))
        self.assertFalse(any("lm_head" in name for name, _ in model.named_parameters()))
        self.assertTrue(
            any(isinstance(layer, DroplessMoELayer) for layer in model.modules())
        )

    def test_fixed_collation_and_reject_invalid_data(self):
        collator = FixedPreferenceCollator(8, 32, 0)
        batch = collator([{"chosen": [1, 2], "rejected": [1, 3, 2]}])
        self.assertEqual(
            set(batch),
            {
                f"{side}_{field}"
                for side in ("chosen", "rejected")
                for field in ("input_ids", "attention_mask")
            },
        )
        self.assertTrue(all(value.shape == (1, 8) for value in batch.values()))
        self.assertEqual(batch["chosen_attention_mask"].sum().item(), 2)
        self.assertEqual(batch["chosen_input_ids"][0, 1].item(), 2)
        for tokens in ([], list(range(9)), [32], [-1], [1.5]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                collator([{"chosen": tokens, "rejected": [1, 2]}])

    def test_resume_sampler_is_same_order_without_consuming_data(self):
        dataset = Mock()
        dataset.__len__ = Mock(return_value=13)
        all_indices = list(EpochSampler(dataset, seed=42))
        self.assertEqual(
            list(EpochSampler(dataset, seed=42, offset=4)), all_indices[4:]
        )
        self.assertNotEqual(list(EpochSampler(dataset, seed=43)), all_indices)
        self.assertEqual(len(set(all_indices)), 13)
        dataset.assert_not_called()

    def test_parameter_and_optimizer_sharding_annotations(self):
        self.assertEqual(parameter_partition((32, 16), 8), ("fsdp", None))
        self.assertEqual(parameter_partition((1, 16), 8), (None, "fsdp"))
        self.assertEqual(parameter_partition((1,), 8), (None,))
        self.assertEqual(parameter_partition((), 8), ())
        runtime = object.__new__(XlaRuntime)
        runtime.xs, runtime.mesh, runtime.devices = Mock(), object(), 8
        model = torch.nn.Linear(16, 1)
        optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
        model(torch.ones(2, 16)).sum().backward()
        optimizer.step()
        runtime.shard(model, optimizer)
        self.assertEqual(
            runtime.xs.mark_sharding.call_count, 6
        )  # weights, bias, four moments
        for call in runtime.xs.mark_sharding.call_args_list:
            tensor, mesh, spec = call.args
            self.assertIs(mesh, runtime.mesh)
            self.assertEqual(spec, parameter_partition(tensor.shape, 8))

    def test_config_validation(self):
        cfg = smoke_model_config()
        validate_model_config(cfg, 8)
        for key, value in (
            ("use_dual_memory", True),
            ("use_auxiliary_losses", True),
            ("capacity_factor", 1.25),
            ("top_k", 3),
            ("head_dim", 7),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                invalid = copy.deepcopy(cfg)
                setattr(invalid, key, value)
                validate_model_config(invalid, 8)
        with self.assertRaises(ValueError):
            validate_model_config(cfg, 17)

    def test_smoke_checkpoint_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            full = str(Path(directory) / "full")
            resumed = str(Path(directory) / "resumed")
            with self.assertLogs("reward_tpu", level="INFO") as logs:
                full_path = main(["--smoke", "--output-dir", full])
            self.assertTrue(any("pairwise_accuracy=" in line for line in logs.output))
            first_path = main(["--smoke", "--max-steps", "1", "--output-dir", resumed])
            first = torch.load(first_path, weights_only=True)
            self.assertEqual((first["step"], first["epoch"], first["batch"]), (1, 0, 2))
            resumed_path = main(
                ["--smoke", "--output-dir", resumed, "--resume", str(first_path)]
            )
            expected = torch.load(full_path, weights_only=True)
            actual = torch.load(resumed_path, weights_only=True)
            self.assertEqual(actual["step"], 2)
            for key, tensor in expected["model"].items():
                torch.testing.assert_close(actual["model"][key], tensor, atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                main(
                    [
                        "--smoke",
                        "--output-dir",
                        resumed,
                        "--resume",
                        str(resumed_path),
                        "--seed",
                        "99",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
