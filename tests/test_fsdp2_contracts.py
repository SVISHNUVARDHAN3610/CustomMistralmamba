from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import unittest
from pathlib import Path

import torch


def _load_fsdp2_train():
    path = Path(__file__).resolve().parents[1] / "pre-training" / "fsdp2_train.py"
    spec = importlib.util.spec_from_file_location("fsdp2_train_contract_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fsdp2_train = _load_fsdp2_train()


class _SequenceSampler:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


class FSDP2ContractTests(unittest.TestCase):
    def test_resume_sampler_starts_at_exact_offset(self) -> None:
        sampler = fsdp2_train._OffsetSampler(_SequenceSampler(list(range(10))), 6)
        self.assertEqual(list(sampler), [6, 7, 8, 9])
        self.assertEqual(len(sampler), 4)

    def test_microbatch_metrics_are_averaged(self) -> None:
        means = fsdp2_train._mean_metric_sums(
            {"loss": torch.tensor(8.0), "ce_loss": torch.tensor(5.0)}, 2
        )
        self.assertEqual(float(means["loss"]), 4.0)
        self.assertEqual(float(means["ce_loss"]), 2.5)

    def test_fsdp2_optimizer_is_adamw_only(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.LayerNorm(4),
        )
        args = argparse.Namespace(
            lr=7.5e-4,
            adam_lr=None,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
            weight_decay=0.1,
            no_muon=False,
            muon_lr=7.5e-4,
        )
        optimizers, use_muon, meta = fsdp2_train.build_fsdp2_optimizers(
            model, args=args, logger=logging.getLogger("fsdp2-contract-test")
        )
        self.assertFalse(use_muon)
        self.assertEqual(meta["optimizer_policy"], "fsdp2_adamw")
        self.assertEqual(len(optimizers), 1)
        self.assertIsInstance(optimizers[0], torch.optim.AdamW)
        optimized = {
            id(param)
            for group in optimizers[0].param_groups
            for param in group["params"]
        }
        expected = {id(param) for param in model.parameters() if param.requires_grad}
        self.assertEqual(optimized, expected)


if __name__ == "__main__":
    unittest.main()
