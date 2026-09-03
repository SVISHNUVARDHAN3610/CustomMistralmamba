from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


def _distributed_grad_sync_worker(rank: int, world_size: int, init_method: str) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        first = torch.nn.Parameter(torch.ones(2, 2))
        second = torch.nn.Parameter(torch.ones(2, 2))
        if rank == 0:
            first.grad = torch.ones_like(first)
            second.grad = None
        else:
            first.grad = None
            second.grad = torch.full_like(second, 3.0)

        fsdp2_train._sync_replicated_param_grads(
            (("first", first), ("second", second)),
            world_size=world_size,
            bucket_cap_mb=1.0,
        )
        torch.testing.assert_close(first.grad, torch.full_like(first, 0.5))
        torch.testing.assert_close(second.grad, torch.full_like(second, 1.5))
    finally:
        torch.distributed.destroy_process_group()


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

    def test_replicated_params_use_model_order_not_set_order(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 2),
        )
        ignored = {model[1].bias, model[0].weight, model[1].weight}
        ordered = fsdp2_train._ordered_replicated_params(model, ignored)
        self.assertEqual(
            [name for name, _ in ordered],
            ["0.weight", "1.weight", "1.bias"],
        )

    def test_replicated_grad_sync_handles_rank_local_missing_grad(self) -> None:
        first = torch.nn.Parameter(torch.ones(2, 2))
        second = torch.nn.Parameter(torch.ones(2, 2))
        first.grad = None
        second.grad = torch.ones_like(second)
        calls: list[torch.Tensor] = []

        def fake_all_reduce(tensor, op=None):
            del op
            calls.append(tensor.detach().clone())
            if len(calls) == 1:
                # Simulate the first parameter being used by the other rank.
                tensor[0, 0] = 1
            else:
                tensor.mul_(2)

        with mock.patch.object(
            torch.distributed, "all_reduce", side_effect=fake_all_reduce
        ):
            fsdp2_train._sync_replicated_param_grads(
                (("first", first), ("second", second)),
                world_size=2,
                bucket_cap_mb=1.0,
            )

        self.assertIsNotNone(first.grad)
        self.assertTrue(torch.equal(first.grad, torch.zeros_like(first)))
        self.assertTrue(torch.equal(second.grad, torch.ones_like(second)))
        self.assertEqual(len(calls), 2)  # one status + one gradient bucket

    def test_replicated_grad_sync_two_process_gloo(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fsdp2_grad_sync_") as tmp_dir:
            init_method = (Path(tmp_dir) / "store").resolve().as_uri()
            torch.multiprocessing.spawn(
                _distributed_grad_sync_worker,
                args=(2, init_method),
                nprocs=2,
                join=True,
            )

    def test_shared_log_formatter_accepts_adamw_only_record(self) -> None:
        import train

        record = {
            "loss": 1.0,
            "ce_loss": 0.9,
            "router_aux_loss": 0.1,
            "router_z_loss": 0.01,
            "recon": 0.02,
            "assoc": 0.03,
            "assoc_scale": 1.0,
            "expert": 0.0,
            "expert_scale": 0.0,
            "grad_norm": 0.5,
            "adam_lr": 3e-4,
        }
        line = train._format_log_line(0, 10, record)
        self.assertIn("adam_lr=3.00e-04", line)
        self.assertNotIn("muon_lr=", line)


if __name__ == "__main__":
    unittest.main()
