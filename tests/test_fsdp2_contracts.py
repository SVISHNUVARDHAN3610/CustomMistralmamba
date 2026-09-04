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

from utils.fsdp2_muon import (
    MuonDTensor,
    adjust_lr_factor,
    zeropower_via_newtonschulz5,
)
from utils.training_logging import format_training_log_line


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


def _distributed_muon_worker(rank: int, world_size: int, init_method: str) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    torch.distributed.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        full_param = torch.arange(28, dtype=torch.float32).reshape(7, 4) / 10
        full_grad = torch.linspace(-1.0, 1.0, 28).reshape(7, 4)
        param = torch.nn.Parameter(distribute_tensor(full_param, mesh, [Shard(0)]))
        param.grad = distribute_tensor(full_grad, mesh, [Shard(0)])

        lr = 1e-2
        weight_decay = 0.1
        momentum = 0.95
        optimizer = MuonDTensor(
            [param],
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=5,
            adjust_lr_fn="match_rms_adamw",
            gather_buffer_size_mb=1e-5,
        )
        optimizer.step()

        momentum_full = full_grad * (1 - momentum)
        nesterov_update = full_grad.lerp(momentum_full, momentum)
        orthogonal = zeropower_via_newtonschulz5(nesterov_update, 5).float()
        expected = full_param * (1 - lr * weight_decay)
        expected.add_(
            orthogonal,
            alpha=-lr * adjust_lr_factor(tuple(full_param.shape), "match_rms_adamw"),
        )
        actual = param.full_tensor()
        torch.testing.assert_close(actual, expected)

        if hasattr(torch.optim, "Muon"):
            reference_param = torch.nn.Parameter(full_param.clone())
            reference_param.grad = full_grad.clone()
            reference_optimizer = torch.optim.Muon(
                [reference_param],
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                ns_steps=5,
                adjust_lr_fn="match_rms_adamw",
            )
            reference_optimizer.step()
            torch.testing.assert_close(actual, reference_param)

        momentum_state = optimizer.state[param]["momentum_buffer"]
        torch.testing.assert_close(momentum_state.full_tensor(), momentum_full)

        full_optimizer_state = fsdp2_train._consolidate_optimizer_state(
            optimizer, DTensor
        )
        restored_param = torch.nn.Parameter(
            distribute_tensor(full_param.clone(), mesh, [Shard(0)])
        )
        restored_optimizer = MuonDTensor([restored_param], gather_buffer_size_mb=1.0)
        restored_optimizer.load_state_dict(
            fsdp2_train._reshard_optimizer_state(
                full_optimizer_state,
                restored_optimizer,
                torch.device("cpu"),
                distribute_tensor,
            )
        )
        restored_momentum = restored_optimizer.state[restored_param]["momentum_buffer"]
        torch.testing.assert_close(restored_momentum.full_tensor(), momentum_full)
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

    def test_fsdp2_optimizer_is_adamw_only_when_disabled(self) -> None:
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
            no_muon=True,
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

    def test_fsdp2_optimizer_routes_hidden_matrices_to_muon(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.LayerNorm(4),
        )
        args = argparse.Namespace(
            lr=7.5e-4,
            muon_lr=1e-3,
            adam_lr=3e-4,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
            weight_decay=0.1,
            no_muon=False,
            muon_momentum=0.95,
            no_muon_nesterov=False,
            muon_ns_steps=5,
            muon_adjust_lr_fn="match_rms_adamw",
            muon_gather_buffer_mb=32.0,
        )
        captured: dict[str, object] = {}

        def fake_muon(params, **kwargs):
            params = list(params)
            captured["params"] = params
            captured["kwargs"] = kwargs
            return torch.optim.SGD(params, lr=kwargs["lr"])

        with mock.patch.object(fsdp2_train, "MuonDTensor", side_effect=fake_muon):
            optimizers, use_muon, meta = fsdp2_train.build_fsdp2_optimizers(
                model, args=args, logger=logging.getLogger("fsdp2-contract-test")
            )

        self.assertTrue(use_muon)
        self.assertEqual(meta["optimizer_policy"], "fsdp2_muon_adamw")
        self.assertEqual(len(optimizers), 2)
        self.assertIsInstance(optimizers[0], torch.optim.SGD)
        self.assertIsInstance(optimizers[1], torch.optim.AdamW)
        self.assertEqual(captured["params"], [model[0].weight])
        self.assertEqual(captured["kwargs"]["gather_buffer_size_mb"], 32.0)

        optimized = {
            id(param)
            for optimizer in optimizers
            for group in optimizer.param_groups
            for param in group["params"]
        }
        expected = {id(param) for param in model.parameters()}
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

    def test_muon_dtensor_two_process_gloo(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fsdp2_muon_") as tmp_dir:
            init_method = (Path(tmp_dir) / "store").resolve().as_uri()
            torch.multiprocessing.spawn(
                _distributed_muon_worker,
                args=(2, init_method),
                nprocs=2,
                join=True,
            )

    def test_shared_log_formatter_accepts_adamw_only_record(self) -> None:
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
        line = format_training_log_line(0, 10, record)
        self.assertIn("adam_lr=3.00e-04", line)
        self.assertNotIn("muon_lr=", line)

        line = format_training_log_line(0, 10, {**record, "muon_lr": 6e-4})
        self.assertIn("muon_lr=6.00e-04", line)
        self.assertIn("adam_lr=3.00e-04", line)

        line = format_training_log_line(
            0, 10, {key: value for key, value in record.items() if key != "adam_lr"}
        )
        self.assertIn("lr=n/a", line)


if __name__ == "__main__":
    unittest.main()
