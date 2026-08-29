"""Fast CPU smoke test for scripts/toy_train.py (~10 steps)."""

from __future__ import annotations

import unittest

import torch

from model.core.builders import count_trainable_params
from model.hybrid.losses import _aux_loss_schedule
from model.hybrid.model import HybridForCausalLM
from scripts.toy_train import build_toy_config


class TestToyTrainSmoke(unittest.TestCase):
    def test_toy_config_param_budget(self) -> None:
        cfg = build_toy_config()
        model = HybridForCausalLM(cfg)
        n = count_trainable_params(model)
        self.assertGreater(n, 4_000_000)
        self.assertLess(n, 6_000_000)

    def test_toy_train_10_steps_finite(self) -> None:
        torch.manual_seed(0)
        cfg = build_toy_config()
        cfg.memory_chunk_size = 256
        model = HybridForCausalLM(cfg).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        max_steps = 10

        for step in range(max_steps):
            ids = torch.randint(0, cfg.vocab_size, (2, 768))
            labels = ids.roll(shifts=-1, dims=1)
            labels[:, -1] = cfg.label_ignore_index
            optimizer.zero_grad(set_to_none=True)
            out = model(
                input_ids=ids,
                labels=labels,
                training_step=step,
                max_training_steps=max_steps,
            )
            assert out.loss is not None
            self.assertTrue(torch.isfinite(out.loss))
            if step == 0:
                assert out.auxiliary_losses is not None
                scale = _aux_loss_schedule(0, max_steps, cfg.assoc_warmup_fraction)
                self.assertEqual(scale, 0.0)
                weighted_assoc = cfg.lambda_assoc * scale * out.auxiliary_losses.assoc
                self.assertEqual(weighted_assoc.item(), 0.0)
            out.loss.backward()
            optimizer.step()


class TestEMASpikeGuard(unittest.TestCase):
    """Issue-2 acceptance: huge-but-finite values are caught and excluded."""

    @staticmethod
    def _load_ema_baseline():
        # train.py imports transformers-dependent utils at module scope; the
        # guard class itself is dependency-free, so load it via AST compile
        # (ruff: noqa for the exec, which is the standard AST-extract trick).
        import ast
        import os

        src_path = os.path.join(os.path.dirname(__file__), "..", "train.py")
        with open(src_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        ns = {"torch": torch}
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "EMABaseline":
                code = compile(
                    ast.Module(body=[node], type_ignores=[]), "train.py", "exec"
                )
                exec(code, ns)  # noqa: S102 - trusted local source, no inputs
                return ns["EMABaseline"]
        raise AssertionError("EMABaseline class not found in train.py")

    def test_spike_detected_and_excluded_from_ema(self) -> None:
        ema_cls = self._load_ema_baseline()
        ema = ema_cls(1, torch.device("cpu"), multiplier=100.0)
        values = torch.tensor([[1.0]] * 20 + [[1e9]] + [[1.0]])
        spikes: list[bool] = []
        for i in range(values.size(0)):
            mask = ema.update(values[i], torch.ones_like(values[i], dtype=torch.bool))
            spikes.append(bool(mask[0]))
        # The 1e9 step (index 20) flags; the return-to-normal step does not.
        self.assertTrue(spikes[20])
        self.assertFalse(spikes[21])
        # The EMA baseline never ingests the spike (no ratchet-up hiding the
        # next spike): after 21 steps of 1.0 it has re-converged toward 1.0.
        self.assertLess(ema.ema.item(), 0.25)
        self.assertGreater(ema.ema.item(), 0.15)

    def test_non_finite_values_never_flag_or_ratchet(self) -> None:
        ema_cls = self._load_ema_baseline()
        ema = ema_cls(1, torch.device("cpu"), multiplier=100.0)
        # NaN is caught by the existing isfinite path, not the spike guard;
        # the EMA must ignore it entirely.
        for _ in range(10):
            mask = ema.update(torch.tensor([1.0]), torch.tensor([True]))
            self.assertFalse(bool(mask[0]))
        mask = ema.update(torch.tensor([float("nan")]), torch.tensor([False]))
        self.assertFalse(bool(mask[0]))
        self.assertTrue(torch.isfinite(ema.ema[0]))

    def test_disabled_multiplier_never_flags(self) -> None:
        ema_cls = self._load_ema_baseline()
        ema = ema_cls(1, torch.device("cpu"), multiplier=0.0)
        for _ in range(8):
            mask = ema.update(torch.tensor([1e30]), torch.tensor([True]))
            self.assertFalse(bool(mask[0]))


if __name__ == "__main__":
    unittest.main()
