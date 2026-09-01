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

    def test_gradient_accumulation_smoke(self) -> None:
        torch.manual_seed(0)
        cfg = build_toy_config()
        cfg.memory_chunk_size = 128
        model = HybridForCausalLM(cfg).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        accum_steps = 2
        optimizer.zero_grad(set_to_none=True)

        for micro in range(accum_steps):
            ids = torch.randint(0, cfg.vocab_size, (1, 256))
            labels = ids.roll(shifts=-1, dims=1)
            out = model(input_ids=ids, labels=labels)
            assert out.loss is not None
            (out.loss / accum_steps).backward()

        # Check gradients exist and are finite
        has_grad = False
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                has_grad = True
                self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue(has_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)



if __name__ == "__main__":
    unittest.main()
