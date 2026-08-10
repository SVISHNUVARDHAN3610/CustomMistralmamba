"""Fast CPU smoke test for scripts/toy_train.py (~10 steps)."""

from __future__ import annotations

import unittest

import torch

from model import HybridForCausalLM, _aux_loss_schedule, count_trainable_params
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


if __name__ == "__main__":
    unittest.main()
