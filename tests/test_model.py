"""Unit tests for Hybrid Mamba–MoE model.py (laptop-safe ~10M param config)."""

from __future__ import annotations

import unittest

import torch

from model import (
    HybridForCausalLM,
    HybridMambaMoEConfig,
    MixtralConfig,
    MixtralForCausalLM,
    count_trainable_params,
)


def _small_hybrid_config(**overrides) -> HybridMambaMoEConfig:
    cfg = HybridMambaMoEConfig(
        vocab_size=3200,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        intermediate_size=512,
        window_size=16,
        num_experts=4,
        top_k=2,
        dropout=0.0,
        capacity_factor=None,
        max_position_embeddings=512,
        mamba_state_size=8,
        mamba_conv_kernel=4,
        mamba_expand=2,
        use_dual_memory=True,
        memory_size=16,
        memory_num_heads=4,
        rms_norm_eps=1e-5,
        init_range=0.02,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestHybridModel(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.cfg = _small_hybrid_config()
        self.model = HybridForCausalLM(self.cfg)
        self.model.eval()

    def test_param_budget(self) -> None:
        n = count_trainable_params(self.model)
        self.assertLess(n, 12_000_000)
        self.assertGreater(n, 1_000_000)

    def test_dt_proj_bias_not_zeroed(self) -> None:
        bias = self.model.model.layers[0].mamba_block.dt_proj.bias
        self.assertTrue(getattr(bias, "_no_reinit", False))
        self.assertFalse(torch.allclose(bias, torch.zeros_like(bias)))

    def test_forward_backward(self) -> None:
        self.model.train()
        ids = torch.randint(0, self.cfg.vocab_size, (2, 24))
        labels = torch.randint(0, self.cfg.vocab_size, (2, 24))
        mask = torch.ones(2, 24, dtype=torch.long)
        out = self.model(input_ids=ids, attention_mask=mask, labels=labels)
        self.assertEqual(tuple(out.logits.shape), (2, 24, self.cfg.vocab_size))
        self.assertIsNotNone(out.loss)
        self.assertTrue(len(out.gate_stats) > 0)
        out.loss.backward()

    def test_window_mask_truncation_no_crash(self) -> None:
        """GQA truncates KV and padding mask together past window_size."""
        ids = torch.randint(0, self.cfg.vocab_size, (2, 40))
        mask = torch.ones(2, 40, dtype=torch.long)
        mask[:, :3] = 0
        out = self.model(input_ids=ids, attention_mask=mask, use_cache=True)
        self.assertEqual(out.logits.shape[1], 40)
        # Cached KV must be <= window
        for k, v in out.past_key_values:
            self.assertLessEqual(k.size(2), self.cfg.window_size)
            self.assertEqual(k.size(2), v.size(2))

    def test_cached_vs_full_logits(self) -> None:
        ids = torch.randint(0, self.cfg.vocab_size, (1, 28))
        full = self.model(input_ids=ids, use_cache=False)

        out = self.model(input_ids=ids[:, :12], use_cache=True)
        pk, mem, mc = out.past_key_values, out.memory_states, out.mamba_caches
        past_seen = 12
        for i in range(12, 28):
            am = torch.ones(1, min(i + 1, self.cfg.window_size), dtype=torch.long)
            # Rebuild from full prefix length for correctness of last-window slice
            full_mask = torch.ones(1, i + 1, dtype=torch.long)
            am = full_mask[:, -self.cfg.window_size :]
            out = self.model(
                input_ids=ids[:, i : i + 1],
                attention_mask=am,
                past_key_values=pk,
                memory_states=mem,
                mamba_caches=mc,
                past_seen_tokens=past_seen,
                use_cache=True,
            )
            pk, mem, mc = out.past_key_values, out.memory_states, out.mamba_caches
            past_seen += 1

        cos = torch.nn.functional.cosine_similarity(
            out.logits[:, -1].float().detach(),
            full.logits[:, -1].float().detach(),
            dim=-1,
        )
        self.assertGreater(float(cos), 0.99)

    def test_memory_persists_across_chunks(self) -> None:
        ids = torch.randint(0, self.cfg.vocab_size, (1, 32))
        chunk1 = ids[:, :16]
        chunk2 = ids[:, 16:]
        out1 = self.model(input_ids=chunk1, use_cache=False)
        self.assertIsNotNone(out1.memory_states)
        mem = out1.memory_states
        # Second chunk with carried memory should run and return updated banks
        out2 = self.model(input_ids=chunk2, memory_states=mem, use_cache=False)
        self.assertEqual(len(out2.memory_states), self.cfg.num_layers)
        # Banks should have changed vs carried-in state
        before = mem[0][0]
        after = out2.memory_states[0][0]
        self.assertFalse(torch.allclose(before, after))

    def test_memory_zeroed_inference_differs(self) -> None:
        """Test-1 hook: zeroed banks vs learned init banks change logits."""
        ids = torch.randint(0, self.cfg.vocab_size, (1, 20))
        mask = torch.ones(1, 20, dtype=torch.long)
        normal = self.model(input_ids=ids, attention_mask=mask)
        zeros = self.model.model.zero_memory_states(
            batch_size=1,
            device=ids.device,
            dtype=self.model.model.embed_tokens.weight.dtype,
        )
        zeroed = self.model(input_ids=ids, attention_mask=mask, memory_states=zeros)
        # After the first layer write, states diverge; logits should not be identical
        # for a randomly initialized model they almost always differ.
        delta = (normal.logits - zeroed.logits).abs().mean().item()
        self.assertGreater(delta, 0.0)

    def test_generate_past_window(self) -> None:
        prompt = torch.randint(0, self.cfg.vocab_size, (2, 10))
        gen = self.model.generate(
            prompt,
            max_new_tokens=20,
            do_sample=False,
            attention_mask=torch.ones_like(prompt),
        )
        self.assertEqual(gen.shape, (2, 30))

    def test_mixtral_baseline_still_imports(self) -> None:
        cfg = MixtralConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            head_dim=64,
            intermediate_size=256,
            window_size=32,
            num_experts=2,
            top_k=1,
            max_position_embeddings=128,
            capacity_factor=None,
        )
        m = MixtralForCausalLM(cfg)
        ids = torch.randint(0, 1000, (1, 8))
        out = m(input_ids=ids)
        self.assertEqual(out.logits.shape[-1], 1000)


if __name__ == "__main__":
    unittest.main()
