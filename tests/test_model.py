"""Unit tests for Hybrid Mamba–MoE model.py (laptop-safe ~10M param config)."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from model import (
    HybridForCausalLM,
    HybridMambaMoEConfig,
    MixtralConfig,
    MixtralForCausalLM,
    build_test3_null_baseline_config,
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
        self.assertLess(n, 20_000_000)
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

    def test_capacity_none_deterministic(self) -> None:
        """Same prompt alone vs batched should match when capacity_factor=None."""
        cfg = _small_hybrid_config(capacity_factor=None)
        model = HybridForCausalLM(cfg).eval()
        prompt = torch.randint(0, cfg.vocab_size, (1, 16))
        alone = model(input_ids=prompt).logits
        batched = model(input_ids=torch.cat([prompt] * 4, dim=0)).logits[:1]
        diff = (alone - batched).abs().max().item()
        self.assertLess(diff, 1e-5)

    def test_memory_write_grad_multi_chunk(self) -> None:
        """Internal chunked BPTT must train memory write parameters."""
        cfg = _small_hybrid_config(memory_chunk_size=8)
        model = HybridForCausalLM(cfg)
        model.train()
        ids = torch.randint(0, cfg.vocab_size, (1, 24))
        labels = torch.randint(0, cfg.vocab_size, (1, 24))
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        write_w = model.model.layers[0].attn_memory_bank.write_gate.weight
        self.assertIsNotNone(write_w.grad)
        self.assertGreater(write_w.grad.abs().sum().item(), 0.0)

    def test_padding_mamba_moe(self) -> None:
        """Valid positions should match between padded and trimmed batches."""
        cfg = _small_hybrid_config()
        model = HybridForCausalLM(cfg).eval()
        ids = torch.randint(1, cfg.vocab_size, (1, 12))
        padded = torch.cat([ids, torch.zeros(1, 4, dtype=torch.long)], dim=1)
        mask = torch.cat(
            [torch.ones(1, 12, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long)],
            dim=1,
        )
        trimmed_logits = model(input_ids=ids).logits
        padded_logits = model(input_ids=padded, attention_mask=mask).logits[:, :12]
        diff = (trimmed_logits - padded_logits).abs().max().item()
        self.assertLess(diff, 1e-4)

    def test_null_baseline_param_match(self) -> None:
        cfg = _small_hybrid_config()
        full = count_trainable_params(HybridForCausalLM(cfg))
        null_cfg = build_test3_null_baseline_config(cfg)
        null_n = count_trainable_params(HybridForCausalLM(null_cfg))
        self.assertFalse(null_cfg.use_dual_memory)
        ratio = null_n / full
        self.assertGreater(ratio, 0.98)
        self.assertLess(ratio, 1.02)

    def test_scan_checkpoint_long_seq(self) -> None:
        """Checkpointed sequential scan should run moderate L without error."""
        cfg = _small_hybrid_config(
            hidden_size=64,
            num_heads=2,
            num_kv_heads=1,
            head_dim=32,
            intermediate_size=128,
            num_layers=1,
            memory_chunk_size=None,
            use_parallel_scan=False,
        )
        model = HybridForCausalLM(cfg)
        model.train()
        ids = torch.randint(0, cfg.vocab_size, (1, 128))
        labels = torch.randint(0, cfg.vocab_size, (1, 128))
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()

    def test_generate_memory_write_interval(self) -> None:
        cfg = _small_hybrid_config(memory_write_interval=4, memory_chunk_size=4)
        model = HybridForCausalLM(cfg).eval()
        torch.manual_seed(42)
        prompt = torch.randint(1, cfg.vocab_size, (1, 6))

        skip_flags: list[bool] = []
        buf_lens: list[int | None] = []
        orig_forward = model.forward

        def spy_forward(*args, **kwargs):
            skip_flags.append(kwargs.get("skip_memory_write", False))
            out = orig_forward(*args, **kwargs)
            if out.write_buffers is not None and out.write_buffers[0] is not None:
                buf_lens.append(out.write_buffers[0][0].size(1))
            else:
                buf_lens.append(None)
            return out

        model.forward = spy_forward  # type: ignore[method-assign]

        gen = model.generate(prompt, max_new_tokens=4, do_sample=False)
        self.assertEqual(gen.shape[1], 10)

        # Prefill is chunked: [0:4] write, [4:6] write, then 4 decode steps.
        self.assertEqual(len(skip_flags), 6)
        self.assertEqual(skip_flags[:2], [False, False])  # prefill chunks write
        # Decode accumulates until interval=4, then writes.
        self.assertEqual(skip_flags[2:], [True, True, True, False])
        # After first three decode skips, buffer holds 1,2,3 tokens; write clears.
        self.assertEqual(buf_lens[2:5], [1, 2, 3])
        self.assertIsNone(buf_lens[5])

    def test_memory_write_all_pad_no_nan(self) -> None:
        cfg = _small_hybrid_config()
        model = HybridForCausalLM(cfg).eval()
        ids = torch.zeros(2, 8, dtype=torch.long)
        mask = torch.zeros(2, 8, dtype=torch.long)
        out = model(input_ids=ids, attention_mask=mask)
        self.assertFalse(torch.isnan(out.logits).any())

    def test_ce_ignore_index_pads(self) -> None:
        cfg = _small_hybrid_config()
        model = HybridForCausalLM(cfg)
        model.train()
        ids = torch.randint(1, cfg.vocab_size, (1, 8))
        labels = ids.clone()
        labels[:, -3:] = cfg.label_ignore_index
        mask = torch.ones(1, 8, dtype=torch.long)
        mask[:, -3:] = 0
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        self.assertIsNotNone(out.loss)
        self.assertFalse(torch.isnan(out.loss))

    def test_mamba_pad_state_matches_trimmed(self) -> None:
        """Right-padded prefill cache should match trimmed sequence cache."""
        cfg = _small_hybrid_config(num_layers=1, memory_chunk_size=None)
        model = HybridForCausalLM(cfg).eval()
        ids = torch.randint(1, cfg.vocab_size, (1, 10))
        padded = torch.cat([ids, torch.zeros(1, 4, dtype=torch.long)], dim=1)
        mask = torch.cat(
            [torch.ones(1, 10, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long)],
            dim=1,
        )
        trimmed = model(input_ids=ids, use_cache=True)
        padded_out = model(input_ids=padded, attention_mask=mask, use_cache=True)
        t_conv, t_ssm = trimmed.mamba_caches[0]
        p_conv, p_ssm = padded_out.mamba_caches[0]
        self.assertLess((t_ssm - p_ssm).abs().max().item(), 1e-4)
        self.assertLess((t_conv - p_conv).abs().max().item(), 1e-4)

    def test_generate_finished_row_zeros_input(self) -> None:
        """Finished batch rows must not feed real token ids into decode forward."""
        cfg = _small_hybrid_config()
        model = HybridForCausalLM(cfg).eval()
        eos = cfg.eos_token_id
        prompt = torch.randint(1, cfg.vocab_size, (2, 4))

        decode_inputs: list[torch.Tensor] = []
        orig_forward = model.forward

        def spy_forward(*args, **kwargs):
            ids = kwargs.get("input_ids")
            if ids is not None and ids.size(1) == 1:
                decode_inputs.append(ids.clone())
            return orig_forward(*args, **kwargs)

        model.forward = spy_forward  # type: ignore[method-assign]

        argmax_calls = {"n": 0}
        real_argmax = torch.argmax

        def mock_argmax(input, dim=-1, keepdim=False):
            if input.dim() == 2 and input.size(0) == 2 and argmax_calls["n"] == 0:
                argmax_calls["n"] += 1
                out = torch.tensor(
                    [[eos], [5]],
                    device=input.device,
                    dtype=torch.long,
                )
                return out if keepdim else out.squeeze(dim)

            return real_argmax(input, dim=dim, keepdim=keepdim)

        with mock.patch.object(torch, "argmax", side_effect=mock_argmax):
            gen = model.generate(
                prompt, max_new_tokens=3, do_sample=False, eos_token_id=eos
            )

        self.assertEqual(gen.shape[0], 2)
        self.assertGreaterEqual(len(decode_inputs), 1)
        # First decode after row 0 hits EOS: finished row must not feed a real token.
        self.assertEqual(decode_inputs[0][0, 0].item(), 0)
        self.assertNotEqual(decode_inputs[0][1, 0].item(), 0)

    def test_moe_capacity_eval_deterministic(self) -> None:
        """Eval mode with capacity_factor should be order-stable across seeds."""
        cfg = _small_hybrid_config(capacity_factor=1.0, num_experts=2, top_k=1)
        model = HybridForCausalLM(cfg).eval()
        ids = torch.randint(1, cfg.vocab_size, (4, 32))

        torch.manual_seed(0)
        logits_a = model(input_ids=ids).logits
        torch.manual_seed(99)
        logits_b = model(input_ids=ids).logits
        diff = (logits_a - logits_b).abs().max().item()
        self.assertLess(diff, 1e-5)


if __name__ == "__main__":
    unittest.main()
