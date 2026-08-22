"""Unit tests for Hybrid Mamba–MoE model package (laptop-safe ~10M param config)."""

from __future__ import annotations

import unittest
import warnings
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
from model.hybrid.layer import _hybrid_layer_forward
from model.hybrid.losses import (
    _aux_loss_schedule,
    _expert_loss_schedule,
    associative_retrieval_loss,
)
from model.hybrid.mamba import (
    MambaBlock,
    _compute_batch_has_padding,
    fused_mamba_scan_available,
    get_mamba_scan_stats,
    reset_mamba_scan_stats,
)
from model.hybrid.memory import (
    CompressiveMemoryBank,
    MemoryWriteBuffer,
    _batched_memory_summarize,
    _write_buffer_token_len,
    batched_dual_memory_read,
    batched_dual_memory_write,
)
from model.layers.moe import DroplessMoELayer, MOERouter, SwiGLUExpert


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
        full = count_trainable_params(HybridForCausalLM(cfg), exclude_training_aux=True)
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
                buf_lens.append(_write_buffer_token_len(out.write_buffers[0]))
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

    def _decode_with_memory(
        self,
        model: HybridForCausalLM,
        prompt: torch.Tensor,
        n_new: int,
        write_interval: int,
    ) -> list:
        """Run prefill + n_new greedy decode steps; return final memory_states."""
        model.eval()
        write_interval = max(1, write_interval)
        batch = prompt.size(0)
        device = prompt.device
        mask = torch.ones_like(prompt)
        out = model.forward(
            input_ids=prompt,
            attention_mask=mask,
            past_seen_tokens=0,
            use_cache=True,
            skip_memory_write=False,
        )
        pk, mem, mc, wb = (
            out.past_key_values,
            out.memory_states,
            out.mamba_caches,
            out.write_buffers,
        )
        past_seen = prompt.size(1)
        tokens_in_buf = 0
        for _ in range(n_new):
            logits = out.logits[:, -1, :]
            tok = torch.argmax(logits, dim=-1, keepdim=True)
            tokens_in_buf += 1
            do_write = tokens_in_buf >= write_interval
            mask = torch.cat(
                [mask, torch.ones(batch, 1, dtype=mask.dtype, device=device)], dim=1
            )
            step_mask = mask
            if step_mask.size(1) > model.config.window_size:
                step_mask = step_mask[:, -model.config.window_size :]
            pos = torch.full((batch, 1), past_seen, dtype=torch.long, device=device)
            out = model.forward(
                input_ids=tok,
                attention_mask=step_mask,
                position_ids=pos,
                past_key_values=pk,
                memory_states=mem,
                mamba_caches=mc,
                write_buffers=wb,
                past_seen_tokens=past_seen,
                use_cache=True,
                skip_memory_write=not do_write,
            )
            pk, mem, mc, wb = (
                out.past_key_values,
                out.memory_states,
                out.mamba_caches,
                out.write_buffers,
            )
            past_seen += 1
            if do_write:
                tokens_in_buf = 0
        if wb is not None and any(b is not None for b in wb):
            mem, _ = model._flush_memory_write_buffers(mem, wb)
        return mem

    def test_memory_write_buffer_equivalence(self) -> None:
        """_flush_memory_write_buffers should match explicit materialize+write."""
        torch.manual_seed(7)
        cfg = _small_hybrid_config(memory_chunk_size=4)
        model = HybridForCausalLM(cfg).eval()
        layer = model.model.layers[0]
        device = next(model.parameters()).device
        mem = layer.init_memory_state(1, device, torch.float32)
        assert mem is not None
        a_mem, s_mem = mem
        h = cfg.hidden_size
        attn_chunks = [torch.randn(1, 2, h), torch.randn(1, 2, h)]
        mamba_chunks = [torch.randn(1, 2, h), torch.randn(1, 2, h)]
        buf = MemoryWriteBuffer(1, h, capacity=4)
        buf.append(attn_chunks[0], mamba_chunks[0])
        buf.append(attn_chunks[1], mamba_chunks[1])

        flushed, _ = model._flush_memory_write_buffers([mem], [buf])
        assert flushed is not None
        buf_attn, buf_mamba = (
            torch.cat(attn_chunks, dim=1),
            torch.cat(mamba_chunks, dim=1),
        )
        exp_a, _, _ = layer.attn_memory_bank.write(buf_attn, a_mem)
        exp_s, _, _ = layer.state_memory_bank.write(buf_mamba, s_mem)

        self.assertLess((flushed[0][0] - exp_a).abs().max().item(), 1e-5)
        self.assertLess((flushed[0][1] - exp_s).abs().max().item(), 1e-5)

    def test_prefill_chunked_memory_writes(self) -> None:
        """Chunked prefill in generate() should match explicit chunk loop."""
        torch.manual_seed(11)
        cfg = _small_hybrid_config(memory_write_interval=4, memory_chunk_size=4)
        prompt = torch.randint(1, cfg.vocab_size, (1, 10))
        model = HybridForCausalLM(cfg).eval()

        final_mem: list | None = None
        orig_forward = model.forward

        def spy_forward(*args, **kwargs):
            nonlocal final_mem
            out = orig_forward(*args, **kwargs)
            final_mem = out.memory_states
            return out

        model.forward = spy_forward  # type: ignore[method-assign]
        gen = model.generate(prompt, max_new_tokens=0, do_sample=False)
        model.forward = orig_forward  # type: ignore[method-assign]

        manual = model.forward(input_ids=prompt[:, :4], use_cache=True)
        pk, mem, mc, wb = (
            manual.past_key_values,
            manual.memory_states,
            manual.mamba_caches,
            manual.write_buffers,
        )
        for start in range(4, 10, 4):
            end = min(start + 4, 10)
            chunk = prompt[:, start:end]
            manual = model.forward(
                input_ids=chunk,
                attention_mask=torch.ones(1, end, dtype=torch.long),
                position_ids=torch.arange(start, end).unsqueeze(0),
                past_key_values=pk,
                memory_states=mem,
                mamba_caches=mc,
                write_buffers=wb,
                past_seen_tokens=start,
                use_cache=True,
                skip_memory_write=False,
            )
            pk, mem, mc, wb = (
                manual.past_key_values,
                manual.memory_states,
                manual.mamba_caches,
                manual.write_buffers,
            )

        assert final_mem is not None
        for layer in range(cfg.num_layers):
            d = (final_mem[layer][0] - mem[layer][0]).abs().max().item()
            self.assertLess(d, 1e-4)
        self.assertEqual(gen.shape[1], 10)

    def test_padding_mask_grid_no_state_corruption(self) -> None:
        cfg = _small_hybrid_config(num_layers=1)
        model = HybridForCausalLM(cfg).eval()
        ids = torch.randint(1, cfg.vocab_size, (4, 6))
        masks = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        init_mem = model.model.init_memory_states(
            4, ids.device, model.model.embed_tokens.weight.dtype
        )
        init_mamba = model.model.allocate_mamba_caches(
            4, ids.device, model.model.embed_tokens.weight.dtype
        )
        out = model(
            input_ids=ids,
            attention_mask=masks,
            memory_states=init_mem,
            mamba_caches=init_mamba,
            use_cache=True,
        )
        self.assertFalse(torch.isnan(out.logits).any())
        self.assertFalse(torch.isinf(out.logits).any())
        row0_mem_before = init_mem[0][0][3].clone()
        row0_mamba_before = init_mamba[0][1][3].clone()
        self.assertLess(
            (out.memory_states[0][0][3] - row0_mem_before).abs().max().item(), 1e-5
        )
        self.assertLess(
            (out.mamba_caches[0][1][3] - row0_mamba_before).abs().max().item(), 1e-4
        )

    def test_finished_row_caches_frozen(self) -> None:
        cfg = _small_hybrid_config(num_layers=1)
        model = HybridForCausalLM(cfg).eval()
        eos = cfg.eos_token_id
        prompt = torch.randint(1, cfg.vocab_size, (2, 4))

        snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
        orig_forward = model.forward

        def spy_forward(*args, **kwargs):
            out = orig_forward(*args, **kwargs)
            ids = kwargs.get("input_ids")
            if ids is not None and ids.size(1) == 1:
                mc = out.mamba_caches[0]
                mem = out.memory_states[0][0]
                snapshots.append((mc[1][0].clone(), mem[0].clone()))
            return out

        model.forward = spy_forward  # type: ignore[method-assign]
        argmax_calls = {"n": 0}
        real_argmax = torch.argmax

        def mock_argmax(input, dim=-1, keepdim=False):
            if input.dim() == 2 and input.size(0) == 2 and argmax_calls["n"] == 0:
                argmax_calls["n"] += 1
                out = torch.tensor([[eos], [5]], device=input.device, dtype=torch.long)
                return out if keepdim else out.squeeze(dim)
            return real_argmax(input, dim=dim, keepdim=keepdim)

        with mock.patch.object(torch, "argmax", side_effect=mock_argmax):
            model.generate(prompt, max_new_tokens=4, do_sample=False, eos_token_id=eos)

        self.assertGreaterEqual(len(snapshots), 2)
        for i in range(1, len(snapshots)):
            self.assertTrue(torch.equal(snapshots[i][0], snapshots[0][0]))
            self.assertTrue(torch.equal(snapshots[i][1], snapshots[0][1]))

    def test_chunked_forward_matches_manual_chunks(self) -> None:
        torch.manual_seed(3)
        cfg = _small_hybrid_config(memory_chunk_size=8)
        model = HybridForCausalLM(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 24))
        auto = model(input_ids=ids)

        memory_states = None
        logits_parts: list[torch.Tensor] = []
        for start in range(0, 24, 8):
            end = start + 8
            chunk_ids = ids[:, start:end]
            chunk_pos = torch.arange(start, end).unsqueeze(0)
            (
                hidden,
                _aux,
                _zloss,
                _,
                memory_states,
                _,
                _,
                _,
                _,
            ) = model.model(
                input_ids=chunk_ids,
                memory_states=memory_states,
                position_ids=chunk_pos,
                use_cache=False,
            )
            logits_parts.append(model.lm_head(hidden))
        manual_logits = torch.cat(logits_parts, dim=1)

        logit_diff = (auto.logits - manual_logits).abs().max().item()
        self.assertLess(logit_diff, 1e-4)
        for layer in range(cfg.num_layers):
            d = (
                (auto.memory_states[layer][0] - memory_states[layer][0])
                .abs()
                .max()
                .item()
            )
            self.assertLess(d, 1e-4)

    def test_chunk_size_ge_seq_len_matches_single_forward(self) -> None:
        torch.manual_seed(5)
        cfg = _small_hybrid_config(memory_chunk_size=32)
        model = HybridForCausalLM(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 24))
        auto = model(input_ids=ids)
        single = model.model(input_ids=ids)
        single_logits = model.lm_head(single[0])
        diff = (auto.logits - single_logits).abs().max().item()
        self.assertLess(diff, 1e-5)

    def test_chunked_gate_stats_running_mean(self) -> None:
        cfg = _small_hybrid_config(memory_chunk_size=8, num_layers=1)
        model = HybridForCausalLM(cfg)
        model.train()
        ids = torch.randint(0, cfg.vocab_size, (1, 24))
        labels = torch.randint(0, cfg.vocab_size, (1, 24))
        out = model(input_ids=ids, labels=labels)
        keys = [k for k in out.gate_stats if "write_gate" in k]
        self.assertTrue(len(keys) > 0)
        for k in keys:
            v = out.gate_stats[k]
            self.assertGreaterEqual(float(v), 0.0)
            self.assertLessEqual(float(v), 1.0)

    def test_chunked_aux_loss_token_weighting(self) -> None:
        """When chunking is not triggered, aux/z losses match across chunk configs."""
        torch.manual_seed(9)
        ids = torch.randint(0, self.cfg.vocab_size, (1, 8))
        labels = torch.randint(0, self.cfg.vocab_size, (1, 8))
        m8 = HybridForCausalLM(_small_hybrid_config(memory_chunk_size=8)).eval()
        m32 = HybridForCausalLM(_small_hybrid_config(memory_chunk_size=32)).eval()
        m32.load_state_dict(m8.state_dict())
        o8 = m8(input_ids=ids, labels=labels)
        o32 = m32(input_ids=ids, labels=labels)
        self.assertLess(
            abs(o8.router_aux_loss.detach() - o32.router_aux_loss.detach()).item(),
            1e-4,
        )
        self.assertLess(
            abs(o8.router_z_loss.detach() - o32.router_z_loss.detach()).item(), 1e-4
        )

    def test_auxiliary_losses_present_when_training(self) -> None:
        self.model.train()
        ids = torch.randint(0, self.cfg.vocab_size, (2, 16))
        labels = torch.randint(0, self.cfg.vocab_size, (2, 16))
        out = self.model(input_ids=ids, labels=labels)
        self.assertIsNotNone(out.auxiliary_losses)
        aux = out.auxiliary_losses
        assert aux is not None
        for name in ("recon", "gate", "read", "fusion", "expert", "ssm", "slot"):
            val = getattr(aux, name)
            self.assertIsNotNone(val)
            self.assertFalse(torch.isnan(val).any())
        self.assertIsNotNone(out.loss)
        out.loss.backward()

    def test_auxiliary_losses_disabled(self) -> None:
        cfg = _small_hybrid_config(use_auxiliary_losses=False)
        model = HybridForCausalLM(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (1, 12))
        labels = torch.randint(0, cfg.vocab_size, (1, 12))
        out = model(input_ids=ids, labels=labels)
        assert out.auxiliary_losses is not None
        aux = out.auxiliary_losses
        self.assertEqual(aux.recon.item(), 0.0)
        self.assertEqual(aux.fusion.item(), 0.0)
        self.assertEqual(aux.expert.item(), 0.0)

    def test_recon_loss_gradients_write_path(self) -> None:
        cfg = _small_hybrid_config(num_layers=1, use_auxiliary_losses=True)
        model = HybridForCausalLM(cfg).train()
        layer = model.model.layers[0]
        layer.attn_memory_bank.write_gate.weight.grad = None
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        labels = torch.randint(0, cfg.vocab_size, (1, 8))
        out = model(input_ids=ids, labels=labels)
        assert out.loss is not None
        out.loss.backward()
        grad = layer.attn_memory_bank.write_gate.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().max().item(), 0.0)

    def test_chunked_path_respects_assoc_warmup(self) -> None:
        cfg = _small_hybrid_config(memory_chunk_size=8, lambda_assoc=1.0)
        model = HybridForCausalLM(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (1, 24))
        labels = torch.randint(0, cfg.vocab_size, (1, 24))
        out = model(
            input_ids=ids,
            labels=labels,
            training_step=0,
            max_training_steps=1000,
        )
        assert out.auxiliary_losses is not None
        scale = _aux_loss_schedule(0, 1000, cfg.assoc_warmup_fraction)
        self.assertEqual(scale, 0.0)
        weighted_assoc = cfg.lambda_assoc * scale * out.auxiliary_losses.assoc
        self.assertEqual(weighted_assoc.item(), 0.0)

    def test_assoc_loss_uses_squared_l2_norm(self) -> None:
        hidden = 64
        bank = CompressiveMemoryBank(hidden, memory_size=8, num_heads=4)
        x = torch.randn(1, 4, hidden)
        new_mem = bank.init_state(1, x.device, x.dtype)
        residual = torch.ones(1, 4)
        torch.manual_seed(0)
        associative_retrieval_loss(
            bank, x, new_mem, residual, sample_count=4, attention_mask=None
        )
        keys = bank.assoc_key(x)
        values = bank.assoc_val(x)
        retrieved = bank.read_query(keys, new_mem)
        err_sum = (retrieved - values).pow(2).sum(dim=-1).mean()
        self.assertGreater(err_sum.item(), 0.0)
        err_mean = (retrieved - values).pow(2).mean(dim=-1).mean()
        self.assertGreater(err_sum.item(), err_mean.item() * (hidden - 1))

    def test_assoc_loss_padded_batch_no_oob(self) -> None:
        """Variable-length rows must not gather at sentinel index seq_len."""
        hidden = 64
        bank = CompressiveMemoryBank(hidden, memory_size=8, num_heads=4)
        seq_len = 32
        x = torch.randn(4, seq_len, hidden)
        new_mem = bank.init_state(4, x.device, x.dtype)
        residual = torch.rand(4, seq_len)
        mask = torch.zeros(4, seq_len, dtype=torch.long)
        lengths = [8, 12, 20, 31]
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        loss = associative_retrieval_loss(
            bank,
            x,
            new_mem,
            residual,
            sample_count=24,
            attention_mask=mask,
        )
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_assoc_loss_padded_batch_cuda(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        hidden = 64
        bank = CompressiveMemoryBank(hidden, memory_size=8, num_heads=4).cuda()
        seq_len = 32
        x = torch.randn(4, seq_len, hidden, device="cuda")
        new_mem = bank.init_state(4, x.device, x.dtype)
        residual = torch.rand(4, seq_len, device="cuda")
        mask = torch.zeros(4, seq_len, dtype=torch.long, device="cuda")
        for i, length in enumerate([8, 12, 20, 31]):
            mask[i, :length] = 1
        loss = associative_retrieval_loss(
            bank,
            x,
            new_mem,
            residual,
            sample_count=24,
            attention_mask=mask,
        )
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_expert_loss_top_k_1_no_crash(self) -> None:
        cfg = _small_hybrid_config(top_k=1, num_experts=4)
        model = HybridForCausalLM(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (1, 12))
        labels = torch.randint(0, cfg.vocab_size, (1, 12))
        out = model(
            input_ids=ids,
            labels=labels,
            training_step=100,
            max_training_steps=100,
        )
        assert out.auxiliary_losses is not None
        self.assertFalse(torch.isnan(out.auxiliary_losses.expert))

    def test_recon_decoder_single_call_per_bank(self) -> None:
        cfg = _small_hybrid_config(memory_chunk_size=8, num_layers=1)
        model = HybridForCausalLM(cfg).train()
        layer = model.model.layers[0]
        calls = {"attn": 0, "state": 0}
        orig_attn = layer.attn_memory_bank.recon_decoder.forward
        orig_state = layer.state_memory_bank.recon_decoder.forward

        def attn_forward(*args, **kwargs):
            calls["attn"] += 1
            return orig_attn(*args, **kwargs)

        def state_forward(*args, **kwargs):
            calls["state"] += 1
            return orig_state(*args, **kwargs)

        layer.attn_memory_bank.recon_decoder.forward = attn_forward
        layer.state_memory_bank.recon_decoder.forward = state_forward
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        labels = torch.randint(0, cfg.vocab_size, (1, 8))
        out = model(input_ids=ids, labels=labels)
        assert out.loss is not None
        out.loss.backward()
        self.assertEqual(calls["attn"], 1)
        self.assertEqual(calls["state"], 1)

    def test_expert_forward_count_no_duplicate_dispatch(self) -> None:
        cfg = _small_hybrid_config(memory_chunk_size=None, num_layers=1)
        model = HybridForCausalLM(cfg).train()
        expert_calls = 0
        orig = SwiGLUExpert.forward

        def counted_forward(self, x):
            nonlocal expert_calls
            expert_calls += 1
            return orig(self, x)

        with mock.patch.object(SwiGLUExpert, "forward", counted_forward):
            ids = torch.randint(0, cfg.vocab_size, (2, 12))
            labels = torch.randint(0, cfg.vocab_size, (2, 12))
            out = model(
                input_ids=ids,
                labels=labels,
                training_step=100,
                max_training_steps=100,
            )
            assert out.loss is not None
            out.loss.backward()
        baseline_calls = 0

        def baseline_forward(self, x):
            nonlocal baseline_calls
            baseline_calls += 1
            return orig(self, x)

        with mock.patch.object(SwiGLUExpert, "forward", baseline_forward):
            cfg_off = _small_hybrid_config(
                memory_chunk_size=None, num_layers=1, use_auxiliary_losses=False
            )
            model_off = HybridForCausalLM(cfg_off).train()
            model_off.load_state_dict(model.state_dict(), strict=False)
            out_off = model_off(input_ids=ids, labels=labels)
            assert out_off.loss is not None
            out_off.loss.backward()
        self.assertEqual(expert_calls, baseline_calls)

    def test_expert_loss_gated_before_warmup(self) -> None:
        cfg = _small_hybrid_config(memory_chunk_size=None, num_layers=1)
        model = HybridForCausalLM(cfg).train()
        expert_calls = 0
        orig = SwiGLUExpert.forward

        def counted_forward(self, x):
            nonlocal expert_calls
            expert_calls += 1
            return orig(self, x)

        with mock.patch.object(SwiGLUExpert, "forward", counted_forward):
            ids = torch.randint(0, cfg.vocab_size, (2, 12))
            labels = torch.randint(0, cfg.vocab_size, (2, 12))
            out = model(
                input_ids=ids,
                labels=labels,
                training_step=0,
                max_training_steps=100,
            )
            assert out.loss is not None
            out.loss.backward()
        self.assertEqual(_expert_loss_schedule(0, 100, cfg.expert_warmup_fraction), 0.0)
        assert out.auxiliary_losses is not None
        self.assertEqual(out.auxiliary_losses.expert.item(), 0.0)

    def test_mamba_right_pad_cache_init(self) -> None:
        cfg = _small_hybrid_config(debug_state_checks=True)
        model = HybridForCausalLM(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 12))
        mask = torch.ones(1, 12, dtype=torch.long)
        out = model(input_ids=ids, attention_mask=mask, use_cache=True)
        self.assertEqual(out.logits.shape[1], 12)
        bad_mask = torch.tensor([[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with self.assertRaises(ValueError):
            model(input_ids=ids, attention_mask=bad_mask, use_cache=True)

    def test_chunked_streaming_ce_matches_full_ce(self) -> None:
        torch.manual_seed(11)
        cfg = _small_hybrid_config(
            memory_chunk_size=8,
            stream_chunked_ce_loss=True,
            return_logits=True,
        )
        model = HybridForCausalLM(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (1, 24))
        labels = torch.randint(0, cfg.vocab_size, (1, 24))
        out = model(input_ids=ids, labels=labels)
        assert out.ce_loss is not None
        assert out.logits is not None
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=cfg.label_ignore_index)
        manual_ce = loss_fct(out.logits.view(-1, cfg.vocab_size), labels.reshape(-1))
        self.assertLess(abs(out.ce_loss.item() - manual_ce.item()), 1e-5)

    def test_chunked_streaming_ce_no_full_logits(self) -> None:
        cfg = _small_hybrid_config(
            memory_chunk_size=8,
            stream_chunked_ce_loss=True,
            return_logits=False,
        )
        model = HybridForCausalLM(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (1, 24))
        labels = torch.randint(0, cfg.vocab_size, (1, 24))
        out = model(input_ids=ids, labels=labels)
        self.assertIsNone(out.logits)
        self.assertIsNotNone(out.loss)
        assert out.loss is not None
        out.loss.backward()

    def test_fused_mamba_scan_matches_pytorch(self) -> None:
        if not fused_mamba_scan_available() or not torch.cuda.is_available():
            self.skipTest("mamba-ssm fused selective_scan requires CUDA")

        hidden_size = 64
        state_size = 8
        block = (
            MambaBlock(
                hidden_size=hidden_size,
                state_size=state_size,
                expand=2,
                use_fused_scan=False,
            )
            .cuda()
            .eval()
        )

        batch_size, seq_len = 2, 32
        d_inner = block.d_inner
        u = torch.randn(batch_size, seq_len, d_inner, device="cuda")
        dt = torch.randn(batch_size, seq_len, d_inner, device="cuda").abs() + 0.01
        a = -torch.exp(block.A_log.float())
        b_param = torch.randn(batch_size, seq_len, state_size, device="cuda")
        c_param = torch.randn(batch_size, seq_len, state_size, device="cuda")
        d_param = block.D.float()

        y_ref, st_ref = MambaBlock._selective_scan(
            u,
            dt,
            a,
            b_param,
            c_param,
            d_param,
            return_final_state=True,
            use_fused_scan=False,
        )
        y_fused, st_fused = MambaBlock._selective_scan(
            u,
            dt,
            a,
            b_param,
            c_param,
            d_param,
            return_final_state=True,
            use_fused_scan=True,
        )
        self.assertLess((y_ref - y_fused).abs().max().item(), 1e-4)
        self.assertLess((st_ref - st_fused).abs().max().item(), 1e-4)

    def test_fused_mamba_forward_matches_pytorch(self) -> None:
        if not fused_mamba_scan_available() or not torch.cuda.is_available():
            self.skipTest("mamba-ssm fused selective_scan requires CUDA")

        hidden_size = 64
        block_ref = (
            MambaBlock(
                hidden_size=hidden_size,
                state_size=8,
                expand=2,
                use_fused_scan=False,
            )
            .cuda()
            .eval()
        )
        block_fused = (
            MambaBlock(
                hidden_size=hidden_size,
                state_size=8,
                expand=2,
                use_fused_scan=True,
            )
            .cuda()
            .eval()
        )
        block_fused.load_state_dict(block_ref.state_dict())

        x = torch.randn(2, 32, hidden_size, device="cuda")
        out_ref, _, st_ref = block_ref(x)
        out_fused, _, st_fused = block_fused(x)
        self.assertLess((out_ref - out_fused).abs().max().item(), 1e-4)
        self.assertLess((st_ref - st_fused).abs().max().item(), 1e-4)

    def test_fused_mamba_unpadded_with_padding_mask(self) -> None:
        if not fused_mamba_scan_available() or not torch.cuda.is_available():
            self.skipTest("mamba-ssm fused selective_scan requires CUDA")

        hidden_size = 64
        block_ref = (
            MambaBlock(
                hidden_size=hidden_size,
                state_size=8,
                expand=2,
                use_fused_scan=False,
            )
            .cuda()
            .eval()
        )
        block_fused = (
            MambaBlock(
                hidden_size=hidden_size,
                state_size=8,
                expand=2,
                use_fused_scan=True,
            )
            .cuda()
            .eval()
        )
        block_fused.load_state_dict(block_ref.state_dict())

        x = torch.randn(1, 14, hidden_size, device="cuda")
        mask = torch.cat(
            [
                torch.ones(1, 10, dtype=torch.long, device="cuda"),
                torch.zeros(1, 4, dtype=torch.long, device="cuda"),
            ],
            dim=1,
        )
        out_ref, _, st_ref = block_ref(x, attention_mask=mask)
        reset_mamba_scan_stats()
        out_fused, _, st_fused = block_fused(x, attention_mask=mask)
        self.assertLess((out_ref - out_fused).abs().max().item(), 1e-4)
        self.assertLess((st_ref - st_fused).abs().max().item(), 1e-4)
        stats = get_mamba_scan_stats()
        self.assertEqual(stats["fused_unpadded_batch"], 1)
        self.assertEqual(stats["pytorch_fallback"], 0)

    def test_fused_mamba_unpadded_backward_no_nan(self) -> None:
        """Regression: in-place pad restore broke autograd → NaNs in training."""
        if not fused_mamba_scan_available() or not torch.cuda.is_available():
            self.skipTest("mamba-ssm fused selective_scan requires CUDA")

        hidden_size = 64
        block = (
            MambaBlock(
                hidden_size=hidden_size,
                state_size=8,
                expand=2,
                use_fused_scan=True,
            )
            .cuda()
            .train()
        )
        x = torch.randn(2, 16, hidden_size, device="cuda", requires_grad=True)
        mask = torch.tensor(
            [[1] * 12 + [0] * 4, [1] * 8 + [0] * 8],
            dtype=torch.long,
            device="cuda",
        )
        reset_mamba_scan_stats()
        out, _, ssm_state = block(x, attention_mask=mask)
        loss = out.pow(2).mean()
        if ssm_state is not None:
            loss = loss + ssm_state.pow(2).mean()
        loss.backward()

        self.assertFalse(torch.isnan(out).any().item())
        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertFalse(torch.isnan(x.grad).any().item())
        self.assertGreater(x.grad.abs().sum().item(), 0.0)
        for name, param in block.named_parameters():
            if param.grad is not None:
                self.assertFalse(
                    torch.isnan(param.grad).any().item(),
                    msg=f"NaN grad in {name}",
                )
        stats = get_mamba_scan_stats()
        self.assertEqual(stats["fused_unpadded_batch"], 1)
        self.assertEqual(stats["pytorch_fallback"], 0)

    def test_unpadded_fused_restore_preserves_autograd(self) -> None:
        """CPU: F.pad+stack restore keeps grad into per-row scan outputs."""

        def fake_fused(u, dt, A, B, C, D, return_final_state=False):
            y = u * 2.0 + dt.sum(dim=-1, keepdim=True)
            st = None
            if return_final_state:
                # Match fused scan final state shape [B, d_inner, n].
                st = B[:, -1, :].unsqueeze(1).expand(-1, u.size(-1), -1).contiguous()
            return y, st

        batch, seq_len, d_inner, n = 2, 8, 4, 3
        u = torch.randn(batch, seq_len, d_inner, requires_grad=True)
        dt = torch.randn(batch, seq_len, d_inner, requires_grad=True)
        A = torch.randn(d_inner, n)
        B = torch.randn(batch, seq_len, n, requires_grad=True)
        C = torch.randn(batch, seq_len, n)
        D = torch.ones(d_inner)
        mask = torch.tensor(
            [[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]],
            dtype=torch.long,
        )
        with mock.patch.object(
            MambaBlock, "_fused_selective_scan", side_effect=fake_fused
        ):
            y, st = MambaBlock._fused_selective_scan_unpadded(
                u, dt, A, B, C, D, mask, return_final_state=True
            )
        self.assertEqual(tuple(y.shape), (batch, seq_len, d_inner))
        self.assertTrue(torch.equal(y[0, 5:], torch.zeros(3, d_inner)))
        self.assertTrue(torch.equal(y[1, 3:], torch.zeros(5, d_inner)))
        loss = y.pow(2).mean() + st.pow(2).mean()
        loss.backward()
        self.assertIsNotNone(u.grad)
        assert u.grad is not None
        self.assertFalse(torch.isnan(u.grad).any().item())
        # Gradients only on valid prefixes.
        self.assertGreater(u.grad[0, :5].abs().sum().item(), 0.0)
        self.assertEqual(u.grad[0, 5:].abs().sum().item(), 0.0)
        self.assertGreater(u.grad[1, :3].abs().sum().item(), 0.0)
        self.assertEqual(u.grad[1, 3:].abs().sum().item(), 0.0)

    def test_hybrid_padded_train_step_no_nan(self) -> None:
        """Full hybrid train step with padding must not NaN after one backward."""
        if not fused_mamba_scan_available() or not torch.cuda.is_available():
            self.skipTest("mamba-ssm fused selective_scan requires CUDA")

        cfg = _small_hybrid_config(
            num_layers=2,
            use_fused_mamba_scan=True,
            use_auxiliary_losses=True,
            memory_chunk_size=8,
            return_logits=False,
            stream_chunked_ce_loss=True,
        )
        model = HybridForCausalLM(cfg).cuda().train()
        ids = torch.randint(0, cfg.vocab_size, (2, 12), device="cuda")
        mask = torch.tensor(
            [[1] * 10 + [0] * 2, [1] * 7 + [0] * 5],
            dtype=torch.long,
            device="cuda",
        )
        labels = ids.clone()
        labels = labels.masked_fill(mask == 0, cfg.label_ignore_index)
        reset_mamba_scan_stats()
        out = model(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
            training_step=5,
            max_training_steps=100,
        )
        self.assertIsNotNone(out.loss)
        assert out.loss is not None
        self.assertFalse(torch.isnan(out.loss).item())
        out.loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertFalse(
                    torch.isnan(param.grad).any().item(),
                    msg=f"NaN grad in {name}",
                )
        stats = get_mamba_scan_stats()
        self.assertGreater(stats["fused_unpadded_batch"], 0)
        self.assertEqual(stats["pytorch_fallback"], 0)
        cfg = _small_hybrid_config(use_auxiliary_losses=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            HybridForCausalLM(cfg)
        self.assertTrue(
            any("use_auxiliary_losses=False" in str(w.message) for w in caught)
        )

    def test_ssm_gamma_survives_state_dict_roundtrip(self) -> None:
        cfg = _small_hybrid_config(num_layers=2, use_auxiliary_losses=True)
        model = HybridForCausalLM(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        model(input_ids=ids, labels=ids)
        gammas_before = model.model.ssm_norm_gammas.clone()
        cal_before = model.model.ssm_gammas_calibrated.clone()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        model2 = HybridForCausalLM(cfg).train()
        model2.load_state_dict(state)
        with mock.patch.object(
            model2.model,
            "calibrate_ssm_norm_thresholds",
            wraps=model2.model.calibrate_ssm_norm_thresholds,
        ) as spy:
            model2(input_ids=ids, labels=ids)
            spy.assert_not_called()
        self.assertTrue(torch.allclose(model2.model.ssm_norm_gammas, gammas_before))
        self.assertTrue(torch.allclose(model2.model.ssm_gammas_calibrated, cal_before))

    def test_blocked_scan_matches_sequential(self) -> None:
        torch.manual_seed(0)
        b, seq_len, d, n = 2, 128, 16, 8
        delta_a = torch.rand(b, seq_len, d, n) * 0.5 + 0.5
        delta_b_u = torch.randn(b, seq_len, d, n)
        blocked = MambaBlock._blocked_associative_scan(
            delta_a, delta_b_u, block_size=32
        )
        sequential = MambaBlock._sequential_associative_scan(delta_a, delta_b_u)
        self.assertLess((blocked - sequential).abs().max().item(), 1e-5)

    def test_scan_dispatch_parallel_for_short_seq(self) -> None:
        with mock.patch.object(
            MambaBlock,
            "_parallel_associative_scan",
            wraps=MambaBlock._parallel_associative_scan,
        ) as spy:
            cfg = _small_hybrid_config(
                use_fused_mamba_scan=False, parallel_scan_fallback_max_len=512
            )
            model = HybridForCausalLM(cfg).train()
            ids = torch.randint(0, cfg.vocab_size, (1, 64))
            model(input_ids=ids, labels=ids)
            self.assertGreater(spy.call_count, 0)

    def test_fused_scan_failure_warns_once(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for fused scan failure test")
        import model.hybrid.mamba as mamba_mod

        mamba_mod._FUSED_SCAN_WARNED = False
        hidden_size = 64
        d_inner = hidden_size * 2
        u = torch.randn(1, 8, d_inner, device="cuda")
        dt = torch.rand(1, 8, d_inner, device="cuda")
        a = -torch.exp(torch.randn(d_inner, 8, device="cuda"))
        b = torch.randn(1, 8, 8, device="cuda")
        c = torch.randn(1, 8, 8, device="cuda")
        d = torch.ones(d_inner, device="cuda")
        with (
            mock.patch.object(
                MambaBlock, "_fused_selective_scan", side_effect=RuntimeError("boom")
            ),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            MambaBlock._selective_scan(
                u, dt, a, b, c, d, use_fused_scan=True, training=False
            )
            MambaBlock._selective_scan(
                u, dt, a, b, c, d, use_fused_scan=True, training=False
            )
        fused_warnings = [
            w for w in caught if "fused selective_scan failed" in str(w.message)
        ]
        self.assertEqual(len(fused_warnings), 1)
        mamba_mod._FUSED_SCAN_WARNED = False

    def test_write_buffer_append_linear_cost(self) -> None:
        buf = MemoryWriteBuffer(2, 16, capacity=4)
        for i in range(3):
            t = torch.randn(2, 1, 16)
            buf.append(t, t)
        self.assertEqual(buf.filled, 3)
        self.assertIsNotNone(buf.attn_buf)
        assert buf.attn_buf is not None
        self.assertGreaterEqual(buf.attn_buf.size(1), 3)
        out_a, _out_m, out_mask = buf.materialize()
        self.assertEqual(out_a.size(1), 3)
        self.assertEqual(out_mask.dtype, torch.bool)
        self.assertTrue(out_mask.all())

    def test_grouped_moe_matches_loop_dispatch(self) -> None:
        torch.manual_seed(3)
        hidden, inter, experts, top_k = 64, 128, 4, 2
        router = MOERouter(hidden, experts, top_k)
        expert_list = torch.nn.ModuleList(
            [SwiGLUExpert(hidden, inter) for _ in range(experts)]
        )
        x = torch.randn(3, 7, hidden)
        grouped = DroplessMoELayer(
            router, expert_list, capacity_factor=None, use_grouped_moe_dispatch=True
        )
        loop = DroplessMoELayer(
            router, expert_list, capacity_factor=None, use_grouped_moe_dispatch=False
        )
        loop.load_state_dict(grouped.state_dict())
        out_g, aux_g, z_g, _ = grouped(x)
        out_l, aux_l, z_l, _ = loop(x)
        self.assertLess((out_g - out_l).abs().max().item(), 1e-5)
        self.assertLess(abs(aux_g.item() - aux_l.item()), 1e-5)
        self.assertLess(abs(z_g.item() - z_l.item()), 1e-5)

    def test_batched_memory_read_matches_separate(self) -> None:
        cfg = _small_hybrid_config(num_layers=1)
        model = HybridForCausalLM(cfg).eval()
        layer = model.model.layers[0]
        x = torch.randn(2, 5, cfg.hidden_size)
        a_mem = layer.attn_memory_bank.init_state(2, x.device, x.dtype)
        s_mem = layer.state_memory_bank.init_state(2, x.device, x.dtype)
        a_sep = layer.attn_memory_bank.read(x, a_mem)
        s_sep = layer.state_memory_bank.read(x, s_mem)
        a_bat, s_bat = batched_dual_memory_read(
            layer.attn_memory_bank, layer.state_memory_bank, x, a_mem, s_mem
        )
        self.assertLess((a_sep - a_bat).abs().max().item(), 1e-5)
        self.assertLess((s_sep - s_bat).abs().max().item(), 1e-5)

    def test_padding_flag_hoisted(self) -> None:
        cfg = _small_hybrid_config(num_layers=3)
        model = HybridForCausalLM(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 12))
        mask = torch.ones(2, 12, dtype=torch.long)
        mask[0, -2:] = 0
        with mock.patch(
            "model.hybrid.model._compute_batch_has_padding",
            wraps=_compute_batch_has_padding,
        ) as spy:
            model(input_ids=ids, attention_mask=mask)
            self.assertEqual(spy.call_count, 1)

    def test_no_double_checkpoint_when_layer_ckpt_on(self) -> None:
        cfg = _small_hybrid_config(
            gradient_checkpointing=True,
            memory_chunk_size=None,
            use_parallel_scan=False,
            parallel_scan_fallback_max_len=8,
            sequential_scan_min_len=64,
        )
        model = HybridForCausalLM(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (1, 32))
        labels = torch.randint(0, cfg.vocab_size, (1, 32))
        with mock.patch(
            "model.hybrid.model.checkpoint",
            wraps=torch.utils.checkpoint.checkpoint,
        ) as spy:
            out = model(input_ids=ids, labels=labels)
            assert out.loss is not None
            forward_calls = spy.call_args_list[: spy.call_count]
        out.loss.backward()

        layer_ckpts = [
            c for c in forward_calls if c[0] and c[0][0] is _hybrid_layer_forward
        ]
        scan_ckpts = [
            c for c in forward_calls if c[0] and c[0][0] is not _hybrid_layer_forward
        ]
        self.assertEqual(len(layer_ckpts), cfg.num_layers)
        self.assertEqual(len(scan_ckpts), 0)

    def test_torch_compile_forward_parity(self) -> None:
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile unavailable")
        torch.manual_seed(0)
        cfg = _small_hybrid_config(use_torch_compile=False, memory_chunk_size=None)
        model = HybridForCausalLM(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            logits_ref = model(input_ids=ids).logits
        compile_backend = "inductor" if torch.cuda.is_available() else "aot_eager"
        try:
            model.model.layers = torch.nn.ModuleList(
                [
                    torch.compile(
                        layer,
                        mode=cfg.torch_compile_mode,
                        backend=compile_backend,
                    )  # type: ignore[arg-type]
                    for layer in model.model.layers
                ]
            )
        except (RuntimeError, TypeError) as exc:
            self.skipTest(f"torch.compile unavailable: {exc}")
        try:
            with torch.no_grad():
                logits_cmp = model(input_ids=ids).logits
        except (RuntimeError, TypeError, OSError) as exc:
            self.skipTest(f"torch.compile unavailable: {exc}")
        assert logits_ref is not None and logits_cmp is not None
        self.assertLess((logits_ref - logits_cmp).abs().max().item(), 1e-3)

    def test_grouped_gemm_moe_matches_loop(self) -> None:
        torch.manual_seed(5)
        hidden, inter, experts, top_k = 64, 128, 4, 2
        router = MOERouter(hidden, experts, top_k)
        expert_list = torch.nn.ModuleList(
            [SwiGLUExpert(hidden, inter) for _ in range(experts)]
        )
        x = torch.randn(3, 7, hidden)
        gemm = DroplessMoELayer(
            router,
            expert_list,
            capacity_factor=None,
            use_grouped_moe_dispatch=False,
            use_grouped_gemm=True,
        )
        loop = DroplessMoELayer(
            router,
            expert_list,
            capacity_factor=None,
            use_grouped_moe_dispatch=False,
            use_grouped_gemm=False,
        )
        loop.load_state_dict(gemm.state_dict())
        out_g, _, _, _ = gemm(x)
        out_l, _, _, _ = loop(x)
        self.assertLess((out_g - out_l).abs().max().item(), 1e-4)

    def test_batched_dual_memory_write_matches_separate(self) -> None:
        cfg = _small_hybrid_config(num_layers=1)
        model = HybridForCausalLM(cfg).eval()
        layer = model.model.layers[0]
        h = cfg.hidden_size
        buf_attn = torch.randn(2, 4, h)
        buf_mamba = torch.randn(2, 4, h)
        a_mem = layer.attn_memory_bank.init_state(2, buf_attn.device, buf_attn.dtype)
        s_mem = layer.state_memory_bank.init_state(2, buf_mamba.device, buf_mamba.dtype)
        exp_a, _, _ = layer.attn_memory_bank.write(buf_attn, a_mem)
        exp_s, _, _ = layer.state_memory_bank.write(buf_mamba, s_mem)
        new_a, _, _, new_s, _, _ = batched_dual_memory_write(
            layer.attn_memory_bank,
            layer.state_memory_bank,
            buf_attn,
            buf_mamba,
            a_mem,
            s_mem,
        )
        self.assertLess((exp_a - new_a).abs().max().item(), 1e-4)
        self.assertLess((exp_s - new_s).abs().max().item(), 1e-4)

    def test_batched_memory_summarize_all_masked_no_nan(self) -> None:
        """All-padding write chunks must not NaN summary attention (chunked training)."""
        cfg = _small_hybrid_config(num_layers=1)
        model = HybridForCausalLM(cfg).eval()
        layer = model.model.layers[0]
        h = cfg.hidden_size
        buf_attn = torch.randn(2, 8, h)
        buf_mamba = torch.randn(2, 8, h)
        kpm = torch.ones(2, 8, dtype=torch.bool)
        a_sum, s_sum = _batched_memory_summarize(
            layer.attn_memory_bank,
            layer.state_memory_bank,
            buf_attn,
            buf_mamba,
            kpm,
            fast_path=False,
        )
        self.assertFalse(torch.isnan(a_sum).any())
        self.assertFalse(torch.isnan(s_sum).any())
        self.assertEqual(a_sum[0].abs().sum().item(), 0.0)
        self.assertEqual(s_sum[0].abs().sum().item(), 0.0)

    def test_chunked_train_short_row_second_chunk_no_nan(self) -> None:
        """valid_len < chunk_size with seq_len > chunk_size must not NaN aux losses."""
        cfg = _small_hybrid_config(
            num_layers=2,
            use_auxiliary_losses=True,
            memory_chunk_size=16,
            stream_chunked_ce_loss=True,
            return_logits=False,
            use_fused_mamba_scan=False,
        )
        model = HybridForCausalLM(cfg).train()
        ids = torch.randint(1, cfg.vocab_size, (2, 32))
        mask = torch.zeros(2, 32, dtype=torch.long)
        mask[0, :] = 1
        mask[1, :10] = 1
        labels = ids.clone()
        labels = labels.masked_fill(mask == 0, cfg.label_ignore_index)
        out = model(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
            training_step=10,
            max_training_steps=100,
        )
        self.assertIsNotNone(out.loss)
        assert out.loss is not None
        self.assertTrue(torch.isfinite(out.loss).item())
        aux = out.auxiliary_losses
        assert aux is not None
        self.assertTrue(torch.isfinite(aux.recon).item())
        self.assertTrue(torch.isfinite(aux.gate).item())

    def test_write_buffer_preserves_validity_mask(self) -> None:
        """Prior pads must stay invalid across appends (no torch.ones reconstruction)."""
        buf = MemoryWriteBuffer(2, 8, capacity=4)
        a0 = torch.randn(2, 3, 8)
        m0 = torch.randn(2, 3, 8)
        mask0 = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        buf.append(a0, m0, mask0)
        a1 = torch.randn(2, 2, 8)
        m1 = torch.randn(2, 2, 8)
        mask1 = torch.tensor([[0, 0], [1, 1]], dtype=torch.bool)
        buf.append(a1, m1, mask1)
        attn, mamba, valid = buf.materialize()
        self.assertEqual(tuple(valid.shape), (2, 5))
        self.assertEqual(valid[0].tolist(), [True, True, False, False, False])
        self.assertEqual(valid[1].tolist(), [True, False, False, True, True])
        # Padded slots stored as zeros.
        self.assertEqual(attn[0, 2].abs().sum().item(), 0.0)
        self.assertEqual(mamba[1, 1].abs().sum().item(), 0.0)

    def test_cloud_style_padded_second_chunk_no_nan(self) -> None:
        """Exact Kaggle trigger: seq=512-like split with one row shorter than chunk."""
        cfg = _small_hybrid_config(
            num_layers=2,
            use_auxiliary_losses=True,
            memory_chunk_size=16,
            stream_chunked_ce_loss=True,
            return_logits=False,
            use_fused_mamba_scan=False,
        )
        model = HybridForCausalLM(cfg).train()
        # Analogous to valid_lens=[512,195] with chunk=256 → second chunk all-pad.
        ids = torch.randint(1, cfg.vocab_size, (2, 32))
        mask = torch.zeros(2, 32, dtype=torch.long)
        mask[0, :] = 1
        mask[1, :12] = 1  # < 16 so chunk [16:32] is all pad for row 1
        labels = ids.clone().masked_fill(mask == 0, cfg.label_ignore_index)
        for step in range(3):
            out = model(
                input_ids=ids,
                attention_mask=mask,
                labels=labels,
                training_step=step,
                max_training_steps=100,
            )
            assert out.loss is not None
            self.assertTrue(
                torch.isfinite(out.loss).item(),
                msg=f"non-finite loss at step={step}",
            )
            aux = out.auxiliary_losses
            assert aux is not None
            self.assertTrue(torch.isfinite(aux.recon).item())
            self.assertTrue(torch.isfinite(aux.gate).item())
            out.loss.backward()
            model.zero_grad(set_to_none=True)

    def test_decode_accumulate_fast_path_buffer_lens(self) -> None:
        buf = MemoryWriteBuffer(1, 16, capacity=8)
        t1 = torch.randn(1, 1, 16)
        buf.append_single_token(t1, t1)
        buf.append_single_token(t1, t1)
        self.assertEqual(buf.filled, 2)
        buf.append(torch.randn(1, 2, 16), torch.randn(1, 2, 16))
        self.assertEqual(buf.filled, 4)

        cfg = _small_hybrid_config(memory_write_interval=4, memory_chunk_size=4)
        model = HybridForCausalLM(cfg).eval()
        prompt = torch.randint(1, cfg.vocab_size, (1, 6))
        decode_flags: list[bool] = []
        orig_forward = model.forward

        def spy_forward(*args, **kwargs):
            decode_flags.append(kwargs.get("decode_accumulate_only", False))
            return orig_forward(*args, **kwargs)

        model.forward = spy_forward  # type: ignore[method-assign]
        model.generate(prompt, max_new_tokens=4, do_sample=False)
        self.assertIn(True, decode_flags)

    def test_cuda_graph_decode_parity(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for CUDA graph decode test")
        torch.manual_seed(9)
        cfg = _small_hybrid_config(
            memory_write_interval=8,
            memory_chunk_size=8,
            use_cuda_graph=True,
        )
        model = HybridForCausalLM(cfg).eval().cuda()
        prompt = torch.randint(1, cfg.vocab_size, (1, 6), device="cuda")
        eager_cfg = _small_hybrid_config(
            memory_write_interval=8,
            memory_chunk_size=8,
            use_cuda_graph=False,
        )
        eager = HybridForCausalLM(eager_cfg).eval().cuda()
        eager.load_state_dict(model.state_dict())
        gen_graph = model.generate(prompt, max_new_tokens=4, do_sample=False)
        gen_eager = eager.generate(prompt, max_new_tokens=4, do_sample=False)
        self.assertTrue(torch.equal(gen_graph, gen_eager))

    def test_qk_norm_forward_backward(self) -> None:
        cfg = _small_hybrid_config(use_qk_norm=True)
        model = HybridForCausalLM(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        labels = torch.randint(0, cfg.vocab_size, (2, 16))
        out = model(input_ids=ids, labels=labels)
        self.assertIsNotNone(out.loss)
        out.loss.backward()
        q_norm_w = model.model.layers[0].attention_block.q_norm.weight
        self.assertIsNotNone(q_norm_w.grad)
        self.assertTrue(torch.isfinite(q_norm_w.grad).all())

    def test_attention_sinks_forward_and_generate(self) -> None:
        cfg = _small_hybrid_config(window_size=8, attention_sink_size=2)
        model = HybridForCausalLM(cfg).eval()
        prompt = torch.randint(1, cfg.vocab_size, (2, 16))
        gen = model.generate(prompt, max_new_tokens=10, do_sample=False)
        self.assertEqual(gen.shape, (2, 26))

    def test_layer_types_interleaving(self) -> None:
        cfg = _small_hybrid_config(num_layers=2, layer_types=["mamba_only", "attn_only"])
        model = HybridForCausalLM(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        labels = torch.randint(0, cfg.vocab_size, (2, 16))
        out = model(input_ids=ids, labels=labels)
        self.assertIsNotNone(out.loss)
        out.loss.backward()
        self.assertEqual(out.logits.shape, (2, 16, cfg.vocab_size))

    def test_shared_rope_cache(self) -> None:
        cfg = _small_hybrid_config(num_layers=2)
        model = HybridForCausalLM(cfg)
        layer0_rope = model.model.layers[0].attention_block.rotary_emb
        layer1_rope = model.model.layers[1].attention_block.rotary_emb
        self.assertIs(layer0_rope, layer1_rope)
        self.assertIs(layer0_rope, model.model.rotary_emb)

    def test_optimizer_partitioning_no_weight_decay(self) -> None:
        from model.core.builders import build_adamw_param_groups, split_muon_adam_params

        cfg = _small_hybrid_config()
        model = HybridForCausalLM(cfg)
        adam_params, _muon_params, inventory, param_names = split_muon_adam_params(model)

        # A_log and D must be in AdamW
        a_log_found = any("A_log" in name for name in inventory["adamw"])
        d_found = any("D" in name for name in inventory["adamw"])
        self.assertTrue(a_log_found, "A_log must be in AdamW parameter group")
        self.assertTrue(d_found, "D must be in AdamW parameter group")

        # Test build_adamw_param_groups
        groups = build_adamw_param_groups(adam_params, weight_decay=0.1, name_lookup=param_names)
        self.assertEqual(len(groups), 2)
        decay_group = next(g for g in groups if g["weight_decay"] > 0)
        self.assertEqual(decay_group["weight_decay"], 0.1)
        no_decay_group = next(g for g in groups if g["weight_decay"] == 0.0)

        # No-decay group must contain A_log, D, norm, embed
        no_decay_names = [param_names[id(p)] for p in no_decay_group["params"]]
        self.assertTrue(any("A_log" in n for n in no_decay_names))
        self.assertTrue(any("D" in n for n in no_decay_names))
        self.assertTrue(any("norm" in n for n in no_decay_names))


if __name__ == "__main__":
    unittest.main()
