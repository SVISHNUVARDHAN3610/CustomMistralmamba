"""Tiny CPU-only PPO algorithm, rollout, integration and production-guard tests."""

import copy
import json
import logging
import random
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "post-training"))

from RLHF.config import PPOConfig, RewardConfig, smoke_config
from RLHF.evaluation_metrics import RewardStatistics
from RLHF.ppo import (
    PolicyValueModel,
    assert_frozen,
    collect_rollout,
    generalized_advantage_estimate,
    masked_mean,
    ppo_objective,
)
from RLHF.ppo_dataset import PromptShardDataset, prompt_feed
from RLHF.reward_model import RewardModel
from RLHF.rlhf_dataset import SmokeTokenizer
from RLHF.train_reward_model import (
    RewardBackend,
    load_checkpoint,
    require_mamba_kernels,
    resolve_precision,
)
from RLHF.train_rlhf import (
    FAMILY,
    create_smoke_inputs,
    evaluate_policy,
    prepare_sources,
    run_training,
)

from model.core.config import HybridMambaMoEConfig
from model.hybrid.mamba import MambaBlock


def tiny_config():
    return HybridMambaMoEConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        intermediate_size=24,
        num_experts=2,
        top_k=1,
        mamba_state_size=4,
        window_size=32,
        max_position_embeddings=32,
        use_dual_memory=True,
        memory_chunk_size=4,
        memory_size=2,
        memory_num_heads=2,
        use_auxiliary_losses=True,
        use_fused_mamba_scan=False,
    )


class PPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ratio_clipping_and_masking(self):
        cfg = PPOConfig()
        old = torch.zeros(1, 4)
        new = torch.tensor([[2.0, 0.5, 100.0, -100.0]]).log()
        new[0, 2:] = 0
        advantage = torch.tensor([[1.0, -1.0, 999.0, 999.0]])
        mask = torch.tensor([[True, True, False, False]])
        zero = torch.zeros_like(old)
        loss, metrics = ppo_objective(
            new, old, zero, zero, advantage, zero, zero, mask, cfg
        )
        self.assertAlmostEqual(
            metrics["policy_loss"].item(), -(1.2 - 0.8) / 2, places=6
        )
        self.assertAlmostEqual(metrics["clip_fraction"].item(), 1)
        self.assertTrue(torch.isfinite(loss))
        _, metrics = ppo_objective(
            old, old, zero, zero, advantage, zero, zero, mask, cfg
        )
        self.assertEqual(metrics["approx_kl"].item(), 0)
        self.assertEqual(metrics["clip_fraction"].item(), 0)
        self.assertEqual(masked_mean(advantage, mask).item(), 0)

    def test_gae_terminal_and_padding(self):
        rewards = torch.tensor([[0.0, 0.0, 1.0, 100.0], [2.0, 100.0, 100.0, 100.0]])
        values = torch.tensor([[0.1, 0.2, 0.3, 90.0], [0.5, 90.0, 90.0, 90.0]])
        mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
        adv, ret = generalized_advantage_estimate(rewards, values, mask, 1, 1)
        torch.testing.assert_close(
            adv, torch.tensor([[0.9, 0.8, 0.7, 0.0], [1.5, 0.0, 0.0, 0.0]])
        )
        torch.testing.assert_close(
            ret, torch.tensor([[1.0, 1.0, 1.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
        )
        adv, _ = generalized_advantage_estimate(rewards, values, mask, 0.9, 0.5)
        self.assertAlmostEqual(adv[0, 0].item(), 0.25325, places=5)

    def test_value_clipping(self):
        z = torch.zeros(1, 1)
        mask = torch.ones(1, 1, dtype=torch.bool)
        _, metrics = ppo_objective(
            z, z, torch.ones(1, 1), z, z, torch.ones(1, 1), z, mask, PPOConfig()
        )
        self.assertAlmostEqual(metrics["value_loss"].item(), 0.32, places=6)

    def models(self):
        torch.manual_seed(1)
        policy = PolicyValueModel(tiny_config())
        reference = copy.deepcopy(policy).requires_grad_(False).eval()
        cfg = smoke_config("unused")
        cfg.model.vocab_size = 32
        reward = RewardModel(cfg.model).requires_grad_(False).eval()
        backend = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
        return policy, reference, reward, backend

    def test_rollout_replay_and_frozen_models(self):
        policy, reference, reward, backend = self.models()
        cfg = PPOConfig(max_prompt_tokens=6, max_new_tokens=3)
        reference_before = {k: v.clone() for k, v in reference.state_dict().items()}
        reward_before = {k: v.clone() for k, v in reward.state_dict().items()}
        # Force variable EOS lengths while preserving real autoregressive forwards.
        with patch(
            "RLHF.ppo.sample_tokens",
            side_effect=[
                torch.tensor([2, 4]),
                torch.tensor([5, 2]),
                torch.tensor([6, 7]),
            ],
        ):
            rollout = collect_rollout(
                policy, reference, reward, [[1, 3], [1, 4, 5]], cfg, backend, 0, 2
            )
        self.assertEqual(
            rollout.response_mask.tolist(), [[True, False, False], [True, True, False]]
        )
        self.assertEqual(rollout.attention_mask.sum(-1).tolist(), [3, 5])
        self.assertEqual(rollout.eos.tolist(), [True, True])
        logp, values, entropy = policy(
            rollout.input_ids,
            rollout.attention_mask,
            rollout.positions,
            tokens=rollout.tokens,
        )
        self.assertEqual(values.shape, (2, 3))
        torch.testing.assert_close(
            logp[rollout.response_mask],
            rollout.old_logprobs[rollout.response_mask],
            atol=1e-6,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            rollout.old_logprobs[rollout.response_mask],
            rollout.reference_logprobs[rollout.response_mask],
            atol=1e-6,
            rtol=1e-5,
        )
        loss, _ = ppo_objective(
            logp,
            rollout.old_logprobs,
            values,
            rollout.old_values,
            rollout.advantages,
            rollout.returns,
            entropy,
            rollout.response_mask,
            cfg,
        )
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        loss.backward()
        optimizer.step()
        self.assertIsNotNone(policy.value_head.weight.grad)
        for model, before in ((reference, reference_before), (reward, reward_before)):
            assert_frozen(model)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_rollout_determinism(self):
        policy, reference, reward, backend = self.models()
        cfg = PPOConfig(max_prompt_tokens=4, max_new_tokens=3)
        outputs = []
        for _ in range(2):
            torch.manual_seed(15)
            outputs.append(
                collect_rollout(policy, reference, reward, [[1, 4]], cfg, backend, 0, 2)
            )
        torch.testing.assert_close(outputs[0].tokens, outputs[1].tokens)
        torch.testing.assert_close(outputs[0].raw_rewards, outputs[1].raw_rewards)

    def test_policy_diagnostic_isolates_rng_and_cursor(self):
        policy, reference, reward, _ = self.models()
        with tempfile.TemporaryDirectory() as directory:
            cfg = smoke_config(directory)
            cfg.rlhf = PPOConfig(max_prompt_tokens=4, max_new_tokens=3)
            backend = RewardBackend(cfg, logging.getLogger("ppo-test"), smoke=True)
            cursor = {"shard": 3, "batch": 2, "raw_start": 4}
            original = copy.deepcopy(cursor)
            rng, python_rng = torch.get_rng_state(), random.getstate()
            reports = [
                evaluate_policy(
                    policy,
                    reference,
                    reward,
                    [[1, 4]],
                    cfg,
                    backend,
                    SmokeTokenizer(),
                    0,
                )
                for _ in range(2)
            ]
            self.assertEqual(reports[0], reports[1])
            self.assertEqual(cursor, original)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(python_rng, random.getstate())

    def test_loss_is_invariant_to_microbatch_partition(self):
        config = PPOConfig()
        mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
        old = torch.zeros(2, 3)
        advantages = torch.tensor([[1.0, 2.0, -1.0], [3.0, 999.0, 999.0]])
        weight = torch.tensor(0.1, requires_grad=True)
        loss, _ = ppo_objective(
            weight.expand(2, 3), old, old, old, advantages, old, old, mask, config
        )
        expected = torch.autograd.grad(loss, weight)[0]
        partitioned = torch.tensor(0.1, requires_grad=True)
        for row in range(2):
            ix = slice(row, row + 1)
            loss, _ = ppo_objective(
                partitioned.expand(1, 3),
                old[ix],
                old[ix],
                old[ix],
                advantages[ix],
                old[ix],
                old[ix],
                mask[ix],
                config,
            )
            (loss * mask[ix].sum() / mask.sum()).backward()
        torch.testing.assert_close(partitioned.grad, expected)

    def test_replay_across_memory_chunks_and_temperature(self):
        policy, reference, reward, backend = self.models()
        cfg = PPOConfig(max_prompt_tokens=5, max_new_tokens=5, temperature=0.7)
        with patch("RLHF.ppo.sample_tokens", return_value=torch.tensor([4, 5])):
            rollout = collect_rollout(
                policy, reference, reward, [[1, 3], [1, 3, 4]], cfg, backend, 0, 2
            )
        new, _, _ = policy(
            rollout.input_ids,
            rollout.attention_mask,
            rollout.positions,
            tokens=rollout.tokens,
            temperature=cfg.temperature,
        )
        torch.testing.assert_close(new, rollout.old_logprobs, atol=1e-6, rtol=1e-5)

    def test_precision_and_production_guard(self):
        self.assertEqual(resolve_precision("auto", bf16_supported=False), "fp32")
        self.assertEqual(resolve_precision("auto", bf16_supported=True), "bf16")
        with self.assertRaises(ValueError):
            resolve_precision("fp16", bf16_supported=True)
        cfg = RewardConfig()
        cfg.system.precision = "fp16"
        with self.assertRaises(ValueError):
            cfg.validate()
        block = MambaBlock(16, state_size=4)
        require_mamba_kernels(block, False, torch.device("cpu"))
        with self.assertRaises(RuntimeError):
            require_mamba_kernels(block, True, torch.device("cpu"))
        block.require_fused_scan = True
        with self.assertRaisesRegex(RuntimeError, "Production RLHF"):
            block(torch.randn(1, 3, 16))

    def test_legacy_diagnostic_interval_and_approximate_stats(self):
        cfg = RewardConfig.from_dict({"system": {"eval_interval": 17}})
        self.assertEqual(cfg.system.diagnostic_eval_interval, 17)
        cfg = RewardConfig.from_dict(
            {"system": {"eval_interval": 17, "diagnostic_eval_interval": 23}}
        )
        self.assertEqual(cfg.system.diagnostic_eval_interval, 23)
        a, b = RewardStatistics("approximate", 8), RewardStatistics("exact")
        try:
            for start in range(0, 100, 10):
                values = torch.arange(start, start + 10)
                a.update(values + 1, values)
                b.update(values + 1, values)
            self.assertLessEqual(len(a.samples["reward"][1]), 8)
            for key in (
                "pairwise_accuracy",
                "chosen_reward_mean",
                "chosen_reward_std",
                "margin_mean",
            ):
                self.assertEqual(a.result()[key], b.result()[key])
            self.assertEqual(a.result()["percentile_strategy"], "approximate")
        finally:
            a.close()
            b.close()

    def test_prompt_pipeline_and_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = smoke_config(directory)
            cfg.rlhf = PPOConfig(max_prompt_tokens=32)
            backend = RewardBackend(cfg, logging.getLogger("ppo-test"), smoke=True)
            feed = prompt_feed(cfg, SmokeTokenizer(), backend, smoke=True)
            try:
                path = feed.wait(0)
                dataset = PromptShardDataset(path)
                first = dataset[0].copy()
                cursor = {
                    "shard": 0,
                    "raw_start": dataset.metadata["raw_start"],
                    "native_start": dataset.metadata["native_start"],
                }
                dataset.close()
                self.assertLessEqual(
                    len(list(Path(feed.args.cache_dir).glob("*.bin"))),
                    cfg.data.buffer_size,
                )
            finally:
                feed.close()
            second = prompt_feed(cfg, SmokeTokenizer(), backend, cursor, smoke=True)
            try:
                dataset = PromptShardDataset(second.wait(0))
                self.assertEqual(first.tolist(), dataset[0].tolist())
                dataset.close()
            finally:
                second.close()
            cfg.rlhf.prompt_dataset = "nvidia/HelpSteer2"
            with self.assertRaises(ValueError):
                cfg.validate()

    def test_checkpoint_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = create_smoke_inputs(str(Path(directory) / "full"))
            prepare_sources(cfg, smoke=True)
            complete = run_training(cfg, smoke=True)
            partial_cfg = copy.deepcopy(cfg)
            partial_cfg.system.output_dir = str(Path(directory) / "resumed")
            first = run_training(partial_cfg, smoke=True, stop_after=1)
            partial_cfg.system.resume_from_checkpoint = first
            resumed = run_training(partial_cfg, smoke=True)
            # Compare DCP model/critic and optimizer state after exact fresh-rollout resume.
            loaded = []
            for path, settings in ((complete, cfg), (resumed, partial_cfg)):
                source = torch.load(settings.rlhf.policy_checkpoint, weights_only=True)
                model = PolicyValueModel(
                    HybridMambaMoEConfig.from_dict(source["config"])
                )
                optimizer = torch.optim.AdamW(model.parameters())
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
                backend = RewardBackend(
                    settings, logging.getLogger("ppo-test"), smoke=True
                )
                # Match production optimizer groups before loading DCP.
                args = SimpleNamespace(
                    no_muon=True,
                    lr=cfg.training.learning_rate,
                    adam_lr=None,
                    muon_lr=None,
                    weight_decay=cfg.training.weight_decay,
                    adam_beta1=0.9,
                    adam_beta2=0.95,
                    adam_eps=1e-8,
                )
                optimizer = backend.base.build_fsdp2_optimizers(
                    model, args=args, logger=backend.logger
                )[0][0]
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
                metadata = load_checkpoint(
                    path, model, optimizer, scheduler, settings, backend, family=FAMILY
                )
                self.assertEqual(metadata["step"], 2)
                self.assertEqual(metadata["cursor"]["optimizer_steps"], 4)
                loaded.append((model.state_dict(), optimizer.state_dict()))
            for key in loaded[0][0]:
                torch.testing.assert_close(
                    loaded[0][0][key], loaded[1][0][key], atol=0, rtol=0
                )
            for index, state in loaded[0][1]["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(
                        value, loaded[1][1]["state"][index][key], atol=0, rtol=0
                    )
            records = [
                json.loads(line)
                for line in Path(cfg.system.output_dir, "ppo_metrics.jsonl")
                .read_text()
                .splitlines()
            ]
            diagnostic = [r for r in records if "evaluation" in r]
            self.assertEqual(len({r["subset_sha256"] for r in diagnostic}), 1)


if __name__ == "__main__":
    unittest.main()
