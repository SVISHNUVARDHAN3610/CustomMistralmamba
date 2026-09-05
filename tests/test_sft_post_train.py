"""CPU SFT integration and FSDP2 adapter contracts (no Hub downloads)."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from model.core.config import HybridMambaMoEConfig
from model.hybrid.model import HybridForCausalLM


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "post-training" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sft = load_script("sft_post_train")
fsdp = load_script("sft_fsdp2_post_train")


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 3
    name_or_path = "tiny"

    def __len__(self):
        return 64

    def get_vocab(self):
        return {str(i): i for i in range(64)}

    def encode(self, text, add_special_tokens=False):
        return [4 + ord(c) % 60 for c in text]


def make_shard(directory, index=0, count=6, window=17):
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / f"shard_{index:06d}"
    tokens = (np.arange(count * window).reshape(count, window) % 50 + 4).astype("<u4")
    tokens[:, 0] = 1
    tokens[:, -1] = 2
    mask = np.zeros((count, window), dtype=np.uint8)
    for i in range(count):
        mask[i, -(i % 5 + 2) :] = 1
    tokens.tofile(base.with_suffix(".bin"))
    mask.tofile(base.with_suffix(".mask"))
    base.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": "sft-role-text-v1",
                "dtype": "<u4",
                "num_tokens": tokens.size,
                "seq_len": window,
            }
        )
    )
    return base.with_suffix(".bin")


class TestSFTTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.logger = logging.getLogger("sft-test")
        cls.logger.addHandler(logging.NullHandler())

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def checkpoint(self, directory):
        torch.manual_seed(5)
        cfg = HybridMambaMoEConfig(
            vocab_size=64,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            intermediate_size=32,
            window_size=16,
            num_experts=2,
            top_k=1,
            dropout=0.1,
            max_position_embeddings=16,
            mamba_state_size=4,
            mamba_conv_kernel=2,
            mamba_expand=1,
            use_dual_memory=True,
            memory_size=4,
            memory_num_heads=4,
            memory_chunk_size=8,
            use_fused_mamba_scan=False,
            use_auxiliary_losses=True,
        )
        model = HybridForCausalLM(cfg)
        path = directory / "pretrained.pth"
        torch.save(
            {
                "config": cfg.to_dict(),
                "model_state_dict": model.state_dict(),
                "global_step": 9000,
            },
            path,
        )
        return path

    def args(self, root, pretrained, run="run", extra=()):
        run_dir = root / run
        run_dir.mkdir(exist_ok=True)
        return sft.parse_args(
            [
                "--pretrained-checkpoint",
                str(pretrained),
                "--run-dir",
                str(run_dir),
                "--cache-dir",
                str(root / "cache"),
                "--offline-shards",
                "--device",
                "cpu",
                "--no-amp",
                "--no-muon",
                "--max-steps",
                "2",
                "--warmup-steps",
                "1",
                "--grad-accum-steps",
                "2",
                "--save-interval",
                "1",
                "--seq-len",
                "16",
                "--tokens-per-shard",
                str(17 * 6),
                *extra,
            ]
        )

    def run_cpu(self, args):
        with patch.object(
            sft.AutoTokenizer, "from_pretrained", return_value=Tokenizer()
        ):
            sft.run_training(args, sft.SingleGPUBackend(args, self.logger), self.logger)

    def test_real_cpu_warm_start_changes_weights_and_uses_sft_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretrained = self.checkpoint(root)
            make_shard(root / "cache")
            args = self.args(root, pretrained)
            self.run_cpu(args)
            saved = sft.read_checkpoint(root / "run")
            initial = sft.read_checkpoint(pretrained)
            self.assertEqual(saved["global_step"], 2)
            self.assertEqual(saved["current_batch_idx"], 4)
            self.assertEqual(saved["sft_runtime"]["family"], "sft_single_gpu_v1")
            self.assertTrue(
                any(
                    not torch.equal(value, initial["model_state_dict"][key])
                    for key, value in saved["model_state_dict"].items()
                )
            )
            self.assertTrue((root / "cache" / "shard_000000.bin").exists())

    def test_interrupted_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretrained = self.checkpoint(root)
            make_shard(root / "cache")
            self.run_cpu(self.args(root, pretrained, run="reference"))
            args = self.args(root, pretrained, run="resumed")
            original = sft.train_window

            def interrupt(model, batches, optimizers, args, backend, step):
                if step == 1:
                    raise RuntimeError("simulated interruption")
                return original(model, batches, optimizers, args, backend, step)

            with (
                patch.object(sft, "train_window", side_effect=interrupt),
                self.assertRaisesRegex(RuntimeError, "simulated"),
            ):
                self.run_cpu(args)
            partial = sft.read_checkpoint(root / "resumed")
            self.assertEqual(partial["global_step"], 1)
            args.resume = str(root / "resumed")
            args.pretrained_checkpoint = None
            self.run_cpu(args)
            final = sft.read_checkpoint(root / "resumed")
            reference = sft.read_checkpoint(root / "reference")
            for key, value in reference["model_state_dict"].items():
                torch.testing.assert_close(
                    final["model_state_dict"][key], value, rtol=0, atol=0
                )
            self.assertEqual(final["current_batch_idx"], reference["current_batch_idx"])
            self.assertEqual(final["schedulers"], reference["schedulers"])

    def test_loss_weighting_matches_global_token_mean_gradient(self):
        parameter = torch.tensor(0.7, requires_grad=True)
        counts = [2, 7, 3, 4]
        means = [parameter.square(), parameter * 2, parameter**3, parameter * 5]
        losses = [
            sft.accumulated_loss(
                SimpleNamespace(ce_loss=ce, loss=ce), n, sum(counts), 2, 2
            )
            for ce, n in zip(means, counts)
        ]
        distributed = sum(losses) / 2  # FSDP averages the two ranks' gradients.
        expected = sum(ce * n for ce, n in zip(means, counts)) / sum(counts)
        actual_grad = torch.autograd.grad(distributed, parameter, retain_graph=True)[0]
        torch.testing.assert_close(
            actual_grad, torch.autograd.grad(expected, parameter)[0]
        )

    def test_sampler_ranks_do_not_overlap_and_resume_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = make_shard(Path(directory), count=9)
            args = SimpleNamespace(seq_len=16, batch_size=2, seed=99)
            rank_inputs = []
            for rank in range(2):
                backend = SimpleNamespace(
                    rank=rank, world=2, device=torch.device("cpu")
                )
                ds, loader = sft.shard_loader(path, args, backend, 0, 0)
                batches = list(loader)
                ds.close()
                rank_inputs.append(torch.cat([x[:, 1] for x, _ in batches]).tolist())
                ds, resumed = sft.shard_loader(path, args, backend, 0, 1)
                rest = list(resumed)
                ds.close()
                torch.testing.assert_close(rest[0][0], batches[1][0])
            self.assertFalse(set(rank_inputs[0]) & set(rank_inputs[1]))

    def test_heldout_validation_preserves_training_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = sft.read_checkpoint(self.checkpoint(root))
            model = HybridForCausalLM(
                HybridMambaMoEConfig.from_dict(checkpoint["config"])
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.config.return_logits = False
            model.train()
            make_shard(root / "validation", count=3)
            args = SimpleNamespace(
                validation_dir=str(root / "validation"),
                seq_len=16,
                batch_size=2,
                seed=3,
                val_batches=2,
                no_amp=True,
            )
            backend = SimpleNamespace(device=torch.device("cpu"))
            result = sft.evaluate(model, args, backend)
            self.assertTrue(np.isfinite(result))
            self.assertTrue(model.training)
            self.assertFalse(model.config.use_cache)

    def test_resume_contract_rejects_pretraining_and_topology_drift(self):
        backend = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            sft.restore_training_state({}, {"world_size": 2}, [], [], backend)
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            sft.restore_training_state(
                {"sft_runtime": {"world_size": 1}}, {"world_size": 2}, [], [], backend
            )

    def test_retained_producer_recovers_published_shard_before_state_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            producer = sft.RetainedShardProducer(
                directory,
                tokenizer=Tokenizer(),
                seq_len=32,
                tokens_per_shard=32,
                log_fn=lambda _: None,
            )
            messages = [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ]
            producer._append_conversation(messages)
            producer._pad_window()
            tokens, mask = producer.token_buffer[:], producer.loss_mask_buffer[:]
            producer._write_shard(32)
            base = Path(directory) / "shard_000000.bin"
            before = base.read_bytes()
            replay = sft.RetainedShardProducer(
                directory,
                tokenizer=Tokenizer(),
                seq_len=32,
                tokens_per_shard=32,
                log_fn=lambda _: None,
            )
            replay.token_buffer, replay.loss_mask_buffer = tokens, mask
            replay._write_shard(32)
            self.assertEqual(replay.current_shard_idx, 1)
            self.assertEqual(base.read_bytes(), before)
            replay.current_shard_idx = 0
            replay.token_buffer, replay.loss_mask_buffer = [10] * 32, mask
            with self.assertRaisesRegex(ValueError, "differs"):
                replay._write_shard(32)

    def test_fsdp_adapter_wraps_layers_before_root_and_uses_replicated_helpers(self):
        backend = fsdp.FSDP2Backend.__new__(fsdp.FSDP2Backend)
        backend.device = torch.device("cpu")
        backend.world = 2
        calls = []
        layers = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList(layers)
        model.model.calibrate_ssm_norm_thresholds = lambda: None
        backend.base = SimpleNamespace(
            _prepare_fsdp2_custom_math_params=lambda model: set(),
            _ordered_replicated_params=lambda model, ignored: (),
        )
        backend.api = {
            "MixedPrecisionPolicy": lambda **kw: kw,
            "fully_shard": lambda module, **kw: calls.append((module, kw)),
        }
        backend.wrap(model, SimpleNamespace(gradient_checkpointing=True))
        self.assertEqual([m for m, _ in calls], layers + [model])
        self.assertFalse(calls[0][1]["reshard_after_forward"])
        self.assertEqual(calls[-1][1]["mp_policy"]["reduce_dtype"], torch.float32)

    def test_fsdp_builds_existing_dtensor_muon_optimizer(self):
        backend = fsdp.FSDP2Backend.__new__(fsdp.FSDP2Backend)
        backend.base = fsdp.load_pretraining_fsdp2()
        backend.logger = self.logger
        with patch.object(
            backend.base, "build_fsdp2_optimizers", return_value="result"
        ) as build:
            self.assertEqual(backend.build_optimizers("model", "args"), "result")
            build.assert_called_once_with("model", args="args", logger=self.logger)


if __name__ == "__main__":
    unittest.main()
