"""Mixed CPU training test for Hybrid Mamba-MoE (1M to 5M parameters).

Runs 50 steps with 1 epoch on CPU with seq_len=128, batch_size=2, using a
mixture of padded, unpadded, associative, and structured token sequences.
Validates all loss terms, auxiliary losses, activations, and parameter
gradients for NaN/Inf values at every step.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.nn.utils import clip_grad_norm_

from model.core.builders import count_trainable_params
from model.core.config import HybridMambaMoEConfig
from model.hybrid.model import HybridForCausalLM


def build_small_cpu_test_config(
    vocab_size: int = 512,
    hidden_size: int = 128,
    num_layers: int = 4,
    intermediate_size: int = 256,
    num_experts: int = 4,
    top_k: int = 2,
    use_qk_norm: bool = True,
    attention_sink_size: int = 4,
) -> HybridMambaMoEConfig:
    """Target 1M - 5M parameter model configuration."""
    return HybridMambaMoEConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        intermediate_size=intermediate_size,
        window_size=32,
        num_experts=num_experts,
        top_k=top_k,
        dropout=0.0,
        capacity_factor=None,
        max_position_embeddings=512,
        mamba_state_size=8,
        mamba_conv_kernel=4,
        mamba_expand=2,
        use_dual_memory=True,
        memory_size=16,
        memory_num_heads=4,
        memory_chunk_size=64,
        stream_chunked_ce_loss=True,
        return_logits=False,
        use_auxiliary_losses=True,
        use_qk_norm=use_qk_norm,
        attention_sink_size=attention_sink_size,
        final_logit_z_loss_coef=1e-4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def generate_mixed_training_batch(
    step: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    pad_token_id: int,
    ignore_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates a mixture of data:

    - Case A: Fully dense uniform token sequence.
    - Case B: Padded sequence with trailing pad tokens.
    - Case C: Key-value associative repetition sequence.
    - Case D: Highly repeated / low-entropy token sequences.
    """
    mode = step % 4
    input_ids = torch.randint(4, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    if mode == 1:
        # Padded sequence: second sequence is shorter (e.g. length 80 of 128)
        valid_len = max(32, seq_len - 32 - (step % 32))
        input_ids[1, valid_len:] = pad_token_id
        attention_mask[1, valid_len:] = 0

    elif mode == 2:
        # Key-Value associative pattern: repeated key-value pairs
        key = torch.randint(10, 30, (batch_size, 3), device=device)
        val = torch.randint(30, 50, (batch_size, 2), device=device)
        # Place key-val at pos 10 and query key at pos 100
        input_ids[:, 10:13] = key
        input_ids[:, 13:15] = val
        input_ids[:, 100:103] = key
        input_ids[:, 103:105] = val

    elif mode == 3:
        # Low entropy repetition sequence
        repeat_tok = (step % (vocab_size - 10)) + 5
        input_ids[:, ::2] = repeat_tok

    # Build causal next-token labels
    labels = input_ids.roll(shifts=-1, dims=1)
    labels[:, -1] = ignore_index
    labels = labels.masked_fill(attention_mask == 0, ignore_index)
    next_valid = attention_mask.roll(shifts=-1, dims=1)
    next_valid[:, -1] = 0
    labels = labels.masked_fill(next_valid == 0, ignore_index)

    return input_ids, attention_mask, labels


class TestMixedCPUTraining(unittest.TestCase):
    def test_mixed_cpu_training_50_steps(self) -> None:
        torch.manual_seed(42)
        device = torch.device("cpu")
        cfg = build_small_cpu_test_config()
        model = HybridForCausalLM(cfg).to(device)

        param_count = count_trainable_params(model)
        print(
            f"\n[TestMixedCPUTraining] Trainable params: {param_count:,} (Budget: 1M - 5M)"
        )
        self.assertGreaterEqual(
            param_count, 1_000_000, "Model must have at least 1M parameters"
        )
        self.assertLessEqual(
            param_count, 5_000_000, "Model must have at most 5M parameters"
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        num_steps = 50
        batch_size = 2
        seq_len = 128

        model.train()
        for step in range(num_steps):
            input_ids, attention_mask, labels = generate_mixed_training_batch(
                step=step,
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=cfg.vocab_size,
                pad_token_id=cfg.pad_token_id,
                ignore_index=cfg.label_ignore_index,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                training_step=step,
                max_training_steps=num_steps,
            )

            # 1. Validate total loss
            self.assertIsNotNone(out.loss, f"Loss is None at step {step}")
            self.assertTrue(
                torch.isfinite(out.loss).item(),
                f"Non-finite total loss {out.loss.item()} at step {step}",
            )

            # 2. Validate CE loss
            if out.ce_loss is not None:
                self.assertTrue(
                    torch.isfinite(out.ce_loss).item(),
                    f"Non-finite ce_loss {out.ce_loss.item()} at step {step}",
                )

            # 3. Validate Router Aux and Z loss
            if out.router_aux_loss is not None:
                self.assertTrue(
                    torch.isfinite(out.router_aux_loss).item(),
                    f"Non-finite router_aux_loss {out.router_aux_loss.item()} at step {step}",
                )
            if out.router_z_loss is not None:
                self.assertTrue(
                    torch.isfinite(out.router_z_loss).item(),
                    f"Non-finite router_z_loss {out.router_z_loss.item()} at step {step}",
                )

            # 4. Validate all 8 Auxiliary Loss Breakdown terms
            aux = out.auxiliary_losses
            self.assertIsNotNone(aux, f"Auxiliary losses is None at step {step}")
            for term_name in (
                "recon",
                "assoc",
                "gate",
                "read",
                "fusion",
                "expert",
                "ssm",
                "slot",
            ):
                val = getattr(aux, term_name)
                self.assertIsNotNone(
                    val, f"Aux loss {term_name} is None at step {step}"
                )
                self.assertTrue(
                    torch.isfinite(val).item(),
                    f"Non-finite auxiliary loss {term_name}={val.item()} at step {step}",
                )

            # 5. Backward pass
            out.loss.backward()

            # 6. Validate gradients across all trainable parameters
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.assertTrue(
                        torch.isfinite(param.grad).all().item(),
                        f"Non-finite gradient in parameter '{name}' at step {step}",
                    )

            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 7. Validate updated weights
            for name, param in model.named_parameters():
                self.assertTrue(
                    torch.isfinite(param).all().item(),
                    f"Non-finite weight values in parameter '{name}' at step {step}",
                )

            if (step + 1) % 10 == 0 or step == 0:
                print(
                    f"  Step {step + 1:2d}/{num_steps} | "
                    f"Loss: {out.loss.item():.4f} | "
                    f"CE: {out.ce_loss.item() if out.ce_loss is not None else 0.0:.4f} | "
                    f"Recon: {aux.recon.item():.4f} | "
                    f"Assoc: {aux.assoc.item():.4f} | "
                    f"Gate: {aux.gate.item():.4f} | "
                    f"Fusion: {aux.fusion.item():.4f} | "
                    f"SSM: {aux.ssm.item():.4f} | "
                    f"Slot: {aux.slot.item():.4f}"
                )


def run_standalone_training() -> None:
    test = TestMixedCPUTraining()
    test.test_mixed_cpu_training_50_steps()
    print(
        "\n[SUCCESS] All 50 steps of mixed CPU training completed with zero NaNs / Infs!\n"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--unittest":
        unittest.main(argv=[sys.argv[0]])
    else:
        run_standalone_training()
