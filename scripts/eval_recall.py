"""Synthetic associative retrieval & needle-in-a-haystack recall harness.

Evaluates long-range factual recall and runs scientific falsification tests:
  - Condition 1 (Memory-On): Full Hybrid model with active dual memory.
  - Condition 2 (Memory-Zeroed / Test 1): Zeroed memory states at test time.
  - Condition 3 (Null Baseline / Test 3): Parameter-matched SSM-only model (no memory).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F

from model.core.builders import build_test3_null_baseline_config, count_trainable_params
from model.core.config import HybridMambaMoEConfig
from model.hybrid.model import HybridForCausalLM


def build_eval_config(
    vocab_size: int = 512,
    hidden_size: int = 128,
    num_layers: int = 4,
    use_qk_norm: bool = True,
    attention_sink_size: int = 4,
) -> HybridMambaMoEConfig:
    return HybridMambaMoEConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        intermediate_size=192,
        window_size=64,
        num_experts=4,
        top_k=2,
        dropout=0.0,
        capacity_factor=None,
        max_position_embeddings=4096,
        mamba_state_size=8,
        mamba_conv_kernel=4,
        mamba_expand=2,
        use_dual_memory=True,
        memory_size=16,
        memory_num_heads=4,
        memory_chunk_size=128,
        stream_chunked_ce_loss=False,
        return_logits=True,
        use_auxiliary_losses=True,
        use_qk_norm=use_qk_norm,
        attention_sink_size=attention_sink_size,
    )


def generate_synthetic_needle_sample(
    vocab_size: int,
    seq_len: int,
    needle_depth: float,
    device: torch.device,
    key_len: int = 3,
    val_len: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Generates a synthetic haystack sequence with a key-value needle inserted."""
    haystack = torch.randint(20, vocab_size, (1, seq_len), device=device)

    # Distinct tokens for key and value
    key_tokens = torch.randint(5, 15, (1, key_len), device=device)
    val_tokens = torch.randint(15, 20, (1, val_len), device=device)

    needle_pos = max(
        0, min(seq_len - key_len - val_len - key_len - 5, int(seq_len * needle_depth))
    )
    haystack[:, needle_pos : needle_pos + key_len] = key_tokens
    haystack[:, needle_pos + key_len : needle_pos + key_len + val_len] = val_tokens

    # Append retrieval query at the end
    query_pos = seq_len - key_len
    haystack[:, query_pos:] = key_tokens

    target_val = val_tokens
    return haystack, target_val, needle_pos, query_pos


@torch.no_grad()
def evaluate_retrieval_loss(
    model: HybridForCausalLM,
    input_ids: torch.Tensor,
    target_val: torch.Tensor,
    zero_memory: bool = False,
) -> tuple[float, bool]:
    """Evaluates cross-entropy loss and exact-match on target value tokens."""
    model.eval()
    memory_states = None
    if zero_memory:
        memory_states = model.model.zero_memory_states(
            batch_size=input_ids.size(0),
            device=input_ids.device,
            dtype=model.model.embed_tokens.weight.dtype,
        )

    out = model(input_ids=input_ids, memory_states=memory_states, use_cache=False)
    assert out.logits is not None

    # Predict target tokens immediately following the query at the end
    pred_logits = out.logits[:, -1, :]  # shape [1, vocab_size]
    first_target = target_val[:, 0]

    loss = F.cross_entropy(pred_logits, first_target).item()
    pred_tok = pred_logits.argmax(dim=-1)
    exact_match = bool((pred_tok == first_target).item())

    return loss, exact_match


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic Recall & Falsification Harness"
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument(
        "--needle-depth", type=float, default=0.1, help="Depth in sequence (0.0 to 0.9)"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = build_eval_config()
    model_hybrid = HybridForCausalLM(cfg).to(device)
    n_hybrid = count_trainable_params(model_hybrid)

    null_cfg = build_test3_null_baseline_config(cfg)
    model_null = HybridForCausalLM(null_cfg).to(device)
    n_null = count_trainable_params(model_null)

    print("=" * 72)
    print("SCIENTIFIC RECALL & FALSIFICATION EVALUATION")
    print("=" * 72)
    print(f"Sequence length:  {args.seq_len}")
    print(
        f"Needle depth:     {args.needle_depth:.2f} (distance ~{int(args.seq_len * (1 - args.needle_depth))} tokens)"
    )
    print(f"Number of trials: {args.num_samples}")
    print(f"Hybrid params:    {n_hybrid:,} (use_dual_memory=True)")
    print(
        f"Null params:      {n_null:,} (use_dual_memory=False, state={null_cfg.mamba_state_size})"
    )
    print("-" * 72)

    losses_on, losses_zeroed, losses_null = [], [], []
    matches_on, matches_zeroed, matches_null = 0, 0, 0

    for _ in range(args.num_samples):
        input_ids, target_val, _needle_pos, _query_pos = (
            generate_synthetic_needle_sample(
                vocab_size=cfg.vocab_size,
                seq_len=args.seq_len,
                needle_depth=args.needle_depth,
                device=device,
            )
        )

        l_on, m_on = evaluate_retrieval_loss(
            model_hybrid, input_ids, target_val, zero_memory=False
        )
        l_zero, m_zero = evaluate_retrieval_loss(
            model_hybrid, input_ids, target_val, zero_memory=True
        )
        l_null, m_null = evaluate_retrieval_loss(
            model_null, input_ids, target_val, zero_memory=False
        )

        losses_on.append(l_on)
        losses_zeroed.append(l_zero)
        losses_null.append(l_null)
        if m_on:
            matches_on += 1
        if m_zero:
            matches_zeroed += 1
        if m_null:
            matches_null += 1

    mean_on = sum(losses_on) / len(losses_on)
    mean_zero = sum(losses_zeroed) / len(losses_zeroed)
    mean_null = sum(losses_null) / len(losses_null)

    print(
        f"Condition 1 (Memory-On):     Mean CE = {mean_on:.4f} | Exact Match = {matches_on}/{args.num_samples}"
    )
    print(
        f"Condition 2 (Test 1 Zeroed):  Mean CE = {mean_zero:.4f} | Exact Match = {matches_zeroed}/{args.num_samples}"
    )
    print(
        f"Condition 3 (Test 3 Null):    Mean CE = {mean_null:.4f} | Exact Match = {matches_null}/{args.num_samples}"
    )
    print("-" * 72)
    delta_test1 = mean_zero - mean_on
    delta_test3 = mean_null - mean_on
    print(f"Test 1 Memory Delta (Zeroed - On): {delta_test1:+.4f}")
    print(f"Test 3 Baseline Delta (Null - On): {delta_test3:+.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
