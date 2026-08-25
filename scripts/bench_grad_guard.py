"""A/B benchmark for the ``--grad-nan-guard sanitize`` pass cost.

Measures, on the SAME model and batch:

  A. forward + backward + clip_grad_norm_          (guard "monitor" cost)
  B. forward + backward + clip_grad_norm_ +        (guard "sanitize" cost)
     nan_to_num_ over every gradient
  C. the sanitize pass alone (clip re-applied each rep)

so the sanitize overhead can be read off directly in ms/step and % of
step time, and extrapolated: its cost scales with total gradient BYTES,
which are printed alongside.

Defaults are laptop-safe (~5M-param toy config). CPU timings are NOT
representative of production sizing — run this on the GPU host with e.g.::

    python scripts/bench_grad_guard.py --layers 24 --seq-len 2048 \
        --batch-size 8 --iters 100

The script refuses large parameter counts on CPU so the dev-laptop rule
(never run production-scale configs locally) stays enforced.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from toy_train import build_toy_config

from model.hybrid.model import HybridForCausalLM

# Laptop-safe ceiling: anything above this on CPU is refused.
CPU_PARAM_LIMIT = 20_000_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument(
        "--layers",
        type=int,
        default=None,
        help="Override num_layers (default: keep the toy config's value).",
    )
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Bypass the CPU parameter-count guard (GPU hosts don't need it).",
    )
    return p.parse_args()


def sanitize_pass(model: HybridForCausalLM) -> None:
    """Mirrors train.py's --grad-nan-guard=sanitize block exactly."""
    for prm in model.parameters():
        if prm.grad is not None:
            torch.nan_to_num_(prm.grad, nan=0.0, posinf=0.0, neginf=0.0)


def time_ms(fn, warmup: int, iters: int, sync: bool) -> float:
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    sync = device.type == "cuda"

    cfg = build_toy_config()
    if args.layers is not None:
        cfg = dataclasses.replace(cfg, num_layers=args.layers)
    if args.seq_len > cfg.max_position_embeddings:
        cfg = dataclasses.replace(cfg, max_position_embeddings=args.seq_len)

    torch.manual_seed(0)
    model = HybridForCausalLM(cfg).to(device)
    model.train()
    n_params = sum(prm.numel() for prm in model.parameters())
    grad_bytes = n_params * 4  # fp32 grads

    if device.type == "cpu" and n_params > CPU_PARAM_LIMIT and not args.force:
        print(
            f"REFUSED: {n_params / 1e6:.1f}M params on CPU exceeds the "
            f"{CPU_PARAM_LIMIT // 1_000_000}M laptop-safe limit. Run this on a "
            "GPU host, or shrink --layers/--seq-len, or pass --force."
        )
        return 1

    ids = torch.randint(
        0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device
    )
    labels = torch.randint(
        0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device
    )

    def full_step(with_sanitize: bool):
        model.zero_grad(set_to_none=True)
        out = model(
            input_ids=ids, labels=labels, training_step=0, max_training_steps=args.iters
        )
        assert out.loss is not None
        out.loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        if with_sanitize:
            sanitize_pass(model)

    base_ms = time_ms(lambda: full_step(False), args.warmup, args.iters, sync)
    guard_ms = time_ms(lambda: full_step(True), args.warmup, args.iters, sync)

    # Segment C: with grads already populated, re-time clip alone vs
    # clip+sanitize; the difference isolates the sanitize pass itself.
    out = model(
        input_ids=ids, labels=labels, training_step=0, max_training_steps=args.iters
    )
    assert out.loss is not None
    out.loss.backward()
    clip_only_ms = time_ms(
        lambda: clip_grad_norm_(model.parameters(), 1.0), args.warmup, args.iters, sync
    )
    clip_plus_ms = time_ms(
        lambda: (clip_grad_norm_(model.parameters(), 1.0), sanitize_pass(model)),
        args.warmup,
        args.iters,
        sync,
    )

    print("=" * 64)
    print(
        f"device={device.type}  params={n_params / 1e6:.2f}M  "
        f"grad_bytes={grad_bytes / 1e6:.0f}MB  "
        f"bs={args.batch_size}  seq={args.seq_len}  layers={cfg.num_layers}"
    )
    print(f"A  fwd+bwd+clip              : {base_ms:8.3f} ms/step")
    print(f"B  fwd+bwd+clip+sanitize     : {guard_ms:8.3f} ms/step")
    print(
        f"B-A sanitize overhead        : {guard_ms - base_ms:8.3f} ms/step "
        f"({(guard_ms - base_ms) / max(base_ms, 1e-9) * 100:.2f}% of step)"
    )
    print(f"C  sanitize pass alone       : {clip_plus_ms - clip_only_ms:8.3f} ms/pass")
    print("=" * 64)
    print("Extrapolate linearly in grad_bytes to your production param count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
