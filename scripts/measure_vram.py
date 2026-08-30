"""GPU VRAM measurement harness for the Hybrid model (cloud hosts only).

Measures torch.cuda.max_memory_allocated / max_memory_reserved for the
production-equivalent setup (batch 8, seq 986, hidden 512, 8 layers, 8
experts, dual memory, chunked CE, bf16 autocast) with gradient
checkpointing off vs on, from a clean process each time.

Usage (on a CUDA machine):
    python scripts/measure_vram.py off     # baseline only
    python scripts/measure_vram.py on      # checkpointing only
    python scripts/measure_vram.py both    # default

CPU-only hosts refuse to run (assert), because CPU allocator numbers are
not comparable to VRAM.
"""

from __future__ import annotations

import json
import sys

import torch

from model import HybridForCausalLM, HybridMambaMoEConfig


def build_config(checkpointing: bool) -> HybridMambaMoEConfig:
    return HybridMambaMoEConfig(
        vocab_size=32000,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        num_kv_heads=8,
        head_dim=64,
        intermediate_size=512,
        window_size=512,
        num_experts=8,
        top_k=2,
        dropout=0.0,
        capacity_factor=None,
        max_position_embeddings=4096,
        mamba_state_size=16,
        mamba_conv_kernel=4,
        mamba_expand=2,
        use_dual_memory=True,
        memory_size=48,
        memory_num_heads=8,
        memory_chunk_size=512,
        stream_chunked_ce_loss=True,
        return_logits=False,
        use_auxiliary_losses=True,
        use_fused_mamba_scan=True,
        gradient_checkpointing=checkpointing,
    )


def measure(cfg: HybridMambaMoEConfig, batch: int = 8, seq: int = 986, steps: int = 3):
    assert torch.cuda.is_available(), "VRAM measurement requires a CUDA host"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    model = HybridForCausalLM(cfg).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        ids = torch.randint(0, cfg.vocab_size, (batch, seq), device="cuda")
        labels = torch.randint(0, cfg.vocab_size, (batch, seq), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=ids,
                labels=labels,
                training_step=50,
                max_training_steps=100,
            )
        out.loss.backward()
        opt.step()
        del out, ids, labels
    torch.cuda.synchronize()
    return {
        "peak_allocated_GB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_GB": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }


def main() -> None:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device — run this on the GPU host (Kaggle/cloud).")
        raise SystemExit(1)
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    results = {}
    if which in ("off", "both"):
        results["ckpt_off"] = measure(build_config(False))
    if which in ("on", "both"):
        results["ckpt_on"] = measure(build_config(True))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
