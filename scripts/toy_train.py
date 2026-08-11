"""Minimal toy training smoke test for Hybrid Mamba-MoE (~5M params)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.nn.utils import clip_grad_norm_

from model import (
    HybridForCausalLM,
    HybridMambaMoEConfig,
    count_trainable_params,
    log_mamba_backend,
)


def build_toy_config() -> HybridMambaMoEConfig:
    """~5M trainable params (excluding training-only aux modules from budget check)."""
    return HybridMambaMoEConfig(
        vocab_size=512,
        hidden_size=128,
        num_layers=5,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        intermediate_size=192,
        window_size=64,
        num_experts=4,
        top_k=2,
        dropout=0.0,
        capacity_factor=None,
        max_position_embeddings=1024,
        mamba_state_size=8,
        mamba_conv_kernel=4,
        mamba_expand=2,
        use_dual_memory=True,
        memory_size=32,
        memory_num_heads=4,
        memory_chunk_size=256,
        stream_chunked_ce_loss=True,
        return_logits=False,
        use_auxiliary_losses=True,
    )


def _weighted_terms(
    model: HybridForCausalLM, out, step: int, max_steps: int
) -> dict[str, float]:
    cfg = model.config
    aux = out.auxiliary_losses
    assert aux is not None
    from model import _aux_loss_schedule, _expert_loss_schedule

    assoc_scale = _aux_loss_schedule(step, max_steps, cfg.assoc_warmup_fraction)
    expert_scale = _expert_loss_schedule(step, max_steps, cfg.expert_warmup_fraction)
    return {
        "recon_w": float((cfg.lambda_recon * aux.recon).item()),
        "assoc_w": float((cfg.lambda_assoc * assoc_scale * aux.assoc).item()),
        "gate_w": float((cfg.lambda_gate * aux.gate).item()),
        "read_w": float((cfg.lambda_read * aux.read).item()),
        "fusion_w": float((cfg.lambda_fusion * aux.fusion).item()),
        "expert_w": float((cfg.lambda_expert * expert_scale * aux.expert).item()),
        "ssm_w": float((cfg.lambda_ssm * aux.ssm).item()),
        "slot_w": float((cfg.lambda_slot * aux.slot).item()),
        "assoc_scale": assoc_scale,
        "expert_scale": expert_scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Toy Hybrid Mamba-MoE training smoke test"
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=768)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--log-jsonl",
        type=Path,
        default=Path("toy_train_log.jsonl"),
        help="Optional JSONL log path (set empty to disable)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile decoder layers"
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        help="torch.compile mode (default, reduce-overhead, max-autotune)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = build_toy_config()
    if args.compile:
        cfg.use_torch_compile = True
        cfg.torch_compile_mode = args.compile_mode
    print(log_mamba_backend(cfg))
    model = HybridForCausalLM(cfg).to(device)
    n_params = count_trainable_params(model)
    print(f"trainable_params={n_params:,} (target ~5M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log_path = args.log_jsonl if str(args.log_jsonl) else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    for step in range(args.steps):
        model.train()
        input_ids = torch.randint(
            0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device
        )
        labels = input_ids.roll(shifts=-1, dims=1)
        labels[:, -1] = cfg.label_ignore_index

        optimizer.zero_grad(set_to_none=True)
        out = model(
            input_ids=input_ids,
            labels=labels,
            training_step=step,
            max_training_steps=args.steps,
        )
        assert out.loss is not None
        out.loss.backward()
        clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        aux = out.auxiliary_losses
        assert aux is not None
        weighted = _weighted_terms(model, out, step, args.steps)
        record = {
            "step": step,
            "loss": float(out.loss.item()),
            "ce_loss": float(out.ce_loss.item()) if out.ce_loss is not None else None,
            "router_aux_loss": float(out.router_aux_loss.item())
            if out.router_aux_loss is not None
            else None,
            "router_z_loss": float(out.router_z_loss.item())
            if out.router_z_loss is not None
            else None,
            "recon": float(aux.recon.item()),
            "assoc": float(aux.assoc.item()),
            "gate": float(aux.gate.item()),
            "read": float(aux.read.item()),
            "fusion": float(aux.fusion.item()),
            "expert": float(aux.expert.item()),
            "ssm": float(aux.ssm.item()),
            "slot": float(aux.slot.item()),
            **weighted,
            "gate_stats": {
                k: float(v.item()) for k, v in (out.gate_stats or {}).items()
            },
        }

        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        if step % args.log_every == 0 or step == args.steps - 1:
            assoc_tag = "warm" if weighted["assoc_scale"] == 0.0 else "on"
            expert_tag = "warm" if weighted["expert_scale"] == 0.0 else "on"
            print(
                f"step={step} ce={record['ce_loss']:.3f} "
                f"aux={record['router_aux_loss']:.4f} z={record['router_z_loss']:.4f} "
                f"recon={record['recon']:.3f} assoc={record['assoc']:.3f}({assoc_tag}) "
                f"expert={record['expert']:.3f}({expert_tag}) gate={record['gate']:.3f} "
                f"total={record['loss']:.3f}"
            )


if __name__ == "__main__":
    main()
