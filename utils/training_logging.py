"""Dependency-light formatting helpers shared by training entry points."""

from __future__ import annotations

from typing import Any


def format_training_log_line(step: int, max_steps: int, record: dict[str, Any]) -> str:
    """Format a training record for Muon+AdamW or AdamW-only trainers."""
    assoc_tag = "warm" if record.get("assoc_scale") == 0.0 else "on"
    smooth = record.get("ce_smooth")
    smooth_str = f"{smooth:.6f}" if smooth is not None else "n/a"
    val_ce = record.get("val_ce_loss")
    val_str = f"{val_ce:.6f}" if val_ce is not None else "n/a"
    assoc_norm = record.get("assoc_norm")
    assoc_norm_str = f"assoc_norm={assoc_norm:.6f} " if assoc_norm is not None else ""
    expert_val = record.get("expert", 0.0)
    if record.get("expert_scale", 0.0) == 0.0 and expert_val == 0.0:
        expert_str = "expert=off "
    else:
        expert_tag = "warm" if record.get("expert_scale") == 0.0 else "on"
        expert_str = f"expert={expert_val:.6f}({expert_tag}) "

    lr_parts = []
    if record.get("muon_lr") is not None:
        lr_parts.append(f"muon_lr={record['muon_lr']:.2e}")
    if record.get("adam_lr") is not None:
        lr_parts.append(f"adam_lr={record['adam_lr']:.2e}")
    lr_str = " ".join(lr_parts) if lr_parts else "lr=n/a"

    return (
        f"step={step}/{max_steps} "
        f"shard={record.get('shard_idx', 0)} "
        f"loss={record['loss']:.6f} ce={record['ce_loss']:.6f} "
        f"ce_smooth={smooth_str} val_ce={val_str} "
        f"router_aux={record['router_aux_loss']:.6f} "
        f"router_z={record['router_z_loss']:.6f} "
        f"recon={record['recon']:.6f} assoc={record['assoc']:.6f}({assoc_tag}) "
        f"{assoc_norm_str}"
        f"{expert_str}"
        f"grad_norm={record['grad_norm']:.4f} "
        f"step_time={record.get('step_time_s', float('nan')):.3f}s "
        f"{lr_str}"
    )
