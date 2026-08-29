"""One-epoch IMDB cloud training smoke test for Hybrid Mamba-MoE.

Loads rows from ``stanfordnlp/imdb``, tokenizes with
``UIC-AI-lab/llama2-tokenizer``, and runs training with validation CE,
rolling train-CE smoothing, cosine LR (with warmup), and optional early
stopping. Default model is ~200M trainable params (vocab 32000; measured,
excluding aux-only modules). Intended for GPU cloud environments
(Colab T4/A100, etc.) — not laptop CPU runs.

Dependencies (install on the cloud host):
    pip install datasets transformers torch
    pip install causal-conv1d mamba-ssm   # optional fused Mamba speedup
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.core.builders import count_trainable_params
from model.core.config import HybridMambaMoEConfig
from model.core.constants import MEMORY_NAN_FIX_ID
from model.hybrid.losses import _aux_loss_schedule, _expert_loss_schedule
from model.hybrid.mamba import (
    fused_mamba_scan_available,
    get_mamba_scan_stats,
    log_mamba_backend,
    probe_mamba_scan_timing,
    reset_mamba_scan_stats,
)
from model.hybrid.model import HybridForCausalLM


def resolve_tokenizer_vocab_size(
    tokenizer: AutoTokenizer, tokenized_dataset: object | None = None
) -> int:
    """
    Embedding rows must cover every token id the tokenizer can emit.
    Llama-style tokenizers often report vocab_size=32000 but still use id 32000.
    """
    max_id = 0
    if tokenized_dataset is not None and len(tokenized_dataset) > 0:
        for row in tokenized_dataset:
            ids = row["input_ids"]
            max_id = max(max_id, int(ids.max().item()))
    else:
        probe = tokenizer("hello world", return_tensors="pt")
        max_id = int(probe["input_ids"].max().item())

    vocab_candidates = [
        max_id + 1,
        len(tokenizer),
        int(getattr(tokenizer, "vocab_size", 0) or 0),
    ]
    return max(vocab_candidates)


def build_causal_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """Next-token labels; ignore padding and targets that fall on pad positions."""
    labels = input_ids.roll(shifts=-1, dims=1)
    labels[:, -1] = ignore_index
    labels = labels.masked_fill(attention_mask == 0, ignore_index)
    next_valid = attention_mask.roll(shifts=-1, dims=1)
    next_valid[:, -1] = 0
    labels = labels.masked_fill(next_valid == 0, ignore_index)
    return labels


def validate_token_batch(
    input_ids: torch.Tensor,
    vocab_size: int,
    labels: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> None:
    max_id = int(input_ids.max().item())
    min_id = int(input_ids.min().item())
    if min_id < 0 or max_id >= vocab_size:
        raise ValueError(
            f"input_ids out of range [{min_id}, {max_id}] for vocab_size={vocab_size}. "
            "Increase model vocab_size or clamp/filter token ids."
        )
    if labels is not None:
        active = labels != ignore_index
        if active.any():
            label_max = int(labels[active].max().item())
            label_min = int(labels[active].min().item())
            if label_min < 0 or label_max >= vocab_size:
                raise ValueError(
                    f"labels out of range [{label_min}, {label_max}] for "
                    f"vocab_size={vocab_size}."
                )


def build_cloud_config(vocab_size: int) -> HybridMambaMoEConfig:
    """~150M trainable params with Llama-2 vocabulary size (vocab 32000)."""
    hidden_size = 512
    num_heads = 8
    head_dim = 64
    return HybridMambaMoEConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=10,
        num_heads=num_heads,
        num_kv_heads=2,
        head_dim=head_dim,
        intermediate_size=1472,
        window_size=128,
        num_experts=4,
        top_k=2,
        dropout=0.0,
        capacity_factor=None,
        max_position_embeddings=2048,
        mamba_state_size=16,
        mamba_conv_kernel=4,
        mamba_expand=2,
        use_dual_memory=True,
        memory_size=48,
        memory_num_heads=8,
        memory_chunk_size=256,
        stream_chunked_ce_loss=True,
        return_logits=False,
        use_auxiliary_losses=True,
        use_fused_mamba_scan=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def _weighted_terms(
    model: HybridForCausalLM, out, step: int, max_steps: int
) -> dict[str, float]:
    cfg = model.config
    aux = out.auxiliary_losses
    assert aux is not None
    assoc_scale = _aux_loss_schedule(step, max_steps, cfg.assoc_warmup_fraction)
    expert_scale = _expert_loss_schedule(step, max_steps, cfg.expert_warmup_fraction)
    return {
        "recon_w": float((cfg.lambda_recon * aux.recon).item()),
        "assoc_w": float((cfg.lambda_assoc * assoc_scale * aux.assoc).item()),
        "assoc_norm_w": float((cfg.lambda_assoc_norm * aux.assoc_norm).item()),
        "gate_w": float((cfg.lambda_gate * aux.gate).item()),
        "read_w": float((cfg.lambda_read * aux.read).item()),
        "fusion_w": float((cfg.lambda_fusion * aux.fusion).item()),
        "expert_w": float((cfg.lambda_expert * expert_scale * aux.expert).item()),
        "ssm_w": float((cfg.lambda_ssm * aux.ssm).item()),
        "slot_w": float((cfg.lambda_slot * aux.slot).item()),
        "assoc_scale": assoc_scale,
        "expert_scale": expert_scale,
    }


def _scalar_finite_label(value: torch.Tensor | None) -> str:
    """Human-readable scalar value with nan/inf markers for debug logs."""
    if value is None:
        return "none"
    scalar = float(value.detach().float().item())
    if math.isnan(scalar):
        return "nan"
    if math.isinf(scalar):
        return "inf"
    return f"{scalar:.8g}"


def _non_finite_loss_diagnosis(
    model: HybridForCausalLM,
    out,
    step: int,
    max_steps: int,
    attention_mask: torch.Tensor,
) -> dict[str, object]:
    """Identify which loss terms are non-finite and capture values for logging."""
    cfg = model.config
    aux = out.auxiliary_losses
    weighted = _weighted_terms(model, out, step, max_steps)

    router_aux_w: torch.Tensor | None = None
    router_z_w: torch.Tensor | None = None
    if out.router_aux_loss is not None:
        router_aux_w = cfg.router_aux_loss_coef * out.router_aux_loss
    if out.router_z_loss is not None:
        router_z_w = cfg.router_z_loss_coef * out.router_z_loss

    raw_terms: dict[str, torch.Tensor | None] = {
        "loss": out.loss,
        "ce_loss": out.ce_loss,
        "router_aux": out.router_aux_loss,
        "router_z": out.router_z_loss,
        "router_aux_w": router_aux_w,
        "router_z_w": router_z_w,
    }
    if aux is not None:
        for name in (
            "recon",
            "assoc",
            "assoc_norm",
            "gate",
            "read",
            "fusion",
            "expert",
            "ssm",
            "slot",
        ):
            raw_terms[name] = getattr(aux, name)

    non_finite_terms = [
        name
        for name, tensor in raw_terms.items()
        if tensor is not None and not torch.isfinite(tensor).all().item()
    ]

    valid_lens = [int(x) for x in attention_mask.sum(dim=1).tolist()]
    seq_len = int(attention_mask.size(1))

    values = {name: _scalar_finite_label(tensor) for name, tensor in raw_terms.items()}
    for key, val in weighted.items():
        if key.endswith("_w"):
            values[key] = f"{val:.8g}"

    return {
        "non_finite_terms": non_finite_terms,
        "values": values,
        "assoc_scale": weighted["assoc_scale"],
        "expert_scale": weighted["expert_scale"],
        "batch_valid_lens": valid_lens,
        "seq_len": seq_len,
        "has_padding": any(length < seq_len for length in valid_lens),
    }


def _format_non_finite_warning(step: int, diagnosis: dict[str, object]) -> str:
    bad = diagnosis["non_finite_terms"]
    values = diagnosis["values"]
    valid_lens = diagnosis["batch_valid_lens"]
    return (
        f"WARNING: non-finite loss at step={step} "
        f"non_finite_terms={bad} "
        f"valid_lens={valid_lens} seq_len={diagnosis['seq_len']} "
        f"has_padding={diagnosis['has_padding']} "
        f"assoc_scale={diagnosis['assoc_scale']:.4f} "
        f"expert_scale={diagnosis['expert_scale']:.4f} "
        f"loss={values['loss']} ce={values['ce_loss']} "
        f"router_aux={values['router_aux']} router_z={values['router_z']} "
        f"router_aux_w={values['router_aux_w']} router_z_w={values['router_z_w']} "
        f"recon={values['recon']} assoc={values['assoc']} assoc_norm={values['assoc_norm']} "
        f"gate={values['gate']} "
        f"read={values['read']} fusion={values['fusion']} expert={values['expert']} "
        f"ssm={values['ssm']} slot={values['slot']} "
        f"recon_w={values['recon_w']} assoc_w={values['assoc_w']} "
        f"assoc_norm_w={values['assoc_norm_w']} "
        f"gate_w={values['gate_w']} read_w={values['read_w']} "
        f"fusion_w={values['fusion_w']} expert_w={values['expert_w']} "
        f"ssm_w={values['ssm_w']} slot_w={values['slot_w']}"
    )


def _collate_batch(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, item in enumerate(batch):
        length = item["input_ids"].size(0)
        input_ids[i, :length] = item["input_ids"]
        attention_mask[i, :length] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _load_imdb_splits(
    tokenizer: AutoTokenizer,
    max_rows: int,
    val_rows: int,
    batch_size: int,
    seq_len: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader | None, object]:
    from datasets import load_dataset

    dataset = load_dataset("stanfordnlp/imdb", split="train")
    val_rows = min(val_rows, len(dataset))
    total_rows = min(max_rows + val_rows, len(dataset))
    dataset = dataset.select(range(total_rows)).shuffle(seed=seed)

    val_dataset = dataset.select(range(val_rows)) if val_rows > 0 else None
    train_dataset = dataset.select(range(val_rows, total_rows))

    def tokenize_batch(examples: dict) -> dict:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=seq_len,
            padding=False,
        )

    tokenized_train = train_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing IMDB train",
    )
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask"])

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    train_loader = DataLoader(
        tokenized_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate_batch(batch, pad_id),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader: DataLoader | None = None
    if val_dataset is not None and len(val_dataset) > 0:
        tokenized_val = val_dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing IMDB val",
        )
        tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask"])
        val_loader = DataLoader(
            tokenized_val,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: _collate_batch(batch, pad_id),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, tokenized_train


class RollingAverage:
    """Fixed-window mean for noisy per-step train CE."""

    def __init__(self, window: int) -> None:
        self._window = max(1, window)
        self._values: deque[float] = deque(maxlen=self._window)

    def update(self, value: float) -> None:
        self._values.append(value)

    @property
    def mean(self) -> float | None:
        if not self._values:
            return None
        return sum(self._values) / len(self._values)


def _resolve_warmup_steps(warmup_steps: int, total_steps: int) -> int:
    if warmup_steps > 0:
        return min(warmup_steps, max(1, total_steps - 1))
    return min(100, max(1, total_steps // 10))


def _build_lr_lambda(
    warmup_steps: int, total_steps: int, min_lr_ratio: float
) -> callable:
    decay_steps = max(1, total_steps - warmup_steps)
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


@torch.no_grad()
def evaluate_val_ce(
    model: HybridForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
    ignore_index: int,
    training_step: int,
    max_training_steps: int,
    use_amp: bool = False,
) -> float:
    """Token-weighted validation CE (eval mode, CE only)."""
    model.eval()
    total_ce = 0.0
    total_tokens = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = build_causal_labels(input_ids, attention_mask, ignore_index)
        validate_token_batch(
            input_ids,
            model.config.vocab_size,
            labels=labels,
            ignore_index=ignore_index,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                training_step=training_step,
                max_training_steps=max_training_steps,
            )
        assert out.ce_loss is not None
        active = int((labels != ignore_index).sum().item())
        if active == 0:
            continue
        total_ce += float(out.ce_loss.item()) * active
        total_tokens += active
    model.train()
    if total_tokens == 0:
        return float("inf")
    return total_ce / total_tokens


def _format_loss_line(step: int, max_steps: int, record: dict[str, float]) -> str:
    assoc_tag = "warm" if record["assoc_scale"] == 0.0 else "on"
    expert_tag = "warm" if record["expert_scale"] == 0.0 else "on"
    smooth = record.get("train_ce_smooth")
    smooth_str = f"{smooth:.8f}" if smooth is not None else "n/a"
    val_ce = record.get("val_ce")
    val_str = f"{val_ce:.8f}" if val_ce is not None else "n/a"
    return (
        f"step={step}/{max_steps} "
        f"epoch={record.get('epoch', 0)} "
        f"lr={record.get('lr', 0.0):.2e} "
        f"loss={record['loss']:.8f} "
        f"ce={record['ce_loss']:.8f} "
        f"ce_smooth={smooth_str} "
        f"val_ce={val_str} "
        f"router_aux={record['router_aux_loss']:.8f} "
        f"router_z={record['router_z_loss']:.8f} "
        f"recon={record['recon']:.8f} "
        f"assoc={record['assoc']:.8f}({assoc_tag}) "
        f"assoc_norm={record.get('assoc_norm', 0.0):.8f} "
        f"gate={record['gate']:.8f} "
        f"read={record['read']:.8f} "
        f"fusion={record['fusion']:.8f} "
        f"expert={record['expert']:.8f}({expert_tag}) "
        f"ssm={record['ssm']:.8f} "
        f"slot={record['slot']:.8f} "
        f"recon_w={record['recon_w']:.8f} "
        f"assoc_w={record['assoc_w']:.8f} "
        f"assoc_norm_w={record.get('assoc_norm_w', 0.0):.8f} "
        f"gate_w={record['gate_w']:.8f} "
        f"read_w={record['read_w']:.8f} "
        f"fusion_w={record['fusion_w']:.8f} "
        f"expert_w={record['expert_w']:.8f} "
        f"ssm_w={record['ssm_w']:.8f} "
        f"slot_w={record['slot_w']:.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IMDB cloud training smoke test with val CE and cosine LR"
    )
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--val-rows", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smooth-window", type=int, default=50)
    parser.add_argument(
        "--val-every",
        type=int,
        default=200,
        help="Run validation every N train steps (0 disables)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="LR warmup steps (0 = auto: min(100, total_steps//10))",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="Cosine floor as a fraction of --lr",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after N val checks without val_ce improvement (0 disables)",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="UIC-AI-lab/llama2-tokenizer",
    )
    parser.add_argument(
        "--log-jsonl",
        type=Path,
        default=Path("cloud_train_log.jsonl"),
        help="Optional JSONL log path (set empty to disable)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile decoder layers",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        help="torch.compile mode (default, reduce-overhead, max-autotune)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable FP16 autocast + GradScaler (full-precision forward on CUDA)",
    )
    args = parser.parse_args()

    if args.device == "cpu":
        print("Warning: running on CPU — this script is intended for cloud GPU hosts.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader, tokenized_train = _load_imdb_splits(
        tokenizer=tokenizer,
        max_rows=args.max_rows,
        val_rows=args.val_rows,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = _resolve_warmup_steps(args.warmup_steps, total_steps)
    vocab_size = resolve_tokenizer_vocab_size(tokenizer, tokenized_train)
    print(
        f"imdb_train_rows={args.max_rows} imdb_val_rows={args.val_rows} "
        f"batches_per_epoch={steps_per_epoch} epochs={args.epochs} "
        f"total_steps={total_steps}"
    )
    print(
        f"tokenizer_vocab_size={tokenizer.vocab_size} "
        f"resolved_model_vocab_size={vocab_size} "
        f"pad_token_id={tokenizer.pad_token_id}"
    )
    print(
        f"fused_mamba_scan_available={fused_mamba_scan_available()} "
        f"warmup_steps={warmup_steps} val_every={args.val_every} "
        f"smooth_window={args.smooth_window}"
    )

    cfg = build_cloud_config(vocab_size=vocab_size)
    cfg.pad_token_id = tokenizer.pad_token_id or 0
    cfg.bos_token_id = tokenizer.bos_token_id or 1
    cfg.eos_token_id = tokenizer.eos_token_id or 2
    if args.compile:
        cfg.use_torch_compile = True
        cfg.torch_compile_mode = args.compile_mode

    print(log_mamba_backend(cfg))
    if device.type == "cuda" and not fused_mamba_scan_available():
        print(
            "WARNING: mamba-ssm not installed. Install with: "
            "pip install causal-conv1d mamba-ssm"
        )
    if device.type == "cuda":
        print(
            probe_mamba_scan_timing(
                cfg, batch_size=2, seq_len=min(args.seq_len, 512), device=device
            )
        )
    model = HybridForCausalLM(cfg).to(device)
    n_params = count_trainable_params(model)
    print(f"trainable_params={n_params:,} (target ~200M) vocab_size={cfg.vocab_size}")
    import model as model_pkg

    print(f"model_source={model_pkg.__file__} memory_nan_fix={MEMORY_NAN_FIX_ID}")

    use_amp = device.type == "cuda" and not args.no_amp
    if use_amp:
        print("mixed_precision=fp16 torch.autocast(cuda) + GradScaler enabled")
    elif device.type == "cuda" and args.no_amp:
        print("mixed_precision=disabled (--no-amp, full precision on CUDA)")
    else:
        print("mixed_precision=disabled (CPU or non-CUDA device)")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_build_lr_lambda(warmup_steps, total_steps, args.min_lr_ratio),
    )
    ce_smooth = RollingAverage(args.smooth_window)
    # `--log-jsonl ""` means "disable" per the flag's help; str(Path("")) is
    # "." (truthy), so test the raw string, not the Path.
    log_path = args.log_jsonl if args.log_jsonl and args.log_jsonl != "." else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_ce = float("inf")
    patience_left = args.early_stop_patience
    last_val_ce: float | None = None
    global_step = 0
    stop_training = False
    reset_mamba_scan_stats()

    for epoch in range(args.epochs):
        if stop_training:
            break
        for batch in train_loader:
            model.train()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = build_causal_labels(
                input_ids, attention_mask, cfg.label_ignore_index
            )
            validate_token_batch(
                input_ids,
                cfg.vocab_size,
                labels=labels,
                ignore_index=cfg.label_ignore_index,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    training_step=global_step,
                    max_training_steps=total_steps,
                )
            assert out.loss is not None
            if not torch.isfinite(out.loss):
                diagnosis = _non_finite_loss_diagnosis(
                    model, out, global_step, total_steps, attention_mask
                )
                print(_format_non_finite_warning(global_step, diagnosis))
                if log_path is not None:
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "event": "non_finite_loss",
                                    "step": global_step,
                                    "epoch": epoch,
                                    **diagnosis,
                                }
                            )
                            + "\n"
                        )
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue
            if scaler is not None:
                scaler.scale(out.loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                out.loss.backward()
                clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            scheduler.step()

            step_ce = float(out.ce_loss.item()) if out.ce_loss is not None else 0.0
            ce_smooth.update(step_ce)

            run_val = (
                val_loader is not None
                and args.val_every > 0
                and global_step % args.val_every == 0
            )
            if run_val:
                last_val_ce = evaluate_val_ce(
                    model,
                    val_loader,
                    device,
                    cfg.label_ignore_index,
                    training_step=global_step,
                    max_training_steps=total_steps,
                    use_amp=use_amp,
                )
                if last_val_ce < best_val_ce:
                    best_val_ce = last_val_ce
                    if args.early_stop_patience > 0:
                        patience_left = args.early_stop_patience
                elif args.early_stop_patience > 0:
                    patience_left -= 1
                    if patience_left <= 0:
                        print(
                            f"early_stop step={global_step} "
                            f"best_val_ce={best_val_ce:.6f} "
                            f"val_ce={last_val_ce:.6f}"
                        )
                        stop_training = True

            aux = out.auxiliary_losses
            assert aux is not None
            weighted = _weighted_terms(model, out, global_step, total_steps)
            current_lr = float(scheduler.get_last_lr()[0])
            record = {
                "step": global_step,
                "epoch": epoch,
                "lr": current_lr,
                "loss": float(out.loss.item()),
                "ce_loss": step_ce,
                "train_ce_smooth": ce_smooth.mean,
                "val_ce": last_val_ce,
                "best_val_ce": best_val_ce if best_val_ce != float("inf") else None,
                "router_aux_loss": float(out.router_aux_loss.item())
                if out.router_aux_loss is not None
                else 0.0,
                "router_z_loss": float(out.router_z_loss.item())
                if out.router_z_loss is not None
                else 0.0,
                "recon": float(aux.recon.item()),
                "assoc": float(aux.assoc.item()),
                "assoc_norm": float(aux.assoc_norm.item()),
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

            if global_step % args.log_every == 0 or global_step == total_steps - 1:
                print(_format_loss_line(global_step, total_steps, record))
                scan_stats = get_mamba_scan_stats()
                print(
                    f"  mamba_scan_stats fused_full={scan_stats['fused_full_batch']} "
                    f"fused_unpadded={scan_stats['fused_unpadded_batch']} "
                    f"pytorch_fallback={scan_stats['pytorch_fallback']}"
                )
                for key, val in record["gate_stats"].items():
                    print(f"  gate_stats[{key}]={val:.8f}")
                if last_val_ce is not None:
                    print(
                        f"  val_summary best_val_ce={best_val_ce:.8f} "
                        f"last_val_ce={last_val_ce:.8f}"
                    )

            global_step += 1
            if stop_training:
                break

    if val_loader is not None and global_step > 0:
        final_val_ce = evaluate_val_ce(
            model,
            val_loader,
            device,
            cfg.label_ignore_index,
            training_step=global_step - 1,
            max_training_steps=total_steps,
            use_amp=use_amp,
        )
        print(
            f"training_complete steps={global_step} "
            f"best_val_ce={best_val_ce:.6f} final_val_ce={final_val_ce:.6f}"
        )
    else:
        print(f"training_complete steps={global_step}")


if __name__ == "__main__":
    main()
