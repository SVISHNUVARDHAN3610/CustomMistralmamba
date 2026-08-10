"""One-epoch IMDB cloud training smoke test for Hybrid Mamba-MoE.

Loads 5 000 rows from ``stanfordnlp/imdb``, tokenizes with
``UIC-AI-lab/llama2-tokenizer``, and runs a single training epoch.
Default model is ~150M trainable params (vocab 32000). Intended for GPU
cloud environments (Colab T4/A100, etc.) — not laptop CPU runs.

Dependencies (install on the cloud host):
    pip install datasets transformers torch
    pip install causal-conv1d mamba-ssm   # optional fused Mamba speedup
"""

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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import (
    HybridForCausalLM,
    HybridMambaMoEConfig,
    _aux_loss_schedule,
    _expert_loss_schedule,
    count_trainable_params,
    fused_mamba_scan_available,
)


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
        "gate_w": float((cfg.lambda_gate * aux.gate).item()),
        "read_w": float((cfg.lambda_read * aux.read).item()),
        "fusion_w": float((cfg.lambda_fusion * aux.fusion).item()),
        "expert_w": float((cfg.lambda_expert * expert_scale * aux.expert).item()),
        "ssm_w": float((cfg.lambda_ssm * aux.ssm).item()),
        "slot_w": float((cfg.lambda_slot * aux.slot).item()),
        "assoc_scale": assoc_scale,
        "expert_scale": expert_scale,
    }


def _load_imdb_dataloader(
    tokenizer: AutoTokenizer,
    max_rows: int,
    batch_size: int,
    seq_len: int,
    num_workers: int,
) -> tuple[DataLoader, object]:
    from datasets import load_dataset

    dataset = load_dataset("stanfordnlp/imdb", split="train")
    dataset = dataset.select(range(min(max_rows, len(dataset))))

    def tokenize_batch(examples: dict) -> dict:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=seq_len,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing IMDB",
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        max_len = max(item["input_ids"].size(0) for item in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, item in enumerate(batch):
            length = item["input_ids"].size(0)
            input_ids[i, :length] = item["input_ids"]
            attention_mask[i, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    loader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, tokenized


def _format_loss_line(step: int, max_steps: int, record: dict[str, float]) -> str:
    assoc_tag = "warm" if record["assoc_scale"] == 0.0 else "on"
    expert_tag = "warm" if record["expert_scale"] == 0.0 else "on"
    return (
        f"step={step}/{max_steps} "
        f"loss={record['loss']:.8f} "
        f"ce={record['ce_loss']:.8f} "
        f"router_aux={record['router_aux_loss']:.8f} "
        f"router_z={record['router_z_loss']:.8f} "
        f"recon={record['recon']:.8f} "
        f"assoc={record['assoc']:.8f}({assoc_tag}) "
        f"gate={record['gate']:.8f} "
        f"read={record['read']:.8f} "
        f"fusion={record['fusion']:.8f} "
        f"expert={record['expert']:.8f}({expert_tag}) "
        f"ssm={record['ssm']:.8f} "
        f"slot={record['slot']:.8f} "
        f"recon_w={record['recon_w']:.8f} "
        f"assoc_w={record['assoc_w']:.8f} "
        f"gate_w={record['gate_w']:.8f} "
        f"read_w={record['read_w']:.8f} "
        f"fusion_w={record['fusion_w']:.8f} "
        f"expert_w={record['expert_w']:.8f} "
        f"ssm_w={record['ssm_w']:.8f} "
        f"slot_w={record['slot_w']:.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-epoch IMDB cloud training smoke test"
    )
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
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
    args = parser.parse_args()

    if args.device == "cpu":
        print(
            "Warning: running on CPU — this script is intended for cloud GPU hosts."
        )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader, tokenized = _load_imdb_dataloader(
        tokenizer=tokenizer,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
    )
    max_steps = len(dataloader)
    vocab_size = resolve_tokenizer_vocab_size(tokenizer, tokenized)
    print(f"imdb_rows={args.max_rows} batches_per_epoch={max_steps}")
    print(
        f"tokenizer_vocab_size={tokenizer.vocab_size} "
        f"resolved_model_vocab_size={vocab_size} "
        f"pad_token_id={tokenizer.pad_token_id}"
    )
    print(f"fused_mamba_scan_available={fused_mamba_scan_available()}")

    cfg = build_cloud_config(vocab_size=vocab_size)
    cfg.pad_token_id = tokenizer.pad_token_id or 0
    cfg.bos_token_id = tokenizer.bos_token_id or 1
    cfg.eos_token_id = tokenizer.eos_token_id or 2

    model = HybridForCausalLM(cfg).to(device)
    n_params = count_trainable_params(model)
    print(f"trainable_params={n_params:,} (target ~150M) vocab_size={cfg.vocab_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log_path = args.log_jsonl if str(args.log_jsonl) else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for batch in dataloader:
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
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            training_step=global_step,
            max_training_steps=max_steps,
        )
        assert out.loss is not None
        out.loss.backward()
        clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        aux = out.auxiliary_losses
        assert aux is not None
        weighted = _weighted_terms(model, out, global_step, max_steps)
        record = {
            "step": global_step,
            "loss": float(out.loss.item()),
            "ce_loss": float(out.ce_loss.item()) if out.ce_loss is not None else 0.0,
            "router_aux_loss": float(out.router_aux_loss.item())
            if out.router_aux_loss is not None
            else 0.0,
            "router_z_loss": float(out.router_z_loss.item())
            if out.router_z_loss is not None
            else 0.0,
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

        if global_step % args.log_every == 0 or global_step == max_steps - 1:
            print(_format_loss_line(global_step, max_steps, record))
            for key, val in record["gate_stats"].items():
                print(f"  gate_stats[{key}]={val:.8f}")

        global_step += 1

    print(f"epoch_complete steps={global_step}")


if __name__ == "__main__":
    main()
