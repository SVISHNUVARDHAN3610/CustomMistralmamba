"""Cyclic validation on Salesforce/wikitext (validation split)."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from model.hybrid.model import HybridForCausalLM


def build_causal_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """Next-token labels; ignore padding and invalid target positions."""
    labels = input_ids.roll(shifts=-1, dims=1)
    labels[:, -1] = ignore_index
    labels = labels.masked_fill(attention_mask == 0, ignore_index)
    next_valid = attention_mask.roll(shifts=-1, dims=1)
    next_valid[:, -1] = 0
    labels = labels.masked_fill(next_valid == 0, ignore_index)
    return labels


def _collate_wikitext_batch(
    batch: list[dict[str, torch.Tensor]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, item in enumerate(batch):
        length = item["input_ids"].size(0)
        input_ids[i, :length] = item["input_ids"]
        attention_mask[i, :length] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


class _WikiTextRowDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        bos_id: int,
        eos_id: int,
    ) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        raw = self.tokenizer.encode(text, add_special_tokens=False)
        # Match training producer: BOS + tokens + EOS, capped at seq_len+1 for labels.
        doc = [self.bos_id] + raw + [self.eos_id]
        doc = doc[: self.seq_len + 1]
        if len(doc) < 2:
            doc = [self.bos_id, self.eos_id]
        ids = torch.tensor(doc, dtype=torch.long)
        return {"input_ids": ids}


class _WikiTextPackedDataset(Dataset):
    """Packed-context validation windows (matches training data packing).

    Training packs documents contiguously: the producer extends one token
    buffer with ``[BOS] + doc + [EOS]`` per document and slices ``seq_len + 1``
    windows, so a training sequence can span document boundaries and every
    token count equals ``seq_len``. Row-independent scoring instead feeds each
    wikitext row as its own (often very short) sequence — short rows produce
    few tokens at inflated per-token loss (no left context), so the resulting
    val loss is systematically higher and noisier than the training CE.

    This dataset replicates the training packing over a FIXED, non-rotating
    slice of rows: tokenize each row as ``[BOS] + raw + [EOS]``, concatenate
    the whole eval set into one stream, and slice into ``seq_len + 1`` windows
    (input = window[:-1], label = window[1:], i.e. pre-shifted like
    ``MmapShardDataset``). Returns ``(input_ids, labels)`` so no roll-based
    relabeling is needed downstream.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        bos_id: int,
        eos_id: int,
    ) -> None:
        stream: list[int] = []
        for text in texts:
            raw = tokenizer.encode(text, add_special_tokens=False)
            if not raw:
                continue
            stream.extend([bos_id] + raw + [eos_id])
        if len(stream) < seq_len + 1:
            raise ValueError(
                f"Packed validation slice has {len(stream)} tokens < "
                f"seq_len+1 ({seq_len + 1}); increase num_rows."
            )
        n_windows = len(stream) // (seq_len + 1)
        usable = n_windows * (seq_len + 1)
        # Stack windows: [n_windows, seq_len+1].
        self.windows = torch.tensor(stream[:usable], dtype=torch.long).view(
            n_windows, seq_len + 1
        )

    def __len__(self) -> int:
        return self.windows.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        window = self.windows[idx]
        # Pre-shifted labels exactly like MmapShardDataset.__getitem__:
        # input = chunk[:-1], labels = chunk[1:]. Do NOT shift again in loss.
        return {"input_ids": window[:-1], "labels": window[1:]}


class WikiTextCyclicValidator:
    """Cycles through wikitext validation rows, ``num_rows`` at a time.

    Two scoring modes:

    - **packed** (default, ``mode='packed'``): a FIXED slice of
      ``eval_rows`` rows (first ``eval_rows`` non-empty rows of the split)
      is tokenized once into one contiguous ``[BOS] doc [EOS] [BOS] doc
      [EOS] ...`` stream and sliced into full ``seq_len`` windows with
      pre-shifted labels — identical packing to training. The same windows
      are scored at every call, removing the rotating-cursor sampling noise;
      cross-document boundaries see left context, as in training.
    - **rows** (legacy, ``mode='rows'``): each row scored independently with
      per-row BOS/EOS, right-padding, and a rotating cursor of ``num_rows``
      rows per call. Kept for comparison with historical metrics.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        seq_len: int,
        num_rows: int = 50,
        batch_size: int = 10,
        dataset_config: str = "wikitext-2-raw-v1",
        bos_id: int = 1,
        eos_id: int = 2,
        pad_token_id: int = 0,
        start_index: int = 0,
        mode: str = "packed",
        eval_rows: int = 500,
    ) -> None:
        if mode not in ("packed", "rows"):
            raise ValueError(
                f"validation mode must be 'packed' or 'rows', got {mode!r}"
            )
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", dataset_config, split="validation")
        self.texts = [row["text"] for row in ds if row["text"].strip()]
        if not self.texts:
            raise ValueError(
                f"No non-empty rows in Salesforce/wikitext validation ({dataset_config})"
            )

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_rows = num_rows
        self.batch_size = batch_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_token_id = pad_token_id
        self.dataset_config = dataset_config
        self.dataset_size = len(self.texts)
        self.cursor = start_index % self.dataset_size
        self.mode = mode
        self.eval_rows = eval_rows
        # Packed windows are deterministic for a given dataset/tokenizer, so
        # they are built lazily on first evaluate() (tokenizing 500+ rows in
        # __init__ would slow training startup) and cached thereafter.
        self._packed_cache: dict[str, _WikiTextPackedDataset] = {}

    @property
    def state_dict(self) -> dict[str, Any]:
        return {
            "cursor": self.cursor,
            "dataset_config": self.dataset_config,
            "dataset_size": self.dataset_size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("dataset_size") != self.dataset_size:
            # Dataset revision mismatch; restart cycle safely.
            self.cursor = 0
            return
        self.cursor = int(state.get("cursor", 0)) % self.dataset_size

    def _next_texts(self) -> tuple[list[str], int, int]:
        """Return the next validation window and its [start, end) cursor range."""
        start = self.cursor
        indices = [(start + i) % self.dataset_size for i in range(self.num_rows)]
        self.cursor = (start + self.num_rows) % self.dataset_size
        texts = [self.texts[i] for i in indices]
        end = (start + self.num_rows) % self.dataset_size
        return texts, start, end

    def _packed_dataset(self) -> _WikiTextPackedDataset:
        """Build (once) the fixed packed eval set over the first eval_rows."""
        cached = self._packed_cache.get(self.dataset_config)
        if cached is None:
            texts = self.texts[: self.eval_rows]
            cached = _WikiTextPackedDataset(
                texts,
                self.tokenizer,
                seq_len=self.seq_len,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
            )
            self._packed_cache[self.dataset_config] = cached
        return cached

    @staticmethod
    def _collate_packed(
        batch: list[dict[str, torch.Tensor]], pad_token_id: int
    ) -> dict[str, torch.Tensor]:
        # All packed windows are exactly seq_len long — just stack them.
        del pad_token_id
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }

    @torch.no_grad()
    def evaluate(
        self,
        model: HybridForCausalLM,
        *,
        device: torch.device,
        global_step: int,
        max_training_steps: int,
        use_amp: bool,
        amp_dtype: torch.dtype,
        ignore_index: int,
    ) -> dict[str, Any]:
        if self.mode == "packed":
            return self._evaluate_packed(
                model,
                device=device,
                global_step=global_step,
                max_training_steps=max_training_steps,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

        texts, row_start, row_end = self._next_texts()
        row_ds = _WikiTextRowDataset(
            texts,
            self.tokenizer,
            seq_len=self.seq_len,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
        )
        loader = DataLoader(
            row_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda batch: _collate_wikitext_batch(batch, self.pad_token_id),
        )

        was_training = model.training
        model.eval()

        totals: dict[str, float] = {
            "loss": 0.0,
            "ce_loss": 0.0,
            "router_aux_loss": 0.0,
            "router_z_loss": 0.0,
        }
        token_weight = 0
        # Router aux/z are per-batch means (like the model's internal aux
        # losses), so they must be averaged per batch — weighting them by
        # active-token counts silently rescales them with each batch's padding.
        batch_count = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = build_causal_labels(input_ids, attention_mask, ignore_index)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    training_step=global_step,
                    max_training_steps=max_training_steps,
                )

            active = int((labels != ignore_index).sum().item())
            if active == 0 or out.loss is None:
                continue

            weight = float(active)
            token_weight += active
            batch_count += 1
            totals["loss"] += float(out.loss.item()) * weight
            if out.ce_loss is not None:
                totals["ce_loss"] += float(out.ce_loss.item()) * weight
            if out.router_aux_loss is not None:
                totals["router_aux_loss"] += float(out.router_aux_loss.item())
            if out.router_z_loss is not None:
                totals["router_z_loss"] += float(out.router_z_loss.item())

        if was_training:
            model.train()

        if token_weight == 0:
            return {
                "event": "validation",
                "mode": "rows",
                "step": global_step,
                "val_loss": float("inf"),
                "val_ce_loss": float("inf"),
                "val_router_aux_loss": float("inf"),
                "val_router_z_loss": float("inf"),
                "val_rows": self.num_rows,
                "val_batch_size": self.batch_size,
                "val_row_start": row_start,
                "val_row_end": row_end,
                "val_cursor": self.cursor,
                "val_dataset": f"Salesforce/wikitext/{self.dataset_config}",
            }

        return {
            "event": "validation",
            "mode": "rows",
            "step": global_step,
            "val_loss": totals["loss"] / token_weight,
            "val_ce_loss": totals["ce_loss"] / token_weight,
            "val_router_aux_loss": totals["router_aux_loss"] / max(batch_count, 1),
            "val_router_z_loss": totals["router_z_loss"] / max(batch_count, 1),
            "val_rows": self.num_rows,
            "val_batch_size": self.batch_size,
            "val_row_start": row_start,
            "val_row_end": row_end,
            "val_cursor": self.cursor,
            "val_dataset": f"Salesforce/wikitext/{self.dataset_config}",
        }

    @torch.no_grad()
    def _evaluate_packed(
        self,
        model: HybridForCausalLM,
        *,
        device: torch.device,
        global_step: int,
        max_training_steps: int,
        use_amp: bool,
        amp_dtype: torch.dtype,
    ) -> dict[str, Any]:
        packed_ds = self._packed_dataset()
        loader = DataLoader(
            packed_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda batch: self._collate_packed(batch, self.pad_token_id),
        )

        was_training = model.training
        model.eval()

        totals: dict[str, float] = {
            "loss": 0.0,
            "ce_loss": 0.0,
            "router_aux_loss": 0.0,
            "router_z_loss": 0.0,
        }
        token_weight = 0
        batch_count = 0
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, dtype=torch.long, device=device
        )

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            n_rows = input_ids.size(0)
            # Packed windows are always full-length with no padding.
            batch_mask = attention_mask[:n_rows]

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                out = model(
                    input_ids=input_ids,
                    attention_mask=batch_mask,
                    labels=labels,
                    training_step=global_step,
                    max_training_steps=max_training_steps,
                )

            # No ignore_index positions exist in packed windows.
            active = int(labels.numel())
            if out.loss is None:
                continue

            weight = float(active)
            token_weight += active
            batch_count += 1
            totals["loss"] += float(out.loss.item()) * weight
            if out.ce_loss is not None:
                totals["ce_loss"] += float(out.ce_loss.item()) * weight
            if out.router_aux_loss is not None:
                totals["router_aux_loss"] += float(out.router_aux_loss.item())
            if out.router_z_loss is not None:
                totals["router_z_loss"] += float(out.router_z_loss.item())

        if was_training:
            model.train()

        base = {
            "event": "validation",
            "mode": "packed",
            "step": global_step,
            "val_batch_size": self.batch_size,
            "val_windows": len(packed_ds),
            "val_eval_rows": min(self.eval_rows, self.dataset_size),
            "val_dataset": f"Salesforce/wikitext/{self.dataset_config}",
        }
        if token_weight == 0:
            base.update(
                {
                    "val_loss": float("inf"),
                    "val_ce_loss": float("inf"),
                    "val_router_aux_loss": float("inf"),
                    "val_router_z_loss": float("inf"),
                }
            )
            return base

        base.update(
            {
                "val_loss": totals["loss"] / token_weight,
                "val_ce_loss": totals["ce_loss"] / token_weight,
                "val_router_aux_loss": totals["router_aux_loss"] / max(batch_count, 1),
                "val_router_z_loss": totals["router_z_loss"] / max(batch_count, 1),
            }
        )
        return base
