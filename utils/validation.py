"""Cyclic validation on Salesforce/wikitext (validation split).

T-3 fix: ``PackedWindowValidator`` evaluates on a fixed packed token stream
with identical tokenizer/context/shift semantics as training, making the
validation CE directly comparable to training CE.  The legacy
``WikiTextCyclicValidator`` (row-based, padded) is preserved for backward
compatibility but should not be used for convergence decisions.
"""

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


class WikiTextCyclicValidator:
    """Cycles through wikitext validation rows, ``num_rows`` at a time."""

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
    ) -> None:
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


class _PackedWindowDataset(Dataset):
    """Fixed packed token stream split into non-overlapping (input, label) windows.

    Identical shift semantics to ``MmapShardDataset.__getitem__``:
    ``input_ids = stream[start:start+seq_len]``,
    ``labels    = stream[start+1:start+seq_len+1]``.
    """

    def __init__(self, token_stream: list[int], seq_len: int) -> None:
        self._stream = token_stream
        self._seq_len = seq_len
        # Number of complete windows that fit in the stream.
        self._n_windows = (len(token_stream) - 1) // seq_len

    def __len__(self) -> int:
        return self._n_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self._seq_len
        chunk = self._stream[start : start + self._seq_len + 1]
        ids = torch.tensor(chunk, dtype=torch.long)
        return ids[:-1], ids[1:]  # input_ids, labels


class PackedWindowValidator:
    """T-3 fix: evaluate on packed continuous token windows.

    Unlike ``WikiTextCyclicValidator``, this validator concatenates a fixed
    snapshot of Salesforce/wikitext validation texts into a single packed token
    stream and scores it on non-overlapping ``seq_len``-token windows — exactly
    matching training data tokenization and loss reduction semantics.

    The resulting ``packed_val_ce`` metric is directly comparable to training CE
    and enables reliable convergence decisions.  The legacy row-based validator
    is still available for comparison under ``val_ce_loss`` in the JSONL log.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        seq_len: int,
        num_texts: int = 200,
        batch_size: int = 8,
        dataset_config: str = "wikitext-2-raw-v1",
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> None:
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset("Salesforce/wikitext", dataset_config, split="validation")
        texts = [row["text"] for row in ds if row["text"].strip()]
        if not texts:
            raise ValueError(
                f"No non-empty rows in Salesforce/wikitext validation ({dataset_config})"
            )
        # Use the first num_texts texts (fixed snapshot for reproducibility).
        texts = texts[:num_texts]

        # Build a packed token stream: BOS + tokens + EOS per document,
        # concatenated into one continuous uint-id list.
        stream: list[int] = []
        for text in texts:
            raw = tokenizer.encode(text, add_special_tokens=False)
            stream.extend([bos_id] + raw + [eos_id])

        self._dataset = _PackedWindowDataset(stream, seq_len)
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._dataset_config = dataset_config
        self._num_texts = num_texts
        self._stream_len = len(stream)

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
        """Return a dict with ``packed_val_ce`` and metadata."""
        loader = DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            shuffle=False,
            drop_last=False,
        )

        was_training = model.training
        model.eval()

        total_ce = 0.0
        total_tokens = 0
        n_windows = 0

        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                out = model(
                    input_ids=input_ids,
                    labels=labels,
                    training_step=global_step,
                    max_training_steps=max_training_steps,
                )

            if out.ce_loss is None or out.loss is None:
                continue

            n_tokens = int((labels != ignore_index).sum().item())
            if n_tokens == 0:
                continue

            total_ce += float(out.ce_loss.item()) * n_tokens
            total_tokens += n_tokens
            n_windows += input_ids.size(0)

        if was_training:
            model.train()

        if total_tokens == 0:
            return {
                "event": "packed_validation",
                "step": global_step,
                "packed_val_ce": float("inf"),
                "packed_val_windows": 0,
                "packed_val_tokens": 0,
                "packed_val_dataset": f"Salesforce/wikitext/{self._dataset_config}",
            }

        return {
            "event": "packed_validation",
            "step": global_step,
            "packed_val_ce": total_ce / total_tokens,
            "packed_val_windows": n_windows,
            "packed_val_tokens": total_tokens,
            "packed_val_dataset": f"Salesforce/wikitext/{self._dataset_config}",
            "packed_stream_len": self._stream_len,
            "packed_num_texts": self._num_texts,
        }
