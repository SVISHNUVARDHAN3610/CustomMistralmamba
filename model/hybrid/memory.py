"""Compressive memory banks and write buffers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.core.dtype import _is_low_precision, _promote_fp32, _restore_dtype
from model.core.fsdp import local_dtensor
from model.hybrid.losses import MemoryReconstructionDecoder


def _assert_right_padded_attention_mask(
    attention_mask: Tensor, debug_state_checks: bool
) -> None:
    """Valid tokens must form a left prefix (right-padding), not interior holes."""
    if not debug_state_checks:
        return
    valid = attention_mask.bool()
    for b in range(valid.size(0)):
        row = valid[b]
        n_valid = int(row.sum().item())
        if n_valid == 0:
            continue
        prefix = row[:n_valid]
        if not prefix.all() or row[n_valid:].any():
            raise ValueError(
                "Mamba prefill cache init requires right-padded attention_mask "
                "(valid tokens form a left prefix)."
            )


def _materialize_write_buffer(
    write_buffer: MemoryWriteBuffer | None,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """Materialize pre-allocated write buffer into tensors for memory write.

    Returns ``(attn, mamba, valid_mask)`` where ``valid_mask`` is ``[B, L]``
    bool (True = real token). Empty buffers return ``(None, None, None)``.
    """
    if write_buffer is None or write_buffer.filled == 0:
        return None, None, None
    return write_buffer.materialize()


def _write_buffer_token_len(
    write_buffer: MemoryWriteBuffer | None,
) -> int:
    if write_buffer is None:
        return 0
    return write_buffer.token_len()


class MemoryWriteBuffer:
    """Pre-allocated buffer for chunked memory writes (amortized O(k) append).

    Stores a per-token validity mask with the branch outputs so write/aux
    paths never reconstruct prior pads as ``torch.ones`` (which incorrectly
    treats padding as valid when buffers span multiple appends).
    """

    __slots__ = (
        "attn_buf",
        "batch_size",
        "capacity",
        "filled",
        "hidden_size",
        "mamba_buf",
        "mask_buf",
    )

    def __init__(
        self,
        batch_size: int,
        hidden_size: int,
        capacity: int = 512,
    ) -> None:
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.capacity = max(1, capacity)
        self.filled = 0
        self.attn_buf: Tensor | None = None
        self.mamba_buf: Tensor | None = None
        self.mask_buf: Tensor | None = None

    def _ensure_capacity(
        self, add_tokens: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        needed = self.filled + add_tokens
        if self.attn_buf is None:
            cap = max(self.capacity, needed)
            self.attn_buf = torch.zeros(
                self.batch_size, cap, self.hidden_size, device=device, dtype=dtype
            )
            self.mamba_buf = torch.zeros(
                self.batch_size, cap, self.hidden_size, device=device, dtype=dtype
            )
            self.mask_buf = torch.zeros(
                self.batch_size, cap, device=device, dtype=torch.bool
            )
            self.capacity = cap
            return
        if needed > self.capacity:
            new_cap = max(self.capacity * 2, needed)
            assert self.mamba_buf is not None and self.mask_buf is not None
            new_attn = torch.zeros(
                self.batch_size, new_cap, self.hidden_size, device=device, dtype=dtype
            )
            new_mamba = torch.zeros(
                self.batch_size, new_cap, self.hidden_size, device=device, dtype=dtype
            )
            new_mask = torch.zeros(
                self.batch_size, new_cap, device=device, dtype=torch.bool
            )
            new_attn[:, : self.filled] = self.attn_buf[:, : self.filled]
            new_mamba[:, : self.filled] = self.mamba_buf[:, : self.filled]
            new_mask[:, : self.filled] = self.mask_buf[:, : self.filled]
            self.attn_buf = new_attn
            self.mamba_buf = new_mamba
            self.mask_buf = new_mask
            self.capacity = new_cap

    @staticmethod
    def _normalize_valid_mask(
        valid_mask: Tensor | None,
        batch_size: int,
        add: int,
        device: torch.device,
    ) -> Tensor:
        if valid_mask is None:
            return torch.ones(batch_size, add, device=device, dtype=torch.bool)
        mask = valid_mask.bool()
        if mask.dim() == 1:
            mask = mask.unsqueeze(-1)
        if mask.dim() != 2 or mask.size(0) != batch_size or mask.size(1) != add:
            raise ValueError(
                f"valid_mask must be [B, add]=[{batch_size}, {add}], got {tuple(mask.shape)}."
            )
        return mask

    def append(
        self,
        attn: Tensor,
        mamba: Tensor,
        valid_mask: Tensor | None = None,
    ) -> None:
        add = attn.size(1)
        self._ensure_capacity(add, attn.device, attn.dtype)
        assert (
            self.attn_buf is not None
            and self.mamba_buf is not None
            and self.mask_buf is not None
        )
        mask = self._normalize_valid_mask(valid_mask, self.batch_size, add, attn.device)
        # Keep pad slots zero so a wrong write_mask cannot attend to junk.
        keep = mask.unsqueeze(-1)
        self.attn_buf[:, self.filled : self.filled + add] = torch.where(
            keep, attn, torch.zeros_like(attn)
        )
        self.mamba_buf[:, self.filled : self.filled + add] = torch.where(
            keep, mamba, torch.zeros_like(mamba)
        )
        self.mask_buf[:, self.filled : self.filled + add] = mask
        self.filled += add

    def append_single_token(
        self,
        attn: Tensor,
        mamba: Tensor,
        valid_mask: Tensor | None = None,
    ) -> None:
        """Fast path for decode: append one token without realloc after warm-up."""
        if self.attn_buf is None or self.filled >= self.capacity:
            self._ensure_capacity(1, attn.device, attn.dtype)
        assert (
            self.attn_buf is not None
            and self.mamba_buf is not None
            and self.mask_buf is not None
        )
        mask = self._normalize_valid_mask(valid_mask, self.batch_size, 1, attn.device)
        keep = mask.unsqueeze(-1)
        self.attn_buf[:, self.filled : self.filled + 1] = torch.where(
            keep, attn, torch.zeros_like(attn)
        )
        self.mamba_buf[:, self.filled : self.filled + 1] = torch.where(
            keep, mamba, torch.zeros_like(mamba)
        )
        self.mask_buf[:, self.filled : self.filled + 1] = mask
        self.filled += 1

    def materialize(self) -> tuple[Tensor, Tensor, Tensor]:
        assert (
            self.attn_buf is not None
            and self.mamba_buf is not None
            and self.mask_buf is not None
        )
        return (
            self.attn_buf[:, : self.filled],
            self.mamba_buf[:, : self.filled],
            self.mask_buf[:, : self.filled],
        )

    def token_len(self) -> int:
        return self.filled


class CompressiveMemoryBank(nn.Module):
    """
    Fixed-size (m slots) gated read/write memory bank.

    Multi-head scaled-dot attention over a small bank (m << L).
    """

    def __init__(
        self,
        hidden_size: int,
        memory_size: int = 64,
        num_heads: int = 8,
        recon_decoder_heads: int = 2,
        enable_aux_modules: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by memory_num_heads ({num_heads})."
            )
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.init_memory = nn.Parameter(torch.randn(memory_size, hidden_size) * 0.02)
        self.summary_query = nn.Parameter(torch.randn(memory_size, hidden_size) * 0.02)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.write_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.write_update = nn.Linear(hidden_size, hidden_size)

        if enable_aux_modules:
            self.recon_decoder = MemoryReconstructionDecoder(
                hidden_size, recon_decoder_heads
            )
            self.assoc_key = nn.Linear(hidden_size, hidden_size, bias=False)
            self.assoc_val = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.recon_decoder = None
            self.assoc_key = None
            self.assoc_val = None

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        return (
            local_dtensor(self.init_memory)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .to(device=device, dtype=dtype)
            .clone()
        )

    @staticmethod
    def _key_padding_mask(attention_mask: Tensor | None) -> Tensor | None:
        """Convert 1=keep / 0=pad mask to key_padding_mask (True = ignore)."""
        if attention_mask is None:
            return None
        if attention_mask.dim() != 2:
            raise ValueError(
                "CompressiveMemoryBank expects a 2D attention_mask [B, L]."
            )
        return ~attention_mask.bool()

    def _shape_heads(self, t: Tensor, seq_len: int) -> Tensor:
        # [B, L, H] -> [B, heads, L, head_dim]
        b = t.size(0)
        return t.reshape(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _attend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        fast_path: bool = False,
    ) -> Tensor:
        """
        query: [B, Q, H], key/value: [B, K, H]
        key_padding_mask: [B, K] True = ignore
        """
        bsz, q_len, _ = query.shape
        k_len = key.size(1)
        if fast_path and q_len * k_len <= 256:
            q = self.q_proj(query).view(bsz, q_len, self.num_heads, self.head_dim)
            k = self.k_proj(key).view(bsz, k_len, self.num_heads, self.head_dim)
            v = self.v_proj(value).view(bsz, k_len, self.num_heads, self.head_dim)
            scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * self.scale
            if key_padding_mask is not None:
                scores = scores.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )
                all_masked = key_padding_mask.all(dim=-1)
                if all_masked.any():
                    scores = scores.clone()
                    scores[all_masked] = 0.0
            attn = F.softmax(scores, dim=-1)
            if key_padding_mask is not None and all_masked.any():
                attn = attn.clone()
                attn[all_masked] = 0.0
            out = torch.einsum("bhqk,bkhd->bqhd", attn, v)
            out = out.reshape(bsz, q_len, self.hidden_size)
            return self.out_proj(out)

        q = self._shape_heads(self.q_proj(query), q_len)
        k = self._shape_heads(self.k_proj(key), k_len)
        v = self._shape_heads(self.v_proj(value), k_len)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            all_masked = key_padding_mask.all(dim=-1)  # [B]
            if all_masked.any():
                # Avoid NaN softmax when a row has no valid keys.
                scores = scores.clone()
                scores[all_masked] = 0.0
        attn = F.softmax(scores, dim=-1)
        if key_padding_mask is not None and all_masked.any():
            attn = attn.clone()
            attn[all_masked] = 0.0
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.out_proj(out)

    def read(
        self,
        x: Tensor,
        memory: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        del attention_mask  # queries may be zeroed by caller for pads
        return self._attend(x, memory, memory, key_padding_mask=None)

    def read_query(self, query: Tensor, memory: Tensor) -> Tensor:
        return self._attend(query, memory, memory, key_padding_mask=None)

    def write(
        self,
        x: Tensor,
        memory: Tensor,
        attention_mask: Tensor | None = None,
        fast_path: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        kpm = self._key_padding_mask(attention_mask)
        batch_size = x.size(0)
        query = local_dtensor(self.summary_query).unsqueeze(0).expand(batch_size, -1, -1)
        chunk_summary = self._attend(
            query, x, x, key_padding_mask=kpm, fast_path=fast_path
        )

        gate = torch.sigmoid(
            self.write_gate(torch.cat([memory, chunk_summary], dim=-1))
        )
        new_memory = gate * memory + (1.0 - gate) * self.write_update(chunk_summary)
        if kpm is not None:
            all_masked_rows = kpm.all(dim=-1)
            if all_masked_rows.any():
                new_memory = new_memory.clone()
                new_memory[all_masked_rows] = memory[all_masked_rows]
        return new_memory, gate, chunk_summary


def batched_dual_memory_read(
    bank_a: CompressiveMemoryBank,
    bank_b: CompressiveMemoryBank,
    x: Tensor,
    mem_a: Tensor,
    mem_b: Tensor,
) -> tuple[Tensor, Tensor]:
    """Single batched attention for both memory banks (stacked projections)."""
    bsz, seq_len, hidden = x.shape
    m_len = mem_a.size(1)
    num_heads = bank_a.num_heads
    head_dim = bank_a.head_dim
    scale = bank_a.scale

    queries = torch.cat([x, x], dim=0)
    keys = torch.cat([mem_a, mem_b], dim=0)
    values = keys
    bank_idx = torch.arange(2 * bsz, device=x.device) // bsz

    q_w = torch.stack(
        [local_dtensor(bank_a.q_proj.weight), local_dtensor(bank_b.q_proj.weight)],
        dim=0,
    )
    k_w = torch.stack(
        [local_dtensor(bank_a.k_proj.weight), local_dtensor(bank_b.k_proj.weight)],
        dim=0,
    )
    v_w = torch.stack(
        [local_dtensor(bank_a.v_proj.weight), local_dtensor(bank_b.v_proj.weight)],
        dim=0,
    )
    out_w = torch.stack(
        [
            local_dtensor(bank_a.out_proj.weight),
            local_dtensor(bank_b.out_proj.weight),
        ],
        dim=0,
    )

    q = torch.bmm(queries, q_w[bank_idx].transpose(1, 2))
    k = torch.bmm(keys, k_w[bank_idx].transpose(1, 2))
    v = torch.bmm(values, v_w[bank_idx].transpose(1, 2))

    q = q.view(2 * bsz, seq_len, num_heads, head_dim)
    k = k.view(2 * bsz, m_len, num_heads, head_dim)
    v = v.view(2 * bsz, m_len, num_heads, head_dim)

    # Promote v alongside q/k: outside autocast (native-bf16 weights) a
    # fp32-attn x bf16-v einsum raises a dtype-mismatch RuntimeError. Under
    # autocast this is a no-op (autocast already widened both operands).
    if _is_low_precision(q.dtype):
        q = _promote_fp32(q)
        k = _promote_fp32(k)
        v = _promote_fp32(v)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    attn = F.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", attn, v)
    out = out.reshape(2 * bsz, seq_len, hidden)
    # Restore activation dtype before the out_proj bmm so the stacked weights
    # (native weight dtype) and `out` always match outside autocast too.
    out = _restore_dtype(out, queries.dtype)
    out = torch.bmm(out, out_w[bank_idx].transpose(1, 2))

    return out[:bsz], out[bsz:]


def _batched_memory_summarize(
    bank_a: CompressiveMemoryBank,
    bank_b: CompressiveMemoryBank,
    buf_a: Tensor,
    buf_b: Tensor,
    key_padding_mask: Tensor | None,
    fast_path: bool,
) -> tuple[Tensor, Tensor]:
    """Batched summary_query attention for dual-bank writes."""
    bsz = buf_a.size(0)
    m_len = bank_a.memory_size
    num_heads = bank_a.num_heads
    head_dim = bank_a.head_dim
    scale = bank_a.scale
    buf_len = buf_a.size(1)

    queries = torch.cat(
        [
            local_dtensor(bank_a.summary_query).unsqueeze(0).expand(bsz, -1, -1),
            local_dtensor(bank_b.summary_query).unsqueeze(0).expand(bsz, -1, -1),
        ],
        dim=0,
    )
    keys = torch.cat([buf_a, buf_b], dim=0)
    values = keys
    bank_idx = torch.arange(2 * bsz, device=buf_a.device) // bsz

    q_w = torch.stack(
        [local_dtensor(bank_a.q_proj.weight), local_dtensor(bank_b.q_proj.weight)],
        dim=0,
    )
    k_w = torch.stack(
        [local_dtensor(bank_a.k_proj.weight), local_dtensor(bank_b.k_proj.weight)],
        dim=0,
    )
    v_w = torch.stack(
        [local_dtensor(bank_a.v_proj.weight), local_dtensor(bank_b.v_proj.weight)],
        dim=0,
    )
    out_w = torch.stack(
        [
            local_dtensor(bank_a.out_proj.weight),
            local_dtensor(bank_b.out_proj.weight),
        ],
        dim=0,
    )

    q = torch.bmm(queries, q_w[bank_idx].transpose(1, 2))
    k = torch.bmm(keys, k_w[bank_idx].transpose(1, 2))
    v = torch.bmm(values, v_w[bank_idx].transpose(1, 2))
    q = q.view(2 * bsz, m_len, num_heads, head_dim)
    k = k.view(2 * bsz, buf_len, num_heads, head_dim)
    v = v.view(2 * bsz, buf_len, num_heads, head_dim)
    # Promote v alongside q/k (see batched_dual_memory_read): native-bf16
    # callers crash on a fp32-attn x bf16-v einsum without autocast.
    if _is_low_precision(q.dtype):
        q = _promote_fp32(q)
        k = _promote_fp32(k)
        v = _promote_fp32(v)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    all_masked: Tensor | None = None
    if key_padding_mask is not None:
        kpm = torch.cat([key_padding_mask, key_padding_mask], dim=0)
        scores = scores.masked_fill(kpm.unsqueeze(1).unsqueeze(2), float("-inf"))
        all_masked = kpm.all(dim=-1)
        if all_masked.any():
            # Match CompressiveMemoryBank._attend: softmax over all -inf keys is NaN.
            scores = scores.clone()
            scores[all_masked] = 0.0
    attn = F.softmax(scores, dim=-1)
    if all_masked is not None and all_masked.any():
        attn = attn.clone()
        attn[all_masked] = 0.0
    out = torch.einsum("bhqk,bkhd->bqhd", attn, v)
    out = out.reshape(2 * bsz, m_len, bank_a.hidden_size)
    # Restore buffer dtype before the stacked out_proj bmm (native-bf16 safe).
    out = _restore_dtype(out, keys.dtype)
    out = torch.bmm(out, out_w[bank_idx].transpose(1, 2))

    return out[:bsz], out[bsz:]


def batched_dual_memory_write(
    bank_a: CompressiveMemoryBank,
    bank_b: CompressiveMemoryBank,
    buf_attn: Tensor,
    buf_mamba: Tensor,
    mem_a: Tensor,
    mem_b: Tensor,
    attention_mask: Tensor | None = None,
    fast_path: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fused write for both memory banks (stacked projections)."""
    kpm: Tensor | None = None
    if attention_mask is not None:
        kpm = ~attention_mask.bool()

    a_summary, s_summary = _batched_memory_summarize(
        bank_a, bank_b, buf_attn, buf_mamba, kpm, fast_path
    )

    gate_w = torch.stack(
        [
            local_dtensor(bank_a.write_gate.weight),
            local_dtensor(bank_b.write_gate.weight),
        ],
        dim=0,
    )
    gate_b = torch.stack(
        [
            local_dtensor(bank_a.write_gate.bias)
            if bank_a.write_gate.bias is not None
            else torch.zeros(
                bank_a.hidden_size, device=buf_attn.device, dtype=buf_attn.dtype
            ),
            local_dtensor(bank_b.write_gate.bias)
            if bank_b.write_gate.bias is not None
            else torch.zeros(
                bank_b.hidden_size, device=buf_attn.device, dtype=buf_attn.dtype
            ),
        ],
        dim=0,
    )
    update_w = torch.stack(
        [
            local_dtensor(bank_a.write_update.weight),
            local_dtensor(bank_b.write_update.weight),
        ],
        dim=0,
    )
    bsz = buf_attn.size(0)
    bank_idx = torch.arange(2 * bsz, device=buf_attn.device) // bsz

    mem_stacked = torch.cat([mem_a, mem_b], dim=0)
    summary_stacked = torch.cat([a_summary, s_summary], dim=0)
    gate_in = torch.cat([mem_stacked, summary_stacked], dim=-1)
    out_dtype = mem_stacked.dtype
    if _is_low_precision(out_dtype):
        gate_logits = torch.bmm(
            _promote_fp32(gate_in),
            gate_w.float()[bank_idx].transpose(1, 2),
        ) + gate_b.float()[bank_idx].unsqueeze(1)
        gate = torch.sigmoid(gate_logits).to(out_dtype)
        updates = torch.bmm(
            _promote_fp32(summary_stacked),
            update_w.float()[bank_idx].transpose(1, 2),
        ).to(out_dtype)
    else:
        gate_logits = torch.bmm(gate_in, gate_w[bank_idx].transpose(1, 2)) + gate_b[
            bank_idx
        ].unsqueeze(1)
        gate = torch.sigmoid(gate_logits)
        updates = torch.bmm(summary_stacked, update_w[bank_idx].transpose(1, 2))
    new_mem_stacked = gate * mem_stacked + (1.0 - gate) * updates
    new_a = new_mem_stacked[:bsz]
    new_s = new_mem_stacked[bsz:]

    if kpm is not None:
        all_masked_rows = kpm.all(dim=-1)
        if all_masked_rows.any():
            new_a = new_a.clone()
            new_s = new_s.clone()
            new_a[all_masked_rows] = mem_a[all_masked_rows]
            new_s[all_masked_rows] = mem_b[all_masked_rows]
            # Aux losses use summary/gate before memory restore; keep them finite.
            a_summary = a_summary.clone()
            s_summary = s_summary.clone()
            a_summary[all_masked_rows] = 0.0
            s_summary[all_masked_rows] = 0.0
            # Avoid `gate[:bsz][mask] = ...` (advanced-index on a slice may not
            # write back into `gate`). Sanitize halves explicitly.
            a_gate = gate[:bsz].clone()
            s_gate = gate[bsz:].clone()
            a_gate[all_masked_rows] = 0.5
            s_gate[all_masked_rows] = 0.5
            return new_a, a_gate, a_summary, new_s, s_gate, s_summary

    a_gate = gate[:bsz]
    s_gate = gate[bsz:]
    return new_a, a_gate, a_summary, new_s, s_gate, s_summary


HybridMemoryState = tuple[Tensor, Tensor]
