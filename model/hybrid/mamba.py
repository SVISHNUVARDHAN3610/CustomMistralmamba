"""Mamba selective-SSM block and scan backends."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from model.core.config import HybridMambaMoEConfig, MambaCache
from model.hybrid.memory import _assert_right_padded_attention_mask


def _validate_hybrid_cache_states(
    config: HybridMambaMoEConfig,
    num_layers: int,
    batch_size: int,
    memory_states: list | None,
    mamba_caches: list | None,
    write_buffers: list | None,
    past_key_values: list | None,
    active_batch_mask: Tensor | None,
) -> None:
    if not config.debug_state_checks:
        return
    if memory_states is not None:
        assert len(memory_states) == num_layers
    if mamba_caches is not None:
        assert len(mamba_caches) == num_layers
    if write_buffers is not None:
        assert len(write_buffers) == num_layers
        for buf in write_buffers:
            if buf is not None:
                assert buf.filled >= 0
                if buf.attn_buf is not None:
                    assert buf.attn_buf.size(0) == batch_size
    if past_key_values is not None:
        assert len(past_key_values) == num_layers
    if active_batch_mask is not None:
        assert active_batch_mask.dtype == torch.bool
        assert active_batch_mask.size(0) == batch_size


_SELECTIVE_SCAN_FN: Callable[..., Tensor] | None = None
_SELECTIVE_SCAN_PROBE_DONE = False
_FUSED_SCAN_WARNED = False
_MAMBA_SCAN_STATS: dict[str, int] = {
    "fused_full_batch": 0,
    "fused_unpadded_batch": 0,
    "pytorch_fallback": 0,
}


def get_mamba_scan_stats() -> dict[str, int]:
    """Training-time counters for which Mamba selective-scan backend was used."""
    return dict(_MAMBA_SCAN_STATS)


def reset_mamba_scan_stats() -> None:
    """Reset Mamba scan backend counters (e.g. at the start of a training run)."""
    for key in _MAMBA_SCAN_STATS:
        _MAMBA_SCAN_STATS[key] = 0


def fused_mamba_scan_available() -> bool:
    """True when mamba-ssm fused selective_scan CUDA kernels can be imported."""
    return _load_selective_scan_fn() is not None


def log_mamba_backend(config: HybridMambaMoEConfig | None = None) -> str:
    """Log which Mamba selective-scan backend will be used. Returns summary string."""
    fused = fused_mamba_scan_available()
    if config is None:
        config = HybridMambaMoEConfig()
    if fused and config.use_fused_mamba_scan:
        msg = (
            "Mamba backend: fused CUDA selective_scan (mamba-ssm); "
            "padded batches use per-row unpadded fused scan"
        )
    elif config.use_parallel_scan:
        msg = "Mamba backend: Hillis-Steele parallel scan (explicit use_parallel_scan=True)"
    else:
        msg = (
            "Mamba backend: PyTorch fallback "
            f"(parallel L<={config.parallel_scan_fallback_max_len}, "
            f"blocked {config.blocked_scan_min_len}<L<={config.sequential_scan_min_len}, "
            f"sequential L>{config.sequential_scan_min_len})"
        )
        if torch.cuda.is_available() and not fused:
            warnings.warn(
                "CUDA is available but mamba-ssm is not installed; training will use "
                "slow PyTorch scan fallbacks. Install mamba-ssm for production runs.",
                stacklevel=2,
            )
    return msg


def probe_mamba_scan_timing(
    config: HybridMambaMoEConfig | None = None,
    batch_size: int = 2,
    seq_len: int = 512,
    device: torch.device | None = None,
) -> str:
    """One-step timing probe: fused vs PyTorch fallback selective scan."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config is None:
        config = HybridMambaMoEConfig(
            hidden_size=128,
            mamba_state_size=8,
            mamba_expand=2,
        )
    if not torch.cuda.is_available():
        return "mamba_scan_probe: skipped (CPU only)"

    import time

    block = MambaBlock(
        hidden_size=config.hidden_size,
        state_size=config.mamba_state_size,
        expand=config.mamba_expand,
        use_fused_scan=False,
    ).to(device)
    x = torch.randn(batch_size, seq_len, config.hidden_size, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        block(x)
    torch.cuda.synchronize()
    fallback_ms = (time.perf_counter() - t0) * 1000

    if fused_mamba_scan_available():
        block_fused = MambaBlock(
            hidden_size=config.hidden_size,
            state_size=config.mamba_state_size,
            expand=config.mamba_expand,
            use_fused_scan=True,
        ).to(device)
        block_fused.load_state_dict(block.state_dict())
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            block_fused(x)
        torch.cuda.synchronize()
        fused_ms = (time.perf_counter() - t0) * 1000
        return (
            f"mamba_scan_probe: fused={fused_ms:.1f}ms fallback={fallback_ms:.1f}ms "
            f"speedup={fallback_ms / max(fused_ms, 1e-6):.2f}x"
        )
    return f"mamba_scan_probe: fallback={fallback_ms:.1f}ms (mamba-ssm not installed)"


def _compute_batch_has_padding(attention_mask: Tensor | None, seq_len: int) -> bool:
    """Single sync point per forward for padding detection."""
    if attention_mask is None:
        return False
    if attention_mask.dim() != 2:
        return True
    if attention_mask.size(1) < seq_len:
        return True
    return not attention_mask[:, -seq_len:].all().item()


def _load_selective_scan_fn() -> Callable[..., Tensor] | None:
    """Lazy import of mamba-ssm selective_scan_fn (optional dependency)."""
    global _SELECTIVE_SCAN_FN, _SELECTIVE_SCAN_PROBE_DONE
    if _SELECTIVE_SCAN_PROBE_DONE:
        return _SELECTIVE_SCAN_FN
    _SELECTIVE_SCAN_PROBE_DONE = True
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        _SELECTIVE_SCAN_FN = selective_scan_fn
    except ImportError:
        _SELECTIVE_SCAN_FN = None
    return _SELECTIVE_SCAN_FN


def _attention_mask_has_padding(attention_mask: Tensor | None, seq_len: int) -> bool:
    if attention_mask is None:
        return False
    if attention_mask.dim() != 2:
        return True
    if attention_mask.size(1) < seq_len:
        return True
    return not attention_mask[:, -seq_len:].all().item()


class MambaBlock(nn.Module):
    """
    Selective SSM (Mamba / S6).

    Prefill/training uses fused CUDA selective_scan from `mamba-ssm` when
    available (`use_fused_scan=True`, CUDA, no padding). Otherwise falls back
    to length-aware PyTorch scans: parallel (short L), blocked (medium L),
    sequential+checkpoint (very long L). Optional Hillis-Steele parallel scan
    (`use_parallel_scan=True`) bypasses fused CUDA.
    Decode uses allocate_inference_cache() + step().
    """

    def __init__(
        self,
        hidden_size: int,
        state_size: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        use_parallel_scan: bool = False,
        use_fused_scan: bool = True,
        parallel_scan_fallback_max_len: int = 4096,
        blocked_scan_chunk_size: int = 256,
        blocked_scan_min_len: int = 4096,
        sequential_scan_min_len: int = 65536,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.d_inner = expand * hidden_size
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(hidden_size / 16)
        self.use_parallel_scan = use_parallel_scan
        self.use_fused_scan = use_fused_scan
        self.parallel_scan_fallback_max_len = parallel_scan_fallback_max_len
        self.blocked_scan_chunk_size = blocked_scan_chunk_size
        self.blocked_scan_min_len = blocked_scan_min_len
        self.sequential_scan_min_len = sequential_scan_min_len

        self.in_proj = nn.Linear(hidden_size, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=conv_kernel,
            groups=self.d_inner,
            padding=conv_kernel - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * state_size, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Softplus(dt_bias) ~ Uniform[dt_min, dt_max] at init (official Mamba).
        # Marked _no_reinit so HybridForCausalLM._init_weights does not zero it.
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # type: ignore[attr-defined]

        A = torch.arange(1, state_size + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]
        self.out_proj = nn.Linear(self.d_inner, hidden_size, bias=False)

    def allocate_inference_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> MambaCache:
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        ssm_dtype = torch.float32
        conv_state = torch.zeros(
            batch_size,
            self.d_inner,
            self.conv_kernel,
            device=device,
            dtype=conv_dtype,
        )
        ssm_state = torch.zeros(
            batch_size,
            self.d_inner,
            self.state_size,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def forward(
        self,
        x: Tensor,
        cache: MambaCache | None = None,
        use_cache: bool = False,
        attention_mask: Tensor | None = None,
        active_batch_mask: Tensor | None = None,
        debug_state_checks: bool = False,
        batch_has_padding: bool | None = None,
        mamba_internal_checkpoint: bool = True,
        layer_checkpointing_active: bool = False,
    ) -> tuple[Tensor, MambaCache | None, Tensor | None]:
        """
        x: [B, L, hidden_size]
        If use_cache and L==1 and cache is provided, runs a single decode step.
        Otherwise runs full-sequence prefill (parallel scan); when use_cache,
        returns updated (conv_state, ssm_state) for subsequent steps.
        """
        _, seq_len, _ = x.shape

        if use_cache and cache is not None and seq_len == 1:
            out, cache_out = self.step(
                x, cache[0], cache[1], active_batch_mask=active_batch_mask
            )
            return out, cache_out, cache[1]

        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        x_conv = x_in.transpose(1, 2)
        if use_cache:
            # Keep last conv_kernel *valid* tokens as the rolling conv buffer.
            if attention_mask is not None and attention_mask.dim() == 2:
                token_mask = attention_mask[:, -seq_len:]
                _assert_right_padded_attention_mask(token_mask, debug_state_checks)
                # Right-padding assumption: valid prefix length per row.
                # Vectorized gather (no per-row .item() host syncs): row b
                # fills columns [K - take_b, K) with x_in[b, vl_b - take_b :
                # vl_b]^T where take_b = min(K, vl_b); every other column is
                # zeroed via the validity mask (rows with vl == 0 stay all
                # zero) — identical to the previous per-row loop.
                k = self.conv_kernel
                valid_lens = token_mask.sum(dim=1)  # [B]
                take = torch.clamp(valid_lens, max=k)  # [B]
                # Source position for destination column j: vl - K + j;
                # clamped into range for lanes the mask discards.
                src_idx = (
                    valid_lens.unsqueeze(1) - k + torch.arange(k, device=x.device)
                ).clamp(min=0, max=seq_len - 1)  # [B, K]
                dst_valid = (
                    torch.arange(k, device=x.device).unsqueeze(0)
                    >= (k - take).unsqueeze(1)
                ).to(x_in.dtype)  # [B, K]; all-zero rows when vl == 0
                conv_state = (
                    x_in.gather(
                        dim=1,
                        index=src_idx.unsqueeze(-1).expand(-1, -1, x_in.size(2)),
                    ).transpose(1, 2)
                    * dst_valid.unsqueeze(1)
                ).contiguous()  # [B, d_inner, K]; step() shifts it in-place
            else:
                pad = max(self.conv_kernel - seq_len, 0)
                conv_state = F.pad(x_conv, (pad, 0))[
                    :, :, -self.conv_kernel :
                ].contiguous()
        else:
            conv_state = None

        x_conv = self.conv1d(x_conv)[..., :seq_len]
        x_conv = F.silu(x_conv).transpose(1, 2)

        x_dbl = self.x_proj(x_conv)
        dt, B_param, C_param = torch.split(
            x_dbl, [self.dt_rank, self.state_size, self.state_size], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log.float())
        y, ssm_state = self._selective_scan(
            x_conv,
            dt,
            A,
            B_param,
            C_param,
            self.D,
            return_final_state=True,
            use_parallel_scan=self.use_parallel_scan,
            use_fused_scan=self.use_fused_scan,
            training=self.training,
            attention_mask=attention_mask,
            batch_has_padding=batch_has_padding,
            parallel_scan_fallback_max_len=self.parallel_scan_fallback_max_len,
            blocked_scan_chunk_size=self.blocked_scan_chunk_size,
            blocked_scan_min_len=self.blocked_scan_min_len,
            sequential_scan_min_len=self.sequential_scan_min_len,
            mamba_internal_checkpoint=mamba_internal_checkpoint,
            layer_checkpointing_active=layer_checkpointing_active,
        )
        y = y * F.silu(z)
        out = self.out_proj(y)

        new_cache: MambaCache | None = None
        if use_cache:
            assert conv_state is not None and ssm_state is not None
            new_cache = (conv_state, ssm_state)
        return out, new_cache, ssm_state

    def step(
        self,
        x: Tensor,
        conv_state: Tensor,
        ssm_state: Tensor,
        active_batch_mask: Tensor | None = None,
    ) -> tuple[Tensor, MambaCache]:
        """Single-token decode. x: [B, 1, hidden_size]."""
        assert x.size(1) == 1
        dtype = x.dtype

        prev_conv = conv_state.clone()
        prev_ssm = ssm_state.clone()

        xz = self.in_proj(x.squeeze(1))
        x_in, z = xz.chunk(2, dim=-1)

        # Shift conv buffer in-place (slice copy, no full tensor roll).
        conv_state[:, :, :-1].copy_(conv_state[:, :, 1:])
        conv_state[:, :, -1] = x_in
        x_conv = torch.sum(conv_state * self.conv1d.weight.squeeze(1), dim=-1)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x_conv = F.silu(x_conv).to(dtype=dtype)

        x_dbl = self.x_proj(x_conv)
        dt, B_param, C_param = torch.split(
            x_dbl, [self.dt_rank, self.state_size, self.state_size], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log.float())
        dt_f = dt.float()
        dA = torch.exp(dt_f.unsqueeze(-1) * A)  # [B, d_inner, n]
        dB_u = (
            dt_f.unsqueeze(-1)
            * B_param.float().unsqueeze(1)
            * x_conv.float().unsqueeze(-1)
        )
        ssm_state = ssm_state.float() * dA + dB_u
        y = (ssm_state * C_param.float().unsqueeze(1)).sum(dim=-1)
        y = (y + x_conv.float() * self.D.float()).to(dtype)
        y = y * F.silu(z)
        out = self.out_proj(y).unsqueeze(1)

        if active_batch_mask is not None and (~active_batch_mask).any():
            inactive = ~active_batch_mask
            conv_state = conv_state.clone()
            ssm_state = ssm_state.clone()
            conv_state[inactive] = prev_conv[inactive]
            ssm_state[inactive] = prev_ssm[inactive]
            out = out.clone()
            out[inactive] = 0

        return out, (conv_state, ssm_state)

    @staticmethod
    def _parallel_associative_scan(delta_a: Tensor, delta_b_u: Tensor) -> Tensor:
        """
        Hillis-Steele inclusive scan: h_t = delta_a_t * h_{t-1} + delta_b_u_t.
        O(L log L) work and training memory — use only when use_parallel_scan.
        """
        seq_len = delta_a.size(1)
        a = delta_a
        b = delta_b_u
        n = 1
        while n < seq_len:
            a_prev = a[:, :-n]
            b_prev = b[:, :-n]
            a_curr = a[:, n:]
            b_curr = b[:, n:]
            a = torch.cat([a[:, :n], a_curr * a_prev], dim=1)
            b = torch.cat([b[:, :n], a_curr * b_prev + b_curr], dim=1)
            n *= 2
        return b

    @staticmethod
    def _sequential_associative_scan(delta_a: Tensor, delta_b_u: Tensor) -> Tensor:
        """O(L) work sequential scan; pair with checkpoint during training."""
        _, seq_len, _, _ = delta_a.shape
        state = torch.zeros_like(delta_b_u[:, 0])
        outputs = []
        for t in range(seq_len):
            state = delta_a[:, t] * state + delta_b_u[:, t]
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    @staticmethod
    def _blocked_associative_scan(
        delta_a: Tensor, delta_b_u: Tensor, block_size: int
    ) -> Tensor:
        """Vectorized scan within blocks; carry state between blocks."""
        _, seq_len, _, _ = delta_a.shape
        state = torch.zeros_like(delta_b_u[:, 0])
        outputs: list[Tensor] = []
        for start in range(0, seq_len, block_size):
            end = min(start + block_size, seq_len)
            block_a = delta_a[:, start:end]
            block_b = delta_b_u[:, start:end]
            if start > 0:
                block_b = block_b.clone()
                block_b[:, 0] = block_a[:, 0] * state + block_b[:, 0]
            block_out = MambaBlock._parallel_associative_scan(block_a, block_b)
            outputs.append(block_out)
            state = block_out[:, -1]
        return torch.cat(outputs, dim=1)

    @classmethod
    def _run_associative_scan(
        cls,
        delta_a: Tensor,
        delta_b_u: Tensor,
        *,
        use_parallel_scan: bool,
        training: bool,
        seq_len: int,
        parallel_scan_fallback_max_len: int,
        blocked_scan_chunk_size: int,
        blocked_scan_min_len: int,
        sequential_scan_min_len: int,
        mamba_internal_checkpoint: bool = True,
        layer_checkpointing_active: bool = False,
    ) -> Tensor:
        use_scan_checkpoint = (
            training and mamba_internal_checkpoint and not layer_checkpointing_active
        )
        if use_parallel_scan or seq_len <= parallel_scan_fallback_max_len:
            return cls._parallel_associative_scan(delta_a, delta_b_u)
        if seq_len <= sequential_scan_min_len:
            scan_fn = lambda a, b: cls._blocked_associative_scan(
                a, b, blocked_scan_chunk_size
            )
            if use_scan_checkpoint:
                return checkpoint(scan_fn, delta_a, delta_b_u, use_reentrant=False)
            return scan_fn(delta_a, delta_b_u)
        if use_scan_checkpoint:
            return checkpoint(
                cls._sequential_associative_scan,
                delta_a,
                delta_b_u,
                use_reentrant=False,
            )
        return cls._sequential_associative_scan(delta_a, delta_b_u)

    @staticmethod
    def _fused_selective_scan(
        u: Tensor,
        dt: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
        return_final_state: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """
        mamba-ssm fused CUDA selective scan.
        u, dt: [B, L, d_inner]; A: [d_inner, n]; B, C: [B, L, n]; D: [d_inner]
        """
        selective_scan_fn = _load_selective_scan_fn()
        if selective_scan_fn is None:
            raise RuntimeError("mamba-ssm selective_scan_fn is not available.")

        input_dtype = u.dtype
        u_t = u.transpose(1, 2).contiguous()
        dt_t = dt.transpose(1, 2).contiguous()
        b_t = B.transpose(1, 2).contiguous()
        c_t = C.transpose(1, 2).contiguous()

        result = selective_scan_fn(
            u_t.float(),
            dt_t.float(),
            A.float(),
            b_t.float(),
            c_t.float(),
            D.float(),
            delta_bias=None,
            delta_softplus=False,
            return_last_state=return_final_state,
        )
        if return_final_state:
            y_t, final_state = result
            return y_t.transpose(1, 2).to(input_dtype), final_state
        y_t = result
        return y_t.transpose(1, 2).to(input_dtype), None

    @classmethod
    def _fused_selective_scan_unpadded(
        cls,
        u: Tensor,
        dt: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
        attention_mask: Tensor,
        return_final_state: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Run fused selective_scan on each row's valid prefix (right-padded mask).
        Outputs are restored to [B, L, d_inner] with padding positions zeroed.
        Final SSM state per row comes from the last valid token (not pad columns).

        Uses cat/pad + stack (not in-place slice assignment) so autograd keeps a
        path from the fused scan outputs back into the Mamba branch. In-place
        writes into a zero buffer break that path and can NaN training within
        a few optimizer steps when dual-memory aux losses are active.
        """
        batch_size, seq_len, d_inner = u.shape
        state_size = B.size(-1)
        device = u.device
        dtype = u.dtype

        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must be 2D [B, L].")
        if attention_mask.size(1) < seq_len:
            raise ValueError(
                f"attention_mask length {attention_mask.size(1)} < seq_len {seq_len}."
            )

        token_mask = attention_mask[:, -seq_len:].bool()
        valid_lens = token_mask.sum(dim=1)

        y_rows: list[Tensor] = []
        state_rows: list[Tensor] = []
        for b in range(batch_size):
            vl = int(valid_lens[b].item())
            if vl <= 0:
                y_rows.append(torch.zeros(seq_len, d_inner, device=device, dtype=dtype))
                if return_final_state:
                    state_rows.append(
                        torch.zeros(d_inner, state_size, device=device, dtype=dtype)
                    )
                continue
            y_b, st_b = cls._fused_selective_scan(
                u[b : b + 1, :vl, :],
                dt[b : b + 1, :vl, :],
                A,
                B[b : b + 1, :vl, :],
                C[b : b + 1, :vl, :],
                D,
                return_final_state=return_final_state,
            )
            # Pad seq dim on the right; keeps gradient into y_b[:, :vl].
            if vl < seq_len:
                y_padded = F.pad(y_b, (0, 0, 0, seq_len - vl))
            else:
                y_padded = y_b
            y_rows.append(y_padded[0])
            if return_final_state:
                if st_b is not None:
                    state_rows.append(st_b[0])
                else:
                    state_rows.append(
                        torch.zeros(d_inner, state_size, device=device, dtype=dtype)
                    )

        y = torch.stack(y_rows, dim=0)
        ssm_state = torch.stack(state_rows, dim=0) if return_final_state else None
        return y, ssm_state

    @classmethod
    def _selective_scan(
        cls,
        u: Tensor,
        dt: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
        return_final_state: bool = False,
        use_parallel_scan: bool = False,
        use_fused_scan: bool = True,
        training: bool = False,
        attention_mask: Tensor | None = None,
        batch_has_padding: bool | None = None,
        parallel_scan_fallback_max_len: int = 4096,
        blocked_scan_chunk_size: int = 256,
        blocked_scan_min_len: int = 4096,
        sequential_scan_min_len: int = 65536,
        mamba_internal_checkpoint: bool = True,
        layer_checkpointing_active: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """
        u, dt: [B, L, d_inner]; A: [d_inner, n]; B, C: [B, L, n]; D: [d_inner]
        On pad positions (attention_mask==0), apply identity state transition
        so SSM state does not decay through padding.
        """
        global _FUSED_SCAN_WARNED
        seq_len = u.size(1)
        has_padding = (
            batch_has_padding
            if batch_has_padding is not None
            else _attention_mask_has_padding(attention_mask, seq_len)
        )
        fused_fn_available = _load_selective_scan_fn() is not None
        can_use_fused = (
            use_fused_scan
            and not use_parallel_scan
            and u.is_cuda
            and fused_fn_available
        )
        if can_use_fused and not has_padding:
            try:
                _MAMBA_SCAN_STATS["fused_full_batch"] += 1
                return cls._fused_selective_scan(
                    u,
                    dt,
                    A,
                    B,
                    C,
                    D,
                    return_final_state=return_final_state,
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                if not _FUSED_SCAN_WARNED:
                    warnings.warn(
                        f"mamba-ssm fused selective_scan failed ({type(exc).__name__}: "
                        f"{exc}); falling back to PyTorch scan.",
                        stacklevel=2,
                    )
                    _FUSED_SCAN_WARNED = True
        if can_use_fused and has_padding and attention_mask is not None:
            try:
                _MAMBA_SCAN_STATS["fused_unpadded_batch"] += 1
                return cls._fused_selective_scan_unpadded(
                    u,
                    dt,
                    A,
                    B,
                    C,
                    D,
                    attention_mask,
                    return_final_state=return_final_state,
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                if not _FUSED_SCAN_WARNED:
                    warnings.warn(
                        f"mamba-ssm unpadded fused selective_scan failed "
                        f"({type(exc).__name__}: {exc}); falling back to PyTorch scan.",
                        stacklevel=2,
                    )
                    _FUSED_SCAN_WARNED = True

        _MAMBA_SCAN_STATS["pytorch_fallback"] += 1
        input_dtype = u.dtype
        u_f = u.float()
        dt_f = dt.float()
        B_f = B.float()
        C_f = C.float()

        delta_a = torch.exp(dt_f.unsqueeze(-1) * A)  # [B, L, d_inner, n]
        delta_b_u = dt_f.unsqueeze(-1) * B_f.unsqueeze(2) * u_f.unsqueeze(-1)

        token_mask: Tensor | None = None
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("MambaBlock expects 2D attention_mask [B, L].")
            if attention_mask.size(1) < seq_len:
                raise ValueError(
                    f"attention_mask length {attention_mask.size(1)} < seq_len {seq_len}."
                )
            token_mask = attention_mask[:, -seq_len:].to(dtype=delta_a.dtype)
            m = token_mask.unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
            # Pad steps: h_t = 1 * h_{t-1} + 0
            delta_a = delta_a * m + (1.0 - m)
            delta_b_u = delta_b_u * m

        states = cls._run_associative_scan(
            delta_a,
            delta_b_u,
            use_parallel_scan=use_parallel_scan,
            training=training,
            seq_len=seq_len,
            parallel_scan_fallback_max_len=parallel_scan_fallback_max_len,
            blocked_scan_chunk_size=blocked_scan_chunk_size,
            blocked_scan_min_len=blocked_scan_min_len,
            sequential_scan_min_len=sequential_scan_min_len,
            mamba_internal_checkpoint=mamba_internal_checkpoint,
            layer_checkpointing_active=layer_checkpointing_active,
        )

        y = (states * C_f.unsqueeze(2)).sum(dim=-1)
        y = y + u_f * D.float()
        if token_mask is not None:
            y = y * token_mask.unsqueeze(-1)
        final_state = states[:, -1].contiguous() if return_final_state else None
        return y.to(input_dtype), final_state
