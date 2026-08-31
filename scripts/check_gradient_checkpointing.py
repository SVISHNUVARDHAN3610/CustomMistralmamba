"""Standalone gradient-checkpointing and resume-contract smoke check.

Runs the repository's representative ~5M-parameter hybrid architecture with
checkpointing off and on. CUDA runs assert an actual reduction in peak allocated
activation memory; CPU runs report that memory assertion as SKIP while still
checking gradients, finiteness, cache lifecycle, and checkpoint/resume state.
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.core.builders import count_trainable_params
from model.hybrid.model import HybridForCausalLM
from scripts.toy_train import build_toy_config


@dataclass
class CaseResult:
    enabled: bool
    peak_bytes: int | None
    losses: list[float]
    adjacent_grad_l1: float
    embedding_output_requires_grad: bool
    finite: bool
    model: HybridForCausalLM
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler


def _configure_model(enabled: bool, device: torch.device) -> HybridForCausalLM:
    torch.manual_seed(1234)
    cfg = build_toy_config()
    # Exercise whole-layer checkpointing without introducing a second chunked
    # BPTT axis into this focused comparison.
    cfg.memory_chunk_size = None
    cfg.use_fused_mamba_scan = False
    model = HybridForCausalLM(cfg).to(device)
    model.get_input_embeddings().weight.requires_grad_(False)
    if enabled:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.gradient_checkpointing_disable()
    model.train()
    return model


def _dummy_batch(
    cfg,
    device: torch.device,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.randint(
        0,
        cfg.vocab_size,
        (batch_size, seq_len),
        device=device,
    )
    labels = ids.roll(shifts=-1, dims=1)
    labels[:, -1] = cfg.label_ignore_index
    return ids, labels


def _one_step(
    model: HybridForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ids: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    max_steps: int,
) -> tuple[float, float, bool]:
    optimizer.zero_grad(set_to_none=True)
    out = model(
        input_ids=ids,
        labels=labels,
        training_step=step,
        max_training_steps=max_steps,
    )
    assert out.loss is not None
    out.loss.backward()

    adjacent = model.model.layers[0].attention_block.q_proj.weight.grad
    adjacent_l1 = 0.0 if adjacent is None else float(adjacent.abs().sum().item())
    finite = bool(torch.isfinite(out.loss).item())
    for parameter in model.parameters():
        if parameter.grad is not None:
            finite = finite and bool(torch.isfinite(parameter.grad).all().item())
    optimizer.step()
    scheduler.step()
    return float(out.loss.item()), adjacent_l1, finite


def _run_case(
    enabled: bool,
    device: torch.device,
    *,
    steps: int,
    batch_size: int,
    seq_len: int,
) -> CaseResult:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = _configure_model(enabled, device)
    parameter_count = count_trainable_params(model)
    if not 4_000_000 <= parameter_count <= 6_000_000:
        raise AssertionError(
            f"Expected representative ~5M model, got {parameter_count:,} parameters."
        )
    embedding_output_requires_grad = bool(
        model.get_input_embeddings()(
            torch.zeros((1, 1), dtype=torch.long, device=device)
        ).requires_grad
    )
    if enabled and not embedding_output_requires_grad:
        raise AssertionError("Frozen embedding output did not enter autograd.")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=3e-4,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    batches = [
        _dummy_batch(model.config, device, batch_size, seq_len)
        for _ in range(steps + 1)
    ]

    # Warm up lazy kernels and allocate optimizer state outside the measured
    # region. Both modes start their measured steps from the same allocator
    # baseline and model seed.
    _one_step(model, optimizer, scheduler, *batches[0], 0, steps + 1)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = 0

    losses: list[float] = []
    adjacent_grad_l1 = 0.0
    finite = True
    for step, batch in enumerate(batches[1:], start=1):
        loss, adjacent_grad_l1, step_finite = _one_step(
            model,
            optimizer,
            scheduler,
            *batch,
            step,
            steps + 1,
        )
        losses.append(loss)
        finite = finite and step_finite

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device) - baseline
    else:
        peak_bytes = None

    if adjacent_grad_l1 <= 0.0:
        raise AssertionError(
            "Layer adjacent to frozen embeddings received zero/no gradients."
        )
    if not finite:
        raise AssertionError("Non-finite loss or gradient detected.")

    # Eval/generation may cache; returning to train must disable it again.
    model.eval()
    if not model.config.use_cache:
        raise AssertionError("use_cache was not re-enabled for eval/inference.")
    model.train()
    if enabled and model.config.use_cache:
        raise AssertionError("use_cache remained enabled for checkpointed training.")

    return CaseResult(
        enabled=enabled,
        peak_bytes=peak_bytes,
        losses=losses,
        adjacent_grad_l1=adjacent_grad_l1,
        embedding_output_requires_grad=embedding_output_requires_grad,
        finite=finite,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def _check_resume_round_trip(result: CaseResult, device: torch.device) -> None:
    # Importing the production trainer here keeps the model-only comparison
    # usable until this point while ensuring the real save/load code is tested.
    import train

    logger = logging.getLogger("check_gradient_checkpointing")
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    dl_generator = torch.Generator().manual_seed(99)

    with tempfile.TemporaryDirectory(prefix="gc_resume_check_") as temp_dir:
        checkpoint_dir = Path(temp_dir)
        train.save_checkpoint(
            model=result.model,
            optimizers=[result.optimizer],
            schedulers=[result.scheduler],
            global_step=len(result.losses) + 1,
            current_shard_idx=7,
            checkpoint_dir=checkpoint_dir,
            logger=logger,
            use_muon=False,
            extra_payload={"dl_generator_state": dl_generator.get_state()},
        )
        payload = torch.load(
            checkpoint_dir / train.CHECKPOINT_FILENAME,
            map_location="cpu",
            weights_only=True,
        )
        runtime = payload["training_runtime"]
        assert runtime["gradient_checkpointing"] is True
        assert runtime["gradient_checkpointing_use_reentrant"] is False
        assert runtime["use_cache"] is False
        assert runtime["distributed_strategy"] == "single_process"
        assert runtime["ddp_find_unused_parameters"] is None
        assert runtime["ddp_static_graph"] is None

        resumed = _configure_model(True, device)
        resumed_optimizer = torch.optim.AdamW(
            [p for p in resumed.parameters() if p.requires_grad], lr=3e-4
        )
        resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
            resumed_optimizer, lambda _: 1.0
        )
        resumed_dl_generator = torch.Generator().manual_seed(0)
        step, shard = train.load_checkpoint(
            model=resumed,
            optimizers=[resumed_optimizer],
            schedulers=[resumed_scheduler],
            checkpoint_dir=checkpoint_dir,
            device=device,
            logger=logger,
            use_muon=False,
            dl_generator=resumed_dl_generator,
        )
        assert (step, shard) == (len(result.losses) + 1, 7)
        assert len(resumed_optimizer.state) == len(result.optimizer.state)
        assert resumed_scheduler.last_epoch == result.scheduler.last_epoch
        assert torch.equal(
            resumed_dl_generator.get_state(), dl_generator.get_state()
        )
        assert torch.equal(torch.get_rng_state(), payload["rng_state"]["torch"])

        stale_cache = copy.deepcopy(payload)
        stale_cache["training_runtime"]["use_cache"] = True
        resumed.train()
        train.validate_resume_runtime_contract(
            stale_cache,
            resumed,
            logger,
            distributed_strategy="single_process",
        )
        assert resumed.config.use_cache is False

        bad_reentrant = copy.deepcopy(payload)
        bad_reentrant["training_runtime"][
            "gradient_checkpointing_use_reentrant"
        ] = True
        try:
            train.validate_resume_runtime_contract(
                bad_reentrant,
                resumed,
                logger,
                distributed_strategy="single_process",
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("Reentrant checkpointing mismatch was not rejected.")

        bad_ddp = copy.deepcopy(payload)
        bad_ddp["training_runtime"].update(
            {
                "distributed_strategy": "ddp",
                "ddp_find_unused_parameters": True,
                "ddp_static_graph": True,
            }
        )
        try:
            train.validate_resume_runtime_contract(
                bad_ddp,
                resumed,
                logger,
                distributed_strategy="single_process",
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("DDP reducer-contract mismatch was not rejected.")


def _format_mib(value: int | None) -> str:
    return "N/A (CPU)" if value is None else f"{value / 2**20:.2f} MiB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for the lightweight comparison.",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be >= 1")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    batch_size = args.batch_size or (2 if device.type == "cuda" else 1)
    seq_len = args.seq_len or (256 if device.type == "cuda" else 32)

    off = _run_case(
        False,
        device,
        steps=args.steps,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    off_peak = off.peak_bytes
    del off.model, off.optimizer, off.scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    on = _run_case(
        True,
        device,
        steps=args.steps,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    if off_peak is not None and on.peak_bytes is not None:
        if on.peak_bytes >= off_peak:
            raise AssertionError(
                "Gradient checkpointing did not reduce peak allocated GPU memory: "
                f"off={_format_mib(off_peak)}, on={_format_mib(on.peak_bytes)}."
            )
        memory_status = "PASS"
    else:
        memory_status = "SKIP (CUDA unavailable)"

    _check_resume_round_trip(on, device)

    print("gradient checkpointing check")
    print(f"device={device} batch={batch_size} seq={seq_len} steps={args.steps}")
    print(f"peak GPU memory off: {_format_mib(off_peak)}")
    print(f"peak GPU memory on : {_format_mib(on.peak_bytes)}")
    print(f"memory reduction   : {memory_status}")
    print(
        "frozen-input gradient: PASS "
        f"(adjacent grad L1={on.adjacent_grad_l1:.6g})"
    )
    print("finite loss/gradients: PASS")
    print("use_cache train/eval lifecycle: PASS")
    print("stale saved use_cache=True repair: PASS")
    print("use_reentrant=False save/resume contract: PASS")
    print("optimizer/scheduler/RNG resume: PASS")
    print("DDP find_unused/static_graph mismatch rejection: PASS")


if __name__ == "__main__":
    main()
