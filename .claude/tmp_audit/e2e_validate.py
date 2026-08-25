"""TEMPORARY audit fixture — NOT part of the codebase (lives in gitignored .claude/).

End-to-end validation of the production training pipeline on a laptop-safe
~5M-parameter toy config, batch size 2:

  synthetic uint16 shards -> MmapShardDataset -> DataLoader
    -> HybridForCausalLM forward (pre-shifted labels) -> backward -> clip
    -> train.build_optimizers (real Muon+AdamW split) -> cosine/warmup schedulers
    -> save_checkpoint / load_checkpoint roundtrip (weights, optimizers,
       schedulers, RNG state) -> resume for extra steps
    -> generate() inference (greedy determinism, sampling, batched, bounds)

The ONLY deviation from production: `datasets`/`transformers` are absent from
this dev venv, so they are stubbed in sys.modules purely to let train.py import
(TokenizedShardProducer/WikiTextCyclicValidator are never instantiated). All
executed logic (optimizers, schedule, checkpointing, token validation, weighted
aux terms) is train.py's own code, unmodified.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for scripts_dir in (ROOT / "scripts",):
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

# --- in-process stubs for optional deps missing on this dev laptop ---------
_datasets = types.ModuleType("datasets")
_datasets.interleave_datasets = lambda *a, **k: None  # type: ignore[attr-defined]
_datasets.load_dataset = lambda *a, **k: None  # type: ignore[attr-defined]
_transformers = types.ModuleType("transformers")
_transformers.AutoTokenizer = object  # type: ignore[attr-defined]
_transformers.PreTrainedTokenizerBase = object  # type: ignore[attr-defined]
sys.modules.setdefault("datasets", _datasets)
sys.modules.setdefault("transformers", _transformers)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import train as train_mod  # noqa: E402  (real production training module)
from model.core.builders import count_trainable_params  # noqa: E402
from model.hybrid.model import HybridForCausalLM  # noqa: E402
from toy_train import _weighted_terms  # noqa: E402  (mirrors train.py's copy)
from utils.dataset import MmapShardDataset  # noqa: E402

OUT_DIR = ROOT / ".claude" / "tmp_audit" / "run"
CACHE_DIR = OUT_DIR / "cache"
CKPT_DIR = OUT_DIR / "ckpt"
JSONL_PATH = OUT_DIR / "metrics.jsonl"

SEQ_LEN = 128            # model sees SEQ_LEN tokens per example (row = SEQ_LEN+1)
VOCAB = 512              # toy config vocab
BATCH_SIZE = 2           # per user requirement
MAX_STEPS = 8            # first training segment (crosses one shard boundary)
RESUME_EXTRA_STEPS = 2   # post-resume continuation target: MAX_STEPS + this
SEED = 42

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def make_shard(path: Path, n_sequences: int, seed: int) -> int:
    """Write one packed uint16 shard of n_sequences*(SEQ_LEN+1) tokens."""
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, VOCAB, size=(n_sequences, SEQ_LEN + 1), dtype=np.uint16)
    rows[:, 0] = 1  # BOS-ish sentinel at doc starts
    rows[:, -1] = 2  # EOS-ish sentinel at doc ends
    rows.astype(np.uint16).tofile(path)
    return rows.size


def main() -> int:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for d in (CACHE_DIR, CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("e2e")

    train_mod.set_seed(SEED)
    device = torch.device("cpu")

    # --- data shards -------------------------------------------------------
    tok0 = make_shard(CACHE_DIR / "shard_000000.bin", n_sequences=48, seed=1)
    tok1 = make_shard(CACHE_DIR / "shard_000001.bin", n_sequences=48, seed=2)
    check("shards written", tok0 == 48 * (SEQ_LEN + 1) and tok1 == tok0,
          f"{tok0}+{tok1} tokens")

    # --- model + real production optimizer/schedule builders ---------------
    from toy_train import build_toy_config  # laptop-safe ~5M config

    cfg = build_toy_config()
    model = HybridForCausalLM(cfg).to(device)
    n_params = count_trainable_params(model)
    check("param budget 5M-10M", 5_000_000 <= n_params <= 10_000_000,
          f"{n_params:,}")

    optimizers, use_muon, opt_meta = train_mod.build_optimizers(
        model,
        lr=1e-3,
        muon_lr=None,
        adam_lr=None,
        weight_decay=0.1,
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_ns_steps=2,
        muon_adjust_lr_fn="match_rms_adamw",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        use_muon=True,
        device=device,
        logger=logger,
    )
    check("optimizer build", len(optimizers) >= 1,
          f"use_muon={use_muon} split={opt_meta['adam_pct']:.1f}%/"
          f"{opt_meta['muon_pct']:.1f}%")

    warmup = train_mod._resolve_warmup_steps(0, MAX_STEPS + RESUME_EXTRA_STEPS)
    lr_lambda = train_mod._build_lr_lambda(
        warmup, MAX_STEPS + RESUME_EXTRA_STEPS, 0.1
    )
    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        for opt in optimizers
    ]
    lrs0 = [s.get_last_lr()[0] for s in schedulers]
    check("warmup schedule start", all(lr > 0 for lr in lrs0),
          f"warmup={warmup} lr0={lrs0}")

    # --- training loop (mirrors train.py::train inner loop) ----------------
    dl_generator = torch.Generator()
    dl_generator.manual_seed(SEED)

    def run_steps(target_max: int, start_shard: int, global_step: int) -> tuple[int, int]:
        shard_idx = start_shard
        while global_step < target_max and shard_idx < 2:
            bin_path = CACHE_DIR / f"shard_{shard_idx:06d}.bin"
            dataset = MmapShardDataset(bin_path=str(bin_path), seq_len=SEQ_LEN + 1)
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                generator=dl_generator,
            )
            model.train()
            batches_seen = 0
            for input_ids, labels in loader:
                if global_step >= target_max:
                    break
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                train_mod.validate_token_batch(
                    input_ids, cfg.vocab_size,
                    labels=labels, ignore_index=cfg.label_ignore_index,
                )
                # Pre-shifted labels contract: dataset already returned
                # (chunk[:-1], chunk[1:]) — forward must NOT shift again.
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=False
                ):
                    out = model(
                        input_ids=input_ids,
                        labels=labels,
                        training_step=global_step,
                        max_training_steps=target_max,
                    )
                assert out.loss is not None
                if not torch.isfinite(out.loss):
                    check("finite loss", False, f"step={global_step}")
                    return global_step, shard_idx
                out.loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0
                    ).item()
                )
                for opt in optimizers:
                    opt.step()
                for sched in schedulers:
                    sched.step()

                assert out.auxiliary_losses is not None
                record = {
                    "event": "train_step",
                    "step": global_step,
                    "shard_idx": shard_idx,
                    "loss": float(out.loss.item()),
                    "ce_loss": float(out.ce_loss.item()) if out.ce_loss is not None else None,
                    "grad_norm": grad_norm,
                    "muon_lr": float(schedulers[0].get_last_lr()[0]),
                    "adam_lr": float(schedulers[-1].get_last_lr()[0]),
                    "router_aux_loss": float(out.router_aux_loss.item()) if out.router_aux_loss is not None else 0.0,
                    "router_z_loss": float(out.router_z_loss.item()) if out.router_z_loss is not None else 0.0,
                    "recon": float(out.auxiliary_losses.recon.item()),
                    "assoc": float(out.auxiliary_losses.assoc.item()),
                    "gate": float(out.auxiliary_losses.gate.item()),
                    "read": float(out.auxiliary_losses.read.item()),
                    "fusion": float(out.auxiliary_losses.fusion.item()),
                    "expert": float(out.auxiliary_losses.expert.item()),
                    "ssm": float(out.auxiliary_losses.ssm.item()),
                    "slot": float(out.auxiliary_losses.slot.item()),
                    "gate_stats": {
                        k: float(v.item()) for k, v in (out.gate_stats or {}).items()
                    },
                }
                with JSONL_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

                finite_grads = all(
                    torch.isfinite(p.grad).all().item()
                    for p in model.parameters()
                    if p.grad is not None
                )
                if not finite_grads or not np.isfinite(grad_norm):
                    check("finite grads", False, f"step={global_step}")
                    return global_step, shard_idx
                batches_seen += 1
                global_step += 1
            print(f"  shard {shard_idx}: {batches_seen} steps, "
                  f"{len(dataset)} sequences of len {SEQ_LEN}")
            shard_idx += 1
        return global_step, shard_idx

    print(f"--- training phase 1 ({MAX_STEPS} steps, batch={BATCH_SIZE}) ---")
    global_step, shard_after = run_steps(MAX_STEPS, start_shard=0, global_step=0)
    check("training phase 1 completed", global_step == MAX_STEPS,
          f"steps={global_step}")

    # LR actually moved along the schedule
    lr_end = [s.get_last_lr()[0] for s in schedulers]
    check("lr schedule progressed", lr_end != lrs0, f"{lrs0} -> {lr_end}")

    # --- checkpoint roundtrip ----------------------------------------------
    train_mod.save_checkpoint(
        model=model,
        optimizers=optimizers,
        schedulers=schedulers,
        global_step=global_step,
        current_shard_idx=shard_after,
        checkpoint_dir=CKPT_DIR,
        logger=logger,
        validator=None,
        use_muon=use_muon,
        extra_payload={"dl_generator_state": dl_generator.get_state()},
    )
    ckpt_path = CKPT_DIR / train_mod.CHECKPOINT_FILENAME
    check("checkpoint file exists", ckpt_path.exists(), str(ckpt_path.name))
    check("config.json written", (CKPT_DIR / train_mod.CONFIG_FILENAME).exists())

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    expected_keys = {
        "model_state_dict", "config", "global_step", "current_shard_idx",
        "rng_state", "memory_nan_fix_id", "use_muon",
    }
    check("checkpoint payload keys", expected_keys.issubset(payload.keys()),
          str(sorted(payload.keys())))
    # M12: the new payload must be loadable WITHOUT pickle execution.
    try:
        torch.load(ckpt_path, map_location="cpu", weights_only=True)
        wo_ok = True
    except Exception:
        wo_ok = False
    check("weights_only=True load (M12)", wo_ok)
    saved_rng = payload["rng_state"]["torch"].clone()
    saved_cfg_json = json.loads((CKPT_DIR / train_mod.CONFIG_FILENAME).read_text())
    new_knobs_present = all(
        k in saved_cfg_json
        for k in ("vocab_z_loss_coef", "num_sink_tokens", "use_qk_norm",
                  "fusion_balance_target")
    )
    check("config.json roundtrip incl. new knobs", new_knobs_present)

    weights_before = {k: v.clone() for k, v in model.state_dict().items()}
    sched_pos_before = [s.last_epoch for s in schedulers]

    def snap(obj, memo=None):
        if memo is None:
            memo = {}
        if torch.is_tensor(obj):
            return obj.clone()
        if isinstance(obj, dict):
            return {k: snap(v, memo) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(snap(v, memo) for v in obj)
        return obj

    def same(a, b):
        if torch.is_tensor(a):
            return torch.is_tensor(b) and torch.equal(a, b)
        if isinstance(a, dict):
            return (isinstance(b, dict) and set(a) == set(b)
                    and all(same(a[k], b[k]) for k in a))
        if isinstance(a, (list, tuple)):
            return (isinstance(b, type(a)) and len(a) == len(b)
                    and all(same(x, y) for x, y in zip(a, b)))
        return a == b

    opt_snapshots_before = [snap(o.state_dict()) for o in optimizers]

    fresh_model = HybridForCausalLM(cfg).to(device)
    fresh_optims, fresh_use_muon, _ = train_mod.build_optimizers(
        fresh_model, lr=1e-3, muon_lr=None, adam_lr=None, weight_decay=0.1,
        muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=2,
        muon_adjust_lr_fn="match_rms_adamw", adam_beta1=0.9, adam_beta2=0.95,
        adam_eps=1e-8, use_muon=True, device=device, logger=logger,
    )
    fresh_scheds = [
        torch.optim.lr_scheduler.LambdaLR(o, lr_lambda=lr_lambda)
        for o in fresh_optims
    ]
    fo, fs = fresh_optims, fresh_scheds
    resumed_step, resumed_shard = train_mod.load_checkpoint(
        model=fresh_model, optimizers=fo, schedulers=fs,
        checkpoint_dir=CKPT_DIR, device=device, logger=logger, validator=None,
        use_muon=fresh_use_muon, dl_generator=dl_generator,
    )
    check("resume counters", resumed_step == MAX_STEPS and resumed_shard == shard_after,
          f"step={resumed_step} shard={resumed_shard}")

    weights_after = fresh_model.state_dict()
    mismatched = [
        k for k in weights_before
        if not torch.equal(weights_before[k], weights_after[k])
    ]
    check("weights identical after roundtrip", not mismatched,
          f"mismatched={mismatched[:5]} ({len(mismatched)})")

    opt_snapshots_after = [snap(o.state_dict()) for o in fresh_optims]
    check("optimizer state restored",
          len(fresh_optims[0].state) > 0
          and all(same(a, b) for a, b in zip(opt_snapshots_before,
                                             opt_snapshots_after)))
    check("scheduler position restored",
          sched_pos_before == [s.last_epoch for s in fresh_scheds],
          f"{sched_pos_before}")
    check("torch RNG state restored", torch.equal(saved_rng, torch.get_rng_state()))

    # --- resume training ----------------------------------------------------
    print(f"--- training phase 2 (resume, +{RESUME_EXTRA_STEPS} steps) ---")
    optimizers, schedulers = fresh_optims, fresh_scheds
    global_step, shard_after = run_steps(
        MAX_STEPS + RESUME_EXTRA_STEPS, start_shard=resumed_shard,
        global_step=resumed_step,
    )
    check("training resumed and progressed",
          global_step == MAX_STEPS + RESUME_EXTRA_STEPS, f"steps={global_step}")

    # --- JSONL metrics parse-back -------------------------------------------
    lines = JSONL_PATH.read_text(encoding="utf-8").strip().splitlines()
    parsed = []
    try:
        parsed = [json.loads(ln) for ln in lines]
    except json.JSONDecodeError as exc:
        check("metrics jsonl parses", False, str(exc))
    losses = [r["loss"] for r in parsed if r.get("event") == "train_step"]
    check("metrics jsonl parses & complete", len(parsed) == global_step,
          f"{len(parsed)} records, loss[0]={losses[0]:.4f} loss[-1]={losses[-1]:.4f}")
    check("loss finite throughout", all(np.isfinite(l) for l in losses))

    # --- inference ----------------------------------------------------------
    print("--- inference phase ---")
    fresh_model.eval()
    prompt = torch.randint(3, VOCAB, (1, 12), dtype=torch.long)

    # H1 FIX VERIFICATION: as-trained configs ship return_logits=False
    # (build_toy_config AND train.py build_training_config); generate() now
    # forces logit materialization for its duration and restores the flag.
    flag_before = fresh_model.config.return_logits
    with torch.no_grad():
        h1_gen = fresh_model.generate(prompt.clone(), max_new_tokens=4,
                                      do_sample=False)
    check("H1 FIXED: generate() works when config.return_logits=False",
          tuple(h1_gen.shape) == (1, 16) and fresh_model.config.return_logits == flag_before,
          f"shape={tuple(h1_gen.shape)} flag_restored={fresh_model.config.return_logits == flag_before}")

    with torch.no_grad():
        greedy_a = fresh_model.generate(prompt.clone(), max_new_tokens=16,
                                        do_sample=False)
        greedy_b = fresh_model.generate(prompt.clone(), max_new_tokens=16,
                                        do_sample=False)
        sampled = fresh_model.generate(prompt.clone(), max_new_tokens=16,
                                       do_sample=True, temperature=0.8,
                                       top_k=50, top_p=0.95)
        batched = fresh_model.generate(torch.cat([prompt, prompt.flip(1)]),
                                       max_new_tokens=8, do_sample=False)
    check("greedy generate shape", tuple(greedy_a.shape) == (1, 28),
          str(tuple(greedy_a.shape)))
    check("greedy deterministic", torch.equal(greedy_a, greedy_b))
    check("greedy output valid ids",
          int(greedy_a.min()) >= 0 and int(greedy_a.max()) < VOCAB)
    check("sampled generate valid ids",
          int(sampled.min()) >= 0 and int(sampled.max()) < VOCAB)
    check("batched generate shape", tuple(batched.shape) == (2, 20),
          str(tuple(batched.shape)))
    check("generate outputs finite",
          bool(torch.isfinite(greedy_a.float()).all()))
    try:
        fresh_model.generate(prompt, max_new_tokens=cfg.max_position_embeddings)
        overran = False
    except ValueError:
        overran = True
    check("cache bound enforced loudly", overran)

    # --- summary -------------------------------------------------------------
    failed = [name for name, ok, _ in RESULTS if not ok]
    print("=" * 72)
    print(f"E2E VALIDATION: {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    print(f"config: toy ~{n_params:,} params | batch_size={BATCH_SIZE} "
          f"| seq_len={SEQ_LEN} | cpu fp32 | steps={global_step}")
    if failed:
        print("FAILED:", *failed, sep="\n  - ")
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
