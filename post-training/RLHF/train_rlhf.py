"""PPO policy optimization: torchrun in the cloud; --smoke is tiny/offline/CPU."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "post-training"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset

from model.core.config import HybridMambaMoEConfig
from model.hybrid.mamba import fused_mamba_scan_available
from model.hybrid.model import HybridForCausalLM
from RLHF.config import PPOConfig, RewardConfig, smoke_config
from RLHF.evaluation_metrics import isolated_evaluation_rng
from RLHF.ppo import (
    PolicyValueModel,
    Rollout,
    assert_frozen,
    collect_rollout,
    ppo_objective,
)
from RLHF.ppo_dataset import PromptShardDataset, prompt_collator, prompt_feed
from RLHF.reward_model import RewardModel, parameter_counts
from RLHF.train_reward_model import (
    RewardBackend,
    checkpoint_metadata,
    initialize_tokenizer,
    load_checkpoint,
    load_pretraining_fsdp2,
    require_mamba_kernels,
    save_checkpoint,
    sft,
)

FAMILY = "hybrid_ppo_dcp_v1"


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def emit(backend, cfg, record):
    if backend.rank == 0:
        backend.logger.info("ppo %s", json.dumps(record))
        with Path(cfg.system.output_dir, "ppo_metrics.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(record) + "\n")


def load_sft_policy(cfg, backend, tokenizer, *, with_value):
    """Reuse the actual SFT consolidated checkpoint contract, rank-serial CPU load.

    CPU initialization avoids ever placing a full policy/reference on each GPU.
    FSDP shards parameters before inference; host RAM must hold one checkpoint.
    """
    model = None
    for turn in range(backend.world):
        error = None
        if turn == backend.rank:
            try:
                checkpoint = sft.read_checkpoint(cfg.rlhf.policy_checkpoint)
                contract = checkpoint.get("sft_runtime", {})
                if not contract.get("family", "").startswith("sft_"):
                    raise ValueError(
                        "policy_checkpoint must be a repository SFT checkpoint"
                    )
                if contract.get("tokenizer") != cfg.data.tokenizer_name:
                    raise ValueError("SFT and Reward Model must use the same tokenizer")
                if hasattr(tokenizer, "get_vocab"):
                    vocab_hash = hashlib.sha256(
                        json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
                    ).hexdigest()
                    if contract.get("vocab_hash") != vocab_hash:
                        raise ValueError(
                            "SFT tokenizer vocabulary differs from reward tokenizer"
                        )
                policy_config = HybridMambaMoEConfig.from_dict(checkpoint["config"])
                policy_config.use_torch_compile = False
                policy_config.gradient_checkpointing = False
                for name in ("bos_token_id", "eos_token_id"):
                    if getattr(policy_config, name) != getattr(tokenizer, name):
                        raise ValueError(f"SFT/tokenizer {name} mismatch")
                if policy_config.vocab_size != cfg.model.vocab_size:
                    raise ValueError("Policy and reward vocabularies must match")
                if (
                    cfg.rlhf.max_prompt_tokens + cfg.rlhf.max_new_tokens
                    > policy_config.max_position_embeddings
                ):
                    raise ValueError("PPO sequence exceeds policy context")
                model = PolicyValueModel(policy_config, with_value=with_value)
                missing, unexpected = model.load_state_dict(
                    checkpoint["model_state_dict"], strict=False
                )
                if (
                    set(missing)
                    != (
                        {"value_head.weight", "value_head.bias"}
                        if with_value
                        else set()
                    )
                    or unexpected
                ):
                    raise ValueError(
                        f"SFT architecture mismatch: {missing}, {unexpected}"
                    )
                del checkpoint
                if not with_value:
                    model.requires_grad_(False)
            except Exception as exc:  # noqa: BLE001 -- propagate loading failure to all ranks
                error = str(exc)
        if backend.world > 1:
            messages = [error]
            torch.distributed.broadcast_object_list(messages, src=turn)
            error = messages[0]
        if error:
            raise RuntimeError(f"SFT initialization rank {turn}: {error}")
    require_mamba_kernels(model, cfg.system.require_fused_mamba, backend.device)
    wrap_cfg = copy.deepcopy(cfg)
    wrap_cfg.system.activation_checkpointing &= with_value
    return backend.wrap(model, wrap_cfg).eval()


def concatenate_rollouts(parts):
    # Detach to CPU between phases: keep only the current optimization microbatch on GPU.
    return Rollout(
        **{
            f.name: torch.cat([getattr(p, f.name).detach().cpu() for p in parts])
            for f in fields(Rollout)
        }
    )


def collect_batch(policy, reference, reward, prompts, cfg, backend, tokenizer):
    parts = []
    size = cfg.rlhf.rollout_microbatch_size
    for i in range(0, len(prompts), size):
        part = collect_rollout(
            policy,
            reference,
            reward,
            prompts[i : i + size],
            cfg.rlhf,
            backend,
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
        )
        parts.append(
            Rollout(**{f.name: getattr(part, f.name).cpu() for f in fields(Rollout)})
        )
    return concatenate_rollouts(parts)


def rollout_metrics(rollout, backend):
    """Rank-zero exact diagnostics for this bounded rollout, never full vocab logits."""
    mask = rollout.response_mask
    local = {
        "raw_reward_model_score": rollout.raw_rewards.tolist(),
        "kl": (rollout.old_logprobs - rollout.reference_logprobs)[mask].tolist(),
        "KL_penalty": (rollout.raw_rewards - rollout.token_rewards.sum(-1)).tolist(),
        "final_rl_reward": rollout.token_rewards.sum(-1).tolist(),
        "response_length": mask.sum(-1).tolist(),
        "eos_rate": rollout.eos.float().tolist(),
        "advantage": rollout.advantages[mask].tolist(),
        "return": rollout.returns[mask].tolist(),
        "value": rollout.old_values[mask].tolist(),
    }
    all_ranks = [local]
    if backend.world > 1:
        all_ranks = [None] * backend.world if backend.rank == 0 else None
        torch.distributed.gather_object(local, all_ranks, dst=0)
    result = None
    if backend.rank == 0:
        result = {}
        for name in local:
            values = torch.tensor(
                [v for group in all_ranks for v in group[name]], dtype=torch.float64
            )
            result[name + "_mean"] = values.mean().item()
            result[name + "_std"] = values.std(unbiased=False).item()
            result[name + "_p95"] = values.quantile(0.95).item()
        result["mean_kl"] = result["kl_mean"]
        result["mean_reward"] = result["raw_reward_model_score_mean"]
        result["reward_std"] = result["raw_reward_model_score_std"]
        result["completion_rate"] = result["eos_rate_mean"]
        returns = torch.tensor(
            [v for g in all_ranks for v in g["return"]], dtype=torch.float64
        )
        values = torch.tensor(
            [v for g in all_ranks for v in g["value"]], dtype=torch.float64
        )
        variance = returns.var(unbiased=False)
        result["explained_variance"] = (
            (1 - (returns - values).var(unbiased=False) / variance).item()
            if variance > 0
            else None
        )
    return backend.broadcast(result)


def normalize_advantages(rollout, backend):
    values = rollout.advantages[rollout.response_mask].double().to(backend.device)
    count_sum = backend.sum(
        torch.stack((values.new_tensor(values.numel()), values.sum()))
    )
    mean = count_sum[1] / count_sum[0]
    variance = backend.sum((values - mean).square().sum()) / count_sum[0]
    rollout.advantages = (
        ((rollout.advantages - mean.cpu()) / (variance.cpu().sqrt() + 1e-8))
        .float()
        .masked_fill(~rollout.response_mask, 0)
    )


def optimize_rollout(policy, rollout, optimizer, scheduler, cfg, backend):
    """Global token-weighted accumulation; old rollout tensors are immutable."""
    policy.eval()  # gradients enabled, dropout/auxiliary calibration disabled
    if cfg.rlhf.advantage_normalization:
        normalize_advantages(rollout, backend)
    metrics, total_tokens, updates = {}, 0, 0
    batch_size = cfg.rlhf.ppo_minibatch_size
    accumulation = cfg.training.gradient_accumulation_steps
    optimizer.zero_grad(set_to_none=True)
    for _ in range(cfg.rlhf.ppo_epochs):
        indices = torch.randperm(len(rollout.raw_rewards))
        minibatches = list(indices.split(batch_size))
        for start in range(0, len(minibatches), accumulation):
            window = minibatches[start : start + accumulation]
            global_tokens = backend.sum(
                torch.tensor(
                    sum(rollout.response_mask[ix].sum().item() for ix in window),
                    device=backend.device,
                    dtype=torch.float64,
                )
            )
            for j, ix in enumerate(window):
                batch = {
                    f.name: getattr(rollout, f.name)[ix].to(
                        backend.device, non_blocking=True
                    )
                    for f in fields(Rollout)
                }
                backend.sync_backward(policy, j == len(window) - 1)
                with backend.autocast():
                    logprobs, values, entropy = policy(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["positions"],
                        tokens=batch["tokens"],
                        temperature=cfg.rlhf.temperature,
                    )
                loss, observed = ppo_objective(
                    logprobs,
                    batch["old_logprobs"],
                    values,
                    batch["old_values"],
                    batch["advantages"],
                    batch["returns"],
                    entropy,
                    batch["response_mask"],
                    cfg.rlhf,
                )
                tokens = batch["response_mask"].sum()
                (loss * tokens * backend.world / global_tokens).backward()
                for key, value in observed.items():
                    metrics[key] = metrics.get(key, 0) + value.double() * tokens
                total_tokens += tokens.item()
            norm, _ = backend.optimizer_step(
                policy, optimizer, cfg.training.gradient_clip_norm
            )
            scheduler.step()
            updates += 1
    backend.sync_backward(policy, True)
    count = backend.sum(
        torch.tensor(total_tokens, device=backend.device, dtype=torch.float64)
    )
    result = {
        key: (backend.sum(value) / count).item() for key, value in metrics.items()
    }
    result.update(gradient_norm=norm.item(), learning_rate=scheduler.get_last_lr()[0])
    return result, updates


def diagnostic_prompts(cfg, tokenizer, backend, smoke):
    rows = []
    with isolated_evaluation_rng():
        feed = prompt_feed(cfg, tokenizer, backend, diagnostic=True, smoke=smoke)
        try:
            shard = 0
            while len(rows) < cfg.rlhf.diagnostic_prompts:
                path = feed.wait(shard)
                if path is None:
                    raise ValueError("Not enough valid policy diagnostic prompts")
                dataset = PromptShardDataset(path)
                try:
                    rows.extend(
                        dataset[i].tolist()
                        for i in range(
                            min(len(dataset), cfg.rlhf.diagnostic_prompts - len(rows))
                        )
                    )
                finally:
                    dataset.close()
                feed.consume(path)
                shard += 1
        finally:
            feed.close()
    return rows


def evaluate_policy(policy, reference, reward, rows, cfg, backend, tokenizer, step):
    with isolated_evaluation_rng():
        torch.manual_seed(cfg.seed + 7001)
        rollout = collect_batch(
            policy, reference, reward, rows, cfg, backend, tokenizer
        )
        # Diagnostics are replicated: report one observation per prompt.
        reporting = SimpleNamespace(
            rank=backend.rank, world=1, broadcast=backend.broadcast
        )
        record = rollout_metrics(rollout, reporting)
    record.update(
        step=step,
        evaluation="policy_fixed_prompts",
        subset_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
    )
    emit(backend, cfg, record)
    return record


def run_training(
    cfg: RewardConfig,
    *,
    smoke: bool = False,
    stop_after: int | None = None,
    preflight: bool = False,
) -> str:
    cfg.validate()
    if cfg.rlhf is None:
        raise ValueError("PPO requires an rlhf configuration section")
    logger = load_pretraining_fsdp2()._setup_logging(Path(cfg.system.output_dir))
    feed = None
    try:
        backend = RewardBackend(cfg, logger, smoke)
        if cfg.system.require_fused_mamba and not fused_mamba_scan_available():
            raise RuntimeError(
                "require_fused_mamba: install a compatible mamba-ssm CUDA build"
            )
        if preflight and (
            smoke
            or backend.world != 2
            or backend.precision != "bf16"
            or not cfg.system.require_fused_mamba
        ):
            raise ValueError(
                "--preflight requires two cloud CUDA GPUs, BF16 and required fused Mamba"
            )
        sources = None
        if backend.rank == 0:
            try:
                prepare_sources(cfg, smoke)
                sources = {"config": cfg.to_dict()}
            except Exception as exc:  # noqa: BLE001 -- do not strand peers during checkpoint/Hub reads
                sources = {"error": str(exc)}
        sources = backend.broadcast(sources)
        if "error" in sources:
            raise RuntimeError(f"PPO source initialization failed: {sources['error']}")
        cfg.__dict__.update(RewardConfig.from_dict(sources["config"]).__dict__)
        if cfg.rlhf.max_prompt_tokens + cfg.rlhf.max_new_tokens > cfg.model.max_length:
            raise ValueError(
                "PPO context exceeds Reward Model context; no silent truncation"
            )
        if cfg.data.pairs_per_shard < backend.world * cfg.rlhf.rollout_batch_size:
            raise ValueError("pairs_per_shard must fit a global PPO rollout batch")
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        tokenizer = initialize_tokenizer(cfg, smoke)
        policy = load_sft_policy(cfg, backend, tokenizer, with_value=True)
        reference = load_sft_policy(cfg, backend, tokenizer, with_value=False)
        reward = RewardModel(cfg.model).requires_grad_(False).eval()
        parameter_counts(reward, enforce_size=not smoke)
        require_mamba_kernels(reward, cfg.system.require_fused_mamba, backend.device)
        frozen_cfg = copy.deepcopy(cfg)
        frozen_cfg.system.activation_checkpointing = False
        reward = backend.wrap(reward, frozen_cfg)
        load_checkpoint(cfg.rlhf.reward_model_checkpoint, reward, cfg=cfg)
        reward.eval()
        assert_frozen(reference)
        assert_frozen(reward)
        args = SimpleNamespace(
            no_muon=True,
            lr=cfg.training.learning_rate,
            adam_lr=None,
            muon_lr=None,
            weight_decay=cfg.training.weight_decay,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
        )
        optimizer = backend.base.build_fsdp2_optimizers(
            policy, args=args, logger=logger
        )[0][0]
        # Conditional experts/memory and auxiliary-only parameters can have no
        # gradients in a rollout. Materialize Adam's sharded state via the public
        # DCP helper so the save/load schema does not depend on which routes fired.
        # That helper takes a zero-LR dummy step; reset counters before real PPO.
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

        get_optimizer_state_dict(policy, optimizer)
        for state in optimizer.state.values():
            state["step"].zero_()
        per_rollout = (
            math.ceil(
                (cfg.rlhf.rollout_batch_size // cfg.rlhf.ppo_minibatch_size)
                / cfg.training.gradient_accumulation_steps
            )
            * cfg.rlhf.ppo_epochs
        )
        horizon = cfg.training.max_steps * per_rollout
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            sft.pretrain._build_lr_lambda(
                int(horizon * cfg.training.warmup_ratio), horizon, 0.1
            ),
        )
        step, cursor = (
            0,
            {
                "epoch": 0,
                "shard": 0,
                "batch": 0,
                "raw_start": 0,
                "native_start": None,
                "optimizer_steps": 0,
            },
        )
        if cfg.system.resume_from_checkpoint:
            metadata = load_checkpoint(
                cfg.system.resume_from_checkpoint,
                policy,
                optimizer,
                scheduler,
                cfg,
                backend,
                family=FAMILY,
            )
            step, cursor = metadata["step"], metadata["cursor"]
        # Distinct rollout RNG streams after identical initialization; resume restores them.
        else:
            torch.manual_seed(cfg.seed + backend.rank)
        rows = (
            diagnostic_prompts(cfg, tokenizer, backend, smoke)
            if cfg.system.diagnostic_eval_enabled
            else None
        )
        if rows is not None:
            evaluate_policy(
                policy, reference, reward, rows, cfg, backend, tokenizer, step
            )
        target = min(cfg.training.max_steps, stop_after or cfg.training.max_steps)
        if cfg.data.pairs_per_shard < backend.world * cfg.rlhf.rollout_batch_size:
            raise ValueError(
                "pairs_per_shard must fit a global PPO rollout batch (prompt records in PPO)"
            )
        last_saved = None
        while step < target and cursor["epoch"] < cfg.training.num_epochs:
            feed = prompt_feed(cfg, tokenizer, backend, cursor, smoke=smoke)
            try:
                while step < target:
                    path = feed.wait(cursor["shard"])
                    if path is None:
                        cursor.update(
                            epoch=cursor["epoch"] + 1,
                            shard=0,
                            batch=0,
                            raw_start=0,
                            native_start=None,
                        )
                        break
                    dataset = PromptShardDataset(path)
                    sampler = DistributedSampler(
                        dataset,
                        num_replicas=backend.world,
                        rank=backend.rank,
                        seed=cfg.seed,
                        shuffle=True,
                        drop_last=True,
                    )
                    sampler.set_epoch(cursor["shard"] + cursor["epoch"] * 1000003)
                    indices = list(sampler)[
                        cursor["batch"] * cfg.rlhf.rollout_batch_size :
                    ]
                    loader = DataLoader(
                        Subset(dataset, indices),
                        batch_size=cfg.rlhf.rollout_batch_size,
                        drop_last=True,
                        collate_fn=prompt_collator,
                        num_workers=cfg.data.num_workers,
                        generator=torch.Generator().manual_seed(cfg.seed),
                        **(
                            {"prefetch_factor": cfg.data.prefetch_factor}
                            if cfg.data.num_workers
                            else {}
                        ),
                    )
                    try:
                        for prompts in loader:
                            start = time.monotonic()
                            rollout = collect_batch(
                                policy,
                                reference,
                                reward,
                                prompts,
                                cfg,
                                backend,
                                tokenizer,
                            )
                            report = rollout_metrics(rollout, backend)
                            optimization, updates = optimize_rollout(
                                policy, rollout, optimizer, scheduler, cfg, backend
                            )
                            assert_frozen(reference)
                            assert_frozen(reward)
                            step += 1
                            cursor.update(
                                batch=cursor["batch"] + 1,
                                optimizer_steps=cursor["optimizer_steps"] + updates,
                            )
                            elapsed = time.monotonic() - start
                            report.update(
                                optimization,
                                step=step,
                                optimizer_steps=cursor["optimizer_steps"],
                                epoch=cursor["epoch"],
                                elapsed_seconds=elapsed,
                                prompts_per_second=len(prompts)
                                * backend.world
                                / elapsed,
                                gpu_memory_bytes=torch.cuda.max_memory_allocated()
                                if backend.device.type == "cuda"
                                else 0,
                            )
                            emit(backend, cfg, report)
                            del rollout
                            if (
                                rows is not None
                                and step % cfg.system.diagnostic_eval_interval == 0
                            ):
                                evaluate_policy(
                                    policy,
                                    reference,
                                    reward,
                                    rows,
                                    cfg,
                                    backend,
                                    tokenizer,
                                    step,
                                )
                            if step % cfg.system.save_interval == 0 or step == target:
                                last_saved = (
                                    Path(cfg.system.output_dir)
                                    / f"checkpoint-{step:08d}"
                                )
                                save_checkpoint(
                                    last_saved,
                                    policy,
                                    optimizer,
                                    scheduler,
                                    cfg,
                                    backend,
                                    cursor,
                                    step,
                                    family=FAMILY,
                                )
                            if step >= target:
                                break
                    finally:
                        del loader
                        dataset.close()
                    if step < target:
                        cursor.update(
                            shard=cursor["shard"] + 1,
                            batch=0,
                            raw_start=dataset.metadata["raw_end"],
                            native_start=dataset.metadata["native_end"],
                        )
                        feed.consume(path)
            finally:
                feed.close()
                feed = None
        if step == 0:
            raise ValueError("No complete PPO rollout batch available")
        final_path = Path(cfg.system.output_dir) / f"checkpoint-{step:08d}"
        if last_saved != final_path and not final_path.exists():
            save_checkpoint(
                final_path,
                policy,
                optimizer,
                scheduler,
                cfg,
                backend,
                cursor,
                step,
                family=FAMILY,
            )
        if rows is not None:
            evaluate_policy(
                policy, reference, reward, rows, cfg, backend, tokenizer, step
            )
        if preflight:
            emit(
                backend,
                cfg,
                {
                    "preflight": "resumed_update_completed"
                    if cfg.system.resume_from_checkpoint
                    else "checkpoint_saved_resume_required",
                    "step": step,
                },
            )
        return str(final_path)
    finally:
        if feed is not None:
            feed.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def prepare_sources(cfg, smoke=False):
    """Pin source identity before saving configs; frozen models are referenced once."""
    if not cfg.rlhf.policy_checkpoint or not cfg.rlhf.reward_model_checkpoint:
        raise ValueError("Set policy_checkpoint and reward_model_checkpoint")
    path = Path(cfg.rlhf.policy_checkpoint)
    if path.is_dir():
        path /= sft.pretrain.CHECKPOINT_FILENAME
    for name, source in (
        ("policy_sha256", path),
        (
            "reward_metadata_sha256",
            Path(cfg.rlhf.reward_model_checkpoint) / "trainer.pt",
        ),
    ):
        digest = file_digest(source)
        if getattr(cfg.rlhf, name) and getattr(cfg.rlhf, name) != digest:
            raise ValueError(f"Frozen checkpoint identity changed: {source}")
        setattr(cfg.rlhf, name, digest)
    saved = RewardConfig.from_dict(
        checkpoint_metadata(cfg.rlhf.reward_model_checkpoint)["config"]
    )
    cfg.model = saved.model
    cfg.data.tokenizer_name = saved.data.tokenizer_name
    cfg.data.tokenizer_revision = saved.data.tokenizer_revision
    if not smoke:
        from huggingface_hub import HfApi

        cfg.rlhf.prompt_revision = (
            HfApi()
            .dataset_info(cfg.rlhf.prompt_dataset, revision=cfg.rlhf.prompt_revision)
            .sha
        )


def create_smoke_inputs(output):
    """Create tiny deterministic *checkpoint fixtures*, never train an SFT/RM job."""
    cfg = smoke_config(output)
    cfg.rlhf = PPOConfig(
        max_prompt_tokens=32,
        max_new_tokens=4,
        rollout_batch_size=2,
        ppo_minibatch_size=1,
        ppo_epochs=2,
        diagnostic_prompts=2,
    )
    cfg.training.gradient_accumulation_steps = 2
    cfg.data.pairs_per_shard = 4
    fixture = Path(output) / "fixtures"
    fixture.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    policy_config = HybridMambaMoEConfig(
        vocab_size=256,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        intermediate_size=24,
        num_experts=2,
        top_k=1,
        mamba_state_size=4,
        max_position_embeddings=64,
        window_size=64,
        use_dual_memory=True,
        memory_size=2,
        memory_num_heads=2,
        memory_chunk_size=16,
        use_auxiliary_losses=True,
        dropout=0.1,
        use_fused_mamba_scan=False,
    )
    policy_path = fixture / "sft.pth"
    if not policy_path.exists():
        torch.save(
            {
                "config": policy_config.to_dict(),
                "model_state_dict": HybridForCausalLM(policy_config).state_dict(),
                "sft_runtime": {
                    "family": "sft_single_gpu_v1",
                    "tokenizer": cfg.data.tokenizer_name,
                },
            },
            policy_path,
        )
    reward_path = fixture / "reward"
    if not reward_path.exists():
        logger = load_pretraining_fsdp2()._setup_logging(fixture)
        try:
            backend = RewardBackend(cfg, logger, smoke=True)
            reward = RewardModel(cfg.model)
            optimizer = torch.optim.AdamW(reward.parameters())
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
            save_checkpoint(
                reward_path, reward, optimizer, scheduler, cfg, backend, {}, 0
            )
        finally:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
    cfg.rlhf.policy_checkpoint = str(policy_path.resolve())
    cfg.rlhf.reward_model_checkpoint = str(reward_path.resolve())
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--write-config")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Cloud only: two BF16 GPUs, two rollout updates, checkpoint then explicit resume",
    )
    args = parser.parse_args(argv)
    if args.smoke and args.preflight:
        parser.error("--smoke and --preflight are mutually exclusive")
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("--stop-after must be positive")
    if args.write_config:
        cfg = RewardConfig(rlhf=PPOConfig())
        cfg.system.output_dir = "runs/ppo"
        cfg.system.precision = "bf16"
        cfg.save_pretrained(args.write_config)
        return
    if args.resume:
        cfg = RewardConfig.from_dict(checkpoint_metadata(args.resume, FAMILY)["config"])
        cfg.system.resume_from_checkpoint = args.resume
    elif args.smoke:
        torch.set_num_threads(1)
        cfg = create_smoke_inputs(args.output_dir or "runs/ppo_smoke")
    elif args.config:
        cfg = RewardConfig.from_pretrained(args.config)
    else:
        parser.error("Supply --config or --smoke")
    if args.output_dir:
        cfg.system.output_dir = args.output_dir
    if args.smoke:
        torch.set_num_threads(1)
        if cfg.model.d_model > 32 or cfg.training.max_steps > 2:
            parser.error("--smoke accepts only tiny synthetic checkpoints")
    if args.preflight:
        cfg.system.precision = "bf16"
        cfg.system.require_fused_mamba = True
        cfg.training.max_steps = 2
        cfg.training.gradient_accumulation_steps = 2
        cfg.system.save_interval = cfg.system.diagnostic_eval_interval = 1
        cfg.rlhf.rollout_batch_size = 2
        cfg.rlhf.rollout_microbatch_size = cfg.rlhf.ppo_minibatch_size = 1
        cfg.rlhf.ppo_epochs = 1
        cfg.rlhf.diagnostic_prompts = 2
        cfg.rlhf.prompt_limit = 1024
        cfg.rlhf.max_prompt_tokens = min(cfg.rlhf.max_prompt_tokens, 128)
        cfg.rlhf.max_new_tokens = min(cfg.rlhf.max_new_tokens, 8)
    return run_training(
        cfg, smoke=args.smoke, stop_after=args.stop_after, preflight=args.preflight
    )


if __name__ == "__main__":
    main()
