"""Cloud-only bounded shard benchmark; never launched by normal training."""

import argparse
import copy
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REWARD_MODEL_DIR = ROOT / "post-training" / "RLHF" / "Reward-model"
TRAIN_DIR = REWARD_MODEL_DIR / "train"
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

for directory in (ROOT / "post-training", REWARD_MODEL_DIR, TRAIN_DIR):
    dir_str = str(directory)
    if dir_str not in sys.path:
        sys.path.append(dir_str)

import torch
from config import RewardConfig
from reward_model import RewardModel, pairwise_loss, parameter_counts
from rlhf_dataset import PreferenceShardProducer
from train_reward_model import (
    PreferenceFeed,
    RewardBackend,
    initialize_tokenizer,
    load_pretraining_fsdp2,
    loader_for_shard,
    pin_revisions,
    require_mamba_kernels,
    score_batch,
)


class TimedProducer(PreferenceShardProducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.publish_seconds = 0.0
        self.published_bytes = self.published_shards = 0

    def _publish_shard(self, array, metadata):
        start = time.monotonic()
        super()._publish_shard(array, metadata)
        self.publish_seconds += time.monotonic() - start
        self.published_bytes += array.nbytes
        self.published_shards += 1


def process_io():
    """Linux kernel-accounted process I/O; cache hits differ from physical reads."""
    path = Path("/proc/self/io")
    if not path.exists():
        return {}
    return {
        k: int(v)
        for k, v in (line.split(":") for line in path.read_text().splitlines())
    }


class GPUUtilization:
    """Optional low-rate nvidia-smi telemetry, with explicit unavailable results."""

    def __init__(self):
        self.samples = []
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                self.samples.append([float(v) for v in output.splitlines()])
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.error = str(exc)
                return
            self.stop.wait(1)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=6)
        return {"samples_by_visible_host_gpu": self.samples, "error": self.error}


def benchmark(cfg, backend, pairs):
    tokenizer = initialize_tokenizer(cfg)
    results = []
    for size in (64, 256, 512, 1024):
        variant = copy.deepcopy(cfg)
        variant.data.pairs_per_shard = size
        variant.data.num_workers = (
            0  # process CPU/I/O counters include tokenizer/reader
        )
        torch.manual_seed(cfg.seed)
        model = RewardModel(cfg.model)
        parameter_counts(model)
        require_mamba_kernels(model, True, backend.device)
        model = backend.wrap(model, variant).train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.training.learning_rate, foreach=False
        )
        feed = PreferenceFeed(
            variant, tokenizer, backend, "train", producer_class=TimedProducer
        )
        telemetry = GPUUtilization() if backend.rank == 0 else None
        if telemetry:
            telemetry.thread.start()
        start, cpu_start, io_start = time.monotonic(), os.times(), process_io()
        consumed = 0
        wait_seconds = 0.0
        logical_read_bytes = 0
        shard = 0
        try:
            while consumed * backend.world < pairs:
                waiting = time.monotonic()
                path = feed.wait(shard)
                wait_seconds += time.monotonic() - waiting
                if path is None:
                    raise ValueError(
                        "Benchmark dataset exhausted before requested pair budget"
                    )
                logical_read_bytes += Path(path).stat().st_size
                dataset, loader = loader_for_shard(
                    path, variant, tokenizer, backend, shard
                )
                try:
                    iterator = iter(loader)
                    while consumed * backend.world < pairs:
                        waiting = time.monotonic()
                        batch = next(iterator, None)
                        wait_seconds += time.monotonic() - waiting
                        if batch is None:
                            break
                        chosen, rejected = score_batch(model, batch, backend)
                        pairwise_loss(chosen, rejected).backward()
                        backend.optimizer_step(
                            model, optimizer, cfg.training.gradient_clip_norm
                        )
                        consumed += chosen.numel()
                finally:
                    dataset.close()
                feed.consume(path)
                shard += 1
            torch.cuda.synchronize()
        finally:
            feed.close()
            gpu = telemetry.close() if telemetry else None
        elapsed = time.monotonic() - start
        cpu_end, io_end = os.times(), process_io()
        counts = (
            backend.sum(
                torch.tensor(
                    [consumed, wait_seconds], device=backend.device, dtype=torch.float64
                )
            )
            .cpu()
            .tolist()
        )
        if backend.rank == 0:
            result = {
                "pairs_per_shard": size,
                "pairs": int(counts[0]),
                "seconds_including_startup": elapsed,
                "pairs_per_second": counts[0] / elapsed,
                "mean_rank_data_wait_seconds": counts[1] / backend.world,
                "rank0_cpu_percent_one_core": 100
                * (cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system)
                / elapsed,
                "rank0_producer_publish_seconds": feed.producer.publish_seconds,
                "rank0_published_shards_including_prefetch": feed.producer.published_shards,
                "rank0_published_bytes_including_prefetch": feed.producer.published_bytes,
                "logical_shard_read_bytes_per_rank": logical_read_bytes,
                "rank0_process_io_delta": {
                    k: io_end[k] - io_start[k] for k in io_start
                },
                "gpu_utilization": gpu,
                "peak_gpu_memory_rank0": torch.cuda.max_memory_allocated(),
                "method": "same-seed fresh model, fixed pair count, one AdamW step per batch; startup and queue shutdown included",
            }
            backend.logger.info("shard benchmark %s", json.dumps(result))
            results.append(result)
        del chosen, rejected, iterator, loader, model, optimizer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if backend.rank == 0:
        Path(cfg.system.output_dir, "shard_benchmark.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cloud-benchmark", action="store_true", required=True)
    parser.add_argument(
        "--pairs",
        type=int,
        default=2048,
        help="Consumed global pairs per candidate, max 8192",
    )
    args = parser.parse_args()
    if not 1024 <= args.pairs <= 8192:
        parser.error("Use a bounded pair count between 1024 and 8192")
    cfg = RewardConfig.from_pretrained(args.config)
    cfg.system.require_fused_mamba = True
    cfg.system.precision = "bf16"
    logger = load_pretraining_fsdp2()._setup_logging(Path(cfg.system.output_dir))
    try:
        backend = RewardBackend(cfg, logger)
        if 64 < backend.world * cfg.training.batch_size:
            raise ValueError("Smallest candidate must fit one global batch")
        if 64 % (backend.world * cfg.training.batch_size):
            raise ValueError("Each candidate must divide into complete global batches")
        if args.pairs % (backend.world * cfg.training.batch_size):
            raise ValueError("--pairs must be divisible by global batch size")
        pin_revisions(cfg, backend)
        benchmark(cfg, backend, args.pairs)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
