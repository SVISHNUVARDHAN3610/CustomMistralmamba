"""CPU reward diagnostics: stable moments and disk-backed exact percentiles."""

from __future__ import annotations

import random
import tempfile
from contextlib import ExitStack, contextmanager

import numpy as np
import torch


@contextmanager
def isolated_evaluation_rng():
    """Restore Python, NumPy, CPU and rank-local CUDA RNGs, even on failure."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class RewardStatistics:
    """Exact pooled population statistics; only one batch resides in RAM.

    Float64 Chan/Welford moments avoid cancellation. Four temporary float64
    columns store chosen/rejected/margin/pooled reward values. NumPy partitions
    writable mmaps in-place for exact linear-interpolated quantiles. Disk usage
    is 40 bytes per pair per group; files are deleted by close().
    """

    def __init__(self):
        self.resources = ExitStack()
        self.files = {
            key: self.resources.enter_context(tempfile.TemporaryFile())  # noqa: SIM115 -- lifetime managed by ExitStack
            for key in ("chosen_reward", "rejected_reward", "margin", "reward")
        }
        self.moments = {
            key: [0, 0.0, 0.0, float("inf"), -float("inf")] for key in self.files
        }
        self.positive = self.zero = self.negative = 0
        self.loss_sum = 0.0

    def update(self, chosen, rejected):
        chosen = torch.as_tensor(chosen).detach().cpu().double().numpy().reshape(-1)
        rejected = torch.as_tensor(rejected).detach().cpu().double().numpy().reshape(-1)
        if chosen.shape != rejected.shape or not (
            np.isfinite(chosen).all() and np.isfinite(rejected).all()
        ):
            raise ValueError("Rewards must be finite, equally sized preference arrays")
        if not len(chosen):
            return
        margin = chosen - rejected
        self.positive += int((margin > 0).sum())
        self.zero += int((margin == 0).sum())
        self.negative += int((margin < 0).sum())
        self.loss_sum += float(np.logaddexp(0, -margin).sum())
        for key, values in zip(
            self.files, (chosen, rejected, margin, np.concatenate((chosen, rejected)))
        ):
            n, mean, m2, low, high = self.moments[key]
            size, batch_mean = len(values), float(values.mean())
            delta = batch_mean - mean
            m2 += float(
                np.square(values - batch_mean).sum()
            ) + delta * delta * n * size / (n + size)
            self.moments[key] = [
                n + size,
                mean + delta * size / (n + size),
                m2,
                min(low, float(values.min())),
                max(high, float(values.max())),
            ]
            values.tofile(self.files[key])

    def result(self):
        pairs = self.positive + self.zero + self.negative
        if not pairs:
            raise ValueError("No valid preference pairs were scored")
        result = {
            "pairs": pairs,
            "pairwise_accuracy": self.positive / pairs,
            "fraction_positive_margin": self.positive / pairs,
            "fraction_zero_margin": self.zero / pairs,
            "fraction_negative_margin": self.negative / pairs,
            "loss": self.loss_sum / pairs,
        }
        for key, handle in self.files.items():
            n, mean, m2, low, high = self.moments[key]
            handle.flush()
            values = np.memmap(handle, dtype=np.float64, mode="r+", shape=(n,))
            try:
                percentiles = np.quantile(
                    values, [0.50, 0.95, 0.99], overwrite_input=True
                )
            finally:
                values._mmap.close()
            for suffix, value in zip(
                ("mean", "std", "min", "max", "p50", "p95", "p99"),
                (mean, np.sqrt(max(0.0, m2 / n)), low, high, *percentiles),
            ):
                result[f"{key}_{suffix}"] = float(value)
        # Existing log consumers and the alternate evaluation spelling remain valid.
        for alias, key in {
            "chosen_reward": "chosen_reward_mean",
            "rejected_reward": "rejected_reward_mean",
            "reward_margin": "margin_mean",
            "mean_chosen_reward": "chosen_reward_mean",
            "mean_rejected_reward": "rejected_reward_mean",
            "mean_reward_margin": "margin_mean",
            "std_reward_margin": "margin_std",
        }.items():
            result[alias] = result[key]
        result["reward_scale_diagnostics"] = {
            key: result[key]
            for key in (
                "chosen_reward_mean",
                "chosen_reward_std",
                "rejected_reward_mean",
                "rejected_reward_std",
                "margin_mean",
                "margin_std",
                "margin_p50",
                "margin_p95",
                "margin_p99",
            )
        }
        return result

    def close(self):
        self.resources.close()
