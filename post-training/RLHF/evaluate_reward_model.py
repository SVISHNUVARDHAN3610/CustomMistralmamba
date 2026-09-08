"""Independently load a reward checkpoint and score held-out HelpSteer2 preferences."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "post-training"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import torch

from RLHF.config import RewardConfig
from RLHF.reward_model import RewardModel
from RLHF.train_reward_model import (
    RewardBackend,
    checkpoint_metadata,
    evaluate,
    initialize_tokenizer,
    load_checkpoint,
    load_pretraining_fsdp2,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="runs/reward_evaluation")
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Bounded sample; omitted evaluates the entire preference release",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Offline tiny HelpSteer2 schema fixture on CPU",
    )
    args = parser.parse_args(argv)
    cfg = RewardConfig.from_dict(checkpoint_metadata(args.checkpoint)["config"])
    if args.smoke and cfg.model.d_model > 32:
        parser.error("--smoke accepts only tiny checkpoints")
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive")
    cfg.system.output_dir = args.output_dir
    logger = load_pretraining_fsdp2()._setup_logging(Path(args.output_dir))
    try:
        backend = RewardBackend(cfg, logger, args.smoke)
        tokenizer = initialize_tokenizer(cfg, args.smoke)
        model = backend.wrap(RewardModel(cfg.model), cfg)
        load_checkpoint(args.checkpoint, model, cfg=cfg)
        result = evaluate(
            model,
            cfg,
            tokenizer,
            backend,
            "helpsteer2",
            args.smoke,
            args.max_batches or (2 if args.smoke else None),
        )
        if backend.rank == 0:
            logger.info("HelpSteer2 %s", json.dumps(result))
            Path(args.output_dir, "helpsteer2_metrics.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
        return result
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
