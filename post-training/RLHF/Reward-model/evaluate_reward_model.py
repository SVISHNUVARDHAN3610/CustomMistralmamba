"""Independently score UF test, HH-RLHF test and held-out HelpSteer2 preferences."""

import argparse
import json
import sys
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
from reward_model import RewardModel
from train_reward_model import (
    RewardBackend,
    checkpoint_metadata,
    evaluate,
    evaluate_full,
    initialize_tokenizer,
    load_checkpoint,
    load_pretraining_fsdp2,
    write_evaluation,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="runs/reward_evaluation")
    parser.add_argument("--percentile-strategy", choices=("exact", "approximate"))
    parser.add_argument("--percentile-sample-size", type=int)
    parser.add_argument(
        "--dataset",
        choices=("all", "ultrafeedback_test", "hh_rlhf_test", "helpsteer2"),
        default="all",
    )
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
    if args.percentile_strategy:
        cfg.system.percentile_strategy = args.percentile_strategy
    if args.percentile_sample_size is not None:
        cfg.system.percentile_sample_size = args.percentile_sample_size
    cfg.validate()
    logger = load_pretraining_fsdp2()._setup_logging(Path(args.output_dir))
    try:
        backend = RewardBackend(cfg, logger, args.smoke)
        tokenizer = initialize_tokenizer(cfg, args.smoke)
        model = backend.wrap(RewardModel(cfg.model), cfg)
        load_checkpoint(args.checkpoint, model, cfg=cfg)
        result = (evaluate_full if args.dataset == "all" else evaluate)(
            model,
            cfg,
            tokenizer,
            backend,
            **({} if args.dataset == "all" else {"purpose": args.dataset}),
            smoke=args.smoke,
            max_batches=args.max_batches or (2 if args.smoke else None),
        )
        report = result if args.dataset == "all" else {args.dataset: result}
        write_evaluation(
            report,
            cfg,
            backend,
            "Full" if args.max_batches is None and not args.smoke else "Sample",
            "standalone",
        )
        if backend.rank == 0:
            Path(
                args.output_dir,
                "full_evaluation_metrics.json"
                if args.dataset == "all"
                else f"{args.dataset}_metrics.json",
            ).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
