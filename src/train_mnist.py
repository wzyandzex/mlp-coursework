"""MNIST 基线训练 CLI 入口（P4-DATA-TRAIN-01）。

用法（在项目根目录）：

    python -m src.train_mnist --help
    python -m src.train_mnist --download          # 显式下载缺失的 MNIST 文件
    python -m src.train_mnist                     # 用已就绪的数据训练基线

默认不联网；只有传入 --download 才会下载数据。
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

from .mnist_data import (
    DEFAULT_SPLIT_SEED,
    TRAIN_SIZE,
    VAL_SIZE,
    download_mnist,
    load_mnist_split,
)
from .training import run_experiment

DEFAULT_LEARNING_RATE = 0.1  # 默认值经真实 MNIST 小规模冒烟实验确认，见 README
DEFAULT_MODEL_SEED = 42
BASELINE_NETWORK = {
    "input_dim": 784,
    "hidden_dims": [128],
    "output_dim": 10,
    "hidden_activation": "relu",
    "initializer": "auto",
}


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text!r}")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(f"must be a positive finite number, got {text!r}")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.train_mnist",
        description=(
            "Train the hand-written NumPy MLP baseline (784-[128]-10, ReLU, SGD) on MNIST. "
            "Trains on 20000 / validates on 5000 samples split from the official 60000 "
            "training images; the official 10000 test images are evaluated once after "
            "model selection."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mnist/raw"),
        help="Directory containing the four MNIST IDX gzip files (default: data/mnist/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mnist_baseline"),
        help="Root directory for run artifacts (default: outputs/mnist_baseline).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly download missing MNIST files into --data-dir before training. "
        "Without this flag the command never touches the network.",
    )
    parser.add_argument("--epochs", type=_positive_int, default=30, help="Max epochs (default: 30).")
    parser.add_argument("--batch-size", type=_positive_int, default=128, help="Mini-batch size (default: 128).")
    parser.add_argument(
        "--learning-rate",
        type=_positive_float,
        default=DEFAULT_LEARNING_RATE,
        help=f"SGD learning rate, positive finite (default: {DEFAULT_LEARNING_RATE}).",
    )
    parser.add_argument(
        "--seed",
        type=_non_negative_int,
        default=DEFAULT_MODEL_SEED,
        help="Seed for model init and per-epoch shuffling (default: 42).",
    )
    parser.add_argument(
        "--split-seed",
        type=_non_negative_int,
        default=DEFAULT_SPLIT_SEED,
        help="Seed for the stratified train/validation split (default: 2026).",
    )
    parser.add_argument(
        "--train-size",
        type=_positive_int,
        default=TRAIN_SIZE,
        help="Number of training samples drawn from the official 60000 (default: 20000).",
    )
    parser.add_argument(
        "--val-size",
        type=_positive_int,
        default=VAL_SIZE,
        help="Number of validation samples drawn from the official 60000 (default: 5000).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run directory name under --output-dir; defaults to a timestamp.",
    )
    return parser


def _config_data_dir(path: Path) -> str:
    """配置文件里不写绝对用户路径：相对 cwd 可表达的用相对路径，否则只留目录名。"""
    try:
        relative = os.path.relpath(path, Path.cwd())
    except ValueError:
        return path.name
    if relative.startswith(".."):
        return path.name
    return relative.replace("\\", "/")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.download:
        download_mnist(args.data_dir)

    try:
        split = load_mnist_split(
            args.data_dir,
            train_size=args.train_size,
            val_size=args.val_size,
            split_seed=args.split_seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "hint: pass --download to fetch the files automatically, or place them manually.",
            file=sys.stderr,
        )
        return 1

    config = {
        "task": "P4-DATA-TRAIN-01",
        "network": dict(BASELINE_NETWORK),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "eval_batch_size": 1000,
        },
        "seeds": {"model_seed": args.seed, "split_seed": args.split_seed},
        "data": {
            "source_dir": _config_data_dir(args.data_dir),
            "train_size": args.train_size,
            "val_size": args.val_size,
            "split_strategy": split.strategy,
        },
    }

    print(
        f"[data] train={len(split.train_images)} validation={len(split.val_images)} "
        f"test={len(split.test_images)} (split_seed={args.split_seed})"
    )
    try:
        summary = run_experiment(
            train_images=split.train_images,
            train_labels=split.train_labels,
            val_images=split.val_images,
            val_labels=split.val_labels,
            test_images=split.test_images,
            test_labels=split.test_labels,
            config=config,
            output_root=args.output_dir,
            run_name=args.run_name,
            verbose=True,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"[best] epoch {summary['best_epoch']} val_loss={summary['best_validation_loss']:.4f} "
          f"val_acc={summary['best_validation_accuracy']:.4f}")
    print(f"[test] loss={summary['test_loss']:.4f} accuracy={summary['test_accuracy']:.4f}")
    print(f"[artifacts] {summary['run_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
