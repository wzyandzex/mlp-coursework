# -*- coding: utf-8 -*-
"""E9-A 完整 60,000 训练样本驱动（P5.3-E9 预注册；不改任何冻结文件）。

冻结的 `training.train_model` 要求非空验证集（空验证集显式抛错），而 E9-A 协议
是 train_size=60000 / val_size=0 的无验证集固定预算训练，故按 e9.md §3 用本驱动
复刻其循环语义：完全相同的模型/优化器构造、RNG 派生、小批量迭代、L2 约定与
在线训练指标；逐 epoch 评估与 best-epoch 跟踪仅在提供验证集时启用（用于校验模式）。

用法：
    python train_full60k.py --mode verify    # 20k/5k 复现校验，对账 e6_final_seed404
    python train_full60k.py --mode full60k --seed 404   # 60k 正式运行（3 seed 各一次）

校验门槛（e9.md §3）：verify 模式产物与冻结 e6_final_seed404 的 history.csv
（除 elapsed_seconds 外逐位一致）、best epoch/验证指标、checkpoint 全部参数数组
逐位一致；通过后才允许 full60k。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from src.losses import softmax_cross_entropy       # noqa: E402
from src.mlp import MLP                            # noqa: E402
from src.mnist_data import load_mnist_raw, load_mnist_split, preprocess_images  # noqa: E402
from src.optimizers import MomentumSGD             # noqa: E402
from src.training import (                          # noqa: E402
    _iterate_minibatches,
    add_l2_gradient,
    evaluate,
    l2_penalty,
    load_checkpoint,
    save_checkpoint,
    write_history_csv,
)

E6_REFERENCE = REPO / "outputs" / "p5.3" / "E6_final" / "e6_final_seed404"
OUT_ROOT = REPO / "outputs" / "p5.3" / "E9_budget_data"
FINAL_CONFIG = {  # E6 最终配置（与 e6_final config.json 的有效字段一致）
    "hidden_dims": [256], "hidden_activation": "relu", "initializer": "auto",
    "optimizer": "momentum_sgd", "learning_rate": 0.05, "momentum": 0.9,
    "l2_lambda": 0.0001, "epochs": 30, "batch_size": 128, "eval_batch_size": 1000,
}


def build_model_and_optimizer(model_seed):
    model = MLP(input_dim=784, hidden_dims=FINAL_CONFIG["hidden_dims"], output_dim=10,
                hidden_activation=FINAL_CONFIG["hidden_activation"],
                initializer=FINAL_CONFIG["initializer"], seed=model_seed)
    optimizer = MomentumSGD(FINAL_CONFIG["learning_rate"], momentum=FINAL_CONFIG["momentum"])
    return model, optimizer


def train_loop(model, optimizer, train_X, train_y, val_X, val_y, *, model_seed, epochs):
    """逐句复刻 train_model 的 none 调度 / 无早停 / raw-best 路径；val 可为 None。"""
    lam = FINAL_CONFIG["l2_lambda"]
    batch_size = FINAL_CONFIG["batch_size"]
    eval_batch = FINAL_CONFIG["eval_batch_size"]
    rng = np.random.default_rng(model_seed)   # 与 train_model 相同的 RNG 派生
    n_samples = train_X.shape[0]
    history, best = [], None
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        effective_lr = FINAL_CONFIG["learning_rate"]      # lr_scheduler="none"
        optimizer.learning_rate = effective_lr
        loss_sum = 0.0
        objective_sum = 0.0
        correct = 0
        seen = 0
        for batch_number, batch_idx in enumerate(
            _iterate_minibatches(n_samples, batch_size, rng), start=1
        ):
            batch_X = train_X[batch_idx]
            batch_y = train_y[batch_idx]
            logits, cache = model.forward(batch_X, return_cache=True)
            loss, dlogits = softmax_cross_entropy(logits, batch_y)
            grads = model.backward(dlogits, cache)
            if lam > 0.0:
                penalty = l2_penalty(model.params, lam)
                grads = add_l2_gradient(grads, model.params, lam)
            else:
                penalty = 0.0
            objective = loss + penalty
            optimizer.step(model.params, grads)
            if not (np.isfinite(loss) and np.isfinite(objective)):
                raise RuntimeError(f"non-finite loss/objective at epoch {epoch} batch {batch_number}")
            loss_sum += float(loss) * len(batch_idx)
            objective_sum += float(objective) * len(batch_idx)
            correct += int(np.sum(np.argmax(logits, axis=1) == batch_y))
            seen += len(batch_idx)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            "train_objective_loss": objective_sum / seen,
            "lr": effective_lr,
            "elapsed_seconds": time.perf_counter() - epoch_started,
            # 无验证集时保留 CSV 列结构（写空值），与 train_model 的列定义一致
            "val_loss": None,
            "val_accuracy": None,
        }
        if val_X is not None:
            val_loss, val_accuracy = evaluate(model, val_X, val_y, batch_size=eval_batch)
            if not (np.isfinite(val_loss) and np.isfinite(val_accuracy)):
                raise RuntimeError(f"non-finite validation metrics at epoch {epoch}")
            row["val_loss"] = val_loss
            row["val_accuracy"] = val_accuracy
            if best is None or val_loss < best["val_loss"]:   # raw-best：严格下降，并列取更早
                best = {"val_loss": val_loss, "val_accuracy": val_accuracy, "epoch": epoch,
                        "params": model.get_params_copy()}
        history.append(row)
    result = {"history": history, "total_seconds": time.perf_counter() - started,
              "best": best,
              "final_params": model.get_params_copy()}
    return result


def write_config(run_dir, *, model_seed, train_size, val_size):
    config = {
        "task": "P5.3-E9A-FULL60K",
        "driver": "train_full60k.py（e9.md §3：复刻 train_model 循环语义的独立驱动，"
                  "train_model 本身要求非空验证集，无法直接跑 60k/0）",
        "network": {"input_dim": 784, "hidden_dims": FINAL_CONFIG["hidden_dims"], "output_dim": 10,
                    "hidden_activation": FINAL_CONFIG["hidden_activation"],
                    "initializer": FINAL_CONFIG["initializer"]},
        "training": {"epochs": FINAL_CONFIG["epochs"], "batch_size": FINAL_CONFIG["batch_size"],
                     "learning_rate": FINAL_CONFIG["learning_rate"],
                     "eval_batch_size": FINAL_CONFIG["eval_batch_size"],
                     "optimizer": FINAL_CONFIG["optimizer"], "momentum": FINAL_CONFIG["momentum"],
                     "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "lr_scheduler": "none",
                     "step_size": 10, "gamma": 0.5, "early_stopping": False, "patience": 5,
                     "min_delta": 0.0001, "l2_lambda": FINAL_CONFIG["l2_lambda"]},
        "seeds": {"model_seed": model_seed, "split_seed": 2026},
        "data": {"source_dir": "data/mnist/raw", "train_size": train_size, "val_size": val_size,
                 "split_strategy": "proportional stratified random split per class (largest-remainder "
                                   "quotas, per-class shuffle with explicit seed)",
                 "protocol_note": "full 60000-sample training WITHOUT validation holdout; "
                                  "no best-epoch selection; final-epoch params are the deliverable"},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("verify", "full60k"), required=True)
    parser.add_argument("--seed", type=int, default=404)
    args = parser.parse_args()

    split_seed = 2026
    if args.mode == "verify":
        run_dir = OUT_ROOT / "_driver_check" / "verify_e6_seed404_replica"
        if run_dir.exists():
            raise SystemExit(f"refusing to overwrite: {run_dir}")
        run_dir.mkdir(parents=True)
        split = load_mnist_split(REPO / "data" / "mnist" / "raw", train_size=20000,
                                 val_size=5000, split_seed=split_seed)
        write_config(run_dir, model_seed=args.seed, train_size=20000, val_size=5000)
        model, optimizer = build_model_and_optimizer(args.seed)
        result = train_loop(model, optimizer, split.train_images, split.train_labels,
                            split.val_images, split.val_labels,
                            model_seed=args.seed, epochs=FINAL_CONFIG["epochs"])
        write_history_csv(result["history"], run_dir / "history.csv")
        save_checkpoint(result["best"]["params"], run_dir / "best_model.npz")
        summary = {"status": "success", "run_dir": str(run_dir), "mode": "verify",
                   "best_epoch": result["best"]["epoch"],
                   "best_validation_loss": result["best"]["val_loss"],
                   "best_validation_accuracy": result["best"]["val_accuracy"],
                   "stop_reason": "max_epochs"}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
        # ---- 与冻结 e6_final_seed404 逐位对账 ----
        ref_summary = json.loads((E6_REFERENCE / "summary.json").read_text(encoding="utf-8"))
        checks = {
            "best_epoch": summary["best_epoch"] == ref_summary["best_epoch"],
            "best_validation_loss": summary["best_validation_loss"] == ref_summary["best_validation_loss"],
            "best_validation_accuracy": summary["best_validation_accuracy"] == ref_summary["best_validation_accuracy"],
        }
        ref_rows = [line for line in (E6_REFERENCE / "history.csv").read_text(encoding="utf-8")
                    .strip().splitlines()]
        new_rows = [line for line in (run_dir / "history.csv").read_text(encoding="utf-8")
                    .strip().splitlines()]
        # 按列名剔除 elapsed_seconds 列再逐行比较
        header = ref_rows[0].split(",")
        keep = [i for i, name in enumerate(header) if name != "elapsed_seconds"]
        ref_cmp = [",".join(r.split(",")[i] for i in keep) for r in ref_rows]
        new_cmp = [",".join(r.split(",")[i] for i in keep) for r in new_rows]
        checks["history_bit_exact"] = ref_cmp == new_cmp
        with np.load(E6_REFERENCE / "best_model.npz", allow_pickle=False) as ref_npz:
            ref_params = {k: ref_npz[k] for k in ref_npz.files}
        new_params = load_checkpoint(run_dir / "best_model.npz", ref_params)
        checks["checkpoint_bit_exact"] = all(
            np.array_equal(ref_params[k], new_params[k]) and ref_params[k].dtype == new_params[k].dtype
            for k in ref_params)
        (run_dir / "verification.json").write_text(
            json.dumps({"reference": str(E6_REFERENCE), "checks": checks,
                        "all_pass": all(checks.values())}, indent=2), encoding="utf-8")
        print(json.dumps(checks, indent=1))
        if not all(checks.values()):
            raise SystemExit("VERIFICATION FAILED — full60k runs are NOT allowed (e9.md §3)")
        print("[verify] ALL PASS — driver may run full60k")
        return

    # ---- full60k ----
    run_dir = OUT_ROOT / "e9a_full60k" / f"e9a_full60k_seed{args.seed}"
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite: {run_dir}")
    gate = OUT_ROOT / "_driver_check" / "verify_e6_seed404_replica" / "verification.json"
    if not gate.is_file() or not json.loads(gate.read_text(encoding="utf-8"))["all_pass"]:
        raise SystemExit("verify gate not passed — run `--mode verify` first (e9.md §3)")
    run_dir.mkdir(parents=True)
    # 60k 全量不走切分函数（冻结 split_train_validation 要求 n_val>0）：
    # 直接加载官方 60,000 训练样本，预处理与 load_mnist_split 内部完全一致
    train_images, train_labels, test_images, test_labels = load_mnist_raw(
        REPO / "data" / "mnist" / "raw")
    train_images = preprocess_images(train_images)
    if train_images.shape[0] != 60000 or test_images.shape[0] != 10000:
        raise SystemExit(f"unexpected dataset sizes: train={train_images.shape[0]} "
                         f"test={test_images.shape[0]}")
    write_config(run_dir, model_seed=args.seed, train_size=60000, val_size=0)
    model, optimizer = build_model_and_optimizer(args.seed)
    result = train_loop(model, optimizer, train_images, train_labels,
                        None, None, model_seed=args.seed, epochs=FINAL_CONFIG["epochs"])
    write_history_csv(result["history"], run_dir / "history.csv")
    save_checkpoint(result["final_params"], run_dir / "final_model.npz")
    summary = {
        "status": "success", "run_dir": str(run_dir),
        "experiment_role": "supplement_e9a", "mode": "full60k",
        "test_evaluation": "not_run",          # 测试集前向由独立脚本另行执行并披露
        "stop_reason": "max_epochs",
        "best_epoch": None,                     # 无验证集，无 best-epoch 选择
        "best_validation_loss": None, "best_validation_accuracy": None,
        "test_loss": None, "test_accuracy": None,
        "params_delivered": "final_epoch",
        "sample_counts": {"train": 60000, "validation": 0, "test": 10000},
        "epochs_completed": len(result["history"]),
        "total_seconds": result["total_seconds"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    last = result["history"][-1]
    print(f"[full60k] seed={args.seed} epochs={summary['epochs_completed']} "
          f"final train_loss={last['train_loss']:.6f} train_acc={last['train_accuracy']:.4f} "
          f"({summary['total_seconds']:.0f}s) -> {run_dir}")


if __name__ == "__main__":
    main()
