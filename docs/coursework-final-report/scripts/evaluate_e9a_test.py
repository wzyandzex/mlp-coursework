# -*- coding: utf-8 -*-
"""E9-A 60k 模型的测试集评估（P5.3-E9 预注册 §5 预先声明的第三次测试集前向，2026-11-08）。

目的：量化完整 60,000 训练样本相对 20k 协议的泛化变化（报告 §13.2 局限 #1 的补充研究）。
纪律：只读加载 E9-A 的 final_model.npz（经 load_checkpoint 安全重建，最终 epoch 参数），
对官方 10,000 测试样本各前向传播一次；不训练、不选型、不改任何冻结产物。
本评估不用于任何选型或结论变更；交付模型仍为 E6 预注册产物。
测试集前向簿记：E6 原始评估（1）→ 2026-10-30 的 ROC/AUC 补充评估（2）→ 本次（3）。

用法：python evaluate_e9a_test.py   （输出 derived_data/e9a_full60k_test_eval.json）
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from src.mlp import MLP                       # noqa: E402
from src.mnist_data import load_mnist_raw, preprocess_images  # noqa: E402
from src.training import (                    # noqa: E402
    confusion_matrix,
    evaluate_with_predictions,
    load_checkpoint,
    macro_average,
    per_class_metrics,
)

E9A_ROOT = REPO / "outputs" / "p5.3" / "E9_budget_data" / "e9a_full60k"
GATE = (REPO / "outputs" / "p5.3" / "E9_budget_data" / "_driver_check"
        / "verify_e6_seed404_replica" / "verification.json")
OUT = Path(__file__).resolve().parents[1] / "derived_data"
SEEDS = [404, 505, 606]


def main():
    if not GATE.is_file() or not json.loads(GATE.read_text(encoding="utf-8"))["all_pass"]:
        raise SystemExit("60k driver verify gate not passed — refusing to evaluate")
    _, _, test_images, test_labels = load_mnist_raw(REPO / "data" / "mnist" / "raw")
    test_X = preprocess_images(test_images)
    test_y = test_labels
    reference = MLP(input_dim=784, hidden_dims=[256], output_dim=10,
                    hidden_activation="relu", initializer="auto", seed=0).params
    results = []
    for seed in SEEDS:
        run_dir = E9A_ROOT / f"e9a_full60k_seed{seed}"
        params = load_checkpoint(run_dir / "final_model.npz", reference)
        model = MLP(input_dim=784, hidden_dims=[256], output_dim=10,
                    hidden_activation="relu", initializer="auto", seed=seed)
        model.set_params(params)
        test_loss, test_accuracy, predictions = evaluate_with_predictions(model, test_X, test_y)
        matrix = confusion_matrix(test_y, predictions, 10)
        per_class = per_class_metrics(matrix)
        macro = macro_average(per_class)
        results.append({
            "seed": seed,
            "model_source": f"outputs/p5.3/E9_budget_data/e9a_full60k/e9a_full60k_seed{seed}/final_model.npz",
            "params": "final_epoch（无验证集，无 best-epoch 选择）",
            "test_loss": test_loss, "test_accuracy": test_accuracy,
            "confusion_matrix": matrix.tolist(),
            "per_class": per_class, "macro_average": macro,
        })
        print(f"[seed {seed}] test_loss={test_loss:.6f} accuracy={test_accuracy:.6f} "
              f"macro_f1={macro['f1']:.6f}")
    losses = [r["test_loss"] for r in results]
    accs = [r["test_accuracy"] for r in results]
    out = {
        "preregistration": "e9.md（P5.3-E9 预注册 V0.1）§5",
        "preregistration_statement": "第三次测试集前向于 e9.md §5 预先声明（2026-11-08；"
        "前向簿记：E6 原始评估、2026-10-30 ROC/AUC 补充评估、本次）",
        "purpose": "量化完整 60000 训练样本的泛化变化；不用于选型，不替换 E6 交付模型",
        "protocol_differences": "60k 无验证集：无早停、无 best-epoch 选择，评估对象为 "
                                "30 epoch 固定预算的最终 epoch 参数（E1–E6 为验证集选最佳 epoch）",
        "per_seed": results,
        "group_stats": {
            "test_loss": {"mean": float(np.mean(losses)), "pstdev_ddof0": float(np.std(losses, ddof=0))},
            "test_accuracy": {"mean": float(np.mean(accs)), "pstdev_ddof0": float(np.std(accs, ddof=0))},
        },
        "reference_e6_20k": "test loss 0.087566 ± 0.001882；accuracy 97.35%（报告 §12.2，冻结值）",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e9a_full60k_test_eval.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {OUT / 'e9a_full60k_test_eval.json'}")


if __name__ == "__main__":
    sys.exit(main())
