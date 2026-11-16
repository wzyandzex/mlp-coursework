# -*- coding: utf-8 -*-
"""E9 分析（e9.md 预注册）：预算 100 epoch 对照 + 完整 60k 数据规模研究。

只读输入：
  outputs/p5.3/E9_budget_data/e9b_budget100/   6 个 100-epoch run
  outputs/p5.3/E9_budget_data/e9a_full60k/     3 个 60k run（驱动先经逐位校验门槛）
  outputs/p5.3/E6_final/                        E6 冻结 30ep 参照（只读）
  本目录 derived_data/e9a_full60k_test_eval.json（evaluate_e9a_test.py 产物）
输出：derived_data/e9_budget_data.json
含强制校验：e9b no_es 组 history.csv 前 30 epoch 数值列与 E6 对应 run 逐位一致
（e9.md §2 确定性校验），不一致则退出非零。
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # 报告根目录（脚本位于 scripts/ 子目录）
REPO = Path(__file__).resolve().parents[4]
E9B = REPO / "outputs" / "p5.3" / "E9_budget_data" / "e9b_budget100"
E9A = REPO / "outputs" / "p5.3" / "E9_budget_data" / "e9a_full60k"
E6 = REPO / "outputs" / "p5.3" / "E6_final"
OUT = HERE / "derived_data"
SEEDS = [404, 505, 606]


def read_history(path: Path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    return rows


def numeric_row(row, drop=("elapsed_seconds",)):
    return {k: v for k, v in row.items() if k not in drop}


def load_e9b_run(tag, seed):
    run_dir = E9B / f"e9b_budget100_{tag}_seed{seed}"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    history = read_history(run_dir / "history.csv")
    if summary["status"] != "success" or summary["test_evaluation"] != "not_run":
        raise SystemExit(f"run discipline violation: {run_dir.name}")
    return summary, history


def main():
    # ---- E9-B ----
    e9b = {}
    checks = {}
    for seed in SEEDS:
        e6_summary = json.loads((E6 / f"e6_final_seed{seed}" / "summary.json").read_text(encoding="utf-8"))
        e6_hist = read_history(E6 / f"e6_final_seed{seed}" / "history.csv")
        for tag in ("no_es", "es"):
            summary, history = load_e9b_run(tag, seed)
            e9b[f"{tag}_seed{seed}"] = {
                "best_validation_loss": summary["best_validation_loss"],
                "best_validation_accuracy": summary["best_validation_accuracy"],
                "best_epoch": summary["best_epoch"],
                "stop_reason": summary["stop_reason"],
                "epochs_completed": summary["epochs_completed"],
                "total_seconds": summary["total_seconds"],
            }
            if tag == "no_es":
                # 确定性校验：前 30 epoch 与 E6 逐位一致（除 elapsed_seconds）
                mismatch = [i + 1 for i in range(30)
                            if numeric_row(history[i]) != numeric_row(e6_hist[i])]
                checks[f"first30_epochs_bit_exact_vs_E6_seed{seed}"] = not mismatch
                if mismatch:
                    print(f"MISMATCH at epochs {mismatch} (seed {seed})",
                          file=sys.stderr)
    if not all(checks.values()):
        raise SystemExit(f"E9-B deterministic check FAILED: {checks}")

    def summarize(key_prefix, field):
        vals = np.array([e9b[f"{key_prefix}_seed{s}"][field] for s in SEEDS], dtype=float)
        return {"mean": float(vals.mean()), "pstdev_ddof0": float(vals.std(ddof=0)),
                "values": vals.tolist()}

    e6_stats = {"best_validation_loss": [], "best_validation_accuracy": [], "best_epoch": []}
    for seed in SEEDS:
        s = json.loads((E6 / f"e6_final_seed{seed}" / "summary.json").read_text(encoding="utf-8"))
        e6_stats["best_validation_loss"].append(s["best_validation_loss"])
        e6_stats["best_validation_accuracy"].append(s["best_validation_accuracy"])
        e6_stats["best_epoch"].append(s["best_epoch"])
    e6_group = {k: {"mean": float(np.mean(v)), "pstdev_ddof0": float(np.std(v, ddof=0)),
                    "values": v} for k, v in e6_stats.items()}

    budget = {
        "e6_30ep_frozen": e6_group,
        "budget100_no_es": {f: summarize("no_es", f) for f in
                            ("best_validation_loss", "best_validation_accuracy", "best_epoch")},
        "budget100_es": {f: summarize("es", f) for f in
                         ("best_validation_loss", "best_validation_accuracy", "best_epoch")},
        "per_run": e9b,
    }

    # ---- E9-A ----
    e9a = {"per_run": {}}
    for seed in SEEDS:
        run_dir = E9A / f"e9a_full60k_seed{seed}"
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        history = read_history(run_dir / "history.csv")
        last = history[-1]
        e9a["per_run"][f"seed{seed}"] = {
            "final_train_loss": float(last["train_loss"]),
            "final_train_objective_loss": float(last["train_objective_loss"]),
            "final_train_accuracy": float(last["train_accuracy"]),
            "epochs_completed": summary["epochs_completed"],
            "sample_counts": summary["sample_counts"],
        }
    test_eval = json.loads((OUT / "e9a_full60k_test_eval.json").read_text(encoding="utf-8"))
    e9a["test_eval_reference"] = {
        "preregistration_statement": test_eval["preregistration_statement"], "group_stats": test_eval["group_stats"],
        "per_seed": [{"seed": r["seed"], "test_loss": r["test_loss"],
                      "test_accuracy": r["test_accuracy"],
                      "macro_f1": r["macro_average"]["f1"]} for r in test_eval["per_seed"]],
        "protocol_differences": test_eval["protocol_differences"],
    }

    out = {
        "preregistration": "e9.md（P5.3-E9 预注册 V0.1）",
        "deterministic_checks": checks,
        "budget_study": budget,
        "full60k_study": e9a,
        "note": "3 seed 描述性统计（ddof=0）；E9 无预注册推断检验，不使用'显著'措辞；"
                "60k 模型不替换 E6 交付模型（e9.md §4）",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e9_budget_data.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("checks:", checks)
    for f in ("best_validation_loss", "best_validation_accuracy", "best_epoch"):
        print(f"[budget] {f}: 30ep={e6_group[f]['mean']:.6f} "
              f"100ep_no_es={budget['budget100_no_es'][f]['mean']:.6f} "
              f"100ep_es={budget['budget100_es'][f]['mean']:.6f}")
    for seed in SEEDS:
        r = e9a["per_run"][f"seed{seed}"]
        print(f"[60k seed {seed}] final train_loss={r['final_train_loss']:.6f} "
              f"train_acc={r['final_train_accuracy']:.4f}")
    print(f"[done] {OUT / 'e9_budget_data.json'}")


if __name__ == "__main__":
    main()
