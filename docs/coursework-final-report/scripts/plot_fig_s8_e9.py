# -*- coding: utf-8 -*-
"""图 S8（E9）：(A) 100 epoch 预算下的验证损失曲线；(B) 预算对照摘要；(C) 20k vs 60k
训练曲线与测试准确率对照。只读 E6/E9 产物与 derived_data 汇总。"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]   # 报告根目录（脚本位于 scripts/ 子目录）
REPO = Path(__file__).resolve().parents[4]
E9B = REPO / "outputs" / "p5.3" / "E9_budget_data" / "e9b_budget100"
E9A = REPO / "outputs" / "p5.3" / "E9_budget_data" / "e9a_full60k"
E6 = REPO / "outputs" / "p5.3" / "E6_final"
SUMMARY = json.loads((HERE / "derived_data" / "e9_budget_data.json").read_text(encoding="utf-8"))
TEST_EVAL = json.loads((HERE / "derived_data" / "e9a_full60k_test_eval.json").read_text(encoding="utf-8"))
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [404, 505, 606]
OKABE = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73"}


def history(path, cols=("val_loss", "train_loss")):
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    return {c: np.array([float(r[c]) for r in rows if r[c] != ""]) for c in cols}


fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), constrained_layout=True)
axA, axB, axC = axes

# ---- A：100ep no_es 验证损失曲线 ----
for seed, color in zip(SEEDS, [OKABE["blue"], OKABE["vermillion"], OKABE["green"]]):
    h = history(E9B / f"e9b_budget100_no_es_seed{seed}" / "history.csv", cols=("val_loss",))
    ep = np.arange(1, len(h["val_loss"]) + 1)
    axA.plot(ep, h["val_loss"], lw=0.9, color=color, alpha=0.9, label=f"seed {seed}")
    best = SUMMARY["budget_study"]["per_run"][f"no_es_seed{seed}"]["best_epoch"]
    axA.plot(best, h["val_loss"][best - 1], "*", ms=7, color=color)
axA.axvspan(0, 30, color="#DDDDDD", alpha=0.5, zorder=0)
axA.text(15.5, axA.get_ylim()[1] * 0.985, "E6 budget\n(30 epochs)", ha="center", va="top",
         fontsize=7.2, color="#555555")
axA.set_xlabel("epoch", fontsize=8.6)
axA.set_ylabel("validation loss", fontsize=8.6)
axA.set_title("(A) 100 epochs, no-ES (star = best)", fontsize=8.8, loc="left")
axA.legend(fontsize=6.8, loc="upper right", frameon=False)
axA.tick_params(labelsize=7.4)
axA.grid(lw=0.4, alpha=0.5)
axA.set_axisbelow(True)

# ---- B：预算对照摘要（best val loss 均值±std，逐 seed 点）----
groups = [("30 ep\n(E6 frozen)", SUMMARY["budget_study"]["e6_30ep_frozen"]["best_validation_loss"], OKABE["green"]),
          ("100 ep\nno-ES", SUMMARY["budget_study"]["budget100_no_es"]["best_validation_loss"], OKABE["blue"]),
          ("100 ep\nES", SUMMARY["budget_study"]["budget100_es"]["best_validation_loss"], OKABE["vermillion"])]
for i, (label, st, color) in enumerate(groups):
    axB.bar(i, st["mean"], width=0.62, color=color, alpha=0.75,
            yerr=st["pstdev_ddof0"], capsize=3, error_kw={"lw": 0.9})
    for j, v in enumerate(st["values"]):
        axB.plot(i, v, "o", ms=3, color="#333333", alpha=0.75, zorder=3)
axB.set_xticks(range(3), [g[0] for g in groups], fontsize=7.4)
axB.set_ylabel("mean best val loss", fontsize=8.6)
axB.set_title("(B) budget comparison (3 seeds)", fontsize=8.8, loc="left")
axB.tick_params(axis="y", labelsize=7.4)
axB.grid(axis="y", lw=0.4, alpha=0.5)
axB.set_axisbelow(True)

# ---- C：20k vs 60k 训练曲线 ----
for seed in SEEDS:
    h20 = history(E6 / f"e6_final_seed{seed}" / "history.csv", cols=("train_loss",))
    axC.plot(np.arange(1, 31), h20["train_loss"], "--", lw=0.9, color=OKABE["blue"], alpha=0.55)
for seed in SEEDS:
    h60 = history(E9A / f"e9a_full60k_seed{seed}" / "history.csv", cols=("train_loss",))
    axC.plot(np.arange(1, len(h60["train_loss"]) + 1), h60["train_loss"], "-", lw=1.0,
             color=OKABE["vermillion"], alpha=0.8)
acc60 = TEST_EVAL["group_stats"]["test_accuracy"]["mean"] * 100
e6_acc = 97.35
axC.plot([], [], "--", lw=1.0, color=OKABE["blue"], label=f"20k (E6) -> test {e6_acc:.2f}%")
axC.plot([], [], "-", lw=1.0, color=OKABE["vermillion"], label=f"60k (E9-A) -> test {acc60:.2f}%")
axC.set_xlabel("epoch", fontsize=8.6)
axC.set_ylabel("train loss (log scale)", fontsize=8.6)
axC.set_yscale("log")
axC.set_title("(C) data scale: 20k vs 60k", fontsize=8.8, loc="left")
axC.legend(fontsize=6.6, loc="upper right", frameon=False)
axC.tick_params(labelsize=7.4)
axC.grid(lw=0.4, alpha=0.5)
axC.set_axisbelow(True)

for ext in ("png", "svg"):
    fig.savefig(FIG_DIR / f"fig_s8_e9_budget_data.{ext}", dpi=300)
print(f"[done] {FIG_DIR / 'fig_s8_e9_budget_data.png'}")
