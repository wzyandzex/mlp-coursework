# -*- coding: utf-8 -*-
"""E9 前置补充（报告 §12.3/§12.5 深化，对应 §13.4 展望第⑤条）：易混类别对误差专项分析。

数据源：derived_data/e6_test_predictions_seed{404,505,606}.csv —— 2026-10-30 补充评估产物
（报告 §12.5），本脚本只做只读分析，不接触模型与测试集原件。

输出：
  derived_data/e6_confusion_pairs.json     逐对错误统计（逐 seed + 3 seed 合并口径）
  figures/fig_s5_error_pairs.png/.svg      图 S5（A 高频混淆对；B 10×10 错误率热图；
                                           C 重点方向对（4↔9、7↔2、7↔9）错误置信度分布）
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]   # 报告根目录（脚本位于 scripts/ 子目录）
ROOT = Path(__file__).resolve().parents[4]
PRED = HERE / "derived_data"          # 逐样本预测 CSV 随本报告目录交付
FIG_DIR = HERE / "figures"
OUT = HERE / "derived_data"
SEEDS = [404, 505, 606]
FOCUS = [(4, 9), (9, 4), (7, 2), (2, 7), (7, 9), (9, 7)]
OKABE = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
         "orange": "#E69F00", "sky": "#56B4E9", "purple": "#CC79A7"}


def load_predictions(seed):
    path = PRED / f"e6_test_predictions_seed{seed}.csv"
    true_l, pred_l, correct, probs = [], [], [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            true_l.append(int(row["true_label"]))
            pred_l.append(int(row["pred_label"]))
            correct.append(int(row["correct"]))
            probs.append([float(row[f"p_{c}"]) for c in range(10)])
    return (np.array(true_l), np.array(pred_l), np.array(correct), np.array(probs))


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    for seed in SEEDS:
        true_l, pred_l, correct, probs = load_predictions(seed)
        support = np.bincount(true_l, minlength=10)
        pair_count = defaultdict(int)
        pair_conf = defaultdict(list)
        for i in np.where(correct == 0)[0]:
            t, p = int(true_l[i]), int(pred_l[i])
            pair_count[(t, p)] += 1
            pair_conf[(t, p)].append(float(probs[i, p]))   # 错误样本被预测类别的概率（逐样本）
        per_seed[seed] = {"support": support.tolist(),
                          "n_errors": int((correct == 0).sum()),
                          "pair_counts": {f"{t}->{p}": c for (t, p), c in sorted(pair_count.items())},
                          "pair_mean_conf": {f"{t}->{p}": float(np.mean(v))
                                             for (t, p), v in sorted(pair_conf.items())}}
        per_seed[seed]["_raw"] = (true_l, pred_l, correct, probs)

    # 3 seed 合并口径：计数均值±std（跨 seed），错误率 = 计数 / 该类 support（同一测试集，support 逐 seed 相同）
    all_pairs = sorted({k for s in SEEDS for k in per_seed[s]["pair_counts"]},
                       key=lambda k: (-np.mean([per_seed[s]["pair_counts"].get(k, 0) for s in SEEDS]), k))
    pair_stats = []
    for key in all_pairs:
        t, p = map(int, key.split("->"))
        counts = np.array([per_seed[s]["pair_counts"].get(key, 0) for s in SEEDS], dtype=float)
        sup = np.mean([per_seed[s]["support"][t] for s in SEEDS])
        confs = [per_seed[s]["pair_mean_conf"].get(key) for s in SEEDS if key in per_seed[s]["pair_mean_conf"]]
        pair_stats.append({
            "pair": key, "true": t, "pred": p,
            "count_mean": float(counts.mean()), "count_pstdev": float(counts.std(ddof=0)),
            "counts_per_seed": {str(s): per_seed[s]["pair_counts"].get(key, 0) for s in SEEDS},
            "error_rate_per_true_support": float(counts.mean() / sup),
            "mean_confidence_on_errors": float(np.mean(confs)) if confs else None,
        })

    # 10×10 错误率矩阵（行=true，列=pred，按行 support 归一，3 seed 计数平均）
    rate = np.zeros((10, 10))
    for stat in pair_stats:
        rate[stat["true"], stat["pred"]] = stat["error_rate_per_true_support"]
    support_mean = np.mean([per_seed[s]["support"] for s in SEEDS], axis=0)

    out = {
        "source": "e6_test_predictions_seed{404,505,606}.csv（2026-10-30 补充评估产物，只读分析）",
        "per_seed": {str(s): {k: v for k, v in per_seed[s].items() if k != "_raw"} for s in SEEDS},
        "test_support_mean_per_class": support_mean.tolist(),
        "pair_stats_sorted_by_count": pair_stats,
        "focus_directional_pairs": [p for p in pair_stats
                                    if (p["true"], p["pred"]) in FOCUS],
        "n_test_per_seed": 10000, "n_total_3seed": 30000,
    }
    (OUT / "e6_confusion_pairs.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 图 S5 ----
    fig = plt.figure(figsize=(7.2, 6.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])
    top = [p for p in pair_stats if p["count_mean"] >= 3][:12][::-1]

    # A：高频混淆对
    axA = fig.add_subplot(gs[0, 0])
    labels = [f'{p["true"]}→{p["pred"]}' for p in top]
    means = [p["count_mean"] for p in top]
    stds = [p["count_pstdev"] for p in top]
    axA.barh(range(len(top)), means, xerr=stds, color=OKABE["blue"], alpha=0.85,
             error_kw={"lw": 0.9, "capsize": 2})
    for i, p in enumerate(top):
        for j, s in enumerate(SEEDS):
            axA.plot(p["counts_per_seed"][str(s)], i, "o", ms=2.6, color=OKABE["vermillion"],
                     alpha=0.8, zorder=3)
    axA.set_yticks(range(len(top)), labels, fontsize=8.0)
    axA.set_xlabel("errors per seed (mean ± std; dots = seeds)", fontsize=9.0)
    axA.set_title("(A) top confusion pairs (top 12)", fontsize=9.5, loc="left")
    axA.tick_params(axis="x", labelsize=8.0)
    axA.set_axisbelow(True)
    axA.grid(axis="x", lw=0.4, alpha=0.5)

    # B：10×10 错误率热图
    axB = fig.add_subplot(gs[0, 1])
    im = axB.imshow(rate * 100, cmap="Blues", vmin=0, vmax=max(rate[rate > 0].max() * 100, 1.0))
    for t in range(10):
        for p in range(10):
            if t == p or rate[t, p] < 0.0004:
                continue
            axB.text(p, t, f"{rate[t, p] * 100:.2f}", ha="center", va="center",
                     fontsize=6.4, color="#111111")
    for (t, p) in FOCUS:
        axB.add_patch(plt.Rectangle((p - 0.5, t - 0.5), 1, 1, fill=False,
                                    edgecolor=OKABE["vermillion"], lw=1.4))
    axB.set_xticks(range(10), range(10), fontsize=8.0)
    axB.set_yticks(range(10), range(10), fontsize=8.0)
    axB.set_xlabel("predicted class", fontsize=9.0)
    axB.set_ylabel("true class", fontsize=9.0)
    axB.set_title("(B) error rate (% / class support)", fontsize=9.5, loc="left")
    fig.colorbar(im, ax=axB, fraction=0.046, pad=0.03)

    # C：重点方向对的错误置信度（逐样本）
    axC = fig.add_subplot(gs[1, :])
    data, labels_c = [], []
    for (t, p) in FOCUS:
        pooled_conf = []
        for s in SEEDS:
            true_l, pred_l, correct, probs = per_seed[s]["_raw"]
            m = (true_l == t) & (pred_l == p) & (correct == 0)
            pooled_conf.extend(probs[m][:, p].tolist())
        data.append(pooled_conf)
        labels_c.append(f"{t}→{p}\n(n={len(pooled_conf)})")
    bp = axC.boxplot(data, tick_labels=labels_c, showfliers=False, patch_artist=True,
                     medianprops={"color": "#111111", "lw": 1.1},
                     boxprops={"lw": 0.8}, whiskerprops={"lw": 0.8}, capprops={"lw": 0.8})
    for patch, c in zip(bp["boxes"], [OKABE["blue"], OKABE["sky"], OKABE["vermillion"],
                                      OKABE["orange"], OKABE["green"], OKABE["purple"]]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    rng = np.random.default_rng(2026)
    for i, d in enumerate(data, 1):
        if len(d) > 1:
            jitter = rng.uniform(-0.13, 0.13, size=len(d))
            axC.plot(i + jitter, d, ".", ms=1.6, color="#333333", alpha=0.45, zorder=2)
    axC.set_ylabel("softmax prob. of predicted class (errors)", fontsize=9.0)
    axC.set_title("(C) focus directional pairs — confidence when wrong "
                  "(3-seed pooled, n = error count)", fontsize=9.5, loc="left")
    axC.tick_params(axis="x", labelsize=8.0)
    axC.tick_params(axis="y", labelsize=8.0)
    axC.set_ylim(0, 1.02)

    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"fig_s5_error_pairs.{ext}", dpi=300)
    print(f"[done] pairs={len(pair_stats)} -> {OUT / 'e6_confusion_pairs.json'}")
    for s in SEEDS:
        per_seed[s].pop("_raw")


if __name__ == "__main__":
    main()
