# -*- coding: utf-8 -*-
"""图 S7（E8）：4 个预注册主对比的精确置换检验零分布与观测差。数据源为
derived_data/e8_significance_tests.json（analyze_e8.py 生成；零分布由逐 seed 值
就地完全枚举重算，与检验脚本同一实现语义）。"""
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]   # 报告根目录（脚本位于 scripts/ 子目录）
DATA = json.loads((HERE / "derived_data" / "e8_significance_tests.json").read_text(encoding="utf-8"))
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OKABE = {"blue": "#0072B2", "vermillion": "#D55E00"}

TITLE = {
    "C1_momentum_vs_adam": "C1 momentum lr=0.05 vs adam lr=0.003",
    "C2_arch128_vs_256": "C2 [128] vs [256]",
    "C3_relu_vs_tanh": "C3 ReLU vs tanh (auto init)",
    "C4_lambda0_vs_0.0001": "C4 L2 lambda 0 vs 1e-4",
}

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
for ax, comp in zip(axes.ravel(), DATA["primary_comparisons"]):
    vals = np.array(comp["arm1_stats"]["values"] + comp["arm2_stats"]["values"])
    n1 = len(comp["arm1_stats"]["values"])
    n = len(vals)
    obs = comp["test"]["mean_diff_obs"]
    combos = np.array(list(itertools.combinations(range(n), n1)), dtype=np.int64)
    mask = np.zeros((combos.shape[0], n), dtype=bool)
    mask[np.arange(combos.shape[0])[:, None], combos] = True
    sums = (vals[None, :] * mask).sum(axis=1)
    diffs = sums / n1 - (vals.sum() - sums) / (n - n1)
    scale = 1e3
    ax.hist(diffs * scale, bins=80, density=True, color=OKABE["blue"], alpha=0.75)
    ax.axvline(obs * scale, color=OKABE["vermillion"], lw=1.4)
    ax.axvline(-obs * scale, color=OKABE["vermillion"], lw=1.0, ls=":")
    sig = comp["significant_at_familywise_0.05"]
    ax.set_title(f"{TITLE[comp['comparison']]}\n"
                 rf"$\Delta_{{obs}}$={obs * scale:+.2f}$\times$1e-3, "
                 f"p_holm={comp['p_holm_adjusted']:.4f}"
                 f"{'  (*)' if sig else ''}",
                 fontsize=8.2, loc="left")
    ax.set_xlabel("permutation distribution of mean-difference (x1e-3)", fontsize=8.2)
    ax.tick_params(labelsize=7.6)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
fig.suptitle("E8 exact permutation tests, C(20,10)=184,756 splits, Holm alpha=0.05 "
             "(solid: observed, dotted: mirrored)", fontsize=8.0)
for ext in ("png", "svg"):
    fig.savefig(FIG_DIR / f"fig_s7_e8_permutation.{ext}", dpi=300)
print(f"[done] {FIG_DIR / 'fig_s7_e8_permutation.png'}")
