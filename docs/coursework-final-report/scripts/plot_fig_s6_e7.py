# -*- coding: utf-8 -*-
"""图 S6（E7）：[256] 组件复验排序对照 + 激活×初始化因子热图。数据源为
derived_data/e7_validation_on_256.json（build_e7_analysis.py 生成，只读）。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]   # 报告根目录（脚本位于 scripts/ 子目录）
DATA = json.loads((HERE / "derived_data" / "e7_validation_on_256.json").read_text(encoding="utf-8"))
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OKABE = {"blue": "#0072B2", "vermillion": "#D55E00", "gray": "#999999"}

BLOCKS = [
    ("A", ["relu", "tanh", "sigmoid"], "E7-A activation"),
    ("B", ["lambda0", "lambda0.0001", "lambda0.001"], "E7-B L2"),
    ("C", ["fixed_no_es", "fixed_es", "step_no_es", "step_es"], "E7-C strategy"),
]
LABEL = {"relu": "ReLU", "tanh": "tanh", "sigmoid": "sigmoid",
         "lambda0": "λ=0", "lambda0.0001": "λ=1e-4", "lambda0.001": "λ=1e-3",
         "fixed_no_es": "fixed/noES", "fixed_es": "fixed/ES",
         "step_no_es": "step/noES", "step_es": "step/ES"}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True,
                               gridspec_kw={"width_ratios": [1.45, 1.0]})

# ---- A：[128] vs [256] 逐组均值（主指标 best val loss）----
x, ticklabels, seps, centers = [], [], [], []
x0 = 0.0
for block, names, blabel in BLOCKS:
    start = x0
    for name in names:
        x.append(x0)
        ticklabels.append(LABEL[name])
        x0 += 1.0
    seps.append((x0 - 0.5, blabel))
    centers.append((start + x0 - 1.0) / 2.0)
    x0 += 0.7
for i, xi in enumerate(x):
    block, name = [b for b, ns, _ in BLOCKS if any(n == ticklabels[i] or LABEL[n] == ticklabels[i] for n in ns)][0], ticklabels[i]
    name_key = [n for n in LABEL if LABEL[n] == name][0]
    ref = DATA["frozen_reference_128"][f"{block}:{name_key}"]["best_validation_loss"]
    new = DATA["groups_256"][f"{block}:{name_key}"]["best_validation_loss"]
    axA.plot([xi, xi], [ref["mean"], new["mean"]], color=OKABE["gray"], lw=0.8, zorder=1)
    axA.errorbar(xi - 0.09, ref["mean"], yerr=ref["std"], fmt="o", ms=3.6, lw=0.8,
                 color=OKABE["blue"], capsize=2, zorder=3,
                 label="[128] (E3/E4/E5 frozen)" if i == 0 else None)
    axA.errorbar(xi + 0.09, new["mean"], yerr=new["pstdev_ddof0"], fmt="D", ms=3.4, lw=0.8,
                 color=OKABE["vermillion"], capsize=2, zorder=3,
                 label="[256] (E7 revalidation)" if i == 0 else None)
for pos, blabel in seps[:-1]:
    axA.axvline(pos + 0.35, color="#CCCCCC", lw=0.6, ls=":")
axA.set_xticks(x, ticklabels, rotation=30, ha="right", fontsize=7.5)
axA.set_ylabel("mean best validation loss", fontsize=9.0)
axA.set_title("(A) E7-A/B/C: component ranking on [256]", fontsize=9.5, loc="left", pad=10)
axA.legend(fontsize=7.2, loc="upper left", frameon=False)
axA.tick_params(axis="y", labelsize=8.0)
all_means = [DATA["frozen_reference_128"][f"{b}:{n}"]["best_validation_loss"]["mean"]
             for b, ns, _ in BLOCKS for n in ns] + \
            [DATA["groups_256"][f"{b}:{n}"]["best_validation_loss"]["mean"]
             for b, ns, _ in BLOCKS for n in ns]
axA.set_ylim(None, max(all_means) * 1.09)
axA.grid(axis="y", lw=0.4, alpha=0.5)
axA.set_axisbelow(True)

# ---- B：E7-D 因子热图（[128]，显式初始化）+ E3 auto 列 ----
acts = ["relu", "tanh", "sigmoid"]
inits = ["he_normal", "xavier_normal", "auto(E3)"]
grid = np.full((3, 3), np.nan)
for r, act in enumerate(acts):
    for c, init in enumerate(inits):
        if init == "auto(E3)":
            grid[r, c] = DATA["Q2_auto_diagonal_reference_128"][f"{act}+auto(E3,[128])"]
        else:
            grid[r, c] = DATA["Q2_factorial_128_explicit_init"][f"{act}+{init}"]["best_validation_loss"]["mean"]
im = axB.imshow(grid * 100, cmap="Blues")
gmin, gmax = np.nanmin(grid) * 100, np.nanmax(grid) * 100
for r in range(3):
    for c in range(3):
        norm = (grid[r, c] * 100 - gmin) / (gmax - gmin)   # 与 cmap 线性映射一致
        axB.text(c, r, f"{grid[r, c] * 100:.2f}", ha="center", va="center", fontsize=8.2,
                 color="#FFFFFF" if norm > 0.55 else "#111111",
                 fontweight="bold" if c == 2 or grid[r, c] == np.nanmin(grid) else "normal")
axB.add_patch(plt.Rectangle((2 - 0.5, 0 - 0.5), 1, 1, fill=False, edgecolor=OKABE["vermillion"], lw=1.3))
axB.set_xticks(range(3), ["He normal", "Xavier normal", "auto (E3)"], fontsize=7.8)
axB.set_yticks(range(3), ["ReLU", "tanh", "sigmoid"], fontsize=8.0)
axB.set_title("(B) E7-D factorial ([128])", fontsize=9.5, loc="left")
axB.set_xlabel("initializer (values: mean best val loss x100)", fontsize=8.4)

for ext in ("png", "svg"):
    fig.savefig(FIG_DIR / f"fig_s6_e7_validation.{ext}", dpi=300)
print(f"[done] {FIG_DIR / 'fig_s6_e7_validation.png'}")
