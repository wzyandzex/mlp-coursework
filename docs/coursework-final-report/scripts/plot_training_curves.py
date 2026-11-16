# -*- coding: utf-8 -*-
"""P6.2-R2 training loss curves (fig_s2), read-only from history.csv.

Course requirement: "输出训练过程中损失值随迭代的变化曲线" -- all 66 P5.3 runs
already write per-epoch history.csv; this figure surfaces them.

Panel A: E6 final config, 3 model seeds -- train/val cross-entropy vs epoch.
Panel B: E1 confirmation, 9 runs (3 configs x 3 seeds) -- val loss vs epoch.
No aggregated statistic is invented here; every line is one run's own log.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]   # 报告根目录（脚本位于 scripts/ 子目录）
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[4]

OKABE = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73"}
SEED_STYLE = {404: ("-", "o"), 505: ("--", "s"), 606: (":", "^")}
CONFIG_COLOR = {"momentum": OKABE["vermillion"], "sgd": OKABE["blue"], "adam": OKABE["green"]}
CONFIG_LABEL = {"momentum": "momentum_sgd lr=0.05", "sgd": "sgd lr=0.2", "adam": "adam lr=0.003"}


def read_history(path: Path):
    epochs, train, val = [], [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
    return epochs, train, val


fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.6), constrained_layout=True)

# ---- panel A: E6 ------------------------------------------------------------
ax = axes[0]
for seed in (404, 505, 606):
    ep, tr, va = read_history(ROOT / "outputs/p5.3/E6_final" / f"e6_final_seed{seed}" / "history.csv")
    ls, mk = SEED_STYLE[seed]
    ax.plot(ep, tr, ls, color=OKABE["blue"], lw=1.4, marker=mk, markersize=3,
            markevery=5, label=f"seed {seed} train")
    ax.plot(ep, va, ls, color=OKABE["vermillion"], lw=1.4, marker=mk, markersize=3,
            markevery=5, label=f"seed {seed} val")
ax.set_xlabel("epoch", fontsize=9.0)
ax.set_ylabel("cross-entropy loss", fontsize=9.0)
ax.set_title("E6 final config [256]/ReLU/MomentumSGD lr=0.05: train (blue) vs validation (red) loss",
             fontsize=9.5)
ax.set_xticks(range(0, 31, 5))
ax.tick_params(labelsize=8.0)
ax.legend(fontsize=7.2, ncol=3, loc="upper right")
ax.grid(True, lw=0.3, alpha=0.4)

# ---- panel B: E1 confirmation ----------------------------------------------
ax = axes[1]
config_seen = set()
for run_dir in sorted((ROOT / "outputs/p5.3/E1_confirmation").iterdir()):
    if not run_dir.is_dir():
        continue
    name = run_dir.name
    key = "adam" if "adam" in name else ("momentum" if "momentum" in name else "sgd")
    ep, _, va = read_history(run_dir / "history.csv")
    label = CONFIG_LABEL[key] if key not in config_seen else None
    config_seen.add(key)
    ax.plot(ep, va, "-", color=CONFIG_COLOR[key], lw=1.1, alpha=0.85, label=label)
ax.set_xlabel("epoch", fontsize=9.0)
ax.set_ylabel("validation loss", fontsize=9.0)
ax.set_title("E1 confirmation: validation loss per run (3 configs x seeds 101/202/303, thin lines)",
             fontsize=9.5)
ax.set_xticks(range(0, 31, 5))
ax.tick_params(labelsize=8.0)
ax.legend(fontsize=7.6, loc="upper right")
ax.grid(True, lw=0.3, alpha=0.4)

for ext in ("png", "svg"):
    out = FIG_DIR / f"fig_s2_training_curves.{ext}"
    fig.savefig(out, dpi=300)
    assert out.is_file() and out.stat().st_size > 0, out
plt.close(fig)
print("[fig] fig_s2_training_curves: png+svg written")
