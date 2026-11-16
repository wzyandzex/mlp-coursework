# -*- coding: utf-8 -*-
"""P6.2-R1 network / data-flow schematic for report section 3.

This is a structural illustration only: it reads no experiment data and does
not change any result figure.  Output: figures/fig_s1_network_dataflow.png/.svg
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OKABE = {
    "blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
    "sky": "#56B4E9", "orange": "#E69F00", "gray": "#999999",
}

fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.grid(False)

W, H = 22.0, 26.0
xs = [1.0, 26.0, 51.0, 76.0]
y_top, y_bot = 61.0, 10.0

top = [
    ("Input X\n(m x 784)", OKABE["sky"]),
    ("Affine 1\nZ1 = XW1 + b1\nA1 = ReLU(Z1)\n(m x 256)", OKABE["blue"]),
    ("Affine 2\nZ2 = A1W2 + b2\nlogits Z2 (m x 10)", OKABE["green"]),
    ("Softmax + CE\np = softmax(Z2)\nJ = -(1/m) SUM log p_y", OKABE["vermillion"]),
]
bot = [
    ("dW1 = X^T Delta1\ndb1 = SUM Delta1", OKABE["gray"]),
    ("Delta1 = (Delta2 W2^T)\n* ReLU'(Z1)", OKABE["gray"]),
    ("dW2 = A1^T Delta2\ndb2 = SUM Delta2", OKABE["orange"]),
    ("dlogits\nDelta2 = (P - Y)/m", OKABE["orange"]),
]

for x, (txt, color) in zip(xs, top):
    ax.add_patch(FancyBboxPatch((x, y_top), W, H,
                                boxstyle="round,pad=0.3,rounding_size=0.8",
                                fc=color, ec="black", lw=0.8, alpha=0.25))
    ax.text(x + W / 2, y_top + H / 2, txt, ha="center", va="center", fontsize=7.9)

for x, (txt, color) in zip(xs, bot):
    ax.add_patch(FancyBboxPatch((x, y_bot), W, H,
                                boxstyle="round,pad=0.3,rounding_size=0.8",
                                fc=color, ec="black", lw=0.8, alpha=0.20,
                                linestyle="--"))
    ax.text(x + W / 2, y_bot + H * 0.62, txt, ha="center", va="center", fontsize=7.5)

for i in range(3):
    ax.add_patch(FancyArrowPatch((xs[i] + W, y_top + H / 2), (xs[i + 1], y_top + H / 2),
                                 arrowstyle="-|>", mutation_scale=12, color="black", lw=1.0))
for i in range(3, 0, -1):
    ax.add_patch(FancyArrowPatch((xs[i], y_bot + H * 0.26), (xs[i - 1] + W, y_bot + H * 0.26),
                                 arrowstyle="-|>", mutation_scale=12, color="black", lw=1.0))

ax.add_patch(FancyArrowPatch((xs[3] + W / 2, y_top), (xs[3] + W / 2, y_bot + H),
                             arrowstyle="-|>", mutation_scale=12, color="black",
                             lw=1.0, linestyle=(0, (4, 3))))
ax.text(xs[3] + W / 2 - 1.5, (y_top + y_bot + H) / 2, "loss\ngradient", fontsize=7.0,
        ha="right", va="center")

ax.text(50, 97, "Forward pass (rows = samples; every hidden layer is affine + activation)",
        ha="center", va="center", fontsize=8.8, fontweight="bold")
ax.text(50, 2.5, "Backward pass (chain rule; gradients are batch-averaged; no parameter update here)",
        ha="center", va="center", fontsize=8.4)

for ext in ("png", "svg"):
    out = FIG_DIR / f"fig_s1_network_dataflow.{ext}"
    fig.savefig(out, dpi=300)
    assert out.is_file() and out.stat().st_size > 0, out
plt.close(fig)
print("[fig] fig_s1_network_dataflow: png+svg written")
