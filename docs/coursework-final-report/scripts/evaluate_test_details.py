# -*- coding: utf-8 -*-
"""Supplemental test-set evaluation for report section 12.5 (disclosed forward pass).

The course assignment asks for ROC/AUC and test-sample displays.  Producing
per-sample outputs requires one more forward pass over the sealed test set,
pre-declared as a supplemental evaluation and disclosed in report 13.2.8.
This script NEVER retrains and never changes any frozen artifact:

1. rebuild each E6 model exactly ([256]/relu/auto, seeds 404/505/606) from its
   frozen best_model.npz via the project's own safe loader;
2. forward-propagate the official 10,000-sample test set once per seed;
3. REPRODUCE the sealed summary metrics (loss / accuracy / confusion matrix /
   per-class P-R-F1) and assert bit-level agreement -- this doubles as a
   determinism check of the original evaluation;
4. emit per-sample predictions, one-vs-rest ROC/AUC (pure NumPy, tie-corrected
   Mann-Whitney), and two figures: fig_s3 (ROC) and fig_s4 (test samples).

Outputs (all under derived_data/ and figures/): nothing else is touched.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.mnist_data import load_idx_images, load_idx_labels, preprocess_images  # noqa: E402
from src.mlp import MLP  # noqa: E402
from src.losses import softmax  # noqa: E402
from src.training import load_checkpoint  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
DERIVED = HERE / "derived_data"
FIG_DIR = HERE / "figures"
SEEDS = [404, 505, 606]
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9",
         "#E69F00", "#000000", "#F0E442", "#8B4513", "#999999"]

for d in (DERIVED, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- test set (official 10,000; never split, never augmented) ---------------
raw = ROOT / "data" / "mnist" / "raw"
X = preprocess_images(load_idx_images(raw / "t10k-images-idx3-ubyte.gz"))
y = load_idx_labels(raw / "t10k-labels-idx1-ubyte.gz").astype(np.int64)
assert X.shape == (10000, 784) and y.shape == (10000,), (X.shape, y.shape)


def auc_binary(scores: np.ndarray, pos: np.ndarray) -> float:
    """Tie-corrected Mann-Whitney AUC for one-vs-rest scores."""
    n = scores.size
    n_pos = int(pos.sum())
    n_neg = n - n_pos
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)  # average rank of the tie group
        i = j
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_points(scores: np.ndarray, pos: np.ndarray):
    """(fpr, tpr) arrays from descending unique thresholds."""
    order = np.argsort(-scores, kind="mergesort")
    lab = pos[order].astype(np.int64)
    tp = np.cumsum(lab)
    fp = np.cumsum(1 - lab)
    tpr = np.concatenate(([0.0], tp / tp[-1]))
    fpr = np.concatenate(([0.0], fp / fp[-1]))
    return fpr, tpr


report = {"method": "second approved forward pass over the sealed test set "
                    "(author approval 2026-10-30); no retraining; frozen "
                    "best_model.npz; reproduction of sealed summary asserted",
          "seeds": {}}

for seed in SEEDS:
    run_dir = ROOT / "outputs" / "p5.3" / "E6_final" / f"e6_final_seed{seed}"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    model = MLP(784, [256], 10, "relu", "auto", seed=seed)
    model.params = load_checkpoint(run_dir / "best_model.npz", model.params)
    logits = model.forward(X)
    probs = softmax(logits)
    pred = probs.argmax(axis=1).astype(np.int64)
    eps = 1e-12
    loss = float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], eps, 1.0))))

    conf = np.bincount(y * 10 + pred, minlength=100).reshape(10, 10).astype(np.int64)
    conf_match = bool(np.array_equal(conf, np.asarray(summary["test_confusion_matrix"], dtype=np.int64)))
    acc = float(np.mean(pred == y))
    acc_match = bool(abs(acc - float(summary["test_accuracy"])) <= 1e-12)
    loss_match = bool(abs(loss - float(summary["test_loss"])) <= 1e-9)

    # per-class P/R/F1 recomputed from the confusion matrix
    per_class = []
    for c in range(10):
        tp = float(conf[c, c])
        fp = float(conf[:, c].sum() - tp)
        fn = float(conf[c, :].sum() - tp)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per_class.append({"class": c, "precision": p, "recall": r, "f1": f})
    rec_match = all(
        abs(per_class[c]["f1"] - float(summary["test_per_class_metrics"][c]["f1"])) <= 1e-12
        for c in range(10)
    )

    if not (conf_match and acc_match and loss_match and rec_match):
        raise SystemExit(f"REPRODUCTION MISMATCH for seed {seed}: conf={conf_match} "
                         f"acc={acc_match} loss={loss_match} perclass={rec_match} -- aborting")

    # one-vs-rest ROC / AUC
    roc, aucs = {}, {}
    for c in range(10):
        pos = (y == c)
        aucs[c] = auc_binary(probs[:, c], pos)
        fpr, tpr = roc_points(probs[:, c], pos)
        roc[c] = (fpr.tolist(), tpr.tolist())
    macro_auc = float(np.mean([aucs[c] for c in range(10)]))

    # per-sample predictions CSV (probabilities display-rounded to 6 dp)
    csv_path = DERIVED / f"e6_test_predictions_seed{seed}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "true_label", "pred_label", "correct"] + [f"p_{c}" for c in range(10)])
        for i in range(len(y)):
            w.writerow([i, int(y[i]), int(pred[i]), int(pred[i] == y[i])]
                       + [f"{probs[i, c]:.6f}" for c in range(10)])

    report["seeds"][str(seed)] = {
        "accuracy_recomputed": acc, "loss_recomputed": loss,
        "confusion_matrix_match": conf_match, "accuracy_match": acc_match,
        "loss_match": loss_match, "per_class_match": rec_match,
        "per_class_auc_ovr": {str(c): aucs[c] for c in range(10)},
        "macro_auc_ovr": macro_auc,
    }
    print(f"seed {seed}: acc={acc:.6f} loss={loss:.6f} conf_match={conf_match} "
          f"macro_auc={macro_auc:.6f}")

# fig_s3: one combined 3-panel figure (one panel per seed, 10 OVR curves each).
# AUC ~0.9995 pushes every curve onto the top-left corner, so each panel adds
# a zoomed inset of the corner region (FPR<=0.05, TPR>=0.95) -- without it the
# curves are invisible at report scale.
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), constrained_layout=True)
for ax, seed in zip(axes, SEEDS):
    info = report["seeds"][str(seed)]
    run_dir = ROOT / "outputs" / "p5.3" / "E6_final" / f"e6_final_seed{seed}"
    model = MLP(784, [256], 10, "relu", "auto", seed=seed)
    model.params = load_checkpoint(run_dir / "best_model.npz", model.params)
    probs = softmax(model.forward(X))
    curves = [roc_points(probs[:, c], (y == c)) for c in range(10)]
    for c, (fpr, tpr) in enumerate(curves):
        ax.plot(fpr, tpr, lw=0.9, color=OKABE[c])
    ax.plot([0, 1], [0, 1], "--", lw=0.7, color="gray")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
    ax.set_title(f"seed {seed}\nmacro AUC={info['macro_auc_ovr']:.6f}", fontsize=8.5)
    ax.tick_params(labelsize=6.8)
    axins = ax.inset_axes([0.38, 0.05, 0.60, 0.50])
    for c, (fpr, tpr) in enumerate(curves):
        axins.plot(fpr, tpr, lw=1.0, color=OKABE[c])
    axins.set_xlim(0, 0.05)
    axins.set_ylim(0.95, 1.0)
    axins.tick_params(labelsize=5.5)
    axins.grid(True, lw=0.25, alpha=0.5)
    ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.7)
axes[0].set_ylabel("true positive rate", fontsize=8.0)
axes[1].set_xlabel("false positive rate (one-vs-rest; inset = FPR<=0.05 / TPR>=0.95 corner)",
                   fontsize=8.0)
fig.savefig(FIG_DIR / "fig_s3_roc.png", dpi=300)
fig.savefig(FIG_DIR / "fig_s3_roc.svg")
plt.close(fig)

# ---- fig_s4: misclassified test samples (seed 404) -------------------------
seed = 404
pred_csv = np.genfromtxt(DERIVED / f"e6_test_predictions_seed{seed}.csv", delimiter=",",
                         skip_header=1, usecols=(1, 2), dtype=np.int64)
true_lab, pred_lab = pred_csv[:, 0], pred_csv[:, 1]
wrong = np.where(true_lab != pred_lab)[0]
pairs = {}
for i in wrong:
    pairs.setdefault((int(true_lab[i]), int(pred_lab[i])), []).append(i)

# recompute probs for seed 404 to rank "most confident mistakes"
run_dir = ROOT / "outputs" / "p5.3" / "E6_final" / f"e6_final_seed{seed}"
model = MLP(784, [256], 10, "relu", "auto", seed=seed)
model.params = load_checkpoint(run_dir / "best_model.npz", model.params)
probs404 = softmax(model.forward(X))

pair_counts = sorted(((len(v), k) for k, v in pairs.items()), reverse=True)[:6]
picked = []
for _, (t, p) in pair_counts:
    idxs = pairs[(t, p)]
    idxs = sorted(idxs, key=lambda i: -probs404[i, p])[:2]
    picked += [(i, t, p) for i in idxs]

raw_images = load_idx_images(raw / "t10k-images-idx3-ubyte.gz")  # (10000,28,28) uint8
fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.6), constrained_layout=True)
for ax, (i, t, p) in zip(axes.ravel(), picked):
    ax.imshow(raw_images[i], cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"true {t} -> pred {p}  (p={probs404[i, p]:.2f})", fontsize=8.5)
    ax.axis("off")
fig.suptitle("seed 404: two most confident mistakes in each of the top-6 confused pairs",
             fontsize=9.5)
fig.savefig(FIG_DIR / "fig_s4_test_samples.png", dpi=300)
fig.savefig(FIG_DIR / "fig_s4_test_samples.svg")
plt.close(fig)

(DERIVED / "e6_roc_auc.json").write_text(
    json.dumps({s: report["seeds"][s]["per_class_auc_ovr"] | {"macro": report["seeds"][s]["macro_auc_ovr"]}
                for s in report["seeds"]}, indent=2), encoding="utf-8")
(DERIVED / "e6_supplemental_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("[done] predictions CSV x3, ROC/AUC, fig_s3, fig_s4 written; "
      "reproduction of sealed summary: 3/3 seeds bit-matched")
