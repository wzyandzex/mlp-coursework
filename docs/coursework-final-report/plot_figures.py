# -*- coding: utf-8 -*-
"""P6.2 chart generation for the mlp-coursework final report.

Run with the isolated report venv (matplotlib only; project source is NOT imported):

    <report-venv>/python.exe outputs/p5.3/P6.2/plot_figures.py

Everything is READ-ONLY with respect to existing experiment artifacts:
the script only parses aggregator JSON and per-run JSON/CSV written by the
training pipeline, recomputes group statistics with the Python standard
library, writes full-precision derived data into derived_data/, and renders
fixed-layout figures into figures/.  It never imports src.*, never touches
best_model.npz for prediction, and never recomputes model outputs.

Data sources per figure (all under outputs/p5.3/ unless noted):

  fig01  experiment flow ...... structural summary of stages/run counts,
         cross-checked against outputs/p5.3/<stage>/ directory listings and
         each run's config.json; E0 numbers read from
         outputs/mnist_baseline/baseline_30ep_consistency_fix/summary.json.
  fig02  E1 optimizer confirmation ... _aggregate_E1_confirmation/runs.json
         + groups.json; per-run summary.json best_validation_loss.
         Metric: best_validation_loss; bars = group mean; error bars =
         population std across model seeds (statistics.pstdev, ddof=0).
  fig03  E2 architecture ............ _aggregate_E2_architecture/runs.json
         + groups.json; same metric; parameter counts computed from
         hidden_dims via 784*h1+h1 + sum(h_i*h_{i+1}+h_{i+1}) + h_last*10+10.
  fig04  E3 activation+initializer .. _aggregate_E3_activation/runs.json
         + groups.json; metric: best_validation_loss.  NOTE: each bar is a
         (activation, auto-initializer) COMBINATION (ReLU+He normal,
         Tanh/Sigmoid+Xavier normal); the comparison is not attributable to
         the activation function alone.
  fig05  E4 L2 ...................... _aggregate_E4_l2/runs.json + groups.json;
         lambda is plotted as a CATEGORICAL axis (0, 1e-4, 1e-3) on a linear
         scale; lambda=0 is never placed on a log axis.
  fig06  E5 strategy tradeoff ....... _aggregate_E5_training_strategy/runs.json
         + groups.json; left panel: mean best_validation_loss +/- pop std;
         right panel: mean trainer total_seconds per strategy (efficiency
         metric, separate axes; early-stopping groups may finish < 30 epochs).
  fig07  E6 confusion matrices ...... E6_final/<run>/summary.json
         test_confusion_matrix (rows = true class, cols = predicted class).
         Each seed's matrix is row-normalized BY ITS OWN ROW SUMS; raw count
         matrices are never summed across seeds.
  fig08  E6 per-class F1 ............ E6_final/<run>/summary.json
         test_per_class_metrics[].f1; bars = cross-seed mean, error bars =
         population std (ddof=0) across the three model seeds 404/505/606.

Consistency policy: for every group, mean (statistics.fmean) and population
std (statistics.pstdev) are recomputed from per-run summary.json values and
compared against the aggregator's groups.json stats.  Any deviation above
1e-9 aborts the affected figure/data file BEFORE anything is written for it.
"""
import csv
import json
import math
import os
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --------------------------------------------------------------------------
# fixed global style (deterministic; no GUI, no manual tweaking)
# --------------------------------------------------------------------------
FIG_DIR = Path(__file__).resolve().parent / "figures"
DERIVED_DIR = Path(__file__).resolve().parent / "derived_data"
P53 = Path(__file__).resolve().parents[2] / "p5.3"          # outputs/p5.3
REPO_ROOT = Path(__file__).resolve().parents[3]             # repo root

OKABE = {   # colorblind-safe Okabe-Ito palette
    "blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
    "purple": "#CC79A7", "orange": "#E69F00", "sky": "#56B4E9",
    "black": "#000000", "yellow": "#F0E442",
}
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300, "font.family": "DejaVu Sans",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linestyle": "--", "axes.axisbelow": True,
})
TOL = 1e-9
ERROR_BAR_NOTE = "Error bars: population std across model seeds (ddof=0)"


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fail(msg):
    print("[ABORT] " + msg)
    sys.exit(1)


# --------------------------------------------------------------------------
# stage registry + strict structural checks
# --------------------------------------------------------------------------
STAGES = {
    "E1_exploration":     dict(agg="_aggregate_E1_exploration",     n_runs=9,  n_groups=9, test_eval="not_run"),
    "E1_confirmation":    dict(agg="_aggregate_E1_confirmation",    n_runs=9,  n_groups=3, test_eval="not_run"),
    "E2_architecture":    dict(agg="_aggregate_E2_architecture",    n_runs=15, n_groups=5, test_eval="not_run"),
    "E3_activation":      dict(agg="_aggregate_E3_activation",      n_runs=9,  n_groups=3, test_eval="not_run"),
    "E4_l2":              dict(agg="_aggregate_E4_l2",              n_runs=9,  n_groups=3, test_eval="not_run"),
    "E5_training_strategy": dict(agg="_aggregate_E5_training_strategy", n_runs=12, n_groups=4, test_eval="not_run"),
    "E6_final":           dict(agg="_aggregate_E6_final",           n_runs=3,  n_groups=1, test_eval="executed"),
}
TOTAL_RUNS = 66

runs_by_stage = {}      # stage -> {run_name: summary_dict}
cfg_by_stage = {}       # stage -> {run_name: config_dict}

for stage, meta in STAGES.items():
    sdir = P53 / stage
    if not sdir.is_dir():
        fail(f"missing stage directory: {sdir}")
    dirs = sorted(d for d in sdir.iterdir() if d.is_dir())
    if len(dirs) != meta["n_runs"]:
        fail(f"{stage}: expected {meta['n_runs']} run dirs, found {len(dirs)}")
    names = [d.name for d in dirs]
    if len(set(names)) != len(names):
        fail(f"{stage}: duplicate run directory names")
    rep = jload(P53 / meta["agg"] / "report.json")
    if rep.get("total_runs") != meta["n_runs"] or rep.get("n_groups") != meta["n_groups"]:
        fail(f"{stage}: aggregator report.json counts mismatch: {rep}")
    if rep.get("strict") is not True or rep.get("strict_failed") is not False:
        fail(f"{stage}: aggregator strict state unexpected: {rep}")
    counts = rep.get("counts", {})
    for bad in ("incomplete", "invalid", "legacy"):
        if counts.get(bad, 0) != 0:
            fail(f"{stage}: aggregator counts[{bad}] = {counts[bad]} (expected 0)")
    rj = jload(P53 / meta["agg"] / "runs.json")
    agg_names = [r["run_name"] for r in rj["runs"]]
    if set(agg_names) != set(names):
        fail(f"{stage}: runs.json run names differ from disk listing")
    summaries, configs = {}, {}
    for d in dirs:
        s = jload(d / "summary.json")
        c = jload(d / "config.json")
        if s.get("status") != "success":
            fail(f"{stage}/{d.name}: status={s.get('status')}")
        if s.get("test_evaluation") != meta["test_eval"]:
            fail(f"{stage}/{d.name}: test_evaluation={s.get('test_evaluation')} (expected {meta['test_eval']})")
        if c["seeds"]["split_seed"] != 2026 or c["data"]["sample_counts"] != {"train": 20000, "validation": 5000, "test": 10000}:
            fail(f"{stage}/{d.name}: unexpected split/samples")
        rows = sum(1 for _ in open(d / "history.csv", encoding="utf-8")) - 1
        if rows != s["epochs_completed"]:
            fail(f"{stage}/{d.name}: history rows {rows} != epochs_completed {s['epochs_completed']}")
        summaries[d.name] = s
        configs[d.name] = c
    runs_by_stage[stage], cfg_by_stage[stage] = summaries, configs

total_on_disk = sum(len(v) for v in runs_by_stage.values())
if total_on_disk != TOTAL_RUNS:
    fail(f"total P5.3 run count {total_on_disk} != expected {TOTAL_RUNS}")
pilot_dir = P53 / "E1_exploration" / "E1_sgd_lr0p05_seed42"
if not pilot_dir.is_dir():
    fail("A0 pilot run E1_sgd_lr0p05_seed42 missing from E1_exploration")

print(f"[check] {TOTAL_RUNS} runs across {len(STAGES)} stages, all success, "
      f"roles/flags consistent, pilot present exactly once")


# --------------------------------------------------------------------------
# group recompute + reconciliation against groups.json
# --------------------------------------------------------------------------
def stage_groups(stage):
    return jload(P53 / STAGES[stage]["agg"] / "groups.json")["groups"]


def recompute_and_check(stage, metric="best_validation_loss"):
    """Return {group_id: (label, runs, mean, pstdev)} after reconciliation."""
    out = {}
    for g in stage_groups(stage):
        vals = [runs_by_stage[stage][rn][metric] for rn in g["run_paths"]]
        mean, std = statistics.fmean(vals), statistics.pstdev(vals)
        dev = max(abs(mean - g["stats"][metric]["mean"]),
                  abs(std - g["stats"][metric]["std"]))
        if dev > TOL:
            fail(f"{stage} group {g['group_id']}: {metric} recompute dev {dev:.3e} > {TOL}")
        out[g["group_id"]] = dict(n=g["n_runs"], run_paths=list(g["run_paths"]),
                                  mean=mean, std=std, vals=vals)
    return out


def recompute_acc(stage, group_run_paths, metric="best_validation_accuracy"):
    vals = [runs_by_stage[stage][rn][metric] for rn in group_run_paths]
    return statistics.fmean(vals), statistics.pstdev(vals)


FIG_DIR.mkdir(parents=True, exist_ok=True)
DERIVED_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, stem):
    for ext in ("png", "svg"):
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(out)                    # png dpi=300 via rcParams
        if not out.is_file() or out.stat().st_size == 0:
            fail(f"output missing or empty: {out}")
        if ext == "png":
            from PIL import Image
            with Image.open(out) as im:
                im.verify()
            if Image.open(out).size == (0, 0):
                fail(f"degenerate PNG: {out}")
        else:
            try:
                ET.parse(out)
            except ET.ParseError as e:
                fail(f"invalid SVG XML: {out}: {e}")
    plt.close(fig)
    print(f"[fig] {stem}: png+svg written and verified")


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[data] {path.name}: {len(rows)} rows")


# ==========================================================================
# derived data (full precision) — written before/with each figure
# ==========================================================================

# ---- E1 exploration -------------------------------------------------------
e1x_rows = []
for rn, s in runs_by_stage["E1_exploration"].items():
    c = cfg_by_stage["E1_exploration"][rn]
    e1x_rows.append([rn, c["training"]["optimizer"], c["training"]["learning_rate"],
                     c["seeds"]["model_seed"], c["seeds"]["split_seed"],
                     repr(s["best_validation_loss"]), repr(s["best_validation_accuracy"]),
                     repr(s["total_seconds"]), rn == "E1_sgd_lr0p05_seed42",
                     f"outputs/p5.3/E1_exploration/{rn}/summary.json"])
e1x_rows.sort(key=lambda r: (r[1], r[2]))
write_csv(DERIVED_DIR / "e1_exploration_candidates.csv",
          ["run_name", "optimizer", "learning_rate", "model_seed", "split_seed",
           "best_validation_loss", "best_validation_accuracy", "total_seconds",
           "is_a0_pilot", "source_path"], e1x_rows)

# ---- E1 confirmation ------------------------------------------------------
e1c_groups = {}
for gid, g in recompute_and_check("E1_confirmation").items():
    c = cfg_by_stage["E1_confirmation"][g["run_paths"][0]]["training"]
    label = f"{c['optimizer']} lr={c['learning_rate']}"
    m_acc, s_acc = recompute_acc("E1_confirmation", g["run_paths"])
    e1c_groups[label] = dict(g, mean_acc=m_acc, std_acc=s_acc)
    for rn in g["run_paths"]:
        s = runs_by_stage["E1_confirmation"][rn]
write_csv(DERIVED_DIR / "e1_confirmation.csv",
          ["config_label", "n", "mean_best_validation_loss", "pstdev_best_validation_loss",
           "mean_best_validation_accuracy", "pstdev_best_validation_accuracy",
           "run_names", "source"],
          [[lbl, g["n"], repr(g["mean"]), repr(g["std"]), repr(g["mean_acc"]), repr(g["std_acc"]),
            " ".join(sorted(g["run_paths"])),
            "outputs/p5.3/_aggregate_E1_confirmation/groups.json + per-run summary.json"]
           for lbl, g in sorted(e1c_groups.items())])

# ---- E2 architecture ------------------------------------------------------
def param_count(hidden_dims):
    dims = [784] + list(hidden_dims) + [10]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))

e2_groups = {}
rows = []
for gid, g in recompute_and_check("E2_architecture").items():
    hd = cfg_by_stage["E2_architecture"][g["run_paths"][0]]["network"]["hidden_dims"]
    pc = param_count(hd)
    m_acc, s_acc = recompute_acc("E2_architecture", g["run_paths"])
    e2_groups[tuple(hd)] = dict(g, hd=hd, pc=pc, mean_acc=m_acc, std_acc=s_acc)
    for rn in g["run_paths"]:
        rows.append([rn, json.dumps(hd), pc, cfg_by_stage["E2_architecture"][rn]["seeds"]["model_seed"],
                     repr(runs_by_stage["E2_architecture"][rn]["best_validation_loss"]),
                     repr(runs_by_stage["E2_architecture"][rn]["best_validation_accuracy"])])
rows.sort(key=lambda r: (json.loads(r[1]), r[3]))
write_csv(DERIVED_DIR / "e2_architecture.csv",
          ["run_name", "hidden_dims", "param_count", "model_seed",
           "best_validation_loss", "best_validation_accuracy"], rows)

# ---- E3 activation + initializer -----------------------------------------
E3_INIT = {"relu": "He normal", "tanh": "Xavier normal", "sigmoid": "Xavier normal"}
e3_groups = {}
rows = []
for gid, g in recompute_and_check("E3_activation").items():
    act = cfg_by_stage["E3_activation"][g["run_paths"][0]]["network"]["hidden_activation"]
    label = f"{act} + {E3_INIT[act]} (auto)"
    m_acc, s_acc = recompute_acc("E3_activation", g["run_paths"])
    e3_groups[act] = dict(g, label=label, mean_acc=m_acc, std_acc=s_acc)
    for rn in g["run_paths"]:
        rows.append([rn, act, E3_INIT[act], cfg_by_stage["E3_activation"][rn]["seeds"]["model_seed"],
                     repr(runs_by_stage["E3_activation"][rn]["best_validation_loss"]),
                     repr(runs_by_stage["E3_activation"][rn]["best_validation_accuracy"])])
rows.sort(key=lambda r: (r[1], r[3]))
write_csv(DERIVED_DIR / "e3_activation.csv",
          ["run_name", "hidden_activation", "auto_initializer", "model_seed",
           "best_validation_loss", "best_validation_accuracy"], rows)

# ---- E4 L2 ----------------------------------------------------------------
e4_groups = {}
rows = []
for gid, g in recompute_and_check("E4_l2").items():
    lam = cfg_by_stage["E4_l2"][g["run_paths"][0]]["training"]["l2_lambda"]
    label = f"lambda={lam:g}"
    m_acc, s_acc = recompute_acc("E4_l2", g["run_paths"])
    e4_groups[lam] = dict(g, label=label, mean_acc=m_acc, std_acc=s_acc)
    for rn in g["run_paths"]:
        rows.append([rn, repr(cfg_by_stage["E4_l2"][rn]["training"]["l2_lambda"]),
                     cfg_by_stage["E4_l2"][rn]["seeds"]["model_seed"],
                     repr(runs_by_stage["E4_l2"][rn]["best_validation_loss"]),
                     repr(runs_by_stage["E4_l2"][rn]["best_validation_accuracy"])])
rows.sort(key=lambda r: (float(r[1]), r[2]))
write_csv(DERIVED_DIR / "e4_l2.csv",
          ["run_name", "l2_lambda", "model_seed",
           "best_validation_loss", "best_validation_accuracy"], rows)

# ---- E5 strategy ----------------------------------------------------------
e5_groups = {}
rows = []
for gid, g in recompute_and_check("E5_training_strategy").items():
    c = cfg_by_stage["E5_training_strategy"][g["run_paths"][0]]["training"]
    sched = {"none": "fixed"}.get(c["lr_scheduler"], c["lr_scheduler"])
    es = c["early_stopping"]
    label = f"{sched}_{'es' if es else 'no_es'}"
    m_acc, s_acc = recompute_acc("E5_training_strategy", g["run_paths"])
    secs = [runs_by_stage["E5_training_strategy"][rn]["total_seconds"] for rn in g["run_paths"]]
    eps = [runs_by_stage["E5_training_strategy"][rn]["epochs_completed"] for rn in g["run_paths"]]
    e5_groups[label] = dict(g, sched=sched, es=es, mean_acc=m_acc, std_acc=s_acc,
                            mean_seconds=statistics.fmean(secs),
                            epochs_range=(min(eps), max(eps)))
    for rn in g["run_paths"]:
        s = runs_by_stage["E5_training_strategy"][rn]
        rows.append([rn, sched, es, cfg_by_stage["E5_training_strategy"][rn]["seeds"]["model_seed"],
                     repr(s["best_validation_loss"]), repr(s["best_validation_accuracy"]),
                     repr(s["total_seconds"]), s["epochs_completed"]])
rows.sort(key=lambda r: (r[1], r[2], r[3]))
write_csv(DERIVED_DIR / "e5_strategy.csv",
          ["run_name", "lr_scheduler", "early_stopping", "model_seed",
           "best_validation_loss", "best_validation_accuracy",
           "total_seconds", "epochs_completed"], rows)

# ---- E6 final -------------------------------------------------------------
e6 = runs_by_stage["E6_final"]
seed_order = [404, 505, 606]
by_seed = {cfg_by_stage["E6_final"][rn]["seeds"]["model_seed"]: rn for rn in e6}
if sorted(by_seed) != sorted(seed_order):
    fail(f"E6 final seeds {sorted(by_seed)} != expected {seed_order}")

rows = []
for seed in seed_order:
    s = e6[by_seed[seed]]
    rows.append([by_seed[seed], seed, repr(s["best_validation_loss"]), repr(s["best_validation_accuracy"]),
                 repr(s["test_loss"]), repr(s["test_accuracy"]),
                 repr(s["test_macro_avg"]["precision"]), repr(s["test_macro_avg"]["recall"]),
                 repr(s["test_macro_avg"]["f1"]), repr(s["total_seconds"]),
                 s["epochs_completed"], s["stop_reason"]])
rows.sort(key=lambda r: r[1])
write_csv(DERIVED_DIR / "e6_final_by_seed.csv",
          ["run_name", "model_seed", "best_validation_loss", "best_validation_accuracy",
           "test_loss", "test_accuracy", "macro_precision", "macro_recall", "macro_f1",
           "total_seconds", "epochs_completed", "stop_reason"], rows)

# cross-seed descriptive stats, reconciled against the E6 group entry
g6 = stage_groups("E6_final")[0]
e6_agg = {}
for metric in ("test_loss", "test_accuracy"):
    vals = [e6[by_seed[x]][metric] for x in seed_order]
    m, sd = statistics.fmean(vals), statistics.pstdev(vals)
    if abs(m - g6["stats"][metric]["mean"]) > TOL or abs(sd - g6["stats"][metric]["std"]) > TOL:
        fail(f"E6 group {metric} reconciliation failed")
    e6_agg[metric] = (m, sd)
for metric in ("precision", "recall", "f1"):
    vals = [e6[by_seed[x]]["test_macro_avg"][metric] for x in seed_order]
    e6_agg["macro_" + metric] = (statistics.fmean(vals), statistics.pstdev(vals))
print("[check] E6 cross-seed aggregates reconciled with groups.json")

rows = []
class_tables = {}
for seed in seed_order:
    s = e6[by_seed[seed]]
    per_class = s["test_per_class_metrics"]
    cm = s["test_confusion_matrix"]
    if len(cm) != 10 or any(len(r) != 10 for r in cm):
        fail(f"E6 seed {seed}: confusion matrix is not 10x10")
    if sum(sum(r) for r in cm) != s["sample_counts"]["test"]:
        fail(f"E6 seed {seed}: confusion matrix total != test sample count")
    class_tables[seed] = per_class
    for i, m in enumerate(per_class):
        rows.append([by_seed[seed], seed, i, m["support"],
                     repr(m["precision"]), repr(m["recall"]), repr(m["f1"])])
rows.sort(key=lambda r: (r[1], r[2]))
write_csv(DERIVED_DIR / "e6_class_metrics_by_seed.csv",
          ["run_name", "model_seed", "class_index", "support",
           "precision", "recall", "f1"], rows)

cm_data = {"definition": "rows = true class, columns = predicted class; raw counts per single test evaluation of 10,000 samples",
           "normalization_for_display": "each seed's matrix divided by its own row sums (row-normalized rate); raw matrices are never summed across seeds",
           "source": "outputs/p5.3/E6_final/<run>/summary.json:test_confusion_matrix",
           "matrices": {}}
for seed in seed_order:
    cm = e6[by_seed[seed]]["test_confusion_matrix"]
    rn = [[cm[i][j] / sum(cm[i]) for j in range(10)] for i in range(10)]
    cm_data["matrices"][str(seed)] = {"run_name": by_seed[seed], "raw": cm, "row_normalized": rn}
with open(DERIVED_DIR / "e6_confusion_matrices.json", "w", encoding="utf-8") as f:
    json.dump(cm_data, f, indent=1)
print("[data] e6_confusion_matrices.json written")

# E0 historical baseline (P4 product, read-only reference)
e0 = jload(REPO_ROOT / "outputs/mnist_baseline/baseline_30ep_consistency_fix/summary.json")
print(f"[check] E0 baseline: val_loss={e0['best_validation_loss']:.6f} "
      f"test_acc={e0['test_accuracy']}")


# ==========================================================================
# figures
# ==========================================================================

def bar_fig(stem, title, ylabel, items, colors, note=ERROR_BAR_NOTE,
            second_line_labels=None, figsize=(7.6, 4.6)):
    """items: list of (label, mean, std). Categorical x, zero-based y."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    xs = range(len(items))
    ax.bar(xs, [m for _, m, _ in items], yerr=[s for _, _, s in items],
           capsize=5, color=colors, edgecolor="black", linewidth=0.6, width=0.62)
    for x, (_, m, s) in zip(xs, items):
        ax.annotate(f"{m:.6f}", (x, m + s), ha="center", va="bottom", fontsize=8.5,
                    xytext=(0, 2), textcoords="offset points")
    ax.set_ylim(0, max(m + s for _, m, s in items) * 1.14)
    ax.set_xticks(list(xs))
    labels = [lab for lab, _, _ in items]
    if second_line_labels:
        ax.set_xticklabels([f"{lab}\n{sub}" for lab, sub in zip(labels, second_line_labels)])
    else:
        ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if note:
        ax.set_xlabel(note, fontsize=8)
    save_fig(fig, stem)


# ---- fig01: experiment flow ----------------------------------------------
fig, ax = plt.subplots(figsize=(13.6, 4.4), constrained_layout=True)
ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off"); ax.grid(False)
chain = [
    ("E1 exploration", "optimizer x lr grid\n9 runs, seed 42\nSGD/MomSGD/Adam"),
    ("E1 confirmation", "3 configs x 3 seeds\n9 runs\nmom-SGD lr=0.05 wins"),
    ("E2 architecture", "5 archs x 3 seeds\n15 runs\n[256] wins"),
    ("E3 act+init", "3 combos x 3 seeds\n9 runs\nReLU+He wins"),
    ("E4 L2", "3 lambdas x 3 seeds\n9 runs\nlambda=1e-4 wins"),
    ("E5 strategy", "2x2 strategies x 3 seeds\n12 runs\nfixed, no ES wins"),
    ("E6 final", "frozen config\n3 seeds 404/505/606\ntest set, one-shot"),
]
fills = [OKABE["sky"], OKABE["blue"], OKABE["green"], OKABE["green"],
         OKABE["green"], OKABE["green"], OKABE["vermillion"]]
w, h, y0 = 12.4, 17, 22
for i, ((title, body), fill) in enumerate(zip(chain, fills)):
    x = 1.5 + i * 13.6
    box = FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.35",
                         fc=fill, ec="black", lw=0.9, alpha=0.28 if i < 6 else 0.5)
    ax.add_patch(box)
    ax.text(x + w / 2, y0 + h - 3.1, title, ha="center", va="center",
            fontsize=9.3, fontweight="bold")
    ax.text(x + w / 2, y0 + h / 2 - 2.2, body, ha="center", va="center", fontsize=7.6)
    if i:
        ax.add_patch(FancyArrowPatch((x - 1.35, y0 + h / 2), (x + 0.12, y0 + h / 2),
                                     arrowstyle="-|>", mutation_scale=13, color="black", lw=1.1))
ax.add_patch(FancyBboxPatch((6, 2.2), 30, 11, boxstyle="round,pad=0.35",
                            fc="lightgray", ec="black", lw=0.9, linestyle="--"))
ax.text(21, 7.7, "E0 historical baseline (P4 product)\nSGD lr=0.1, [128], seed 42 — read-only reference\nNOT counted among the 66 P5.3 runs",
        ha="center", va="center", fontsize=7.8)
ax.add_patch(FancyArrowPatch((36.6, 7.7), (52, 21.6), arrowstyle="-|>",
                             mutation_scale=13, color="black", lw=1.0,
                             linestyle=(0, (4, 3))))
ax.text(52.5, 4.5, "Total: 66 P5.3 runs (each run counted once; the A0 pilot is one of the 9 E1 exploration runs).\n"
                   "All stages share split seed 2026 and 20000/5000/10000 samples; the official 10k test set is only touched by E6.",
        ha="left", va="center", fontsize=8.2)
ax.set_title("Experiment flow E0 -> E6 (preregistered sequence; validation set drives E1-E5 selection, test set used only by E6)",
             fontsize=11)
save_fig(fig, "fig01_experiment_flow")

# ---- fig02: E1 confirmation ----------------------------------------------
order = ["momentum_sgd lr=0.05", "sgd lr=0.2", "adam lr=0.003"]
items = [(lbl, e1c_groups[lbl]["mean"], e1c_groups[lbl]["std"]) for lbl in order]
bar_fig("fig02_e1_optimizer_confirmation",
        "E1 confirmation: mean best validation loss by optimizer (3 seeds each)",
        "best validation loss (mean over seeds)", items,
        [OKABE["vermillion"], OKABE["blue"], OKABE["green"]])

# ---- fig03: E2 architecture ----------------------------------------------
arch_order = sorted(e2_groups.values(), key=lambda g: g["pc"])
items = [(f"[{','.join(map(str, g['hd']))}]", g["mean"], g["std"]) for g in arch_order]
sub = [f"{g['pc']:,} params" for g in arch_order]
bar_fig("fig03_e2_architecture",
        "E2: mean best validation loss by architecture (3 seeds each)",
        "best validation loss (mean over seeds)", items,
        [OKABE["blue"]] * 3 + [OKABE["vermillion"]] + [OKABE["blue"]],
        second_line_labels=sub)

# ---- fig04: E3 activation + initializer ----------------------------------
act_order = ["relu", "tanh", "sigmoid"]
items = [(e3_groups[a]["label"], e3_groups[a]["mean"], e3_groups[a]["std"]) for a in act_order]
bar_fig("fig04_e3_activation_initialization",
        "E3: activation + auto-initializer COMBINATIONS (3 seeds each)",
        "best validation loss (mean over seeds)", items,
        [OKABE["vermillion"], OKABE["blue"], OKABE["sky"]],
        note="Coupled comparison: activation and auto-initializer change together;\n" + ERROR_BAR_NOTE)

# ---- fig05: E4 L2 ---------------------------------------------------------
lam_order = sorted(e4_groups)
items = [(e4_groups[l]["label"], e4_groups[l]["mean"], e4_groups[l]["std"]) for l in lam_order]
bar_fig("fig05_e4_l2",
        "E4: mean best validation loss by L2 lambda (categorical axis, 3 seeds each)",
        "best validation loss (mean over seeds)", items,
        [OKABE["sky"], OKABE["vermillion"], OKABE["blue"]],
        note="Lambda treated as discrete categories on a linear scale\n(lambda=0 is never log-transformed); " + ERROR_BAR_NOTE)

# ---- fig06: E5 strategy tradeoff -----------------------------------------
strat_order = ["fixed_no_es", "fixed_es", "step_no_es", "step_es"]
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)
colors6 = [OKABE["vermillion"], OKABE["orange"], OKABE["blue"], OKABE["sky"]]
for ax, (vals_fn, ylab, title) in zip(axes, [
        (lambda g: (g["mean"], g["std"]), "best validation loss (mean over seeds)", "Primary metric: validation loss"),
        (lambda g: (g["mean_seconds"], 0.0), "mean trainer seconds (efficiency)", "Efficiency: mean trainer time")]):
    items = [(lbl, *vals_fn(e5_groups[lbl])) for lbl in strat_order]
    ax.bar(range(len(items)), [m for _, m, _ in items],
           yerr=[s for _, _, s in items] if title.startswith("Primary") else None,
           capsize=5, color=colors6, edgecolor="black", linewidth=0.6, width=0.62)
    for x, (_, m, s) in enumerate(items):
        txt = f"{m:.6f}" if title.startswith("Primary") else f"{m:.2f} s"
        ax.annotate(txt, (x, m + (s or 0)), ha="center", va="bottom", fontsize=8.5,
                    xytext=(0, 2), textcoords="offset points")
    top = max(m + (s or 0) for _, m, s in items)
    ax.set_ylim(0, top * 1.14)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([lbl for lbl, _, _ in items], fontsize=8.6)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=10.5)
axes[0].set_xlabel("training strategy (3 seeds each)\n" + ERROR_BAR_NOTE, fontsize=8)
ep = {lbl: e5_groups[lbl]["epochs_range"] for lbl in strat_order}
axes[1].set_xlabel("training strategy\nepochs_completed ranges: " +
                   ", ".join(f"{k} {v[0]}-{v[1]}" for k, v in ep.items()) +
                   "\nearly-stopping groups may finish before the 30-epoch budget", fontsize=8)
fig.suptitle("E5: training strategy tradeoff (2x2; 3 seeds each; separate panels, no shared dual axis)", fontsize=11)
save_fig(fig, "fig06_e5_strategy_tradeoff")

# ---- fig07: E6 confusion matrices ----------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=True)
for ax, seed in zip(axes, seed_order):
    m = cm_data["matrices"][str(seed)]["row_normalized"]
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            v = m[i][j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.4,
                    color="white" if v > 0.55 else "black")
    ax.set_title(f"seed {seed} ({by_seed[seed]})", fontsize=10)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.grid(False)
    ax.label_outer() if ax is not axes[0] else None
cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.015)
cbar.set_label("row-normalized rate (per-seed row sums)")
fig.suptitle("E6: 10x10 confusion matrices on the official 10,000-sample test set\n"
             "one panel per model seed; each matrix normalized by its own row sums (raw counts never summed across seeds)", fontsize=11)
save_fig(fig, "fig07_e6_confusion_matrices")

# ---- fig08: E6 per-class F1 ----------------------------------------------
f1_mean, f1_std = [], []
for i in range(10):
    vals = [class_tables[s][i]["f1"] for s in seed_order]
    f1_mean.append(statistics.fmean(vals))
    f1_std.append(statistics.pstdev(vals))
fig, ax = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)
ax.errorbar(range(10), f1_mean, yerr=f1_std, fmt="o", capsize=5, markersize=6.5,
            color=OKABE["blue"], ecolor=OKABE["vermillion"], elinewidth=1.3, capthick=1.3)
for x, (m, s) in enumerate(zip(f1_mean, f1_std)):
    ax.annotate(f"{m:.4f}", (x, m + s), ha="center", va="bottom", fontsize=8,
                xytext=(3, 4), textcoords="offset points")
ax.set_xticks(range(10))
ax.set_xticklabels([str(i) for i in range(10)])
ax.set_ylim(0.90, 1.002)
ax.set_xlabel("class index\n" + ERROR_BAR_NOTE +
              "\nsupport is per-model (expected ~1,000 per class; exact counts in e6_class_metrics_by_seed.csv)",
              fontsize=8)
ax.set_ylabel("per-class F1 (cross-seed mean)")
ax.set_title("E6: per-class F1 on the test set, mean over model seeds 404/505/606")
save_fig(fig, "fig08_e6_per_class_f1")

# --------------------------------------------------------------------------
print("\n[done] all derived data files and 8 figures (png+svg) generated")
print(f"[done] figures dir: {FIG_DIR}")
print(f"[done] derived dir: {DERIVED_DIR}")
