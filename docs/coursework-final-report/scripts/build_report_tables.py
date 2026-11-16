# -*- coding: utf-8 -*-
"""Build Markdown tables for the coursework report appendices (A-C).

Reads only frozen config/summary artifacts and the derived E6 per-class CSV.
Writes Markdown snippets under tables/ (never modifies any run artifact).
"""
import csv
import json
import os
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
P53 = ROOT / "outputs" / "p5.3"
OUT = Path(__file__).resolve().parents[1] / "tables"
OUT.mkdir(parents=True, exist_ok=True)

STAGES = [
    ("E1_exploration", "E1 exploration"),
    ("E1_confirmation", "E1 confirmation"),
    ("E2_architecture", "E2 architecture"),
    ("E3_activation", "E3 activation"),
    ("E4_l2", "E4 L2"),
    ("E5_training_strategy", "E5 strategy"),
    ("E6_final", "E6 final"),
]
OPT = {"sgd": "sgd", "momentum_sgd": "momentum", "adam": "adam"}


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def key_config(c):
    net, tr = c["network"], c["training"]
    dims = "[" + ",".join(str(x) for x in net["hidden_dims"]) + "]"
    lam = tr["l2_lambda"]
    lam_s = "0" if lam == 0 else f"{lam:g}"
    es = "ES" if tr["early_stopping"] else "noES"
    sched = tr["lr_scheduler"]
    return (f"{dims}/{net['hidden_activation']}/{OPT[tr['optimizer']]}/"
            f"lr{tr['learning_rate']:g}/λ{lam_s}/{sched}/{es}")


# ---- Appendix A: all 66 runs ------------------------------------------------
rows = []
for dirname, label in STAGES:
    for d in sorted(p for p in (P53 / dirname).iterdir() if p.is_dir()):
        c = load(d / "config.json")
        s = load(d / "summary.json")
        rows.append((label, d.name, c["seeds"]["model_seed"],
                     c["experiment_role"], s["status"], key_config(c)))
lines = ["| 阶段 | run 名 | model seed | role | status | 关键配置 |",
         "| --- | --- | ---: | --- | --- | --- |"]
for r in rows:
    lines.append("| " + " | ".join(str(x) for x in r) + " |")
(OUT / "appendix_a_66_runs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("appendix A rows:", len(rows))

# ---- Appendix B: group-level independent recomputation ----------------------
rows_b = []
for dirname, label in STAGES:
    agg = P53 / f"_aggregate_{dirname}" / "groups.json"
    groups = load(agg)["groups"]
    for g in groups:
        vals = {"best_validation_loss": [], "best_validation_accuracy": [], "total_seconds": []}
        for rp in g["run_paths"]:
            s = load(P53 / dirname / rp / "summary.json")
            for k in vals:
                vals[k].append(s[k])
        means = {k: st.fmean(v) for k, v in vals.items()}
        stds = {k: st.pstdev(v) for k, v in vals.items()}
        dev = 0.0
        for k in vals:
            dev = max(dev, abs(means[k] - g["stats"][k]["mean"]),
                      abs(stds[k] - g["stats"][k]["std"]))
        dev_s = "0" if dev <= 1e-12 else f"{dev:.2e}"
        # Short group label
        cfg = g["group_config"]
        net, tr = cfg["network"], cfg["training"]
        if dirname == "E1_confirmation":
            glabel = f"E1 conf · {OPT[tr['optimizer']]} lr{tr['learning_rate']:g}"
        elif dirname == "E1_exploration":
            glabel = f"E1 explore · {OPT[tr['optimizer']]} lr{tr['learning_rate']:g}"
        elif dirname == "E2_architecture":
            glabel = "E2 · [" + ",".join(str(x) for x in net["hidden_dims"]) + "]"
        elif dirname == "E3_activation":
            init = "He" if net["hidden_activation"] == "relu" else "Xavier"
            glabel = f"E3 · {net['hidden_activation']}+{init}"
        elif dirname == "E4_l2":
            glabel = f"E4 · λ={tr['l2_lambda']:g}"
        elif dirname == "E5_training_strategy":
            sched = "fixed" if tr["lr_scheduler"] == "none" else "step"
            es = "es" if tr["early_stopping"] else "no_es"
            glabel = f"E5 · {sched}_{es}"
        else:
            glabel = "E6 · final [256]"
        rows_b.append((glabel, g["n_runs"],
                       f"{means['best_validation_loss']:.6f} ± {stds['best_validation_loss']:.6f}",
                       f"{means['best_validation_accuracy']:.6f}",
                       f"{means['total_seconds']:.3f}", dev_s))
lines = ["| 组 | n | best_val_loss (mean ± pstdev) | best_val_acc | total_seconds | max abs Δ vs groups.json |",
         "| --- | ---: | ---: | ---: | ---: | ---: |"]
for r in rows_b:
    lines.append("| " + " | ".join(str(x) for x in r) + " |")
(OUT / "appendix_b_recompute.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("appendix B rows:", len(rows_b))

# ---- Appendix C: per-seed per-class metrics ---------------------------------
rows_c = []
with open(Path(__file__).resolve().parent / "derived_data" / "e6_class_metrics_by_seed.csv",
          encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        rows_c.append((r["model_seed"], r["class_index"], r["support"],
                       f"{float(r['precision']):.6f}", f"{float(r['recall']):.6f}",
                       f"{float(r['f1']):.6f}"))
rows_c.sort(key=lambda r: (r[0], int(r[1])))
lines = ["| seed | class | support | precision | recall | F1 |",
         "| ---: | ---: | ---: | ---: | ---: | ---: |"]
for r in rows_c:
    lines.append("| " + " | ".join(str(x) for x in r) + " |")
(OUT / "appendix_c_per_seed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("appendix C rows:", len(rows_c))
