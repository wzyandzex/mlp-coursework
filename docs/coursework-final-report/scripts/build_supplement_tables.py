# -*- coding: utf-8 -*-
"""§14（E7/E8/E9）与附录 I 的表格生成：只读 derived_data 汇总 JSON 与 run 产物，
输出 tables/ 下的 markdown 片段（§14 正文引用）。与 build_report_tables.py 同风格。"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]   # 报告根目录（脚本位于 scripts/ 子目录）
REPO = Path(__file__).resolve().parents[4]
P53 = REPO / "outputs" / "p5.3"
TABLES = HERE / "tables"
E7 = json.loads((HERE / "derived_data" / "e7_validation_on_256.json").read_text(encoding="utf-8"))
E8 = json.loads((HERE / "derived_data" / "e8_significance_tests.json").read_text(encoding="utf-8"))


def _load_optional(name):
    path = HERE / "derived_data" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


E9 = _load_optional("e9_budget_data.json")          # E9 落盘前容错跳过对应表
E9T = _load_optional("e9a_full60k_test_eval.json")
OPT = {"sgd": "sgd", "momentum_sgd": "momentum", "adam": "adam"}
SEEDS = [101, 202, 303]

TABLES.mkdir(parents=True, exist_ok=True)


def f6(x):
    return f"{x:.6f}"


def f4(x):
    return f"{x:.4f}"


def ms(stat, dec=6):
    return f"{stat['mean']:.{dec}f} ± {stat.get('pstdev_ddof0', stat.get('std', 0)):.{dec}f}"


def key_config(c):
    net, tr = c["network"], c["training"]
    dims = "[" + ",".join(str(x) for x in net["hidden_dims"]) + "]"
    lam = tr["l2_lambda"]
    lam_s = "0" if lam == 0 else f"{lam:g}"
    es = "ES" if tr["early_stopping"] else "noES"
    base = (f"{dims}/{net['hidden_activation']}/{OPT[tr['optimizer']]}/"
            f"lr{tr['learning_rate']:g}/λ{lam_s}/{tr['lr_scheduler']}/{es}")
    if tr["epochs"] != 30:
        base += f"/{tr['epochs']}ep"
    return base


def write(name, lines):
    (TABLES / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[table] {name} ({len(lines)} lines)")


# ---- T14-1：E7 Q1 排序对照 ----
lines = ["| Block | 组 | [128] 冻结（mean ± std） | [256] 复验（mean ± pstdev） | [256] argmin 与 [128] 一致 |",
         "| --- | --- | ---: | ---: | --- |"]
BLOCK_GROUPS = {"A": ["relu", "tanh", "sigmoid"],
                "B": ["lambda0", "lambda0.0001", "lambda0.001"],
                "C": ["fixed_no_es", "fixed_es", "step_no_es", "step_es"]}
GLABEL = {"relu": "ReLU+auto", "tanh": "tanh+auto", "sigmoid": "sigmoid+auto",
          "lambda0": "λ=0", "lambda0.0001": "λ=1e-4", "lambda0.001": "λ=1e-3",
          "fixed_no_es": "fixed/noES", "fixed_es": "fixed/ES",
          "step_no_es": "step/noES", "step_es": "step/ES"}
for block, names in BLOCK_GROUPS.items():
    comp = E7["Q1_ranking_comparison"][block]
    for name in names:
        ref = E7["frozen_reference_128"][f"{block}:{name}"]["best_validation_loss"]
        new = E7["groups_256"][f"{block}:{name}"]["best_validation_loss"]
        # 仅在 [128] 的 argmin 组上标注 [256] 的 argmin 是否与其一致
        match = ("**是**" if comp["argmin_match"] else "否") if name == comp["ranking_128"][0] else ""
        lines.append(f"| E7-{block} | {GLABEL[name]} | {ref['mean']:.6f} ± {ref['std']:.6f} "
                     f"| {new['mean']:.6f} ± {new['pstdev_ddof0']:.6f} | {match} |")
write("t14_e7_ranking.md", lines)

# ---- T14-2：E7-D 因子表 ----
acts = ["relu", "tanh", "sigmoid"]
lines = ["| 激活 \\ 初始化 | He normal | Xavier normal | auto（E3，[128]） |", "| --- | ---: | ---: | ---: |"]
for act in acts:
    row = [act]
    for init in ("he_normal", "xavier_normal"):
        row.append(f6(E7["Q2_factorial_128_explicit_init"][f"{act}+{init}"]["best_validation_loss"]["mean"]))
    row.append(f6(E7["Q2_auto_diagonal_reference_128"][f"{act}+auto(E3,[128])"]))
    lines.append("| " + " | ".join(row) + " |")
write("t14_e7_factorial.md", lines)

# ---- T14-3：E8 检验表 ----
CLABEL = {"C1_momentum_vs_adam": "C1 优化器：momentum 0.05 vs adam 0.003",
          "C2_arch128_vs_256": "C2 架构：[128] vs [256]",
          "C3_relu_vs_tanh": "C3 激活+初始化：ReLU vs tanh（auto）",
          "C4_lambda0_vs_0.0001": "C4 正则：λ=0 vs λ=1e-4"}
lines = ["| 对比 | 臂 1（mean ± pstdev） | 臂 2（mean ± pstdev） | Δ（臂1−臂2） | p exact | p Holm | 7v7 p | 结论 |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
for c in E8["primary_comparisons"]:
    a1, a2 = c["arm1_stats"], c["arm2_stats"]
    concl = "显著" if c["significant_at_familywise_0.05"] else "未达阈值"
    lines.append(
        f"| {CLABEL[c['comparison']]} "
        f"| {ms(a1)} | {ms(a2)} | {c['test']['mean_diff_obs']:+.6f} "
        f"| {c['p_raw']:.4f} | {c['p_holm_adjusted']:.4f} "
        f"| {c['sensitivity_new_seeds_only_7v7']['p_two_sided_exact']:.4f} | {concl} |")
write("t14_e8_tests.md", lines)

# ---- T14-4：E9-B 预算对照 ----
if E9 is not None:
    b = E9["budget_study"]
    lines = ["| 训练预算 | best val loss（mean ± pstdev） | best val acc（mean ± pstdev） | best epoch（mean） | 完成epoch（mean） |",
             "| --- | ---: | ---: | ---: | ---: |"]
    e6ep = sum(b["e6_30ep_frozen"]["best_epoch"]["values"]) / 3
    lines.append(f"| 30 ep（E6 冻结参照） | {ms(b['e6_30ep_frozen']['best_validation_loss'])} "
                 f"| {ms(b['e6_30ep_frozen']['best_validation_accuracy'], 4)} "
                 f"| {e6ep:.1f} | 30.0 |")
    for key, label in (("budget100_no_es", "100 ep，无早停"), ("budget100_es", "100 ep，早停")):
        d = b[key]
        ep_mean = sum(d["best_epoch"]["values"]) / 3
        tag = "no_es" if key == "budget100_no_es" else "es"
        done = sum(b["per_run"][f"{tag}_seed{s}"]["epochs_completed"] for s in (404, 505, 606)) / 3
        lines.append(f"| {label} | {ms(d['best_validation_loss'])} "
                     f"| {ms(d['best_validation_accuracy'], 4)} | {ep_mean:.1f} | {done:.1f} |")
    write("t14_e9_budget.md", lines)

# ---- T14-5：E9-A 60k 测试评估 ----
if E9T is not None:
    lines = ["| 模型 | test loss | test accuracy | macro F1 | 备注 |", "| --- | ---: | ---: | ---: | --- |"]
    for r in E9T["per_seed"]:
        lines.append(f"| E9-A 60k seed {r['seed']} | {r['test_loss']:.6f} | {r['test_accuracy'] * 100:.2f}% "
                     f"| {r['macro_average']['f1']:.6f} | 30 ep 最终参数 |")
    g = E9T["group_stats"]
    lines.append(f"| **60k 均值 ± pstdev** | **{g['test_loss']['mean']:.6f} ± {g['test_loss']['pstdev_ddof0']:.6f}** "
                 f"| **{g['test_accuracy']['mean'] * 100:.2f}% ± {g['test_accuracy']['pstdev_ddof0'] * 100:.2f} pp** "
                 f"| — | 跨 3 seed 描述性统计 |")
    lines.append(f"| E6 20k 参照（冻结） | 0.087566 ± 0.001882 | 97.35% ± 0.01 pp | 0.973271 ± 0.000123 | 验证集选最佳 epoch |")
    write("t14_e9_full60k.md", lines)

# ---- 附录 I：E7/E8/E9 全部运行清单 ----
STAGES = [("E7_validation_on_256", "E7"),
          ("E8_significance", "E8"),
          ("E9_budget_data/e9b_budget100", "E9-B"),
          ("E9_budget_data/e9a_full60k", "E9-A")]
lines = ["| 阶段 | run 名 | model seed | role | status | 关键配置 |",
         "| --- | --- | ---: | --- | --- | --- |"]
n = 0
for dirname, label in STAGES:
    root = P53 / dirname
    if not root.is_dir():
        print(f"[appendix I] skip missing stage dir: {dirname}")
        continue
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (d / "summary.json").is_file():
            print(f"[appendix I] skip incomplete run dir: {d.name}")
            continue
        c = json.loads((d / "config.json").read_text(encoding="utf-8"))
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        cfg = key_config(c)
        if "full60k" in dirname:
            cfg += "/train60k/val0/最终epoch参数"
        lines.append(f"| {label} | {d.name} | {c['seeds']['model_seed']} "
                     f"| {c.get('experiment_role', '—')} | {s['status']} | {cfg} |")
        n += 1
write("appendix_i_e7e8e9_runs.md", lines)
print(f"[appendix I] {n} runs")
