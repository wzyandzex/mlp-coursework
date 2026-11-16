# -*- coding: utf-8 -*-
"""E8 显著性分析（e8.md 预注册）：精确置换检验 + Holm–Bonferroni。

每臂 10 seed（既有 3 + 新增 7）的 best_validation_loss 做 4 个预注册主对比：
两样本置换检验，双侧 p，完全枚举 C(20,10)=184,756 种分组（纯 NumPy，无随机成分）；
族内 Holm 校正（familywise α=0.05）。敏感性分析：仅 7 新 seed（C(14,7)=3,432）。
次要指标 best_validation_accuracy 只报告均值±std 与方向（预注册禁用"显著"措辞）。

用法：python analyze_e8.py   （输出 derived_data/e8_significance_tests.json）
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
P53 = REPO / "outputs" / "p5.3"
OUT = Path(__file__).resolve().parents[1] / "derived_data"
SEEDS_OLD = [101, 202, 303]
SEEDS_NEW = [707, 808, 909, 1010, 1111, 1212, 1313]

# 臂定义（e8.md §2）：既有 run 目录模式 + 新 run 目录模式
ARMS = {
    "M_momentum_lr0.05": ("E1_confirmation/E1_confirm_momentum_sgd_lr0p05_seed{}",
                          "E8_significance/e8_M_momentum_seed{}"),
    "A_adam_lr0.003": ("E1_confirmation/E1_confirm_adam_lr0p003_seed{}",
                       "E8_significance/e8_A_adam_seed{}"),
    "P256_arch256": ("E2_architecture/arch256_seed{}",
                     "E8_significance/e8_P256_seed{}"),
    "T_tanh_xavier": ("E3_activation/tanh_seed{}",
                      "E8_significance/e8_T_tanh_seed{}"),
    "L1_lambda0.0001": ("E4_l2/E4_l2_lambda0p0001_seed{}",
                        "E8_significance/e8_L1_lambda0p0001_seed{}"),
}
COMPARISONS = [  # (标签, 臂1, 臂2)——恰好 4 个，预注册，不得增删
    ("C1_momentum_vs_adam", "M_momentum_lr0.05", "A_adam_lr0.003"),
    ("C2_arch128_vs_256", "M_momentum_lr0.05", "P256_arch256"),
    ("C3_relu_vs_tanh", "M_momentum_lr0.05", "T_tanh_xavier"),
    ("C4_lambda0_vs_0.0001", "M_momentum_lr0.05", "L1_lambda0.0001"),
]
CONSISTENCY_FIELDS = [
    ("network", "hidden_dims"), ("network", "hidden_activation"), ("network", "initializer"),
    ("training", "epochs"), ("training", "batch_size"), ("training", "learning_rate"),
    ("training", "optimizer"), ("training", "momentum"), ("training", "beta1"),
    ("training", "beta2"), ("training", "eps"), ("training", "lr_scheduler"),
    ("training", "step_size"), ("training", "gamma"), ("training", "early_stopping"),
    ("training", "patience"), ("training", "min_delta"), ("training", "l2_lambda"),
    ("seeds", "split_seed"), ("data", "train_size"), ("data", "val_size"),
]


def load_run(path: Path):
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "success":
        raise SystemExit(f"run not success: {path}")
    if summary.get("test_evaluation") != "not_run" or summary.get("test_loss") is not None:
        raise SystemExit(f"test-set discipline violated in {path}")
    return config, summary


def load_arm(arm):
    old_pat, new_pat = ARMS[arm]
    runs = []
    for s in SEEDS_OLD:
        config, summary = load_run(P53 / old_pat.format(s))
        runs.append({"seed": s, "provenance": "frozen_batch", "config": config,
                     "summary": summary, "path": old_pat.format(s)})
    for s in SEEDS_NEW:
        config, summary = load_run(P53 / new_pat.format(s))
        runs.append({"seed": s, "provenance": "e8_new_batch", "config": config,
                     "summary": summary, "path": new_pat.format(s)})
    # 配置一致性：10 个 run 的关键字段必须逐字段相等
    reference = {f"{sec}.{key}": runs[0]["config"][sec][key] for sec, key in CONSISTENCY_FIELDS}
    mismatches = [
        {"field": field, "run": r["path"],
         "value": r["config"][field.split(".")[0]][field.split(".")[1]]}
        for r in runs for field, ref in reference.items()
        if r["config"][field.split(".")[0]][field.split(".")[1]] != ref
    ]
    if mismatches:
        raise SystemExit(f"config inconsistency in arm {arm}: {mismatches}")
    return runs, reference


def exact_permutation_test(vals1, vals2):
    """两样本置换检验，双侧 p，完全枚举全部 C(n1+n2, n1) 个分组。"""
    vals = np.concatenate([np.asarray(vals1, dtype=np.float64),
                           np.asarray(vals2, dtype=np.float64)])
    n1, n = len(vals1), len(vals)
    obs = vals[:n1].mean() - vals[n1:].mean()
    combos = np.array(list(itertools.combinations(range(n), n1)), dtype=np.int64)
    mask = np.zeros((combos.shape[0], n), dtype=bool)     # (n_combos, n) 成员表
    mask[np.arange(combos.shape[0])[:, None], combos] = True
    sums = (vals[None, :] * mask).sum(axis=1)
    perm_diff = sums / n1 - (vals.sum() - sums) / (n - n1)
    extreme = int(np.count_nonzero(np.abs(perm_diff) >= abs(obs)))
    return {
        "n_per_arm": [n1, n - n1],
        "n_permutations": int(combos.shape[0]),
        "mean_diff_obs": obs,
        "p_two_sided_exact": extreme / combos.shape[0],
        "count_extreme": extreme,
    }


def holm_bonferroni(p_values):
    """Holm–Bonferroni 逐步校正（族wise α 语义），返回与输入同序的校正 p。"""
    m = len(p_values)
    order = np.argsort(p_values)
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adj[idx] = min(1.0, running)
    return adj


def describe(vals):
    v = np.asarray(vals, dtype=np.float64)
    return {"mean": float(v.mean()), "pstdev_ddof0": float(v.std(ddof=0)),
            "values": [float(x) for x in v], "seeds": SEEDS_OLD + SEEDS_NEW}


def main():
    arms, references = {}, {}
    for arm in ARMS:
        runs, reference = load_arm(arm)
        arms[arm] = runs
        references[arm] = reference
    metric = "best_validation_loss"
    primary = []
    for label, a1, a2 in COMPARISONS:
        v1 = [r["summary"][metric] for r in arms[a1]]
        v2 = [r["summary"][metric] for r in arms[a2]]
        test = exact_permutation_test(v1, v2)
        v1_new = [r["summary"][metric] for r in arms[a1] if r["provenance"] == "e8_new_batch"]
        v2_new = [r["summary"][metric] for r in arms[a2] if r["provenance"] == "e8_new_batch"]
        sens = exact_permutation_test(v1_new, v2_new)
        primary.append({
            "comparison": label, "arm1": a1, "arm2": a2, "metric": metric,
            "arm1_stats": describe(v1), "arm2_stats": describe(v2),
            "secondary_metric_best_validation_accuracy": {
                "arm1": describe([r["summary"]["best_validation_accuracy"] for r in arms[a1]]),
                "arm2": describe([r["summary"]["best_validation_accuracy"] for r in arms[a2]])},
            "test": test,
            "sensitivity_new_seeds_only_7v7": {
                "n_per_arm": sens["n_per_arm"], "n_permutations": sens["n_permutations"],
                "mean_diff_obs": sens["mean_diff_obs"],
                "p_two_sided_exact": sens["p_two_sided_exact"]},
        })
    p_raw = np.array([c["test"]["p_two_sided_exact"] for c in primary])
    p_adj = holm_bonferroni(p_raw)
    for c, raw, adj in zip(primary, p_raw, p_adj):
        c["p_raw"] = float(raw)
        c["p_holm_adjusted"] = float(adj)
        c["significant_at_familywise_0.05"] = bool(adj < 0.05)
        c["conclusion"] = ("significant_under_preregistered_holm_0.05" if adj < 0.05
                           else "not_significant_at_preregistered_threshold")
    # 配置一致性引用 + 既有 run 哈希引用留痕（只读引用声明）
    out = {
        "preregistration": "e7.md 同级 e8.md（P5.3-E8 预注册 V0.1）",
        "metric_primary": metric,
        "method": "two-sample permutation test on difference of means, two-sided, "
                  "exact full enumeration; Holm-Bonferroni familywise alpha=0.05 over "
                  "exactly 4 preregistered comparisons",
        "arms": {arm: {"config_reference": references[arm],
                       "runs": [{"seed": r["seed"], "provenance": r["provenance"],
                                 "path": r["path"],
                                 "best_validation_loss": r["summary"][metric],
                                 "best_validation_accuracy":
                                     r["summary"]["best_validation_accuracy"],
                                 "best_epoch": r["summary"]["best_epoch"]}
                                for r in arms[arm]]}
                 for arm in ARMS},
        "primary_comparisons": primary,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e8_significance_tests.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for c in primary:
        print(f"{c['comparison']}: Δ={c['test']['mean_diff_obs']:+.6f} "
              f"p_exact={c['p_raw']:.6f} p_holm={c['p_holm_adjusted']:.6f} -> {c['conclusion']}")
    print(f"[done] {OUT / 'e8_significance_tests.json'}")


if __name__ == "__main__":
    sys.exit(main())
