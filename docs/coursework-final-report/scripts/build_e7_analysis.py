# -*- coding: utf-8 -*-
"""E7 分析（e7.md 预注册）：[256] 组件复验 vs [128] 冻结结论的排序对照 + 因子解耦表。

只读输入：
  outputs/p5.3/E7_validation_on_256/*/          新增 39 run
  outputs/p5.3/E2_architecture/arch256_seed*    复用控制组（E7-A relu 组 = E7-B λ0 组）
  outputs/p5.3/_aggregate_E3_activation/groups.json 等  [128] 冻结参照（E3/E4/E5）
输出：derived_data/e7_validation_on_256.json
"""
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
P53 = REPO / "outputs" / "p5.3"
E7 = P53 / "E7_validation_on_256"
OUT = Path(__file__).resolve().parents[1] / "derived_data"
SEEDS = [101, 202, 303]

# [128] 冻结参照：block -> 组名 -> (聚合目录, run 路径前缀)
REF_128 = {
    "A": {"relu": ("_aggregate_E3_activation", "relu_seed"),
          "tanh": ("_aggregate_E3_activation", "tanh_seed"),
          "sigmoid": ("_aggregate_E3_activation", "sigmoid_seed")},
    "B": {"lambda0": ("_aggregate_E4_l2", "E4_l2_lambda0_seed"),
          "lambda0.0001": ("_aggregate_E4_l2", "E4_l2_lambda0p0001_seed"),
          "lambda0.001": ("_aggregate_E4_l2", "E4_l2_lambda0p001_seed")},
    "C": {"fixed_no_es": ("_aggregate_E5_training_strategy", "e5_fixed_no_es_seed"),
          "fixed_es": ("_aggregate_E5_training_strategy", "e5_fixed_es_seed"),
          "step_no_es": ("_aggregate_E5_training_strategy", "e5_step_no_es_seed"),
          "step_es": ("_aggregate_E5_training_strategy", "e5_step_es_seed")},
}
# E7 新 run 前缀 -> (block, [256] 组名)（e7bc 前缀同时归属 B 与 C，见 SHARED）
NEW_PREFIX = [
    (r"e7a_tanh256_seed(\d+)", {"A": "tanh"}),
    (r"e7a_sigmoid256_seed(\d+)", {"A": "sigmoid"}),
    (r"e7bc_l2lambda0p0001_fixednoes_seed(\d+)", {"B": "lambda0.0001", "C": "fixed_no_es"}),
    (r"e7b_l2lambda0p001_seed(\d+)", {"B": "lambda0.001"}),
    (r"e7c_fixed_es_seed(\d+)", {"C": "fixed_es"}),
    (r"e7c_step_no_es_seed(\d+)", {"C": "step_no_es"}),
    (r"e7c_step_es_seed(\d+)", {"C": "step_es"}),
    (r"e7d_(relu|tanh|sigmoid)_(he_normal|xavier_normal)_seed(\d+)", "D"),  # 特判
]
SHARED_NOTE = "e7bc_l2lambda0p0001_fixednoes 是同一批运行服务 E7-B（λ=0.0001 组）与 E7-C（fixed_no_es 组），e7.md §4 设计性共享"
REUSED_NOTE = "relu/[256]/λ0 控制组复用 E2_architecture/arch256_seed{101,202,303}（同配置已冻结存在，只读引用，e7.md §4）"


def load_run_summary(path: Path):
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "success" or summary.get("test_evaluation") != "not_run":
        raise SystemExit(f"run discipline violation: {path.name}")
    return config, summary


def group_stats(summaries):
    loss = np.array([s["best_validation_loss"] for s in summaries], dtype=float)
    acc = np.array([s["best_validation_accuracy"] for s in summaries], dtype=float)
    epoch = np.array([s["best_epoch"] for s in summaries], dtype=float)
    return {"n": len(summaries),
            "best_validation_loss": {"mean": float(loss.mean()), "pstdev_ddof0": float(loss.std(ddof=0)),
                                     "values": loss.tolist()},
            "best_validation_accuracy": {"mean": float(acc.mean()), "pstdev_ddof0": float(acc.std(ddof=0))},
            "best_epoch_mean": float(epoch.mean())}


def ref_groups(agg_dir):
    groups = json.loads((P53 / agg_dir / "groups.json").read_text(encoding="utf-8"))["groups"]
    by_prefix = {}
    for grp in groups:
        prefix = re.sub(r"\d+$", "", grp["run_paths"][0])   # "tanh_seed101" -> "tanh_seed"
        st = grp["stats"]
        by_prefix[prefix] = {
            "n": grp["n_runs"],
            "best_validation_loss": {"mean": st["best_validation_loss"]["mean"],
                                     "std": st["best_validation_loss"]["std"]},
            "best_validation_accuracy": {"mean": st["best_validation_accuracy"]["mean"],
                                         "std": st["best_validation_accuracy"]["std"]},
            "best_epoch_mean": st["best_epoch"]["mean"],
            "group_id": grp["group_id"],
        }
    return by_prefix


def ranking(loss_map):
    return [k for k, _ in sorted(loss_map.items(), key=lambda kv: kv[1])]


def main():
    # ---- [256] 新 run 分组 ----
    groups_256 = {}   # (block, name) -> list of summaries
    seen_runs = []
    for run_dir in sorted(p for p in E7.iterdir() if p.is_dir()):
        config, summary = load_run_summary(run_dir)
        name = run_dir.name
        seen_runs.append(name)
        mapped = False
        for pattern, mapping in NEW_PREFIX:
            m = re.fullmatch(pattern, name)
            if not m:
                continue
            mapped = True
            if mapping == "D":
                key = {"D": f"{m.group(1)}+{m.group(2)}"}
                seed = int(m.group(3))
            else:
                key, seed = mapping, int(m.group(1))
            for block, gname in key.items():   # e7bc 前缀同时进 B 与 C 两个组
                groups_256.setdefault((block, gname), []).append((seed, summary))
            break
        if not mapped:
            raise SystemExit(f"unmapped E7 run: {name}")
    # 复用的 E2 arch256 控制组（只读）
    reused = []
    for s in SEEDS:
        config, summary = load_run_summary(P53 / "E2_architecture" / f"arch256_seed{s}")
        reused.append((s, summary))
        groups_256.setdefault(("A", "relu"), []).append((s, summary))
        groups_256.setdefault(("B", "lambda0"), []).append((s, summary))
    for (block, gname), items in groups_256.items():
        items.sort(key=lambda x: x[0])
        if [s for s, _ in items] != SEEDS:
            raise SystemExit(f"group {(block, gname)} seeds != {SEEDS}: {[s for s, _ in items]}")

    stats_256 = {f"{block}:{gname}": group_stats([s for _, s in items])
                 for (block, gname), items in sorted(groups_256.items())}

    # ---- [128] 冻结参照 ----
    refs = {}
    for block, groups in REF_128.items():
        cache = {}
        for gname, (agg, prefix) in groups.items():
            if agg not in cache:
                cache[agg] = ref_groups(agg)
            refs[f"{block}:{gname}"] = cache[agg][prefix]

    # ---- 排序对照（e7.md §6 Q1 判定）----
    comparison = {}
    for block, ref_names in (("A", ["relu", "tanh", "sigmoid"]),
                             ("B", ["lambda0", "lambda0.0001", "lambda0.001"]),
                             ("C", ["fixed_no_es", "fixed_es", "step_no_es", "step_es"])):
        loss_128 = {n: refs[f"{block}:{n}"]["best_validation_loss"]["mean"] for n in ref_names}
        loss_256 = {n: stats_256[f"{block}:{n}"]["best_validation_loss"]["mean"] for n in ref_names}
        r128, r256 = ranking(loss_128), ranking(loss_256)
        comparison[block] = {
            "groups": ref_names,
            "mean_best_val_loss_128": loss_128, "ranking_128": r128,
            "mean_best_val_loss_256": loss_256, "ranking_256": r256,
            "argmin_match": r128[0] == r256[0],
            "full_order_match": r128 == r256,
            "conclusion": ("ranking identical" if r128 == r256 else
                           f"argmin {'matches' if r128[0] == r256[0] else 'DIFFERS'} "
                           f"([128] argmin={r128[0]}, [256] argmin={r256[0]})"),
        }

    # ---- E7-D 因子表 ----
    factorial = {}
    for act in ("relu", "tanh", "sigmoid"):
        for init in ("he_normal", "xavier_normal"):
            factorial[f"{act}+{init}"] = stats_256[f"D:{act}+{init}"]
    # E3 auto 对角线参照（[128]）
    auto_diag = {f"{act}+auto(E3,[128])": refs[f"A:{act}"]["best_validation_loss"]["mean"]
                 for act in ("relu", "tanh", "sigmoid")}

    out = {
        "preregistration": "e7.md（P5.3-E7 预注册 V0.1）",
        "n_new_runs": len(seen_runs), "n_runs_expected": 39,
        "reused_control": REUSED_NOTE, "shared_runs": SHARED_NOTE,
        "groups_256": stats_256,
        "frozen_reference_128": refs,
        "Q1_ranking_comparison": comparison,
        "Q2_factorial_128_explicit_init": factorial,
        "Q2_auto_diagonal_reference_128": auto_diag,
        "note": "全部为 3 seed 描述性统计（ddof=0）；E7 内不做显著性声明（推断由 E8 承担）",
    }
    if len(seen_runs) != 39:
        out["warning"] = f"run count {len(seen_runs)} != 39"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e7_validation_on_256.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for block, comp in comparison.items():
        print(f"[Q1-{block}] {comp['conclusion']}")
    print("[Q2] factorial:")
    for k, v in factorial.items():
        print(f"  {k:24s} {v['best_validation_loss']['mean']:.6f} ± {v['best_validation_loss']['pstdev_ddof0']:.6f}")
    print(f"[done] {OUT / 'e7_validation_on_256.json'}")


if __name__ == "__main__":
    main()
