"""跨 run 汇总器测试：全部使用临时目录与手写合成 JSON fixture。

不训练模型、不下载数据、不读取 best_model.npz、不产生真实实验产物。
"""

import csv
import hashlib
import json
import statistics
from pathlib import Path

import pytest

from src.experiments import aggregate, main


# ---------- 合成 fixture 工具 ----------


def _make_config(*, experiment_role="exploration", evaluate_test=False, model_seed=42,
                 split_seed=2026, l2_lambda=0.0, learning_rate=0.1, hidden_dims=(128,),
                 optimizer="sgd"):
    return {
        "task": "P5-TEST",
        "network": {"input_dim": 784, "hidden_dims": list(hidden_dims), "output_dim": 10,
                    "hidden_activation": "relu", "initializer": "auto"},
        "training": {"epochs": 30, "batch_size": 128, "learning_rate": learning_rate,
                     "eval_batch_size": 1000, "optimizer": optimizer, "momentum": 0.9,
                     "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "lr_scheduler": "none",
                     "step_size": 10, "gamma": 0.5, "early_stopping": False,
                     "patience": 5, "min_delta": 1e-4, "l2_lambda": l2_lambda},
        "seeds": {"model_seed": model_seed, "split_seed": split_seed},
        "data": {"source_dir": "data/mnist/raw", "train_size": 20000, "val_size": 5000,
                 "split_strategy": "stratified",
                 "sample_counts": {"train": 20000, "validation": 5000, "test": 10000}},
        "environment": {"python": "3.11.4", "numpy": "2.4.6", "platform": "test"},
        "experiment_role": experiment_role,
        "evaluate_test": evaluate_test,
    }


def _make_summary(*, experiment_role="exploration", test_evaluation="not_run",
                  test_loss=None, test_accuracy=None, best_val_loss=0.5,
                  best_val_acc=0.9, best_epoch=2, stop_reason="max_epochs",
                  epochs_completed=3, total_seconds=1.0, status="success"):
    return {
        "status": status,
        "run_dir": f"runs/{experiment_role}",  # 平台相关路径：聚合器不得依赖它
        "experiment_role": experiment_role,
        "test_evaluation": test_evaluation,
        "stop_reason": stop_reason,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_validation_accuracy": best_val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "sample_counts": {"train": 20000, "validation": 5000, "test": 10000},
        "epochs_completed": epochs_completed,
        "total_seconds": total_seconds,
    }


def _write_run(root, name, config=None, summary=None, *, write_config=True,
               write_summary=True, config_text=None, summary_text=None, history=None):
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_text is not None:
        (run_dir / "config.json").write_text(config_text, encoding="utf-8")
    elif write_config and config is not None:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if summary_text is not None:
        (run_dir / "summary.json").write_text(summary_text, encoding="utf-8")
    elif write_summary and summary is not None:
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if history is not None:
        (run_dir / "history.csv").write_text(history, encoding="utf-8")
    return run_dir


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _final_matrix(num_classes=10, diag=930, errors=((0, 1, 700),)):
    """构造总数 = diag*num_classes + sum(errors) 的方形整数混淆矩阵。

    默认参数下总数 10000、对角线 9300，对角线占比 0.93。
    """
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for i in range(num_classes):
        matrix[i][i] = diag
    for i, j, count in errors:
        matrix[i][j] += count
    return matrix


def _final_metrics_fields(matrix, *, test_loss=0.2):
    """用测试文件内独立手写的纯 Python 实现从矩阵导出分类指标字段。

    与被测实现相互独立，用于构造合法 final fixture 并交叉复核汇总器输出。
    """
    size = len(matrix)
    per_class = []
    for cls in range(size):
        tp = matrix[cls][cls]
        support = sum(matrix[cls])
        predicted = sum(matrix[row][cls] for row in range(size))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1": f1,
                          "support": support})
    macro = {key: sum(entry[key] for entry in per_class) / size
             for key in ("precision", "recall", "f1")}
    total = sum(sum(row) for row in matrix)
    accuracy = sum(matrix[i][i] for i in range(size)) / total
    return {
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        "test_confusion_matrix": matrix,
        "test_per_class_metrics": per_class,
        "test_macro_avg": macro,
    }


# ---------- 1. 单个成功 run 的解析 ----------


def test_single_baseline_run_parsed(tmp_path):
    root = tmp_path / "outputs"
    config = _make_config(experiment_role="baseline", evaluate_test=True)
    summary = _make_summary(experiment_role="baseline", test_evaluation="executed",
                            test_loss=0.12, test_accuracy=0.96)
    _write_run(root, "run_baseline", config=config, summary=summary,
               history="epoch,train_loss\n1,0.7\n2,0.5\n")
    report = aggregate(root=root, output=root / "_aggregate")
    assert report["total_runs"] == 1
    assert report["counts"] == {"success": 1, "incomplete": 0, "invalid": 0, "legacy": 0}

    runs = _read_json(root / "_aggregate" / "runs.json")["runs"]
    assert len(runs) == 1
    record = runs[0]
    assert record["run_name"] == "run_baseline"
    assert record["run_path"] == "run_baseline"
    assert record["status"] == "success"
    assert record["experiment_role"] == "baseline"
    assert record["test_evaluation"] == "executed"
    assert record["test_loss"] == 0.12 and record["test_accuracy"] == 0.96
    assert record["history_rows"] == 2
    assert record["model_seed"] == 42 and record["split_seed"] == 2026
    assert record["optimizer"] == "sgd" and record["l2_lambda"] == 0.0
    assert record["network"]["hidden_dims"] == [128]

    groups = _read_json(root / "_aggregate" / "groups.json")["groups"]
    assert len(groups) == 1
    group = groups[0]
    assert group["n_runs"] == 1
    assert group["experiment_role"] == "baseline"
    assert group["stats"]["best_validation_loss"]["mean"] == pytest.approx(0.5)
    assert group["stats"]["best_validation_loss"]["std"] == 0.0  # 单 run 组标准差为 0
    assert group["stats"]["test_loss"]["mean"] == pytest.approx(0.12)
    assert group["stats"]["test_loss"]["n"] == 1


# ---------- 2. 仅 model_seed 不同的 exploration run 分组与手算统计 ----------


def test_model_seed_only_runs_grouped_with_hand_computed_stats(tmp_path):
    root = tmp_path / "outputs"
    losses = [0.5, 0.7, 0.9]
    for i, loss in enumerate(losses):
        _write_run(root, f"run_seed_{i}",
                   config=_make_config(model_seed=100 + i),
                   summary=_make_summary(best_val_loss=loss,
                                         best_val_acc=0.9 + 0.01 * i,
                                         total_seconds=10.0 + i))
    aggregate(root=root, output=root / "_aggregate")
    groups = _read_json(root / "_aggregate" / "groups.json")["groups"]
    assert len(groups) == 1
    group = groups[0]
    assert group["n_runs"] == 3
    assert group["stats"]["best_validation_loss"]["mean"] == pytest.approx(0.7)
    assert group["stats"]["best_validation_loss"]["std"] == pytest.approx(
        statistics.pstdev(losses)
    )
    assert group["stats"]["best_validation_accuracy"]["mean"] == pytest.approx(
        statistics.fmean([0.9, 0.91, 0.92])
    )
    assert group["stats"]["total_seconds"]["mean"] == pytest.approx(11.0)
    assert group["run_paths"] == [f"run_seed_{i}" for i in range(3)]  # 稳定排序
    # exploration 组的测试指标不进入统计：null + 原因
    assert group["stats"]["test_loss"]["mean"] is None
    assert group["stats"]["test_loss"]["n"] == 0
    assert "not_run" in group["stats"]["test_loss"]["reason"]
    # group_id 可由规范化配置复算，且规范化配置保留完整内容（仅去掉 model_seed）
    assert group["group_id"] == "g" + hashlib.sha256(
        group["canonical_config"].encode("utf-8")
    ).hexdigest()[:12]
    assert "model_seed" not in group["group_config"]["seeds"]
    assert group["group_config"]["seeds"]["split_seed"] == 2026
    assert group["group_config"]["network"]["hidden_dims"] == [128]


# ---------- 3. 除 model_seed 外的任何差异都不得合组 ----------


def test_differences_other_than_model_seed_prevent_grouping(tmp_path):
    root = tmp_path / "outputs"
    _write_run(root, "base", config=_make_config(model_seed=1), summary=_make_summary())
    _write_run(root, "diff_lr",
               config=_make_config(model_seed=1, learning_rate=0.2), summary=_make_summary())
    _write_run(root, "diff_l2",
               config=_make_config(model_seed=1, l2_lambda=1e-3), summary=_make_summary())
    _write_run(root, "diff_split",
               config=_make_config(model_seed=1, split_seed=7), summary=_make_summary())
    _write_run(root, "diff_net",
               config=_make_config(model_seed=1, hidden_dims=(64,)), summary=_make_summary())
    _write_run(root, "diff_optimizer",
               config=_make_config(model_seed=1, optimizer="adam"), summary=_make_summary())
    _write_run(root, "diff_role",
               config=_make_config(model_seed=1, experiment_role="baseline", evaluate_test=True),
               summary=_make_summary(experiment_role="baseline",
                                     test_evaluation="executed",
                                     test_loss=0.1, test_accuracy=0.95))
    _write_run(root, "diff_eval",
               config=_make_config(model_seed=2, experiment_role="final", evaluate_test=True),
               summary={**_make_summary(experiment_role="final",
                                        test_evaluation="executed",
                                        test_loss=0.2, test_accuracy=0.93),
                        # 当前 schema 的 final run 必须携带完整分类指标字段
                        **_final_metrics_fields(_final_matrix())})
    report = aggregate(root=root, output=root / "_aggregate")
    groups = _read_json(root / "_aggregate" / "groups.json")["groups"]
    assert report["counts"]["success"] == 8
    assert len(groups) == 8  # 没有任何两个 run 被错误合并
    assert all(group["n_runs"] == 1 for group in groups)


# ---------- 5. 角色/评估状态/指标矛盾 → invalid，不静默修复 ----------


def test_role_metric_contradictions_marked_invalid(tmp_path):
    root = tmp_path / "outputs"
    # exploration 声称已执行测试评估
    _write_run(root, "bad_exp",
               config=_make_config(experiment_role="exploration", evaluate_test=False),
               summary=_make_summary(experiment_role="exploration",
                                     test_evaluation="executed",
                                     test_loss=0.3, test_accuracy=0.9))
    # baseline 声称未评估
    _write_run(root, "bad_base",
               config=_make_config(experiment_role="baseline", evaluate_test=True),
               summary=_make_summary(experiment_role="baseline", test_evaluation="not_run"))
    # executed 但指标为 null
    _write_run(root, "bad_null",
               config=_make_config(experiment_role="final", evaluate_test=True),
               summary=_make_summary(experiment_role="final", test_evaluation="executed"))
    # not_run 却给了指标（0 也不行）
    _write_run(root, "bad_notrun_value", config=_make_config(),
               summary=_make_summary(experiment_role="exploration",
                                     test_evaluation="not_run",
                                     test_loss=0.0, test_accuracy=0.5))
    # config 与 summary 角色不一致
    _write_run(root, "bad_role_mismatch",
               config=_make_config(experiment_role="final", evaluate_test=True),
               summary=_make_summary(experiment_role="baseline",
                                     test_evaluation="executed",
                                     test_loss=0.2, test_accuracy=0.9))
    # evaluate_test 与角色不匹配
    _write_run(root, "bad_config_flag",
               config=_make_config(experiment_role="baseline", evaluate_test=False),
               summary=_make_summary(experiment_role="baseline", test_evaluation="not_run"))
    report = aggregate(root=root, output=root / "_aggregate")
    assert report["counts"]["invalid"] == 6
    assert report["counts"]["success"] == 0
    runs = {record["run_name"]: record
            for record in _read_json(root / "_aggregate" / "runs.json")["runs"]}
    for name in ("bad_exp", "bad_base", "bad_null", "bad_notrun_value",
                 "bad_role_mismatch", "bad_config_flag"):
        assert runs[name]["status"] == "invalid"
        assert runs[name]["diagnostics"]
    assert _read_json(root / "_aggregate" / "groups.json")["groups"] == []


# ---------- 6/7. null 不被改写成 0；缺失与损坏文件的处理 ----------


def test_null_metrics_not_coerced_to_zero(tmp_path):
    root = tmp_path / "outputs"
    _write_run(root, "exp", config=_make_config(), summary=_make_summary())
    aggregate(root=root, output=root / "_aggregate")
    with open(root / "_aggregate" / "runs.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["test_loss"] == ""  # CSV 空单元格表示 null（文档约定）
    assert rows[0]["test_accuracy"] == ""
    runs = _read_json(root / "_aggregate" / "runs.json")["runs"]
    assert runs[0]["test_loss"] is None
    assert runs[0]["test_accuracy"] is None
    # CSV 中的嵌套结构是紧凑 JSON 字符串，而不是 Python repr
    assert rows[0]["config_json"].startswith('{"')
    assert "'" not in rows[0]["config_json"]


def _assert_standard_json(text: str) -> None:
    """断言文本是标准 JSON：值层面不含 NaN/Infinity 字面量。"""
    def reject(name):
        raise AssertionError(f"non-standard JSON constant {name} found in output")

    json.loads(text, parse_constant=reject)


def test_missing_and_corrupt_files_handled(tmp_path):
    root = tmp_path / "outputs"
    _write_run(root, "no_summary", config=_make_config(), write_summary=False)
    _write_run(root, "no_config", summary=_make_summary(), write_config=False)
    _write_run(root, "corrupt_config", config_text="{not json", summary=_make_summary())
    _write_run(root, "corrupt_summary", config=_make_config(), summary_text="{oops")
    nan_summary_text = json.dumps(_make_summary()).replace("0.5", "NaN")
    _write_run(root, "nan_summary", config=_make_config(), summary_text=nan_summary_text)
    _write_run(root, "inf_config", config_text=json.dumps(_make_config()).replace("0.1", "1e400"),
               summary=_make_summary())
    _write_run(root, "bad_types", config=_make_config(),
               summary=_make_summary(best_val_loss="0.5"))  # 指标是字符串
    report = aggregate(root=root, output=root / "_aggregate")
    runs = {record["run_name"]: record
            for record in _read_json(root / "_aggregate" / "runs.json")["runs"]}
    assert runs["no_summary"]["status"] == "incomplete"
    assert runs["no_config"]["status"] == "incomplete"
    assert runs["corrupt_config"]["status"] == "invalid"
    assert runs["corrupt_summary"]["status"] == "invalid"
    assert runs["nan_summary"]["status"] == "invalid"
    assert runs["inf_config"]["status"] == "invalid"
    assert runs["bad_types"]["status"] == "invalid"
    assert report["counts"]["invalid"] == 5
    assert report["counts"]["incomplete"] == 2
    # 标准 JSON 输出在值层面不含 NaN/Infinity 字面量（诊断字符串除外）
    for name in ("runs.json", "groups.json", "report.json"):
        _assert_standard_json((root / "_aggregate" / name).read_text(encoding="utf-8"))


def test_legacy_run_without_role_fields(tmp_path):
    root = tmp_path / "outputs"
    # P4/P5.2-B 时代的 run：config 没有 experiment_role，summary 没有 test_evaluation
    legacy_config = _make_config()
    legacy_config.pop("experiment_role")
    legacy_config.pop("evaluate_test")
    legacy_summary = _make_summary(test_loss=0.9, test_accuracy=0.95)
    legacy_summary.pop("test_evaluation")
    _write_run(root, "old_run", config=legacy_config, summary=legacy_summary)
    report = aggregate(root=root, output=root / "_aggregate")
    runs = _read_json(root / "_aggregate" / "runs.json")["runs"]
    assert runs[0]["status"] == "legacy"
    assert any("test_evaluation" in diagnostic for diagnostic in runs[0]["diagnostics"])
    assert runs[0]["test_loss"] == 0.9  # 已有数值原样保留，但不参与分组/测试统计
    assert _read_json(root / "_aggregate" / "groups.json")["groups"] == []
    assert report["counts"]["legacy"] == 1


# ---------------------------------------------------------------------------
# P5.2-D-R1 整改：当前 schema 严格校验、legacy/null 区分与极端数值防护
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "section, key",
    [
        ("training", "optimizer"),
        ("training", "learning_rate"),
        ("seeds", "model_seed"),
        ("network", "input_dim"),
        ("environment", "python"),
    ],
)
def test_missing_required_config_fields_marked_invalid(tmp_path, section, key):
    # 删除任一当前 schema 必需字段都必须判 invalid（而非 success/legacy）
    root = tmp_path / "outputs"
    config = _make_config()
    del config[section][key]
    _write_run(root, "missing_field", config=config, summary=_make_summary())
    report = aggregate(root=root, output=root / "_aggregate")
    runs = {record["run_name"]: record
            for record in _read_json(root / "_aggregate" / "runs.json")["runs"]}
    assert runs["missing_field"]["status"] == "invalid"
    assert any(key in diagnostic for diagnostic in runs["missing_field"]["diagnostics"])
    assert report["counts"]["success"] == 0
    assert report["counts"]["legacy"] == 0  # 键存在于 config → 不算 legacy


@pytest.mark.parametrize(
    "section, key, value, token",
    [
        ("training", "batch_size", True, "batch_size"),          # bool 冒充 int
        ("training", "learning_rate", "0.1", "learning_rate"),   # 字符串冒充数字
        ("training", "gamma", 1.5, "gamma"),                     # 越上界
        ("training", "gamma", 0.0, "gamma"),                     # 越下界（开区间）
        ("training", "momentum", 1.0, "momentum"),               # 越上界（开区间）
        ("training", "l2_lambda", -0.1, "l2_lambda"),            # 负数
        ("network", "hidden_activation", {"x": 1}, "hidden_activation"),  # 对象冒充字符串
        ("seeds", "model_seed", False, "model_seed"),            # bool 冒充非负整数
    ],
)
def test_invalid_field_types_and_ranges_marked_invalid(tmp_path, section, key, value, token):
    root = tmp_path / "outputs"
    config = _make_config()
    config[section][key] = value
    _write_run(root, "bad_type", config=config, summary=_make_summary())
    aggregate(root=root, output=root / "_aggregate")
    runs = {record["run_name"]: record
            for record in _read_json(root / "_aggregate" / "runs.json")["runs"]}
    assert runs["bad_type"]["status"] == "invalid"
    assert any(token in diagnostic for diagnostic in runs["bad_type"]["diagnostics"])


def test_summary_status_missing_or_not_success_is_invalid(tmp_path):
    root = tmp_path / "outputs"
    missing_status = _make_summary()
    del missing_status["status"]
    _write_run(root, "no_status", config=_make_config(), summary=missing_status)
    failed = _make_summary(status="failed")
    _write_run(root, "failed_run", config=_make_config(), summary=failed)
    report = aggregate(root=root, output=root / "_aggregate")
    runs = {record["run_name"]: record
            for record in _read_json(root / "_aggregate" / "runs.json")["runs"]}
    assert runs["no_status"]["status"] == "invalid"
    assert any("missing required field 'status'" in diagnostic
               for diagnostic in runs["no_status"]["diagnostics"])
    assert runs["failed_run"]["status"] == "invalid"
    assert any("'status' must be 'success'" in diagnostic
               for diagnostic in runs["failed_run"]["diagnostics"])
    assert report["counts"]["success"] == 0


def test_legacy_requires_missing_key_while_explicit_null_is_invalid(tmp_path):
    root = tmp_path / "outputs"
    # 键缺失 → legacy（两处分别成立）
    role_missing = _make_config()
    del role_missing["experiment_role"]
    _write_run(root, "role_key_missing", config=role_missing,
               summary=_make_summary(test_evaluation="not_run"))
    te_missing = _make_summary()
    del te_missing["test_evaluation"]
    _write_run(root, "te_key_missing", config=_make_config(), summary=te_missing)
    # 键存在但值为显式 null → 当前 schema 的无效值 → invalid（不归为 legacy）
    role_null = _make_config()
    role_null["experiment_role"] = None
    _write_run(root, "role_null", config=role_null, summary=_make_summary())
    te_null = _make_summary()
    te_null["test_evaluation"] = None
    _write_run(root, "te_null", config=_make_config(), summary=te_null)
    report = aggregate(root=root, output=root / "_aggregate")
    runs = {record["run_name"]: record
            for record in _read_json(root / "_aggregate" / "runs.json")["runs"]}
    assert runs["role_key_missing"]["status"] == "legacy"
    assert runs["te_key_missing"]["status"] == "legacy"
    assert runs["role_null"]["status"] == "invalid"
    assert any("experiment_role" in diagnostic and "None" in diagnostic
               for diagnostic in runs["role_null"]["diagnostics"])
    assert runs["te_null"]["status"] == "invalid"
    assert any("test_evaluation" in diagnostic and "None" in diagnostic
               for diagnostic in runs["te_null"]["diagnostics"])
    assert report["counts"] == {"success": 0, "incomplete": 0, "invalid": 2, "legacy": 2}


def test_legacy_present_field_damage_takes_precedence_over_legacy(tmp_path):
    # experiment_role 键缺失（legacy 候选），但已存在的 test_evaluation 明确非法
    # → 优先判 invalid，不把 schema 损坏掩盖成 legacy
    root = tmp_path / "outputs"
    config = _make_config()
    del config["experiment_role"]
    _write_run(root, "damaged", config=config,
               summary=_make_summary(test_evaluation="banana"))
    report = aggregate(root=root, output=root / "_aggregate")
    runs = _read_json(root / "_aggregate" / "runs.json")["runs"]
    assert runs[0]["status"] == "invalid"
    assert any("banana" in diagnostic for diagnostic in runs[0]["diagnostics"])
    assert any("instead of legacy" in diagnostic for diagnostic in runs[0]["diagnostics"])


def test_sample_counts_mismatch_marked_invalid(tmp_path):
    root = tmp_path / "outputs"
    config = _make_config()  # sample_counts: train=20000
    summary = _make_summary()
    summary["sample_counts"]["train"] = 19000  # 与 config 不一致
    _write_run(root, "mismatch", config=config, summary=summary)
    aggregate(root=root, output=root / "_aggregate")
    runs = {record["run_name"]: record
            for record in _read_json(root / "_aggregate" / "runs.json")["runs"]}
    assert runs["mismatch"]["status"] == "invalid"
    assert any("sample_counts" in diagnostic and "does not match" in diagnostic
               for diagnostic in runs["mismatch"]["diagnostics"])


# ---------------------------------------------------------------------------
# P5.2-D-R2 整改：config.data.sample_counts 必需键校验
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample_counts, missing_inner_key",
    [
        ("missing", None),                                 # 键整体缺失（R2 缺陷场景）
        (None, None),                                      # 显式 null
        ({"train": 20000, "validation": 5000}, "test"),    # 缺 test
        ({"train": 20000, "test": 10000}, "validation"),   # 缺 validation
        ({"validation": 5000, "test": 10000}, "train"),    # 缺 train
    ],
)
def test_missing_or_malformed_sample_counts_marked_invalid(
    tmp_path, sample_counts, missing_inner_key
):
    root = tmp_path / "outputs"
    config = _make_config()
    if sample_counts == "missing":
        del config["data"]["sample_counts"]  # 键整体缺失：旧实现会静默判 success
    else:
        config["data"]["sample_counts"] = sample_counts
    _write_run(root, "bad_counts", config=config, summary=_make_summary())
    report = aggregate(root=root, output=root / "_aggregate")
    runs = {record["run_name"]: record
            for record in _read_json(root / "_aggregate" / "runs.json")["runs"]}
    assert runs["bad_counts"]["status"] == "invalid"
    assert any("sample_counts" in diagnostic
               for diagnostic in runs["bad_counts"]["diagnostics"])
    if missing_inner_key is not None:  # 内层键缺失时诊断定位到具体字段
        assert any(
            missing_inner_key in diagnostic and "missing required field" in diagnostic
            for diagnostic in runs["bad_counts"]["diagnostics"]
        )
    # invalid run 不得进入成功分组
    assert report["counts"]["success"] == 0
    assert report["counts"]["invalid"] == 1
    assert _read_json(root / "_aggregate" / "groups.json")["groups"] == []


def test_missing_sample_counts_default_mode_and_strict(tmp_path):
    # 默认模式：保留完整汇总与诊断、退出码 0；--strict：非零退出码
    root = tmp_path / "outputs"
    config = _make_config()
    del config["data"]["sample_counts"]
    _write_run(root, "no_counts", config=config, summary=_make_summary())
    report = aggregate(root=root, output=root / "_aggregate")
    assert report["counts"]["invalid"] == 1
    assert (root / "_aggregate" / "runs.json").is_file()
    assert (root / "_aggregate" / "groups.json").is_file()
    assert (root / "_aggregate" / "report.json").is_file()
    problems = report["problems"]
    assert len(problems) == 1 and problems[0]["run_path"] == "no_counts"
    assert any("sample_counts" in diagnostic for diagnostic in problems[0]["diagnostics"])
    assert main(["aggregate", "--root", str(root), "--output", str(root / "_aggregate"),
                 "--overwrite", "--strict"]) == 1


def test_extreme_integer_values_do_not_crash_aggregation(tmp_path):
    # 极大整数（10**400）曾让 math.isfinite 抛未捕获 OverflowError 导致整个
    # 汇总中止；修复后必须可靠标记 invalid，输出仍是可严格解析的标准 JSON
    root = tmp_path / "outputs"
    config = _make_config()
    config["training"]["epochs"] = 10 ** 400  # 极大整数冒充 epochs（不崩溃即可）
    summary = _make_summary(best_val_loss=10 ** 400)  # 极大整数冒充有限数值 → invalid
    _write_run(root, "huge", config=config, summary=summary)
    report = aggregate(root=root, output=root / "_aggregate")
    runs = {record["run_name"]: record
            for record in _read_json(root / "_aggregate" / "runs.json")["runs"]}
    assert runs["huge"]["status"] == "invalid"
    assert any("best_validation_loss" in diagnostic
               for diagnostic in runs["huge"]["diagnostics"])
    assert report["counts"]["invalid"] == 1
    for name in ("runs.json", "groups.json", "report.json"):
        text = (root / "_aggregate" / name).read_text(encoding="utf-8")

        def reject(constant):
            raise AssertionError(f"non-standard JSON constant {constant} in {name}")

        json.loads(text, parse_constant=reject)  # 严格 JSON parser 可读取
    assert (root / "_aggregate" / "runs.csv").read_text(encoding="utf-8")  # CSV 非空可读


# ---------- 8. 输入与输出目录的边界行为 ----------


def test_invalid_root_fails(tmp_path):
    with pytest.raises(ValueError, match="not an existing directory"):
        aggregate(root=tmp_path / "missing", output=tmp_path / "agg")


def test_empty_root_writes_empty_aggregate(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    report = aggregate(root=root, output=root / "_aggregate")
    assert report["total_runs"] == 0
    assert report["n_groups"] == 0
    assert _read_json(root / "_aggregate" / "runs.json")["runs"] == []
    assert _read_json(root / "_aggregate" / "groups.json")["groups"] == []
    assert _read_json(root / "_aggregate" / "report.json")["total_runs"] == 0


def test_existing_output_dir_requires_overwrite(tmp_path):
    root = tmp_path / "outputs"
    _write_run(root, "run_a", config=_make_config(), summary=_make_summary())
    out = root / "_aggregate"
    aggregate(root=root, output=out)
    with pytest.raises(ValueError, match="already exists"):
        aggregate(root=root, output=out)  # 默认拒绝覆盖
    report = aggregate(root=root, output=out, overwrite=True)
    assert report["total_runs"] == 1


# ---------- 9. 汇总输出目录不会被二次运行当成 run ----------


def test_aggregate_output_not_recognized_as_run(tmp_path):
    root = tmp_path / "outputs"
    _write_run(root, "run_a", config=_make_config(), summary=_make_summary())
    aggregate(root=root, output=root / "_aggregate")
    report = aggregate(root=root, output=root / "_aggregate", overwrite=True)
    assert report["total_runs"] == 1  # _aggregate 未被识别为 run
    runs = _read_json(root / "_aggregate" / "runs.json")["runs"]
    assert all(record["run_path"] != "_aggregate" for record in runs)


# ---------- 10. 确定性：稳定排序与逐字节相同的重复输出 ----------


def test_outputs_are_deterministic(tmp_path):
    root = tmp_path / "outputs"
    for i, loss in enumerate([0.3, 0.2, 0.4]):
        _write_run(root, f"run_{i}",
                   config=_make_config(model_seed=100 + i),
                   summary=_make_summary(best_val_loss=loss))
    aggregate(root=root, output=root / "_agg_a")
    aggregate(root=root, output=root / "_agg_b")
    for name in ("runs.json", "groups.json", "report.json", "runs.csv", "groups.csv"):
        assert (root / "_agg_a" / name).read_bytes() == (root / "_agg_b" / name).read_bytes()


# ---------- 11. 默认模式容忍损坏 run；--strict 返回非零 ----------


def test_default_mode_tolerates_invalid_and_strict_fails(tmp_path):
    root = tmp_path / "outputs"
    _write_run(root, "good", config=_make_config(), summary=_make_summary())
    _write_run(root, "bad", config_text="{broken", summary_text="{also")
    report = aggregate(root=root, output=root / "_aggregate")
    assert report["counts"]["success"] == 1  # 默认模式保留其它有效 run
    assert report["strict_failed"] is False

    report_strict = aggregate(root=root, output=root / "_aggregate",
                              overwrite=True, strict=True)
    assert report_strict["strict_failed"] is True
    # CLI 退出码：strict 非零、默认零
    assert main(["aggregate", "--root", str(root), "--output", str(root / "_aggregate"),
                 "--overwrite", "--strict"]) == 1
    assert main(["aggregate", "--root", str(root), "--output", str(root / "_aggregate"),
                 "--overwrite"]) == 0


# ---------- 12. 输出目录与源 run 重合必须拒绝 ----------


def test_output_overlapping_source_run_refused(tmp_path):
    root = tmp_path / "outputs"
    run_dir = _write_run(root, "run_a", config=_make_config(), summary=_make_summary())
    with pytest.raises(ValueError, match="overlaps"):
        aggregate(root=root, output=run_dir)
    with pytest.raises(ValueError, match="overlaps"):
        aggregate(root=root, output=run_dir / "nested")  # 输出位于 run 目录内部
    with pytest.raises(ValueError, match="input root"):
        aggregate(root=root, output=root)  # 输出与输入根相同


# ---------- 13. 汇总过程不修改任何输入文件 ----------


def test_aggregation_does_not_modify_inputs(tmp_path):
    root = tmp_path / "outputs"
    run_dir = _write_run(root, "run_a", config=_make_config(), summary=_make_summary(),
                         history="epoch,train_loss\n1,0.6\n")
    before = {path: path.read_bytes() for path in sorted(run_dir.iterdir())}
    aggregate(root=root, output=root / "_aggregate")
    for path, content in before.items():
        assert path.read_bytes() == content


# ---------- 14. CLI：--help 与合成端到端 ----------


def test_cli_help_documents_isolation_rules(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["aggregate", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for token in ("--root", "--output", "--strict", "--overwrite", "exploration", "not_run"):
        assert token in out  # 参数与测试集隔离规则说明可见


def test_cli_aggregate_end_to_end(tmp_path, capsys):
    root = tmp_path / "outputs"
    for i in range(2):
        _write_run(root, f"run_{i}",
                   config=_make_config(model_seed=i),
                   summary=_make_summary(best_val_loss=0.4 + 0.1 * i))
    exit_code = main(["aggregate", "--root", str(root), "--output", str(root / "_aggregate")])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[aggregate]" in out
    assert "success=2" in out
    assert (root / "_aggregate" / "groups.csv").is_file()
    assert (root / "_aggregate" / "report.json").is_file()


def test_cli_missing_root_returns_nonzero(tmp_path, capsys):
    exit_code = main(["aggregate", "--root", str(tmp_path / "nope"),
                      "--output", str(tmp_path / "agg")])
    assert exit_code == 1
    assert "error" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# P5.3-E6-A 新增：final run 分类指标的 schema 校验、保留与跨 seed 组统计。
# 全部使用手写合成 JSON fixture（2 类小矩阵便于手算），不读取真实测试集。
# ---------------------------------------------------------------------------


def _make_final_run(root, name, *, model_seed, matrix, test_loss):
    """写一个 schema 合法的合成 final run（矩阵维度决定 config.output_dim）。"""
    config = _make_config(experiment_role="final", evaluate_test=True, model_seed=model_seed,
                          l2_lambda=0.0001)
    config["network"]["output_dim"] = len(matrix)
    total = sum(sum(row) for row in matrix)
    config["data"]["sample_counts"]["test"] = total
    summary = {**_make_summary(experiment_role="final", test_evaluation="executed"),
               **_final_metrics_fields(matrix, test_loss=test_loss)}
    summary["sample_counts"]["test"] = total
    _write_run(root, name, config=config, summary=summary)


def test_final_run_record_validated_and_retained(tmp_path):
    """合法 final run → success；runs.json / runs.csv 无损保留分类指标。"""
    root = tmp_path / "outputs"
    matrix = _final_matrix()
    _make_final_run(root, "final_a", model_seed=404, matrix=matrix, test_loss=0.2)
    report = aggregate(root=root, output=root / "_aggregate", strict=True)
    assert report["counts"] == {"success": 1, "incomplete": 0, "invalid": 0, "legacy": 0}
    runs = _read_json(root / "_aggregate" / "runs.json")["runs"]
    record = runs[0]
    assert record["status"] == "success" and record["experiment_role"] == "final"
    assert record["test_confusion_matrix"] == matrix
    # class0：TP=930，predicted=930（列和），support=1630（行和，含 700 个错分）
    assert record["test_per_class_metrics"][0]["precision"] == pytest.approx(1.0)
    assert record["test_per_class_metrics"][0]["recall"] == pytest.approx(930 / 1630)
    assert record["test_per_class_metrics"][0]["support"] == 1630
    # class1：TP=930，predicted=1630（吸收 700 个错分），support=930
    assert record["test_per_class_metrics"][1]["precision"] == pytest.approx(930 / 1630)
    assert record["test_per_class_metrics"][1]["recall"] == pytest.approx(1.0)
    # macro recall = (930/1630 + 1.0 + 8×1.0) / 10
    assert record["test_macro_avg"]["recall"] == pytest.approx((930 / 1630 + 9.0) / 10)
    with open(root / "_aggregate" / "runs.csv", newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    # CSV 复杂结构为紧凑规范化 JSON（双引号、无空格），不是 Python repr
    assert json.loads(row["test_confusion_matrix_json"]) == matrix
    assert "'" not in row["test_confusion_matrix_json"]
    assert json.loads(row["test_macro_avg_json"]) == record["test_macro_avg"]


def test_final_missing_fields_invalid(tmp_path):
    root = tmp_path / "outputs"
    config = _make_config(experiment_role="final", evaluate_test=True)
    summary = _make_summary(experiment_role="final", test_evaluation="executed",
                            test_loss=0.2, test_accuracy=0.93)
    _write_run(root, "final_nofields", config=config, summary=summary)
    report = aggregate(root=root, output=root / "_aggregate", strict=True)
    assert report["counts"]["invalid"] == 1
    runs = _read_json(root / "_aggregate" / "runs.json")["runs"]
    assert ("final summary missing required field 'test_confusion_matrix'"
            in runs[0]["diagnostics"])


def test_final_corrupt_metrics_invalid(tmp_path):
    """矩阵形状 / support / macro / 总数 / 对角线-准确率 / 浮点元素任一损坏 → invalid。"""
    def mutate(name, summary):
        if name == "shape":
            summary["test_confusion_matrix"][0] = [8, 2, 0]  # 行宽不一致 → 非方形
        elif name == "support":
            summary["test_per_class_metrics"][0]["support"] = 999
        elif name == "macro":
            summary["test_macro_avg"]["precision"] = 0.123456
        elif name == "total":
            summary["sample_counts"]["test"] = 9999
        elif name == "diag":
            summary["test_accuracy"] = 0.99
        elif name == "float_entry":
            summary["test_confusion_matrix"][0][0] = 90.0
        return summary

    for mutation in ("shape", "support", "macro", "total", "diag", "float_entry"):
        root = tmp_path / f"outputs_{mutation}"
        matrix = _final_matrix()
        config = _make_config(experiment_role="final", evaluate_test=True, model_seed=404)
        config["network"]["output_dim"] = len(matrix)
        total = sum(sum(row) for row in matrix)
        config["data"]["sample_counts"]["test"] = total
        summary = {**_make_summary(experiment_role="final", test_evaluation="executed"),
                   **_final_metrics_fields(matrix)}
        summary["sample_counts"]["test"] = total
        summary = mutate(mutation, summary)
        _write_run(root, "final_bad", config=config, summary=summary)
        report = aggregate(root=root, output=root / "_aggregate", strict=True)
        runs = _read_json(root / "_aggregate" / "runs.json")["runs"]
        assert report["counts"]["invalid"] == 1, mutation
        assert runs[0]["status"] == "invalid", mutation
        assert runs[0]["diagnostics"], mutation


def test_final_matrix_dim_must_match_output_dim(tmp_path):
    root = tmp_path / "outputs"
    config = _make_config(experiment_role="final", evaluate_test=True, model_seed=404)
    config["network"]["output_dim"] = 3  # 与 2×2 矩阵不一致
    config["data"]["sample_counts"]["test"] = 100
    summary = {**_make_summary(experiment_role="final", test_evaluation="executed"),
               **_final_metrics_fields(_final_matrix())}
    summary["sample_counts"]["test"] = 100
    _write_run(root, "final_dim", config=config, summary=summary)
    aggregate(root=root, output=root / "_aggregate", strict=True)
    runs = _read_json(root / "_aggregate" / "runs.json")["runs"]
    assert runs[0]["status"] == "invalid"
    assert any("output_dim" in d for d in runs[0]["diagnostics"])


def test_non_final_and_legacy_without_fields_unaffected(tmp_path):
    """baseline/exploration 缺新字段仍为 success；legacy 判定不变。"""
    root = tmp_path / "outputs"
    _write_run(root, "base", config=_make_config(experiment_role="baseline", evaluate_test=True),
               summary=_make_summary(experiment_role="baseline", test_evaluation="executed",
                                     test_loss=0.1, test_accuracy=0.95))
    _write_run(root, "exp", config=_make_config(), summary=_make_summary())
    legacy_config = _make_config()
    legacy_config.pop("experiment_role")
    legacy_summary = _make_summary()
    legacy_summary.pop("test_evaluation")
    _write_run(root, "old", config=legacy_config, summary=legacy_summary)
    report = aggregate(root=root, output=root / "_aggregate", strict=True)
    assert report["counts"] == {"success": 2, "incomplete": 0, "invalid": 0, "legacy": 1}
    runs = {r["run_name"]: r for r in _read_json(root / "_aggregate" / "runs.json")["runs"]}
    assert runs["base"]["status"] == "success" and runs["exp"]["status"] == "success"
    assert runs["old"]["status"] == "legacy"
    for name in ("base", "exp", "old"):
        assert runs[name]["test_confusion_matrix"] is None


def test_three_final_runs_group_stats(tmp_path):
    """3 个合成 final run → 单一 3-seed 组；组级 macro/逐类统计可独立复算。"""
    root = tmp_path / "outputs"
    matrix_a = [[90, 0], [0, 10]]   # acc 1.0；class1 predicted=10, TP=10 → precision 1.0
    matrix_b = [[80, 10], [10, 0]]  # acc 0.8；class1 predicted=10, TP=0  → precision 0.0
    matrix_c = [[70, 20], [0, 10]]  # acc 0.8；class0 p=1.0, r=7/9      → f1 = 7/8
    for name, seed, matrix, loss in (("f1", 404, matrix_a, 0.1),
                                     ("f2", 505, matrix_b, 0.3),
                                     ("f3", 606, matrix_c, 0.3)):
        _make_final_run(root, name, model_seed=seed, matrix=matrix, test_loss=loss)
    report = aggregate(root=root, output=root / "_aggregate", strict=True)
    assert report["counts"]["success"] == 3 and report["strict_failed"] is False

    runs = _read_json(root / "_aggregate" / "runs.json")["runs"]
    # 混淆矩阵保留在 run 级且互不相同（不跨 seed 求和）
    assert [r["test_confusion_matrix"] for r in runs] == [matrix_a, matrix_b, matrix_c]

    groups = _read_json(root / "_aggregate" / "groups.json")["groups"]
    assert len(groups) == 1
    group = groups[0]
    assert group["n_runs"] == 3 and group["experiment_role"] == "final"
    assert group["test_evaluation"] == "executed"
    stats = group["stats"]
    assert "test_confusion_matrix" not in stats  # 组级不得出现求和矩阵
    assert "per-model statistics on the fixed test set" in stats["test_classification_note"]
    assert "must not be summed across seeds" in stats["test_classification_note"]

    # 组级 macro：用 runs.json 的原始值经 statistics 独立复算
    macro_by_key = {key: [r["test_macro_avg"][key] for r in runs]
                    for key in ("precision", "recall", "f1")}
    for key, values in macro_by_key.items():
        entry = stats["test_macro_avg"][key]
        assert entry["n"] == 3
        assert entry["mean"] == pytest.approx(statistics.fmean(values))
        assert entry["std"] == pytest.approx(statistics.pstdev(values))

    # 逐类统计：键为类别 0..C-1；每类 mean/std/n 可独立复算
    assert set(stats["test_per_class_metrics"]) == {"0", "1"}
    per_class_by_key = {
        (cls, key): [r["test_per_class_metrics"][cls][key] for r in runs]
        for cls in range(2) for key in ("precision", "recall", "f1")
    }
    for (cls, key), values in per_class_by_key.items():
        entry = stats["test_per_class_metrics"][str(cls)][key]
        assert entry["mean"] == pytest.approx(statistics.fmean(values))
        assert entry["std"] == pytest.approx(statistics.pstdev(values))
        assert entry["n"] == 3

    # 手算抽查：零预测 precision=0；f1=7/8；support 是逐模型计数（不跨 seed 相加）
    runs_by_name = {r["run_name"]: r for r in runs}
    assert runs_by_name["f2"]["test_per_class_metrics"][1]["precision"] == 0.0
    assert runs_by_name["f3"]["test_per_class_metrics"][0]["f1"] == pytest.approx(7 / 8)
    assert stats["test_per_class_metrics"]["1"]["precision"]["mean"] == pytest.approx(
        (1.0 + 0.0 + 1 / 3) / 3
    )
    assert stats["test_per_class_metrics"]["0"]["support"] == {"mean": 90.0, "std": 0.0, "n": 3}
    assert stats["test_per_class_metrics"]["1"]["support"] == {"mean": 10.0, "std": 0.0, "n": 3}


# ---------------------------------------------------------------------------
# P5.3-E6-A-R1 新增：逐类 P/R/F1 必须与混淆矩阵重算值逐项一致。
# 冻结公式：tp=M[c][c]；support=行和；predicted=列和；predicted=0 → precision=0.0；
# support=0 → recall=0.0；P+R=0 → f1=0.0。容差 abs 1e-9、rel_tol=0.0。
# ---------------------------------------------------------------------------


def _tampered_final_summary(key, cls, wrong_value):
    """构造"仅某一类某个指标被篡改、macro 同步调整"的 final summary。

    混淆矩阵、support、总数、accuracy 与其它逐类值全部保持合法——证明校验
    捕获的是"逐类值与矩阵不符"，而非范围/macro 等既有检查。
    """
    matrix = _final_matrix()
    fields = _final_metrics_fields(matrix)
    per_class = [dict(entry) for entry in fields["test_per_class_metrics"]]
    assert wrong_value != per_class[cls][key]  # 篡改值必须确实偏离矩阵重算值
    per_class[cls][key] = wrong_value
    fields["test_per_class_metrics"] = per_class
    # 同步调整 macro，使其仍是篡改后逐类值的不加权平均
    fields["test_macro_avg"] = {
        k: sum(e[k] for e in per_class) / len(per_class)
        for k in ("precision", "recall", "f1")
    }
    return {**_make_summary(experiment_role="final", test_evaluation="executed"), **fields}


@pytest.mark.parametrize("key", ("precision", "recall", "f1"))
@pytest.mark.parametrize("cls", (0, 1))
def test_final_per_class_metrics_must_match_matrix(tmp_path, key, cls):
    """篡改某类 P/R/F1（范围内）并同步修正 macro → 仍必须判 invalid。"""
    root = tmp_path / "outputs"
    config = _make_config(experiment_role="final", evaluate_test=True, model_seed=404)
    summary = _tampered_final_summary(key, cls, 0.5)
    _write_run(root, "final_tampered", config=config, summary=summary)
    report = aggregate(root=root, output=root / "_aggregate", strict=True)
    assert report["counts"]["invalid"] == 1
    runs = _read_json(root / "_aggregate" / "runs.json")["runs"]
    assert runs[0]["status"] == "invalid"
    assert any(
        f"'test_per_class_metrics'[{cls}]" in d and f"{key!r}" in d
        and "recomputed from the confusion matrix" in d
        for d in runs[0]["diagnostics"]
    ), runs[0]["diagnostics"]


def test_zero_denominator_rules_consistent_with_matrix(tmp_path):
    """predicted=0 / support=0（含 predicted>0 组合）时冻结零分母值应通过。"""
    root = tmp_path / "outputs"
    # class1：support=0、predicted=0、tp=0 → p=r=f=0.0
    _make_final_run(root, "zd_never", model_seed=404, matrix=[[5, 0], [0, 0]], test_loss=0.2)
    # class1：support=0 但 predicted=3（列和来自 class0 的错分）→ p=r=f=0.0
    _make_final_run(root, "zd_no_support", model_seed=505, matrix=[[5, 3], [0, 0]],
                    test_loss=0.3)
    report = aggregate(root=root, output=root / "_aggregate", strict=True)
    assert report["counts"] == {"success": 2, "incomplete": 0, "invalid": 0, "legacy": 0}
    runs = {r["run_name"]: r for r in _read_json(root / "_aggregate" / "runs.json")["runs"]}
    for name in ("zd_never", "zd_no_support"):
        assert runs[name]["status"] == "success"
        assert runs[name]["test_per_class_metrics"][1]["precision"] == 0.0
        assert runs[name]["test_per_class_metrics"][1]["recall"] == 0.0
        assert runs[name]["test_per_class_metrics"][1]["f1"] == 0.0


@pytest.mark.parametrize(
    "matrix, key, wrong_value",
    [
        ([[5, 0], [0, 0]], "precision", 0.5),  # predicted=0 → 正确值为 0.0
        ([[5, 3], [0, 0]], "recall", 0.9),     # support=0 → 正确值为 0.0
        ([[5, 0], [0, 0]], "f1", 0.4),         # P+R=0 → 正确值为 0.0
    ],
)
def test_zero_denominator_violations_rejected(tmp_path, matrix, key, wrong_value):
    """零分母类别写入非零指标（macro 同步）→ 与矩阵重算不符，判 invalid。"""
    root = tmp_path / "outputs"
    fields = _final_metrics_fields(matrix)
    cls = 1
    assert fields["test_per_class_metrics"][cls][key] == 0.0  # 冻结规则下的正确值
    per_class = [dict(e) for e in fields["test_per_class_metrics"]]
    per_class[cls][key] = wrong_value
    fields["test_per_class_metrics"] = per_class
    fields["test_macro_avg"] = {
        k: sum(e[k] for e in per_class) / len(per_class)
        for k in ("precision", "recall", "f1")
    }
    config = _make_config(experiment_role="final", evaluate_test=True, model_seed=404)
    config["network"]["output_dim"] = len(matrix)
    total = sum(sum(row) for row in matrix)
    config["data"]["sample_counts"]["test"] = total
    summary = {**_make_summary(experiment_role="final", test_evaluation="executed"), **fields}
    summary["sample_counts"]["test"] = total
    _write_run(root, "zd_bad", config=config, summary=summary)
    report = aggregate(root=root, output=root / "_aggregate", strict=True)
    assert report["counts"]["invalid"] == 1
    runs = _read_json(root / "_aggregate" / "runs.json")["runs"]
    assert any(
        f"'test_per_class_metrics'[{cls}]" in d and f"{key!r}" in d
        and "recomputed from the confusion matrix" in d
        for d in runs[0]["diagnostics"]
    ), runs[0]["diagnostics"]


def test_final_per_class_inconsistency_default_vs_strict(tmp_path, capsys):
    """默认模式保留诊断并标 invalid；--strict 退出码非零。"""
    root = tmp_path / "outputs"
    config = _make_config(experiment_role="final", evaluate_test=True, model_seed=404)
    summary = _tampered_final_summary("precision", 0, 0.5)
    _write_run(root, "final_tampered", config=config, summary=summary)
    # 默认模式：退出码 0、诊断保留、状态 invalid
    report_default = aggregate(root=root, output=root / "_agg_default")
    assert report_default["counts"]["invalid"] == 1
    assert report_default["strict_failed"] is False
    runs = _read_json(root / "_agg_default" / "runs.json")["runs"]
    assert runs[0]["status"] == "invalid"
    assert any("recomputed from the confusion matrix" in d for d in runs[0]["diagnostics"])
    # --strict：strict_failed=True 且 CLI 非零退出
    report_strict = aggregate(root=root, output=root / "_agg_strict", strict=True)
    assert report_strict["strict_failed"] is True
    assert main(["aggregate", "--root", str(root),
                 "--output", str(root / "_agg_cli"), "--strict"]) == 1


def test_metric_consistency_fix_not_extended_to_non_final(tmp_path):
    """修复不得扩展到旧记录：exploration 携带（即便不一致的）额外字段仍按既有 schema 处理。"""
    root = tmp_path / "outputs"
    summary = _make_summary()
    summary["test_confusion_matrix"] = [[1, 0], [0, 1]]
    summary["test_per_class_metrics"] = [
        {"precision": 0.1, "recall": 0.2, "f1": 0.3, "support": 1},
        {"precision": 0.4, "recall": 0.5, "f1": 0.6, "support": 1},
    ]
    summary["test_macro_avg"] = {"precision": 0.7, "recall": 0.8, "f1": 0.9}
    _write_run(root, "exp_extra", config=_make_config(), summary=summary)
    report = aggregate(root=root, output=root / "_aggregate", strict=True)
    assert report["counts"]["success"] == 1
    runs = _read_json(root / "_aggregate" / "runs.json")["runs"]
    assert runs[0]["status"] == "success" and runs[0]["experiment_role"] == "exploration"