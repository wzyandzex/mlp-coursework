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
               summary=_make_summary(experiment_role="final",
                                     test_evaluation="executed",
                                     test_loss=0.2, test_accuracy=0.93))
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
