"""跨 run 结果汇总器（P5.2-D）：扫描、校验、分组、统计与报告。

本模块是只读工具：只读取已保存的 config.json / summary.json / history.csv，
校验、汇总并输出 runs/groups 摘要。它绝不训练模型、绝不评估测试集、绝不
根据任何指标选择"最佳实验"，也不修改任何源 run 产物。

运行角色与测试集隔离纪律（与 P5.2-C 契约一致，违反即 invalid）
------------------------------------------------
* exploration：validation-only——test_evaluation 必须为 "not_run"，
  test_loss / test_accuracy 必须为 JSON null；
* baseline / final：test_evaluation 必须为 "executed"，且测试指标必须是
  有限数值；
* null 不是 0：缺失或未评估的指标不参与均值/标准差，也不被补零；
* 旧 schema（缺少 experiment_role 或 test_evaluation 字段）标记为 legacy，
  不自动推导其测试集状态。

确定性
------
* run 目录按相对路径字典序扫描与排序（符号链接去重）；
* JSON 输出固定键序/缩进且 allow_nan=False（标准 JSON，无 NaN/Infinity）；
* CSV 列序固定；不写入任何时间戳，相同输入重复运行产生逐字节相同输出；
* 汇总输出使用临时文件 + os.replace 原子替换；路径一律相对 --root。
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import sys
from pathlib import Path

SCHEMA_VERSION = "1.0"

STATUS_SUCCESS = "success"
STATUS_INCOMPLETE = "incomplete"
STATUS_INVALID = "invalid"
STATUS_LEGACY = "legacy"

ROLES = ("baseline", "exploration", "final")
EVALUATION_STATES = ("executed", "not_run")
# 角色 → 是否必须执行测试评估（P5.2-C 契约）
ROLE_EXPECTS_EVALUATION = {"baseline": True, "exploration": False, "final": True}

RUN_CSV_FIELDS = (
    "run_name",
    "run_path",
    "status",
    "experiment_role",
    "test_evaluation",
    "optimizer",
    "learning_rate",
    "lr_scheduler",
    "early_stopping",
    "patience",
    "min_delta",
    "l2_lambda",
    "network_json",
    "model_seed",
    "split_seed",
    "history_rows",
    "best_epoch",
    "best_validation_loss",
    "best_validation_accuracy",
    "stop_reason",
    "epochs_completed",
    "total_seconds",
    "test_loss",
    "test_accuracy",
    "diagnostics_json",
    "config_json",
)

GROUP_CSV_FIELDS = (
    "group_id",
    "n_runs",
    "experiment_role",
    "test_evaluation",
    "best_validation_loss_mean",
    "best_validation_loss_std",
    "best_validation_accuracy_mean",
    "best_validation_accuracy_std",
    "epochs_completed_mean",
    "epochs_completed_std",
    "total_seconds_mean",
    "total_seconds_std",
    "best_epoch_mean",
    "best_epoch_std",
    "test_loss_mean",
    "test_loss_std",
    "test_loss_n",
    "test_accuracy_mean",
    "test_accuracy_std",
    "test_accuracy_n",
    "run_paths_json",
    "group_config_json",
)


def _reject_json_constant(name: str) -> float:
    """json.loads 的 parse_constant 钩子：NaN/Infinity 不是合法输入。"""
    raise ValueError(f"non-standard JSON numeric constant {name!r} (NaN/Infinity is not allowed)")


def _is_finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _load_json_file(path: Path):
    """读取并解析 JSON；返回 (data, error)。解析失败时 data 为 None。

    parse_constant 钩子把 NaN/Infinity 字面量变成解析错误（它们不是标准 JSON）。
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"cannot read {path.name}: {exc}"
    try:
        return json.loads(text, parse_constant=_reject_json_constant), None
    except ValueError as exc:
        return None, f"invalid JSON in {path.name}: {exc}"


def _dumpable(data) -> bool:
    """数据能否以标准 JSON（无 NaN/Infinity）序列化。"""
    try:
        json.dumps(data, allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False


def discover_run_dirs(root: Path, exclude: Path | None = None) -> list:
    """发现 root 下的 run 目录（深度 1）。

    * 含 config.json 或 summary.json 的子目录才算 run，避免把任意目录/文件
      当作实验；
    * exclude（汇总输出目录）与符号链接重复指向的目录被显式排除；
    * 返回按相对路径（posix）字典序排序的列表。
    """
    exclude_resolved = exclude.resolve() if exclude is not None else None
    run_dirs = []
    seen_resolved = set()
    for entry in sorted(root.iterdir(), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        resolved = entry.resolve()
        if exclude_resolved is not None and resolved == exclude_resolved:
            continue
        if resolved in seen_resolved:  # 符号链接重复指向同一目录
            continue
        seen_resolved.add(resolved)
        if (entry / "config.json").is_file() or (entry / "summary.json").is_file():
            run_dirs.append(entry)
    return sorted(run_dirs, key=lambda item: item.relative_to(root).as_posix())


def _sanitize_json_value(value):
    """把非有限浮点（NaN/Inf，例如 1e400 字面量解析结果）替换为 None，
    保证记录可以被写为标准 JSON（allow_nan=False）。"""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    return value


def _fill_fields(record: dict, config: dict | None, summary: dict | None) -> None:
    """尽力从（可能不完整或损坏的）config/summary 提取字段；缺失即为 None。"""
    training = config.get("training") if isinstance(config, dict) else None
    if isinstance(training, dict):
        for key in ("optimizer", "learning_rate", "lr_scheduler", "early_stopping",
                    "patience", "min_delta", "l2_lambda"):
            record[key] = _sanitize_json_value(training.get(key))
    seeds = config.get("seeds") if isinstance(config, dict) else None
    if isinstance(seeds, dict):
        record["model_seed"] = _sanitize_json_value(seeds.get("model_seed"))
        record["split_seed"] = _sanitize_json_value(seeds.get("split_seed"))
    if isinstance(config, dict):
        record["network"] = _sanitize_json_value(config.get("network"))
    if isinstance(summary, dict):
        for key in ("best_epoch", "best_validation_loss", "best_validation_accuracy",
                    "stop_reason", "epochs_completed", "total_seconds",
                    "test_loss", "test_accuracy"):
            record[key] = _sanitize_json_value(summary.get(key))
        if record["experiment_role"] is None:
            record["experiment_role"] = summary.get("experiment_role")


def _validate_current_schema(config: dict, summary: dict, diagnostics: list) -> bool:
    """P5.2-C schema 的完整校验：返回 True 表示可作为 success 记录。"""
    ok = True
    role = config.get("experiment_role")
    if role not in ROLES:
        diagnostics.append(f"invalid experiment_role {role!r} in config.json")
        ok = False
    eval_state = summary.get("test_evaluation")
    if eval_state not in EVALUATION_STATES:
        diagnostics.append(f"invalid test_evaluation {eval_state!r} in summary.json")
        ok = False
    summary_role = summary.get("experiment_role")
    if summary_role != role:
        diagnostics.append(
            f"experiment_role mismatch between config.json ({role!r}) and "
            f"summary.json ({summary_role!r})"
        )
        ok = False
    if role in ROLE_EXPECTS_EVALUATION and eval_state in EVALUATION_STATES:
        expects_executed = ROLE_EXPECTS_EVALUATION[role]
        if (eval_state == "executed") != expects_executed:
            diagnostics.append(
                f"experiment_role {role!r} requires test_evaluation="
                f"{'executed' if expects_executed else 'not_run'}, got {eval_state!r}"
            )
            ok = False
    config_evaluate = config.get("evaluate_test")
    if not isinstance(config_evaluate, bool):
        diagnostics.append(f"evaluate_test in config.json must be a bool, got {config_evaluate!r}")
        ok = False
    elif role in ROLE_EXPECTS_EVALUATION and config_evaluate != ROLE_EXPECTS_EVALUATION[role]:
        diagnostics.append(
            f"experiment_role {role!r} is inconsistent with evaluate_test={config_evaluate!r} "
            f"in config.json (expected {ROLE_EXPECTS_EVALUATION[role]!r})"
        )
        ok = False
    if eval_state == "executed":
        for name in ("test_loss", "test_accuracy"):
            value = summary.get(name)
            if not _is_finite_number(value):
                diagnostics.append(
                    f"summary field {name!r} must be a finite number when test_evaluation "
                    f"is 'executed', got {value!r}"
                )
                ok = False
    elif eval_state == "not_run":
        for name in ("test_loss", "test_accuracy"):
            value = summary.get(name)
            if value is not None:
                diagnostics.append(
                    f"summary field {name!r} must be null when test_evaluation is "
                    f"'not_run', got {value!r}"
                )
                ok = False
    for name in ("best_validation_loss", "best_validation_accuracy",
                 "epochs_completed", "total_seconds"):
        if not _is_finite_number(summary.get(name)):
            diagnostics.append(
                f"summary field {name!r} must be a finite number, got {summary.get(name)!r}"
            )
            ok = False
    best_epoch = summary.get("best_epoch")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int):
        diagnostics.append(f"summary field 'best_epoch' must be an integer, got {best_epoch!r}")
        ok = False
    stop_reason = summary.get("stop_reason")
    if stop_reason not in ("max_epochs", "early_stopping"):
        diagnostics.append(f"summary field 'stop_reason' has unexpected value {stop_reason!r}")
        ok = False
    for name, section in (("training", config.get("training")),
                          ("seeds", config.get("seeds")),
                          ("network", config.get("network"))):
        if not isinstance(section, dict):
            diagnostics.append(f"config.json missing {name!r} object")
            ok = False
    return ok


def build_run_record(run_dir: Path, root: Path) -> dict:
    """读取并校验单个 run 目录，生成标准化的 run-level 记录。

    状态判定顺序：文件缺失 → incomplete；JSON 损坏或含 NaN/Infinity → invalid；
    缺少 experiment_role / test_evaluation 字段 → legacy（旧 schema）；
    其余按 P5.2-C 契约完整校验 → success 或 invalid。任何失败路径都不静默修正。
    """
    record = {
        "run_name": run_dir.name,
        "run_path": run_dir.relative_to(root).as_posix(),
        "status": None,
        "experiment_role": None,
        "test_evaluation": None,
        "optimizer": None,
        "learning_rate": None,
        "lr_scheduler": None,
        "early_stopping": None,
        "patience": None,
        "min_delta": None,
        "l2_lambda": None,
        "network": None,
        "model_seed": None,
        "split_seed": None,
        "history_rows": None,
        "best_epoch": None,
        "best_validation_loss": None,
        "best_validation_accuracy": None,
        "stop_reason": None,
        "epochs_completed": None,
        "total_seconds": None,
        "test_loss": None,
        "test_accuracy": None,
        "diagnostics": [],
        "config": None,
    }
    diagnostics = record["diagnostics"]
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    config_missing = not config_path.is_file()
    summary_missing = not summary_path.is_file()
    if config_missing:
        diagnostics.append("missing config.json")
    if summary_missing:
        diagnostics.append("missing summary.json")
    history_path = run_dir / "history.csv"
    if history_path.is_file():
        # history.csv 是可选输入：缺失只记诊断，不影响 summary 有效字段的读取
        try:
            with open(history_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames
                if fieldnames is None or "epoch" not in fieldnames:
                    diagnostics.append("history.csv has no 'epoch' column")
                else:
                    record["history_rows"] = sum(1 for _ in reader)
        except (OSError, csv.Error) as exc:
            diagnostics.append(f"cannot read history.csv: {exc}")
    else:
        diagnostics.append("history.csv missing (optional for aggregation)")

    config = summary = None
    parse_errors = []
    if not config_missing:
        config, error = _load_json_file(config_path)
        if error is not None:
            parse_errors.append(error)
    if not summary_missing:
        summary, error = _load_json_file(summary_path)
        if error is not None:
            parse_errors.append(error)
    # 非标准 JSON 数值（NaN/Infinity 字面量或 1e400 溢出成 Inf）一旦出现在
    # 任何已解析文件中，该 run 即为 invalid
    for name, data in (("config.json", config), ("summary.json", summary)):
        if isinstance(data, dict) and not _dumpable(data):
            parse_errors.append(f"{name} contains values outside standard JSON (NaN/Infinity)")
    if parse_errors:
        record["status"] = STATUS_INVALID
        diagnostics.extend(parse_errors)
        _fill_fields(record, config if isinstance(config, dict) else {},
                     summary if isinstance(summary, dict) else {})
        return record

    if config_missing or summary_missing:
        record["status"] = STATUS_INCOMPLETE
        _fill_fields(record, config if isinstance(config, dict) else {},
                     summary if isinstance(summary, dict) else {})
        return record

    config = record["config"] = _sanitize_json_value(config) if isinstance(config, dict) else None
    summary = _sanitize_json_value(summary) if isinstance(summary, dict) else None
    if config is None or summary is None:
        # JSON 合法但不是对象（如数组/标量）：无法按 schema 校验
        record["status"] = STATUS_INVALID
        diagnostics.append("config.json and summary.json must both contain JSON objects")
        _fill_fields(record, config or {}, summary or {})
        return record

    # 旧 schema：缺少角色/评估状态字段 → legacy，不自动推导测试集状态
    if config.get("experiment_role") is None or summary.get("test_evaluation") is None:
        record["status"] = STATUS_LEGACY
        missing = []
        if config.get("experiment_role") is None:
            missing.append("experiment_role in config.json")
        if summary.get("test_evaluation") is None:
            missing.append("test_evaluation in summary.json")
        diagnostics.append(
            "run predates the P5.2-C schema (missing " + "; ".join(missing) + "); "
            "its test-set state cannot be verified and is not derived"
        )
        _fill_fields(record, config, summary)
        return record

    if _validate_current_schema(config, summary, diagnostics):
        record["status"] = STATUS_SUCCESS
    else:
        record["status"] = STATUS_INVALID
    record["experiment_role"] = config.get("experiment_role")
    record["test_evaluation"] = summary.get("test_evaluation")
    _fill_fields(record, config, summary)
    return record


def _group_payload(config: dict) -> dict:
    """规范化分组配置：完整 config 的规范化拷贝，仅移除 seeds.model_seed。"""
    payload = json.loads(json.dumps(config, sort_keys=True, allow_nan=False))
    seeds = payload.get("seeds")
    if isinstance(seeds, dict):
        seeds.pop("model_seed", None)
    return payload


def _mean_std(values: list) -> dict:
    """均值与总体标准差（ddof=0）；空列表返回 null 并由调用方补充原因。"""
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": statistics.fmean(values), "std": statistics.pstdev(values), "n": n}


def build_groups(success_records: list) -> list:
    """对成功 run 按规范化配置（仅排除 seeds.model_seed）分组并统计。

    * group_id = 规范化分组 JSON 的 SHA-256 前 12 位（前缀 'g'），
      同时保留完整规范化配置，确保可追溯；
    * 标准差为总体标准差（ddof=0），单 run 组为 0.0；
    * 测试指标统计只对实际执行了测试评估的组计算；exploration 组为 null
      并记录原因。
    """
    grouped: dict[str, dict] = {}
    for record in success_records:
        payload = _group_payload(record["config"])
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False)
        group_id = "g" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
        group = grouped.setdefault(
            group_id,
            {"group_id": group_id, "group_config": payload,
             "canonical_config": canonical, "members": []},
        )
        group["members"].append(record)

    groups = []
    for group_id in sorted(grouped):
        group = grouped[group_id]
        members = sorted(group["members"], key=lambda item: item["run_path"])
        stats = {}
        for metric in ("best_validation_loss", "best_validation_accuracy",
                       "epochs_completed", "total_seconds", "best_epoch"):
            stats[metric] = _mean_std([member[metric] for member in members])
        test_evaluation = members[0]["test_evaluation"]
        if test_evaluation == "executed":
            for metric in ("test_loss", "test_accuracy"):
                values = [member[metric] for member in members
                          if isinstance(member.get(metric), (int, float))
                          and not isinstance(member.get(metric), bool)]
                entry = _mean_std(values)
                if entry["n"] < len(members):
                    entry["reason"] = "some runs in this group have missing test metrics"
                stats[metric] = entry
        else:
            reason = (
                "test_evaluation is 'not_run' for this group; test metrics are "
                "excluded from statistics (validation-only runs)"
            )
            stats["test_loss"] = {"mean": None, "std": None, "n": 0, "reason": reason}
            stats["test_accuracy"] = {"mean": None, "std": None, "n": 0, "reason": reason}
        groups.append(
            {
                "group_id": group_id,
                "group_config": group["group_config"],
                "canonical_config": group["canonical_config"],
                "run_paths": [member["run_path"] for member in members],
                "n_runs": len(members),
                "experiment_role": members[0]["experiment_role"],
                "test_evaluation": test_evaluation,
                "stats": stats,
            }
        )
    return groups


def _atomic_write(path: Path, text: str) -> None:
    """临时文件写入后 os.replace 原子替换，避免留下半截输出文件。"""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(tmp, path)


def _write_json(path: Path, payload) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False) + "\n"
    _atomic_write(path, text)


def _write_csv(path: Path, fieldnames: tuple, rows: list) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    _atomic_write(path, buffer.getvalue())


def _csv_value(value):
    """CSV 单元格的确定性序列化：None → 空单元格，bool → true/false，
    嵌套结构 → 紧凑规范化 JSON（不是 Python repr）。空单元格表示 null/缺失，
    权威取值以 runs.json / groups.json 为准。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    return str(value)


def _run_csv_row(record: dict) -> dict:
    row = {key: _csv_value(record.get(key)) for key in RUN_CSV_FIELDS
           if key not in ("network_json", "diagnostics_json", "config_json")}
    row["network_json"] = _csv_value(record.get("network"))
    row["diagnostics_json"] = _csv_value(record.get("diagnostics"))
    row["config_json"] = _csv_value(record.get("config"))
    return row


def _group_csv_row(group: dict) -> dict:
    row = {key: _csv_value(group.get(key)) for key in GROUP_CSV_FIELDS
           if key not in ("run_paths_json", "group_config_json")}
    for metric in ("best_validation_loss", "best_validation_accuracy",
                   "epochs_completed", "total_seconds", "best_epoch",
                   "test_loss", "test_accuracy"):
        entry = group["stats"][metric]
        row[f"{metric}_mean"] = _csv_value(entry["mean"])
        row[f"{metric}_std"] = _csv_value(entry["std"])
        if "n" in entry:
            row[f"{metric}_n"] = _csv_value(entry["n"])
    row["run_paths_json"] = _csv_value(group["run_paths"])
    row["group_config_json"] = _csv_value(group["group_config"])
    return row


def aggregate(root, output, strict: bool = False, overwrite: bool = False) -> dict:
    """扫描 root 下的 run 目录，校验、汇总并把输出写入 output 目录。

    只读取源 run 产物；输出写入独立目录（已存在时默认拒绝，除非 overwrite，
    且 overwrite 只替换汇总文件本身）。返回 report 字典；冲突与非法输入以
    ValueError 报告，由 CLI 转换为退出码。
    """
    root = Path(root)
    output = Path(output)
    if not root.is_dir():
        raise ValueError(f"input root {str(root)!r} is not an existing directory.")

    # 先做全量发现与冲突检测（不对输出目录排除），再排除汇总输出目录本身，
    # 否则"输出目录恰好是某个源 run 目录"的冲突会被漏检
    all_run_dirs = discover_run_dirs(root)
    root_resolved = root.resolve()
    output_resolved = output.resolve()
    if output_resolved == root_resolved:
        raise ValueError("output directory must not be the same as the input root.")
    for run_dir in all_run_dirs:
        run_resolved = run_dir.resolve()
        if output_resolved == run_resolved or run_resolved in output_resolved.parents:
            raise ValueError(
                f"output directory {str(output)!r} overlaps source run directory "
                f"{run_dir.relative_to(root).as_posix()!r}; source runs must never "
                f"be overwritten."
            )
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"output path {str(output)!r} exists and is not a directory.")
        if not overwrite:
            raise ValueError(
                f"output directory {str(output)!r} already exists; pass --overwrite "
                f"to replace its aggregation files (source runs are never touched)."
            )
    run_dirs = [run_dir for run_dir in all_run_dirs
                if run_dir.resolve() != output_resolved]  # 汇总输出目录自身不算 run

    records = [build_run_record(run_dir, root) for run_dir in run_dirs]
    success_records = [record for record in records if record["status"] == STATUS_SUCCESS]
    groups = build_groups(success_records)
    counts = {
        "success": sum(1 for r in records if r["status"] == STATUS_SUCCESS),
        "incomplete": sum(1 for r in records if r["status"] == STATUS_INCOMPLETE),
        "invalid": sum(1 for r in records if r["status"] == STATUS_INVALID),
        "legacy": sum(1 for r in records if r["status"] == STATUS_LEGACY),
    }
    strict_failed = bool(strict and any(
        record["status"] in (STATUS_INCOMPLETE, STATUS_INVALID) for record in records
    ))
    report = {
        "schema_version": SCHEMA_VERSION,
        "total_runs": len(records),
        "counts": counts,
        "n_groups": len(groups),
        "strict": bool(strict),
        "strict_failed": strict_failed,
        "problems": [
            {"run_path": record["run_path"], "status": record["status"],
             "diagnostics": record["diagnostics"]}
            for record in records if record["status"] != STATUS_SUCCESS
        ],
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "report.json", report)
    _write_json(output / "runs.json", {"schema_version": SCHEMA_VERSION, "runs": records})
    _write_json(output / "groups.json", {"schema_version": SCHEMA_VERSION, "groups": groups})
    _write_csv(output / "runs.csv", RUN_CSV_FIELDS,
               [_run_csv_row(record) for record in records])
    _write_csv(output / "groups.csv", GROUP_CSV_FIELDS,
               [_group_csv_row(group) for group in groups])
    return report


_ISOLATION_RULES = """test-set isolation rules:
  exploration runs are validation-only: their summary must record
  test_evaluation='not_run' with null test_loss/test_accuracy; baseline and
  final runs must have executed the test evaluation with finite metrics.
  This tool validates that contract, marks contradicting runs as invalid,
  never re-evaluates a test set, and never treats null as 0.
  Group statistics for test metrics are computed only for groups whose runs
  actually executed the test evaluation.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.experiments",
        description="Cross-run aggregation for the hand-written MLP coursework "
        "(read-only: it never trains, evaluates or selects models).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    aggregate_parser = subparsers.add_parser(
        "aggregate",
        help="Scan a run-output root and write runs/groups summaries.",
        description=(
            "Discover run directories under --root (depth 1, containing config.json "
            "or summary.json), validate each against the P5.2-C schema, write "
            "runs.csv/runs.json/groups.csv/groups.json/report.json into --output, "
            "and report diagnostics. Source run directories are never modified."
        ),
        epilog=_ISOLATION_RULES
        + "examples:\n"
        + "  python -m src.experiments aggregate --root outputs --output outputs/_aggregate\n",
    )
    aggregate_parser.add_argument(
        "--root", required=True, type=Path,
        help="Existing directory that contains the run directories to aggregate.",
    )
    aggregate_parser.add_argument(
        "--output", required=True, type=Path,
        help="Directory for the aggregation outputs. It must not overlap any source "
        "run directory; if it already exists, --overwrite is required.",
    )
    aggregate_parser.add_argument(
        "--strict", action="store_true",
        help="Exit with a non-zero code when any run is incomplete or invalid "
        "(diagnostics are still written for every run).",
    )
    aggregate_parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace the aggregation files in --output if it already exists "
        "(source run directories are never touched).",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "aggregate":
        parser.error(f"unknown command {args.command!r}")
    try:
        report = aggregate(root=args.root, output=args.output,
                           strict=args.strict, overwrite=args.overwrite)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    counts = report["counts"]
    print(
        f"[aggregate] runs: total={report['total_runs']} "
        f"success={counts['success']} incomplete={counts['incomplete']} "
        f"invalid={counts['invalid']} legacy={counts['legacy']}"
    )
    print(f"[aggregate] groups: {report['n_groups']}")
    for problem in report["problems"]:
        print(f"[aggregate] {problem['status']}: {problem['run_path']} "
              f"({'; '.join(problem['diagnostics'])})")
    if report["strict"] and report["strict_failed"]:
        print("[aggregate] strict mode: incomplete/invalid runs found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
