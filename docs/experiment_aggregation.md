# 跨 run 结果汇总器使用文档（P5.2-D）

`src/experiments.py` 是一个**只读**的跨 run 汇总工具：扫描运行输出根目录下的
run 目录，校验每个 run 的 config.json / summary.json（可选读取 history.csv），
输出机器可读与表格化的 run-level / group-level 汇总。它**不会训练模型、
不会评估测试集、不会根据任何指标选择"最佳实验"**，也不修改任何源 run 产物。

## CLI 调用方式

```bash
# 推荐用法：汇总 outputs 下的全部 run，输出到 outputs/_aggregate
python -m src.experiments aggregate --root outputs --output outputs/_aggregate

# 严格模式：存在 incomplete/invalid run 时以非零退出码结束（诊断仍会完整输出）
python -m src.experiments aggregate --root outputs --output outputs/_aggregate --strict

# 已存在汇总目录时，需显式允许替换汇总文件（只影响汇总文件本身）
python -m src.experiments aggregate --root outputs --output outputs/_aggregate --overwrite

python -m src.experiments aggregate --help   # 参数与测试集隔离规则说明
```

| 参数 | 说明 |
| --- | --- |
| `--root` | 必须存在的输入根目录，其深度 1 子目录中含 `config.json` 或 `summary.json` 者被识别为 run |
| `--output` | 汇总输出目录；不得与输入根相同、不得与任何源 run 目录重合（含位于 run 目录内部）；已存在时需 `--overwrite` |
| `--strict` | 存在 incomplete/invalid run 时返回非零退出码；默认模式仍输出完整诊断与其它有效 run 的汇总 |
| `--overwrite` | 替换 `--output` 中已存在的汇总文件；绝不触碰源 run 目录 |

## run 发现规则

* 仅扫描 `--root` 的**深度 1 子目录**；含 `config.json` **或** `summary.json`
  的子目录被识别为 run（两者皆无的目录——如 `_aggregate` 自身、杂项文件——不是 run）；
* 汇总输出目录被显式排除，二次运行不会把它识别为 run；
* 符号链接重复指向同一目录时只计一次；扫描与输出均按相对路径（posix）字典序稳定排序。

## run 状态枚举

| 状态 | 含义 |
| --- | --- |
| `success` | config/summary 齐全、可解析、且完全通过 P5.2-C 契约校验（含角色-评估-指标三角一致） |
| `incomplete` | 缺少 config.json 或 summary.json（例如训练中断的 run） |
| `invalid` | JSON 损坏、包含 NaN/Infinity、schema 字段缺失/类型错误，或角色、test_evaluation 与测试指标互相矛盾（**绝不静默修正**） |
| `legacy` | 旧 schema（缺少 `experiment_role` 或 `test_evaluation` 字段，如 P4/P5.2-B 时代的 run）；已有数值原样保留，但不参与分组与测试统计，其测试集状态**不自动推导** |

默认模式下单个损坏 run 不会影响其它有效 run 的汇总；`--strict` 会在完成全部
诊断后返回非零退出码。

## 测试集隔离纪律

汇总器按 P5.2-C 契约对每个 run 做三角一致性校验：

* `exploration` ↔ `test_evaluation="not_run"` ↔ `test_loss`/`test_accuracy` 为 JSON null；
* `baseline` / `final` ↔ `test_evaluation="executed"` ↔ 测试指标为有限数值；
* config 与 summary 中的 `experiment_role`、`evaluate_test` 必须与上述规则一致。

矛盾（例如 exploration 声称已评估、executed 却给出 null、not_run 却给出 0）
会使该 run 变为 `invalid` 并附明确诊断。**null 不等于 0**：未评估的指标不参与
均值/标准差，也不会被补零。汇总器只读取已保存的指标。

## 输出 schema

输出目录包含 5 个文件（原子写入：临时文件 + `os.replace`）：

### runs.json / runs.csv（run-level，每 run 一条）

字段：`run_name`、`run_path`（相对 `--root` 的 posix 路径）、`status`、
`experiment_role`、`test_evaluation`、`optimizer`、`learning_rate`、`lr_scheduler`、
`early_stopping`、`patience`、`min_delta`、`l2_lambda`、`network`（含结构与激活函数）、
`model_seed`、`split_seed`、`history_rows`、`best_epoch`、`best_validation_loss`、
`best_validation_accuracy`、`stop_reason`、`epochs_completed`、`total_seconds`、
`test_loss`、`test_accuracy`、`diagnostics`（诊断字符串数组）、`config`（完整原始配置）。

CSV 与 JSON 字段一一对应；CSV 中 None 写作空单元格、bool 写作 true/false、
嵌套结构写作紧凑规范化 JSON 字符串。**CSV 空单元格表示 null/缺失，权威取值以
JSON 为准。**

### groups.json / groups.csv（group-level，每组一条）

只有 `status=success` 的 run 参与分组。分组键 = **规范化完整 config 仅移除
`seeds.model_seed`**——即网络结构、激活/初始化、全部训练超参（optimizer、
learning_rate、scheduler、early_stopping、patience、min_delta、l2_lambda 等）、
数据标识与切分、`seeds.split_seed`、experiment_role、evaluate_test 必须完全一致
才合并；不同 experiment_role 永不同组。

每个分组包含：`group_id`（规范化分组 JSON 的 SHA-256 前 12 位，前缀 `g`）、
`group_config`（完整规范化配置，不只存哈希）、`canonical_config`、`run_paths`、
`n_runs`、`experiment_role`、`test_evaluation` 与 `stats`。

`stats` 对 `best_validation_loss`、`best_validation_accuracy`、`epochs_completed`、
`total_seconds`、`best_epoch`（描述性）给出 `mean`/`std`/`n`：**标准差为总体
标准差（ddof=0），单 run 组为 0.0**。`test_loss`/`test_accuracy` 的统计只对
实际执行了测试评估的组计算；exploration 组为 null 并附 `reason`；部分缺失时
记录有效样本数 `n`，不补零。

### report.json

`schema_version`、`total_runs`、`counts`（四种状态的数量）、`n_groups`、
`strict`、`strict_failed`、`problems`（每个非 success run 的路径/状态/诊断）。
不包含时间戳：相同输入重复运行产生逐字节相同的输出。

## 本工具不做什么

不训练、不评估测试集、不选择模型或超参数、不生成图表（matplotlib 等作图
依赖需后续单独批准）、不实现混淆矩阵或每类指标（P5.3/P5.4 范围）、不读取
`best_model.npz`、不修改任何源 run 文件。
