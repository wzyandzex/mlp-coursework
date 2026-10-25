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
| `invalid` | JSON 损坏、包含 NaN/Infinity、当前 schema 必需字段缺失/显式 null/类型错误/数值越界、`sample_counts` 不一致，或角色、test_evaluation 与测试指标互相矛盾（**绝不静默修正**） |
| `legacy` | **仅当 `experiment_role`（config）或 `test_evaluation`（summary）的键确实不存在**时判定（如 P4/P5.2-B 时代的 run）；已有数值原样保留，但不参与分组与测试统计，其测试集状态**不自动推导**。键存在但值为 JSON null 属于 `invalid`，不是 legacy |

默认模式下单个损坏 run 不会影响其它有效 run 的汇总；`--strict` 会在完成全部
诊断后返回非零退出码。

## 当前 schema 校验规则（P5.2-C，严格模式）

汇总器按当前实际产出的 schema 对每个非 legacy run 做完整校验；任何缺键、显式
null、类型错误或越界数值都会写入**定位到字段**的诊断并判 `invalid`，绝不静默
补默认值。`legacy` 的判定前提是：缺失键之外、所有**已存在**的相关字段均合法——
若已存在字段有明确非法的枚举值/类型（例如 `test_evaluation="banana"`），优先判
`invalid` 而非 legacy（该优先级由测试覆盖）。

### config.json 必需字段与约束

| 字段 | 约束 |
| --- | --- |
| `task` | 非空字符串 |
| `network` | 对象：`input_dim`/`output_dim` 为正整数；`hidden_dims` 为非空正整数列表；`hidden_activation` ∈ {relu, sigmoid, tanh}；`initializer` ∈ {auto, he_normal, xavier_normal} |
| `training` | 对象：`epochs`/`batch_size`/`eval_batch_size`/`step_size`/`patience` 为正整数；`learning_rate`/`eps` 为 >0 的有限数；`momentum`/`beta1`/`beta2` ∈ [0, 1)；`gamma` ∈ (0, 1]；`min_delta`/`l2_lambda` ≥ 0；`optimizer` ∈ {sgd, momentum_sgd, adam}；`lr_scheduler` ∈ {none, step}；`early_stopping` 为 JSON boolean |
| `seeds` | 对象：`model_seed`/`split_seed` 为非负整数 |
| `data` | 对象：`source_dir`/`split_strategy` 为非空字符串；`train_size`/`val_size` 为正整数；`sample_counts` 为含非负整数 `train`/`validation`/`test` 的对象（**该键整体缺失同样判 invalid**，键存在但为 null/非对象亦然） |
| `environment` | 对象：`python`/`numpy`/`platform` 为非空字符串 |
| `experiment_role` | ∈ {baseline, exploration, final} |
| `evaluate_test` | 必须是真正的 JSON boolean（0/1/字符串不算），且与角色匹配 |

整数型字段拒绝 bool；字符串字段拒绝 null/数字/对象；浮点字段必须有限；不得
静默转换类型。

### summary.json 必需字段与约束

| 字段 | 约束 |
| --- | --- |
| `status` | 必须为 `"success"`；`failed` 或其它值判 invalid |
| `run_dir` | 非空字符串 |
| `experiment_role` / `test_evaluation` / `stop_reason` | 枚举同上；`stop_reason` ∈ {max_epochs, early_stopping} |
| `best_epoch` | 正整数，且 ≤ `epochs_completed` |
| `best_validation_loss` | ≥ 0 的有限数 |
| `best_validation_accuracy` | ∈ [0, 1] 的有限数 |
| `test_loss` / `test_accuracy` | executed 时为有限数值（loss ≥ 0、accuracy ∈ [0,1]）；not_run 时必须为 JSON null |
| `epochs_completed` | 正整数 |
| `total_seconds` | ≥ 0 的有限数 |
| `sample_counts` | 含非负整数 `train`/`validation`/`test` 的对象，且与 `config.data.sample_counts` 完全一致 |
| `test_confusion_matrix` / `test_per_class_metrics` / `test_macro_avg` | **仅 `experiment_role="final"` 必需**（见下节）；baseline/exploration/legacy 缺这些字段是正常状态，不判 invalid |

跨文件一致性：config 与 summary 的 `experiment_role` 必须一致；角色、
`evaluate_test` 与 `test_evaluation` 必须满足隔离矩阵；`sample_counts` 必须一致。

## final run 分类指标（P5.3-E6-A）

仅 `experiment_role="final"` 的 run 在 summary.json 中携带以下三个字段；
它们由**恢复最佳 checkpoint 后的一次测试前向**同时导出，与 `test_loss` /
`test_accuracy` 严格同源。

### 字段定义

* `test_confusion_matrix`：C×C 整数二维数组（C = `config.network.output_dim`，
  且 ≥ 2）。**行 = 真实类别，列 = 预测类别**；`matrix[i][j]` 表示真实类别为 i、
  预测类别为 j 的样本数。元素为非负整数，矩阵总数等于 `sample_counts.test`。
* `test_per_class_metrics`：恰有 C 个对象（按类别 0..C-1 排序），每类含
  `precision`、`recall`、`f1`（[0,1] 内有限数值）与 `support`（非负整数，等于
  混淆矩阵对应**行和**）。**零分母规则**：某类预测正例数为 0 时 precision 记
  0.0；真实 support 为 0 时 recall 记 0.0；precision+recall 为 0 时 f1 记 0.0。
  逐类与 macro 的数值字段必须是可安全表示为有限浮点的数——JSON 中合法但超出
  浮点表示范围的极大整数（如 10**400）判 `invalid` 并附定位诊断，绝不使汇总器
  以未捕获 OverflowError 崩溃。
* `test_macro_avg`：`precision`/`recall`/`f1` 各为 C 类逐类指标的**不加权算术
  平均**（不按 support 加权、不剔除零 support 类别），汇总器按容差 1e-9 与逐类
  直接重算值核对。

单 run 一致性约束（违反任一条即 invalid）：`sum(matrix) == sample_counts.test`；
`sum(diagonal(matrix)) / sum(matrix)` 与 `test_accuracy` 在 1e-9 容差内一致；
逐类 `support` 与矩阵行和逐类相等；`test_macro_avg` 与逐类重算值一致；
**逐类 `precision`/`recall`/`f1` 必须与混淆矩阵按冻结公式重算的值逐项一致**
（`tp = matrix[c][c]`，`support` = 行和，`predicted` = 列和；`predicted=0` 时
precision 记 0.0，`support=0` 时 recall 记 0.0，`precision+recall=0` 时 f1 记
0.0；容差 abs 1e-9、rel_tol=0.0。权威来源是混淆矩阵本身——同步篡改逐类值
与 macro 无法通过校验）。

### 汇总器行为

* **校验**：final 记录缺任一字段、矩阵非方形/维度与 `output_dim` 不符/含浮点
  或负数元素/support 与行和不符/macro 与重算不符/总数与 `sample_counts.test`
  不符/对角线占比与 `test_accuracy` 不符 → 该 run 判 `invalid` 并附定位诊断。
* **保留**：`runs.json` 原样保留每个 final run 的三个字段；`runs.csv` 以
  `test_confusion_matrix_json` / `test_per_class_metrics_json` /
  `test_macro_avg_json` 三列写入紧凑规范化 JSON（非 Python repr）；非 final
  run 的这三列为空单元格。
* **组级统计**：final 组（`test_evaluation="executed"`）的 `stats` 额外包含
  `test_macro_avg` 与 `test_per_class_metrics`（键为类别索引字符串），每项给出
  `mean`/`std`（ddof=0）/`n`；并附 `test_classification_note` 说明口径。
  **语义注记**：组级统计是固定测试集上的**逐模型统计**——每个成员的 support
  是该模型对同一批固定测试样本的类别计数，跨 seed 相加不构成独立样本数；
  **混淆矩阵保留在 run 级，不做跨 seed 矩阵求和**（组级不输出任何"求和矩阵"）。
* **兼容性**：baseline/exploration/legacy 记录不要求这些字段；`runs.csv` 新列
  对旧记录为空单元格；缺字段的旧产物不会被误判为损坏。

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
`test_loss`、`test_accuracy`、`test_confusion_matrix`、`test_per_class_metrics`、
`test_macro_avg`（后三者仅 final run 非空）、`diagnostics`（诊断字符串数组）、
`config`（完整原始配置）。

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
依赖需后续单独批准）、不读取 `best_model.npz`、不修改任何源 run 文件。
（final run 的分类指标自 P5.3-E6-A 起由训练器在 summary 中产出，汇总器负责
校验、保留与组级统计，见上文"final run 分类指标"一节。）
