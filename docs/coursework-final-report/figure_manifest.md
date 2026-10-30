# P6.2 图表清单与溯源登记（figure manifest）

- 生成日期：2026-10-25
- 生成环境：独立虚拟环境（仓库之外），Matplotlib 3.10.9，Agg 后端；详见 `report_environment.txt`
- 复现命令（仓库根目录执行）：`../mlp-coursework-report-venv/Scripts/python.exe outputs/p5.3/P6.2/plot_figures.py`
- 脚本只读解析聚合 JSON 与逐 run `summary.json`/`history.csv`，用标准库 `statistics.fmean`/`statistics.pstdev` 重算组级统计并与 `groups.json` 对账（偏差 > 1e-9 即中止），随后写入 `derived_data/`（完整精度）并渲染 `figures/`（每图 PNG 300 dpi + SVG）。**不复制粘贴任何数值作为图表数据源。**

## 图表登记

| 图号 | 研究问题 | 源文件（字段） | run/group ids | 聚合计算 | 误差条 | 输出 |
| --- | --- | --- | --- | --- | --- | --- |
| fig01_experiment_flow | E0–E6 流程与各阶段变量/运行数如何组织 | 阶段目录清单 + 各 run `config.json`（变量/seed/样本量）；E0 数值：`outputs/mnist_baseline/baseline_30ep_consistency_fix/summary.json` | 66 个 P5.3 run（结构化枚举，A0 Pilot 计入 E1 exploration 一次）；E0 为 P4 产物单独标注 | 结构汇总：9+9+15+9+9+12+3=66 | 无 | `.png/.svg` |
| fig02_e1_optimizer_confirmation | E1 三优化器确认组的主指标比较 | `_aggregate_E1_confirmation/runs.json` + 各 run `summary.json:best_validation_loss` | 组 `gc5f232da94d6`（mom-SGD lr0.05）、`g2b1fddd38f86`（sgd lr0.2）、`gf5aa7702adbd`（adam lr0.003），各 n=3（seeds 101/202/303） | `fmean` 组均值；与 groups.json 对账零偏差 | 跨 seed pstdev (ddof=0) | `.png/.svg` |
| fig03_e2_architecture | 五种架构的主指标-参数量权衡 | `_aggregate_E2_architecture/runs.json` + summary；参数量按 `784·h₁+h₁+Σ(hᵢ·hᵢ₊₁+hᵢ₊₁)+hₖ·10+10` 由 hidden_dims 推得 | 组 `ga2da0b47750c`[256]、`geccea6bbf90f`[256,128]、`gc5f232da94d6`[128]、`g79b1b53d972d`[128,64]、`g5a58c9a29298`[64]，各 n=3 | 组均值 ± pstdev；架构为离散类别轴 | 跨 seed pstdev (ddof=0) | `.png/.svg` |
| fig04_e3_activation_initialization | 三种"激活+auto 初始化"组合的比较 | `_aggregate_E3_activation/runs.json` + summary | 组 `gc5f232da94d6`（relu+He）、`g4ca3d63c481c`（tanh+Xavier）、`g9b808fbb929b`（sigmoid+Xavier），各 n=3 | 组均值 ± pstdev；标签含初始化器以明示耦合比较 | 跨 seed pstdev (ddof=0) | `.png/.svg` |
| fig05_e4_l2 | L2 λ 的主指标比较 | `_aggregate_E4_l2/runs.json` + summary | 组 `g950f9b6c6dd4`（λ=1e-4）、`g818eda627e6d`（λ=1e-3）、`gc5f232da94d6`（λ=0），各 n=3 | 组均值 ± pstdev；λ 为离散类别、线性轴，λ=0 不做 log 变换 | 跨 seed pstdev (ddof=0) | `.png/.svg` |
| fig06_e5_strategy_tradeoff | 训练策略的性能-耗时权衡 | `_aggregate_E5_training_strategy/runs.json` + summary（`best_validation_loss`、`total_seconds`、`epochs_completed`） | 组 `g950f9b6c6dd4`（fixed_no_es）、`gc94c954ff912`（fixed_es）、`gcc5015f80874`（step_no_es）、`g9c5eb3f5e685`（step_es），各 n=3 | 左面板组均值 ± pstdev；右面板 `fmean(total_seconds)`；两面板分离，无共享双轴 | 左面板跨 seed pstdev (ddof=0)；右面板无误差条（纯效率均值，精确值见 CSV） | `.png/.svg` |
| fig07_e6_confusion_matrices | 三个最终模型在测试集上的混淆结构 | `E6_final/e6_final_seed{404,505,606}/summary.json:test_confusion_matrix` | run `e6_final_seed404/505/606`（model seeds 404/505/606） | 每 seed 各自矩阵按**自身行和**行归一化后分面展示；原始计数矩阵**不相加** | 无（矩阵单元为行归一化率） | `.png/.svg` |
| fig08_e6_per_class_f1 | 逐类 F1 的跨 seed 均值与波动 | `E6_final/<run>/summary.json:test_per_class_metrics[].f1` | 同上 3 个 run，10 类 × 3 seed | 每类 `fmean`/`pstdev`（ddof=0）；support 逐模型列出（不相加） | 跨 seed pstdev (ddof=0) | `.png/.svg` |

## 命名映射说明

P6.1 大纲第 5 节的"图 7（E6 逐类 F1 条形图 + 混淆矩阵）"在本阶段按规格拆分为 fig07（混淆矩阵）与 fig08（逐类 F1），其余图号顺序与大纲一致；此为规格允许的一次性命名调整，无其他改名。

## 复算与对账记录（本次运行实际结果）

- 结构检查：7 阶段 run 目录数 9/9/15/9/9/12/3 与聚合器 `report.json` 的 total_runs/n_groups 一致；strict=true、strict_failed=false、problems=[]；counts 中 incomplete/invalid/legacy 全为 0；runs.json 与磁盘 run 名一一对应；A0 Pilot 恰出现一次。
- 全部组级 `fmean`/`pstdev` 与 `groups.json` 对账：**最大偏差 0.00e+00**。
- E6：3 个 run 的 seeds 恰为 404/505/606；混淆矩阵均为 10×10 且总数=10,000；逐类指标从矩阵重算与 `test_per_class_metrics` 零偏差；macro=逐类算术平均（偏差 ≤ 2.22e-16）；跨 seed 聚合与 `groups.json` 的 test_loss/test_accuracy 组统计一致。
- E0 基线读取正常（val_loss 0.125406 / test_acc 0.9627）。

## 哈希保护

- 制图前对 237 个源文件建立 SHA-256 清单（7 个聚合目录全部 runs/groups/report JSON、66 run 的 config.json/summary.json/history.csv、E6 三个 best_model.npz、E0 基线 config/summary、全部 11 份 P5.3 报告、P6.1 两份交付文件）；清单存于仓库外工作区 `_p62_source_hashes_before.json`，组合清单 SHA-256 为 `b9ed9bcf59713531744d8f347e4c61c906a84d0420eebf203b32d4d87d614ddc`。
- 制图完成后复算比对：**237 个文件 0 变化、0 缺失**。

## 自检结果

8 张图 × 2 格式 = 16 个文件全部存在且非空；PNG 均通过 PIL `Image.open().verify()`；SVG 均通过 `xml.etree.ElementTree` 解析（合法 XML）；并经人工查看核验版式（无文字重叠、无截断轴、无 3D、无双轴误导）。
