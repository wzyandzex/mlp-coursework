# 提交说明（README for Submission）

本目录是《手写 NumPy MLP 的 MNIST 分类实验》课程报告的**最终交付包**（发布候选），由 P6.3 阶段从项目 `outputs/p5.3/P6.2/` 的已验收成果整理而成。

## 提交内容

| 文件/目录 | 说明 |
| --- | --- |
| `P6.2_final_report.md` | 报告权威源文件（Markdown，内容以此为准） |
| `P6.2_final_report.docx` / `P6.2_final_report.pdf` | 由上述 Markdown 与同目录图表生成的提交版（A4，20 页，含目录与页码） |
| `figures/` | 报告全部 8 张图（PNG 300 dpi + SVG 矢量，共 16 个文件） |
| `derived_data/` | 图表派生数据（9 个 CSV/JSON，完整精度，可溯源到聚合产物） |
| `figure_manifest.md` | 每张图的数据源、字段、run/group id、统计口径与复现命令 |
| `plot_figures.py` | 图表生成脚本（只读解析实验产物，可重复运行） |
| `report_environment.txt` | 实验运行环境与图表/文档生成环境的完整记录 |
| `submission_notes.md` | 交付范围、统计口径与证据缺口说明 |
| `SHA256SUMS.txt` | 本目录全部文件（不含自身）的 SHA-256 清单 |

## 阅读建议

- 快速阅读：`P6.2_final_report.pdf`（或 `.docx`）；内容与 `.md` 完全一致。
- 复核数字：任何数值都可按 `figure_manifest.md` 的登记回到 `derived_data/` 与项目实验产物对账。
- 关键结论（摘要）：在 20,000 训练样本 / 30 epoch 预算下，按预注册主指标依次选定 MomentumSGD lr=0.05、[256]、ReLU+He normal、L2 λ=0.0001、固定学习率无早停；最终模型在官方 10,000 样本测试集上取得 test loss 0.087566 ± 0.001882、accuracy 97.35% ± 0.01 pp、macro F1 0.973271 ± 0.000123（跨 3 个模型种子的描述性统计，ddof=0，非置信区间）。

## 格式与证据状态

- DOCX/PDF 为**通用中文学术排版**版本：本地证据中未找到教师提供的模板或评分标准（rubric），不能保证符合未提供的课程特殊格式要求；如需按指定模板排版，请以 `P6.2_final_report.md` 为内容源重新导出。
- 报告不包含作者/学校/教师信息（证据中不存在，未虚构）；如需署名请自行补填。
- 实验全链路（E0–E6，66 个运行）的原始产物保留在项目本地并经哈希保护；本包只含报告、最终图表、派生数据与追溯材料，不含原始训练产物。
