# 手写 MLP 课程作业 —— MNIST 训练闭环、对照实验链与最终报告（P4–P6）

## 当前阶段与模块职责

在阶段 3 的手写 NumPy 核心（P3-CORE-01 已关闭）与阶段 4 的 MNIST 训练闭环之上，P5.3 按
预注册流程完成了对照实验链 E0–E6，P6 生成最终课程报告与图表（交付包见
`docs/coursework-final-report/`）。

| 模块 | 职责 |
| --- | --- |
| `src/activations.py` | **（阶段 3，本阶段未改动）** ReLU/Sigmoid/Tanh 及导数、按名称解析；Sigmoid 数值稳定实现。 |
| `src/losses.py` | **（阶段 3，本阶段未改动）** 稳定 Softmax 与 `softmax_cross_entropy`（log-sum-exp，批次平均梯度 `(P−Y)/m`）。 |
| `src/mlp.py` | **（阶段 3，本阶段未改动）** MLP：He/Xavier 初始化、forward（含 cache）、backward、predict、参数备份/恢复。 |
| `src/optimizers.py` | SGD/MomentumSGD/Adam：两阶段更新（候选值有限性验证 + 原子就地提交）。 |
| `src/mnist_data.py` | **（本阶段新增）** IDX gzip 解析（magic/维度/payload/标签范围校验）、显式下载（临时文件→结构校验→原子替换，双镜像）、预处理（float64/255 → (N,784)）、按比例分层随机切分。 |
| `src/training.py` | **（本阶段新增）** mini-batch 训练循环、按样本数加权的批量评估、val_loss 选最佳 checkpoint（并列取更早 epoch）、npz checkpoint 保存/安全加载、运行目录与产物编排；P5.3 扩展学习率调度（step）、早停、L2 正则化、测试集隔离与 run 角色。 |
| `src/experiments.py` | **（P5.3 新增）** 跨 run 严格聚合器（strict 模式、schema 校验、组级统计汇总）。 |
| `src/train_mnist.py` | **（本阶段新增）** 命令行入口：参数校验、显式下载、组装配置并调用训练编排。 |
| `tests/` | 阶段 3 全部既有测试保留；新增 `test_mnist_data.py` 与 `test_training.py` 中的数据/切分/训练/产物/CLI 测试（全部离线）。 |

## 最终报告与交付包

P6 阶段课程报告（内容权威源为 Markdown，DOCX/PDF 为通用学术排版导出版，20 页，含目录与页码）：

- 报告：[P6.2_final_report.md](docs/coursework-final-report/P6.2_final_report.md) ·
  [P6.2_final_report.docx](docs/coursework-final-report/P6.2_final_report.docx) ·
  [P6.2_final_report.pdf](docs/coursework-final-report/P6.2_final_report.pdf)
- 同目录含全部图表（`figures/`，PNG 300dpi + SVG）、派生数据（`derived_data/`）、
  图表溯源清单（`figure_manifest.md`）、绘图脚本（`plot_figures.py`）、环境记录
  （`report_environment.txt`）与哈希清单（`SHA256SUMS.txt`）。
- 说明：本地未找到教师模板/rubric，DOCX/PDF 为通用学术排版版本；报告不含作者/课程信息。

## 环境与安装

* Python 3.11.4（Windows）、numpy 2.4.6、pytest 9.1.1

```bash
pip install -r requirements.txt
```

## MNIST 数据

* 四个标准 IDX gzip 文件存放于 `data/mnist/raw/`（目录可用 `--data-dir` 更改）。
* **显式下载**（默认不联网，只有加 `--download` 才访问网络）：

```powershell
# Windows PowerShell（在项目根目录）
python -m src.train_mnist --download
```

```bash
# 通用 shell
python -m src.train_mnist --download
```

* 下载源（按顺序尝试，TLS 验证不关闭）：
  `https://ossci-datasets.s3.amazonaws.com/mnist/` 与
  `https://storage.googleapis.com/cvdf-datasets/mnist/`。
* **离线手动准备**：从上述任一来源下载四个文件
  （`train-images-idx3-ubyte.gz`、`train-labels-idx1-ubyte.gz`、
  `t10k-images-idx3-ubyte.gz`、`t10k-labels-idx1-ubyte.gz`）放入 `data/mnist/raw/` 即可，
  不加 `--download` 也能正常运行。
* `data/` 与 `outputs/` 已被 `.gitignore` 排除，原始数据与训练产物不进仓库。

## 预处理与数据切分

* 预处理：uint8 → float64 / 255.0，展平 (N, 784)，取值 [0, 1]；标签 int64 类别索引 0..9；
  不做 one-hot、不做标准化/PCA/增强。
* 切分方案（已批准的 20k/5k/10k）：train/validation 只从官方 60,000 训练样本中
  **按类别分层随机**抽取（每类按比例配额、最大余数法保证总数精确、类内用显式 seed 打乱），
  官方 10,000 测试样本整体保留、不参与切分与训练。
  切分策略说明：该方式形成**类别均衡的子集**，不等同于完整 60,000 样本训练。
* seed 职责：`--split-seed`（默认 2026）决定切分；`--seed`（默认 42）决定模型初始化与
  每个 epoch 的打乱顺序（同一 Generator 顺序派生，逐 epoch 不同但整体可复现）。
  相同数据 + 相同 seed 必产生相同划分与相同训练历史。

## 基线网络与默认训练参数

* 网络：784 → [128] → 10，隐藏层 ReLU，输出层线性 logits + Softmax 交叉熵；
  初始化 auto（ReLU 隐藏层 He 正态，输出层 Xavier 正态）。
* 训练：SGD，batch size 128，最多 30 epoch，学习率 0.1。
* **学习率依据（真实 MNIST 小规模冒烟实验，3 epoch，20k/5k 切分，seed 42）**：

| 学习率 | epoch1 val_loss | epoch2 val_loss | epoch3 val_loss | 观察 |
| --- | --- | --- | --- | --- |
| 0.05 | 0.5242 | 0.4086 | 0.3630 | 收敛慢 |
| **0.1** | 0.4088 | 0.3486 | 0.3195 | 稳定下降（采用） |
| 0.3 | 0.3090 | 0.3071 | 0.3108 | 早期即震荡回落 |

* 该配置是基础基线，不声称最优；训练指标为“在线指标”（参数更新前的 logits、按样本数加权），
  验证指标为 epoch 结束后对完整验证集的批量评估（loss 按样本数加权）。

## 命令行

```bash
python -m src.train_mnist --help          # 查看全部参数与默认值
python -m src.train_mnist --download      # 下载缺失文件并训练
python -m src.train_mnist                 # 用已就绪的数据按默认配置训练
python -m src.train_mnist --epochs 5 --learning-rate 0.2 --run-name quick_check
```

参数非法（epochs/batch-size ≤ 0、learning-rate 非正或非有限、seed 为负等）会在训练开始前报错；
数据文件缺失时错误信息会列出期望文件名、目录及获取方式。

## 运行产物

每次运行写入 `outputs/mnist_baseline/<run>/`（或 `--output-dir` 指定的根目录）下的独立目录，
已存在的运行目录不会被覆盖（时间戳命名或 `--run-name` 显式指定）：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 完整有效配置（网络、训练参数、seed、切分方案、样本数、Python/numpy/平台版本）；不写绝对用户路径 |
| `history.csv` | 每 epoch 的 `epoch, train_loss, train_accuracy, val_loss, val_accuracy, elapsed_seconds` |
| `best_model.npz` | 最佳验证 checkpoint 的参数数组（无 pickle；加载 `allow_pickle=False` 并校验键/形状/有限性） |
| `summary.json` | `status, best_epoch, best_validation_loss, best_validation_accuracy, test_loss, test_accuracy, sample_counts, epochs_completed, total_seconds` |

## 测试集使用纪律

测试集（官方 10,000 样本）不参与训练、切分与超参数调整；只在训练全部 epoch 并按验证集
`val_loss` 选出最佳 checkpoint **之后**评估一次。运行记录中 `best_validation_*` 与 `test_*`
分开记录。早停与学习率调度自 P5.3 起实现（CLI 可选启用，默认关闭）。

## 实验执行状态

**已实际执行（真实 MNIST，非合成数据）**：

* 数据：`--download` 成功获取四个标准文件（字节数与官方一致：9,912,422 / 28,881 / 1,648,877 / 4,542 字节），
  解析校验通过（60000/10000 张 28×28 图）。
* 学习率冒烟：0.05 / 0.1 / 0.3 × 3 epoch（结果见上表）。
* 正式基线：`python -m src.train_mnist`（30 epoch，lr 0.1，batch 128，seed 42/2026），
  运行目录 `outputs/mnist_baseline/baseline_30ep_consistency_fix`（测试 accuracy 0.9627）。
* P5.3 预注册对照实验链 E0–E6 已全部完成并通过独立审计：E1 优化器/学习率（探索 9 run +
  三种子确认 9 run）、E2 架构（15 run）、E3 激活函数与自动初始化组合（9 run）、
  E4 L2 正则化（9 run）、E5 训练策略（12 run）、E6 冻结配置最终评估（3 run），共 66 个
  P5.3 运行；E6 最终模型在官方 10,000 样本测试集上 test accuracy 97.35%（跨 3 seed
  描述性统计，ddof=0）。实验结论、统计口径与全部局限详见最终报告。

**后续工作**：仅作展望（在 [256] 架构上复核组件选择、解耦激活与初始化、完整 60k 训练集等），
见最终报告第 13.4 节；本仓库不再自行追加实验。

## 运行测试

在项目根目录执行：

```bash
python -m pytest -q
# 或
python -m pytest tests -q
```

## 实现声明

核心 MLP、Softmax/交叉熵及其梯度、SGD、数据读取与训练闭环均为本项目使用 NumPy 与
Python 标准库手写实现，未使用 PyTorch/TensorFlow/Keras/JAX/scikit-learn 等任何机器学习
框架，也未引入 requests/tqdm 等额外依赖。本 README 只记录实际运行验证过的结果。
