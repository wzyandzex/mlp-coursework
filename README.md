# 手写 MLP 课程作业 —— 阶段 3 · 第一批实现（P3-CORE-01）

## 当前阶段与范围

本仓库当前只包含**基础 MLP 核心**（纯 NumPy 手写）及其自动化正确性测试：

* 激活函数（ReLU / Sigmoid / Tanh）及导数；
* 数值稳定的 Softmax 与 Softmax + 交叉熵（含批次平均梯度）；
* MLP 参数初始化（He / Xavier 正态）、前向传播（输出线性 logits）、反向传播；
* 基础 SGD 优化器；
* 单元测试 + 数值梯度检查 + 合成数据训练冒烟测试。

**尚未实现**（属于后续阶段，勿在本阶段期待）：MNIST 下载/解析、正式训练与验证管线、
早停、Momentum/Adam、多组扩展实验、ROC/AUC、最终实验报告、GUI。

## 环境

* Python 3.11.4（Windows，解释器 `D:\Python3\python.exe`）
* numpy 2.4.6
* pytest 9.1.1

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行测试

在项目根目录（本 README 所在目录）执行：

```bash
python -m pytest -q
```

或等价的：

```bash
python -m pytest tests -q
```

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `src/activations.py` | ReLU / Sigmoid / Tanh 及导数；按名称的前向与导数解析接口。Sigmoid 用 `exp(-|z|)` 双分支实现，极端输入无溢出；ReLU 在 z=0 处导数约定为 0。 |
| `src/losses.py` | 逐行减最大值的稳定 Softmax；`softmax_cross_entropy(logits, y)` 用 log-sum-exp 返回批次平均损失与 `dlogits = (P − Y_onehot)/m`（已含 1/m，调用方不再除 batch size）；完整的 logits/标签校验。 |
| `src/mlp.py` | `MLP` 类：W_l ∈ (d_in, d_out)、b_l ∈ (1, d_out)；隐藏层逐层激活、输出层直接给线性 logits；初始化 auto 规则——ReLU 隐藏层用 He 正态 `sqrt(2/fan_in)`，Sigmoid/Tanh 隐藏层与输出层用 Xavier 正态 `sqrt(2/(fan_in+fan_out))`，偏置为零；`forward(X, return_cache=...)`、`backward(dlogits, cache)`（只算梯度、不更新参数、不再除 m）、`predict_proba` / `predict`、深复制备份 `get_params_copy()` / `set_params()`；随机性只来自显式 `numpy.random.Generator` 或 seed，每层实际策略记录在 `layer_initializers` 中可检查。 |
| `src/optimizers.py` | 基础 SGD：`step(params, grads)` 执行 `θ ← θ − η·g`，就地更新；更新前校验学习率、键集合、形状与有限性，校验失败不产生部分更新。 |

## 实现声明

* 核心 MLP、Softmax/交叉熵及其梯度、SGD 均为本项目**使用 NumPy 手写实现**，
  未使用 PyTorch / TensorFlow / Keras / JAX 等框架或任何机器学习库的现成实现。
* 数组计算统一 float64；批次约定“每行一个样本”；标签为整数类别索引。
* 数值梯度检查（中心差分 ε=1e-5，相对误差阈值 < 1e-5）覆盖 Sigmoid/Tanh 的单隐层
  与双隐层网络全部权重与偏置；ReLU 检查前先断言所有预激活远离 0。
* 训练冒烟测试使用固定 seed 的 NumPy 合成三簇数据，断言损失有限、显著下降且可复现。
* 本 README 不包含任何未经运行验证的性能结果。
