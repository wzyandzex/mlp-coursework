"""数值稳定的 Softmax 与 Softmax 交叉熵损失（纯 NumPy 实现）。

约定
----
* logits 形状 (m, K)：每行一个样本，K 个类别；
* 标签 y 形状 (m,)，取整数类别索引，满足 0 <= y[i] < K；
* 损失与梯度均为批次平均：
    J = -(1/m) * Σ_i log p_{i, y_i}
    ∂J/∂Z = (P - Y_onehot) / m
  dlogits 已含 1/m，调用方不得再次除以 batch size。
"""

import numpy as np


def _as_logit_matrix(logits) -> np.ndarray:
    """校验并转换 logits 为 float64 二维非空数组；总是返回新数组。"""
    raw = np.asarray(logits)
    if np.issubdtype(raw.dtype, np.complexfloating) or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"logits must be a real numeric array, got dtype {raw.dtype}.")
    if raw.ndim != 2:
        raise ValueError(f"logits must be 2-D (m, K), got {raw.ndim}-D with shape {raw.shape}.")
    if raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError(f"logits must be non-empty, got shape {raw.shape}.")
    arr = raw.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("logits contain non-finite values (NaN or Inf).")
    return arr


def _as_label_vector(y, batch_size: int, num_classes: int) -> np.ndarray:
    """校验标签并转换为 int64 一维数组；拒绝越界与非整数浮点（不静默截断）。"""
    raw = np.asarray(y)
    if raw.ndim != 1:
        raise ValueError(f"labels must be 1-D of length {batch_size}, got shape {raw.shape}.")
    if raw.shape[0] != batch_size:
        raise ValueError(f"labels length {raw.shape[0]} does not match batch size {batch_size}.")
    if np.issubdtype(raw.dtype, np.floating):
        # 浮点标签只在其值全部为精确整数时可解释为类别索引；否则拒绝截断
        if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
            raise ValueError(
                "float labels must be finite integral class indices; refusing to truncate."
            )
        labels = raw.astype(np.int64)
    elif np.issubdtype(raw.dtype, np.integer):
        labels = raw.astype(np.int64)
    else:
        raise TypeError(f"labels must be integer class indices, got dtype {raw.dtype}.")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(
            f"labels must satisfy 0 <= y < {num_classes}; got min={labels.min()}, max={labels.max()}."
        )
    return labels


def softmax(logits) -> np.ndarray:
    """逐行稳定 Softmax，返回 (m, K) 概率矩阵。

    每行先减去该行最大值再取指数：s_ik = z_ik - max_j z_ij，
    p_ik = exp(s_ik) / Σ_j exp(s_ij)，保证 exp 的指数不超过 0，避免上溢。
    """
    z = _as_logit_matrix(logits)
    shifted = z - z.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits, y) -> tuple[float, np.ndarray]:
    """Softmax + 交叉熵（批次平均），返回 (mean_loss, dlogits)。

    使用 log-sum-exp 稳定计算：
        log p_{i, y_i} = s_{i, y_i} - log Σ_j exp(s_ij)，s = logits - 每行最大值，
    避免先计算可能下溢为 0 的概率再取对数而产生无穷大。
    由于每行最大值平移后 exp 和 >= 1，log 恒安全。

    dlogits = (P - Y_onehot) / m 已是批次平均（等价于按标签索引的实现，
    不显式构造 one-hot 矩阵），调用方不应再除以 batch size。
    """
    z = _as_logit_matrix(logits)
    labels = _as_label_vector(y, z.shape[0], z.shape[1])
    m = z.shape[0]

    shifted = z - z.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    sum_exp = exp.sum(axis=1)
    log_sum_exp = np.log(sum_exp)
    picked = shifted[np.arange(m), labels]
    mean_loss = float(np.mean(log_sum_exp - picked))

    probs = exp / sum_exp[:, None]
    dlogits = probs.copy()
    dlogits[np.arange(m), labels] -= 1.0
    dlogits /= m
    return mean_loss, dlogits
