"""测试辅助（仅供 tests/ 使用）：中心差分数值梯度、相对误差与常用封装。"""

import numpy as np

from src.losses import softmax_cross_entropy


def numerical_gradients(loss_fn, param_arrays, eps: float = 1e-5) -> dict:
    """对 param_arrays（dict[str, ndarray]）中每个元素做中心有限差分。

    loss_fn: 无参可调用，返回标量损失；内部须读取 param_arrays 中的同一批数组
    （对 MLP 传入其 self.params 即可，forward 每次都读取这些数组）。
    为让模型感知扰动，本函数就地临时修改数组元素，每个元素检查后恢复原值。

        g_j = (J(θ_j + ε) - J(θ_j - ε)) / (2ε)
    """
    grads = {}
    for key, arr in param_arrays.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"param_arrays[{key!r}] must be a numpy.ndarray for in-place perturbation."
            )
        grad = np.zeros_like(arr, dtype=np.float64)
        iterator = np.nditer(arr, flags=["multi_index"])
        while not iterator.finished:
            index = iterator.multi_index
            original = arr[index]
            arr[index] = original + eps
            loss_plus = float(loss_fn())
            arr[index] = original - eps
            loss_minus = float(loss_fn())
            arr[index] = original
            grad[index] = (loss_plus - loss_minus) / (2.0 * eps)
            iterator.iternext()
        grads[key] = grad
    return grads


def relative_error(numeric, analytic) -> float:
    """相对误差 E = ||g_num − g_ana||₂ / max(1e-12, ||g_num||₂ + ||g_ana||₂)。"""
    numeric = np.asarray(numeric, dtype=np.float64)
    analytic = np.asarray(analytic, dtype=np.float64)
    numerator = np.linalg.norm(numeric - analytic)
    denominator = max(1e-12, np.linalg.norm(numeric) + np.linalg.norm(analytic))
    return float(numerator / denominator)


def model_loss(mlp, X, y) -> float:
    """forward + Softmax 交叉熵的标量损失（供数值梯度扰动调用）。"""
    logits = mlp.forward(X)
    loss, _ = softmax_cross_entropy(logits, y)
    return loss


def analytic_grads(mlp, X, y) -> dict:
    """forward(return_cache=True) + 交叉熵 + backward，返回解析梯度字典。"""
    logits, cache = mlp.forward(X, return_cache=True)
    _, dlogits = softmax_cross_entropy(logits, y)
    return mlp.backward(dlogits, cache)
