"""激活函数及其导数（纯 NumPy 实现）。

支持 relu、sigmoid、tanh，并提供按名称的前向/导数解析接口。

数学定义
--------
ReLU:    f(z) = max(0, z),         f'(z) = 1（z > 0），0（z <= 0，含 z = 0 处的次梯度约定）
Sigmoid: f(z) = 1 / (1 + exp(-z)), f'(z) = f(z) * (1 - f(z))
Tanh:    f(z) = tanh(z),           f'(z) = 1 - f(z)^2
"""

import numpy as np

SUPPORTED_ACTIVATIONS: tuple[str, ...] = ("relu", "sigmoid", "tanh")


def _as_float_array(z, name: str = "z") -> np.ndarray:
    """转换为 float64 ndarray；拒绝复数/非数值类型与非有限值。总是返回新数组。"""
    raw = np.asarray(z)
    if np.issubdtype(raw.dtype, np.complexfloating) or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must be a real numeric array, got dtype {raw.dtype}.")
    arr = raw.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN or Inf).")
    return arr


def relu(z) -> np.ndarray:
    """ReLU：f(z) = max(0, z)。"""
    arr = _as_float_array(z)
    return np.maximum(arr, 0.0)


def sigmoid(z) -> np.ndarray:
    """数值稳定的 Sigmoid：f(z) = 1 / (1 + exp(-z))。

    令 e = exp(-|z|) ∈ [0, 1]，则
        f(z) = 1 / (1 + e)（z >= 0）；  f(z) = e / (1 + e)（z < 0）。
    两个分支都只计算 exp(-|z|)，因此极大/极小输入都不会上溢出警告或产生 NaN。
    """
    arr = _as_float_array(z)
    e = np.exp(-np.abs(arr))
    return np.where(arr >= 0.0, 1.0 / (1.0 + e), e / (1.0 + e))


def tanh(z) -> np.ndarray:
    """Tanh：f(z) = tanh(z)，输出范围 (-1, 1)。"""
    arr = _as_float_array(z)
    return np.tanh(arr)


def relu_grad(z) -> np.ndarray:
    """ReLU 导数：z > 0 处为 1；z <= 0（含 z = 0，按任务约定）处为 0。"""
    arr = _as_float_array(z)
    return (arr > 0.0).astype(np.float64)


def sigmoid_grad(z, activated=None) -> np.ndarray:
    """Sigmoid 导数：f'(z) = a * (1 - a)，其中 a = f(z)。

    activated 允许传入已算好的前向输出 a；为 None 时内部重新计算。
    """
    a = sigmoid(z) if activated is None else _as_float_array(activated, "activated")
    return a * (1.0 - a)


def tanh_grad(z, activated=None) -> np.ndarray:
    """Tanh 导数：f'(z) = 1 - a^2，其中 a = tanh(z)。"""
    a = tanh(z) if activated is None else _as_float_array(activated, "activated")
    return 1.0 - a * a


def check_activation_name(name) -> str:
    """校验激活函数名称，返回规范化（小写）名称；非法名称抛出 TypeError/ValueError。"""
    if not isinstance(name, str):
        raise TypeError(f"activation name must be a string, got {type(name).__name__}.")
    key = name.strip().lower()
    if key not in _FORWARD:
        raise ValueError(
            f"unsupported activation {name!r}; supported: {', '.join(SUPPORTED_ACTIVATIONS)}."
        )
    return key


def activate(name: str, z) -> np.ndarray:
    """按名称计算前向激活；未知名称抛出 ValueError。"""
    return _FORWARD[check_activation_name(name)](z)


def activation_gradient(name: str, z, activated=None) -> np.ndarray:
    """按名称计算激活导数。

    activated 允许传入已算好的前向输出 a = f(z)（Sigmoid/Tanh 的导数用 a 表示
    更高效）；ReLU 的导数由 z 的符号决定，忽略该参数。
    """
    key = check_activation_name(name)
    if key == "relu":
        return relu_grad(z)
    return _GRAD[key](z, activated)


_FORWARD = {"relu": relu, "sigmoid": sigmoid, "tanh": tanh}
_GRAD = {"relu": relu_grad, "sigmoid": sigmoid_grad, "tanh": tanh_grad}
