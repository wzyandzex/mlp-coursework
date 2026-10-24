"""优化器模块：本阶段仅实现基础 SGD。

更新公式：θ ← θ − η·g（η 为学习率，g 为批次平均梯度）。
参数更新就地执行；本模块不负责计算梯度。Momentum / Adam 留待后续授权阶段。
"""

import numpy as np


def _is_real_number(value) -> bool:
    """是否为实数标量（int/float/np.integer/np.floating；bool 不算数）。"""
    return not isinstance(value, bool) and isinstance(value, (int, float, np.integer, np.floating))


class SGD:
    """基础随机梯度下降（固定学习率，对小批/全批均适用）。

    用法：optimizer.step(params, grads)，params 与 grads 为同键的字典，
    值为形状一致的 numpy 数组；更新就地修改 params 中的数组本身。
    """

    def __init__(self, learning_rate: float) -> None:
        if not _is_real_number(learning_rate):
            raise TypeError(f"learning_rate must be a real number, got {type(learning_rate).__name__}.")
        lr = float(learning_rate)
        if not np.isfinite(lr):
            raise ValueError(f"learning_rate must be finite, got {learning_rate!r}.")
        if lr <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate!r}.")
        self.learning_rate: float = lr

    def step(self, params: dict, grads: dict) -> None:
        """执行一步更新 params[key] ← params[key] − η·grads[key]（就地修改）。

        先完整校验（键集合、形状、有限性），全部通过后才统一更新，
        因此任何校验失败都不会产生“部分更新”。
        """
        if not isinstance(params, dict):
            raise TypeError(f"params must be a dict, got {type(params).__name__}.")
        if not isinstance(grads, dict):
            raise TypeError(f"grads must be a dict, got {type(grads).__name__}.")

        param_keys, grad_keys = set(params), set(grads)
        if param_keys != grad_keys:
            missing = sorted(grad_keys - param_keys)
            unexpected = sorted(param_keys - grad_keys)
            raise ValueError(
                f"params and grads key sets must match; "
                f"keys missing from params: {missing}; unexpected keys in params: {unexpected}."
            )

        checked: list[tuple[np.ndarray, np.ndarray]] = []
        for key in sorted(param_keys):
            p, g = params[key], grads[key]
            if not isinstance(p, np.ndarray):
                raise TypeError(
                    f"param {key!r} must be a numpy.ndarray for in-place update, "
                    f"got {type(p).__name__}."
                )
            if not np.issubdtype(p.dtype, np.floating):
                raise TypeError(f"param {key!r} must have a floating dtype, got {p.dtype}.")
            if not isinstance(g, np.ndarray):
                raise TypeError(f"grad {key!r} must be a numpy.ndarray, got {type(g).__name__}.")
            if np.issubdtype(g.dtype, np.complexfloating) or not np.issubdtype(g.dtype, np.number):
                raise TypeError(f"grad {key!r} must be a real numeric array, got dtype {g.dtype}.")
            if p.shape != g.shape:
                raise ValueError(f"shape mismatch for {key!r}: param {p.shape} vs grad {g.shape}.")
            if not np.all(np.isfinite(p)):
                raise ValueError(f"param {key!r} contains non-finite values; refusing to update.")
            if not np.all(np.isfinite(g)):
                raise ValueError(f"grad {key!r} contains non-finite values; refusing to update.")
            checked.append((p, g))

        eta = self.learning_rate
        for p, g in checked:
            p -= eta * g
