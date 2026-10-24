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

        分两个阶段保证“失败时无部分写入”：

        1. 准备阶段：完成全部输入校验（键集合、形状、有限性）后，在受控的
           浮点异常环境（np.errstate 显式忽略 over/invalid，随后自行检查结果）
           下计算每个参数的候选更新值到临时数组，并验证其全部有限。
           有限的 η 与 g 相乘仍可能溢出（如 2.0 × 1e308 → Inf），因此候选值
           必须验证：任一候选出现 NaN/Inf 即抛出异常，所有原参数逐元素不变。
        2. 提交阶段：所有候选都通过验证后，才统一就地写回原参数数组
           （p[...] = candidate），保持 params[key] 指向的 ndarray 对象身份不变；
           梯度数组全程不被修改。
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

        checked: list[tuple[str, np.ndarray, np.ndarray]] = []
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
            checked.append((key, p, g))

        # 准备阶段：计算全部候选更新并验证有限性（此时不写任何参数）。
        # 溢出/无效运算在此被显式受控（忽略警告、随后自行检查结果），不依赖警告被忽略的默认行为。
        eta = self.learning_rate
        candidates: list[tuple[np.ndarray, np.ndarray]] = []
        with np.errstate(over="ignore", invalid="ignore"):
            for key, p, g in checked:
                candidate = p - eta * g
                if not np.all(np.isfinite(candidate)):
                    raise ValueError(
                        f"update for {key!r} would produce non-finite values "
                        f"(overflow or invalid operation); refusing to update any parameter."
                    )
                candidates.append((p, candidate))

        # 提交阶段：全部候选有效后统一就地写回，数组对象身份不变。
        for p, candidate in candidates:
            p[...] = candidate
