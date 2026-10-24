"""优化器模块：基础 SGD、Momentum SGD 与 Adam（均为手写实现）。

公式与安全约定遵循《P5 实验约定冻结 V0.1》：

* SGD：θ ← θ − η·g（无状态，两阶段原子更新）；
* MomentumSGD：v_t = momentum·v_{t-1} + g_t，θ_t = θ_{t-1} − η·v_t（v_0 = 0，
  无 dampening / Nesterov）；
* Adam：一阶/二阶矩估计 + t=1 起的偏置修正，
  θ_t = θ_{t-1} − η·m_hat/(sqrt(v_hat) + eps)（eps 位于 sqrt(v_hat) 之后，
  无 AMSGrad / weight decay 变体）。

有状态优化器（MomentumSGD/Adam）的实例在首次成功更新时绑定该次传入的
params 字典对象与各参数 ndarray 对象；之后更换字典、替换参数数组或键/形状/
dtype 契约变化都会被拒绝。所有优化器的每次更新都先全量校验输入、在受控
浮点环境下计算全部候选参数与候选状态并验证有限性，然后才统一就地提交；
任何失败都不产生部分写入，参数数组对象身份不变，梯度数组不被修改。

参数更新就地执行；本模块不负责计算梯度。训练器接入新优化器属于后续批次，
不在本模块范围内。
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

        分三个阶段保证“失败时无部分写入”：

        1. 准备阶段：完成全部输入校验（键集合、形状、有限性）后，在受控的
           浮点异常环境（np.errstate 显式忽略 over/invalid，随后自行检查结果）
           下计算每个参数的候选更新值到临时数组，并验证其在计算 dtype 下全部
           有限。有限的 η 与 g 相乘仍可能溢出（如 2.0 × 1e308 → Inf），因此
           候选值必须验证：任一候选出现 NaN/Inf 即抛出异常，所有原参数逐元素
           不变。
        2. 目标 dtype 转换验证：候选在计算 dtype 下有限，不代表能安全写回更窄
           的参数 dtype（如 float64 候选 -1e40 写入 float32 参数会静默变成
           -Inf）。先对全部候选做目标 dtype 转换（dtype 相同则直接复用候选），
           再对全部转换结果检查有限性；任一转换结果出现 NaN/Inf 即抛出异常，
           所有原参数逐元素不变。
        3. 提交阶段：全部转换后的候选通过验证后，才统一就地写回原参数数组
           （p[...] = cast），保持 params[key] 指向的 ndarray 对象身份与 dtype
           不变；梯度数组全程不被修改。
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
        candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
        with np.errstate(over="ignore", invalid="ignore"):
            for key, p, g in checked:
                candidate = p - eta * g
                if not np.all(np.isfinite(candidate)):
                    raise ValueError(
                        f"update for {key!r} would produce non-finite values "
                        f"(overflow or invalid operation); refusing to update any parameter."
                    )
                candidates.append((key, p, candidate))

        # 目标 dtype 转换验证：候选在计算 dtype 下有限，不代表能安全写回更窄的
        # 参数 dtype（如 float64 候选 -1e40 写入 float32 参数会静默变成 -Inf）。
        # 先对全部候选做目标 dtype 转换（显式管理浮点告警），再全部检查，最后
        # 才统一提交——任何失败都不写入任何参数。
        casted: list[tuple[str, np.ndarray, np.ndarray]] = []
        with np.errstate(over="ignore", invalid="ignore"):
            for key, p, candidate in candidates:
                if candidate.dtype == p.dtype:
                    casted.append((key, p, candidate))
                else:
                    casted.append((key, p, candidate.astype(p.dtype)))
        for key, p, cast in casted:
            if not np.all(np.isfinite(cast)):
                raise ValueError(
                    f"update for {key!r} overflows after casting the candidate to the "
                    f"target parameter dtype {p.dtype}; refusing to update any parameter."
                )
        # 提交阶段：全部转换后的候选通过验证后统一就地写回，数组对象身份不变。
        for _key, p, cast in casted:
            p[...] = cast


class _StatefulOptimizer:
    """MomentumSGD 与 Adam 的公共骨架：参数归属绑定、输入校验与两阶段原子提交。

    与 SGD 的关系：SGD 保持独立实现、行为不变；本基类只服务有状态优化器。

    绑定语义（P5 实验约定冻结 V0.1）：
    * 首次成功完成 step 后，实例绑定该次传入的 params 字典对象与各参数
      ndarray 对象；
    * 之后传入其他字典、替换参数数组，或键/形状/dtype 契约变化，都会抛出
      ValueError 且不触碰参数与内部状态；
    * 首次调用失败时不绑定、不留下任何半初始化状态。
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
        self._bound_params: dict | None = None
        self._bound_arrays: dict[str, np.ndarray] = {}
        self._bound_shapes: dict[str, tuple] = {}
        self._bound_dtypes: dict[str, np.dtype] = {}
        self._state: dict = {}
        self._timestep: int = 0  # Adam 的调用步数 t；MomentumSGD 不使用

    def _check_binding_contract(self, params: dict) -> None:
        """绑定后的归属契约校验；任何不一致在读取参数之前就拒绝。"""
        if self._bound_params is None:
            return
        if params is not self._bound_params:
            raise ValueError(
                "this optimizer instance is bound to the params dict of its first successful "
                "update; step() must always receive that same dict, or a new optimizer "
                "instance must be created for a new training run."
            )
        if set(params) != set(self._bound_arrays):
            raise ValueError(
                "parameter keys changed after this optimizer bound to its params dict; "
                f"bound keys: {sorted(self._bound_arrays)}, current keys: {sorted(params)}."
            )
        for key in sorted(self._bound_arrays):
            current = params[key]
            if current is not self._bound_arrays[key]:
                raise ValueError(
                    f"parameter array {key!r} was replaced after this optimizer bound to it; "
                    f"the accumulated optimizer state would no longer match. Reuse the "
                    f"original ndarray or create a new optimizer instance."
                )
            if current.shape != self._bound_shapes[key]:
                raise ValueError(
                    f"parameter {key!r} shape changed from {self._bound_shapes[key]} "
                    f"to {current.shape}."
                )
            if current.dtype != self._bound_dtypes[key]:
                raise ValueError(
                    f"parameter {key!r} dtype changed from {self._bound_dtypes[key]} "
                    f"to {current.dtype}."
                )

    def _validate_inputs(self, params: dict, grads: dict) -> list:
        """全量输入校验（键集合、dtype、形状、可写性、有限性），返回按 key 排序的 (key, p, g)。"""
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
            if not p.flags.writeable:
                raise ValueError(
                    f"param {key!r} is a read-only array; in-place update is impossible."
                )
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
        return checked

    def _compute_candidates(self, checked: list):
        """计算全部候选参数与候选状态；不得修改 self 或任何输入数组。

        返回 (param_candidates, new_state, new_timestep)：
        * param_candidates: (key, param_array, new_value) 列表；
        * new_state: 整份新状态（首步在此惰性初始化）；
        * new_timestep: None 表示该优化器无全局时间步。
        """
        raise NotImplementedError

    def step(self, params: dict, grads: dict) -> None:
        """执行一步更新；先全量校验并计算候选，验证有限（含写回目标 dtype
        转换后的溢出检查）后统一提交（失败零写入）。"""
        if not isinstance(params, dict):
            raise TypeError(f"params must be a dict, got {type(params).__name__}.")
        if not isinstance(grads, dict):
            raise TypeError(f"grads must be a dict, got {type(grads).__name__}.")
        self._check_binding_contract(params)
        checked = self._validate_inputs(params, grads)
        param_candidates, new_state, new_timestep = self._compute_candidates(checked)
        for key, _p, candidate in param_candidates:
            if not np.all(np.isfinite(candidate)):
                raise ValueError(
                    f"update for {key!r} would produce non-finite values "
                    f"(overflow or invalid operation); refusing to update any parameter or state."
                )
        for key, state_value in new_state.items():
            arrays = state_value if isinstance(state_value, tuple) else (state_value,)
            for arr in arrays:
                if not np.all(np.isfinite(arr)):
                    raise ValueError(
                        f"candidate optimizer state for {key!r} would contain non-finite "
                        f"values; refusing to update any parameter or state."
                    )
        # 目标 dtype 转换验证：候选在计算 dtype 下有限，不代表能安全写回更窄的
        # 参数 dtype（如 float64 候选 -1e40 写入 float32 参数会静默变成 -Inf）。
        # 先对全部候选做目标 dtype 转换（显式管理浮点告警），再全部检查，最后
        # 才统一提交——任何失败都不写入任何参数、状态或时间步。
        casted: list[tuple[str, np.ndarray, np.ndarray]] = []
        with np.errstate(over="ignore", invalid="ignore"):
            for key, p, candidate in param_candidates:
                if candidate.dtype == p.dtype:
                    casted.append((key, p, candidate))
                else:
                    casted.append((key, p, candidate.astype(p.dtype)))
        for key, p, cast in casted:
            if not np.all(np.isfinite(cast)):
                raise ValueError(
                    f"update for {key!r} overflows after casting the candidate to the "
                    f"target parameter dtype {p.dtype}; refusing to update any parameter "
                    f"or state."
                )
        # 统一提交：参数就地写回（对象身份不变），状态与时间步整体替换
        for _key, p, cast in casted:
            p[...] = cast
        self._state = new_state
        if new_timestep is not None:
            self._timestep = new_timestep
        if self._bound_params is None:  # 首次成功更新后才绑定
            self._bound_params = params
            self._bound_arrays = {key: p for key, p, _ in checked}
            self._bound_shapes = {key: p.shape for key, p, _ in checked}
            self._bound_dtypes = {key: p.dtype for key, p, _ in checked}


class MomentumSGD(_StatefulOptimizer):
    """带动量的随机梯度下降（无 dampening / Nesterov）。

    公式（P5 实验约定冻结 V0.1）：
        v_t = momentum * v_{t-1} + g_t
        theta_t = theta_{t-1} - learning_rate * v_t
    v_0 为全零数组，因此首步 v_1 = g_1，等价于普通 SGD。
    learning_rate 为有限正实数；momentum 为有限实数且 0 <= momentum < 1。
    """

    def __init__(self, learning_rate: float, momentum: float = 0.9) -> None:
        super().__init__(learning_rate)
        if not _is_real_number(momentum):
            raise TypeError(f"momentum must be a real number, got {type(momentum).__name__}.")
        m = float(momentum)
        if not np.isfinite(m):
            raise ValueError(f"momentum must be finite, got {momentum!r}.")
        if not 0.0 <= m < 1.0:
            raise ValueError(f"momentum must satisfy 0 <= momentum < 1, got {momentum!r}.")
        self.momentum: float = m

    def _compute_candidates(self, checked: list):
        param_candidates = []
        new_state: dict[str, np.ndarray] = {}
        eta, mu = self.learning_rate, self.momentum
        with np.errstate(over="ignore", invalid="ignore"):
            for key, p, g in checked:
                v_prev = self._state.get(key)
                if v_prev is None:  # 惰性初始化：v_0 = 0
                    v_prev = np.zeros_like(p)
                v_new = mu * v_prev + g
                new_param = p - eta * v_new
                param_candidates.append((key, p, new_param))
                new_state[key] = v_new
        return param_candidates, new_state, None


class Adam(_StatefulOptimizer):
    """Adam 优化器（带偏置修正，无 AMSGrad / weight decay 变体）。

    公式（P5 实验约定冻结 V0.1）：
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t * g_t)
        m_hat = m_t / (1 - beta1**t)
        v_hat = v_t / (1 - beta2**t)
        theta_t = theta_{t-1} - learning_rate * m_hat / (sqrt(v_hat) + eps)
    t 从 1 起算；epsilon 位于 sqrt(v_hat) 之后。
    learning_rate 与 eps 为有限正实数；beta1、beta2 为有限实数且 0 <= beta < 1。
    """

    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        for name, value in (("beta1", beta1), ("beta2", beta2)):
            if not _is_real_number(value):
                raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
            beta = float(value)
            if not np.isfinite(beta):
                raise ValueError(f"{name} must be finite, got {value!r}.")
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"{name} must satisfy 0 <= {name} < 1, got {value!r}.")
        if not _is_real_number(eps):
            raise TypeError(f"eps must be a real number, got {type(eps).__name__}.")
        eps_value = float(eps)
        if not np.isfinite(eps_value):
            raise ValueError(f"eps must be finite, got {eps!r}.")
        if eps_value <= 0.0:
            raise ValueError(f"eps must be positive, got {eps!r}.")
        self.beta1: float = float(beta1)
        self.beta2: float = float(beta2)
        self.eps: float = eps_value

    def _compute_candidates(self, checked: list):
        param_candidates = []
        new_state: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        eta, b1, b2, eps = self.learning_rate, self.beta1, self.beta2, self.eps
        t_new = self._timestep + 1
        bias1 = 1.0 - b1 ** t_new
        bias2 = 1.0 - b2 ** t_new
        with np.errstate(over="ignore", invalid="ignore"):
            for key, p, g in checked:
                moments = self._state.get(key)
                if moments is None:  # 惰性初始化：m_0 = v_0 = 0
                    m_prev = np.zeros_like(p)
                    v_prev = np.zeros_like(p)
                else:
                    m_prev, v_prev = moments
                m_new = b1 * m_prev + (1.0 - b1) * g
                v_new = b2 * v_prev + (1.0 - b2) * (g * g)
                m_hat = m_new / bias1
                v_hat = v_new / bias2
                new_param = p - eta * m_hat / (np.sqrt(v_hat) + eps)
                param_candidates.append((key, p, new_param))
                new_state[key] = (m_new, v_new)
        return param_candidates, new_state, t_new
