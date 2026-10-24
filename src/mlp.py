"""手写多层感知机（MLP）：参数初始化、前向传播、反向传播与推理辅助。

形状约定（每行一个样本，全程 float64）
--------------------------------------
设层宽列表为 [d_0, d_1, ..., d_L]，其中 d_0 = input_dim，d_L = output_dim。
对第 l 层（l = 1..L，1-based）：

* W_l 形状 (d_{l-1}, d_l)，b_l 形状 (1, d_l)；
* Z_l = A_{l-1} @ W_l + b_l；
* 隐藏层 A_l = f(Z_l)（f 为该层激活函数）；
* 输出层不施加激活：logits = Z_L。Softmax 在损失函数 / predict_proba 中计算。

对 batch size 为 m：X ∈ (m, d_0)，logits ∈ (m, d_L)，各层梯度与参数形状一致。

反向传播接收损失模块给出的 dJ/dlogits（批次平均），返回与 self.params
键一一对应的梯度字典；本模块不更新参数，更新由优化器单独执行。
"""

import numpy as np

from .activations import activate, activation_gradient, check_activation_name
from .losses import softmax

_INITIALIZER_CHOICES = ("auto", "he_normal", "xavier_normal")


def _validate_positive_int(value, name: str) -> int:
    """校验“正整数”参数；bool 视为非整数拒绝。"""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return int(value)


def _as_float_matrix(arr, name: str) -> np.ndarray:
    """转换为 float64 二维数组；拒绝复数/非数值类型、非二维与非有限值。总是返回新数组。"""
    raw = np.asarray(arr)
    if np.issubdtype(raw.dtype, np.complexfloating) or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must be a real numeric array, got dtype {raw.dtype}.")
    if raw.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {raw.ndim}-D with shape {raw.shape}.")
    out = raw.astype(np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values (NaN or Inf).")
    return out


def _check_matrix(arr, name: str) -> np.ndarray:
    """校验 arr 为实数数值型二维有限数组；返回原数组（不复制，供只读检查）。"""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}.")
    if np.issubdtype(arr.dtype, np.complexfloating) or not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must be a real numeric array, got dtype {arr.dtype}.")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {arr.ndim}-D with shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN or Inf).")
    return arr


class MLP:
    """多层感知机（不硬编码 MNIST，输入/输出维度由调用方提供）。

    参数
    ----
    input_dim: 输入特征维度（正整数）。
    hidden_dims: 隐藏层宽度列表，如 [128] 或 [128, 64]；至少一层，每项为正整数。
    output_dim: 输出类别数（正整数）。
    hidden_activation: 字符串（所有隐藏层共用，默认 "relu"），或与 hidden_dims
        等长的字符串列表逐层配置；支持 relu / sigmoid / tanh。
    initializer: "auto"（默认：ReLU 隐藏层用 He 正态，Sigmoid/Tanh 隐藏层与
        输出层用 Xavier 正态）、"he_normal" 或 "xavier_normal"（显式覆盖所有层）。
        He: std = sqrt(2 / fan_in)；Xavier normal: std = sqrt(2 / (fan_in + fan_out))。
        偏置一律初始化为零。
    rng / seed: 二选一。rng 须为 numpy.random.Generator；seed 为非负整数或 None。
        相同 seed（或相同随机序列的 Generator）与相同配置产生可复现的初始参数；
        不依赖 NumPy 全局随机状态。实际每层采用的初始化策略记录在
        self.layer_initializers 中，可检查。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims,
        output_dim: int,
        hidden_activation="relu",
        initializer: str = "auto",
        rng=None,
        seed=None,
    ) -> None:
        self.input_dim = _validate_positive_int(input_dim, "input_dim")
        self.output_dim = _validate_positive_int(output_dim, "output_dim")

        if isinstance(hidden_dims, str) or not isinstance(hidden_dims, (list, tuple, np.ndarray)):
            raise TypeError(
                f"hidden_dims must be a sequence of positive integers, got {type(hidden_dims).__name__}."
            )
        hidden_list = [
            _validate_positive_int(width, f"hidden_dims[{i}]") for i, width in enumerate(hidden_dims)
        ]
        if not hidden_list:
            raise ValueError("hidden_dims must contain at least one hidden layer.")
        self.hidden_dims: list[int] = hidden_list

        if isinstance(hidden_activation, str):
            normalized = check_activation_name(hidden_activation)
            self.hidden_activations: list[str] = [normalized] * len(hidden_list)
        elif isinstance(hidden_activation, (list, tuple)):
            if len(hidden_activation) != len(hidden_list):
                raise ValueError(
                    f"hidden_activation list length {len(hidden_activation)} must match "
                    f"number of hidden layers {len(hidden_list)}."
                )
            self.hidden_activations = [check_activation_name(name) for name in hidden_activation]
        else:
            raise TypeError(
                f"hidden_activation must be a string or a list of strings, "
                f"got {type(hidden_activation).__name__}."
            )

        if not isinstance(initializer, str):
            raise TypeError(f"initializer must be a string, got {type(initializer).__name__}.")
        if initializer not in _INITIALIZER_CHOICES:
            raise ValueError(
                f"unsupported initializer {initializer!r}; choices: {', '.join(_INITIALIZER_CHOICES)}."
            )

        if rng is not None and seed is not None:
            raise ValueError("pass either rng or seed, not both.")
        if rng is not None:
            if not isinstance(rng, np.random.Generator):
                raise TypeError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}.")
            self._rng = rng
        else:
            if seed is not None and (
                isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0
            ):
                raise ValueError(f"seed must be a non-negative integer or None, got {seed!r}.")
            self._rng = np.random.default_rng(seed)

        self.layer_dims: list[int] = [self.input_dim] + self.hidden_dims + [self.output_dim]
        if initializer == "auto":
            self.layer_initializers: list[str] = [
                "he_normal" if act == "relu" else "xavier_normal"
                for act in self.hidden_activations
            ] + ["xavier_normal"]  # 输出 logits 层固定用 Xavier
        else:
            self.layer_initializers = [initializer] * self.num_layers

        self.params: dict[str, np.ndarray] = {}
        for layer in range(1, self.num_layers + 1):
            fan_in = self.layer_dims[layer - 1]
            fan_out = self.layer_dims[layer]
            if self.layer_initializers[layer - 1] == "he_normal":
                std = np.sqrt(2.0 / fan_in)
            else:  # xavier_normal
                std = np.sqrt(2.0 / (fan_in + fan_out))
            self.params[f"W{layer}"] = self._rng.normal(0.0, std, size=(fan_in, fan_out))
            self.params[f"b{layer}"] = np.zeros((1, fan_out), dtype=np.float64)

    @property
    def num_layers(self) -> int:
        """总层数 L（含输出层；隐藏层数为 L - 1）。"""
        return len(self.layer_dims) - 1

    def forward(self, X, return_cache: bool = False):
        """前向传播，返回 logits（不提前 Softmax）。

        X 须为形状 (m, input_dim) 的非空有限二维数组；输入不会被修改。
        return_cache=True 时返回 (logits, cache)：cache 含逐层输入 A（A_0 = X，
        之后为各隐藏层激活输出）与预激活 Z_1..Z_L，供 backward 使用；
        cache 每次调用新建，不会复用上一次前向的结果。
        """
        batch = _as_float_matrix(X, "X")
        if batch.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if batch.shape[1] != self.input_dim:
            raise ValueError(
                f"X feature dimension {batch.shape[1]} does not match input_dim {self.input_dim}."
            )

        a_current = batch
        a_list: list[np.ndarray] = [batch]  # A_0..A_{L-1}：各层输入
        z_list: list[np.ndarray] = []       # Z_1..Z_L：各层预激活
        for layer in range(1, self.num_layers + 1):
            z_current = a_current @ self.params[f"W{layer}"] + self.params[f"b{layer}"]
            z_list.append(z_current)
            if layer < self.num_layers:
                a_current = activate(self.hidden_activations[layer - 1], z_current)
                a_list.append(a_current)
            # 输出层不施加激活，logits = Z_L
        logits = z_list[-1]
        if return_cache:
            return logits, {"A": a_list, "Z": z_list}
        return logits

    def backward(self, dlogits, cache) -> dict[str, np.ndarray]:
        """根据 dJ/dlogits 反向计算所有权重与偏置梯度。

        dlogits 是损失函数返回的批次平均梯度 ∂J/∂Z_L（不是标签，也不是
        未归一化误差），本方法不再除以 batch size。链式法则：

            delta^(L) = dJ/dlogits
            dW^(l) = (A^(l-1))^T @ delta^(l)，  db^(l) = Σ_i delta_i^(l)
            delta^(l-1) = (delta^(l) @ (W^(l))^T) ⊙ f'^(l-1)(Z^(l-1))

        返回 {"W1": ..., "b1": ..., ...}，键与 self.params 一一对应，形状相同。
        不修改任何参数，也不修改 dlogits；缓存非法或维度不相容时报错。
        """
        a_list, z_list = self._validate_cache(cache)
        delta = _as_float_matrix(dlogits, "dlogits")
        batch_size = a_list[0].shape[0]
        if delta.shape != (batch_size, self.output_dim):
            raise ValueError(
                f"dlogits shape {delta.shape} incompatible with batch size {batch_size} "
                f"and output_dim {self.output_dim}."
            )

        grads: dict[str, np.ndarray] = {}
        for layer in range(self.num_layers, 0, -1):
            a_prev = a_list[layer - 1]
            grads[f"W{layer}"] = a_prev.T @ delta
            grads[f"b{layer}"] = delta.sum(axis=0, keepdims=True)
            if layer > 1:
                # A^(l-1) = f(Z^(l-1))：预激活在 Z_list[l-2]，激活输出在 A_list[l-1]
                delta = (delta @ self.params[f"W{layer}"].T) * activation_gradient(
                    self.hidden_activations[layer - 2], z_list[layer - 2], a_list[layer - 1]
                )
        return grads

    def _validate_cache(self, cache):
        """校验 backward 收到的缓存结构，返回 (A_list, Z_list)。"""
        num_layers = self.num_layers
        if not isinstance(cache, dict) or set(cache.keys()) < {"A", "Z"}:
            raise ValueError(
                "cache must be the dict returned by forward(..., return_cache=True) "
                "with keys 'A' and 'Z'."
            )
        a_list, z_list = cache["A"], cache["Z"]
        if not isinstance(a_list, (list, tuple)) or not isinstance(z_list, (list, tuple)):
            raise ValueError("cache['A'] and cache['Z'] must be lists of arrays.")
        if len(a_list) != num_layers or len(z_list) != num_layers:
            raise ValueError(
                f"cache['A']/cache['Z'] must each contain {num_layers} arrays (one per layer), "
                f"got {len(a_list)} and {len(z_list)}."
            )
        for index in range(num_layers):
            z_checked = _check_matrix(z_list[index], f"cache['Z'][{index}]")
            if z_checked.shape[1] != self.layer_dims[index + 1]:
                raise ValueError(
                    f"cache['Z'][{index}] shape {z_checked.shape} does not match "
                    f"layer width {self.layer_dims[index + 1]}."
                )
        batch_size = z_list[0].shape[0]
        for index in range(1, num_layers):
            if z_list[index].shape[0] != batch_size:
                raise ValueError("all cached arrays must share the same batch size.")
        a_input = _check_matrix(a_list[0], "cache['A'][0]")
        if a_input.shape != (batch_size, self.input_dim):
            raise ValueError(
                f"cache['A'][0] shape {a_input.shape} does not match "
                f"(batch_size, input_dim) = ({batch_size}, {self.input_dim})."
            )
        for index in range(1, num_layers):
            a_checked = _check_matrix(a_list[index], f"cache['A'][{index}]")
            if a_checked.shape != (batch_size, self.layer_dims[index]):
                raise ValueError(
                    f"cache['A'][{index}] shape {a_checked.shape} does not match "
                    f"(batch_size, {self.layer_dims[index]})."
                )
        return a_list, z_list

    def predict_proba(self, X) -> np.ndarray:
        """类别概率：forward 得到的 logits 经稳定 Softmax。只推理，不改参数。"""
        return softmax(self.forward(X))

    def predict(self, X) -> np.ndarray:
        """整数预测类别（每行概率的 argmax）。"""
        return np.argmax(self.predict_proba(X), axis=1)

    def get_params_copy(self) -> dict[str, np.ndarray]:
        """返回参数的深复制备份；后续训练不会影响备份。"""
        return {key: value.copy() for key, value in self.params.items()}

    def set_params(self, params) -> None:
        """用给定参数字典恢复参数（就地写回，不与外部数组共享内存）。

        校验键集合、形状与数值有限性；全部通过后才写入，避免部分恢复。
        """
        if not isinstance(params, dict):
            raise TypeError(f"params must be a dict, got {type(params).__name__}.")
        if set(params) != set(self.params):
            missing = sorted(set(self.params) - set(params))
            unexpected = sorted(set(params) - set(self.params))
            raise ValueError(f"parameter keys mismatch; missing: {missing}; unexpected: {unexpected}.")
        staged: dict[str, np.ndarray] = {}
        for key, target in self.params.items():
            value = params[key]
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"parameter {key!r} must be a numpy.ndarray, got {type(value).__name__}."
                )
            if np.issubdtype(value.dtype, np.complexfloating) or not np.issubdtype(value.dtype, np.number):
                raise TypeError(f"parameter {key!r} must be a real numeric array, got dtype {value.dtype}.")
            if value.shape != target.shape:
                raise ValueError(f"parameter {key!r} shape {value.shape} does not match {target.shape}.")
            converted = value.astype(np.float64)
            if not np.all(np.isfinite(converted)):
                raise ValueError(f"parameter {key!r} contains non-finite values.")
            staged[key] = converted
        for key, converted in staged.items():
            self.params[key][...] = converted
