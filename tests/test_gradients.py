"""数值梯度检查：解析梯度 vs 中心有限差分（float64，ε = 1e-5，目标误差 < 1e-5）。"""

import numpy as np

from src.mlp import MLP
from tests.grad_check import analytic_grads, model_loss, numerical_gradients, relative_error

GRAD_CHECK_EPS = 1e-5
REL_ERROR_THRESHOLD = 1e-5


def _check_all_gradients(mlp, X, y, eps: float = GRAD_CHECK_EPS) -> dict:
    """比较全部参数的解析梯度与数值梯度，返回 {键: 相对误差} 并断言达标。"""
    analytic = analytic_grads(mlp, X, y)
    numeric = numerical_gradients(lambda: model_loss(mlp, X, y), mlp.params, eps=eps)
    errors = {key: relative_error(numeric[key], analytic[key]) for key in analytic}
    for key, error in errors.items():
        assert error < REL_ERROR_THRESHOLD, (
            f"{key}: relative error {error:.3e} >= {REL_ERROR_THRESHOLD}"
        )
    return errors


def test_gradients_sigmoid_one_hidden_layer():
    mlp = MLP(4, [5], 3, hidden_activation="sigmoid", seed=42)
    rng = np.random.default_rng(100)
    errors = _check_all_gradients(mlp, rng.normal(size=(3, 4)), rng.integers(0, 3, size=3))
    assert set(errors) == {"W1", "b1", "W2", "b2"}


def test_gradients_tanh_two_hidden_layers():
    mlp = MLP(4, [5, 6], 3, hidden_activation="tanh", seed=7)
    rng = np.random.default_rng(101)
    errors = _check_all_gradients(mlp, rng.normal(size=(3, 4)), rng.integers(0, 3, size=3))
    assert set(errors) == {"W1", "b1", "W2", "b2", "W3", "b3"}


def test_gradients_sigmoid_two_hidden_layers():
    mlp = MLP(4, [5, 6], 3, hidden_activation="sigmoid", seed=11)
    rng = np.random.default_rng(102)
    errors = _check_all_gradients(mlp, rng.normal(size=(4, 4)), rng.integers(0, 3, size=4))
    assert set(errors) == {"W1", "b1", "W2", "b2", "W3", "b3"}


def _min_hidden_preactivation_margin(mlp, X) -> float:
    """所有隐藏层预激活中绝对值的最小值（到 ReLU 不可导点 0 的距离）。"""
    _, cache = mlp.forward(X, return_cache=True)
    return min(float(np.abs(z).min()) for z in cache["Z"][:-1])


def test_gradients_relu_avoids_kink_points():
    mlp = MLP(4, [5, 6], 3, hidden_activation="relu", seed=21)
    rng = np.random.default_rng(103)
    X = rng.normal(size=(3, 4))
    margin = _min_hidden_preactivation_margin(mlp, X)
    # 保证所有被检查的预激活都远离 0（远大于差分步长及其在网络中的传播量级），
    # 有限差分不会跨过 ReLU 的不可导点，与 z=0 处导数取 0 的次梯度约定不冲突。
    assert margin > 1e-2, f"pre-activation margin {margin:.3e} too small for reliable checking"
    errors = _check_all_gradients(mlp, X, rng.integers(0, 3, size=3))
    assert set(errors) == {"W1", "b1", "W2", "b2", "W3", "b3"}
