"""SGD 优化器测试：手工算例、更新方向、累积更新、校验与异常处理。"""

import numpy as np
import pytest

from src.losses import softmax_cross_entropy
from src.mlp import MLP
from src.optimizers import SGD


def test_sgd_hand_computed_update():
    params = {
        "w": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "b": np.array([[0.5, -0.5]]),
    }
    grads = {
        "w": np.array([[0.1, -0.2], [0.3, 0.4]]),
        "b": np.array([[1.0, 2.0]]),
    }
    SGD(learning_rate=0.1).step(params, grads)
    # θ ← θ − η·g，逐元素手算
    np.testing.assert_allclose(
        params["w"], [[1.0 - 0.01, 2.0 + 0.02], [3.0 - 0.03, 4.0 - 0.04]]
    )
    np.testing.assert_allclose(params["b"], [[0.4, -0.7]])


def test_sgd_updates_in_place_and_opposite_to_gradient():
    rng = np.random.default_rng(0)
    params = {"w": rng.normal(size=(4, 3))}
    grads = {"w": rng.normal(size=(4, 3))}
    old = params["w"].copy()
    handle = params["w"]
    SGD(0.5).step(params, grads)
    assert params["w"] is handle  # 就地更新同一数组对象
    np.testing.assert_allclose(params["w"], old - 0.5 * grads["w"])
    assert np.all((params["w"] - old) * grads["w"] < 0.0)  # 每个分量都朝负梯度方向移动


def test_sgd_multiple_steps_accumulate():
    params = {"w": np.array([1.0])}
    grads = {"w": np.array([1.0])}
    optimizer = SGD(0.1)
    for _ in range(10):
        optimizer.step(params, grads)
    np.testing.assert_allclose(params["w"], [0.0], atol=1e-15)  # 允许浮点累积误差


def test_sgd_rejects_invalid_learning_rate():
    for bad in (0, -0.1, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            SGD(bad)
    for bad in ("0.1", None, True, [0.1]):
        with pytest.raises(TypeError):
            SGD(bad)


def test_sgd_key_mismatch():
    params = {"w": np.array([1.0]), "b": np.array([0.0])}
    with pytest.raises(ValueError, match="key"):
        SGD(0.1).step(params, {"w": np.array([0.1])})  # 缺 b
    with pytest.raises(ValueError, match="key"):
        SGD(0.1).step(
            params,
            {"w": np.array([0.1]), "b": np.array([0.0]), "c": np.array([0.0])},  # 多出 c
        )


def test_sgd_shape_mismatch_leaves_params_unchanged():
    params = {"w": np.array([1.0, 2.0])}
    before = params["w"].copy()
    with pytest.raises(ValueError):
        SGD(0.1).step(params, {"w": np.array([1.0])})
    np.testing.assert_array_equal(params["w"], before)


def test_sgd_rejects_non_finite_values():
    with pytest.raises(ValueError):
        SGD(0.1).step({"w": np.array([1.0])}, {"w": np.array([np.nan])})
    with pytest.raises(ValueError):
        SGD(0.1).step({"w": np.array([1.0])}, {"w": np.array([np.inf])})
    with pytest.raises(ValueError):
        SGD(0.1).step({"w": np.array([np.nan])}, {"w": np.array([0.1])})  # 参数非有限


def test_sgd_validation_is_atomic():
    # 一个参数非法时，另一个合法参数也不能被更新
    params = {"a": np.array([1.0]), "b": np.array([1.0])}
    grads = {"a": np.array([0.5]), "b": np.array([np.nan])}
    with pytest.raises(ValueError):
        SGD(0.1).step(params, grads)
    np.testing.assert_array_equal(params["a"], [1.0])


def test_sgd_rejects_overflowing_candidate_update_atomically():
    # 回归（审查 R1）：输入全部有限，但候选更新 η·g 溢出为 Inf 时，
    # 必须整体拒绝，任何参数（包括排序在前、单独更新本不会溢出的那个）
    # 都不得被部分写入，且参数 ndarray 对象身份保持不变。
    params = {
        "a_safe": np.array([1.0, -2.0]),   # 梯度 1.0：单独更新不会溢出
        "b_over": np.array([[3.0]]),       # 梯度 1e308，η=2.0 → 2e308 溢出为 Inf
    }
    grads = {
        "a_safe": np.array([1.0, 1.0]),
        "b_over": np.array([[1e308]]),
    }
    before = {key: value.copy() for key, value in params.items()}
    handles = {key: params[key] for key in params}
    assert sorted(params) == ["a_safe", "b_over"]  # 保证安全参数排在更新序列前面

    with pytest.raises(ValueError, match="non-finite"):
        SGD(2.0).step(params, grads)

    for key in params:
        np.testing.assert_array_equal(params[key], before[key])  # 逐元素完全不变
        assert params[key] is handles[key]                        # 对象身份不变


def test_sgd_rejects_non_dict_and_bad_dtypes():
    with pytest.raises(TypeError):
        SGD(0.1).step([("w", np.array([1.0]))], {"w": np.array([0.1])})  # params 非字典
    with pytest.raises(TypeError):
        SGD(0.1).step({"w": np.array([1.0])}, {"w": [0.1]})  # 梯度不是 ndarray
    with pytest.raises(TypeError):
        SGD(0.1).step({"w": np.array([1, 2])}, {"w": np.array([0.1, 0.2])})  # 参数须为浮点


def test_sgd_updates_mlp_params():
    mlp = MLP(4, [5], 3, hidden_activation="tanh", seed=0)
    X = np.random.default_rng(1).normal(size=(4, 4))
    y = np.array([0, 1, 2, 0])
    before = mlp.get_params_copy()
    logits, cache = mlp.forward(X, return_cache=True)
    _, dlogits = softmax_cross_entropy(logits, y)
    grads = mlp.backward(dlogits, cache)
    SGD(0.1).step(mlp.params, grads)
    assert all(not np.array_equal(mlp.params[k], before[k]) for k in mlp.params)
