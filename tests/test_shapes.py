"""MLP 维度、多层支持与输入校验测试（小维度网络，不依赖 MNIST）。"""

import numpy as np
import pytest

from src.losses import softmax_cross_entropy
from src.mlp import MLP
from tests.grad_check import analytic_grads


@pytest.mark.parametrize("hidden_dims", [[5], [5, 6]])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_forward_prediction_and_gradient_shapes(hidden_dims, batch_size):
    mlp = MLP(input_dim=4, hidden_dims=hidden_dims, output_dim=3, hidden_activation="tanh", seed=0)
    X = np.random.default_rng(1).normal(size=(batch_size, 4))
    y = np.random.default_rng(2).integers(0, 3, size=batch_size)

    logits = mlp.forward(X)
    assert logits.shape == (batch_size, 3)

    proba = mlp.predict_proba(X)
    assert proba.shape == (batch_size, 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(batch_size), atol=1e-12)

    pred = mlp.predict(X)
    assert pred.shape == (batch_size,)
    assert np.issubdtype(pred.dtype, np.integer)
    assert np.all((pred >= 0) & (pred < 3))

    grads = analytic_grads(mlp, X, y)
    assert set(grads) == set(mlp.params)
    for key in grads:
        assert grads[key].shape == mlp.params[key].shape


def test_parameter_shapes_two_hidden_layers():
    mlp = MLP(4, [5, 6], 3, hidden_activation="relu", seed=0)
    assert set(mlp.params) == {"W1", "b1", "W2", "b2", "W3", "b3"}
    assert mlp.params["W1"].shape == (4, 5)
    assert mlp.params["b1"].shape == (1, 5)
    assert mlp.params["W2"].shape == (5, 6)
    assert mlp.params["b2"].shape == (1, 6)
    assert mlp.params["W3"].shape == (6, 3)
    assert mlp.params["b3"].shape == (1, 3)


def test_course_extension_shape_smoke():
    # 课程扩展形状 [128, 64] 只做维度冒烟测试
    mlp = MLP(6, [128, 64], 4, hidden_activation="relu", seed=0)
    X = np.random.default_rng(3).normal(size=(4, 6))
    y = np.random.default_rng(4).integers(0, 4, size=4)
    logits, cache = mlp.forward(X, return_cache=True)
    assert logits.shape == (4, 4)
    _, dlogits = softmax_cross_entropy(logits, y)
    grads = mlp.backward(dlogits, cache)
    assert set(grads) == set(mlp.params)


def test_per_layer_activation_config():
    mlp = MLP(4, [5, 6], 3, hidden_activation=["tanh", "sigmoid"], seed=0)
    assert mlp.hidden_activations == ["tanh", "sigmoid"]
    X = np.random.default_rng(5).normal(size=(3, 4))
    assert mlp.forward(X).shape == (3, 3)


def test_initializer_selection_recorded():
    relu_net = MLP(4, [5], 3, hidden_activation="relu", seed=0)
    assert relu_net.layer_initializers == ["he_normal", "xavier_normal"]
    tanh_net = MLP(4, [5], 3, hidden_activation="tanh", seed=0)
    assert tanh_net.layer_initializers == ["xavier_normal", "xavier_normal"]
    explicit = MLP(4, [5], 3, hidden_activation="relu", initializer="xavier_normal", seed=0)
    assert explicit.layer_initializers == ["xavier_normal", "xavier_normal"]


def test_same_seed_reproducible_params():
    a = MLP(4, [5, 6], 3, hidden_activation="relu", seed=123)
    b = MLP(4, [5, 6], 3, hidden_activation="relu", seed=123)
    c = MLP(4, [5, 6], 3, hidden_activation="relu", seed=124)
    for key in a.params:
        np.testing.assert_array_equal(a.params[key], b.params[key])
        if key.startswith("W"):
            # 偏置恒为零初始化、与 seed 无关；只有权重必须随 seed 变化
            assert not np.array_equal(a.params[key], c.params[key])


def test_generator_rng_accepted_and_reproducible():
    a = MLP(4, [5], 3, rng=np.random.default_rng(7))
    b = MLP(4, [5], 3, rng=np.random.default_rng(7))
    for key in a.params:
        np.testing.assert_array_equal(a.params[key], b.params[key])


def test_biases_initialized_to_zero():
    mlp = MLP(4, [5], 3, seed=0)
    for key, value in mlp.params.items():
        if key.startswith("b"):
            np.testing.assert_array_equal(value, np.zeros_like(value))


def test_constructor_validation():
    with pytest.raises(ValueError):
        MLP(0, [5], 3)  # input_dim 非正
    with pytest.raises(ValueError):
        MLP(4, [5], 0)  # output_dim 非正
    with pytest.raises(TypeError):
        MLP(4.5, [5], 3)  # 非整数
    with pytest.raises(TypeError):
        MLP(True, [5], 3)  # bool 不算整数
    with pytest.raises(ValueError):
        MLP(4, [], 3)  # 至少一个隐藏层
    with pytest.raises(ValueError):
        MLP(4, [0], 3)  # 零宽度
    with pytest.raises(ValueError):
        MLP(4, [-2, 5], 3)  # 负宽度
    with pytest.raises(TypeError):
        MLP(4, [2.5], 3)  # 非整数宽度
    with pytest.raises(TypeError):
        MLP(4, "128", 3)  # 字符串不算层宽序列
    with pytest.raises(ValueError):
        MLP(4, [5], 3, hidden_activation="gelu")  # 不支持的激活
    with pytest.raises(ValueError):
        MLP(4, [5, 6], 3, hidden_activation=["tanh"])  # 列表长度不符
    with pytest.raises(ValueError):
        MLP(4, [5, 6], 3, hidden_activation=["tanh", "sin"])
    with pytest.raises(ValueError):
        MLP(4, [5], 3, initializer="adam")  # 未知初始化策略
    with pytest.raises(ValueError):
        MLP(4, [5], 3, seed=1, rng=np.random.default_rng(1))  # rng/seed 二选一
    with pytest.raises(TypeError):
        MLP(4, [5], 3, rng=123)  # rng 类型错误


def test_forward_input_validation():
    mlp = MLP(4, [5], 3, seed=0)
    with pytest.raises(ValueError):
        mlp.forward(np.zeros(4))  # 一维
    with pytest.raises(ValueError):
        mlp.forward(np.zeros((2, 5)))  # 特征维度不符
    with pytest.raises(ValueError):
        mlp.forward(np.empty((0, 4)))  # 空批次
    with pytest.raises(ValueError):
        mlp.forward(np.array([[np.nan, 0.0, 0.0, 0.0]]))  # NaN
    with pytest.raises(ValueError):
        mlp.forward(np.array([[np.inf, 0.0, 0.0, 0.0]]))  # Inf
    with pytest.raises(TypeError):
        mlp.forward(np.array([["a", "b", "c", "d"]]))  # 非数值
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    x_copy = x.copy()
    mlp.forward(x)
    np.testing.assert_array_equal(x, x_copy)  # 不修改输入


def test_backward_and_cache_validation():
    mlp = MLP(4, [5], 3, hidden_activation="tanh", seed=0)
    X = np.random.default_rng(6).normal(size=(3, 4))
    y = np.array([0, 1, 2])
    logits, cache = mlp.forward(X, return_cache=True)
    _, dlogits = softmax_cross_entropy(logits, y)

    with pytest.raises(ValueError):
        mlp.backward(np.zeros((2, 3)), cache)  # batch 不符
    with pytest.raises(ValueError):
        mlp.backward(np.zeros((3, 2)), cache)  # output_dim 不符
    with pytest.raises(ValueError):
        mlp.backward(np.array([[np.nan, 0.0, 0.0]]), cache)  # 非有限梯度
    with pytest.raises(ValueError):
        mlp.backward(dlogits, {})  # 缺键
    with pytest.raises(ValueError):
        mlp.backward(dlogits, {"A": cache["A"]})  # 缺 Z
    with pytest.raises(ValueError):
        mlp.backward(dlogits, {"A": cache["A"][:-1], "Z": cache["Z"]})  # 长度不符
    bad = {"A": list(cache["A"]), "Z": list(cache["Z"])}
    bad["A"][0] = np.zeros((3, 7))  # A0 形状不符
    with pytest.raises(ValueError):
        mlp.backward(dlogits, bad)


def test_backward_does_not_modify_params_or_dlogits():
    mlp = MLP(4, [5, 6], 3, hidden_activation="sigmoid", seed=0)
    X = np.random.default_rng(8).normal(size=(3, 4))
    y = np.array([0, 2, 1])
    logits, cache = mlp.forward(X, return_cache=True)
    _, dlogits = softmax_cross_entropy(logits, y)
    params_before = {k: v.copy() for k, v in mlp.params.items()}
    dlogits_before = dlogits.copy()
    mlp.backward(dlogits, cache)
    for key in mlp.params:
        np.testing.assert_array_equal(mlp.params[key], params_before[key])
    np.testing.assert_array_equal(dlogits, dlogits_before)


def test_predict_methods_do_not_modify_params():
    mlp = MLP(4, [5], 3, seed=0)
    X = np.random.default_rng(9).normal(size=(5, 4))
    before = {k: v.copy() for k, v in mlp.params.items()}
    mlp.predict_proba(X)
    mlp.predict(X)
    for key in mlp.params:
        np.testing.assert_array_equal(mlp.params[key], before[key])


def test_get_params_copy_is_deep():
    mlp = MLP(4, [5], 3, seed=0)
    backup = mlp.get_params_copy()
    for value in backup.values():
        value[...] += 1.0  # 就地修改导出的副本
    for key in mlp.params:
        assert not np.array_equal(mlp.params[key], backup[key])


def test_set_params_roundtrip():
    mlp = MLP(4, [5], 3, seed=0)
    backup = mlp.get_params_copy()
    for value in mlp.params.values():
        value[...] += 0.5  # 破坏参数
    mlp.set_params(backup)  # 恢复
    for key in mlp.params:
        np.testing.assert_array_equal(mlp.params[key], backup[key])


def test_set_params_validation_and_atomicity():
    mlp = MLP(4, [5], 3, seed=0)
    backup = mlp.get_params_copy()
    before = mlp.get_params_copy()

    bad_key = dict(backup)
    bad_key["W9"] = np.zeros((1, 1))
    with pytest.raises(ValueError):
        mlp.set_params(bad_key)

    bad_shape = dict(backup)
    bad_shape["W1"] = np.zeros((1, 1))
    with pytest.raises(ValueError):
        mlp.set_params(bad_shape)

    bad_value = dict(backup)
    bad_value["W1"] = np.full_like(mlp.params["W1"], np.nan)
    with pytest.raises(ValueError):
        mlp.set_params(bad_value)

    for key in mlp.params:  # 校验失败不得部分写入
        np.testing.assert_array_equal(mlp.params[key], before[key])

    mlp.set_params({k: v.astype(np.float32) for k, v in backup.items()})  # float32 可接受
    for key in mlp.params:
        np.testing.assert_allclose(mlp.params[key], backup[key])
