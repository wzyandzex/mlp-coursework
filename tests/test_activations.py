"""激活函数及其导数的单元测试。"""

import warnings

import numpy as np
import pytest

from src.activations import (
    SUPPORTED_ACTIVATIONS,
    activate,
    activation_gradient,
    check_activation_name,
    relu,
    relu_grad,
    sigmoid,
    sigmoid_grad,
    tanh,
    tanh_grad,
)


def test_relu_forward_values():
    z = np.array([[-2.0, -0.5, 0.0], [0.5, 2.0, 3.0]])
    expected = np.array([[0.0, 0.0, 0.0], [0.5, 2.0, 3.0]])
    out = relu(z)
    np.testing.assert_allclose(out, expected)
    assert out.shape == z.shape


def test_relu_grad_convention_at_zero():
    # 任务约定：z = 0 处导数取 0
    z = np.array([-1.0, -1e-9, 0.0, 1e-9, 1.0])
    np.testing.assert_allclose(relu_grad(z), [0.0, 0.0, 0.0, 1.0, 1.0])


def test_sigmoid_basic_properties():
    z = np.linspace(-4.0, 4.0, 33)
    a = sigmoid(z)
    assert np.all((a > 0.0) & (a < 1.0))
    np.testing.assert_allclose(sigmoid(0.0), 0.5)
    np.testing.assert_allclose(a + sigmoid(-z), 1.0, atol=1e-12)  # f(-z) = 1 - f(z)
    assert np.all(np.diff(a) > 0.0)  # 单调递增


def test_sigmoid_extreme_inputs_stable():
    z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 任何 RuntimeWarning（如溢出）都视为失败
        a = sigmoid(z)
    assert np.all(np.isfinite(a))
    # 极大/极小输入趋于 0/1（-50 处约为 1.9e-22，允许绝对容差）
    np.testing.assert_allclose(a, [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-15)


def test_sigmoid_grad_matches_formula_and_finite_diff():
    z = np.linspace(-3.0, 3.0, 13)
    a = sigmoid(z)
    np.testing.assert_allclose(sigmoid_grad(z), a * (1.0 - a))
    np.testing.assert_allclose(sigmoid_grad(z, a), a * (1.0 - a))  # 传入已算好的 a
    eps = 1e-6
    numeric = (sigmoid(z + eps) - sigmoid(z - eps)) / (2.0 * eps)
    np.testing.assert_allclose(sigmoid_grad(z), numeric, atol=1e-9)
    np.testing.assert_allclose(sigmoid_grad(np.array([0.0])), [0.25])


def test_tanh_properties_and_grad():
    z = np.linspace(-3.0, 3.0, 25)
    a = tanh(z)
    np.testing.assert_allclose(tanh(0.0), 0.0, atol=1e-15)
    assert np.all((a > -1.0) & (a < 1.0))
    np.testing.assert_allclose(tanh(-z), -a, atol=1e-15)  # 奇函数
    np.testing.assert_allclose(tanh_grad(z), 1.0 - a**2)
    eps = 1e-6
    np.testing.assert_allclose(
        tanh_grad(z), (tanh(z + eps) - tanh(z - eps)) / (2.0 * eps), atol=1e-9
    )
    extreme = tanh(np.array([-800.0, 800.0]))
    assert np.all(np.isfinite(extreme))
    np.testing.assert_allclose(extreme, [-1.0, 1.0])


def test_activate_resolver_and_case_insensitivity():
    z = np.linspace(-1.0, 1.0, 7)
    np.testing.assert_allclose(activate("relu", z), relu(z))
    np.testing.assert_allclose(activate("sigmoid", z), sigmoid(z))
    np.testing.assert_allclose(activate("tanh", z), tanh(z))
    np.testing.assert_allclose(activate("ReLU", z), relu(z))
    with pytest.raises(ValueError, match="unsupported activation"):
        activate("gelu", z)


def test_gradient_resolver():
    z = np.linspace(-2.0, 2.0, 9)
    a = tanh(z)
    np.testing.assert_allclose(activation_gradient("tanh", z, a), tanh_grad(z))
    np.testing.assert_allclose(activation_gradient("tanh", z), tanh_grad(z))
    np.testing.assert_allclose(activation_gradient("relu", z), relu_grad(z))
    np.testing.assert_allclose(activation_gradient("sigmoid", z), sigmoid_grad(z))
    with pytest.raises(ValueError, match="unsupported activation"):
        activation_gradient("sin", z)


def test_supported_activation_list():
    assert set(SUPPORTED_ACTIVATIONS) == {"relu", "sigmoid", "tanh"}
    for name in SUPPORTED_ACTIVATIONS:
        assert check_activation_name(name.upper()) == name


def test_invalid_inputs_rejected():
    with pytest.raises(TypeError):
        relu("not an array")
    with pytest.raises(TypeError):
        sigmoid(np.array(["a", "b"]))
    with pytest.raises(TypeError):
        tanh(np.array([1 + 2j]))
    for fn in (relu, sigmoid, tanh, relu_grad):
        with pytest.raises(ValueError, match="non-finite"):
            fn(np.array([np.nan, 1.0]))
        with pytest.raises(ValueError, match="non-finite"):
            fn(np.array([np.inf]))


def test_input_array_not_modified():
    z = np.array([-1.0, 0.0, 2.0])
    original = z.copy()
    for fn in (relu, sigmoid, tanh, relu_grad, sigmoid_grad, tanh_grad):
        fn(z)
    np.testing.assert_array_equal(z, original)


def test_output_is_float64_and_shape_preserved():
    z = np.array([[1, -2], [3, 0]], dtype=np.int32)
    for fn in (relu, sigmoid, tanh):
        out = fn(z)
        assert out.dtype == np.float64
        assert out.shape == z.shape
