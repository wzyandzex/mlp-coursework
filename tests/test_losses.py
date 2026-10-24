"""Softmax 与 Softmax 交叉熵的单元测试（含手算数值核验与数值梯度）。"""

import math
import warnings

import numpy as np
import pytest

from src.losses import softmax, softmax_cross_entropy
from tests.grad_check import numerical_gradients, relative_error


def _reference_ce(logits_rows, labels) -> float:
    """用纯 Python math 独立实现批次平均交叉熵（小数值范围，无溢出风险）。

    loss_i = max_j z_ij + log Σ_j exp(z_ij - max_j z_ij) - z_{i, y_i}
    """
    total = 0.0
    for row, label in zip(logits_rows, labels):
        row_max = max(row)
        sum_exp = sum(math.exp(v - row_max) for v in row)
        total += row_max + math.log(sum_exp) - row[label]
    return total / len(labels)


# ---------- Softmax ----------


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(11)
    p = softmax(rng.normal(size=(6, 7)) * 3.0)
    assert p.shape == (6, 7)
    np.testing.assert_allclose(p.sum(axis=1), np.ones(6), atol=1e-12)
    assert np.all(p >= 0.0)
    assert np.all(np.isfinite(p))


def test_softmax_shift_invariance():
    rng = np.random.default_rng(13)
    z = rng.normal(size=(4, 5))
    for c in (2.5, 123.0, 1e4):
        np.testing.assert_allclose(softmax(z + c), softmax(z), atol=1e-9)


def test_softmax_extreme_logits_finite():
    z = np.array(
        [
            [1000.0, 0.0, -1000.0],
            [-1000.0, -1001.0, -1002.0],
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = softmax(z)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(axis=1), [1.0, 1.0], atol=1e-12)
    assert p[0].argmax() == 0
    assert p[1].argmax() == 0


def test_softmax_rejects_bad_inputs():
    ok = np.random.default_rng(17).normal(size=(3, 4))
    with pytest.raises(ValueError):
        softmax(ok[0])  # 一维
    with pytest.raises(ValueError):
        softmax(ok[np.newaxis])  # 三维
    with pytest.raises(ValueError):
        softmax(np.empty((0, 4)))  # 空批次
    with pytest.raises(ValueError):
        softmax(np.empty((3, 0)))  # 空类别
    with pytest.raises(ValueError):
        softmax(np.array([[np.nan, 1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        softmax(np.array([[np.inf, 1.0], [0.0, 1.0]]))
    with pytest.raises(TypeError):
        softmax(np.array([["a", "b"], ["c", "d"]]))
    with pytest.raises(TypeError):
        softmax(np.array([[1 + 0j, 0.0], [0.0, 1.0]]))


# ---------- Softmax + 交叉熵：数值 ----------


def test_cross_entropy_hand_computed_values():
    # 单样本 [1, 0]，y = 0：损失 = log(1 + e^-1)
    loss, dlogits = softmax_cross_entropy(np.array([[1.0, 0.0]]), np.array([0]))
    np.testing.assert_allclose(loss, math.log(1.0 + math.exp(-1.0)), atol=1e-12)
    # 均匀两类的单样本：损失 = ln 2
    loss2, _ = softmax_cross_entropy(np.array([[0.0, 0.0]]), np.array([1]))
    np.testing.assert_allclose(loss2, math.log(2.0), atol=1e-12)
    # dlogits 手算：(p - onehot) / m，其中 p = [1/(1+e^-1), e^-1/(1+e^-1)]
    p0 = 1.0 / (1.0 + math.exp(-1.0))
    np.testing.assert_allclose(dlogits, [[p0 - 1.0, 1.0 - p0]])


def test_cross_entropy_matches_independent_reference():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(6, 4))
    y = rng.integers(0, 4, size=6)
    loss, dlogits = softmax_cross_entropy(logits, y)
    expected = _reference_ce(logits.tolist(), y.tolist())
    np.testing.assert_allclose(loss, expected, atol=1e-12)
    assert isinstance(loss, float)
    assert dlogits.shape == logits.shape


def test_cross_entropy_is_batch_mean():
    logits = np.array([[1.0, 0.0], [0.0, 2.0]])
    y = np.array([0, 1])
    loss, _ = softmax_cross_entropy(logits, y)
    l0 = softmax_cross_entropy(logits[:1], y[:1])[0]
    l1 = softmax_cross_entropy(logits[1:], y[1:])[0]
    np.testing.assert_allclose(loss, (l0 + l1) / 2.0, atol=1e-12)


def test_cross_entropy_extreme_logits_finite():
    logits = np.array([[1000.0, -1000.0], [0.0, 999.0]])
    y = np.array([0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loss, dlogits = softmax_cross_entropy(logits, y)
    assert np.isfinite(loss)
    assert np.all(np.isfinite(dlogits))


def test_integer_logits_accepted_and_converted():
    loss, dlogits = softmax_cross_entropy(np.array([[2, 0], [0, 1]]), np.array([0, 1]))
    assert np.isfinite(loss)
    assert dlogits.dtype == np.float64
    assert np.all(np.isfinite(dlogits))


# ---------- dlogits 梯度 ----------


def test_dlogits_rows_sum_to_zero():
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(8, 5))
    y = rng.integers(0, 5, size=8)
    _, dlogits = softmax_cross_entropy(logits, y)
    np.testing.assert_allclose(dlogits.sum(axis=1), np.zeros(8), atol=1e-12)


def test_dlogits_matches_numerical_gradient():
    rng = np.random.default_rng(7)
    logits = rng.normal(scale=2.0, size=(3, 4))
    y = np.array([0, 2, 3])

    def loss_fn():
        return softmax_cross_entropy(logits, y)[0]

    numeric = numerical_gradients(loss_fn, {"logits": logits})
    _, analytic = softmax_cross_entropy(logits, y)
    error = relative_error(numeric["logits"], analytic)
    assert error < 1e-6, f"relative error {error:.3e} >= 1e-6"


# ---------- 标签校验 ----------


@pytest.fixture()
def small_logits():
    return np.array([[1.0, 0.2, -0.5], [0.0, 2.0, 0.3], [-1.0, 0.5, 0.8]])


def test_cross_entropy_label_validation(small_logits):
    with pytest.raises(ValueError):
        softmax_cross_entropy(small_logits, np.array([0, 1, 3]))  # 越界
    with pytest.raises(ValueError):
        softmax_cross_entropy(small_logits, np.array([0, 1, -1]))  # 负标签
    with pytest.raises(ValueError):
        softmax_cross_entropy(small_logits, np.array([0, 1]))  # 长度不符
    with pytest.raises(ValueError):
        softmax_cross_entropy(small_logits, np.array([[0], [1], [2]]))  # 二维
    with pytest.raises(ValueError):
        softmax_cross_entropy(small_logits, np.array([0.0, 1.5, 2.0]))  # 非整数浮点
    with pytest.raises(ValueError):
        softmax_cross_entropy(small_logits, np.array([0.0, np.nan, 2.0]))  # NaN
    with pytest.raises(TypeError):
        softmax_cross_entropy(small_logits, np.array(["0", "1", "2"]))
    with pytest.raises(TypeError):
        softmax_cross_entropy(small_logits, np.array([True, False, True]))  # bool 拒绝
    # 整数值的浮点标签可接受（不发生截断），结果与整数标签一致
    loss_int = softmax_cross_entropy(small_logits, np.array([0, 1, 2]))[0]
    loss_float = softmax_cross_entropy(small_logits, np.array([0.0, 1.0, 2.0]))[0]
    assert loss_int == loss_float


def test_cross_entropy_rejects_bad_logits():
    with pytest.raises(ValueError):
        softmax_cross_entropy(np.empty((0, 2)), np.array([]))  # 空批次
    with pytest.raises(ValueError):
        softmax_cross_entropy(np.array([[np.nan, 1.0]]), np.array([0]))


def test_integer_list_labels_accepted():
    # 标签传入 int 列表（可转换为整数数组）应被接受
    loss, _ = softmax_cross_entropy(np.array([[1.0, 0.0], [0.0, 1.0]]), [0, 1])
    assert np.isfinite(loss)
