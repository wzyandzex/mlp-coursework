"""优化器测试：SGD 手工算例与校验，以及 P5.2-A 新增的 MomentumSGD/Adam
（数学正确性、输入校验、状态归属、原子性与溢出恢复）。"""

import numpy as np
import pytest

from src.losses import softmax_cross_entropy
from src.mlp import MLP
from src.optimizers import SGD, Adam, MomentumSGD


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


# ---------------------------------------------------------------------------
# P5.2-A 新增：MomentumSGD 与 Adam（依据《P5 实验约定冻结 V0.1》）
# ---------------------------------------------------------------------------


def _momentum_scalar_trajectory(theta0, grads_seq, lr=0.1, mu=0.9):
    """独立标量递推参考：v = mu*v + g；theta -= lr*v（v_0 = 0）。"""
    v = 0.0
    theta = theta0
    out = []
    for g in grads_seq:
        v = mu * v + g
        theta = theta - lr * v
        out.append(theta)
    return out


def _adam_scalar_trajectory(theta0, grads_seq, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
    """独立标量递推参考：Adam 一阶/二阶矩 + t=1 起偏置修正，eps 在 sqrt(v_hat) 之后。"""
    m = v = 0.0
    theta = theta0
    out = []
    for t, g in enumerate(grads_seq, start=1):
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        m_hat = m / (1.0 - b1 ** t)
        v_hat = v / (1.0 - b2 ** t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        out.append(theta)
    return out


# 每个用例给 (优化器工厂, 对应的标量参考递推)
OPTIMIZER_CASES = [
    (lambda: MomentumSGD(0.1), _momentum_scalar_trajectory),
    (lambda: Adam(0.1), _adam_scalar_trajectory),
]


# ---------- 数学正确性 ----------


def test_momentum_first_step_equals_sgd():
    # v_1 = momentum*0 + g_1 = g_1 → 首步与同学习率的 SGD 完全一致
    rng = np.random.default_rng(0)
    grads = {"w": rng.normal(size=(3, 2)), "b": np.array([[0.5, -0.5]])}
    params_sgd = {"w": rng.normal(size=(3, 2)), "b": rng.normal(size=(1, 2))}
    params_mom = {k: v.copy() for k, v in params_sgd.items()}
    SGD(0.1).step(params_sgd, grads)
    MomentumSGD(0.1, momentum=0.9).step(params_mom, grads)
    for key in params_sgd:
        np.testing.assert_array_equal(params_sgd[key], params_mom[key])


def test_momentum_multi_step_hand_computed():
    # 三步手算：v = 1.0, 1.9, 2.71；theta = 0.9, 0.71, 0.439（恒定正梯度 → 持续下降）
    params = {"w": np.array([1.0])}
    opt = MomentumSGD(0.1, momentum=0.9)
    grads_seq = [1.0, 1.0, 1.0]
    for g in grads_seq:
        opt.step(params, {"w": np.array([g])})
    expected = _momentum_scalar_trajectory(1.0, grads_seq)[-1]
    np.testing.assert_allclose(params["w"], [expected], rtol=1e-15, atol=0)
    np.testing.assert_allclose(params["w"], [0.439], rtol=1e-12, atol=0)
    assert params["w"][0] < 1.0  # 更新方向与梯度相反


def test_adam_first_step_hand_computed():
    # t=1：m_hat = g、v_hat = g²，更新 = lr*g/(|g|+eps)；
    # 该公式同时锁死 eps 的位置（若写成 sqrt(v_hat + eps)，相对差异约 2e-8，远超容差）
    params = {"w": np.array([1.0, -2.0])}
    grads = np.array([0.5, -0.25])
    Adam(0.1).step(params, {"w": grads})
    for i, g in enumerate(grads):
        m = 0.1 * g
        v = 0.001 * g * g
        m_hat = m / (1.0 - 0.9)
        v_hat = v / (1.0 - 0.999)
        expected = [1.0, -2.0][i] - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)
        np.testing.assert_allclose(params["w"][i], expected, rtol=1e-15, atol=0)


def test_adam_first_step_bias_correction_property():
    # t=1 偏置修正使首步更新 = lr*g/(|g|+eps)：两个梯度相差 1e5 倍，步幅都 ≈ lr
    params = {"w": np.array([1.0, -1.0])}
    grads = np.array([100.0, 0.001])
    Adam(0.1).step(params, {"w": grads})
    for i, g in enumerate(grads):
        theta0 = [1.0, -1.0][i]
        expected = theta0 - 0.1 * g / (abs(g) + 1e-8)
        np.testing.assert_allclose(params["w"][i], expected, rtol=1e-12, atol=0)
        assert params["w"][i] - theta0 == pytest.approx(-0.1 if g > 0 else 0.1, rel=1e-4)


def test_adam_multi_step_hand_computed():
    params = {"w": np.array([1.0])}
    opt = Adam(0.1)
    grads_seq = [1.0, 0.5, -0.3]
    for g in grads_seq:
        opt.step(params, {"w": np.array([g])})
    expected = _adam_scalar_trajectory(1.0, grads_seq)[-1]
    np.testing.assert_allclose(params["w"], [expected], rtol=1e-12, atol=0)


@pytest.mark.parametrize("make", [lambda: MomentumSGD(0.1), lambda: Adam(0.1)])
def test_stateful_update_in_place_and_opposite_direction(make):
    params = {"w": np.array([1.0, 2.0])}
    grads = {"w": np.array([0.1, 0.2])}
    old = params["w"].copy()
    handle = params["w"]
    make().step(params, grads)
    assert params["w"] is handle  # 原 ndarray 对象身份不变
    assert np.all((params["w"] - old) * grads["w"] < 0.0)  # 朝负梯度方向移动


# ---------- 超参数校验 ----------


def test_momentum_rejects_invalid_hyperparameters():
    for bad_lr in (0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            MomentumSGD(bad_lr)
    for bad_lr in ("0.1", None, True, [0.1]):
        with pytest.raises(TypeError):
            MomentumSGD(bad_lr)
    for bad_m in (-0.1, 1.0, 1.5, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            MomentumSGD(0.1, momentum=bad_m)
    for bad_m in ("0.9", None, True, [0.9]):
        with pytest.raises(TypeError):
            MomentumSGD(0.1, momentum=bad_m)


def test_momentum_zero_is_valid_boundary():
    # 约定允许 momentum = 0（0 <= momentum < 1）：每步退化为普通 SGD
    params = {"w": np.array([1.0])}
    MomentumSGD(0.1, momentum=0.0).step(params, {"w": np.array([0.5])})
    np.testing.assert_allclose(params["w"], [1.0 - 0.05], rtol=1e-15, atol=0)


def test_adam_rejects_invalid_hyperparameters():
    for bad in (0, -1e-3, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Adam(bad)
        with pytest.raises(ValueError):
            Adam(0.1, eps=bad)
    for bad in ("0.1", None, True, [0.1]):
        with pytest.raises(TypeError):
            Adam(bad)
        with pytest.raises(TypeError):
            Adam(0.1, eps=bad)
    for bad_beta in (-0.1, 1.0, 2.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Adam(0.1, beta1=bad_beta)
        with pytest.raises(ValueError):
            Adam(0.1, beta2=bad_beta)
    for bad_beta in ("0.9", None, True, [0.9]):
        with pytest.raises(TypeError):
            Adam(0.1, beta1=bad_beta)
        with pytest.raises(TypeError):
            Adam(0.1, beta2=bad_beta)


def test_adam_zero_beta_is_valid_boundary():
    # beta = 0 是合法下界：beta1=0 → m_t = g_t，m_hat = g；beta2=0 → v_hat = g²
    params = {"w": np.array([1.0])}
    Adam(0.1, beta1=0.0, beta2=0.0).step(params, {"w": np.array([0.5])})
    expected = 1.0 - 0.1 * 0.5 / (0.5 + 1e-8)
    np.testing.assert_allclose(params["w"], [expected], rtol=1e-14, atol=0)


# ---------- 输入校验与失败路径 ----------


@pytest.mark.parametrize("make", [lambda: MomentumSGD(0.1), lambda: Adam(0.1)])
def test_stateful_input_validation_rejects_and_leaves_params_unchanged(make):
    opt = make()
    params = {"w": np.array([1.0, 2.0])}
    snapshot = params["w"].copy()
    with pytest.raises(TypeError):
        opt.step([("w", np.array([1.0]))], {"w": np.array([0.1])})  # params 非字典
    with pytest.raises(TypeError):
        opt.step(params, [("w", 0.1)])  # grads 非字典
    with pytest.raises(ValueError, match="key"):
        opt.step(params, {"w": np.array([0.1]), "x": np.array([1.0])})  # 键不匹配
    with pytest.raises(TypeError):
        opt.step({"w": np.array([1, 2])}, {"w": np.array([0.1, 0.2])})  # 参数须浮点
    with pytest.raises(TypeError):
        opt.step(params, {"w": [0.1, 0.2]})  # 梯度非 ndarray
    with pytest.raises(ValueError, match="shape"):
        opt.step(params, {"w": np.array([0.1])})  # 形状不匹配
    with pytest.raises(ValueError, match="non-finite"):
        opt.step(params, {"w": np.array([np.nan, 0.0])})  # 非有限梯度
    with pytest.raises(ValueError, match="read-only"):
        params["w"].setflags(write=False)  # 只读参数数组
        try:
            opt.step(params, {"w": np.array([0.1, 0.2])})
        finally:
            params["w"].setflags(write=True)
    np.testing.assert_array_equal(params["w"], snapshot)  # 所有失败路径参数逐元素不变


@pytest.mark.parametrize("make", [lambda: MomentumSGD(0.1), lambda: Adam(0.1)])
def test_first_call_failure_does_not_bind_or_leave_state(make):
    # 首次调用失败：不绑定字典、不初始化状态、不推进时间步——
    # 之后的有效调用必须与全新实例的手算结果完全一致
    opt = make()
    params = {"w": np.array([1.0, 2.0])}
    with pytest.raises(ValueError):
        opt.step(params, {"w": np.array([np.nan, 1.0])})
    np.testing.assert_array_equal(params["w"], [1.0, 2.0])  # 失败零写入
    valid_grads = {"w": np.array([0.3, -0.2])}
    params_fresh = {"w": params["w"].copy()}
    opt.step(params, valid_grads)
    fresh = make()
    fresh.step(params_fresh, valid_grads)
    np.testing.assert_array_equal(params["w"], params_fresh["w"])


# ---------- 状态隔离与归属契约 ----------


@pytest.mark.parametrize("make,trajectory", OPTIMIZER_CASES)
def test_two_instances_are_isolated(make, trajectory):
    grads_seq = [1.0, 0.5, -0.3, 0.2]
    expected = trajectory(1.0, grads_seq)
    params_a, params_b = {"w": np.array([1.0])}, {"w": np.array([1.0])}
    opt_a, opt_b = make(), make()
    for i, g in enumerate(grads_seq):
        opt_a.step(params_a, {"w": np.array([g])})
        np.testing.assert_allclose(params_a["w"], [expected[i]], rtol=1e-12, atol=0)
    opt_b.step(params_b, {"w": np.array([grads_seq[0]])})  # 实例 B 只走一步
    np.testing.assert_allclose(params_b["w"], [expected[0]], rtol=1e-12, atol=0)
    assert not np.allclose(params_a["w"], params_b["w"])  # 两个实例状态互不影响


@pytest.mark.parametrize("make,trajectory", OPTIMIZER_CASES)
def test_misuse_with_different_dict_rejected_and_state_preserved(make, trajectory):
    opt = make()
    grads_seq = [1.0, 1.0]
    expected = trajectory(1.0, grads_seq)
    params_a = {"w": np.array([1.0])}
    opt.step(params_a, {"w": np.array([grads_seq[0]])})  # 绑定 A
    params_b = {"w": np.array([5.0])}
    with pytest.raises(ValueError, match="bound"):
        opt.step(params_b, {"w": np.array([1.0])})  # 误用于另一个字典
    np.testing.assert_array_equal(params_a["w"], [expected[0]])  # A 未被第二次更新
    np.testing.assert_array_equal(params_b["w"], [5.0])  # B 也未被修改
    opt.step(params_a, {"w": np.array([grads_seq[1]])})  # 从失败前状态继续
    np.testing.assert_allclose(params_a["w"], [expected[1]], rtol=1e-12, atol=0)


@pytest.mark.parametrize("make,trajectory", OPTIMIZER_CASES)
def test_replaced_param_array_rejected_and_state_preserved(make, trajectory):
    opt = make()
    grads_seq = [1.0, 1.0]
    expected = trajectory(1.0, grads_seq)
    params = {"w": np.array([1.0])}
    opt.step(params, {"w": np.array([grads_seq[0]])})  # 绑定原数组
    original = params["w"]
    params["w"] = np.array([original[0]])  # 同形状/dtype/值的新数组对象
    with pytest.raises(ValueError, match="replaced"):
        opt.step(params, {"w": np.array([1.0])})
    np.testing.assert_array_equal(original, [expected[0]])  # 原数组未被触碰
    np.testing.assert_array_equal(params["w"], [expected[0]])  # 新数组也未被修改
    params["w"] = original  # 还原后可继续，且与手算轨迹一致
    opt.step(params, {"w": np.array([grads_seq[1]])})
    np.testing.assert_allclose(params["w"], [expected[1]], rtol=1e-12, atol=0)


@pytest.mark.parametrize("make", [lambda: MomentumSGD(0.1), lambda: Adam(0.1)])
def test_dtype_and_shape_contract_change_rejected(make):
    opt = make()
    params = {"w": np.array([1.0])}
    opt.step(params, {"w": np.array([1.0])})
    original = params["w"]
    params["w"] = original.astype(np.float32)  # dtype 契约变化
    with pytest.raises(ValueError):
        opt.step(params, {"w": np.array([1.0], dtype=np.float32)})
    params["w"] = original
    params["w"] = np.zeros((2, 2))  # 形状契约变化
    with pytest.raises(ValueError):
        opt.step(params, {"w": np.zeros((2, 2))})


# ---------- 原子性：候选溢出时参数与状态零写入，且可从失败前状态恢复 ----------


def test_momentum_velocity_overflow_is_atomic_and_resumable():
    opt = MomentumSGD(1e-3, momentum=0.9)
    params = {"w": np.array([1.0])}
    # 第一步：v_1 = 1e308（有限），参数候选 1 - 1e-3*1e308 = 1 - 1e305（有限）→ 成功
    opt.step(params, {"w": np.array([1e308])})
    after_first = 1.0 - 1e-3 * 1e308
    np.testing.assert_allclose(params["w"], [after_first], rtol=1e-15, atol=0)
    # 第二步：v_2 = 0.9*1e308 + 1e308 = 1.9e308 → 速度候选溢出 → 整体拒绝
    with pytest.raises(ValueError, match="non-finite"):
        opt.step(params, {"w": np.array([1e308])})
    np.testing.assert_array_equal(params["w"], [after_first])  # 参数逐元素不变
    # 第三步：小梯度 → 必须与“从失败前状态(v=1e308)继续”的独立手算一致
    opt.step(params, {"w": np.array([1.0])})
    v_before_failure = 1e308
    v_next = 0.9 * v_before_failure + 1.0
    expected = after_first - 1e-3 * v_next
    np.testing.assert_allclose(params["w"], [expected], rtol=1e-12, atol=0)


def test_momentum_param_candidate_overflow_rejected():
    # 候选参数溢出：v 有限（1e308）但 lr=2.0 使参数候选 1 - 2e308 = -Inf → 拒绝
    opt = MomentumSGD(2.0, momentum=0.9)
    params = {"w": np.array([1.0])}
    with pytest.raises(ValueError, match="non-finite"):
        opt.step(params, {"w": np.array([1e308])})
    np.testing.assert_array_equal(params["w"], [1.0])


def test_adam_second_moment_overflow_is_atomic_and_fresh_after_failure():
    opt = Adam(0.1)
    params = {"w": np.array([1.0])}
    # g = 1e200 → g² = 1e400 溢出 → 二阶矩候选非有限 → 首次调用整体失败
    with pytest.raises(ValueError, match="non-finite"):
        opt.step(params, {"w": np.array([1e200])})
    np.testing.assert_array_equal(params["w"], [1.0])  # 参数不变
    # 失败后首次有效调用必须像全新优化器一样从 t=1 开始（无半初始化状态）
    opt.step(params, {"w": np.array([0.5])})
    m = 0.1 * 0.5
    v = 0.001 * 0.25
    expected = 1.0 - 0.1 * (m / (1.0 - 0.9)) / (np.sqrt(v / (1.0 - 0.999)) + 1e-8)
    np.testing.assert_allclose(params["w"], [expected], rtol=1e-14, atol=0)


@pytest.mark.parametrize("make", [lambda: MomentumSGD(0.1), lambda: Adam(0.1)])
def test_stateful_optimizers_do_not_modify_grads(make):
    opt = make()
    grads = {"w": np.array([0.3, -0.4]), "b": np.array([[1.0]])}
    params = {"w": np.array([1.0, 2.0]), "b": np.array([[0.0]])}
    grads_snapshot = {key: value.copy() for key, value in grads.items()}
    opt.step(params, grads)
    opt.step(params, grads)  # 多步后仍不变
    for key in grads:
        np.testing.assert_array_equal(grads[key], grads_snapshot[key])


# ---------------------------------------------------------------------------
# P5.2-A-R1 整改：目标 dtype 转换后的候选值有限性校验
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make", [lambda: MomentumSGD(1e40), lambda: Adam(1e40)])
def test_dtype_cast_overflow_rejected_and_params_untouched(make):
    # float32 参数 + float64 梯度 + lr=1e40：候选在 float64 下为 -1e40（有限），
    # 写回 float32 会静默变成 -Inf。修复后必须在提交前拒绝。
    params = {"w": np.array([0.0], dtype=np.float32)}
    handle = params["w"]
    with pytest.raises(ValueError, match="casting the candidate"):
        make().step(params, {"w": np.array([1.0], dtype=np.float64)})
    np.testing.assert_array_equal(params["w"], np.array([0.0], dtype=np.float32))
    assert params["w"] is handle  # 对象身份不变


@pytest.mark.parametrize("make", [lambda: MomentumSGD(1e40), lambda: Adam(1e40)])
def test_state_not_advanced_after_cast_failure(make):
    # 零梯度二次调用是行为学区分器：若失败调用曾偷偷提交状态
    # （Momentum 速度=1.0 / Adam 矩 m=0.1、v=0.001、t=1），
    # 本调用候选仍会是 ±9e39 量级并再次溢出被拒；只有状态未推进才会成功。
    opt = make()
    params = {"w": np.array([0.0], dtype=np.float32)}
    handle = params["w"]
    with pytest.raises(ValueError):
        opt.step(params, {"w": np.array([1.0], dtype=np.float64)})
    opt.step(params, {"w": np.array([0.0], dtype=np.float64)})  # 必须成功
    np.testing.assert_array_equal(params["w"], np.array([0.0], dtype=np.float32))
    assert params["w"] is handle
    assert params["w"].dtype == np.float32
    assert np.all(np.isfinite(params["w"]))


@pytest.mark.parametrize(
    "make, expected",
    [
        # Momentum: v_1 = 0.5 → theta = 1 - 0.05
        (lambda: MomentumSGD(0.1), 1.0 - 0.1 * 0.5),
        # Adam t=1: m_hat = 0.5, v_hat = 0.25 → theta = 1 - 0.1*0.5/(0.5+1e-8)
        (lambda: Adam(0.1), 1.0 - 0.1 * 0.5 / (np.sqrt(0.25 * 0.001 / 0.001) + 1e-8)),
    ],
)
def test_representable_float32_update_succeeds(make, expected):
    # 可被 float32 表示的正常更新不得被转换检查误拒，且 dtype 保持 float32
    params = {"w": np.array([1.0], dtype=np.float32)}
    handle = params["w"]
    make().step(params, {"w": np.array([0.5], dtype=np.float64)})
    assert params["w"] is handle
    assert params["w"].dtype == np.float32
    assert np.all(np.isfinite(params["w"]))
    np.testing.assert_allclose(params["w"], [expected], rtol=1e-6, atol=0)


@pytest.mark.parametrize("make", [lambda: MomentumSGD(1e40), lambda: Adam(1e40)])
def test_multi_param_atomicity_on_cast_overflow(make):
    # "safe" 的候选可被 float32 表示（≈0 或 ≈-1e8），"over" 的候选溢出（≈-1e40）
    # → 必须整体拒绝，两个参数都零写入
    params = {
        "safe": np.array([1.0], dtype=np.float32),
        "over": np.array([0.0], dtype=np.float32),
    }
    handles = {key: params[key] for key in params}
    grads = {
        "safe": np.array([1e-40], dtype=np.float64),
        "over": np.array([1.0], dtype=np.float64),
    }
    with pytest.raises(ValueError, match="casting the candidate"):
        make().step(params, grads)
    np.testing.assert_array_equal(params["safe"], np.array([1.0], dtype=np.float32))
    np.testing.assert_array_equal(params["over"], np.array([0.0], dtype=np.float32))
    assert params["safe"] is handles["safe"]
    assert params["over"] is handles["over"]


# ---------------------------------------------------------------------------
# P5.2-A-R2 整改：SGD 的目标 dtype 转换安全
# ---------------------------------------------------------------------------


def test_sgd_dtype_cast_overflow_rejected_atomically():
    # float32 参数 + float64 梯度 + lr=1e40：
    # "a_safe"（梯度 1e-40）候选 ≈ 0，可被 float32 表示，且排序在前；
    # "b_over"（梯度 1.0）候选 ≈ -1e40，float64 下有限、写回 float32 会成为 -Inf。
    # 旧实现会先把 a_safe 部分更新、再把 b_over 静默写成 -Inf；修复后必须整体拒绝。
    params = {
        "a_safe": np.array([1.0], dtype=np.float32),
        "b_over": np.array([0.0], dtype=np.float32),
    }
    handles = {key: params[key] for key in params}
    original_dtypes = {key: params[key].dtype for key in params}
    grads = {
        "a_safe": np.array([1e-40], dtype=np.float64),
        "b_over": np.array([1.0], dtype=np.float64),
    }
    with pytest.raises(ValueError, match="casting the candidate"):
        SGD(1e40).step(params, grads)
    # 所有参数逐元素与调用前完全相同
    np.testing.assert_array_equal(params["a_safe"], np.array([1.0], dtype=np.float32))
    np.testing.assert_array_equal(params["b_over"], np.array([0.0], dtype=np.float32))
    # 对象身份与 dtype 保持不变
    assert params["a_safe"] is handles["a_safe"]
    assert params["b_over"] is handles["b_over"]
    assert params["a_safe"].dtype == original_dtypes["a_safe"]
    assert params["b_over"].dtype == original_dtypes["b_over"]


def test_sgd_representable_float32_update_succeeds():
    # 可被 float32 表示的正常更新不得被转换检查误拒
    params = {"w": np.array([1.0], dtype=np.float32)}
    handle = params["w"]
    SGD(0.1).step(params, {"w": np.array([0.5], dtype=np.float64)})
    assert params["w"] is handle  # 对象身份不变
    assert params["w"].dtype == np.float32  # dtype 保持
    assert np.all(np.isfinite(params["w"]))  # 全部有限
    np.testing.assert_allclose(params["w"], [1.0 - 0.1 * 0.5], rtol=1e-6, atol=0)
