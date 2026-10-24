"""小型端到端训练冒烟测试：NumPy 合成三分类数据，损失必须明显下降且可复现。

不使用 scikit-learn 或任何机器学习工具，不涉及 MNIST。
"""

import numpy as np
import pytest

from src.losses import softmax_cross_entropy
from src.mlp import MLP
from src.optimizers import SGD


def _make_three_clusters(n_per_class: int = 60, noise: float = 0.6, seed: int = 0):
    """用 NumPy 生成三个可分离的二维高斯簇。"""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [3.5, 3.0], [-3.5, 3.0]])
    y = np.tile(np.arange(3), n_per_class)
    X = centers[y] + rng.normal(0.0, noise, size=(3 * n_per_class, 2))
    return X, y


def _train_full_batch(mlp, X, y, learning_rate: float, epochs: int):
    """基础训练循环：forward → softmax+CE → backward → SGD step。"""
    optimizer = SGD(learning_rate)
    losses = []
    for _ in range(epochs):
        logits, cache = mlp.forward(X, return_cache=True)
        loss, dlogits = softmax_cross_entropy(logits, y)
        grads = mlp.backward(dlogits, cache)
        optimizer.step(mlp.params, grads)
        losses.append(loss)
    return losses


def _accuracy(mlp, X, y) -> float:
    return float(np.mean(mlp.predict(X) == y))


@pytest.mark.parametrize(
    "hidden_dims, activation, learning_rate",
    [
        ([16], "relu", 1.0),
        ([16, 8], "tanh", 1.0),
    ],
)
def test_training_reduces_loss_and_reaches_high_accuracy(hidden_dims, activation, learning_rate):
    X, y = _make_three_clusters(seed=0)
    mlp = MLP(
        input_dim=2,
        hidden_dims=hidden_dims,
        output_dim=3,
        hidden_activation=activation,
        seed=42,
    )
    losses = _train_full_batch(mlp, X, y, learning_rate, epochs=300)

    assert all(np.isfinite(losses))        # 训练过程中损失保持有限
    assert losses[0] > 0.9                 # 初始损失接近 ln 3 ≈ 1.099（三分类随机水平）
    assert losses[-1] < 0.1 * losses[0]    # 损失相对初始值下降一个数量级
    assert _accuracy(mlp, X, y) >= 0.95    # 可分簇上训练准确率应很高


def test_training_reproducible_with_same_seed():
    X, y = _make_three_clusters(seed=0)

    def run():
        mlp = MLP(2, [16], 3, hidden_activation="relu", seed=42)
        return _train_full_batch(mlp, X, y, 1.0, 50)

    losses_a = run()
    losses_b = run()
    np.testing.assert_allclose(losses_a, losses_b, rtol=1e-13, atol=1e-13)
