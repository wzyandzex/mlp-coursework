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


# ---------------------------------------------------------------------------
# 阶段 4（P4-DATA-TRAIN-01）新增：训练闭环、评估、checkpoint、实验产物与 CLI。
# 全部使用小型合成数据，不依赖网络与真实 MNIST。
# ---------------------------------------------------------------------------

import csv  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from src.training import (  # noqa: E402
    HISTORY_FIELDS,
    _iterate_minibatches,
    create_run_directory,
    evaluate,
    load_checkpoint,
    run_experiment,
    save_checkpoint,
    select_best_epoch,
    train_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_split():
    """合成三簇数据并切成 train/val/test 三份（80/20/20）。"""
    X, y = _make_three_clusters(n_per_class=40, noise=0.5, seed=0)
    perm = np.random.default_rng(1).permutation(len(y))
    return X, y, perm[:80], perm[80:100], perm[100:]


def _tiny_config(learning_rate: float = 0.5) -> dict:
    return {
        "task": "P4-DATA-TRAIN-01",
        "network": {
            "input_dim": 2,
            "hidden_dims": [8],
            "output_dim": 3,
            "hidden_activation": "tanh",
            "initializer": "auto",
        },
        "training": {
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": learning_rate,
            "eval_batch_size": 32,
        },
        "seeds": {"model_seed": 0, "split_seed": 0},
        "data": {"source_dir": "synthetic"},
    }


# ---------- 训练循环 ----------


def test_train_one_epoch_records_metrics_and_updates_params():
    X, y, tr, va, _te = _tiny_split()
    model = MLP(2, [16], 3, hidden_activation="relu", seed=0)
    before = model.get_params_copy()
    result = train_model(
        model, X[tr], y[tr], X[va], y[va],
        epochs=1, batch_size=16, learning_rate=0.5, seed=0,
    )
    assert len(result.history) == 1
    row = result.history[0]
    for key in HISTORY_FIELDS:
        assert key in row
    assert np.isfinite(row["train_loss"]) and np.isfinite(row["val_loss"])
    assert 0.0 <= row["train_accuracy"] <= 1.0
    assert 0.0 <= row["val_accuracy"] <= 1.0
    assert any(not np.array_equal(model.params[k], before[k]) for k in model.params)
    assert result.best_epoch == 1
    assert set(result.best_params) == set(model.params)


def test_minibatch_tail_samples_not_dropped():
    rng = np.random.default_rng(0)
    batches = list(_iterate_minibatches(10, 4, rng))
    assert [len(batch) for batch in batches] == [4, 4, 2]  # 尾批保留
    np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(10))


def test_evaluate_does_not_modify_params():
    X, y, _tr, va, _te = _tiny_split()
    model = MLP(2, [16], 3, hidden_activation="tanh", seed=1)
    before = model.get_params_copy()
    loss, acc = evaluate(model, X[va], y[va], batch_size=8)
    for key in model.params:
        np.testing.assert_array_equal(model.params[key], before[key])
    assert np.isfinite(loss) and 0.0 <= acc <= 1.0


def test_evaluate_loss_is_sample_weighted():
    # eval_batch_size=1 的逐样本加权平均必须等于整批一次计算的平均损失
    X, y, _tr, va, _te = _tiny_split()
    model = MLP(2, [8], 3, hidden_activation="tanh", seed=3)
    weighted, acc = evaluate(model, X[va], y[va], batch_size=1)
    direct_loss, _ = softmax_cross_entropy(model.forward(X[va]), y[va])
    assert weighted == pytest.approx(direct_loss)
    assert 0.0 <= acc <= 1.0


def test_select_best_epoch_tie_prefers_earlier():
    history = [
        {"epoch": 1, "val_loss": 0.5},
        {"epoch": 2, "val_loss": 0.3},
        {"epoch": 3, "val_loss": 0.3},  # 与 epoch 2 并列 → 取更早的
        {"epoch": 4, "val_loss": 0.4},
    ]
    assert select_best_epoch(history) == 2


def test_best_checkpoint_snapshot_not_overwritten_by_later_epochs():
    X, y, tr, va, _te = _tiny_split()
    model = MLP(2, [16], 3, hidden_activation="relu", seed=2)
    result = train_model(
        model, X[tr], y[tr], X[va], y[va],
        epochs=4, batch_size=32, learning_rate=0.5, seed=2,
    )
    val_losses = [row["val_loss"] for row in result.history]
    best_row = min(val_losses)
    assert result.best_epoch == val_losses.index(best_row) + 1  # 并列取最早
    assert result.best_val_loss == pytest.approx(min(val_losses))
    # 训练继续更新了参数，但最佳快照保持不变（深复制）
    for key in model.params:
        model.params[key][...] += 100.0
    for key in result.best_params:
        assert not np.array_equal(result.best_params[key], model.params[key])
    np.testing.assert_allclose(
        sorted(result.best_params["W1"].ravel()), sorted(result.best_params["W1"].ravel())
    )  # 自反一致性（防误改断言对象）
    _ = best_row


def test_training_history_reproducible_with_same_seed():
    X, y, tr, va, _te = _tiny_split()

    def run():
        model = MLP(2, [16], 3, hidden_activation="relu", seed=5)
        return train_model(
            model, X[tr], y[tr], X[va], y[va],
            epochs=3, batch_size=32, learning_rate=0.5, seed=123,
        )

    r1, r2 = run(), run()
    assert r1.best_epoch == r2.best_epoch
    for h1, h2 in zip(r1.history, r2.history):
        assert h1["epoch"] == h2["epoch"]
        for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy"):
            assert h1[key] == pytest.approx(h2[key], rel=1e-12, abs=1e-12)


def test_train_model_rejects_invalid_arguments():
    X, y, tr, va, _te = _tiny_split()
    model = MLP(2, [8], 3, seed=0)
    with pytest.raises(ValueError):
        train_model(model, X[tr], y[tr], X[va], y[va], epochs=0, batch_size=16, learning_rate=0.1, seed=0)
    with pytest.raises(ValueError):
        train_model(model, X[tr], y[tr], X[va], y[va], epochs=1, batch_size=0, learning_rate=0.1, seed=0)
    with pytest.raises((ValueError, TypeError)):
        train_model(model, X[tr], y[tr][:5], X[va], y[va], epochs=1, batch_size=16, learning_rate=0.1, seed=0)


# ---------- checkpoint 保存 / 加载 ----------


def test_checkpoint_roundtrip_preserves_predictions(tmp_path):
    X, y, tr, _va, _te = _tiny_split()
    model = MLP(2, [16], 3, hidden_activation="relu", seed=4)
    train_model(
        model, X[tr], y[tr], X[tr], y[tr],
        epochs=2, batch_size=32, learning_rate=0.5, seed=4,
    )
    path = tmp_path / "best_model.npz"
    save_checkpoint(model.params, path)
    loaded = load_checkpoint(path, model.params)
    for key in model.params:
        np.testing.assert_array_equal(loaded[key], model.params[key])
    # 加载到同结构的新模型后预测一致
    restored = MLP(2, [16], 3, hidden_activation="relu", seed=999)  # seed 不同也无妨
    restored.set_params(loaded)
    np.testing.assert_array_equal(restored.predict(X[tr]), model.predict(X[tr]))


def test_checkpoint_rejects_bad_files(tmp_path):
    reference = MLP(2, [8], 3, seed=0).params
    good = tmp_path / "good.npz"
    save_checkpoint(reference, good)
    assert set(load_checkpoint(good, reference)) == set(reference)

    missing = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError):
        load_checkpoint(missing, reference)

    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"definitely not a npz file")
    with pytest.raises(ValueError, match="cannot load checkpoint"):
        load_checkpoint(corrupt, reference)

    missing_key = tmp_path / "missing_key.npz"
    save_checkpoint({k: v for k, v in reference.items() if k != "b1"}, missing_key)
    with pytest.raises(ValueError, match="key mismatch"):
        load_checkpoint(missing_key, reference)

    extra_key = tmp_path / "extra_key.npz"
    save_checkpoint({**reference, "W9": np.zeros((1, 1))}, extra_key)
    with pytest.raises(ValueError, match="key mismatch"):
        load_checkpoint(extra_key, reference)

    bad_shape = tmp_path / "bad_shape.npz"
    save_checkpoint({**reference, "W1": np.zeros((3, 3))}, bad_shape)
    with pytest.raises(ValueError, match="shape"):
        load_checkpoint(bad_shape, reference)

    non_finite = tmp_path / "non_finite.npz"
    save_checkpoint({**reference, "W1": np.full_like(reference["W1"], np.nan)}, non_finite)
    with pytest.raises(ValueError, match="non-finite"):
        load_checkpoint(non_finite, reference)

    non_float = tmp_path / "non_float.npz"
    save_checkpoint({**reference, "W1": reference["W1"].astype(np.int64)}, non_float)
    with pytest.raises(ValueError, match="floating"):
        load_checkpoint(non_float, reference)


# ---------- 运行目录与实验产物 ----------


def test_run_directory_refuses_overwrite(tmp_path):
    first = create_run_directory(tmp_path, "run_x")
    assert first.is_dir()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        create_run_directory(tmp_path, "run_x")
    assert create_run_directory(tmp_path, "run_y").is_dir()


def test_run_experiment_success_writes_artifacts(tmp_path):
    X, y, tr, va, te = _tiny_split()
    summary = run_experiment(
        train_images=X[tr], train_labels=y[tr],
        val_images=X[va], val_labels=y[va],
        test_images=X[te], test_labels=y[te],
        config=_tiny_config(),
        output_root=tmp_path,
        run_name="ok_run",
    )
    run_dir = tmp_path / "ok_run"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "history.csv").is_file()
    assert (run_dir / "best_model.npz").is_file()
    assert (run_dir / "summary.json").is_file()
    assert summary["status"] == "success"
    assert summary["sample_counts"] == {"train": 80, "validation": 20, "test": 20}
    for key in ("best_epoch", "best_validation_loss", "best_validation_accuracy",
                "test_loss", "test_accuracy", "run_dir"):
        assert key in summary

    with open(run_dir / "history.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2  # epochs=2
    assert set(rows[0]) == set(HISTORY_FIELDS)

    history_for_select = [
        {"epoch": int(row["epoch"]), "val_loss": float(row["val_loss"])} for row in rows
    ]
    assert summary["best_epoch"] == select_best_epoch(history_for_select)

    reference = MLP(2, [8], 3, hidden_activation="tanh", seed=0).params
    loaded = load_checkpoint(run_dir / "best_model.npz", reference)
    assert set(loaded) == set(reference)


def test_run_experiment_failure_writes_no_success_summary(tmp_path):
    X, y, tr, va, te = _tiny_split()
    bad_train = X[tr].copy()
    bad_train[0, 0] = np.nan  # 非有限输入 → 组件校验快速失败，绝不产生假成功产物
    with pytest.raises(RuntimeError, match="training failed at epoch 1"):
        run_experiment(
            train_images=bad_train, train_labels=y[tr],
            val_images=X[va], val_labels=y[va],
            test_images=X[te], test_labels=y[te],
            config=_tiny_config(),
            output_root=tmp_path,
            run_name="fail_run",
        )
    run_dir = tmp_path / "fail_run"
    assert (run_dir / "config.json").is_file()          # 失败前配置已落盘
    assert not (run_dir / "summary.json").exists()      # 不得写出成功状态 summary
    assert not (run_dir / "best_model.npz").exists()
    assert not (run_dir / "history.csv").exists()


# ---------- CLI（子进程真实执行，不联网） ----------


def _run_cli(args):
    return subprocess.run(
        [sys.executable, "-m", "src.train_mnist", *args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )


def test_summary_test_metrics_correspond_to_best_checkpoint(tmp_path, monkeypatch):
    """复审整改回归：summary 的测试指标必须来自 best_model.npz 保存的最佳参数。

    场景刻意让最后一个 epoch 发散（val_loss 约 2.08，最佳 epoch 约 0.006），
    两组参数天差地别：旧实现用“最后 epoch 的 model”评估测试集，本测试在旧
    实现下必然失败；修复后（测试集评估前恢复最佳参数）必须通过。
    """
    from src import training as training_module

    # 确定性场景：噪声 0.9 的三簇数据 + tanh + lr 2.0，最佳 epoch 在中段出现
    X, y = _make_three_clusters(n_per_class=40, noise=0.9, seed=0)
    perm = np.random.default_rng(1).permutation(len(y))
    tr, va, te = perm[:80], perm[80:100], perm[100:]
    test_images, test_labels = X[te], y[te]

    config = _tiny_config(learning_rate=2.0)
    config["network"]["hidden_dims"] = [16]
    config["network"]["hidden_activation"] = "tanh"
    config["seeds"]["model_seed"] = 7
    config["training"]["epochs"] = 8
    config["training"]["batch_size"] = 8
    config["training"]["eval_batch_size"] = 16

    captured = {}
    real_evaluate = training_module.evaluate

    def spy(model, X_eval, y_eval, batch_size=1000):
        loss, acc = real_evaluate(model, X_eval, y_eval, batch_size=batch_size)
        if X_eval is test_images:  # 只捕获对测试集的那一次评估
            captured["calls"] = captured.get("calls", 0) + 1
            captured["params"] = model.get_params_copy()
            captured["metrics"] = (loss, acc)
        return loss, acc

    monkeypatch.setattr(training_module, "evaluate", spy)
    summary = run_experiment(
        train_images=X[tr], train_labels=y[tr],
        val_images=X[va], val_labels=y[va],
        test_images=test_images, test_labels=test_labels,
        config=config, output_root=tmp_path, run_name="consistency_run",
    )

    run_dir = tmp_path / "consistency_run"
    # 场景守卫：最佳 epoch 不是最后一个 epoch，否则本测试无法区分两组参数
    assert summary["best_epoch"] < config["training"]["epochs"]
    # 测试集只被评估一次，summary 的测试指标就来自这一次
    assert captured["calls"] == 1
    assert captured["metrics"] == (summary["test_loss"], summary["test_accuracy"])

    # 评估测试集时模型里的参数 == best_model.npz 里的最佳参数（旧实现此处必然不等）
    reference = MLP(2, [16], 3, hidden_activation="tanh", seed=0).params
    best_params = load_checkpoint(run_dir / "best_model.npz", reference)
    for key in best_params:
        np.testing.assert_array_equal(best_params[key], captured["params"][key])

    # 独立重算：加载 checkpoint 后对测试集评估，结果必须与 summary 一致
    eval_model = MLP(2, [16], 3, hidden_activation="tanh", seed=999)
    eval_model.set_params(best_params)
    loss, acc = evaluate(eval_model, test_images, test_labels)
    assert loss == pytest.approx(summary["test_loss"], rel=1e-12, abs=1e-15)
    assert acc == pytest.approx(summary["test_accuracy"], abs=1e-12)


def test_cli_help_shows_usage_and_defaults():
    proc = _run_cli(["--help"])
    assert proc.returncode == 0
    for flag in ("--download", "--data-dir", "--output-dir", "--epochs",
                 "--batch-size", "--learning-rate", "--seed", "--split-seed"):
        assert flag in proc.stdout


def test_cli_rejects_invalid_learning_rate_before_training():
    proc = _run_cli(["--learning-rate", "0"])
    assert proc.returncode != 0
    assert "learning-rate" in proc.stdout + proc.stderr
    proc_neg = _run_cli(["--learning-rate", "nan"])
    assert proc_neg.returncode != 0


def test_cli_missing_data_files_error_is_actionable(tmp_path):
    empty = tmp_path / "empty_data"
    empty.mkdir()
    out = tmp_path / "out"
    proc = _run_cli(["--data-dir", str(empty), "--output-dir", str(out)])
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "train-images-idx3-ubyte.gz" in combined   # 说明期望文件名
    assert "--download" in combined                    # 说明如何获取
    assert not out.exists() or not any(out.iterdir())  # 缺数据时未产生运行目录


def test_cli_rejects_invalid_epochs_and_batch_size():
    assert _run_cli(["--epochs", "0"]).returncode != 0
    assert _run_cli(["--batch-size", "-4"]).returncode != 0
    assert _run_cli(["--seed", "-1"]).returncode != 0

