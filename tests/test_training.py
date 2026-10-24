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
    write_history_csv,
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


# ---------------------------------------------------------------------------
# P5.2-B 新增：学习率调度、早停、raw-best 分离与优化器注入
# 全部使用合成数据与脚本化验证指标，不依赖 MNIST。
# ---------------------------------------------------------------------------

import src.training as training_module  # noqa: E402
from src.optimizers import Adam, MomentumSGD  # noqa: E402


def _scripted_validation(monkeypatch, losses):
    """把 training.evaluate 替换为按调用次序返回脚本化验证指标的桩。

    第 i 次调用返回 (losses[i], 0.9)；耗尽脚本后重复最后一个值。
    返回调用计数器供断言使用。
    """
    state = {"call": 0}

    def stub(model, X, y, batch_size=1000):
        index = min(state["call"], len(losses) - 1)
        state["call"] += 1
        return losses[index], 0.9

    monkeypatch.setattr(training_module, "evaluate", stub)
    return state


def _early_stop_run(monkeypatch, losses, epochs, patience=5, min_delta=1e-4, early_stopping=True):
    X, y, tr, va, _te = _tiny_split()
    model = MLP(2, [8], 3, hidden_activation="tanh", seed=0)
    _scripted_validation(monkeypatch, losses)
    return train_model(
        model, X[tr], y[tr], X[va], y[va],
        epochs=epochs, batch_size=32, learning_rate=0.1, seed=0,
        early_stopping=early_stopping, patience=patience, min_delta=min_delta,
    )


# ---------- 学习率调度 ----------


def test_history_lr_without_scheduler_is_constant():
    # lr_scheduler 缺省（none）：所有 epoch 使用初始 lr，行为与 P4 一致
    X, y, tr, va, _te = _tiny_split()
    model = MLP(2, [8], 3, hidden_activation="tanh", seed=0)
    result = train_model(
        model, X[tr], y[tr], X[va], y[va],
        epochs=3, batch_size=32, learning_rate=0.25, seed=0,
    )
    assert [row["lr"] for row in result.history] == [0.25, 0.25, 0.25]
    assert all(np.isfinite(row["lr"]) for row in result.history)


def test_step_decay_schedule_boundaries():
    # 冻结公式 lr(epoch) = base * gamma^((epoch-1)//step_size)：
    # 21 个 epoch → [base]*10 + [base/2]*10 + [base/4]，epoch 10 不提前衰减
    X, y, tr, va, _te = _tiny_split()
    model = MLP(2, [8], 3, hidden_activation="tanh", seed=0)
    result = train_model(
        model, X[tr], y[tr], X[va], y[va],
        epochs=21, batch_size=32, learning_rate=0.1, seed=0,
        lr_scheduler="step", step_size=10, gamma=0.5,
    )
    expected = [0.1] * 10 + [0.05] * 10 + [0.025]
    assert [row["lr"] for row in result.history] == expected
    assert result.history[9]["lr"] == 0.1   # epoch 10 仍是 base lr
    assert result.history[10]["lr"] == 0.05  # epoch 11 才首次衰减
    assert result.history[20]["lr"] == 0.025
    assert result.stop_reason == "max_epochs"


def test_injected_optimizer_instance_is_reused_and_scheduled():
    # 注入的同一 optimizer 实例被调度器驱动：step_size=2 时 epoch 1–2 用 base、
    # 3–4 用 gamma 倍、5 用 gamma² 倍；实例 learning_rate 终值与最后 epoch 一致
    # （证明未重建实例）
    X, y, tr, va, _te = _tiny_split()
    model = MLP(2, [8], 3, hidden_activation="tanh", seed=0)
    opt = MomentumSGD(0.2, momentum=0.9)
    result = train_model(
        model, X[tr], y[tr], X[va], y[va],
        epochs=5, batch_size=32, seed=0,
        optimizer=opt, lr_scheduler="step", step_size=2, gamma=0.5,
    )
    assert [row["lr"] for row in result.history] == [0.2, 0.2, 0.1, 0.1, 0.05]
    assert opt.learning_rate == pytest.approx(0.05)
    assert result.stop_reason == "max_epochs"


def test_optimizer_and_learning_rate_are_mutually_exclusive():
    X, y, tr, va, _te = _tiny_split()
    model = MLP(2, [8], 3, seed=0)
    with pytest.raises(ValueError, match="not both"):
        train_model(
            model, X[tr], y[tr], X[va], y[va],
            epochs=1, batch_size=32, learning_rate=0.1, seed=0,
            optimizer=MomentumSGD(0.1),
        )
    with pytest.raises(ValueError, match="either optimizer or learning_rate"):
        train_model(model, X[tr], y[tr], X[va], y[va], epochs=1, batch_size=32, seed=0)
    with pytest.raises(TypeError, match="optimizer must be"):
        train_model(
            model, X[tr], y[tr], X[va], y[va],
            epochs=1, batch_size=32, seed=0, optimizer="sgd",
        )


@pytest.mark.parametrize(
    "kwargs, expected_exception",
    [
        ({"lr_scheduler": "cosine"}, ValueError),
        ({"lr_scheduler": "step", "step_size": 0}, ValueError),
        ({"lr_scheduler": "step", "step_size": -10}, ValueError),
        ({"lr_scheduler": "step", "step_size": True}, ValueError),
        ({"lr_scheduler": "step", "step_size": 2.5}, ValueError),
        ({"lr_scheduler": "step", "gamma": 0.0}, ValueError),
        ({"lr_scheduler": "step", "gamma": -0.5}, ValueError),
        ({"lr_scheduler": "step", "gamma": 1.5}, ValueError),
        ({"lr_scheduler": "step", "gamma": float("nan")}, ValueError),
        ({"lr_scheduler": "step", "gamma": float("inf")}, ValueError),
        ({"lr_scheduler": "step", "gamma": True}, TypeError),
    ],
)
def test_invalid_scheduler_arguments_rejected(kwargs, expected_exception):
    X, y, tr, va, _te = _tiny_split()
    model = MLP(2, [8], 3, seed=0)
    with pytest.raises(expected_exception):
        train_model(
            model, X[tr], y[tr], X[va], y[va],
            epochs=1, batch_size=32, learning_rate=0.1, seed=0, **kwargs,
        )


# ---------- 早停语义（脚本化验证损失序列） ----------


def test_early_stopping_disabled_runs_all_epochs(monkeypatch):
    result = _early_stop_run(monkeypatch, [0.5, 0.4, 0.3, 0.3], epochs=4, early_stopping=False)
    assert len(result.history) == 4
    assert result.stop_reason == "max_epochs"


def test_early_stopping_stops_at_patience_without_extra_epoch(monkeypatch):
    # [1.0]*6，patience=2：ep1 初始化 reference；ep2/ep3 连续不改善 → ep3 停止
    result = _early_stop_run(monkeypatch, [1.0] * 6, epochs=6, patience=2)
    assert len(result.history) == 3
    assert result.stop_reason == "early_stopping"
    assert result.best_epoch == 1
    assert result.history[-1]["epoch"] == 3  # 停止 epoch 计入 history，无额外一轮


def test_min_delta_boundary_counts_as_improvement(monkeypatch):
    # val_loss == reference - min_delta 属于"足够大的改善"（<= 语义）→ 重置计数
    result = _early_stop_run(
        monkeypatch, [1.0, 0.9999, 0.9999], epochs=3, patience=2, min_delta=1e-4
    )
    assert len(result.history) == 3
    assert result.stop_reason == "max_epochs"


def test_small_improvement_updates_raw_best_but_not_patience(monkeypatch):
    # [1.0, 0.995, 0.988, 0.988]，min_delta=0.01，patience=2：
    # ep2: 0.995 < 1.0 → raw-best 更新（best_epoch=2），但 0.995 > 0.99 → bad=1；
    # ep3: 0.988 <= 1.0-0.01 → 改善（若错误地在 ep2 更新 reference，这里就会停止）；
    # ep4: 0.988 <= 0.988-0.01? 否 → bad=1 < 2 → 跑满 4 个 epoch
    result = _early_stop_run(
        monkeypatch, [1.0, 0.995, 0.988, 0.988], epochs=4, patience=2, min_delta=0.01
    )
    assert len(result.history) == 4
    assert result.stop_reason == "max_epochs"
    assert result.best_epoch == 3
    assert result.best_val_loss == pytest.approx(0.988)


def test_patience_exhaustion_keeps_small_improvement_as_best(monkeypatch):
    # [1.0, 0.995, 0.996, 0.997]，min_delta=0.01，patience=2：
    # ep2 raw-best=0.995 但 bad=1；ep3 bad=2 → 停止；best_epoch 指向轻微改善 epoch
    result = _early_stop_run(
        monkeypatch, [1.0, 0.995, 0.996, 0.997], epochs=6, patience=2, min_delta=0.01
    )
    assert len(result.history) == 3
    assert result.stop_reason == "early_stopping"
    assert result.best_epoch == 2
    assert result.best_val_loss == pytest.approx(0.995)


def test_real_improvement_resets_patience(monkeypatch):
    # [1.0, 0.9, 0.9, 0.9]，min_delta=0.05，patience=2：
    # ep2: 0.9 <= 0.95 → 改善并重置；ep3 bad=1；ep4 bad=2 → 停止于 ep4
    # （若无重置语义，ep3 就会停止）
    result = _early_stop_run(
        monkeypatch, [1.0, 0.9, 0.9, 0.9], epochs=6, patience=2, min_delta=0.05
    )
    assert len(result.history) == 4
    assert result.stop_reason == "early_stopping"


def test_min_delta_zero_treats_equal_loss_as_improvement(monkeypatch):
    # min_delta=0（合法下界）：val_loss == reference 即改善，连续平局不早停
    result = _early_stop_run(monkeypatch, [1.0, 1.0, 1.0], epochs=3, patience=2, min_delta=0.0)
    assert len(result.history) == 3
    assert result.stop_reason == "max_epochs"


@pytest.mark.parametrize(
    "losses",
    [[1.0, float("nan")], [1.0, float("inf")], [1.0, -float("inf")]],
)
def test_non_finite_validation_loss_fails_fast(monkeypatch, losses):
    # 非有限验证指标必须快速失败（带 epoch 上下文），不得当作普通 bad epoch
    X, y, tr, va, _te = _tiny_split()
    model = MLP(2, [8], 3, hidden_activation="tanh", seed=0)
    _scripted_validation(monkeypatch, losses)
    with pytest.raises(RuntimeError, match="epoch 2"):
        train_model(
            model, X[tr], y[tr], X[va], y[va],
            epochs=3, batch_size=32, learning_rate=0.1, seed=0, early_stopping=True,
        )


@pytest.mark.parametrize(
    "kwargs, expected_exception",
    [
        ({"patience": 0}, ValueError),
        ({"patience": -1}, ValueError),
        ({"patience": True}, ValueError),
        ({"patience": 2.5}, ValueError),
        ({"patience": "5"}, ValueError),
        ({"min_delta": -1e-4}, ValueError),
        ({"min_delta": float("nan")}, ValueError),
        ({"min_delta": float("inf")}, ValueError),
        ({"min_delta": True}, TypeError),
        ({"min_delta": "1e-4"}, TypeError),
        ({"min_delta": None}, TypeError),
        ({"early_stopping": 1}, TypeError),
    ],
)
def test_invalid_early_stopping_arguments_rejected(kwargs, expected_exception):
    X, y, tr, va, _te = _tiny_split()
    model = MLP(2, [8], 3, seed=0)
    with pytest.raises(expected_exception):
        train_model(
            model, X[tr], y[tr], X[va], y[va],
            epochs=1, batch_size=32, learning_rate=0.1, seed=0,
            early_stopping=True, **kwargs,
        )


# ---------- 早停下的 checkpoint 与产物一致性 ----------


def test_early_stop_artifacts_consistent_with_raw_best(monkeypatch, tmp_path):
    # [1.0, 0.995, 0.996]，patience=2 → 停止于 ep3，但 raw-best 是 ep2；
    # run_experiment 保存的 checkpoint、summary 最佳指标与测试评估前恢复的
    # 参数必须全部指向 ep2（沿用 P4 一致性回归的证明方式）
    from src.training import load_checkpoint

    real_evaluate = training_module.evaluate
    X, y, tr, va, te = _tiny_split()
    test_images = X[te]
    captured = {}
    val_sequence = iter([1.0, 0.995, 0.996])

    def stub(model, X_eval, y_eval, batch_size=1000):
        if X_eval is test_images:  # 测试集评估：走真实计算并捕获恢复后的参数
            loss, acc = real_evaluate(model, X_eval, y_eval, batch_size=batch_size)
            captured["params"] = model.get_params_copy()
            return loss, acc
        return next(val_sequence, 0.996), 0.9  # 验证评估：脚本化 val_loss

    monkeypatch.setattr(training_module, "evaluate", stub)
    config = _tiny_config(learning_rate=0.1)
    config["network"]["hidden_activation"] = "tanh"
    config["training"]["epochs"] = 6
    config["training"]["early_stopping"] = True
    config["training"]["patience"] = 2
    config["training"]["min_delta"] = 0.01
    summary = run_experiment(
        train_images=X[tr], train_labels=y[tr],
        val_images=X[va], val_labels=y[va],
        test_images=test_images, test_labels=y[te],
        config=config, output_root=tmp_path, run_name="early_stop_run",
    )

    assert summary["stop_reason"] == "early_stopping"
    assert summary["best_epoch"] == 2
    assert summary["best_validation_loss"] == pytest.approx(0.995)
    assert summary["epochs_completed"] == 3

    run_dir = tmp_path / "early_stop_run"
    reference = MLP(2, [8], 3, hidden_activation="tanh", seed=0).params
    best_params = load_checkpoint(run_dir / "best_model.npz", reference)
    for key in best_params:  # 测试评估前恢复的参数 == checkpoint（raw-best）
        np.testing.assert_array_equal(best_params[key], captured["params"][key])
    eval_model = MLP(2, [8], 3, hidden_activation="tanh", seed=999)
    eval_model.set_params(best_params)
    loss, acc = real_evaluate(eval_model, test_images, y[te])
    assert loss == pytest.approx(summary["test_loss"])
    assert acc == pytest.approx(summary["test_accuracy"])


def test_history_csv_columns_and_lr_values(tmp_path):
    # history.csv 保留全部旧列且 lr 在末尾；每行 lr 有限并与训练实际使用值一致
    X, y, tr, va, _te = _tiny_split()
    model = MLP(2, [8], 3, hidden_activation="tanh", seed=0)
    result = train_model(
        model, X[tr], y[tr], X[va], y[va],
        epochs=5, batch_size=32, learning_rate=0.2, seed=0,
        lr_scheduler="step", step_size=2, gamma=0.5,
    )
    write_history_csv(result.history, tmp_path / "history.csv")
    with open(tmp_path / "history.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == list(HISTORY_FIELDS)
    assert list(rows[0].keys())[-1] == "lr"
    assert [float(row["lr"]) for row in rows] == [0.2, 0.2, 0.1, 0.1, 0.05]


# ---------- 优化器选择与 config 兼容 ----------


def test_run_experiment_defaults_to_sgd_without_optimizer_key(monkeypatch, tmp_path):
    created = []

    class CountingSGD(training_module.SGD):
        def __init__(self, *args, **kwargs):
            created.append(self)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(training_module, "SGD", CountingSGD)
    X, y, tr, va, te = _tiny_split()
    summary = run_experiment(
        train_images=X[tr], train_labels=y[tr],
        val_images=X[va], val_labels=y[va],
        test_images=X[te], test_labels=y[te],
        config=_tiny_config(),  # 旧式 config：无 optimizer/调度/早停键
        output_root=tmp_path, run_name="sgd_default",
    )
    assert len(created) == 1  # 旧 config 默认 SGD
    assert summary["stop_reason"] == "max_epochs"
    with open(tmp_path / "sgd_default" / "config.json", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["training"]["optimizer"] == "sgd"
    assert saved["training"]["lr_scheduler"] == "none"
    assert saved["training"]["early_stopping"] is False
    assert saved["training"]["patience"] == 5
    assert saved["training"]["min_delta"] == 1e-4
    assert saved["training"]["step_size"] == 10
    assert saved["training"]["gamma"] == 0.5


@pytest.mark.parametrize(
    "optimizer_name, attr_name, optimizer_cls",
    [("momentum_sgd", "MomentumSGD", MomentumSGD), ("adam", "Adam", Adam)],
)
def test_run_experiment_instantiates_selected_optimizer_once(
    monkeypatch, tmp_path, optimizer_name, attr_name, optimizer_cls
):
    created = []

    class Counting(optimizer_cls):
        def __init__(self, *args, **kwargs):
            created.append(self)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(training_module, attr_name, Counting)
    X, y, tr, va, te = _tiny_split()
    config = _tiny_config()
    config["training"]["optimizer"] = optimizer_name
    config["training"]["epochs"] = 3
    summary = run_experiment(
        train_images=X[tr], train_labels=y[tr],
        val_images=X[va], val_labels=y[va],
        test_images=X[te], test_labels=y[te],
        config=config, output_root=tmp_path, run_name=f"run_{optimizer_name}",
    )
    assert len(created) == 1  # 每 run 恰好实例化一次，不跨 epoch 重建
    assert isinstance(created[0], optimizer_cls)
    assert summary["epochs_completed"] == 3
    with open(tmp_path / f"run_{optimizer_name}" / "config.json", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["training"]["optimizer"] == optimizer_name
    if optimizer_name == "momentum_sgd":
        assert saved["training"]["momentum"] == 0.9
    else:
        assert saved["training"]["beta1"] == 0.9
        assert saved["training"]["beta2"] == 0.999
        assert saved["training"]["eps"] == 1e-8


def test_caller_config_not_mutated_by_run_experiment(tmp_path):
    X, y, tr, va, te = _tiny_split()
    config = _tiny_config()
    before = json.loads(json.dumps(config))
    run_experiment(
        train_images=X[tr], train_labels=y[tr],
        val_images=X[va], val_labels=y[va],
        test_images=X[te], test_labels=y[te],
        config=config, output_root=tmp_path, run_name="no_mutation",
    )
    assert config == before  # 调用方原始 config 不被写入默认键


# ---------- CLI ----------


def test_cli_help_shows_new_training_control_options():
    proc = _run_cli(["--help"])
    assert proc.returncode == 0
    for flag in ("--optimizer", "--lr-scheduler", "--early-stopping", "--patience", "--min-delta"):
        assert flag in proc.stdout
    assert "sgd" in proc.stdout  # 默认优化器


def test_cli_rejects_invalid_new_options():
    assert _run_cli(["--optimizer", "adagrad"]).returncode != 0
    assert _run_cli(["--lr-scheduler", "cosine"]).returncode != 0
    assert _run_cli(["--patience", "0"]).returncode != 0
    assert _run_cli(["--min-delta", "-0.5"]).returncode != 0
    assert _run_cli(["--min-delta", "nan"]).returncode != 0

