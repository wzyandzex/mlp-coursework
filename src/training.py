"""训练闭环：mini-batch 训练、批量评估、模型选择、checkpoint 与实验产物。

职责边界
--------
* 训练器只做流程调度：每步调用现有 MLP.forward(return_cache=True)、
  softmax_cross_entropy、MLP.backward 与 SGD.step，不重写任何数学公式；
* dlogits 已含 batch 平均因子，本模块不再除以 batch size；
* 训练指标为“在线指标”：用每次参数更新前的 logits 计算、按 batch 内样本数
  加权累计（见 train_model docstring），整个训练过程前后一致；
* 验证/测试指标在 epoch 结束后对完整数据集批量评估得到；
* 测试集只在训练与基于验证集的模型选择全部完成后评估一次。

模型选择规则
------------
训练全部 epoch（无早停）；以 val_loss 最低的 epoch 为最佳 checkpoint，
val_loss 完全相同时保留更早的 epoch（扫描时用严格小于实现）。
"""

import csv
import json
import platform
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .losses import softmax_cross_entropy
from .mlp import MLP
from .optimizers import SGD

HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "elapsed_seconds",
)


def evaluate(model, X, y, batch_size: int = 1000) -> tuple:
    """批量评估：返回 (样本数加权平均 loss, accuracy)。

    * 评估期间不更新参数、不修改输入数据；
    * loss 用各 batch 的平均损失 × batch 样本数累加后除以总样本数，
      即按样本数正确加权，而不是对不同大小 batch 的均值再做简单平均；
    * accuracy = 正确分类样本数 / 样本总数。argmax(logits) 与
      argmax(softmax(logits)) 等价（Softmax 单调），无需重复计算概率。
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"X must be a non-empty 2-D array, got shape {X.shape}.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"y must be 1-D with length {X.shape[0]}, got shape {y.shape}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    total = X.shape[0]
    loss_sum = 0.0
    correct = 0
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        logits = model.forward(X[start:end])
        batch_loss, _ = softmax_cross_entropy(logits, y[start:end])
        loss_sum += float(batch_loss) * (end - start)
        correct += int(np.sum(np.argmax(logits, axis=1) == y[start:end]))
    return loss_sum / total, correct / total


def _iterate_minibatches(n_samples: int, batch_size: int, rng: np.random.Generator):
    """每个 epoch 生成一个独立、可复现的打乱顺序，并按 batch 切分（含尾部不足整批）。"""
    permutation = rng.permutation(n_samples)
    for start in range(0, n_samples, batch_size):
        yield permutation[start : start + batch_size]


@dataclass
class TrainingResult:
    """一次训练的全部输出：逐 epoch 历史、最佳 checkpoint 及其指标。"""

    history: list = field(default_factory=list)  # 每行键见 HISTORY_FIELDS
    best_epoch: int = 0  # 1-based
    best_params: dict = field(default_factory=dict)
    best_val_loss: float = float("inf")
    best_val_accuracy: float = 0.0
    total_seconds: float = 0.0


def select_best_epoch(history: list) -> int:
    """从历史记录中选出 val_loss 最低的 epoch（1-based）；并列时取更早的 epoch。"""
    if not history:
        raise ValueError("history is empty; nothing to select from.")
    best_row = min(history, key=lambda row: (row["val_loss"], row["epoch"]))
    return int(best_row["epoch"])


def train_model(
    model,
    train_X,
    train_y,
    val_X,
    val_y,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed,
    eval_batch_size: int = 1000,
    verbose: bool = False,
) -> TrainingResult:
    """mini-batch 训练循环（不早停，跑满全部 epoch）。

    * 每 epoch 用 seed 派生的同一个 Generator 顺序生成新的可复现打乱顺序，
      不重复同一固定顺序；
    * 尾部不足整批的样本不丢弃；
    * 每步调用 forward(return_cache=True) → softmax_cross_entropy → backward
      → SGD.step；dlogits 已含批次平均，这里不再除以 batch size；
    * 任何一步出现非有限值（组件校验或显式检查）都会快速失败并指出 epoch/batch；
    * 训练指标是在线指标：用更新前的 logits 计算，按样本数加权；
    * 最佳 checkpoint 只依据 val_loss（并列取更早 epoch），参数深复制快照，
      不会被后续 epoch 的更新覆盖。
    """
    if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs <= 0:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")
    train_X = np.asarray(train_X)
    train_y = np.asarray(train_y)
    val_X = np.asarray(val_X)
    val_y = np.asarray(val_y)
    if train_X.ndim != 2 or train_X.shape[0] == 0:
        raise ValueError(f"train_X must be a non-empty 2-D array, got shape {train_X.shape}.")
    if train_y.ndim != 1 or train_y.shape[0] != train_X.shape[0]:
        raise ValueError(
            f"train_y must be 1-D with length {train_X.shape[0]}, got shape {train_y.shape}."
        )
    if val_X.ndim != 2 or val_X.shape[0] == 0:
        raise ValueError(f"val_X must be a non-empty 2-D array, got shape {val_X.shape}.")
    if val_y.ndim != 1 or val_y.shape[0] != val_X.shape[0]:
        raise ValueError(
            f"val_y must be 1-D with length {val_X.shape[0]}, got shape {val_y.shape}."
        )

    optimizer = SGD(learning_rate)
    rng = np.random.default_rng(seed)
    n_samples = train_X.shape[0]
    history: list = []
    result = TrainingResult()
    started = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for batch_number, batch_idx in enumerate(
            _iterate_minibatches(n_samples, int(batch_size), rng), start=1
        ):
            batch_X = train_X[batch_idx]
            batch_y = train_y[batch_idx]
            try:
                logits, cache = model.forward(batch_X, return_cache=True)
                loss, dlogits = softmax_cross_entropy(logits, batch_y)
                grads = model.backward(dlogits, cache)
                optimizer.step(model.params, grads)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"training failed at epoch {epoch} batch {batch_number}: {exc}"
                ) from exc
            if not np.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss {loss!r} at epoch {epoch} batch {batch_number}; "
                    f"refusing to continue with a corrupt model."
                )
            # 在线训练指标：此处 logits 是参数更新前的输出，按样本数加权累计
            loss_sum += float(loss) * len(batch_idx)
            correct += int(np.sum(np.argmax(logits, axis=1) == batch_y))
            seen += len(batch_idx)

        train_loss = loss_sum / seen
        train_accuracy = correct / seen
        val_loss, val_accuracy = evaluate(model, val_X, val_y, batch_size=eval_batch_size)
        elapsed = time.perf_counter() - epoch_started
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "elapsed_seconds": elapsed,
            }
        )
        # 严格小于：val_loss 并列时保留更早的 epoch
        if val_loss < result.best_val_loss:
            result.best_val_loss = val_loss
            result.best_val_accuracy = val_accuracy
            result.best_epoch = epoch
            result.best_params = model.get_params_copy()  # 深复制快照
        if verbose:
            print(
                f"[epoch {epoch}/{epochs}] train_loss={train_loss:.4f} "
                f"train_acc={train_accuracy:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_accuracy:.4f} ({elapsed:.1f}s)"
            )

    result.history = history
    result.total_seconds = time.perf_counter() - started
    return result


# ---------- checkpoint 保存 / 加载 ----------


def save_checkpoint(params: dict, path) -> None:
    """把参数字典保存为无 pickle 的 .npz 文件（数组键值对）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **params)


def load_checkpoint(path, reference_params: dict) -> dict:
    """安全加载 checkpoint，并对照 reference_params 校验键、形状与有限性。

    加载使用 allow_pickle=False；文件缺失、损坏、键/形状不匹配或含非有限值
    都会得到明确异常，绝不返回静默损坏的参数。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"cannot load checkpoint {path}: {exc}") from exc

    expected_keys = set(reference_params)
    actual_keys = set(arrays)
    if actual_keys != expected_keys:
        raise ValueError(
            f"checkpoint {path} key mismatch; missing: {sorted(expected_keys - actual_keys)}, "
            f"unexpected: {sorted(actual_keys - expected_keys)}."
        )
    for key, reference in reference_params.items():
        arr = arrays[key]
        if arr.shape != np.shape(reference):
            raise ValueError(
                f"checkpoint {path} array {key!r} has shape {arr.shape}, expected {np.shape(reference)}."
            )
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError(f"checkpoint {path} array {key!r} must be floating, got {arr.dtype}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"checkpoint {path} array {key!r} contains non-finite values.")
    return {key: arrays[key].astype(np.float64) for key in arrays}


# ---------- 实验产物 ----------


def create_run_directory(output_root, run_name: str = None) -> Path:
    """在 output_root 下创建本次运行的独立目录；已存在则报错，绝不静默覆盖。"""
    root = Path(output_root)
    name = run_name if run_name else f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = root / name
    if run_dir.exists():
        raise ValueError(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def write_history_csv(history: list, path) -> None:
    """把逐 epoch 历史写成列名稳定的 CSV，便于后续绘图与分析。"""
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HISTORY_FIELDS))
        writer.writeheader()
        for row in history:
            writer.writerow({key: row[key] for key in HISTORY_FIELDS})


def _environment_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def run_experiment(
    *,
    train_images,
    train_labels,
    val_images,
    val_labels,
    test_images,
    test_labels,
    config: dict,
    output_root,
    run_name: str = None,
    verbose: bool = False,
) -> dict:
    """端到端编排：建运行目录 → 写 config → 训练 → 选最佳 → 测试集评估一次 → 保存产物。

    config 需包含 network / training / seeds / data 四节（见 CLI 的组装方式）。
    训练失败时异常向上传播，目录里只留下 config.json，绝不会出现标记为成功
    却缺少关键指标的 summary.json。
    """
    run_dir = create_run_directory(output_root, run_name)
    effective_config = json.loads(json.dumps(config))  # 深复制，避免污染调用方
    effective_config["environment"] = _environment_info()
    effective_config.setdefault("data", {})
    effective_config["data"]["sample_counts"] = {
        "train": int(np.asarray(train_images).shape[0]),
        "validation": int(np.asarray(val_images).shape[0]),
        "test": int(np.asarray(test_images).shape[0]),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(effective_config, handle, indent=2, ensure_ascii=False)

    network = config["network"]
    training_config = config["training"]
    model = MLP(
        input_dim=network["input_dim"],
        hidden_dims=network["hidden_dims"],
        output_dim=network["output_dim"],
        hidden_activation=network.get("hidden_activation", "relu"),
        initializer=network.get("initializer", "auto"),
        seed=config["seeds"]["model_seed"],
    )
    result = train_model(
        model,
        train_images,
        train_labels,
        val_images,
        val_labels,
        epochs=training_config["epochs"],
        batch_size=training_config["batch_size"],
        learning_rate=training_config["learning_rate"],
        seed=config["seeds"]["model_seed"],
        eval_batch_size=training_config.get("eval_batch_size", 1000),
        verbose=verbose,
    )

    # 测试集只在训练与验证集选型全部完成后评估这一次
    test_loss, test_accuracy = evaluate(model, test_images, test_labels)

    write_history_csv(result.history, run_dir / "history.csv")
    save_checkpoint(result.best_params, run_dir / "best_model.npz")
    summary = {
        "status": "success",
        "run_dir": str(run_dir),
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_val_loss,
        "best_validation_accuracy": result.best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "sample_counts": effective_config["data"]["sample_counts"],
        "epochs_completed": len(result.history),
        "total_seconds": result.total_seconds,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary
