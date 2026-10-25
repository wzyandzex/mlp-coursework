"""训练闭环：mini-batch 训练、批量评估、模型选择、checkpoint 与实验产物。

职责边界
--------
* 训练器只做流程调度：每步调用现有 MLP.forward(return_cache=True)、
  softmax_cross_entropy、MLP.backward 与 optimizer.step，不重写任何数学公式；
  优化器由调用方注入，或未注入时按 learning_rate 默认创建 SGD；
* 学习率调度（none / step）在每个 epoch 开始前把有效学习率写入同一 optimizer
  实例的 learning_rate 属性（不重建实例、不清空状态），history 的 lr 列记录
  该 epoch 实际使用值；
* 早停仅监控验证集 val_loss；raw-best checkpoint（原始 val_loss 严格下降，
  并列取更早 epoch）与 early_stop_reference / bad_epochs 计数严格分离，
  min_delta 不影响 raw-best；测试数据不进入训练器；
* dlogits 已含 batch 平均因子，本模块不再除以 batch size；
* 训练指标为“在线指标”：用每次参数更新前的 logits 计算、按 batch 内样本数
  加权累计（见 train_model docstring），整个训练过程前后一致；
* 验证/测试指标在 epoch 结束后对完整数据集批量评估得到；
* 测试集只在训练与基于验证集的模型选择全部完成后评估一次。

模型选择规则
------------
训练至多跑满全部 epoch（早停可提前终止）；以 val_loss 最低的 epoch 为最佳
checkpoint，val_loss 完全相同时保留更早的 epoch（扫描时用严格小于实现）。
"""

import csv
import json
import math
import platform
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .losses import softmax_cross_entropy
from .mlp import MLP
from .optimizers import SGD, MomentumSGD, Adam

HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "elapsed_seconds",
    "lr",
    "train_objective_loss",
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


def evaluate_with_predictions(model, X, y, batch_size: int = 1000) -> tuple:
    """联合评估：同一次前向同时返回 (样本加权 loss, accuracy, 预测标签)。

    与 evaluate() 完全相同的批处理、样本加权与比较规则；额外保留每个样本
    的 argmax 预测（int64），供混淆矩阵等分类指标与 test_loss/test_accuracy
    同源——final 评估不得对测试集做第二次前向。
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
    predictions = np.empty(total, dtype=np.int64)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        logits = model.forward(X[start:end])
        batch_loss, _ = softmax_cross_entropy(logits, y[start:end])
        loss_sum += float(batch_loss) * (end - start)
        batch_pred = np.argmax(logits, axis=1)
        predictions[start:end] = batch_pred
        correct += int(np.sum(batch_pred == y[start:end]))
    return loss_sum / total, correct / total, predictions


def confusion_matrix(y_true, y_pred, num_classes: int) -> "np.ndarray":
    """固定类别顺序的整数混淆矩阵：行=真实类别，列=预测类别。

    严格输入校验：标签必须是一维、等长、非空的整数数组（拒绝浮点/布尔等
    dtype，绝不静默转换）；标签越界即报错；num_classes 为 >= 2 的整数。
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, (int, np.integer)):
        raise ValueError(f"num_classes must be an integer, got {num_classes!r}.")
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}.")
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError(
            f"labels must be 1-D arrays, got shapes {y_true_arr.shape} / {y_pred_arr.shape}."
        )
    if y_true_arr.size == 0:
        raise ValueError("labels must be non-empty.")
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length, got "
            f"{y_true_arr.shape[0]} / {y_pred_arr.shape[0]}."
        )
    for name, arr in (("y_true", y_true_arr), ("y_pred", y_pred_arr)):
        if arr.dtype.kind not in "iu":
            raise ValueError(
                f"{name} must be an integer-dtype array of class labels; got dtype "
                f"{arr.dtype}. Floating-point labels are rejected instead of being "
                f"silently cast to integers."
            )
        if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= num_classes):
            raise ValueError(
                f"{name} contains labels outside [0, {num_classes - 1}]: "
                f"min={int(arr.min())}, max={int(arr.max())}."
            )
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true_arr, y_pred_arr), 1)
    return matrix


def per_class_metrics(confusion) -> list:
    """从混淆矩阵计算每类 precision / recall / f1 / support（按类别 0..C-1 排序）。

    零分母规则（预注册）：predicted 为 0 时 precision=0.0；support 为 0 时
    recall=0.0；precision+recall 为 0 时 f1=0.0。所有输出有限且在 [0,1]。
    """
    arr = np.asarray(confusion)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"confusion must be a square 2-D matrix, got shape {arr.shape}.")
    if arr.shape[0] < 2:
        raise ValueError(f"confusion must have at least 2 classes, got {arr.shape[0]}.")
    if arr.dtype.kind not in "iu":
        raise ValueError(
            f"confusion must be an integer-dtype matrix; got dtype {arr.dtype}."
        )
    if arr.size and int(arr.min()) < 0:
        raise ValueError("confusion counts must be non-negative.")
    metrics = []
    num_classes = arr.shape[0]
    for cls in range(num_classes):
        tp = int(arr[cls, cls])
        support = int(arr[cls, :].sum())
        predicted = int(arr[:, cls].sum())
        precision = tp / predicted if predicted > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        metrics.append(
            {"precision": float(precision), "recall": float(recall),
             "f1": float(f1), "support": support}
        )
    return metrics


def macro_average(per_class: list) -> dict:
    """全部类别逐类指标的不加权算术平均（不按 support 加权、不剔除任何类别）。"""
    if not isinstance(per_class, list) or len(per_class) == 0:
        raise ValueError("per_class must be a non-empty list of per-class metric dicts.")
    keys = ("precision", "recall", "f1")
    for index, entry in enumerate(per_class):
        if not isinstance(entry, dict):
            raise ValueError(f"per_class[{index}] must be a dict, got {entry!r}.")
        for key in keys:
            value = entry.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"per_class[{index}][{key!r}] must be a number, got {value!r}."
                )
            if not math.isfinite(float(value)):
                raise ValueError(f"per_class[{index}][{key!r}] must be finite.")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"per_class[{index}][{key!r}] must be within [0, 1], got {value!r}."
                )
    return {key: float(sum(float(entry[key]) for entry in per_class) / len(per_class))
            for key in keys}


def _iterate_minibatches(n_samples: int, batch_size: int, rng: np.random.Generator):
    """每个 epoch 生成一个独立、可复现的打乱顺序，并按 batch 切分（含尾部不足整批）。"""
    permutation = rng.permutation(n_samples)
    for start in range(0, n_samples, batch_size):
        yield permutation[start : start + batch_size]


def l2_penalty(params: dict, l2_lambda: float) -> float:
    """L2 penalty 项：(λ/2)·Σ_l ||W_l||_F²。

    仅惩罚权重矩阵（键以 "W" 开头），绝不惩罚偏置；λ=0 时恒返回 0.0，
    保证无正则路径与历史训练的数值行为逐位一致。
    """
    if l2_lambda == 0.0:
        return 0.0
    squared_sum = 0.0
    for key, value in params.items():
        if key.startswith("W"):
            squared_sum += float(np.sum(value * value))
    return 0.5 * l2_lambda * squared_sum


def add_l2_gradient(grads: dict, params: dict, l2_lambda: float) -> dict:
    """返回加入 L2 梯度项后的新梯度字典：∇_W J_obj = ∇_W J_data + λ·W。

    * 偏置梯度原样保留（L2 不作用于偏置）；
    * 权重梯度通过 `value + λ·W` 生成新数组，绝不就地修改 backward 返回的
      原始梯度数组（因此 λ 项与 batch size 无关，不会被二次平均）；
    * λ=0 时原样返回传入字典。
    """
    if l2_lambda == 0.0:
        return grads
    regularized: dict = {}
    for key, value in grads.items():
        if key.startswith("W"):
            regularized[key] = value + l2_lambda * params[key]
        else:
            regularized[key] = value
    return regularized


@dataclass
class TrainingResult:
    """一次训练的全部输出：逐 epoch 历史、最佳 checkpoint 及其指标。"""

    history: list = field(default_factory=list)  # 每行键见 HISTORY_FIELDS
    best_epoch: int = 0  # 1-based
    best_params: dict = field(default_factory=dict)
    best_val_loss: float = float("inf")
    best_val_accuracy: float = 0.0
    total_seconds: float = 0.0
    stop_reason: str = "max_epochs"  # 仅允许 "max_epochs" 与 "early_stopping"


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
    learning_rate: float | None = None,
    seed,
    optimizer=None,
    lr_scheduler: str = "none",
    step_size: int = 10,
    gamma: float = 0.5,
    early_stopping: bool = False,
    patience: int = 5,
    min_delta: float = 1e-4,
    l2_lambda: float = 0.0,
    eval_batch_size: int = 1000,
    verbose: bool = False,
) -> TrainingResult:
    """mini-batch 训练循环（最多跑满全部 epoch，可被早停提前终止）。

    学习率来源
    ----------
    * 未传 optimizer 时按 learning_rate 创建 SGD（既有调用方式不变）；
    * 传入 optimizer（SGD/MomentumSGD/Adam 实例）时，base lr 从
      optimizer.learning_rate 读取；两者同时传入会被拒绝，避免冲突来源。

    学习率调度
    ----------
    * "none"（默认）：所有 epoch 使用 base lr，与旧行为一致；
    * "step"：lr(epoch) = base_lr * gamma ** ((epoch - 1) // step_size)，
      即 epoch 1..step_size 用 base，之后每 step_size 个 epoch 乘 gamma；
      有效 lr 在每个 epoch 第一个 mini-batch 之前写入同一 optimizer 实例的
      learning_rate 属性（不重建实例、不清空状态），history 的 lr 列记录
      该 epoch 实际使用值。

    早停语义（监控量仅为 val_loss；测试数据不进入本函数）
    ----------------------------------------------------
    * 第一轮有限的 val_loss 初始化 early_stop_reference，bad_epochs = 0；
    * 当且仅当 val_loss <= early_stop_reference - min_delta 视为“足够大的
      改善”：更新 early_stop_reference 并重置 bad_epochs；否则 bad_epochs += 1；
    * bad_epochs >= patience 时，在完成该 epoch 的 history 与 raw-best 快照
      更新后停止，stop_reason = "early_stopping"；否则为 "max_epochs"；
    * raw-best checkpoint（原始 val_loss 严格下降，并列取更早 epoch）与上述
      计数严格分离：小于 min_delta 的轻微下降也必须更新 raw-best，但不重置
      计数；验证指标非有限时快速失败，不计入 bad_epochs。

    其余语义与 P4 保持一致：每 epoch 独立可复现打乱、尾批不丢弃、在线训练
    指标按样本数加权、非有限值快速失败并指出 epoch/batch、最佳参数深复制
    快照不被后续 epoch 覆盖。

    L2 正则化（冻结约定）
    --------------------
    * J_obj = J_data + (λ/2)·Σ_l ||W_l||_F²，仅惩罚权重矩阵，绝不惩罚偏置；
    * 权重梯度加 λ·W（softmax_cross_entropy 的 dlogits 已含 batch 平均，
      L2 项不得再除以 batch size），偏置梯度不变；
    * penalty 与梯度项均使用本次反向传播之前的模型参数（同一更新前模型）；
    * 不就地修改 backward 返回的原始梯度数组——正则化梯度由 add_l2_gradient
      生成新数组；
    * train_loss 仍是纯训练数据损失（在线、按样本数加权）；train_objective_loss
      为数据损失 + L2 penalty 的训练目标，两者分列记录；λ=0 时两者逐位相等；
    * val_loss / 模型选择 / 测试评估全部保持纯 data loss，不掺 L2 项。
    """
    if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs <= 0:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")
    # 学习率来源：optimizer 与 learning_rate 互斥，避免两个冲突来源
    if optimizer is not None:
        if learning_rate is not None:
            raise ValueError(
                "pass either optimizer or learning_rate, not both; the base learning "
                "rate is read from optimizer.learning_rate."
            )
        if not isinstance(optimizer, (SGD, MomentumSGD, Adam)):
            raise TypeError(
                f"optimizer must be an SGD, MomentumSGD or Adam instance, "
                f"got {type(optimizer).__name__}."
            )
        try:
            base_lr = float(optimizer.learning_rate)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"optimizer.learning_rate must be a real number, got {optimizer.learning_rate!r}."
            ) from exc
        if not np.isfinite(base_lr) or base_lr <= 0.0:
            raise ValueError(
                f"optimizer.learning_rate must be finite and positive, got {optimizer.learning_rate!r}."
            )
    else:
        if learning_rate is None:
            raise ValueError("either optimizer or learning_rate must be provided.")
        optimizer = SGD(learning_rate)  # 构造器完成学习率校验
        base_lr = optimizer.learning_rate
    if lr_scheduler not in ("none", "step"):
        raise ValueError(f"unsupported lr_scheduler {lr_scheduler!r}; choices: 'none', 'step'.")
    if isinstance(step_size, bool) or not isinstance(step_size, (int, np.integer)) or step_size <= 0:
        raise ValueError(f"step_size must be a positive integer, got {step_size!r}.")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float, np.integer, np.floating)):
        raise TypeError(f"gamma must be a real number, got {type(gamma).__name__}.")
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0.0 or gamma > 1.0:
        raise ValueError(
            f"gamma must be a finite positive real number with gamma <= 1, got {gamma!r}."
        )
    if not isinstance(early_stopping, bool):
        raise TypeError(f"early_stopping must be a bool, got {type(early_stopping).__name__}.")
    if isinstance(patience, bool) or not isinstance(patience, (int, np.integer)) or patience <= 0:
        raise ValueError(f"patience must be a positive integer, got {patience!r}.")
    if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float, np.integer, np.floating)):
        raise TypeError(f"min_delta must be a real number, got {type(min_delta).__name__}.")
    min_delta = float(min_delta)
    if not np.isfinite(min_delta) or min_delta < 0.0:
        raise ValueError(
            f"min_delta must be a finite non-negative real number, got {min_delta!r}."
        )
    if isinstance(l2_lambda, bool) or not isinstance(l2_lambda, (int, float, np.integer, np.floating)):
        raise TypeError(f"l2_lambda must be a real number, got {type(l2_lambda).__name__}.")
    l2_lambda = float(l2_lambda)
    if not np.isfinite(l2_lambda) or l2_lambda < 0.0:
        raise ValueError(
            f"l2_lambda must be a finite non-negative real number, got {l2_lambda!r}."
        )
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

    rng = np.random.default_rng(seed)
    n_samples = train_X.shape[0]
    history: list = []
    result = TrainingResult()
    early_stop_reference: float | None = None
    bad_epochs = 0
    stop_reason = "max_epochs"
    started = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        # 学习率调度：在 epoch 第一个 mini-batch 之前生效（无 off-by-one：
        # epoch 1..step_size 用 base lr，step_size+1..2*step_size 用 gamma 倍……）
        if lr_scheduler == "step":
            effective_lr = base_lr * (gamma ** ((epoch - 1) // int(step_size)))
        else:
            effective_lr = base_lr
        optimizer.learning_rate = effective_lr
        loss_sum = 0.0
        objective_sum = 0.0
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
                if l2_lambda > 0.0:
                    # J_obj = J_data + (λ/2)Σ||W||²：penalty 与 λ·W 梯度项都用
                    # 本次反向传播之前的模型参数；不除以 batch size、不修改
                    # backward 返回的原始梯度数组
                    penalty = l2_penalty(model.params, l2_lambda)
                    grads = add_l2_gradient(grads, model.params, l2_lambda)
                else:
                    penalty = 0.0
                objective = loss + penalty
                optimizer.step(model.params, grads)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"training failed at epoch {epoch} batch {batch_number}: {exc}"
                ) from exc
            if not np.isfinite(loss) or not np.isfinite(objective):
                raise RuntimeError(
                    f"non-finite loss {loss!r} or objective {objective!r} at epoch {epoch} "
                    f"batch {batch_number}; refusing to continue with a corrupt model."
                )
            # 在线训练指标：此处 logits 是参数更新前的输出，按样本数加权累计
            loss_sum += float(loss) * len(batch_idx)
            objective_sum += float(objective) * len(batch_idx)
            correct += int(np.sum(np.argmax(logits, axis=1) == batch_y))
            seen += len(batch_idx)

        train_loss = loss_sum / seen
        train_objective_loss = objective_sum / seen
        train_accuracy = correct / seen
        try:
            val_loss, val_accuracy = evaluate(model, val_X, val_y, batch_size=eval_batch_size)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"validation failed at epoch {epoch}: {exc}") from exc
        if not (np.isfinite(val_loss) and np.isfinite(val_accuracy)):
            # 非有限验证指标必须快速失败：不计入 bad_epochs、不更新任何计数器
            raise RuntimeError(
                f"non-finite validation metrics at epoch {epoch}: "
                f"val_loss={val_loss!r}, val_accuracy={val_accuracy!r}; refusing to continue."
            )
        elapsed = time.perf_counter() - epoch_started
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "elapsed_seconds": elapsed,
                "lr": effective_lr,
                "train_objective_loss": train_objective_loss,
            }
        )
        # raw-best checkpoint：只看原始 val_loss 严格下降（并列保留更早 epoch），
        # 与早停计数完全分离——小于 min_delta 的轻微下降也必须更新 raw-best
        if val_loss < result.best_val_loss:
            result.best_val_loss = val_loss
            result.best_val_accuracy = val_accuracy
            result.best_epoch = epoch
            result.best_params = model.get_params_copy()  # 深复制快照
        early_stop_triggered = False
        if early_stopping:
            if early_stop_reference is None:
                # 第一轮有限的 val_loss 初始化参考值
                early_stop_reference = val_loss
                bad_epochs = 0
            elif val_loss <= early_stop_reference - min_delta:
                # “足够大的改善”：更新参考值并重置计数
                early_stop_reference = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= int(patience):
                stop_reason = "early_stopping"
                early_stop_triggered = True
        if verbose:
            print(
                f"[epoch {epoch}/{epochs}] train_loss={train_loss:.4f} "
                f"train_acc={train_accuracy:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_accuracy:.4f} lr={effective_lr:.4g} ({elapsed:.1f}s)"
            )
        if early_stop_triggered:
            if verbose:
                print(
                    f"[early stopping] epoch {epoch}: val_loss 已连续 {bad_epochs} 个 "
                    f"epoch 无足够改善（patience={patience}），停止训练"
                )
            break

    result.history = history
    result.stop_reason = stop_reason
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
    evaluate_test: bool = True,
    experiment_role: str | None = None,
) -> dict:
    """端到端编排：建运行目录 → 写 config → 训练 → 选最佳 → 测试评估 → 保存产物。

    运行角色与测试集隔离（P5.2-C）
    ------------------------------
    * baseline（默认推导：evaluate_test=True）：历史兼容的基线入口，
      必须执行测试评估；
    * exploration（默认推导：evaluate_test=False）：探索/调参 run，
      validation-only——完全跳过测试集 evaluate 调用，summary 的
      test_evaluation="not_run"，test_loss/test_accuracy 为 JSON null；
    * final：完整实验方案冻结后的正式确认评估，必须显式传入；
      任何默认路径都不会推导出 final；
    * 角色/评估组合不匹配（如 exploration+evaluate_test=True）或未知角色
      会在创建运行目录、写入任何产物之前抛出 ValueError；
    * config.json 顶层记录 experiment_role 与 evaluate_test，
      training.l2_lambda 记录实际使用的正则系数。

    config 需包含 network / training / seeds / data 四节（见 CLI 的组装方式）。
    训练失败时异常向上传播，目录里只留下 config.json，绝不会出现标记为成功
    却缺少关键指标的 summary.json。
    """
    if not isinstance(evaluate_test, bool):
        raise TypeError(f"evaluate_test must be a bool, got {type(evaluate_test).__name__}.")
    allowed_roles = ("baseline", "exploration", "final")
    if experiment_role is None:
        # 默认推导：baseline / exploration；任何默认路径都不会产生 final
        experiment_role = "baseline" if evaluate_test else "exploration"
    elif not isinstance(experiment_role, str) or experiment_role not in allowed_roles:
        raise ValueError(
            f"unsupported experiment_role {experiment_role!r}; "
            f"choices: {', '.join(allowed_roles)}."
        )
    expected_evaluation = {"baseline": True, "exploration": False, "final": True}[experiment_role]
    if evaluate_test != expected_evaluation:
        raise ValueError(
            f"experiment_role {experiment_role!r} requires evaluate_test="
            f"{expected_evaluation}, got {evaluate_test}; baseline/final runs evaluate "
            f"the test set, exploration runs are validation-only."
        )

    run_dir = create_run_directory(output_root, run_name)
    effective_config = json.loads(json.dumps(config))  # 深复制，避免污染调用方
    effective_config["environment"] = _environment_info()
    effective_config["experiment_role"] = experiment_role
    effective_config["evaluate_test"] = evaluate_test
    effective_config.setdefault("data", {})
    effective_config["data"]["sample_counts"] = {
        "train": int(np.asarray(train_images).shape[0]),
        "validation": int(np.asarray(val_images).shape[0]),
        "test": int(np.asarray(test_images).shape[0]),
    }
    # 训练控制有效默认值：旧 config 缺键时得到与 P4 行为一致的默认，
    # 并保证 config.json 记录的正是本次 run 实际使用的全部有效超参数
    effective_training = effective_config.setdefault("training", {})
    effective_training.setdefault("optimizer", "sgd")
    effective_training.setdefault("momentum", 0.9)
    effective_training.setdefault("beta1", 0.9)
    effective_training.setdefault("beta2", 0.999)
    effective_training.setdefault("eps", 1e-8)
    effective_training.setdefault("lr_scheduler", "none")
    effective_training.setdefault("step_size", 10)
    effective_training.setdefault("gamma", 0.5)
    effective_training.setdefault("early_stopping", False)
    effective_training.setdefault("patience", 5)
    effective_training.setdefault("min_delta", 1e-4)
    effective_training.setdefault("l2_lambda", 0.0)
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(effective_config, handle, indent=2, ensure_ascii=False)

    network = config["network"]
    training_config = effective_training  # 与 config.json 完全一致的有效配置
    model = MLP(
        input_dim=network["input_dim"],
        hidden_dims=network["hidden_dims"],
        output_dim=network["output_dim"],
        hidden_activation=network.get("hidden_activation", "relu"),
        initializer=network.get("initializer", "auto"),
        seed=config["seeds"]["model_seed"],
    )

    # 每 run 新建优化器实例；整个训练复用同一实例（调度只改 learning_rate 属性）
    optimizer_name = training_config["optimizer"]
    lr_value = training_config["learning_rate"]
    if optimizer_name == "sgd":
        optimizer = SGD(lr_value)
    elif optimizer_name == "momentum_sgd":
        optimizer = MomentumSGD(lr_value, momentum=training_config["momentum"])
    elif optimizer_name == "adam":
        optimizer = Adam(
            lr_value,
            beta1=training_config["beta1"],
            beta2=training_config["beta2"],
            eps=training_config["eps"],
        )
    else:
        raise ValueError(
            f"unsupported optimizer {optimizer_name!r}; "
            f"choices: 'sgd', 'momentum_sgd', 'adam'."
        )

    result = train_model(
        model,
        train_images,
        train_labels,
        val_images,
        val_labels,
        epochs=training_config["epochs"],
        batch_size=training_config["batch_size"],
        seed=config["seeds"]["model_seed"],
        optimizer=optimizer,
        lr_scheduler=training_config["lr_scheduler"],
        step_size=training_config["step_size"],
        gamma=training_config["gamma"],
        early_stopping=training_config["early_stopping"],
        patience=training_config["patience"],
        min_delta=training_config["min_delta"],
        l2_lambda=training_config["l2_lambda"],
        eval_batch_size=training_config.get("eval_batch_size", 1000),
        verbose=verbose,
    )

    if evaluate_test:
        # 测试集只在训练与验证集选型全部完成后评估这一次；评估前必须恢复
        # 最佳 epoch 参数，保证测试指标与 best_model.npz 是同一组参数。
        # 探索路径（evaluate_test=False）完全不进入此分支。
        model.set_params(result.best_params)
        if experiment_role == "final":
            # final 联合评估：test_loss/test_accuracy/预测标签来自同一次前向，
            # 混淆矩阵、逐类指标与 Macro Average 与之严格同源。
            test_loss, test_accuracy, test_predictions = evaluate_with_predictions(
                model, test_images, test_labels
            )
            num_classes = int(network["output_dim"])
            matrix = confusion_matrix(test_labels, test_predictions, num_classes)
            per_class = per_class_metrics(matrix)
            macro = macro_average(per_class)
        else:
            test_loss, test_accuracy = evaluate(model, test_images, test_labels)
        test_evaluation = "executed"
    else:
        # 未评估的指标必须是 JSON null，绝不能写成 0 或伪造数值
        test_loss = None
        test_accuracy = None
        test_evaluation = "not_run"

    write_history_csv(result.history, run_dir / "history.csv")
    save_checkpoint(result.best_params, run_dir / "best_model.npz")
    summary = {
        "status": "success",
        "run_dir": str(run_dir),
        "experiment_role": experiment_role,
        "test_evaluation": test_evaluation,
        "stop_reason": result.stop_reason,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_val_loss,
        "best_validation_accuracy": result.best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "sample_counts": effective_config["data"]["sample_counts"],
        "epochs_completed": len(result.history),
        "total_seconds": result.total_seconds,
    }
    if experiment_role == "final":
        # 仅 final run 携带分类指标字段；baseline/exploration 的既有 summary
        # schema 保持不变（旧产物缺这些字段是正常状态，不是损坏）。
        summary["test_confusion_matrix"] = matrix.tolist()
        summary["test_per_class_metrics"] = per_class
        summary["test_macro_avg"] = macro
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary
