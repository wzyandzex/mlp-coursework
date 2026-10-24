"""MNIST 数据管线：IDX 文件解析、显式下载、预处理与可复现分层切分（纯标准库 + NumPy）。

职责边界
--------
* raw IDX 加载（load_idx_images / load_idx_labels）与预处理、切分相互独立；
* 下载只在调用方显式请求时发生（CLI --download），导入或测试不会联网；
* 本模块不包含训练逻辑。

预处理契约
----------
* 图片 uint8 → np.float64 / 255.0，展平为 (N, 784)，取值 [0, 1]；
* 标签 np.int64 一维 (N,)，类别索引 0..9；不做 one-hot（损失接口接收类别索引）；
* 不做均值方差标准化、PCA、数据增强等任何额外预处理。

切分方案（已批准的 20k/5k/10k）
------------------------------
train/validation 只从官方 60,000 训练样本中按类别分层随机抽取：
每类配额按比例分配（最大余数法保证总数精确），类内用显式 seed 的
Generator 打乱后依次抽取。官方 10,000 测试样本整体保留、不参与切分。
"""

import gzip
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

IMAGE_MAGIC = 2051  # IDX 图片文件 magic number (0x00000803)
LABEL_MAGIC = 2049  # IDX 标签文件 magic number (0x00000801)
IMAGE_ROWS = 28
IMAGE_COLS = 28
NUM_CLASSES = 10

TRAIN_SIZE = 20000
VAL_SIZE = 5000
DEFAULT_SPLIT_SEED = 2026

MNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)

# 数据源集中定义：同一文件按顺序尝试多个镜像；下载不关闭 TLS 验证
_SOURCE_BASES = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)
MNIST_SOURCES = {name: tuple(base + name for base in _SOURCE_BASES) for name in MNIST_FILES}


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"data file not found: {path}")


def _read_gzip_bytes(path: Path) -> bytes:
    """解压 gzip 文件；损坏的 gzip 流给出清晰错误而不是静默截断。"""
    try:
        with gzip.open(path, "rb") as handle:
            return handle.read()
    except (OSError, EOFError) as exc:
        raise ValueError(f"cannot read gzip file {path}: {exc}") from exc


def load_idx_images(path) -> np.ndarray:
    """解析 IDX 图片文件，返回 uint8 数组，形状 (N, 28, 28)。

    校验：magic number、非空、尺寸恰为 28x28、payload 字节长度与 header 一致。
    任何损坏输入抛出带文件名与原因的 ValueError。
    """
    path = Path(path)
    _require_file(path)
    data = _read_gzip_bytes(path)
    if len(data) < 16:
        raise ValueError(
            f"truncated IDX image file {path}: header needs 16 bytes, got {len(data)}."
        )
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != IMAGE_MAGIC:
        raise ValueError(
            f"bad magic number {magic} in {path}; expected {IMAGE_MAGIC} for IDX images."
        )
    if count == 0:
        raise ValueError(f"empty IDX image file {path}.")
    if rows != IMAGE_ROWS or cols != IMAGE_COLS:
        raise ValueError(
            f"unsupported image size {rows}x{cols} in {path}; MNIST requires 28x28."
        )
    payload = data[16:]
    expected = count * rows * cols
    if len(payload) != expected:
        raise ValueError(
            f"payload length mismatch in {path}: expected {expected} bytes for "
            f"{count} images of {rows}x{cols}, got {len(payload)} (file truncated or corrupted)."
        )
    return np.frombuffer(payload, dtype=np.uint8).reshape(count, rows, cols).copy()


def load_idx_labels(path) -> np.ndarray:
    """解析 IDX 标签文件，返回 np.int64 数组，形状 (N,)，取值 0..9。"""
    path = Path(path)
    _require_file(path)
    data = _read_gzip_bytes(path)
    if len(data) < 8:
        raise ValueError(
            f"truncated IDX label file {path}: header needs 8 bytes, got {len(data)}."
        )
    magic, count = struct.unpack(">II", data[:8])
    if magic != LABEL_MAGIC:
        raise ValueError(
            f"bad magic number {magic} in {path}; expected {LABEL_MAGIC} for IDX labels."
        )
    if count == 0:
        raise ValueError(f"empty IDX label file {path}.")
    payload = data[8:]
    if len(payload) != count:
        raise ValueError(
            f"payload length mismatch in {path}: expected {count} labels, got {len(payload)} "
            f"(file truncated or corrupted)."
        )
    labels = np.frombuffer(payload, dtype=np.uint8)
    if labels.size and int(labels.max()) >= NUM_CLASSES:
        raise ValueError(
            f"invalid labels in {path}: values must be 0..{NUM_CLASSES - 1}, found max {int(labels.max())}."
        )
    return labels.astype(np.int64)


def preprocess_images(images) -> np.ndarray:
    """uint8 图片 (N, 28, 28) → float64 (N, 784)，除以 255.0，取值 [0, 1]。"""
    arr = np.asarray(images)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"images must have an integer dtype, got {arr.dtype}.")
    if arr.ndim != 3 or arr.shape[1:] != (IMAGE_ROWS, IMAGE_COLS):
        raise ValueError(
            f"images must have shape (N, {IMAGE_ROWS}, {IMAGE_COLS}), got {arr.shape}."
        )
    out = arr.astype(np.float64).reshape(arr.shape[0], -1) / 255.0
    if not np.all(np.isfinite(out)) or out.min() < 0.0 or out.max() > 1.0:
        raise ValueError("preprocessed images contain values outside [0, 1] or non-finite.")
    return out


def load_mnist_raw(data_dir):
    """加载四个标准 IDX gzip 文件，返回 (train_images, train_labels, test_images, test_labels)。

    图片为 uint8 (N, 28, 28)，标签为 int64 (N,)；并校验每对图片/标签样本数相等。
    """
    directory = Path(data_dir)
    missing = [name for name in MNIST_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing MNIST file(s) in {directory}: {', '.join(missing)}. "
            f"Run with --download to fetch them, or manually place the four standard "
            f"IDX gzip files there: {', '.join(MNIST_FILES)}."
        )
    train_images = load_idx_images(directory / "train-images-idx3-ubyte.gz")
    train_labels = load_idx_labels(directory / "train-labels-idx1-ubyte.gz")
    test_images = load_idx_images(directory / "t10k-images-idx3-ubyte.gz")
    test_labels = load_idx_labels(directory / "t10k-labels-idx1-ubyte.gz")
    if len(train_images) != len(train_labels):
        raise ValueError(
            f"train images/labels count mismatch: {len(train_images)} images vs {len(train_labels)} labels."
        )
    if len(test_images) != len(test_labels):
        raise ValueError(
            f"test images/labels count mismatch: {len(test_images)} images vs {len(test_labels)} labels."
        )
    return train_images, train_labels, test_images, test_labels


def _quotas(counts: np.ndarray, total: int) -> np.ndarray:
    """按比例把 total 个样本配额分配到各类（最大余数法，总和精确等于 total）。

    余数并列时按类别升序分配，保证同一输入总是得到同一配额。
    """
    exact = counts * (total / counts.sum())
    base = np.floor(exact).astype(np.int64)
    remaining = int(total - base.sum())
    order = np.argsort(-(exact - base), kind="stable")
    base[order[:remaining]] += 1
    return base


def split_train_validation(labels, n_train: int, n_val: int, seed) -> tuple[np.ndarray, np.ndarray]:
    """分层随机切分：返回 (train_indices, val_indices)，两集合不重叠。

    每类按比例获得 n_train / n_val 配额（最大余数法），类内用显式 seed 的
    Generator 打乱后依次抽取；相同 labels 与 seed 必产生相同划分。
    不做“排序后取前若干条”的截取。
    """
    arr = np.asarray(labels)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"labels must be a non-empty 1-D array, got shape {arr.shape}.")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"labels must have an integer dtype, got {arr.dtype}.")
    if int(arr.min()) < 0:
        raise ValueError(f"labels must be non-negative class indices, got min {int(arr.min())}.")
    for name, value in (("n_train", n_train), ("n_val", n_val)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0
    ):
        raise ValueError(f"seed must be a non-negative integer or None, got {seed!r}.")
    total = int(arr.size)
    if int(n_train) + int(n_val) > total:
        raise ValueError(
            f"requested {n_train} train + {n_val} validation samples exceed {total} available."
        )

    counts = np.bincount(arr)
    train_quota = _quotas(counts, int(n_train))
    val_quota = _quotas(counts, int(n_val))
    for cls in range(len(counts)):
        need = int(train_quota[cls] + val_quota[cls])
        if need > int(counts[cls]):
            raise ValueError(
                f"not enough samples for class {cls}: has {int(counts[cls])}, "
                f"needs {int(train_quota[cls])} train + {int(val_quota[cls])} validation."
            )

    rng = np.random.default_rng(seed)
    train_idx = np.empty(int(n_train), dtype=np.int64)
    val_idx = np.empty(int(n_val), dtype=np.int64)
    train_pos = 0
    val_pos = 0
    for cls in range(len(counts)):
        class_indices = np.flatnonzero(arr == cls)
        rng.shuffle(class_indices)
        q_train = int(train_quota[cls])
        q_val = int(val_quota[cls])
        train_idx[train_pos : train_pos + q_train] = class_indices[:q_train]
        val_idx[val_pos : val_pos + q_val] = class_indices[q_train : q_train + q_val]
        train_pos += q_train
        val_pos += q_val
    return train_idx, val_idx


@dataclass(frozen=True)
class MNISTSplit:
    """预处理后的训练/验证/测试数据与切分元信息。"""

    train_images: np.ndarray  # (n_train, 784) float64
    train_labels: np.ndarray  # (n_train,) int64
    val_images: np.ndarray
    val_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    split_seed: int
    strategy: str


SPLIT_STRATEGY = (
    "proportional stratified random split per class "
    "(largest-remainder quotas, per-class shuffle with explicit seed); "
    "note: the class-balanced train subset is NOT the full 60000-sample training set"
)


def load_mnist_split(
    data_dir,
    train_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> MNISTSplit:
    """加载 raw MNIST → 预处理 → 从官方 60k 训练样本切出 train/val，官方 10k 作为 test。"""
    train_images_u8, train_labels, test_images_u8, test_labels = load_mnist_raw(data_dir)
    train_idx, val_idx = split_train_validation(train_labels, train_size, val_size, split_seed)
    return MNISTSplit(
        train_images=preprocess_images(train_images_u8[train_idx]),
        train_labels=train_labels[train_idx],
        val_images=preprocess_images(train_images_u8[val_idx]),
        val_labels=train_labels[val_idx],
        test_images=preprocess_images(test_images_u8),
        test_labels=test_labels,
        split_seed=split_seed,
        strategy=SPLIT_STRATEGY,
    )


def _validate_downloaded_gzip(path: Path, name: str) -> None:
    """下载完成后对临时文件做完整 gzip/IDX 结构校验，通过后才允许替换正式文件。"""
    if "images" in name:
        load_idx_images(path)
    elif "labels" in name:
        load_idx_labels(path)
    else:  # pragma: no cover - MNIST_FILES 均可按名称区分
        raise ValueError(f"unknown MNIST file name: {name}")


def _download_file(name: str, url: str, dest: Path, timeout: float) -> None:
    """下载单个文件：写入同目录临时文件 → 结构校验 → 原子替换。

    失败时删除临时残片、保留既有正式文件，并抛出包含文件名与原因的错误。
    """
    tmp = dest.with_name(dest.name + ".part")
    request = Request(url, headers={"User-Agent": "mlp-coursework/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response, open(tmp, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        _validate_downloaded_gzip(tmp, name)
        os.replace(tmp, dest)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"failed to download {name} from {url}: {exc}") from exc


def download_mnist(data_dir, timeout: float = 60.0) -> list:
    """下载缺失的 MNIST 文件到 data_dir（已存在的文件保持不动），返回本次下载的文件名列表。

    只应在用户显式请求（CLI --download）时调用。
    """
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for name in MNIST_FILES:
        dest = directory / name
        if dest.is_file():
            continue
        last_error = None
        for url in MNIST_SOURCES[name]:
            try:
                print(f"[download] {name} from {url} ...")
                _download_file(name, url, dest, timeout)
                downloaded.append(name)
                print(f"[download] saved {dest}")
                last_error = None
                break
            except RuntimeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
    return downloaded
