"""MNIST 数据管线测试：IDX 解析、预处理、分层切分与下载器（全部离线，无真实网络请求）。"""

import gzip
import struct
import urllib.error
from pathlib import Path

import numpy as np
import pytest

from src import mnist_data as md


# ---------- 合成 IDX fixture 工具 ----------


def _idx_images_bytes(n, rows=28, cols=28, seed=0, magic=md.IMAGE_MAGIC):
    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 256, size=n * rows * cols, dtype=np.uint8).tobytes()
    return struct.pack(">IIII", magic, n, rows, cols) + payload


def _idx_labels_bytes(values, magic=md.LABEL_MAGIC):
    values = list(values)
    return struct.pack(">II", magic, len(values)) + bytes(values)


def _write_gz(path: Path, raw: bytes) -> Path:
    path.write_bytes(gzip.compress(raw))
    return path


@pytest.fixture()
def raw_dir(tmp_path):
    """一个含 30 张合成 28x28 图片（每类 3 张）的小型 MNIST 目录。"""
    n = 30
    labels = np.arange(n) % md.NUM_CLASSES
    images = np.zeros((n, 28, 28), dtype=np.uint8)
    rng = np.random.default_rng(1)
    images[:] = rng.integers(0, 256, size=(n, 28, 28), dtype=np.uint8)
    _write_gz(tmp_path / "train-images-idx3-ubyte.gz", _idx_images_bytes(n, seed=2))
    _write_gz(tmp_path / "train-labels-idx1-ubyte.gz", _idx_labels_bytes(labels))
    _write_gz(tmp_path / "t10k-images-idx3-ubyte.gz", _idx_images_bytes(10, seed=3))
    _write_gz(tmp_path / "t10k-labels-idx1-ubyte.gz", _idx_labels_bytes(np.arange(10) % 10))
    return tmp_path


# ---------- IDX 解析 ----------


def test_load_idx_images_valid(tmp_path):
    path = _write_gz(tmp_path / "img.gz", _idx_images_bytes(5, seed=7))
    arr = md.load_idx_images(path)
    assert arr.shape == (5, 28, 28)
    assert arr.dtype == np.uint8


def test_load_idx_labels_valid(tmp_path):
    path = _write_gz(tmp_path / "lab.gz", _idx_labels_bytes([0, 3, 9]))
    arr = md.load_idx_labels(path)
    assert arr.shape == (3,)
    assert arr.dtype == np.int64
    assert arr.tolist() == [0, 3, 9]


def test_load_idx_images_bad_magic(tmp_path):
    path = _write_gz(tmp_path / "img.gz", _idx_images_bytes(2, magic=2052))
    with pytest.raises(ValueError, match="magic number"):
        md.load_idx_images(path)


def test_load_idx_images_truncated_header(tmp_path):
    path = tmp_path / "img.gz"
    path.write_bytes(gzip.compress(b"\x00" * 5))
    with pytest.raises(ValueError, match="truncated"):
        md.load_idx_images(path)


def test_load_idx_images_truncated_payload(tmp_path):
    # header 声称 5 张图，payload 只有 2 张的字节
    raw = struct.pack(">IIII", md.IMAGE_MAGIC, 5, 28, 28) + b"\x00" * (2 * 28 * 28)
    path = _write_gz(tmp_path / "img.gz", raw)
    with pytest.raises(ValueError, match="payload length mismatch"):
        md.load_idx_images(path)


def test_load_idx_images_empty(tmp_path):
    path = _write_gz(tmp_path / "img.gz", _idx_images_bytes(0))
    with pytest.raises(ValueError, match="empty"):
        md.load_idx_images(path)


def test_load_idx_images_rejects_non_28x28(tmp_path):
    path = _write_gz(tmp_path / "img.gz", _idx_images_bytes(2, rows=14, cols=14))
    with pytest.raises(ValueError, match="28x28"):
        md.load_idx_images(path)


def test_load_idx_labels_bad_magic(tmp_path):
    path = _write_gz(tmp_path / "lab.gz", _idx_labels_bytes([0, 1], magic=2048))
    with pytest.raises(ValueError, match="magic number"):
        md.load_idx_labels(path)


def test_load_idx_labels_out_of_range(tmp_path):
    path = _write_gz(tmp_path / "lab.gz", _idx_labels_bytes([0, 10, 2]))
    with pytest.raises(ValueError, match="out of range|invalid labels"):
        md.load_idx_labels(path)


def test_load_idx_labels_truncated_payload(tmp_path):
    raw = struct.pack(">II", md.LABEL_MAGIC, 5) + b"\x01\x02"
    path = _write_gz(tmp_path / "lab.gz", raw)
    with pytest.raises(ValueError, match="payload length mismatch"):
        md.load_idx_labels(path)


def test_corrupt_gzip_stream_rejected(tmp_path):
    path = tmp_path / "img.gz"
    path.write_bytes(b"this is not gzip at all")
    with pytest.raises(ValueError, match="cannot read gzip"):
        md.load_idx_images(path)


def test_missing_file_error_mentions_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="images-idx3"):
        md.load_idx_images(tmp_path / "train-images-idx3-ubyte.gz")


# ---------- 预处理 ----------


def test_preprocess_images_contract():
    images = np.zeros((3, 28, 28), dtype=np.uint8)
    images[0, 0, 0] = 255
    images[1, 5, 5] = 128
    out = md.preprocess_images(images)
    assert out.shape == (3, 784)
    assert out.dtype == np.float64
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out[0, 0] == 1.0
    assert out[1, 5 * 28 + 5] == pytest.approx(128 / 255.0)
    assert out[2].max() == 0.0


def test_preprocess_images_rejects_wrong_shape_and_dtype():
    with pytest.raises(ValueError, match="shape"):
        md.preprocess_images(np.zeros((4, 784), dtype=np.uint8))
    with pytest.raises(TypeError, match="integer dtype"):
        md.preprocess_images(np.zeros((3, 28, 28), dtype=np.float64))


# ---------- load_mnist_raw / load_mnist_split ----------


def test_load_mnist_raw_pair_counts_must_match(raw_dir):
    # 把训练标签换成数量不一致的文件
    _write_gz(raw_dir / "train-labels-idx1-ubyte.gz", _idx_labels_bytes([0, 1]))
    with pytest.raises(ValueError, match="count mismatch"):
        md.load_mnist_raw(raw_dir)


def test_load_mnist_raw_missing_file_message_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="--download"):
        md.load_mnist_raw(tmp_path)


def test_load_mnist_split_shapes_and_metadata(raw_dir):
    split = md.load_mnist_split(raw_dir, train_size=10, val_size=10, split_seed=0)
    assert split.train_images.shape == (10, 784)
    assert split.val_images.shape == (10, 784)
    assert split.test_images.shape == (10, 784)
    for labels in (split.train_labels, split.val_labels, split.test_labels):
        assert labels.dtype == np.int64
        assert set(labels.tolist()) <= set(range(10))
    assert split.train_images.dtype == np.float64
    assert split.split_seed == 0
    # train 与 val 索引不重叠：两者样本必须来自不同的原始行
    assert split.train_images.shape[0] + split.val_images.shape[0] == 20


def test_load_mnist_split_reproducible(raw_dir):
    a = md.load_mnist_split(raw_dir, train_size=10, val_size=10, split_seed=5)
    b = md.load_mnist_split(raw_dir, train_size=10, val_size=10, split_seed=5)
    np.testing.assert_array_equal(a.train_images, b.train_images)
    np.testing.assert_array_equal(a.val_labels, b.val_labels)


# ---------- 分层切分 ----------


def _balanced_labels(n_per_class=10, n_classes=10):
    return np.tile(np.arange(n_classes), n_per_class)


def test_split_sizes_and_disjointness():
    labels = _balanced_labels()
    train_idx, val_idx = md.split_train_validation(labels, 60, 20, seed=0)
    assert len(train_idx) == 60 and len(val_idx) == 20
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0
    assert all(0 <= i < len(labels) for i in train_idx.tolist() + val_idx.tolist())


def test_split_is_stratified_all_classes_present():
    labels = _balanced_labels(n_per_class=10)
    train_idx, val_idx = md.split_train_validation(labels, 50, 20, seed=3)
    assert set(labels[train_idx].tolist()) == set(range(10))
    assert set(labels[val_idx].tolist()) == set(range(10))
    # 每类训练配额应为 5：分层不是简单排序截取
    assert np.all(np.bincount(labels[train_idx], minlength=10) == 5)
    assert np.all(np.bincount(labels[val_idx], minlength=10) == 2)


def test_split_same_seed_reproducible_different_seed_changes():
    labels = _balanced_labels(n_per_class=20)
    t1, v1 = md.split_train_validation(labels, 100, 50, seed=7)
    t2, v2 = md.split_train_validation(labels, 100, 50, seed=7)
    t3, _ = md.split_train_validation(labels, 100, 50, seed=8)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(v1, v2)
    assert not np.array_equal(t1, t3)


def test_split_proportional_on_imbalanced_classes():
    # 类别不均衡：100 个样本中类 0 占 50%、类 1 占 30%、类 2 占 20%
    labels = np.concatenate(
        [np.zeros(50, dtype=np.int64), np.ones(30, dtype=np.int64), np.full(20, 2, dtype=np.int64)]
    )
    train_idx, val_idx = md.split_train_validation(labels, 60, 40, seed=1)
    assert len(train_idx) == 60 and len(val_idx) == 40
    # 比例可整除，配额应精确：train 30/18/12，val 20/12/8
    np.testing.assert_array_equal(np.bincount(labels[train_idx], minlength=10), [30, 18, 12, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(np.bincount(labels[val_idx], minlength=10), [20, 12, 8, 0, 0, 0, 0, 0, 0, 0])


def test_split_largest_remainder_quotas():
    # 比例不可整除时用最大余数法：10 个样本（类 0×5、类 1×3、类 2×2），train=5
    # 精确配额 2.5/1.5/1.0 → 基础 2/1/1，剩余 1 个给余数最大的类 0 → [3,1,1]
    labels = np.array([0] * 5 + [1] * 3 + [2] * 2, dtype=np.int64)
    train_idx, val_idx = md.split_train_validation(labels, 5, 3, seed=0)
    np.testing.assert_array_equal(np.bincount(labels[train_idx], minlength=10), [3, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    # val=3：精确 1.5/0.9/0.6 → 基础 1/0/0，剩余 2 个给类 1、类 2 → [1,1,1]
    np.testing.assert_array_equal(np.bincount(labels[val_idx], minlength=10), [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0


def test_split_rejects_when_total_insufficient():
    labels = _balanced_labels(n_per_class=5)  # 50 个样本
    with pytest.raises(ValueError, match="exceed 50 available"):
        md.split_train_validation(labels, 40, 20, seed=0)


def test_split_rejects_when_single_class_insufficient():
    # 类 9 只有 3 个样本，比例配额会要求 1 train + 1 val，仍够；构造极端：类 0 只有 1 个
    labels = np.array([0] + [1] * 19, dtype=np.int64)
    with pytest.raises(ValueError, match="not enough samples for class"):
        md.split_train_validation(labels, 10, 10, seed=0)


def test_split_works_on_small_data_not_tied_to_60000():
    labels = np.repeat(np.arange(10), 3)  # 30 个样本
    train_idx, val_idx = md.split_train_validation(labels, 10, 10, seed=2)
    assert len(train_idx) == 10 and len(val_idx) == 10


def test_split_rejects_invalid_arguments():
    labels = _balanced_labels(n_per_class=3)
    with pytest.raises(ValueError):
        md.split_train_validation(labels, 0, 5, seed=0)
    with pytest.raises(ValueError):
        md.split_train_validation(labels, 5, -1, seed=0)
    with pytest.raises(ValueError, match="seed"):
        md.split_train_validation(labels, 5, 5, seed=-3)
    with pytest.raises(TypeError):
        md.split_train_validation(labels, 5.0, 5, seed=0)
    with pytest.raises(TypeError):
        md.split_train_validation(labels.astype(np.float64), 5, 5, seed=0)


# ---------- 下载器（stub 网络，绝不发真实请求） ----------


class _FakeResponse:
    """替代 urllib 响应：按块返回预置字节流。"""

    def __init__(self, payload: bytes, chunk_size: int = 1 << 20):
        self._data = payload
        self._pos = 0
        self._chunk = chunk_size

    def read(self, size=-1):
        if self._pos >= len(self._data):
            return b""
        end = len(self._data) if size < 0 else min(self._pos + size, len(self._data))
        chunk = self._data[self._pos : end]
        self._pos = end
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(monkeypatch, handler):
    """handler(url) -> _FakeResponse | 抛异常；记录被请求过的 URL。"""
    requested = []

    def fake_urlopen(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        requested.append(url)
        return handler(url)

    monkeypatch.setattr(md, "urlopen", fake_urlopen)
    return requested


def _preplace_other_files(tmp_path, keep_missing):
    """预置除 keep_missing 之外的三个文件（download 跳过已存在文件，不做校验）。"""
    for name in md.MNIST_FILES:
        if name != keep_missing:
            (tmp_path / name).write_bytes(b"placeholder")


def test_download_missing_file_success(tmp_path, monkeypatch):
    # 预置其他三个文件，只允许下载训练图片；stub 按请求 URL 提供对应的有效 gzip
    valid_payloads = {
        "train-images-idx3-ubyte.gz": _idx_images_bytes(4, seed=9),
    }
    for name in md.MNIST_FILES:
        if name in valid_payloads:
            continue
        (tmp_path / name).write_bytes(b"placeholder")

    def handler(url):
        name = url.rsplit("/", 1)[-1]
        return _FakeResponse(gzip.compress(valid_payloads[name]))

    requested = _patch_urlopen(monkeypatch, handler)
    downloaded = md.download_mnist(tmp_path)
    assert downloaded == ["train-images-idx3-ubyte.gz"]
    assert len(requested) == 1  # 只请求缺失文件，且默认源为第一个镜像
    assert requested[0].startswith("https://ossci-datasets.s3.amazonaws.com/mnist/")
    arr = md.load_idx_images(tmp_path / "train-images-idx3-ubyte.gz")
    assert arr.shape == (4, 28, 28)
    assert not list(tmp_path.glob("*.part"))  # 临时文件已原子替换，无残片


def test_download_all_four_files(tmp_path, monkeypatch):
    payloads = {
        "train-images-idx3-ubyte.gz": _idx_images_bytes(4, seed=1),
        "train-labels-idx1-ubyte.gz": _idx_labels_bytes([0, 1, 2, 3]),
        "t10k-images-idx3-ubyte.gz": _idx_images_bytes(2, seed=2),
        "t10k-labels-idx1-ubyte.gz": _idx_labels_bytes([9, 8]),
    }

    def handler(url):
        name = url.rsplit("/", 1)[-1]
        return _FakeResponse(gzip.compress(payloads[name]))

    _patch_urlopen(monkeypatch, handler)
    downloaded = md.download_mnist(tmp_path)
    assert sorted(downloaded) == sorted(md.MNIST_FILES)
    images, labels, t_images, t_labels = md.load_mnist_raw(tmp_path)
    assert len(images) == 4 and len(labels) == 4 and len(t_images) == 2 and len(t_labels) == 2


def test_download_failure_cleans_temp_and_keeps_existing(tmp_path, monkeypatch):
    # 只留 train-images 缺失：下载它失败时，既有三个文件必须原样保留
    _preplace_other_files(tmp_path, keep_missing="train-images-idx3-ubyte.gz")
    existing = tmp_path / "train-labels-idx1-ubyte.gz"
    before = existing.read_bytes()

    def handler(url):
        raise urllib.error.URLError("network unreachable")

    _patch_urlopen(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="train-images-idx3-ubyte.gz"):
        md.download_mnist(tmp_path)
    for name in md.MNIST_FILES:
        if name != "train-images-idx3-ubyte.gz":
            assert (tmp_path / name).read_bytes() == b"placeholder"  # 既有文件未被破坏
    assert not (tmp_path / "train-images-idx3-ubyte.gz").exists()
    assert not list(tmp_path.glob("*.part"))  # 临时残片已删除


def test_download_invalid_content_rejected_no_atomic_replace(tmp_path, monkeypatch):
    dest = tmp_path / "train-labels-idx1-ubyte.gz"
    _preplace_other_files(tmp_path, keep_missing="train-labels-idx1-ubyte.gz")

    def handler(url):
        # 内容不是合法 IDX（magic 错误）：必须在替换前被拦截
        return _FakeResponse(gzip.compress(struct.pack(">II", 1234, 3) + b"\x00" * 3))

    _patch_urlopen(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="magic number"):
        md.download_mnist(tmp_path)
    assert not dest.exists()  # 校验失败 → 不得替换出正式文件
    assert not list(tmp_path.glob("*.part"))


def test_download_falls_back_to_second_mirror(tmp_path, monkeypatch):
    _preplace_other_files(tmp_path, keep_missing="train-labels-idx1-ubyte.gz")
    calls = []

    def handler(url):
        calls.append(url)
        if url.startswith("https://ossci-datasets.s3.amazonaws.com"):
            raise urllib.error.URLError("mirror 1 down")
        name = url.rsplit("/", 1)[-1]
        return _FakeResponse(gzip.compress(_idx_labels_bytes([0, 1, 2])))

    _patch_urlopen(monkeypatch, handler)
    downloaded = md.download_mnist(tmp_path)
    assert downloaded == ["train-labels-idx1-ubyte.gz"]
    assert len(calls) == 2  # 第一个镜像失败后尝试第二个
    assert md.load_idx_labels(tmp_path / "train-labels-idx1-ubyte.gz").tolist() == [0, 1, 2]
