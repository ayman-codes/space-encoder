"""Curated array construction and dataset profiling."""

from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict

from eos.dataset.contract import DatasetContract
from eos.dataset.manifest import SPLITS, ImageRecord, manifest_hash


class ClassStats(BaseModel):
    """Pixel statistics for one class.

    Args:
        count (int): Number of images.
        mean_r (float): Mean red value in ``[0, 1]``.
        mean_g (float): Mean green value.
        mean_b (float): Mean blue value.
        std_r (float): Red standard deviation.
        std_g (float): Green standard deviation.
        std_b (float): Blue standard deviation.
    """

    model_config = ConfigDict(frozen=True)

    count: int
    mean_r: float
    mean_g: float
    mean_b: float
    std_r: float
    std_g: float
    std_b: float


class DatasetStats(BaseModel):
    """Profile evidence for a curated dataset.

    Args:
        dataset_version (str): Versioned dataset identifier.
        manifest_sha256 (str): Logical manifest fingerprint.
        records (int): Number of images.
        pixel_mean (tuple[float, float, float]): Global channel means.
        pixel_std (tuple[float, float, float]): Global channel deviations.
        class_counts (dict[str, int]): Counts per class.
        split_counts (dict[str, dict[str, int]]): Counts per split and class.
        per_class (dict[str, ClassStats]): Per-class statistics.
    """

    model_config = ConfigDict(frozen=True)

    dataset_version: str
    manifest_sha256: str
    records: int
    pixel_mean: tuple[float, float, float]
    pixel_std: tuple[float, float, float]
    class_counts: dict[str, int]
    split_counts: dict[str, dict[str, int]]
    per_class: dict[str, ClassStats]


class _Accumulator:
    """Online channel statistics accumulator."""

    def __init__(self) -> None:
        """Create an empty accumulator.

        Returns:
            None: Counters and moments are initialized in place.
        """
        self.count = 0
        self.total = np.zeros(3, dtype=np.float64)
        self.squared = np.zeros(3, dtype=np.float64)

    def add(self, values: np.ndarray) -> None:
        """Add an image's normalized pixels.

        Args:
            values (np.ndarray): Array of shape ``(height, width, 3)`` in
                ``[0, 1]``.

        Returns:
            None: The accumulator is updated in place.
        """
        flat = values.reshape(-1, 3)
        self.count += flat.shape[0]
        self.total += flat.sum(axis=0)
        self.squared += np.square(flat).sum(axis=0)

    def mean(self) -> np.ndarray:
        """Return channel means.

        Returns:
            np.ndarray: Three channel means.
        """
        return self.total / max(self.count, 1)

    def std(self) -> np.ndarray:
        """Return channel standard deviations.

        Returns:
            np.ndarray: Three channel deviations.
        """
        mean = self.mean()
        variance = np.maximum(self.squared / max(self.count, 1) - np.square(mean), 0.0)
        return np.sqrt(variance)


def _save_array(path: Path, array: np.ndarray) -> None:
    """Save a NumPy array atomically.

    Args:
        path (Path): Destination path.
        array (np.ndarray): Array to save.

    Returns:
        None: The destination is replaced atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    """Compute a file digest with streaming reads.

    Args:
        path (Path): File to hash.

    Returns:
        str: Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(path: Path, files: list[Path]) -> None:
    """Write a SHA-256 checksum manifest.

    Args:
        path (Path): Destination checksum file.
        files (list[Path]): Files to include.

    Returns:
        None: The checksum file is written atomically.
    """
    lines = [f"{_sha256_file(item)}  {item.name}" for item in files]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def curate_dataset(
    records: list[ImageRecord],
    raw_root: Path,
    output_root: Path,
    contract: DatasetContract,
) -> DatasetStats:
    """Build the image memmap, labels, split indices, and statistics.

    Args:
        records (list[ImageRecord]): Manifest rows with assigned splits.
        raw_root (Path): Extracted dataset root.
        output_root (Path): Curated artifact directory.
        contract (DatasetContract): Dataset contract.

    Returns:
        DatasetStats: Profile evidence for the curated artifacts.

    Raises:
        ValueError: If an image is missing or geometry differs from the
            contract.
    """
    ordered = sorted(records, key=lambda item: item.image_id)
    size = contract.source.image_size
    channels = contract.source.channels
    images = np.empty((len(ordered), size, size, channels), dtype=np.uint8)
    labels = np.empty(len(ordered), dtype=np.int64)
    global_acc = _Accumulator()
    per_class_acc: dict[str, _Accumulator] = defaultdict(_Accumulator)
    class_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, dict[str, int]] = {split: defaultdict(int) for split in SPLITS}

    for index, record in enumerate(ordered):
        path = raw_root / record.relative_path
        if not path.is_file():
            raise ValueError(f"missing image: {path}")
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape != (size, size, channels):
            raise ValueError(f"unexpected geometry for {record.relative_path}: {rgb.shape}")
        images[index] = rgb
        labels[index] = record.class_id
        values = rgb.astype(np.float64) / 255.0
        global_acc.add(values)
        per_class_acc[record.class_name].add(values)
        class_counts[record.class_name] += 1
        if record.split is not None:
            split_counts[record.split][record.class_name] += 1

    output_root.mkdir(parents=True, exist_ok=True)
    images_path = output_root / "images_uint8.npy"
    labels_path = output_root / "labels_int64.npy"
    _save_array(images_path, images)
    _save_array(labels_path, labels)
    for split in SPLITS:
        indices = np.asarray(
            [index for index, record in enumerate(ordered) if record.split == split],
            dtype=np.int64,
        )
        _save_array(output_root / f"{split}_indices.npy", indices)

    mean = global_acc.mean()
    std = global_acc.std()
    class_stats = {
        class_name: ClassStats(
            count=per_class_acc[class_name].count // (size * size),
            mean_r=float(per_class_acc[class_name].mean()[0]),
            mean_g=float(per_class_acc[class_name].mean()[1]),
            mean_b=float(per_class_acc[class_name].mean()[2]),
            std_r=float(per_class_acc[class_name].std()[0]),
            std_g=float(per_class_acc[class_name].std()[1]),
            std_b=float(per_class_acc[class_name].std()[2]),
        )
        for class_name in contract.source.class_names
    }
    stats = DatasetStats(
        dataset_version=contract.dataset_version,
        manifest_sha256=manifest_hash(ordered),
        records=len(ordered),
        pixel_mean=(float(mean[0]), float(mean[1]), float(mean[2])),
        pixel_std=(float(std[0]), float(std[1]), float(std[2])),
        class_counts=dict(sorted(class_counts.items())),
        split_counts={
            split: dict(sorted(values.items())) for split, values in split_counts.items()
        },
        per_class=class_stats,
    )
    stats_path = output_root / "stats.json"
    temporary = stats_path.with_name(f".{stats_path.name}.tmp")
    temporary.write_text(
        json.dumps(stats.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(stats_path)
    write_checksums(
        output_root / "checksums.sha256",
        [
            images_path,
            labels_path,
            *(output_root / f"{split}_indices.npy" for split in SPLITS),
            stats_path,
        ],
    )
    return stats


def verify_arrays(
    records: list[ImageRecord],
    output_root: Path,
) -> None:
    """Verify array shapes, dtypes, and row alignment without loading pixels.

    Args:
        records (list[ImageRecord]): Manifest rows.
        output_root (Path): Curated artifact directory.

    Returns:
        None: Verification passes silently.

    Raises:
        ValueError: If the arrays do not match the manifest.
    """
    images = np.load(output_root / "images_uint8.npy", mmap_mode="r")
    labels = np.load(output_root / "labels_int64.npy", mmap_mode="r")
    if images.shape[0] != len(records):
        raise ValueError("image array row count does not match manifest")
    if labels.shape != (len(records),):
        raise ValueError("label array shape does not match manifest")
    expected = np.asarray([record.class_id for record in records], dtype=np.int64)
    if not np.array_equal(np.asarray(labels), expected):
        raise ValueError("label array is not aligned with manifest order")
