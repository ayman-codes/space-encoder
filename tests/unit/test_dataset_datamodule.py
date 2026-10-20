"""Tests for the curated EuroSAT LightningDataModule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from tests.helpers import build_curated_dataset

from eos.config import DataConfig
from eos.dataset.eurosat_datamodule import EuroSATDataModule
from eos.dataset.manifest import read_manifest


def test_datamodule_batches_are_normalized(tmp_path: Path) -> None:
    """Batches must be CHW float32 patches in ``[0, 1]`` with int64 labels."""
    output = build_curated_dataset(tmp_path)
    config = DataConfig(
        curated_root=output,
        batch_size=4,
        num_workers=0,
        image_size=64,
        channels=3,
        num_classes=2,
    )
    module = EuroSATDataModule(config)
    module.setup()
    assert module.classes == ("Alpha", "Beta")
    assert module.num_classes == 2
    assert module.class_weights.shape == (2,)
    images, labels = next(iter(module.train_dataloader()))
    assert images.shape == (4, 3, 64, 64)
    assert images.dtype == torch.float32
    assert float(images.min()) >= 0.0
    assert float(images.max()) <= 1.0
    assert labels.dtype == torch.int64
    val_images, _ = next(iter(module.val_dataloader()))
    assert val_images.shape[1:] == (3, 64, 64)


def test_datamodule_requires_setup() -> None:
    """Reading classes before setup must fail clearly."""
    module = EuroSATDataModule(DataConfig())
    with pytest.raises(RuntimeError):
        _ = module.classes
    with pytest.raises(RuntimeError):
        _ = module.class_weights


def test_curated_rows_stay_aligned_with_manifest(tmp_path: Path) -> None:
    """Every curated row must carry the class of its manifest record.

    Both the array writer and the Parquet writer independently sort by
    ``image_id``. The alignment is correct only because those two orders agree,
    so it is asserted explicitly rather than assumed.
    """
    output = build_curated_dataset(tmp_path)
    manifest = read_manifest(output / "manifest.parquet", output / "manifest.json")
    labels = np.load(output / "labels_int64.npy", mmap_mode="r")
    assert len(manifest.records) == labels.shape[0]
    for row, record in enumerate(manifest.records):
        assert int(labels[row]) == record.class_id


def test_geometry_mismatch_fails_at_setup(tmp_path: Path) -> None:
    """A configuration that disagrees with the curated array must fail early."""
    output = build_curated_dataset(tmp_path)
    config = DataConfig(curated_root=output, image_size=32, num_classes=2)
    with pytest.raises(ValueError, match="requires"):
        EuroSATDataModule(config).setup()
