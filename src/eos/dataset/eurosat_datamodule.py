"""Lightning data module backed by the curated EuroSAT manifest and memmap."""

from __future__ import annotations

import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from eos.config import DataConfig
from eos.dataset.manifest import ImageRecord, Manifest, read_manifest
from eos.dataset.transforms import build_eval_transform, build_train_transform


def seed_worker(worker_id: int) -> None:
    """Seed Python, NumPy, and Torch inside a dataloader worker.

    Args:
        worker_id (int): Worker index supplied by PyTorch.

    Returns:
        None: Global RNG state is seeded in place.
    """
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class EuroSATDataset(Dataset[tuple[Tensor, Tensor]]):
    """Map-style dataset over a manifest split and a memmapped image array.

    Args:
        records (list[ImageRecord]): Manifest rows for one split.
        indices (list[int]): Row positions in the curated arrays.
        images_path (Path): Image memmap path.
        labels_path (Path): Label array path.
        train (bool): Whether to use stochastic geometry augmentation.
    """

    def __init__(
        self,
        records: list[ImageRecord],
        indices: list[int],
        images_path: Path,
        labels_path: Path,
        train: bool,
    ) -> None:
        """Load memmaps and choose the transform.

        Args:
            records (list[ImageRecord]): Manifest rows for the split.
            indices (list[int]): Row positions in the curated arrays.
            images_path (Path): Image memmap path.
            labels_path (Path): Label array path.
            train (bool): Whether stochastic augmentation is enabled.

        Returns:
            None: The dataset is initialized in place.
        """
        self.records = records
        self.indices = indices
        self.images = np.load(images_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.transform = build_train_transform() if train else build_eval_transform()

    def __len__(self) -> int:
        """Return the number of images in the split.

        Returns:
            int: Split length.
        """
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return one transformed image and its label.

        Args:
            index (int): Split-local index.

        Returns:
            tuple[Tensor, Tensor]: CHW float32 image in ``[0, 1]`` and int64
                label.
        """
        row = self.indices[index]
        image = np.array(self.images[row], copy=True)
        label = int(self.labels[row])
        tensor = self.transform(image)
        return tensor, torch.tensor(label, dtype=torch.int64)


class EuroSATDataModule(LightningDataModule):
    """LightningDataModule for curated EuroSAT arrays.

    Args:
        config (DataConfig): Dataset and dataloader configuration.
        manifest_path (Path | None): Manifest override.
        metadata_path (Path | None): Manifest metadata override.
    """

    def __init__(
        self,
        config: DataConfig,
        manifest_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> None:
        """Resolve manifest paths and platform-safe worker settings.

        Args:
            config (DataConfig): Dataset and dataloader configuration.
            manifest_path (Path | None): Optional manifest override.
            metadata_path (Path | None): Optional metadata override.

        Returns:
            None: The data module is initialized in place.
        """
        super().__init__()
        self.config = config
        self.curated_root = config.curated_root
        self.manifest_path = manifest_path or self.curated_root / "manifest.parquet"
        self.metadata_path = metadata_path or self.curated_root / "manifest.json"
        self.manifest: Manifest | None = None
        self.train_dataset: EuroSATDataset | None = None
        self.val_dataset: EuroSATDataset | None = None
        self.test_dataset: EuroSATDataset | None = None
        self._class_weights: Tensor | None = None
        self._workers = self._resolve_num_workers()

    @property
    def classes(self) -> tuple[str, ...]:
        """Return class names in identifier order.

        Returns:
            tuple[str, ...]: Sorted class names.

        Raises:
            RuntimeError: If the data module has not been set up.
        """
        if self.manifest is None:
            raise RuntimeError("data module must be set up before reading classes")
        mapping = {record.class_id: record.class_name for record in self.manifest.records}
        return tuple(mapping[index] for index in sorted(mapping))

    @property
    def num_classes(self) -> int:
        """Return the number of classes.

        Returns:
            int: Class count.
        """
        return len(self.classes)

    @property
    def class_weights(self) -> Tensor:
        """Return inverse-frequency class weights from the training split.

        Returns:
            Tensor: Float32 tensor of shape ``(num_classes,)``.

        Raises:
            RuntimeError: If the data module has not been set up.
        """
        if self._class_weights is None:
            raise RuntimeError("data module must be set up before reading class weights")
        return self._class_weights

    def _resolve_num_workers(self) -> int:
        """Resolve a safe worker count for the current platform.

        Returns:
            int: Worker count, zero on constrained local environments.
        """
        if self.config.num_workers == 0:
            return 0
        if os.environ.get("EOS_ALLOW_WORKERS") == "1":
            return self.config.num_workers
        if "google.colab" in sys.modules:
            return 0
        if platform.system() == "Windows":
            return 0
        return self.config.num_workers

    def _check_geometry(self, records: list[ImageRecord]) -> None:
        """Verify the curated image array matches the configured geometry.

        The memmap header is read without touching pixel data, so a stale
        dataset version or a changed ``image_size`` fails at setup time instead
        of producing a shape error deep inside the encoder.

        Args:
            records (list[ImageRecord]): All manifest records.

        Returns:
            None: The check passes silently.

        Raises:
            ValueError: If the curated array geometry differs from
                ``(records, image_size, image_size, channels)``.
        """
        images_path = self.curated_root / "images_uint8.npy"
        stored = np.load(images_path, mmap_mode="r").shape
        expected = (
            len(records),
            self.config.image_size,
            self.config.image_size,
            self.config.channels,
        )
        if stored != expected:
            raise ValueError(
                f"curated images at {images_path} have shape {stored} "
                f"but the configuration requires {expected}"
            )

    def _split_dataset(self, records: list[ImageRecord], split: str, train: bool) -> EuroSATDataset:
        """Build a dataset for one split.

        Args:
            records (list[ImageRecord]): All manifest records.
            split (str): Split name.
            train (bool): Whether augmentation is enabled.

        Returns:
            EuroSATDataset: Split dataset.
        """
        indices = [index for index, record in enumerate(records) if record.split == split]
        split_records = [records[index] for index in indices]
        return EuroSATDataset(
            records=split_records,
            indices=indices,
            images_path=self.curated_root / "images_uint8.npy",
            labels_path=self.curated_root / "labels_int64.npy",
            train=train,
        )

    def setup(self, stage: str | None = None) -> None:
        """Load the manifest and construct split datasets.

        The method is idempotent because the runner reads class weights and
        manifest provenance before handing the module to Lightning, which then
        calls ``setup`` again.

        Args:
            stage (str | None): Lightning stage name.

        Returns:
            None: Datasets are stored on the instance.

        Raises:
            ValueError: If the curated directory does not match the manifest
                dataset version, or if the curated image array geometry does not
                match ``image_size`` and ``channels``.
        """
        del stage
        if self.manifest is not None:
            return
        self.manifest = read_manifest(self.manifest_path, self.metadata_path)
        if self.curated_root.name != self.manifest.metadata.dataset_version:
            raise ValueError(
                f"curated directory {self.curated_root.name} does not match "
                f"manifest version {self.manifest.metadata.dataset_version}"
            )
        records = list(self.manifest.records)
        self._check_geometry(records)
        self.train_dataset = self._split_dataset(records, "train", train=True)
        self.val_dataset = self._split_dataset(records, "val", train=False)
        self.test_dataset = self._split_dataset(records, "test", train=False)
        labels = np.asarray(
            [record.class_id for record in records if record.split == "train"], dtype=np.int64
        )
        counts = np.bincount(labels, minlength=self.num_classes).astype(np.float64)
        weights = np.zeros_like(counts)
        present = counts > 0
        weights[present] = len(labels) / (self.num_classes * counts[present])
        self._class_weights = torch.tensor(weights, dtype=torch.float32)

    def _loader(
        self, dataset: EuroSATDataset | None, shuffle: bool
    ) -> DataLoader[tuple[Tensor, Tensor]]:
        """Build a dataloader with reproducible worker seeding.

        Args:
            dataset (EuroSATDataset | None): Split dataset.
            shuffle (bool): Whether to shuffle.

        Returns:
            DataLoader: Configured dataloader.

        Raises:
            RuntimeError: If the dataset has not been created.
        """
        if dataset is None:
            raise RuntimeError("data module must be set up before creating dataloaders")
        generator = torch.Generator().manual_seed(self.config.seed)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self._workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Return the training dataloader.

        Returns:
            DataLoader: Shuffled training loader.
        """
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Return the validation dataloader.

        Returns:
            DataLoader: Deterministic validation loader.
        """
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Return the test dataloader.

        Returns:
            DataLoader: Deterministic test loader.
        """
        return self._loader(self.test_dataset, shuffle=False)
