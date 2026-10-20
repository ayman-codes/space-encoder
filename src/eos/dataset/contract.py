"""Dataset contracts for the EuroSAT ingestion and curation pipeline.

The contract is the immutable description of what the source dataset must be.
Operational settings such as retries and S3 configuration live in the pipeline
configuration, so a contract change is always a deliberate dataset change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, model_validator

from eos.config import FrozenModel


class SourceContract(FrozenModel):
    """Expected properties of the immutable EuroSAT source archive.

    Args:
        urls (tuple[str, ...]): Ordered fallback download URLs.
        archive_sha256 (str): Lowercase SHA-256 digest of the archive.
        archive_bytes (int): Exact archive size in bytes.
        expected_images (int): Expected number of JPEG members.
        expected_classes (int): Expected number of land-cover classes.
        image_size (int): Expected square patch edge length.
        channels (int): Expected number of colour channels.
        class_names (tuple[str, ...]): Sorted class folder names. Position
            defines the stable class identifier used by the model.
        license (str): License statement copied from the source record.
        citation (str): Citation text stored in the dataset card.
    """

    urls: tuple[str, ...] = Field(min_length=1)
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_bytes: int = Field(gt=0)
    expected_images: int = Field(gt=0)
    expected_classes: int = Field(gt=1)
    image_size: int = Field(ge=16, le=512)
    channels: int = Field(ge=1, le=13)
    class_names: tuple[str, ...] = Field(min_length=1)
    license: str = Field(min_length=1)
    citation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_classes(self) -> Self:
        """Validate class count, uniqueness, and deterministic ordering.

        Returns:
            Self: The validated source contract.

        Raises:
            ValueError: If class names are duplicated, unsorted, or do not
                match the declared class count.
        """
        if len(self.class_names) != self.expected_classes:
            raise ValueError("class_names length must equal expected_classes")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be unique")
        if tuple(sorted(self.class_names)) != self.class_names:
            raise ValueError("class_names must be sorted alphabetically")
        return self


class SplitContract(FrozenModel):
    """Deterministic stratified split policy.

    Args:
        train_fraction (float): Training share of each class.
        val_fraction (float): Validation share of each class.
        test_fraction (float): Test share of each class.
        seed (int): Split seed recorded in every manifest row.
        group_aware (bool): Whether near-duplicate groups stay in one split.
        phash_distance (int): Maximum Hamming distance for near-duplicate
            grouping.

    Raises:
        ValueError: If the fractions do not sum to one.
    """

    train_fraction: float = Field(default=0.70, gt=0.0, lt=1.0)
    val_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    seed: int = Field(default=42, ge=0)
    group_aware: bool = True
    phash_distance: int = Field(default=6, ge=0, le=64)

    @model_validator(mode="after")
    def _check_fractions(self) -> Self:
        """Ensure the split fractions form a complete partition.

        Returns:
            Self: The validated split contract.

        Raises:
            ValueError: If the fractions do not sum to one within tolerance.
        """
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to exactly 1.0")
        return self


class ValidationContract(FrozenModel):
    """Blocking and warning thresholds for dataset validation.

    Args:
        max_corrupt_images (int): Maximum accepted undecodable images.
        max_exact_duplicates (int): Maximum accepted byte-identical files.
        max_cross_class_duplicate_groups (int): Warning threshold for
            perceptual duplicate groups spanning different class labels.
        warn_near_duplicate_fraction (float): Warning threshold for the share
            of files in perceptual duplicate groups.
        min_image_variance (float): Minimum per-image pixel variance used to
            detect blank or constant patches.
    """

    max_corrupt_images: int = Field(default=0, ge=0)
    max_exact_duplicates: int = Field(default=0, ge=0)
    max_cross_class_duplicate_groups: int = Field(default=25, ge=0)
    warn_near_duplicate_fraction: float = Field(default=0.02, ge=0.0, le=1.0)
    min_image_variance: float = Field(default=1e-4, gt=0.0)


class DatasetContract(FrozenModel):
    """Complete immutable contract for a versioned dataset.

    Args:
        dataset (str): Stable dataset name.
        version (str): Semantic dataset version without a ``v`` prefix.
        source (SourceContract): Source archive expectations.
        split (SplitContract): Split policy.
        validation (ValidationContract): Validation thresholds.
    """

    dataset: str = Field(default="eurosat-rgb", min_length=1)
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    source: SourceContract
    split: SplitContract = Field(default_factory=SplitContract)
    validation: ValidationContract = Field(default_factory=ValidationContract)

    @property
    def class_ids(self) -> dict[str, int]:
        """Return the stable class-to-identifier mapping.

        Returns:
            dict[str, int]: Mapping from sorted class name to zero-based index.
        """
        return {name: index for index, name in enumerate(self.source.class_names)}

    @property
    def dataset_version(self) -> str:
        """Return the versioned dataset identifier.

        Returns:
            str: Identifier such as ``eurosat-rgb-v1.0.0``.
        """
        return f"{self.dataset}-v{self.version}"

    def fingerprint(self) -> str:
        """Compute a deterministic contract fingerprint.

        Returns:
            str: Lowercase SHA-256 digest of canonical contract JSON.
        """
        payload = self.model_dump(mode="json", exclude_none=True)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_contract(path: Path) -> DatasetContract:
    """Load and validate a dataset contract from YAML.

    Args:
        path (Path): Contract file path.

    Returns:
        DatasetContract: Validated immutable contract.

    Raises:
        FileNotFoundError: If the contract file does not exist.
        ValueError: If the YAML root is not a mapping.
        pydantic.ValidationError: If any contract field is invalid.
    """
    if not path.is_file():
        raise FileNotFoundError(f"dataset contract not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset contract root must be a mapping")
    return DatasetContract.model_validate(payload)
