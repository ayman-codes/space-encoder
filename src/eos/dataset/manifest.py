"""Dataset manifest contracts and Parquet persistence.

A manifest row is the lineage record for exactly one image. The Parquet table
contains only deterministic data; mutable run metadata lives in a separate JSON
document so the logical manifest fingerprint remains stable across rebuilds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from eos.config import FrozenModel
from eos.dataset.contract import DatasetContract

SPLITS: tuple[Literal["train", "val", "test"], ...] = ("train", "val", "test")

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("relative_path", pa.string(), nullable=False),
        pa.field("class_id", pa.int8(), nullable=False),
        pa.field("class_name", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=True),
        pa.field("sha256", pa.string(), nullable=False),
        pa.field("width", pa.int16(), nullable=False),
        pa.field("height", pa.int16(), nullable=False),
        pa.field("channels", pa.int8(), nullable=False),
        pa.field("mode", pa.string(), nullable=False),
        pa.field("file_bytes", pa.int32(), nullable=False),
        pa.field("pixel_min_r", pa.int16(), nullable=False),
        pa.field("pixel_min_g", pa.int16(), nullable=False),
        pa.field("pixel_min_b", pa.int16(), nullable=False),
        pa.field("pixel_max_r", pa.int16(), nullable=False),
        pa.field("pixel_max_g", pa.int16(), nullable=False),
        pa.field("pixel_max_b", pa.int16(), nullable=False),
        pa.field("mean_r", pa.float32(), nullable=False),
        pa.field("mean_g", pa.float32(), nullable=False),
        pa.field("mean_b", pa.float32(), nullable=False),
        pa.field("std_r", pa.float32(), nullable=False),
        pa.field("std_g", pa.float32(), nullable=False),
        pa.field("std_b", pa.float32(), nullable=False),
        pa.field("phash", pa.uint64(), nullable=False),
        pa.field("group_id", pa.string(), nullable=True),
        pa.field("source_member", pa.string(), nullable=False),
        pa.field("source_sha256", pa.string(), nullable=False),
        pa.field("split_seed", pa.int32(), nullable=False),
    ]
)


class ImageRecord(FrozenModel):
    """Lineage and quality record for one source image.

    Args:
        image_id (str): Stable identifier independent of filesystem ordering.
        relative_path (str): Path relative to the extracted dataset root.
        class_id (int): Zero-based class identifier.
        class_name (str): Land-use or land-cover class name.
        split (Literal["train", "val", "test"] | None): Assigned split.
        sha256 (str): Per-file SHA-256 digest.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        channels (int): Number of colour channels.
        mode (str): Pillow image mode, expected to be ``RGB``.
        file_bytes (int): Encoded file size in bytes.
        pixel_min_r (int): Minimum red value.
        pixel_min_g (int): Minimum green value.
        pixel_min_b (int): Minimum blue value.
        pixel_max_r (int): Maximum red value.
        pixel_max_g (int): Maximum green value.
        pixel_max_b (int): Maximum blue value.
        mean_r (float): Mean red value in ``[0, 1]``.
        mean_g (float): Mean green value in ``[0, 1]``.
        mean_b (float): Mean blue value in ``[0, 1]``.
        std_r (float): Red standard deviation in ``[0, 1]``.
        std_g (float): Green standard deviation in ``[0, 1]``.
        std_b (float): Blue standard deviation in ``[0, 1]``.
        phash (int): Unsigned 64-bit perceptual hash.
        group_id (str | None): Near-duplicate group identifier.
        source_member (str): Member path inside the source archive.
        source_sha256 (str): Digest of the immutable source archive.
        split_seed (int): Seed used for split assignment.
    """

    image_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)
    split: Literal["train", "val", "test"] | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    channels: int = Field(ge=1)
    mode: str = Field(min_length=1)
    file_bytes: int = Field(gt=0)
    pixel_min_r: int = Field(ge=0, le=255)
    pixel_min_g: int = Field(ge=0, le=255)
    pixel_min_b: int = Field(ge=0, le=255)
    pixel_max_r: int = Field(ge=0, le=255)
    pixel_max_g: int = Field(ge=0, le=255)
    pixel_max_b: int = Field(ge=0, le=255)
    mean_r: float = Field(ge=0.0, le=1.0)
    mean_g: float = Field(ge=0.0, le=1.0)
    mean_b: float = Field(ge=0.0, le=1.0)
    std_r: float = Field(ge=0.0, le=1.0)
    std_g: float = Field(ge=0.0, le=1.0)
    std_b: float = Field(ge=0.0, le=1.0)
    phash: int = Field(ge=0, le=2**64 - 1)
    group_id: str | None = None
    source_member: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_seed: int = Field(ge=0)


class ManifestMetadata(FrozenModel):
    """Run metadata stored beside the deterministic Parquet manifest.

    Args:
        dataset (str): Dataset name.
        version (str): Dataset version.
        dataset_version (str): Versioned dataset identifier.
        contract_fingerprint (str): Contract digest used for the build.
        manifest_sha256 (str): Logical fingerprint of the manifest rows.
        created_at (str): ISO-8601 creation timestamp.
        git_commit (str): Repository commit that produced the manifest.
        pipeline_version (str): Data pipeline package version.
        run_id (str): Unique build identifier.
        split_seed (int): Split seed.
        split_strategy (str): Human-readable split strategy.
        class_counts (dict[str, int]): Counts per class.
        split_counts (dict[str, dict[str, int]]): Counts per split and class.
        source_archive_sha256 (str): Source archive digest.
    """

    dataset: str
    version: str
    dataset_version: str
    contract_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str
    git_commit: str
    pipeline_version: str
    run_id: str
    split_seed: int
    split_strategy: str
    class_counts: dict[str, int]
    split_counts: dict[str, dict[str, int]]
    source_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Manifest:
    """Loaded manifest with its immutable record list and metadata."""

    records: tuple[ImageRecord, ...]
    metadata: ManifestMetadata


def manifest_hash(records: list[ImageRecord]) -> str:
    """Compute the logical fingerprint of a manifest.

    Args:
        records (list[ImageRecord]): Manifest rows.

    Returns:
        str: SHA-256 digest over sorted lineage fields.
    """
    payload = [
        {
            "image_id": record.image_id,
            "sha256": record.sha256,
            "class_id": record.class_id,
            "split": record.split,
            "group_id": record.group_id,
        }
        for record in sorted(records, key=lambda item: item.image_id)
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def class_counts(records: list[ImageRecord]) -> dict[str, int]:
    """Count records per class.

    Args:
        records (list[ImageRecord]): Manifest rows.

    Returns:
        dict[str, int]: Counts keyed by class name.
    """
    counts: dict[str, int] = {}
    for record in records:
        counts[record.class_name] = counts.get(record.class_name, 0) + 1
    return dict(sorted(counts.items()))


def split_counts(records: list[ImageRecord]) -> dict[str, dict[str, int]]:
    """Count records per split and class.

    Args:
        records (list[ImageRecord]): Manifest rows with assigned splits.

    Returns:
        dict[str, dict[str, int]]: Nested counts keyed by split and class.
    """
    counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    for record in records:
        if record.split is None:
            continue
        per_split = counts[record.split]
        per_split[record.class_name] = per_split.get(record.class_name, 0) + 1
    return {split: dict(sorted(values.items())) for split, values in counts.items()}


def build_metadata(  # noqa: PLR0913, PLR0917
    contract: DatasetContract,
    records: list[ImageRecord],
    run_id: str,
    created_at: str,
    git_commit: str,
    pipeline_version: str,
) -> ManifestMetadata:
    """Build manifest metadata from validated records.

    Args:
        contract (DatasetContract): Dataset contract used for the build.
        records (list[ImageRecord]): Manifest rows.
        run_id (str): Unique build identifier.
        created_at (str): ISO-8601 creation timestamp.
        git_commit (str): Repository commit.
        pipeline_version (str): Pipeline package version.

    Returns:
        ManifestMetadata: Metadata object ready for serialization.
    """
    strategy = "stratified_group" if contract.split.group_aware else "stratified"
    return ManifestMetadata(
        dataset=contract.dataset,
        version=contract.version,
        dataset_version=contract.dataset_version,
        contract_fingerprint=contract.fingerprint(),
        manifest_sha256=manifest_hash(records),
        created_at=created_at,
        git_commit=git_commit,
        pipeline_version=pipeline_version,
        run_id=run_id,
        split_seed=contract.split.seed,
        split_strategy=strategy,
        class_counts=class_counts(records),
        split_counts=split_counts(records),
        source_archive_sha256=contract.source.archive_sha256,
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes through a temporary sibling and atomic replacement.

    Args:
        path (Path): Destination path.
        payload (bytes): File contents.

    Returns:
        None: The destination is replaced atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_manifest(
    records: list[ImageRecord],
    metadata: ManifestMetadata,
    parquet_path: Path,
    metadata_path: Path,
) -> None:
    """Persist a manifest as deterministic Parquet plus JSON metadata.

    Args:
        records (list[ImageRecord]): Manifest rows.
        metadata (ManifestMetadata): Run metadata.
        parquet_path (Path): Destination Parquet path.
        metadata_path (Path): Destination JSON path.

    Returns:
        None: Both files are written atomically.

    Raises:
        ValueError: If the metadata fingerprint does not match the rows.
    """
    ordered = sorted(records, key=lambda item: item.image_id)
    if metadata.manifest_sha256 != manifest_hash(ordered):
        raise ValueError("manifest metadata fingerprint does not match records")
    table = pa.Table.from_pylist([record.model_dump() for record in ordered], schema=PARQUET_SCHEMA)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = parquet_path.with_name(f".{parquet_path.name}.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(parquet_path)
    payload = json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True)
    _atomic_write_bytes(metadata_path, (payload + "\n").encode("utf-8"))


def read_manifest(parquet_path: Path, metadata_path: Path | None = None) -> Manifest:
    """Load a manifest from Parquet and optional JSON metadata.

    Args:
        parquet_path (Path): Parquet manifest path.
        metadata_path (Path | None): JSON metadata path. When omitted, a
            sibling ``manifest.json`` file is used if present.

    Returns:
        Manifest: Loaded records and metadata.

    Raises:
        FileNotFoundError: If the Parquet file or required metadata is missing.
        ValueError: If the stored fingerprint does not match the rows.
    """
    if not parquet_path.is_file():
        raise FileNotFoundError(f"manifest not found: {parquet_path}")
    resolved_metadata = metadata_path or parquet_path.with_name("manifest.json")
    raw = pq.read_table(parquet_path).to_pylist()
    records = tuple(ImageRecord.model_validate(row) for row in raw)
    if not resolved_metadata.is_file():
        raise FileNotFoundError(f"manifest metadata not found: {resolved_metadata}")
    payload = json.loads(resolved_metadata.read_text(encoding="utf-8"))
    metadata = ManifestMetadata.model_validate(payload)
    if metadata.manifest_sha256 != manifest_hash(list(records)):
        raise ValueError("stored manifest fingerprint does not match rows")
    return Manifest(records=records, metadata=metadata)
