"""Synthetic dataset helpers shared by unit and integration tests."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from eos.dataset.contract import DatasetContract, SourceContract
from eos.dataset.curate import curate_dataset
from eos.dataset.manifest import ImageRecord, build_metadata, write_manifest
from eos.dataset.splits import assign_splits
from eos.dataset.validate import scan_images


def write_image(path: Path, seed: int) -> None:
    """Write a deterministic 64x64 RGB JPEG.

    Args:
        path (Path): Destination path.
        seed (int): Random seed controlling the pattern.

    Returns:
        None: The parent directory and image are written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path, format="JPEG", quality=92)


def build_zip(root: Path, archive: Path, class_names: tuple[str, ...], per_class: int) -> None:
    """Build a EuroSAT-shaped zip archive from class folders.

    Args:
        root (Path): Directory containing class folders.
        archive (Path): Destination archive.
        class_names (tuple[str, ...]): Class folder names.
        per_class (int): Images per class.

    Returns:
        None: Images and the archive are written.
    """
    for class_index, class_name in enumerate(class_names):
        for image_index in range(per_class):
            write_image(
                root / class_name / f"{class_name}_{image_index}.jpg",
                seed=class_index * 1000 + image_index,
            )
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        bundle.write(root, arcname="EuroSAT_RGB")
        for class_name in class_names:
            for path in sorted((root / class_name).glob("*.jpg")):
                bundle.write(path, arcname=f"EuroSAT_RGB/{class_name}/{path.name}")


def build_contract(
    class_names: tuple[str, ...],
    expected_images: int,
    archive: Path | None = None,
    archive_sha256: str | None = None,
    archive_bytes: int | None = None,
) -> DatasetContract:
    """Create a synthetic dataset contract.

    Args:
        class_names (tuple[str, ...]): Sorted class names.
        expected_images (int): Expected image count.
        archive (Path | None): Optional archive used to derive size and digest.
        archive_sha256 (str | None): Explicit digest override.
        archive_bytes (int | None): Explicit size override.

    Returns:
        DatasetContract: Synthetic contract.
    """
    digest = archive_sha256 or "0" * 64
    size = archive_bytes or 1
    if archive is not None:
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        size = archive.stat().st_size
    source = SourceContract(
        urls=("https://example.invalid/EuroSAT_RGB.zip",),
        archive_sha256=digest,
        archive_bytes=size,
        expected_images=expected_images,
        expected_classes=len(class_names),
        image_size=64,
        channels=3,
        class_names=class_names,
        license="test",
        citation="test",
    )
    return DatasetContract(dataset="synthetic", version="1.0.0", source=source)


def build_curated_dataset(
    root: Path,
    class_names: tuple[str, ...] = ("Alpha", "Beta"),
    per_class: int = 12,
) -> Path:
    """Build a small complete curated dataset for datamodule and training tests.

    The helper runs the same archive, validate, split, curate, and manifest
    writers as the production pipeline, so a training test exercises the real
    manifest-versus-memmap contract rather than a stub.

    Args:
        root (Path): Temporary directory that receives the raw tree and curated
            output.
        class_names (tuple[str, ...]): Class folder names in identifier order.
        per_class (int): Images written per class.

    Returns:
        Path: Curated dataset directory named after the contract version.
    """
    raw = root / "raw"
    archive = root / "EuroSAT_RGB.zip"
    build_zip(raw, archive, class_names, per_class)
    contract = build_contract(class_names, len(class_names) * per_class, archive=archive)
    records, _ = scan_images(raw, contract)
    records = assign_splits(records, contract)
    output = root / contract.dataset_version
    curate_dataset(records, raw, output, contract)
    metadata = build_metadata(
        contract,
        records,
        "synthetic-run",
        "2026-01-01T00:00:00Z",
        "0" * 40,
        "0.1.0",
    )
    write_manifest(records, metadata, output / "manifest.parquet", output / "manifest.json")
    return output


def make_record(  # noqa: PLR0913, PLR0917
    image_id: str,
    class_id: int,
    class_name: str,
    split: str | None = None,
    phash: int | None = None,
    sha256: str | None = None,
    group_id: str | None = None,
    source_sha256: str | None = None,
) -> ImageRecord:
    """Create a valid manifest record for split and grouping tests.

    Args:
        image_id (str): Stable identifier.
        class_id (int): Class index.
        class_name (str): Class name.
        split (str | None): Optional split assignment.
        phash (int | None): Optional perceptual hash.
        sha256 (str | None): Optional content digest.
        group_id (str | None): Optional duplicate group.
        source_sha256 (str | None): Optional archive digest.

    Returns:
        ImageRecord: Manifest record.
    """
    digest = sha256 or hashlib.sha256(image_id.encode("utf-8")).hexdigest()
    source_digest = source_sha256 or "0" * 64
    hash_value = phash if phash is not None else int.from_bytes(digest.encode("utf-8")[:8], "big")
    return ImageRecord(
        image_id=image_id,
        relative_path=f"{class_name}/{image_id}.jpg",
        class_id=class_id,
        class_name=class_name,
        split=split,
        sha256=digest,
        width=64,
        height=64,
        channels=3,
        mode="RGB",
        file_bytes=1024,
        pixel_min_r=0,
        pixel_min_g=0,
        pixel_min_b=0,
        pixel_max_r=255,
        pixel_max_g=255,
        pixel_max_b=255,
        mean_r=0.5,
        mean_g=0.5,
        mean_b=0.5,
        std_r=0.2,
        std_g=0.2,
        std_b=0.2,
        phash=hash_value,
        group_id=group_id,
        source_member=f"{class_name}/{image_id}.jpg",
        source_sha256=source_digest,
        split_seed=42,
    )


def zip_bytes(root: Path, class_names: tuple[str, ...], per_class: int) -> bytes:
    """Create an in-memory EuroSAT-shaped zip payload.

    Args:
        root (Path): Directory containing class folders.
        class_names (tuple[str, ...]): Class folder names.
        per_class (int): Images per class.

    Returns:
        bytes: Zip payload.
    """
    for class_index, class_name in enumerate(class_names):
        for image_index in range(per_class):
            write_image(
                root / class_name / f"{class_name}_{image_index}.jpg",
                seed=class_index * 1000 + image_index,
            )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as bundle:
        for class_name in class_names:
            for path in sorted((root / class_name).glob("*.jpg")):
                bundle.write(path, arcname=f"EuroSAT_RGB/{class_name}/{path.name}")
    return buffer.getvalue()
