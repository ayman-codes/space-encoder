"""Validation, quarantine, and evidence generation for the EuroSAT pipeline."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Literal

import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict

from eos.dataset.contract import DatasetContract
from eos.dataset.duplicates import (
    DuplicateReport,
    assign_duplicate_groups,
    perceptual_hash,
)
from eos.dataset.manifest import ImageRecord

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLIT_RATIO_TOLERANCE = 0.01


class ValidationCheck(BaseModel):
    """One named validation result with severity and evidence.

    Args:
        name (str): Stable check identifier.
        severity (Literal["critical", "warning", "info"]): Blocking level.
        passed (bool): Whether the check succeeded.
        detail (str): Human-readable evidence.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    severity: Literal["critical", "warning", "info"]
    passed: bool
    detail: str


class QuarantineEntry(BaseModel):
    """A rejected image and the reason it was not published.

    Args:
        relative_path (str): Path relative to the extracted dataset root.
        reason (str): Failure reason.
    """

    model_config = ConfigDict(frozen=True)

    relative_path: str
    reason: str


class ValidationReport(BaseModel):
    """Complete validation evidence for one pipeline run.

    Args:
        dataset_version (str): Versioned dataset identifier.
        contract_fingerprint (str): Contract digest.
        run_id (str): Build identifier.
        created_at (str): ISO-8601 creation timestamp.
        status (Literal["passed", "failed"]): Aggregate status.
        checks (tuple[ValidationCheck, ...]): Individual checks.
        duplicate_report (DuplicateReport | None): Duplicate evidence.
        quarantined (tuple[QuarantineEntry, ...]): Rejected files.
    """

    model_config = ConfigDict(frozen=True)

    dataset_version: str
    contract_fingerprint: str
    run_id: str
    created_at: str
    status: Literal["passed", "failed"]
    checks: tuple[ValidationCheck, ...]
    duplicate_report: DuplicateReport | None = None
    quarantined: tuple[QuarantineEntry, ...] = ()


class ArrayReport(BaseModel):
    """Shape, dtype, and range evidence for curated arrays.

    Args:
        images_shape (tuple[int, ...]): Shape of the image memmap.
        labels_shape (tuple[int, ...]): Shape of the label array.
        images_dtype (str): Image dtype name.
        labels_dtype (str): Label dtype name.
        pixel_min (int): Minimum pixel value.
        pixel_max (int): Maximum pixel value.
        records (int): Number of manifest rows.
    """

    model_config = ConfigDict(frozen=True)

    images_shape: tuple[int, ...]
    labels_shape: tuple[int, ...]
    images_dtype: str
    labels_dtype: str
    pixel_min: int
    pixel_max: int
    records: int


def _check(
    name: str,
    severity: Literal["critical", "warning", "info"],
    passed: bool,
    detail: str,
) -> ValidationCheck:
    """Create a validation check.

    Args:
        name (str): Check identifier.
        severity (Literal["critical", "warning", "info"]): Blocking level.
        passed (bool): Whether the check succeeded.
        detail (str): Evidence text.

    Returns:
        ValidationCheck: Immutable check result.
    """
    return ValidationCheck(name=name, severity=severity, passed=passed, detail=detail)


def validate_archive(archive: Path, contract: DatasetContract) -> list[ValidationCheck]:
    """Validate the source archive before extraction.

    Args:
        archive (Path): Archive file.
        contract (DatasetContract): Dataset contract.

    Returns:
        list[ValidationCheck]: Archive checks.
    """
    checks: list[ValidationCheck] = []
    exists = archive.is_file()
    checks.append(_check("archive.exists", "critical", exists, str(archive)))
    if not exists:
        return checks
    size = archive.stat().st_size
    checks.append(
        _check(
            "archive.bytes",
            "critical",
            size == contract.source.archive_bytes,
            f"expected {contract.source.archive_bytes}, found {size}",
        )
    )
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    actual_sha = digest.hexdigest()
    checks.append(
        _check(
            "archive.sha256",
            "critical",
            actual_sha == contract.source.archive_sha256,
            f"expected {contract.source.archive_sha256}, found {actual_sha}",
        )
    )
    try:
        with zipfile.ZipFile(archive) as bundle:
            bad_member = bundle.testzip()
            checks.append(
                _check(
                    "archive.crc",
                    "critical",
                    bad_member is None,
                    "all members pass CRC" if bad_member is None else f"bad member {bad_member}",
                )
            )
            members = [info for info in bundle.infolist() if not info.is_dir()]
            unsafe = [
                info.filename
                for info in members
                if PurePosixPath(info.filename).is_absolute()
                or ".." in PurePosixPath(info.filename).parts
            ]
            checks.append(
                _check(
                    "archive.paths",
                    "critical",
                    not unsafe,
                    "paths are relative" if not unsafe else ", ".join(unsafe[:5]),
                )
            )
            images = [
                info
                for info in members
                if PurePosixPath(info.filename).suffix.lower() in SUPPORTED_SUFFIXES
            ]
            checks.append(
                _check(
                    "archive.image_count",
                    "critical",
                    len(images) == contract.source.expected_images,
                    f"expected {contract.source.expected_images}, found {len(images)}",
                )
            )
            classes = {
                PurePosixPath(info.filename).parts[-2]
                for info in images
                if len(PurePosixPath(info.filename).parts) >= 2
            }
            expected_classes = set(contract.source.class_names)
            missing_classes = sorted(expected_classes - classes)
            extra_classes = sorted(classes - expected_classes)
            checks.append(
                _check(
                    "archive.classes",
                    "critical",
                    classes == expected_classes,
                    f"missing={missing_classes}, extra={extra_classes}",
                )
            )
    except (zipfile.BadZipFile, OSError) as error:
        checks.append(_check("archive.readable", "critical", False, str(error)))
    return checks


def _read_image_record(
    path: Path,
    root: Path,
    contract: DatasetContract,
    class_name: str,
    class_id: int,
) -> ImageRecord:
    """Decode and profile one image.

    Args:
        path (Path): Image path.
        root (Path): Extracted dataset root.
        contract (DatasetContract): Dataset contract.
        class_name (str): Class folder name.
        class_id (int): Zero-based class identifier.

    Returns:
        ImageRecord: Profiled record.

    Raises:
        UnidentifiedImageError: If Pillow cannot decode the file.
        OSError: If the image cannot be loaded.
        ValueError: If the geometry or mode violates the contract.
    """
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        width, height = image.size
        mode = image.mode
        if (width, height) != (contract.source.image_size, contract.source.image_size):
            raise ValueError(f"unexpected geometry {(width, height)}")
        if mode != "RGB":
            raise ValueError(f"unexpected mode {mode}")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        phash = perceptual_hash(image)
    flat = rgb.reshape(-1, 3)
    values = flat.astype(np.float64) / 255.0
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    pixel_min = flat.min(axis=0)
    pixel_max = flat.max(axis=0)
    relative_path = path.relative_to(root).as_posix()
    return ImageRecord(
        image_id=f"{class_name}/{path.stem}",
        relative_path=relative_path,
        class_id=class_id,
        class_name=class_name,
        split=None,
        sha256=hashlib.sha256(data).hexdigest(),
        width=width,
        height=height,
        channels=int(rgb.shape[2]),
        mode=mode,
        file_bytes=len(data),
        pixel_min_r=int(pixel_min[0]),
        pixel_min_g=int(pixel_min[1]),
        pixel_min_b=int(pixel_min[2]),
        pixel_max_r=int(pixel_max[0]),
        pixel_max_g=int(pixel_max[1]),
        pixel_max_b=int(pixel_max[2]),
        mean_r=float(mean[0]),
        mean_g=float(mean[1]),
        mean_b=float(mean[2]),
        std_r=float(std[0]),
        std_g=float(std[1]),
        std_b=float(std[2]),
        phash=phash,
        group_id=None,
        source_member=relative_path,
        source_sha256=contract.source.archive_sha256,
        split_seed=contract.split.seed,
    )


def scan_images(
    root: Path,
    contract: DatasetContract,
) -> tuple[list[ImageRecord], list[QuarantineEntry]]:
    """Decode and profile every source image, collecting rejects.

    Args:
        root (Path): Directory containing class folders.
        contract (DatasetContract): Dataset contract.

    Returns:
        tuple[list[ImageRecord], list[QuarantineEntry]]: Accepted records and
            rejected files with reasons.
    """
    records: list[ImageRecord] = []
    quarantined: list[QuarantineEntry] = []
    class_ids = contract.class_ids
    for class_name in contract.source.class_names:
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            try:
                record = _read_image_record(path, root, contract, class_name, class_ids[class_name])
            except (UnidentifiedImageError, OSError, ValueError) as error:
                quarantined.append(
                    QuarantineEntry(
                        relative_path=path.relative_to(root).as_posix(),
                        reason=type(error).__name__,
                    )
                )
                continue
            channel_std = max(record.std_r, record.std_g, record.std_b)
            if channel_std < contract.validation.min_image_variance:
                quarantined.append(
                    QuarantineEntry(
                        relative_path=record.relative_path,
                        reason="blank_or_constant_image",
                    )
                )
                continue
            records.append(record)
    return records, quarantined


def quarantine_files(
    root: Path,
    entries: list[QuarantineEntry],
    quarantine_root: Path,
    run_id: str,
) -> Path:
    """Copy rejected files into a run-scoped quarantine directory.

    Args:
        root (Path): Extracted dataset root.
        entries (list[QuarantineEntry]): Rejected files and reasons.
        quarantine_root (Path): Quarantine base directory.
        run_id (str): Build identifier.

    Returns:
        Path: Quarantine directory for this run.
    """
    destination = quarantine_root / run_id
    destination.mkdir(parents=True, exist_ok=True)
    reasons: list[dict[str, str]] = []
    for entry in entries:
        source = root / entry.relative_path
        target = destination / entry.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            shutil.copy2(source, target)
        reasons.append({"relative_path": entry.relative_path, "reason": entry.reason})
    payload = "\n".join(json.dumps(item, sort_keys=True) for item in reasons)
    reasons_path = destination / "reasons.jsonl"
    reasons_path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    return destination


def split_targets(count: int, contract: DatasetContract) -> dict[str, int]:
    """Compute exact per-class split targets.

    Args:
        count (int): Number of records in a class.
        contract (DatasetContract): Dataset contract.

    Returns:
        dict[str, int]: Target counts for train, validation, and test.
    """
    train = int(count * contract.split.train_fraction)
    val = int(count * contract.split.val_fraction)
    return {"train": train, "val": val, "test": count - train - val}


def validate_manifest(
    records: list[ImageRecord],
    contract: DatasetContract,
) -> list[ValidationCheck]:
    """Validate manifest counts, geometry, splits, and group isolation.

    Args:
        records (list[ImageRecord]): Manifest rows.
        contract (DatasetContract): Dataset contract.

    Returns:
        list[ValidationCheck]: Manifest checks.
    """
    checks: list[ValidationCheck] = []
    image_ids = [record.image_id for record in records]
    paths = [record.relative_path for record in records]
    checks.append(
        _check(
            "manifest.image_count",
            "critical",
            len(records) == contract.source.expected_images,
            f"expected {contract.source.expected_images}, found {len(records)}",
        )
    )
    checks.append(
        _check(
            "manifest.image_ids_unique",
            "critical",
            len(set(image_ids)) == len(image_ids),
            f"{len(image_ids) - len(set(image_ids))} duplicate identifiers",
        )
    )
    checks.append(
        _check(
            "manifest.paths_unique",
            "critical",
            len(set(paths)) == len(paths),
            f"{len(paths) - len(set(paths))} duplicate paths",
        )
    )
    expected_classes = set(contract.source.class_names)
    actual_classes = {record.class_name for record in records}
    missing_classes = sorted(expected_classes - actual_classes)
    extra_classes = sorted(actual_classes - expected_classes)
    checks.append(
        _check(
            "manifest.classes",
            "critical",
            actual_classes == expected_classes,
            f"missing={missing_classes}, extra={extra_classes}",
        )
    )
    geometry_ok = all(
        record.width == contract.source.image_size
        and record.height == contract.source.image_size
        and record.channels == contract.source.channels
        and record.mode == "RGB"
        for record in records
    )
    checks.append(
        _check(
            "manifest.geometry",
            "critical",
            geometry_ok,
            f"expected {contract.source.image_size}px RGB",
        )
    )
    split_ok = all(record.split in {"train", "val", "test"} for record in records)
    checks.append(_check("manifest.splits", "critical", split_ok, "every record has a known split"))
    group_splits: dict[str, set[str]] = {}
    for record in records:
        if record.group_id is None or record.split is None:
            continue
        group_splits.setdefault(record.group_id, set()).add(record.split)
    leaking = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    checks.append(
        _check(
            "manifest.group_isolation",
            "critical",
            not leaking,
            f"{len(leaking)} groups span multiple splits",
        )
    )
    counts: dict[str, int] = {}
    split_totals: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for record in records:
        counts[record.class_name] = counts.get(record.class_name, 0) + 1
        if record.split is not None:
            split_totals[record.split] += 1
    total = len(records)
    ratios = {split: value / total if total else 0.0 for split, value in split_totals.items()}
    targets = {
        "train": contract.split.train_fraction,
        "val": contract.split.val_fraction,
        "test": contract.split.test_fraction,
    }
    ratio_ok = all(
        abs(ratios[split] - targets[split]) <= SPLIT_RATIO_TOLERANCE for split in targets
    )
    checks.append(
        _check(
            "manifest.split_ratios",
            "critical",
            ratio_ok,
            ", ".join(f"{split}={ratios[split]:.4f}" for split in targets),
        )
    )
    class_split_ok = True
    details: list[str] = []
    for class_name in contract.source.class_names:
        class_records = [record for record in records if record.class_name == class_name]
        expected = split_targets(len(class_records), contract)
        actual = {
            split: sum(1 for record in class_records if record.split == split) for split in expected
        }
        if any(abs(actual[split] - expected[split]) > 1 for split in expected):
            class_split_ok = False
            details.append(f"{class_name}={actual}")
    checks.append(
        _check(
            "manifest.class_split_balance",
            "critical",
            class_split_ok,
            "; ".join(details) if details else "within one image of target",
        )
    )
    return checks


def validate_arrays(
    records: list[ImageRecord],
    images_path: Path,
    labels_path: Path,
    contract: DatasetContract,
) -> ArrayReport:
    """Validate curated array shape, dtype, and value range.

    Args:
        records (list[ImageRecord]): Manifest rows in array order.
        images_path (Path): Image memmap path.
        labels_path (Path): Label array path.
        contract (DatasetContract): Dataset contract.

    Returns:
        ArrayReport: Array evidence.

    Raises:
        ValueError: If shapes, dtypes, labels, or values violate the contract.
    """
    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    expected_shape = (
        len(records),
        contract.source.image_size,
        contract.source.image_size,
        contract.source.channels,
    )
    if tuple(images.shape) != expected_shape:
        raise ValueError(f"image array shape {images.shape} does not match {expected_shape}")
    if images.dtype != np.uint8:
        raise ValueError(f"image array dtype {images.dtype} is not uint8")
    if tuple(labels.shape) != (len(records),):
        raise ValueError(f"label array shape {labels.shape} is invalid")
    if labels.dtype != np.int64:
        raise ValueError(f"label array dtype {labels.dtype} is not int64")
    expected_labels = np.asarray([record.class_id for record in records], dtype=np.int64)
    if not np.array_equal(np.asarray(labels), expected_labels):
        raise ValueError("label array does not align with manifest order")
    pixel_min = 255
    pixel_max = 0
    for start in range(0, len(records), 1024):
        chunk = np.asarray(images[start : start + 1024])
        pixel_min = min(pixel_min, int(chunk.min()))
        pixel_max = max(pixel_max, int(chunk.max()))
    if pixel_min < 0 or pixel_max > 255:
        raise ValueError("image values fall outside [0, 255]")
    return ArrayReport(
        images_shape=tuple(images.shape),
        labels_shape=tuple(labels.shape),
        images_dtype=str(images.dtype),
        labels_dtype=str(labels.dtype),
        pixel_min=pixel_min,
        pixel_max=pixel_max,
        records=len(records),
    )


def build_report(  # noqa: PLR0913, PLR0917
    contract: DatasetContract,
    checks: list[ValidationCheck],
    run_id: str,
    created_at: str,
    duplicate_report: DuplicateReport | None,
    quarantined: list[QuarantineEntry],
) -> ValidationReport:
    """Build an aggregate validation report.

    Args:
        contract (DatasetContract): Dataset contract.
        checks (list[ValidationCheck]): Individual checks.
        run_id (str): Build identifier.
        created_at (str): ISO-8601 timestamp.
        duplicate_report (DuplicateReport | None): Duplicate evidence.
        quarantined (list[QuarantineEntry]): Rejected files.

    Returns:
        ValidationReport: Aggregate report.
    """
    failed = any(check.severity == "critical" and not check.passed for check in checks)
    return ValidationReport(
        dataset_version=contract.dataset_version,
        contract_fingerprint=contract.fingerprint(),
        run_id=run_id,
        created_at=created_at,
        status="failed" if failed else "passed",
        checks=tuple(checks),
        duplicate_report=duplicate_report,
        quarantined=tuple(quarantined),
    )


def write_report(path: Path, report: ValidationReport) -> None:
    """Write validation evidence atomically as JSON.

    Args:
        path (Path): Destination path.
        report (ValidationReport): Report to serialize.

    Returns:
        None: The file is written atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True)
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_dataset(
    root: Path,
    archive: Path,
    contract: DatasetContract,
    run_id: str,
    created_at: str,
) -> tuple[list[ImageRecord], ValidationReport]:
    """Run archive, image, duplicate, and manifest validation.

    Args:
        root (Path): Extracted dataset root.
        archive (Path): Source archive.
        contract (DatasetContract): Dataset contract.
        run_id (str): Build identifier.
        created_at (str): ISO-8601 timestamp.

    Returns:
        tuple[list[ImageRecord], ValidationReport]: Accepted records and report.
    """
    checks = validate_archive(archive, contract)
    records, quarantined = scan_images(root, contract)
    records, duplicate_report = assign_duplicate_groups(records, contract.split.phash_distance)
    checks.append(
        _check(
            "images.corrupt",
            "critical",
            len(quarantined) <= contract.validation.max_corrupt_images,
            f"{len(quarantined)} rejected images",
        )
    )
    checks.append(
        _check(
            "duplicates.exact",
            "critical",
            duplicate_report.exact_duplicate_extra_files
            <= contract.validation.max_exact_duplicates,
            f"{duplicate_report.exact_duplicate_extra_files} extra exact duplicates",
        )
    )
    checks.append(
        _check(
            "duplicates.cross_class",
            "warning",
            duplicate_report.cross_class_groups
            <= contract.validation.max_cross_class_duplicate_groups,
            f"{duplicate_report.cross_class_groups} cross-class groups",
        )
    )
    duplicate_fraction = duplicate_report.near_duplicate_extra_files / max(len(records), 1)
    checks.append(
        _check(
            "duplicates.fraction",
            "warning",
            duplicate_fraction <= contract.validation.warn_near_duplicate_fraction,
            f"{duplicate_fraction:.4f} of files in perceptual groups",
        )
    )
    report = build_report(contract, checks, run_id, created_at, duplicate_report, quarantined)
    return records, report
