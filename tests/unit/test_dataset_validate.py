"""Tests for archive, image, manifest, and array validation."""

from __future__ import annotations

from pathlib import Path

from tests.helpers import build_contract, build_zip, write_image

from eos.dataset.curate import curate_dataset
from eos.dataset.splits import assign_splits
from eos.dataset.validate import (
    build_report,
    quarantine_files,
    scan_images,
    validate_archive,
    validate_arrays,
    validate_manifest,
    write_report,
)


def build_fixture(tmp_path: Path):
    """Create a two-class synthetic source tree and archive."""
    root = tmp_path / "raw"
    archive = tmp_path / "EuroSAT_RGB.zip"
    classes = ("Alpha", "Beta")
    build_zip(root, archive, classes, per_class=20)
    contract = build_contract(classes, 40, archive=archive)
    return root, archive, contract


def test_archive_and_image_validation(tmp_path: Path) -> None:
    """A conforming archive and image tree must pass every critical check."""
    root, archive, contract = build_fixture(tmp_path)
    checks = validate_archive(archive, contract)
    assert all(check.passed for check in checks if check.severity == "critical")
    records, quarantined = scan_images(root, contract)
    assert len(records) == 40
    assert quarantined == []
    assert {record.class_name for record in records} == {"Alpha", "Beta"}


def test_checksum_mismatch_is_reported(tmp_path: Path) -> None:
    """A modified archive must fail the digest check."""
    _, archive, contract = build_fixture(tmp_path)
    payload = bytearray(archive.read_bytes())
    payload[0] ^= 1
    archive.write_bytes(bytes(payload))
    checks = validate_archive(archive, contract)
    digest_check = next(check for check in checks if check.name == "archive.sha256")
    assert digest_check.passed is False


def test_manifest_and_arrays_validate(tmp_path: Path) -> None:
    """Split assignments and curated arrays must satisfy their contracts."""
    root, _, contract = build_fixture(tmp_path)
    records, _ = scan_images(root, contract)
    split_records = assign_splits(records, contract)
    checks = validate_manifest(split_records, contract)
    assert all(check.passed for check in checks)
    output = tmp_path / "curated"
    stats = curate_dataset(split_records, root, output, contract)
    assert stats.records == 40
    array_report = validate_arrays(
        sorted(split_records, key=lambda record: record.image_id),
        output / "images_uint8.npy",
        output / "labels_int64.npy",
        contract,
    )
    assert array_report.records == 40
    assert array_report.pixel_min >= 0
    assert array_report.pixel_max <= 255


def test_corrupt_and_blank_images_are_quarantined(tmp_path: Path) -> None:
    """Undecodable and constant images must be rejected with evidence."""
    root, _, contract = build_fixture(tmp_path)
    (root / "Alpha" / "broken.jpg").write_bytes(b"not an image")
    write_image(root / "Alpha" / "blank.jpg", seed=1)
    blank = root / "Alpha" / "blank.jpg"
    blank.write_bytes(b"")
    records, quarantined = scan_images(root, contract)
    assert len(records) == 40
    assert {entry.relative_path for entry in quarantined} == {
        "Alpha/broken.jpg",
        "Alpha/blank.jpg",
    }
    destination = quarantine_files(root, quarantined, tmp_path / "quarantine", "run-1")
    assert (destination / "Alpha" / "broken.jpg").is_file()
    assert (destination / "reasons.jsonl").is_file()


def test_validation_report_roundtrip(tmp_path: Path) -> None:
    """Validation evidence must serialize with a deterministic status."""
    _, _, contract = build_fixture(tmp_path)
    checks = validate_archive(tmp_path / "EuroSAT_RGB.zip", contract)
    report = build_report(contract, checks, "run-1", "2026-01-01T00:00:00Z", None, [])
    path = tmp_path / "validation.json"
    write_report(path, report)
    assert report.status == "passed"
    assert path.is_file()
