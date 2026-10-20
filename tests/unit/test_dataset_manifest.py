"""Tests for manifest hashing, persistence, and metadata."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.helpers import build_contract, make_record

from eos.dataset.manifest import (
    ManifestMetadata,
    build_metadata,
    class_counts,
    manifest_hash,
    read_manifest,
    split_counts,
    write_manifest,
)


def records():
    """Build two records with different classes and splits."""
    return [
        make_record("Alpha/a", 0, "Alpha", split="train"),
        make_record("Beta/b", 1, "Beta", split="val"),
    ]


def test_manifest_hash_is_order_independent() -> None:
    """Logical fingerprints must be stable under row ordering changes."""
    left = records()
    right = list(reversed(left))
    assert manifest_hash(left) == manifest_hash(right)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    """A manifest must survive Parquet and JSON persistence unchanged."""
    contract = build_contract(("Alpha", "Beta"), 2)
    rows = records()
    metadata = build_metadata(contract, rows, "run-1", "2026-01-01T00:00:00Z", "abc", "0.1.0")
    parquet = tmp_path / "manifest.parquet"
    metadata_path = tmp_path / "manifest.json"
    write_manifest(rows, metadata, parquet, metadata_path)
    loaded = read_manifest(parquet, metadata_path)
    assert len(loaded.records) == 2
    assert loaded.metadata.manifest_sha256 == metadata.manifest_sha256
    assert loaded.records[0].image_id == "Alpha/a"
    assert class_counts(rows) == {"Alpha": 1, "Beta": 1}
    assert split_counts(rows) == {
        "train": {"Alpha": 1},
        "val": {"Beta": 1},
        "test": {},
    }


def test_manifest_metadata_mismatch_is_rejected(tmp_path: Path) -> None:
    """A stored fingerprint that does not match the rows must fail closed."""
    contract = build_contract(("Alpha", "Beta"), 2)
    rows = records()
    metadata = build_metadata(contract, rows, "run-1", "2026-01-01T00:00:00Z", "abc", "0.1.0")
    parquet = tmp_path / "manifest.parquet"
    metadata_path = tmp_path / "manifest.json"
    write_manifest(rows, metadata, parquet, metadata_path)
    payload = metadata.model_dump(mode="json")
    payload["manifest_sha256"] = "f" * 64
    metadata_path.write_text(
        __import__("json").dumps(payload),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fingerprint"):
        read_manifest(parquet, metadata_path)


def test_missing_manifest_is_rejected(tmp_path: Path) -> None:
    """A missing parquet file must raise a clear error."""
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path / "missing.parquet", tmp_path / "missing.json")


def test_metadata_rejects_wrong_fingerprint(tmp_path: Path) -> None:
    """Writing metadata with a stale fingerprint must fail before persistence."""
    contract = build_contract(("Alpha", "Beta"), 2)
    rows = records()
    metadata = build_metadata(contract, rows, "run-1", "2026-01-01T00:00:00Z", "abc", "0.1.0")
    stale = ManifestMetadata.model_validate(
        {**metadata.model_dump(mode="json"), "manifest_sha256": "e" * 64}
    )
    with pytest.raises(ValueError, match="fingerprint"):
        write_manifest(rows, stale, tmp_path / "manifest.parquet", tmp_path / "manifest.json")
