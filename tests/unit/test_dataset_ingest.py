"""Tests for resilient download and atomic extraction."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from tests.helpers import build_contract, zip_bytes

from eos.config import RetryConfig
from eos.dataset.ingest import download_archive, extract_archive


def make_payload(tmp_path: Path) -> tuple[bytes, object]:
    """Build an in-memory archive and its contract."""
    root = tmp_path / "payload"
    payload = zip_bytes(root, ("Alpha", "Beta"), per_class=2)
    archive = tmp_path / "EuroSAT_RGB.zip"
    archive.write_bytes(payload)
    contract = build_contract(("Alpha", "Beta"), 4, archive=archive)
    return payload, contract


def test_download_and_extract_are_idempotent(tmp_path: Path, httpserver) -> None:
    """A valid source must download once and extract through a flat root."""
    payload, contract = make_payload(tmp_path)
    httpserver.expect_request("/EuroSAT_RGB.zip").respond_with_data(
        payload, content_type="application/zip"
    )
    source = contract.source.model_copy(update={"urls": (httpserver.url_for("/EuroSAT_RGB.zip"),)})
    contract = contract.model_copy(update={"source": source})
    destination = tmp_path / "download.zip"
    retry = RetryConfig(attempts=2, base_delay_seconds=0.01, max_delay_seconds=0.02)
    first = download_archive(contract, destination, retry, "run-1", "2026-01-01T00:00:00Z")
    second = download_archive(contract, destination, retry, "run-2", "2026-01-01T00:00:01Z")
    assert first.skipped is False
    assert second.skipped is True
    extraction = tmp_path / "extracted"
    extracted = extract_archive(destination, extraction, contract, "run-1", "2026-01-01T00:00:00Z")
    repeated = extract_archive(destination, extraction, contract, "run-2", "2026-01-01T00:00:02Z")
    assert extracted.image_count == 4
    assert repeated.skipped is True
    assert (extraction / "Alpha").is_dir()
    assert (extraction / ".synthetic-extraction.json").is_file()


def test_checksum_mismatch_fails_closed(tmp_path: Path, httpserver) -> None:
    """A payload that does not match the contract must be rejected."""
    payload, contract = make_payload(tmp_path)
    httpserver.expect_request("/EuroSAT_RGB.zip").respond_with_data(
        payload[:-1], content_type="application/zip"
    )
    source = contract.source.model_copy(update={"urls": (httpserver.url_for("/EuroSAT_RGB.zip"),)})
    contract = contract.model_copy(update={"source": source})
    retry = RetryConfig(attempts=1, base_delay_seconds=0.01, max_delay_seconds=0.02)
    with pytest.raises(RuntimeError):
        download_archive(
            contract,
            tmp_path / "download.zip",
            retry,
            "run-1",
            "2026-01-01T00:00:00Z",
        )


def test_zip_path_traversal_is_rejected(tmp_path: Path) -> None:
    """Zip members that escape the extraction root must be rejected."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("../evil.jpg", b"payload")
    archive = tmp_path / "evil.zip"
    archive.write_bytes(buffer.getvalue())
    contract = build_contract(("Alpha", "Beta"), 1, archive=archive)
    with pytest.raises(ValueError, match="zip member"):
        extract_archive(archive, tmp_path / "out", contract, "run-1", "2026-01-01T00:00:00Z")
