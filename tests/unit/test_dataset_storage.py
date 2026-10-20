"""Tests for local-first S3 publishing and verification."""

from __future__ import annotations

import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from eos.config import StorageConfig
from eos.dataset.storage import S3Publisher, parse_mode, write_publish_result


def test_parse_mode_rejects_unknown_values() -> None:
    """Only local and hybrid modes are supported."""
    assert parse_mode("local") == "local"
    assert parse_mode("hybrid") == "hybrid"
    with pytest.raises(ValueError, match="storage mode"):
        parse_mode("remote")


@mock_aws
def test_publish_verify_and_download(tmp_path: Path) -> None:
    """Published objects must be verifiable and recoverable with checksums."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    storage = StorageConfig(
        mode="hybrid",
        s3_bucket="test-bucket",
        region="us-east-1",
        verify_checksums=True,
    )
    publisher = S3Publisher(storage, client=client)
    source = tmp_path / "manifest.json"
    source.write_text('{"ok": true}\n', encoding="utf-8")
    result = publisher.publish({"manifests/manifest.json": source}, dry_run=False)
    assert result.files[0].verified is True
    verified = publisher.verify({"manifests/manifest.json": source})
    assert verified.files[0].key.endswith("manifests/manifest.json")
    destination = tmp_path / "downloaded.json"
    downloaded = publisher.download("eos/datasets/eurosat-rgb/manifests/manifest.json", destination)
    assert downloaded.verified is True
    assert json.loads(destination.read_text(encoding="utf-8")) == {"ok": True}
    evidence = tmp_path / "publish.json"
    write_publish_result(evidence, result)
    assert evidence.is_file()


@mock_aws
def test_large_file_upload_is_verified(tmp_path: Path) -> None:
    """Files above the multipart threshold must still be checksum-verified."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    storage = StorageConfig(
        mode="hybrid",
        s3_bucket="test-bucket",
        region="us-east-1",
        verify_checksums=True,
    )
    publisher = S3Publisher(storage, client=client)
    source = tmp_path / "large.npy"
    source.write_bytes(b"x" * (9 * 1024 * 1024))
    result = publisher.publish({"processed/large.npy": source}, dry_run=False)
    assert result.files[0].verified is True
    assert publisher.verify({"processed/large.npy": source}).files[0].verified is True


@mock_aws
def test_doctor_reports_bucket_controls() -> None:
    """Bucket diagnostics must report versioning, encryption, and access state."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    storage = StorageConfig(mode="hybrid", s3_bucket="test-bucket", region="us-east-1")
    report = S3Publisher(storage, client=client).doctor()
    assert report.bucket == "test-bucket"
    assert isinstance(report.versioning, str)
    assert isinstance(report.encryption, str)
    assert isinstance(report.public_access_blocked, bool)


@mock_aws
def test_dry_run_does_not_upload(tmp_path: Path) -> None:
    """Dry-run publishing must report keys without writing objects."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    storage = StorageConfig(mode="hybrid", s3_bucket="test-bucket", region="us-east-1")
    publisher = S3Publisher(storage, client=client)
    source = tmp_path / "data.txt"
    source.write_text("payload", encoding="utf-8")
    result = publisher.publish({"data.txt": source}, dry_run=True)
    assert result.dry_run is True
    assert result.files[0].verified is False
    assert client.list_objects_v2(Bucket="test-bucket").get("KeyCount", 0) == 0


def test_missing_bucket_is_rejected() -> None:
    """A remote publisher cannot exist without a bucket."""
    with pytest.raises(ValueError, match="s3_bucket"):
        S3Publisher(StorageConfig(mode="local"))
