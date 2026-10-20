"""Local-first S3 mirroring with explicit checksum verification."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, ConfigDict

from eos.config import StorageConfig

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

CONTENT_TYPES = {
    ".json": "application/json",
    ".npy": "application/octet-stream",
    ".parquet": "application/vnd.apache.parquet",
    ".sha256": "text/plain",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".md": "text/markdown",
}


class PublishedFile(BaseModel):
    """One object written or verified in the S3 mirror.

    Args:
        key (str): Object key.
        path (str): Local source path.
        sha256 (str): Local digest.
        bytes (int): Object size.
        verified (bool): Whether the remote checksum matched.
    """

    model_config = ConfigDict(frozen=True)

    key: str
    path: str
    sha256: str
    bytes: int
    verified: bool


class PublishResult(BaseModel):
    """Aggregate result of a publish or verification operation.

    Args:
        bucket (str): Destination bucket.
        prefix (str): Destination prefix.
        files (tuple[PublishedFile, ...]): Object evidence.
        dry_run (bool): Whether this was a planning-only operation.
    """

    model_config = ConfigDict(frozen=True)

    bucket: str
    prefix: str
    files: tuple[PublishedFile, ...]
    dry_run: bool


class BucketDoctorReport(BaseModel):
    """Read-only bucket configuration evidence.

    Args:
        bucket (str): Bucket name.
        versioning (str): Versioning status.
        encryption (str): Encryption summary.
        public_access_blocked (bool): Whether all four public access blocks
            are enabled.
        errors (tuple[str, ...]): Non-fatal diagnostic errors.
    """

    model_config = ConfigDict(frozen=True)

    bucket: str
    versioning: str
    encryption: str
    public_access_blocked: bool
    errors: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    """Compute a file digest with streaming reads.

    Args:
        path (Path): File to hash.

    Returns:
        str: Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _content_type(path: Path) -> str:
    """Select a content type from a file suffix.

    Args:
        path (Path): Local file.

    Returns:
        str: MIME type.
    """
    return CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


class S3Publisher:
    """Publish and verify artifacts against an S3 bucket.

    Args:
        storage (StorageConfig): Storage policy containing the bucket and
            prefix.
        client (S3Client | None): Injected client used by tests.

    Raises:
        ValueError: If no bucket is configured.
    """

    def __init__(self, storage: StorageConfig, client: S3Client | None = None) -> None:
        """Create a client for the configured bucket.

        Args:
            storage (StorageConfig): Storage policy containing the bucket.
            client (S3Client | None): Injected client for tests.

        Returns:
            None: The publisher is initialized in place.

        Raises:
            ValueError: If no bucket is configured.
        """
        if not storage.s3_bucket:
            raise ValueError("s3_bucket is required for remote operations")
        self.storage = storage
        self.bucket = storage.s3_bucket
        self.prefix = storage.s3_prefix.strip("/")
        self.client: S3Client = client or boto3.client("s3", region_name=storage.region or None)

    def _key(self, relative: str) -> str:
        """Build a bucket key beneath the configured prefix.

        Args:
            relative (str): Relative object path.

        Returns:
            str: Full object key.
        """
        return f"{self.prefix}/{relative.lstrip('/')}"

    def publish(
        self,
        files: dict[str, Path],
        dry_run: bool,
    ) -> PublishResult:
        """Upload artifacts and verify remote checksums.

        Args:
            files (dict[str, Path]): Mapping from relative key to local path.
            dry_run (bool): When true, only compute local evidence.

        Returns:
            PublishResult: Upload or planning evidence.

        Raises:
            RuntimeError: If a remote checksum does not match.
        """
        evidence: list[PublishedFile] = []
        for relative, path in sorted(files.items()):
            digest = _sha256_file(path)
            key = self._key(relative)
            if not dry_run:
                self._upload(key, path, digest)
            evidence.append(
                PublishedFile(
                    key=key,
                    path=str(path),
                    sha256=digest,
                    bytes=path.stat().st_size,
                    verified=not dry_run,
                )
            )
        return PublishResult(
            bucket=self.bucket,
            prefix=self.prefix,
            files=tuple(evidence),
            dry_run=dry_run,
        )

    def _upload(self, key: str, path: Path, digest: str) -> None:
        """Upload one file and verify the returned checksum.

        Args:
            key (str): Object key.
            path (Path): Local file.
            digest (str): Local SHA-256 digest.

        Returns:
            None: The object is uploaded.

        Raises:
            RuntimeError: If the returned or stored checksum is inconsistent.
        """
        if path.stat().st_size >= 8 * 1024 * 1024:
            self.client.upload_file(
                str(path),
                self.bucket,
                key,
                ExtraArgs={
                    "ContentType": _content_type(path),
                    "Metadata": {"sha256": digest},
                },
            )
        else:
            with path.open("rb") as handle:
                response = self.client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=handle,
                    ContentLength=path.stat().st_size,
                    ContentType=_content_type(path),
                    ChecksumAlgorithm="SHA256",
                    Metadata={"sha256": digest},
                )
            if self.storage.verify_checksums:
                expected = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
                actual = response.get("ChecksumSHA256")
                if actual != expected:
                    raise RuntimeError(f"checksum mismatch for s3://{self.bucket}/{key}")
        if self.storage.verify_checksums:
            head = self.client.head_object(Bucket=self.bucket, Key=key)
            if head.get("Metadata", {}).get("sha256") != digest:
                raise RuntimeError(f"metadata checksum mismatch for s3://{self.bucket}/{key}")

    def verify(self, files: dict[str, Path]) -> PublishResult:
        """Verify local files against remote object metadata.

        Args:
            files (dict[str, Path]): Mapping from relative key to local path.

        Returns:
            PublishResult: Verification evidence.

        Raises:
            RuntimeError: If any object is missing or mismatched.
        """
        evidence: list[PublishedFile] = []
        for relative, path in sorted(files.items()):
            digest = _sha256_file(path)
            key = self._key(relative)
            try:
                head = self.client.head_object(Bucket=self.bucket, Key=key)
            except (ClientError, BotoCoreError) as error:
                raise RuntimeError(
                    f"remote object unavailable: s3://{self.bucket}/{key}"
                ) from error
            remote = head.get("Metadata", {}).get("sha256")
            if remote != digest:
                raise RuntimeError(f"remote checksum mismatch for s3://{self.bucket}/{key}")
            evidence.append(
                PublishedFile(
                    key=key,
                    path=str(path),
                    sha256=digest,
                    bytes=path.stat().st_size,
                    verified=True,
                )
            )
        return PublishResult(
            bucket=self.bucket,
            prefix=self.prefix,
            files=tuple(evidence),
            dry_run=False,
        )

    def doctor(self) -> BucketDoctorReport:
        """Inspect bucket versioning, encryption, and public access controls.

        Returns:
            BucketDoctorReport: Read-only configuration evidence.
        """
        errors: list[str] = []
        versioning = "unknown"
        encryption = "unknown"
        public_blocked = False
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except (ClientError, BotoCoreError) as error:
            errors.append(f"head_bucket: {error}")
        try:
            status = self.client.get_bucket_versioning(Bucket=self.bucket).get("Status", "Disabled")
            versioning = status
        except (ClientError, BotoCoreError) as error:
            errors.append(f"versioning: {error}")
        try:
            rules = self.client.get_bucket_encryption(Bucket=self.bucket)
            algorithms = [
                rule["ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"]
                for rule in rules.get("ServerSideEncryptionConfiguration", {}).get("Rules", [])
                if "ApplyServerSideEncryptionByDefault" in rule
            ]
            encryption = ", ".join(algorithms) if algorithms else "none"
        except (ClientError, BotoCoreError) as error:
            errors.append(f"encryption: {error}")
        try:
            block = self.client.get_public_access_block(Bucket=self.bucket).get(
                "PublicAccessBlockConfiguration", {}
            )
            public_blocked = all(
                block.get(name, False)
                for name in (
                    "BlockPublicAcls",
                    "IgnorePublicAcls",
                    "BlockPublicPolicy",
                    "RestrictPublicBuckets",
                )
            )
        except (ClientError, BotoCoreError) as error:
            errors.append(f"public_access_block: {error}")
        return BucketDoctorReport(
            bucket=self.bucket,
            versioning=versioning,
            encryption=encryption,
            public_access_blocked=public_blocked,
            errors=tuple(errors),
        )

    def download(self, key: str, destination: Path) -> PublishedFile:
        """Download one object and verify its digest before replacing a local file.

        Args:
            key (str): Object key.
            destination (Path): Local destination.

        Returns:
            PublishedFile: Download evidence.

        Raises:
            RuntimeError: If the object is missing or its checksum is invalid.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            metadata = response.get("Metadata", {})
            expected = metadata.get("sha256")
            with temporary.open("wb") as handle:
                for chunk in response["Body"].iter_chunks(1 << 20):
                    handle.write(chunk)
            digest = _sha256_file(temporary)
            if expected != digest:
                raise RuntimeError(f"download checksum mismatch for s3://{self.bucket}/{key}")
            temporary.replace(destination)
        except (ClientError, BotoCoreError, OSError) as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"download failed for s3://{self.bucket}/{key}") from error
        return PublishedFile(
            key=key,
            path=str(destination),
            sha256=digest,
            bytes=destination.stat().st_size,
            verified=True,
        )


def write_publish_result(path: Path, result: PublishResult) -> None:
    """Persist publish evidence as JSON.

    Args:
        path (Path): Destination path.
        result (PublishResult): Publish evidence.

    Returns:
        None: The file is written atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_mode(value: str) -> Literal["local", "hybrid"]:
    """Validate a storage mode string.

    Args:
        value (str): Candidate mode.

    Returns:
        Literal["local", "hybrid"]: Validated mode.

    Raises:
        ValueError: If the value is not supported.
    """
    if value == "local":
        return "local"
    if value == "hybrid":
        return "hybrid"
    raise ValueError(f"unsupported storage mode: {value}")
