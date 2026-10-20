"""Resilient source acquisition and atomic extraction for EuroSAT."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath

import httpx
from pydantic import BaseModel, ConfigDict

from eos.config import RetryConfig
from eos.dataset.contract import DatasetContract

READ_CHUNK_BYTES = 1 << 20
LOGGER = logging.getLogger(__name__)


class DownloadResult(BaseModel):
    """Evidence describing a completed archive acquisition.

    Args:
        url (str): URL that supplied the archive.
        sha256 (str): Archive digest.
        bytes (int): Archive size.
        skipped (bool): Whether an existing valid archive was reused.
        downloaded_at (str): ISO-8601 completion timestamp.
    """

    model_config = ConfigDict(frozen=True)

    url: str
    sha256: str
    bytes: int
    skipped: bool
    downloaded_at: str


class ExtractionResult(BaseModel):
    """Evidence describing a completed atomic extraction.

    Args:
        archive_sha256 (str): Source archive digest.
        destination (str): Extracted dataset root.
        image_count (int): Number of extracted images.
        class_counts (dict[str, int]): Counts per class.
        skipped (bool): Whether an existing current extraction was reused.
        extracted_at (str): ISO-8601 completion timestamp.
    """

    model_config = ConfigDict(frozen=True)

    archive_sha256: str
    destination: str
    image_count: int
    class_counts: dict[str, int]
    skipped: bool
    extracted_at: str


def sha256_file(path: Path) -> str:
    """Compute a file digest with streaming reads.

    Args:
        path (Path): File to hash.

    Returns:
        str: Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_archive(path: Path, contract: DatasetContract) -> bool:
    """Check archive existence, size, and digest.

    Args:
        path (Path): Candidate archive.
        contract (DatasetContract): Dataset contract.

    Returns:
        bool: True when the archive is complete and correct.
    """
    return (
        path.is_file()
        and path.stat().st_size == contract.source.archive_bytes
        and sha256_file(path) == contract.source.archive_sha256
    )


def _write_json(path: Path, payload: BaseModel) -> None:
    """Write a Pydantic model to JSON atomically.

    Args:
        path (Path): Destination path.
        payload (BaseModel): Model to serialize.

    Returns:
        None: The destination is replaced atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    document = json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True)
    temporary.write_text(document + "\n", encoding="utf-8")
    temporary.replace(path)


def _stream_download(
    url: str,
    part: Path,
    expected_bytes: int,
    retry: RetryConfig,
) -> None:
    """Download one URL into a resumable partial file.

    Args:
        url (str): Source URL.
        part (Path): Partial destination file.
        expected_bytes (int): Expected final size.
        retry (RetryConfig): Retry policy.

    Returns:
        None: The partial file is complete.

    Raises:
        RuntimeError: If all attempts fail or the server returns an
            inconsistent payload.
    """
    last_error: Exception | None = None
    for attempt in range(1, retry.attempts + 1):
        offset = part.stat().st_size if part.is_file() else 0
        if offset > expected_bytes:
            part.unlink(missing_ok=True)
            offset = 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with (
                httpx.Client(
                    follow_redirects=True,
                    timeout=retry.timeout_seconds,
                ) as client,
                client.stream("GET", url, headers=headers) as response,
            ):
                response.raise_for_status()
                if offset and response.status_code != httpx.codes.PARTIAL_CONTENT:
                    offset = 0
                    part.write_bytes(b"")
                mode = "ab" if offset else "wb"
                with part.open(mode) as handle:
                    for chunk in response.iter_bytes(READ_CHUNK_BYTES):
                        handle.write(chunk)
            if part.stat().st_size == expected_bytes:
                return
            last_error = RuntimeError(
                f"incomplete download: expected {expected_bytes}, found {part.stat().st_size}"
            )
        except (httpx.HTTPError, OSError) as error:
            last_error = error
            LOGGER.warning(
                "download attempt %s/%s failed for %s: %s",
                attempt,
                retry.attempts,
                url,
                error,
            )
            if attempt < retry.attempts:
                delay = min(
                    retry.max_delay_seconds,
                    retry.base_delay_seconds * (2 ** (attempt - 1)),
                )
                time.sleep(delay)
    raise RuntimeError(f"download failed after {retry.attempts} attempts: {last_error}")


def download_archive(
    contract: DatasetContract,
    destination: Path,
    retry: RetryConfig,
    run_id: str,
    timestamp: str,
) -> DownloadResult:
    """Acquire the source archive with resume, retry, and checksum guards.

    Args:
        contract (DatasetContract): Dataset contract.
        destination (Path): Final archive path.
        retry (RetryConfig): Retry policy.
        run_id (str): Build identifier.
        timestamp (str): ISO-8601 timestamp used for evidence.

    Returns:
        DownloadResult: Acquisition evidence.

    Raises:
        RuntimeError: If every source URL fails or the checksum does not match.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _valid_archive(destination, contract):
        return DownloadResult(
            url=contract.source.urls[0],
            sha256=contract.source.archive_sha256,
            bytes=contract.source.archive_bytes,
            skipped=True,
            downloaded_at=timestamp,
        )
    if destination.is_file():
        corrupt = destination.with_name(f"{destination.name}.corrupt-{run_id}")
        destination.replace(corrupt)
        LOGGER.warning("quarantined invalid archive at %s", corrupt)
    part = destination.with_name(f"{destination.name}.part")
    last_error: Exception | None = None
    for url in contract.source.urls:
        try:
            _stream_download(url, part, contract.source.archive_bytes, retry)
            digest = sha256_file(part)
            if digest != contract.source.archive_sha256:
                corrupt = destination.with_name(f"{destination.name}.corrupt-{run_id}")
                part.replace(corrupt)
                raise RuntimeError(
                    f"checksum mismatch: expected {contract.source.archive_sha256}, found {digest}"
                )
            part.replace(destination)
            return DownloadResult(
                url=url,
                sha256=digest,
                bytes=destination.stat().st_size,
                skipped=False,
                downloaded_at=timestamp,
            )
        except (RuntimeError, OSError) as error:
            last_error = error
            LOGGER.warning("source %s failed: %s", url, error)
    raise RuntimeError(f"all source URLs failed: {last_error}")


def find_dataset_root(destination: Path, contract: DatasetContract) -> Path:
    """Locate the directory containing the expected class folders.

    Args:
        destination (Path): Extraction directory.
        contract (DatasetContract): Dataset contract.

    Returns:
        Path: Directory whose direct children are the class folders.

    Raises:
        RuntimeError: If no candidate has the exact expected class set.
    """
    expected = set(contract.source.class_names)
    candidates = [destination, *[path for path in destination.iterdir() if path.is_dir()]]
    for candidate in candidates:
        classes = {path.name for path in candidate.iterdir() if path.is_dir()}
        if classes == expected:
            return candidate
    raise RuntimeError(f"no directory with the expected classes under {destination}")


def _safe_target(root: Path, member: str) -> Path:
    """Resolve a zip member to a path confined to the extraction root.

    Args:
        root (Path): Temporary extraction root.
        member (str): Zip member name.

    Returns:
        Path: Safe destination path.

    Raises:
        ValueError: If the member path is absolute, contains parent traversal,
            or escapes the extraction root.
    """
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe zip member: {member}")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"zip member escapes root: {member}")
    return target


def _extraction_marker(destination: Path, contract: DatasetContract) -> Path:
    """Return the extraction marker path.

    Args:
        destination (Path): Extracted dataset root.
        contract (DatasetContract): Dataset contract.

    Returns:
        Path: Marker path.
    """
    return destination / f".{contract.dataset}-extraction.json"


def _extraction_is_current(destination: Path, contract: DatasetContract) -> bool:
    """Check whether an existing extraction matches the contract.

    Args:
        destination (Path): Extracted dataset root.
        contract (DatasetContract): Dataset contract.

    Returns:
        bool: True when the marker and class folders are consistent.
    """
    marker = _extraction_marker(destination, contract)
    if not marker.is_file() or not destination.is_dir():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("archive_sha256") != contract.source.archive_sha256:
        return False
    classes = {path.name for path in destination.iterdir() if path.is_dir()}
    return classes == set(contract.source.class_names)


def _count_classes(root: Path, contract: DatasetContract) -> dict[str, int]:
    """Count image files per expected class.

    Args:
        root (Path): Extracted dataset root.
        contract (DatasetContract): Dataset contract.

    Returns:
        dict[str, int]: Counts per class.
    """
    counts: dict[str, int] = {}
    for class_name in contract.source.class_names:
        counts[class_name] = sum(
            1
            for path in (root / class_name).iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    return counts


def extract_archive(  # noqa: PLR0912
    archive: Path,
    destination: Path,
    contract: DatasetContract,
    run_id: str,
    timestamp: str,
) -> ExtractionResult:
    """Extract the archive through a temporary tree and atomic swap.

    Args:
        archive (Path): Verified source archive.
        destination (Path): Final extracted dataset root.
        contract (DatasetContract): Dataset contract.
        run_id (str): Build identifier.
        timestamp (str): ISO-8601 timestamp used for evidence.

    Returns:
        ExtractionResult: Extraction evidence.

    Raises:
        RuntimeError: If the archive is invalid or extraction fails.
    """
    if _extraction_is_current(destination, contract):
        counts = _count_classes(destination, contract)
        return ExtractionResult(
            archive_sha256=contract.source.archive_sha256,
            destination=str(destination),
            image_count=sum(counts.values()),
            class_counts=counts,
            skipped=True,
            extracted_at=timestamp,
        )
    if not _valid_archive(archive, contract):
        raise RuntimeError("archive does not match the dataset contract")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{run_id}")
    previous = destination.with_name(f".{destination.name}.old-{run_id}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as bundle:
            bad_member = bundle.testzip()
            if bad_member is not None:
                raise RuntimeError(f"corrupt zip member: {bad_member}")
            for info in bundle.infolist():
                if info.is_dir():
                    continue
                target = _safe_target(temporary, info.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
        staged_root = find_dataset_root(temporary, contract)
        counts = _count_classes(staged_root, contract)
        if sum(counts.values()) != contract.source.expected_images:
            raise RuntimeError(
                f"expected {contract.source.expected_images} images, found {sum(counts.values())}"
            )
        marker = _extraction_marker(staged_root, contract)
        marker.write_text(
            json.dumps(
                {
                    "archive_sha256": contract.source.archive_sha256,
                    "image_count": sum(counts.values()),
                    "class_counts": counts,
                    "run_id": run_id,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            destination.replace(previous)
        try:
            staged_root.replace(destination)
        except OSError:
            if previous.exists():
                previous.replace(destination)
            raise
        if previous.exists():
            shutil.rmtree(previous)
        if temporary.exists():
            shutil.rmtree(temporary)
        return ExtractionResult(
            archive_sha256=contract.source.archive_sha256,
            destination=str(destination),
            image_count=sum(counts.values()),
            class_counts=counts,
            skipped=False,
            extracted_at=timestamp,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def write_evidence(path: Path, payload: BaseModel) -> None:
    """Write acquisition or extraction evidence.

    Args:
        path (Path): Destination JSON path.
        payload (BaseModel): Evidence model.

    Returns:
        None: The file is written atomically.
    """
    _write_json(path, payload)
