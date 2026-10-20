"""Command-line interface for the EuroSAT data engineering pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlparse

from filelock import FileLock, Timeout

from eos.config import ExperimentConfig, ServiceSettings, StorageConfig, load_config
from eos.dataset.contract import DatasetContract, load_contract
from eos.dataset.curate import curate_dataset, write_checksums
from eos.dataset.ingest import (
    download_archive,
    extract_archive,
    write_evidence,
)
from eos.dataset.manifest import (
    ImageRecord,
    build_metadata,
    read_manifest,
    write_manifest,
)
from eos.dataset.splits import assign_splits, split_summary
from eos.dataset.storage import S3Publisher, write_publish_result
from eos.dataset.validate import (
    ValidationCheck,
    build_report,
    quarantine_files,
    validate_arrays,
    validate_dataset,
    validate_manifest,
    write_report,
)
from eos.utils.logging import configure_logging

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path("configs/base.yaml")
LOGGER = logging.getLogger(__name__)


def _resolve(path: Path) -> Path:
    """Resolve a repository-relative path.

    Args:
        path (Path): Candidate path.

    Returns:
        Path: Absolute path rooted at the repository when relative.
    """
    return path if path.is_absolute() else PROJECT_ROOT / path


def _pipeline_version() -> str:
    """Return the installed project version.

    Returns:
        str: Package version or a safe fallback.
    """
    try:
        return version("eos-autoencoder-engine")
    except PackageNotFoundError:
        return "0.1.0"


def _git_commit() -> str:
    """Return the current short git commit.

    Returns:
        str: Short commit hash or ``unknown`` when unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _now() -> str:
    """Return an ISO-8601 UTC timestamp.

    Returns:
        str: Timestamp with second precision.
    """
    return datetime.now(UTC).isoformat(timespec="seconds")


def _run_id(timestamp: str) -> str:
    """Build a filesystem-safe run identifier.

    Args:
        timestamp (str): ISO-8601 timestamp.

    Returns:
        str: Compact identifier.
    """
    compact = timestamp.replace("-", "").replace(":", "")
    return compact.replace("+0000", "Z")


def _archive_name(contract: DatasetContract) -> str:
    """Derive the archive filename from the first source URL.

    Args:
        contract (DatasetContract): Dataset contract.

    Returns:
        str: Archive filename.
    """
    name = Path(urlparse(contract.source.urls[0]).path).name
    return name or f"{contract.dataset}.zip"


def _paths(config: ExperimentConfig, contract: DatasetContract) -> dict[str, Path]:
    """Build versioned pipeline paths.

    Args:
        config (ExperimentConfig): Experiment configuration.
        contract (DatasetContract): Dataset contract.

    Returns:
        dict[str, Path]: Named absolute paths.
    """
    version = contract.dataset_version
    return {
        "archive": _resolve(config.pipeline.raw_root) / _archive_name(contract),
        "raw": _resolve(config.pipeline.raw_root) / "eurosat",
        "processed": _resolve(config.pipeline.processed_root) / version,
        "manifest": _resolve(config.pipeline.manifest_root) / version,
        "quarantine": _resolve(config.pipeline.quarantine_root),
        "runs": _resolve(config.pipeline.run_root),
    }


def _lock(config: ExperimentConfig) -> FileLock:
    """Create the pipeline run lock.

    Args:
        config (ExperimentConfig): Experiment configuration.

    Returns:
        FileLock: Lock object for the pipeline.
    """
    lock_dir = _resolve(config.pipeline.run_root) / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return FileLock(str(lock_dir / "pipeline.lock"), timeout=0)


def _context(config_path: Path) -> tuple[ExperimentConfig, DatasetContract]:
    """Load configuration and contract.

    Args:
        config_path (Path): Experiment configuration path.

    Returns:
        tuple[ExperimentConfig, DatasetContract]: Loaded context.
    """
    config = load_config(_resolve(config_path))
    settings = ServiceSettings()
    storage = config.pipeline.storage.model_dump()
    if settings.eos_storage_mode:
        storage["mode"] = settings.eos_storage_mode
    if settings.eos_s3_bucket:
        storage["s3_bucket"] = settings.eos_s3_bucket
    if settings.eos_s3_prefix:
        storage["s3_prefix"] = settings.eos_s3_prefix
    if settings.eos_s3_region:
        storage["region"] = settings.eos_s3_region
    if storage != config.pipeline.storage.model_dump():
        config = config.model_copy(
            update={
                "pipeline": config.pipeline.model_copy(
                    update={"storage": StorageConfig.model_validate(storage)}
                )
            }
        )
    contract = load_contract(_resolve(config.pipeline.contract_path))
    return config, contract


def _fetch(
    config: ExperimentConfig,
    contract: DatasetContract,
    run_id: str,
    timestamp: str,
) -> int:
    """Download the source archive.

    Args:
        config (ExperimentConfig): Experiment configuration.
        contract (DatasetContract): Dataset contract.
        run_id (str): Run identifier.
        timestamp (str): ISO-8601 timestamp.

    Returns:
        int: Process exit code.
    """
    paths = _paths(config, contract)
    result = download_archive(contract, paths["archive"], config.pipeline.retry, run_id, timestamp)
    write_evidence(paths["runs"] / run_id / "fetch.json", result)
    LOGGER.info(
        "archive ready path=%s bytes=%s skipped=%s", paths["archive"], result.bytes, result.skipped
    )
    return 0


def _extract(
    config: ExperimentConfig,
    contract: DatasetContract,
    run_id: str,
    timestamp: str,
) -> int:
    """Extract the archive atomically.

    Args:
        config (ExperimentConfig): Experiment configuration.
        contract (DatasetContract): Dataset contract.
        run_id (str): Run identifier.
        timestamp (str): ISO-8601 timestamp.

    Returns:
        int: Process exit code.
    """
    paths = _paths(config, contract)
    result = extract_archive(paths["archive"], paths["raw"], contract, run_id, timestamp)
    write_evidence(paths["runs"] / run_id / "extract.json", result)
    LOGGER.info(
        "dataset extracted root=%s images=%s skipped=%s",
        result.destination,
        result.image_count,
        result.skipped,
    )
    return 0


def _validate(
    config: ExperimentConfig,
    contract: DatasetContract,
    run_id: str,
    timestamp: str,
) -> int:
    """Validate source images and write the base manifest.

    Args:
        config (ExperimentConfig): Experiment configuration.
        contract (DatasetContract): Dataset contract.
        run_id (str): Run identifier.
        timestamp (str): ISO-8601 timestamp.

    Returns:
        int: Process exit code, zero only when validation passes.
    """
    paths = _paths(config, contract)
    records, report = validate_dataset(paths["raw"], paths["archive"], contract, run_id, timestamp)
    paths["manifest"].mkdir(parents=True, exist_ok=True)
    write_report(paths["manifest"] / "validation.json", report)
    if report.quarantined:
        quarantine_files(paths["raw"], list(report.quarantined), paths["quarantine"], run_id)
    if report.status != "passed":
        for check in report.checks:
            if check.severity == "critical" and not check.passed:
                LOGGER.error("validation failed check=%s detail=%s", check.name, check.detail)
        return 1
    metadata = build_metadata(
        contract,
        records,
        run_id,
        timestamp,
        _git_commit(),
        _pipeline_version(),
    )
    write_manifest(
        records,
        metadata,
        paths["processed"] / "base_manifest.parquet",
        paths["manifest"] / "base_manifest.json",
    )
    write_checksums(
        paths["manifest"] / "checksums.sha256",
        [
            paths["processed"] / "base_manifest.parquet",
            paths["manifest"] / "base_manifest.json",
            paths["manifest"] / "validation.json",
        ],
    )
    LOGGER.info("validation passed records=%s manifest=%s", len(records), metadata.manifest_sha256)
    return 0


def _split(
    config: ExperimentConfig,
    contract: DatasetContract,
    run_id: str,
    timestamp: str,
) -> int:
    """Assign deterministic splits and write the final manifest.

    Args:
        config (ExperimentConfig): Experiment configuration.
        contract (DatasetContract): Dataset contract.
        run_id (str): Run identifier.
        timestamp (str): ISO-8601 timestamp.

    Returns:
        int: Process exit code.
    """
    paths = _paths(config, contract)
    base = read_manifest(
        paths["processed"] / "base_manifest.parquet",
        paths["manifest"] / "base_manifest.json",
    )
    records = assign_splits(list(base.records), contract)
    metadata = build_metadata(
        contract,
        records,
        run_id,
        timestamp,
        _git_commit(),
        _pipeline_version(),
    )
    write_manifest(
        records,
        metadata,
        paths["processed"] / "manifest.parquet",
        paths["processed"] / "manifest.json",
    )
    paths["manifest"].mkdir(parents=True, exist_ok=True)
    manifest_target = paths["manifest"] / "manifest.json"
    manifest_temporary = manifest_target.with_name(f".{manifest_target.name}.tmp")
    manifest_temporary.write_text(
        (paths["processed"] / "manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    manifest_temporary.replace(manifest_target)
    summary_target = paths["manifest"] / "split_summary.json"
    summary_temporary = summary_target.with_name(f".{summary_target.name}.tmp")
    summary_temporary.write_text(
        json.dumps(split_summary(records), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_temporary.replace(summary_target)
    write_checksums(
        paths["manifest"] / "checksums.sha256",
        [
            paths["processed"] / "manifest.parquet",
            paths["manifest"] / "manifest.json",
            paths["manifest"] / "split_summary.json",
        ],
    )
    LOGGER.info("splits assigned manifest=%s", metadata.manifest_sha256)
    return 0


def _curate(
    config: ExperimentConfig,
    contract: DatasetContract,
    run_id: str,
) -> int:
    """Build curated arrays and statistics.

    Args:
        config (ExperimentConfig): Experiment configuration.
        contract (DatasetContract): Dataset contract.
        run_id (str): Run identifier.

    Returns:
        int: Process exit code.
    """
    del run_id
    paths = _paths(config, contract)
    manifest = read_manifest(
        paths["processed"] / "manifest.parquet",
        paths["processed"] / "manifest.json",
    )
    stats = curate_dataset(
        list(manifest.records),
        paths["raw"],
        paths["processed"],
        contract,
    )
    LOGGER.info("curated arrays records=%s stats=%s", stats.records, paths["processed"])
    return 0


def _verification_checks(
    config: ExperimentConfig,
    contract: DatasetContract,
) -> tuple[list[ValidationCheck], list[ImageRecord]]:
    """Build manifest and array verification checks.

    Args:
        config (ExperimentConfig): Experiment configuration.
        contract (DatasetContract): Dataset contract.

    Returns:
        tuple[list[ValidationCheck], list[ImageRecord]]: Checks and records.
    """
    paths = _paths(config, contract)
    manifest = read_manifest(
        paths["processed"] / "manifest.parquet",
        paths["processed"] / "manifest.json",
    )
    records = list(manifest.records)
    checks = validate_manifest(records, contract)
    try:
        array_report = validate_arrays(
            records,
            paths["processed"] / "images_uint8.npy",
            paths["processed"] / "labels_int64.npy",
            contract,
        )
        checks.append(
            ValidationCheck(
                name="arrays.shape",
                severity="critical",
                passed=True,
                detail=(
                    f"{array_report.images_shape} pixel range "
                    f"{array_report.pixel_min}-{array_report.pixel_max}"
                ),
            )
        )
    except (OSError, ValueError) as error:
        checks.append(
            ValidationCheck(
                name="arrays.shape",
                severity="critical",
                passed=False,
                detail=str(error),
            )
        )
    return checks, records


def _verify(
    config: ExperimentConfig,
    contract: DatasetContract,
    run_id: str,
    timestamp: str,
) -> int:
    """Verify manifest, split, and array contracts.

    Args:
        config (ExperimentConfig): Experiment configuration.
        contract (DatasetContract): Dataset contract.
        run_id (str): Run identifier.
        timestamp (str): ISO-8601 timestamp.

    Returns:
        int: Process exit code.
    """
    paths = _paths(config, contract)
    checks, records = _verification_checks(config, contract)
    report = build_report(contract, checks, run_id, timestamp, None, [])
    write_report(paths["manifest"] / "verification.json", report)
    if report.status != "passed":
        for check in checks:
            if check.severity == "critical" and not check.passed:
                LOGGER.error("verification failed check=%s detail=%s", check.name, check.detail)
        return 1
    LOGGER.info("verification passed records=%s", len(records))
    return 0


def _publish_files(config: ExperimentConfig, contract: DatasetContract) -> dict[str, Path]:
    """Build the human-readable S3 publish map.

    Args:
        config (ExperimentConfig): Experiment configuration.
        contract (DatasetContract): Dataset contract.

    Returns:
        dict[str, Path]: Relative key to local path mapping.
    """
    paths = _paths(config, contract)
    version = contract.dataset_version
    candidates = {
        f"raw/{_archive_name(contract)}": paths["archive"],
        f"manifests/{version}/manifest.parquet": paths["processed"] / "manifest.parquet",
        f"manifests/{version}/manifest.json": paths["manifest"] / "manifest.json",
        f"manifests/{version}/validation.json": paths["manifest"] / "validation.json",
        f"manifests/{version}/verification.json": paths["manifest"] / "verification.json",
        f"manifests/{version}/checksums.sha256": paths["manifest"] / "checksums.sha256",
        f"processed/{version}/images_uint8.npy": paths["processed"] / "images_uint8.npy",
        f"processed/{version}/labels_int64.npy": paths["processed"] / "labels_int64.npy",
        f"processed/{version}/stats.json": paths["processed"] / "stats.json",
        f"processed/{version}/checksums.sha256": paths["processed"] / "checksums.sha256",
        f"processed/{version}/train_indices.npy": paths["processed"] / "train_indices.npy",
        f"processed/{version}/val_indices.npy": paths["processed"] / "val_indices.npy",
        f"processed/{version}/test_indices.npy": paths["processed"] / "test_indices.npy",
    }
    return {key: path for key, path in candidates.items() if path.is_file()}


def _publish(
    config: ExperimentConfig,
    contract: DatasetContract,
    run_id: str,
    dry_run: bool,
) -> int:
    """Publish versioned artifacts to S3.

    Args:
        config (ExperimentConfig): Experiment configuration.
        contract (DatasetContract): Dataset contract.
        run_id (str): Run identifier.
        dry_run (bool): Whether to plan without uploading.

    Returns:
        int: Process exit code.
    """
    paths = _paths(config, contract)
    if config.pipeline.storage.mode == "local":
        LOGGER.info("storage mode is local; no remote publish requested")
        return 0
    publisher = S3Publisher(config.pipeline.storage)
    result = publisher.publish(_publish_files(config, contract), dry_run=dry_run)
    write_publish_result(paths["runs"] / run_id / "publish.json", result)
    LOGGER.info(
        "publish complete bucket=%s prefix=%s files=%s dry_run=%s",
        result.bucket,
        result.prefix,
        len(result.files),
        result.dry_run,
    )
    return 0


def _doctor(config: ExperimentConfig, contract: DatasetContract) -> int:
    """Inspect remote storage configuration.

    Args:
        config (ExperimentConfig): Experiment configuration.
        contract (DatasetContract): Dataset contract.
        contract: Included for interface symmetry.

    Returns:
        int: Process exit code.
    """
    del contract
    if not config.pipeline.storage.s3_bucket:
        LOGGER.error("no s3_bucket configured; set pipeline.storage.s3_bucket first")
        return 1
    report = S3Publisher(config.pipeline.storage).doctor()
    LOGGER.info("bucket doctor bucket=%s versioning=%s", report.bucket, report.versioning)
    if report.errors:
        for error in report.errors:
            LOGGER.warning("doctor: %s", error)
        return 1
    if report.versioning != "Enabled":
        LOGGER.warning("bucket versioning is %s", report.versioning)
    if not report.public_access_blocked:
        LOGGER.warning("bucket does not block all public access")
    return 0


def _status(config: ExperimentConfig, contract: DatasetContract) -> int:
    """Print the current manifest status.

    Args:
        config (ExperimentConfig): Experiment configuration.
        contract (DatasetContract): Dataset contract.

    Returns:
        int: Process exit code.
    """
    paths = _paths(config, contract)
    manifest_path = paths["processed"] / "manifest.parquet"
    if not manifest_path.is_file():
        LOGGER.info("no curated manifest at %s", manifest_path)
        return 1
    manifest = read_manifest(manifest_path, paths["processed"] / "manifest.json")
    LOGGER.info(
        "status version=%s records=%s manifest=%s storage=%s",
        manifest.metadata.dataset_version,
        len(manifest.records),
        manifest.metadata.manifest_sha256,
        config.pipeline.storage.mode,
    )
    return 0


def _guarded(
    config: ExperimentConfig,
    action: Callable[[], int],
) -> int:
    """Run a stage while holding the pipeline lock.

    Args:
        config (ExperimentConfig): Experiment configuration.
        action (Callable[[], int]): Zero-argument stage function.

    Returns:
        int: Stage exit code.
    """
    try:
        with _lock(config):
            return int(action())
    except Timeout:
        LOGGER.error("another pipeline process holds the run lock")
        return 1


def _dispatch(arguments: argparse.Namespace) -> int:  # noqa: PLR0911
    """Dispatch a parsed CLI command.

    Args:
        arguments (argparse.Namespace): Parsed arguments.

    Returns:
        int: Process exit code.
    """
    config, contract = _context(arguments.config)
    timestamp = _now()
    run_id = _run_id(timestamp)
    command: str = arguments.command
    if command == "fetch":
        return _guarded(config, lambda: _fetch(config, contract, run_id, timestamp))
    if command == "extract":
        return _guarded(config, lambda: _extract(config, contract, run_id, timestamp))
    if command == "validate":
        return _guarded(config, lambda: _validate(config, contract, run_id, timestamp))
    if command == "split":
        return _guarded(config, lambda: _split(config, contract, run_id, timestamp))
    if command == "curate":
        return _guarded(config, lambda: _curate(config, contract, run_id))
    if command == "verify":
        return _guarded(config, lambda: _verify(config, contract, run_id, timestamp))
    if command == "all":
        stages: list[Callable[[], int]] = [
            lambda: _fetch(config, contract, run_id, timestamp),
            lambda: _extract(config, contract, run_id, timestamp),
            lambda: _validate(config, contract, run_id, timestamp),
            lambda: _split(config, contract, run_id, timestamp),
            lambda: _curate(config, contract, run_id),
            lambda: _verify(config, contract, run_id, timestamp),
        ]

        def run_all() -> int:
            """Run every pipeline stage in dependency order.

            Returns:
                int: First non-zero stage code, or zero on success.
            """
            for stage in stages:
                code = stage()
                if code != 0:
                    return code
            return 0

        return _guarded(config, run_all)
    if command == "publish":
        return _guarded(config, lambda: _publish(config, contract, run_id, bool(arguments.dry_run)))
    if command == "doctor":
        return _doctor(config, contract)
    if command == "status":
        return _status(config, contract)
    LOGGER.error("unknown command: %s", command)
    return 2


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(description="EuroSAT data engineering pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "fetch",
        "extract",
        "validate",
        "split",
        "curate",
        "verify",
        "all",
        "doctor",
        "status",
    ):
        subparsers.add_parser(command)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the data pipeline CLI.

    Args:
        argv (list[str] | None): Optional argument override.

    Returns:
        int: Process exit code.
    """
    configure_logging(logging.INFO)
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return _dispatch(arguments)
    except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
        LOGGER.error("pipeline failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
