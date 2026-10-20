"""Tests for CLI parsing, path resolution, and status commands."""

from __future__ import annotations

from pathlib import Path

import yaml
from tests.helpers import build_contract, zip_bytes

from eos.config import load_config
from eos.dataset.cli import (
    _archive_name,
    _paths,
    _pipeline_version,
    _resolve,
    _run_id,
    build_parser,
    main,
)
from eos.dataset.contract import load_contract

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_parser_accepts_publish_dry_run() -> None:
    """The publish command must expose a dry-run switch."""
    parser = build_parser()
    arguments = parser.parse_args(["publish", "--dry-run"])
    assert arguments.command == "publish"
    assert arguments.dry_run is True


def test_paths_are_versioned_and_resolved() -> None:
    """Pipeline paths must include the dataset version."""
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    contract = load_contract(PROJECT_ROOT / "data" / "contracts" / "eurosat-rgb.contract.yaml")
    paths = _paths(config, contract)
    assert paths["processed"].name == "eurosat-rgb-v1.0.0"
    assert paths["manifest"].name == "eurosat-rgb-v1.0.0"
    assert paths["archive"].name == _archive_name(contract)
    assert _resolve(Path("configs/base.yaml")).is_absolute()


def test_run_id_is_filesystem_safe() -> None:
    """Run identifiers must not contain path separators."""
    identifier = _run_id("2026-01-01T00:00:00+00:00")
    assert "/" not in identifier
    assert ":" not in identifier
    assert identifier.endswith("Z")
    assert _pipeline_version()


def test_doctor_explains_missing_bucket() -> None:
    """Doctor must fail clearly when no S3 bucket is configured."""
    assert main(["doctor"]) == 1


def test_missing_config_is_reported() -> None:
    """A missing configuration path must fail without a traceback."""
    assert main(["--config", "configs/missing.yaml", "status"]) == 1


def test_full_pipeline_cli(tmp_path: Path, httpserver) -> None:
    """The CLI must build a complete synthetic dataset without external services."""
    payload_root = tmp_path / "payload"
    payload = zip_bytes(payload_root, ("Alpha", "Beta"), per_class=20)
    archive = tmp_path / "source.zip"
    archive.write_bytes(payload)
    contract = build_contract(("Alpha", "Beta"), 40, archive=archive)
    httpserver.expect_request("/EuroSAT_RGB.zip").respond_with_data(
        payload, content_type="application/zip"
    )
    source = contract.source.model_copy(update={"urls": (httpserver.url_for("/EuroSAT_RGB.zip"),)})
    contract = contract.model_copy(update={"source": source})
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        yaml.safe_dump(contract.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    manifest_root = tmp_path / "manifests"
    runs = tmp_path / "runs"
    config = {
        "data": {
            "curated_root": str(processed_root / "synthetic-v1.0.0"),
            "batch_size": 4,
            "num_workers": 0,
        },
        "pipeline": {
            "contract_path": str(contract_path),
            "raw_root": str(raw_root),
            "processed_root": str(processed_root),
            "manifest_root": str(manifest_root),
            "quarantine_root": str(tmp_path / "quarantine"),
            "run_root": str(runs),
            "retry": {
                "attempts": 2,
                "base_delay_seconds": 0.01,
                "max_delay_seconds": 0.02,
                "timeout_seconds": 10.0,
            },
            "storage": {"mode": "local"},
        },
    }
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    assert main(["--config", str(config_path), "all"]) == 0
    assert main(["--config", str(config_path), "status"]) == 0
    assert main(["--config", str(config_path), "publish"]) == 0
    assert main(["--config", str(config_path), "doctor"]) == 1
    assert (processed_root / "synthetic-v1.0.0" / "images_uint8.npy").is_file()
    assert (manifest_root / "synthetic-v1.0.0" / "verification.json").is_file()
