"""Tests for dataset contract loading and invariants."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from eos.dataset.contract import DatasetContract, load_contract

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PROJECT_ROOT / "data" / "contracts" / "eurosat-rgb.contract.yaml"


def test_load_eurosat_contract() -> None:
    """The repository contract must describe the pinned EuroSAT source."""
    contract = load_contract(CONTRACT_PATH)
    assert contract.dataset_version == "eurosat-rgb-v1.0.0"
    assert contract.source.expected_images == 27000
    assert contract.source.class_names[0] == "AnnualCrop"
    assert contract.class_ids["SeaLake"] == 9
    assert len(contract.fingerprint()) == 64


def test_contract_fingerprint_is_stable() -> None:
    """Contract fingerprints must not depend on call order."""
    left = load_contract(CONTRACT_PATH)
    right = load_contract(CONTRACT_PATH)
    assert left.fingerprint() == right.fingerprint()


def test_missing_contract_is_rejected() -> None:
    """A missing contract path must raise a clear error."""
    with pytest.raises(FileNotFoundError):
        load_contract(PROJECT_ROOT / "data" / "contracts" / "missing.yaml")


def test_unsorted_class_names_are_rejected() -> None:
    """Class ordering defines class identifiers and must be deterministic."""
    payload = load_contract(CONTRACT_PATH).model_dump(mode="json")
    payload["source"]["class_names"] = ["SeaLake", "AnnualCrop"]
    payload["source"]["expected_classes"] = 2
    with pytest.raises(ValidationError):
        DatasetContract.model_validate(payload)


def test_split_fractions_must_sum_to_one() -> None:
    """A split that does not partition the dataset must be rejected."""
    payload = load_contract(CONTRACT_PATH).model_dump(mode="json")
    payload["split"]["train_fraction"] = 0.5
    with pytest.raises(ValidationError):
        DatasetContract.model_validate(payload)
