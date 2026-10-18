"""Tests for the strict configuration contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from eos.config import (
    ASVDConfig,
    DataConfig,
    ExperimentConfig,
    MMDConfig,
    ModelConfig,
    ServiceSettings,
    TrainConfig,
    load_config,
)


def test_defaults_satisfy_every_contract() -> None:
    """The default experiment configuration must validate without overrides."""
    config = ExperimentConfig()
    assert config.model.latent_dim == 128
    assert config.data.image_size == 64
    assert config.train.precision == "bf16-mixed"


def test_base_yaml_matches_defaults() -> None:
    """The shipped base configuration must load from disk."""
    config = load_config(Path("configs/base.yaml"))
    assert config.model.downsample_stages == 4
    assert config.asvd.preserve_energy == pytest.approx(0.95)
    assert config.mmd.bandwidth_multipliers[0] == pytest.approx(0.25)


def test_missing_file_raises() -> None:
    """A missing configuration file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(Path("configs/does-not-exist.yaml"))


def test_unknown_key_is_rejected() -> None:
    """Typo keys must fail instead of being silently ignored."""
    with pytest.raises(ValidationError):
        DataConfig.model_validate({"batch_size": 16, "batc_size": 32})


def test_split_fractions_must_leave_training_data() -> None:
    """Validation and test shares cannot consume the whole dataset."""
    with pytest.raises(ValidationError):
        DataConfig(val_fraction=0.6, test_fraction=0.4)


def test_normalization_must_match_channels() -> None:
    """Per-channel statistics must align with the input channel count."""
    with pytest.raises(ValidationError):
        DataConfig(channels=3, mean=(0.5, 0.5), std=(0.5, 0.5, 0.5))


def test_non_positive_std_is_rejected() -> None:
    """A zero standard deviation would divide by zero during normalization."""
    with pytest.raises(ValidationError):
        DataConfig(std=(0.5, 0.0, 0.5))


def test_reconstruction_lambda_bounds() -> None:
    """The reconstruction weight must stay inside the assignment band."""
    with pytest.raises(ValidationError):
        ModelConfig(recon_lambda=0.9)


def test_reconstruction_alpha_bounds() -> None:
    """The MSE/SSIM mixing weight must stay inside [0, 1]."""
    with pytest.raises(ValidationError):
        ModelConfig(recon_alpha=1.5)


def test_spatial_symmetry_is_enforced() -> None:
    """An input size that cannot be halved four times must fail."""
    with pytest.raises(ValidationError):
        ModelConfig(input_size=70, downsample_stages=4)


def test_warmup_must_fit_epochs() -> None:
    """Warmup cannot exceed the total epoch budget."""
    with pytest.raises(ValidationError):
        TrainConfig(max_epochs=2, warmup_epochs=2)


def test_asvd_energy_band() -> None:
    """ASVD energy retention must stay inside [0.90, 0.95]."""
    with pytest.raises(ValidationError):
        ASVDConfig(preserve_energy=0.8)


def test_mmd_bandwidths_must_increase() -> None:
    """The multi-scale bandwidth grid must be strictly increasing."""
    with pytest.raises(ValidationError):
        MMDConfig(bandwidth_multipliers=(1.0, 0.5))


def test_service_settings_ignore_unrelated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrelated environment variables must not break settings loading."""
    monkeypatch.setenv("GITHUB_TOKEN", "unrelated")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    settings = ServiceSettings()
    assert settings.mlflow_tracking_uri == "http://mlflow:5000"
