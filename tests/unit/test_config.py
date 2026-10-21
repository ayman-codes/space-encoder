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
    assert config.model.recon_lambda_final == pytest.approx(0.1)
    assert config.train.checkpoint_dir == Path("checkpoints")
    assert config.train.report_dir == Path("reports/training")
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


def test_normalization_keys_are_rejected() -> None:
    """Radiometric normalization statistics are not part of the contract.

    Inputs stay in ``[0, 1]`` because the decoder is sigmoid bounded, so a
    configuration that reintroduces ``mean`` or ``std`` must fail loudly.
    """
    with pytest.raises(ValidationError):
        DataConfig.model_validate({"mean": (0.5, 0.5, 0.5)})
    with pytest.raises(ValidationError):
        DataConfig.model_validate({"std": (0.5, 0.5, 0.5)})


def test_reconstruction_lambda_bounds() -> None:
    """The reconstruction weight must stay inside the assignment band."""
    with pytest.raises(ValidationError):
        ModelConfig(recon_lambda=0.9)


def test_reconstruction_final_lambda_bounds() -> None:
    """The terminal reconstruction weight must stay inside the same band."""
    with pytest.raises(ValidationError):
        ModelConfig(recon_lambda_final=0.9)
    with pytest.raises(ValidationError):
        ModelConfig(recon_lambda_final=0.05)


def test_reconstruction_lambda_schedule_must_decay() -> None:
    """The terminal weight cannot exceed the initial weight."""
    with pytest.raises(ValidationError):
        ModelConfig(recon_lambda=0.2, recon_lambda_final=0.3)
    config = ModelConfig(recon_lambda=0.3, recon_lambda_final=0.1)
    assert config.recon_lambda_final == pytest.approx(0.1)


def test_training_output_defaults() -> None:
    """Training paths and observability switches must have contract defaults."""
    config = TrainConfig()
    assert config.checkpoint_dir == Path("checkpoints")
    assert config.report_dir == Path("reports/training")
    assert config.mlflow_experiment == "eos-autoencoder"
    assert config.mlflow_enabled is True
    assert config.job_metrics_enabled is True


def test_reconstruction_alpha_bounds() -> None:
    """The MSE/SSIM mixing weight must stay inside [0, 1]."""
    with pytest.raises(ValidationError):
        ModelConfig(recon_alpha=1.5)


def test_spatial_symmetry_is_enforced() -> None:
    """An input size that cannot be halved four times must fail."""
    with pytest.raises(ValidationError):
        ModelConfig(input_size=70, downsample_stages=4)


def test_ssim_window_must_be_odd_and_fit() -> None:
    """The SSIM window needs a center pixel and must fit the patch."""
    with pytest.raises(ValidationError):
        ModelConfig(ssim_window=10)
    with pytest.raises(ValidationError):
        ModelConfig(ssim_window=65)


def test_ssim_defaults_are_boundary_aware() -> None:
    """The default policy must preserve full coverage without synthetic pixels."""
    config = ModelConfig()
    assert config.ssim_boundary == "masked"
    assert config.ssim_window == 11
    assert config.ssim_sigma == pytest.approx(1.5)
    assert config.head_hidden == 64
    assert config.in_channels == 3


def test_ssim_boundary_is_constrained() -> None:
    """Only the three documented policies are admissible."""
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"ssim_boundary": "reflect"})


def test_warmup_must_fit_epochs() -> None:
    """Warmup cannot exceed the total epoch budget."""
    with pytest.raises(ValidationError):
        TrainConfig(max_epochs=2, warmup_epochs=2)


def test_convergence_regime_defaults() -> None:
    """The stopping and scheduling knobs must carry the planned defaults."""
    config = TrainConfig()
    assert config.lr_scheduler == "cosine"
    assert config.plateau_factor == pytest.approx(0.5)
    assert config.plateau_patience == 5
    assert config.early_stopping_patience == 12
    assert config.early_stopping_min_delta == pytest.approx(1e-3)
    assert config.early_stopping_min_delta_recon == pytest.approx(1e-5)
    assert config.monitor_smoothing_window == 3


def test_plateau_floor_must_sit_below_the_peak_rate() -> None:
    """A plateau floor at the peak rate would freeze the learning rate."""
    with pytest.raises(ValidationError):
        TrainConfig(lr_scheduler="plateau", learning_rate=1e-3, plateau_min_lr=1e-3)


def test_scheduler_choice_is_constrained() -> None:
    """An unknown decay regime must fail at load time."""
    with pytest.raises(ValidationError):
        TrainConfig(lr_scheduler="exponential")


def test_smoothing_window_cannot_be_empty() -> None:
    """A trailing mean needs at least one sample."""
    with pytest.raises(ValidationError):
        TrainConfig(monitor_smoothing_window=0)


def test_run_tag_names_a_variant_without_loosening_the_contract() -> None:
    """The variant slug must be optional and strictly slug-shaped."""
    assert TrainConfig().run_tag == ""
    assert TrainConfig(run_tag="lambda03").run_tag == "lambda03"
    with pytest.raises(ValidationError):
        TrainConfig(run_tag="Lambda 03")
    with pytest.raises(ValidationError):
        TrainConfig(run_tag="../escape")


def test_base_yaml_declares_the_convergence_regime() -> None:
    """The shipped configuration must run to convergence, not to a fixed budget."""
    config = load_config(Path("configs/base.yaml"))
    assert config.train.max_epochs == 100
    assert config.train.lr_scheduler == "plateau"
    assert config.train.early_stopping_patience == 12
    assert config.train.monitor_smoothing_window == 3


def test_asvd_energy_band() -> None:
    """ASVD energy retention must stay inside [0.90, 0.95]."""
    with pytest.raises(ValidationError):
        ASVDConfig(preserve_energy=0.8)


def test_asvd_scope_defaults_to_bottleneck() -> None:
    """The bottleneck projections carry the compressible parameter mass."""
    assert ASVDConfig().scope == "bottleneck"


def test_asvd_scope_is_constrained() -> None:
    """Only the three documented layer sets are admissible."""
    with pytest.raises(ValidationError):
        ASVDConfig.model_validate({"scope": "everything"})


def test_compression_gain_rejects_trivial_factorization() -> None:
    """A gain below the configured floor must be rejected at config time."""
    with pytest.raises(ValidationError):
        ASVDConfig(min_compression_gain=0.9)


def test_class_weighting_is_off_by_default() -> None:
    """Every ablation arm must share the same unweighted objective."""
    assert TrainConfig().class_weights is False


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
