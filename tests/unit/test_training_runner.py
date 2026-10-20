"""Tests for experiment orchestration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eos.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from eos.training import runner as runner_module
from eos.training.runner import (
    _best_checkpoint,
    budget_payload,
    build_trainer,
    normalize_fast_dev_run,
    run_experiment,
    validate_consistency,
    write_json,
)


def _config(tmp_path: Path | None = None) -> ExperimentConfig:
    """Build a CPU experiment configuration for orchestration tests.

    Args:
        tmp_path (Path | None): Optional checkpoint and report root.

    Returns:
        ExperimentConfig: Validated configuration.
    """
    root = tmp_path or Path(".cache/tmp/orchestration-test")
    train = TrainConfig(
        max_epochs=2,
        warmup_epochs=0,
        precision="bf16-mixed",
        accelerator="cpu",
        devices=1,
        seed=7,
        checkpoint_dir=root / "checkpoints",
        report_dir=root / "reports",
        mlflow_enabled=False,
        job_metrics_enabled=False,
    )
    return ExperimentConfig(train=train)


def test_fast_dev_run_normalizes_epochs_and_warmup() -> None:
    """A fast run must fit in one epoch without a warmup stage."""
    config = _config().model_copy(
        update={"train": _config().train.model_copy(update={"fast_dev_run": True})}
    )
    normalized = normalize_fast_dev_run(config)
    assert normalized.train.max_epochs == 1
    assert normalized.train.warmup_epochs == 0
    assert normalized.train.fast_dev_run is True


def test_normal_configuration_is_returned_unchanged() -> None:
    """The normalizer must not alter a full run."""
    config = _config()
    assert normalize_fast_dev_run(config) is config


def test_consistency_rejects_mismatched_sections() -> None:
    """Class, size, and channel mismatches must fail before training."""
    base = _config()
    mismatched_classes = base.model_copy(
        update={"data": DataConfig(num_classes=4), "model": ModelConfig(num_classes=10)}
    )
    with pytest.raises(ValueError, match="num_classes"):
        validate_consistency(mismatched_classes)
    mismatched_size = base.model_copy(
        update={"data": DataConfig(image_size=32), "model": ModelConfig(input_size=64)}
    )
    with pytest.raises(ValueError, match="image_size"):
        validate_consistency(mismatched_size)
    mismatched_channels = base.model_copy(
        update={"data": DataConfig(channels=4), "model": ModelConfig(in_channels=3)}
    )
    with pytest.raises(ValueError, match="channels"):
        validate_consistency(mismatched_channels)
    validate_consistency(base)


def test_budget_payload_contains_the_shared_fields() -> None:
    """Every field needed to prove arm parity must be present."""
    payload = budget_payload(_config())
    assert payload["max_epochs"] == 2
    assert payload["batch_size"] == 128
    assert payload["learning_rate"] == pytest.approx(1e-3)
    assert payload["warmup_epochs"] == 0
    assert payload["seed"] == 7
    assert payload["precision"] == "bf16-mixed"
    assert payload["class_weights"] is False


def test_write_json_replaces_atomically_and_leaves_no_temporary(tmp_path: Path) -> None:
    """The writer must create parent directories and clean its temp sibling."""
    destination = tmp_path / "nested" / "payload.json"
    write_json(destination, {"b": 2, "a": 1})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert list(destination.parent.iterdir()) == [destination]


def test_build_trainer_resolves_cpu_precision_and_limits(tmp_path: Path) -> None:
    """A bf16 request on CPU must resolve to float32 with one-batch limits."""
    config = _config(tmp_path).model_copy(
        update={"train": _config(tmp_path).train.model_copy(update={"fast_dev_run": True})}
    )
    trainer = build_trainer("hybrid", config)
    assert trainer.precision == "32-true"
    assert trainer.gradient_clip_val == pytest.approx(1.0)
    assert trainer.limit_train_batches == 1
    assert trainer.limit_val_batches == 1
    assert len(trainer.callbacks) >= 2
    assert _best_checkpoint(trainer) is None


def test_build_trainer_uses_autoencoder_monitor(tmp_path: Path) -> None:
    """The autoencoder arm must monitor reconstruction error."""
    trainer = build_trainer("autoencoder", _config(tmp_path))
    checkpoint = trainer.checkpoint_callback
    assert checkpoint is not None
    assert checkpoint.monitor == "val/recon_mse"
    assert checkpoint.mode == "min"


def test_best_checkpoint_ignores_non_model_checkpoint_callbacks() -> None:
    """A trainer without a ModelCheckpoint must report no checkpoint."""
    trainer = SimpleNamespace(checkpoint_callback=None)
    assert _best_checkpoint(trainer) is None  # type: ignore[arg-type]


def test_run_experiment_rejects_unknown_mode() -> None:
    """An unsupported arm must fail before reading the dataset."""
    with pytest.raises(ValueError, match="unsupported"):
        run_experiment("segmentation", _config())  # type: ignore[arg-type]


def test_run_experiment_requires_an_existing_resume_checkpoint(tmp_path: Path) -> None:
    """A missing resume checkpoint must fail before data setup."""
    with pytest.raises(FileNotFoundError, match="resume"):
        run_experiment("hybrid", _config(tmp_path), resume=tmp_path / "missing.ckpt")


def test_run_experiment_requires_a_loaded_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A data module that silently skips setup must fail loudly."""

    class FakeDataModule:
        """Data module double with no manifest."""

        def __init__(self, config: DataConfig) -> None:
            """Store the configuration and expose an empty manifest.

            Args:
                config (DataConfig): Data configuration.

            Returns:
                None: The double is initialized in place.
            """
            del config
            self.manifest = None

        def setup(self) -> None:
            """Do nothing.

            Returns:
                None: Setup is intentionally skipped.
            """

    monkeypatch.setattr(runner_module, "EuroSATDataModule", FakeDataModule)
    with pytest.raises(RuntimeError, match="manifest"):
        run_experiment("hybrid", _config())
