"""Tests for the callback stack governing selection and stopping."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from eos.config import TrainConfig
from eos.training.callbacks import TrailingMeanCallback, build_callbacks, checkpoint_monitor
from eos.training.lightning_module import TRAINING_MODES, TrainingMode


@pytest.mark.parametrize(
    ("mode", "monitor", "direction"),
    [
        ("hybrid", "val/macro_f1", "max"),
        ("classifier", "val/macro_f1", "max"),
        ("autoencoder", "val/recon_mse", "min"),
    ],
)
def test_checkpoint_monitor_matches_the_arm(
    mode: TrainingMode,
    monitor: str,
    direction: str,
) -> None:
    """The monitored metric must exist for the arm being trained."""
    assert checkpoint_monitor(mode) == (monitor, direction)


def test_build_callbacks_orders_smoothing_before_the_controls(tmp_path: Path) -> None:
    """The smoothed metric must be published before selection and stopping read it."""
    callbacks = build_callbacks("hybrid", tmp_path, TrainConfig())
    assert len(callbacks) == 4
    smoothing = callbacks[0]
    assert isinstance(smoothing, TrailingMeanCallback)
    assert smoothing.source == "val/macro_f1"
    assert smoothing.target == "val/macro_f1_smooth"
    assert smoothing.window == 3
    checkpoint = callbacks[1]
    assert isinstance(checkpoint, ModelCheckpoint)
    assert checkpoint.monitor == "val/macro_f1_smooth"
    assert checkpoint.mode == "max"
    assert checkpoint.save_top_k == 1
    assert checkpoint.save_last is True
    assert checkpoint.dirpath == str(tmp_path)
    stopping = callbacks[2]
    assert isinstance(stopping, EarlyStopping)
    assert stopping.monitor == "val/macro_f1_smooth"
    assert stopping.mode == "max"
    assert stopping.patience == 12
    assert stopping.min_delta == pytest.approx(1e-3)
    assert isinstance(callbacks[3], LearningRateMonitor)
    assert callbacks[3].logging_interval == "epoch"


def test_smoothing_window_of_one_disables_the_trailing_mean(tmp_path: Path) -> None:
    """A one-sample window must monitor the raw metric with no extra callback."""
    config = TrainConfig(monitor_smoothing_window=1)
    callbacks = build_callbacks("classifier", tmp_path, config)
    assert len(callbacks) == 3
    checkpoint = callbacks[0]
    assert isinstance(checkpoint, ModelCheckpoint)
    assert checkpoint.monitor == "val/macro_f1"


def test_autoencoder_stops_on_reconstruction_with_its_own_bound(tmp_path: Path) -> None:
    """The reconstruction arm must use its scale-appropriate improvement bound."""
    config = TrainConfig()
    callbacks = build_callbacks("autoencoder", tmp_path, config)
    stopping = next(callback for callback in callbacks if isinstance(callback, EarlyStopping))
    assert stopping.monitor == "val/recon_mse_smooth"
    assert stopping.mode == "min"
    assert abs(stopping.min_delta) == pytest.approx(config.early_stopping_min_delta_recon)


def test_every_arm_builds_callbacks(tmp_path: Path) -> None:
    """Every registered arm must produce a valid callback stack."""
    for mode in TRAINING_MODES:
        callbacks = build_callbacks(mode, tmp_path, TrainConfig())
        assert len(callbacks) == 4


def _fake_trainer(values: dict[str, float]) -> SimpleNamespace:
    """Build a trainer stand-in exposing callback metrics and a root device.

    Args:
        values (dict[str, float]): Initial callback metrics.

    Returns:
        SimpleNamespace: Trainer stand-in accepted by the callback hook.
    """
    return SimpleNamespace(
        callback_metrics=dict(values), strategy=SimpleNamespace(root_device="cpu")
    )


def test_trailing_mean_publishes_the_window_average() -> None:
    """The published value must be the mean of the retained window."""
    smoothing = TrailingMeanCallback("val/macro_f1", "val/macro_f1_smooth", 3)
    trainer = _fake_trainer({"val/macro_f1": 0.90})
    smoothing.on_validation_epoch_end(trainer, SimpleNamespace())  # type: ignore[arg-type]
    assert float(trainer.callback_metrics["val/macro_f1_smooth"]) == pytest.approx(0.90)
    trainer.callback_metrics["val/macro_f1"] = 0.96
    smoothing.on_validation_epoch_end(trainer, SimpleNamespace())  # type: ignore[arg-type]
    assert float(trainer.callback_metrics["val/macro_f1_smooth"]) == pytest.approx(0.93)
    trainer.callback_metrics["val/macro_f1"] = 0.99
    smoothing.on_validation_epoch_end(trainer, SimpleNamespace())  # type: ignore[arg-type]
    assert float(trainer.callback_metrics["val/macro_f1_smooth"]) == pytest.approx(0.95)
    trainer.callback_metrics["val/macro_f1"] = 0.60
    smoothing.on_validation_epoch_end(trainer, SimpleNamespace())  # type: ignore[arg-type]
    assert float(trainer.callback_metrics["val/macro_f1_smooth"]) == pytest.approx(
        (0.96 + 0.99 + 0.60) / 3
    )


def test_trailing_mean_ignores_a_missing_metric() -> None:
    """A run without the raw metric must publish nothing instead of failing."""
    smoothing = TrailingMeanCallback("val/macro_f1", "val/macro_f1_smooth", 3)
    trainer = _fake_trainer({})
    smoothing.on_validation_epoch_end(trainer, SimpleNamespace())  # type: ignore[arg-type]
    assert "val/macro_f1_smooth" not in trainer.callback_metrics


def test_trailing_mean_state_round_trips_through_checkpoints() -> None:
    """A resumed run must continue the window instead of restarting it."""
    smoothing = TrailingMeanCallback("val/macro_f1", "val/macro_f1_smooth", 3)
    trainer = _fake_trainer({"val/macro_f1": 0.90})
    smoothing.on_validation_epoch_end(trainer, SimpleNamespace())  # type: ignore[arg-type]
    trainer.callback_metrics["val/macro_f1"] = 0.96
    smoothing.on_validation_epoch_end(trainer, SimpleNamespace())  # type: ignore[arg-type]
    restored = TrailingMeanCallback("val/macro_f1", "val/macro_f1_smooth", 3)
    restored.load_state_dict(smoothing.state_dict())
    trainer.callback_metrics["val/macro_f1"] = 0.99
    restored.on_validation_epoch_end(trainer, SimpleNamespace())  # type: ignore[arg-type]
    assert float(trainer.callback_metrics["val/macro_f1_smooth"]) == pytest.approx(0.95)


def test_trailing_mean_rejects_an_empty_window() -> None:
    """A window below one is meaningless and must fail at construction."""
    with pytest.raises(ValueError, match="window"):
        TrailingMeanCallback("val/macro_f1", "val/macro_f1_smooth", 0)


def test_trailing_mean_publishes_a_tensor() -> None:
    """The published value must be a tensor so Lightning accepts it as a metric."""
    smoothing = TrailingMeanCallback("val/macro_f1", "val/macro_f1_smooth", 2)
    trainer = _fake_trainer({"val/macro_f1": 0.5})
    smoothing.on_validation_epoch_end(trainer, SimpleNamespace())  # type: ignore[arg-type]
    assert isinstance(trainer.callback_metrics["val/macro_f1_smooth"], torch.Tensor)
