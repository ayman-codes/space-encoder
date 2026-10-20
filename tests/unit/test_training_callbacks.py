"""Tests for checkpoint and learning-rate callbacks."""

from __future__ import annotations

from pathlib import Path

import pytest
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from eos.training.callbacks import build_callbacks, checkpoint_monitor
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


def test_build_callbacks_returns_checkpoint_and_lr_monitor(tmp_path: Path) -> None:
    """Both callbacks must be present and configured for epoch logging."""
    callbacks = build_callbacks("hybrid", tmp_path)
    assert len(callbacks) == 2
    checkpoint = callbacks[0]
    assert isinstance(checkpoint, ModelCheckpoint)
    assert checkpoint.monitor == "val/macro_f1"
    assert checkpoint.mode == "max"
    assert checkpoint.save_top_k == 1
    assert checkpoint.save_last is True
    assert checkpoint.dirpath == str(tmp_path)
    assert isinstance(callbacks[1], LearningRateMonitor)
    assert callbacks[1].logging_interval == "epoch"


def test_every_arm_builds_callbacks(tmp_path: Path) -> None:
    """Every registered arm must produce a valid callback list."""
    for mode in TRAINING_MODES:
        callbacks = build_callbacks(mode, tmp_path)
        assert len(callbacks) == 2
