"""Callbacks shared by the three ablation arms.

Every control surface — checkpoint selection, early stopping, and the plateau
learning rate — observes one monitored metric per arm. When the configured
smoothing window exceeds one, a trailing-mean callback publishes that metric's
smoothed variant at validation epoch end, before any control callback evaluates
it at validation end, so selection and stopping act on a signal that epoch-level
training noise cannot dominate.
"""

from __future__ import annotations

from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from eos.config import TrainConfig
from eos.training.lightning_module import (
    TrainingMode,
    checkpoint_monitor,
    monitor_name,
    stopping_min_delta,
)

__all__ = ["TrainingMode", "build_callbacks", "checkpoint_monitor", "monitor_name"]


class TrailingMeanCallback(Callback):
    """Publish a trailing mean of the monitored validation metric.

    The retained history is exported through the Lightning checkpoint state, so
    a resumed run continues its window instead of restarting it.

    Args:
        source (str): Raw metric name logged at validation epoch end.
        target (str): Smoothed metric name published next to it.
        window (int): Number of most recent values entering the mean.

    Raises:
        ValueError: If ``window`` is below one.
    """

    def __init__(self, source: str, target: str, window: int) -> None:
        """Store the metric names and the smoothing window.

        Args:
            source (str): Raw metric name logged at validation epoch end.
            target (str): Smoothed metric name published next to it.
            window (int): Number of most recent values entering the mean.

        Returns:
            None: The callback is initialized with an empty history.

        Raises:
            ValueError: If ``window`` is below one.
        """
        if window < 1:
            raise ValueError("window must be at least one")
        self.source = source
        self.target = target
        self.window = window
        self._values: list[float] = []

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Append the raw value and publish the trailing mean.

        Args:
            trainer (Trainer): Active Lightning trainer.
            pl_module (LightningModule): Module being trained.

        Returns:
            None: The smoothed value is written to ``trainer.callback_metrics``.
        """
        del pl_module
        value = trainer.callback_metrics.get(self.source)
        if value is None:
            return
        self._values.append(float(value))
        del self._values[: -self.window]
        trainer.callback_metrics[self.target] = torch.tensor(
            fmean(self._values),
            device=trainer.strategy.root_device,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return the smoothing history for checkpointing.

        Returns:
            dict[str, Any]: Mapping holding the retained raw values.
        """
        return {"values": list(self._values)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the smoothing history from a checkpoint.

        Args:
            state_dict (dict[str, Any]): Previously exported state.

        Returns:
            None: The history is restored in place.
        """
        self._values = [float(value) for value in state_dict.get("values", [])]


def build_callbacks(
    mode: TrainingMode,
    checkpoint_dir: Path,
    config: TrainConfig,
) -> list[Callback]:
    """Build the callback stack shared by every ablation arm.

    The stack is ordered so the trailing mean is published before the control
    callbacks evaluate at validation end: smoothing, checkpointing, early
    stopping, and learning-rate monitoring. Checkpointing and stopping monitor
    the same metric with the same direction; the autoencoder arm stops on
    validation reconstruction MSE with its own improvement bound.

    Args:
        mode (TrainingMode): Ablation arm selector.
        checkpoint_dir (Path): Directory that receives ``.ckpt`` files.
        config (TrainConfig): Trainer configuration carrying the stopping rule,
            the smoothing window, and the per-arm improvement bounds.

    Returns:
        list[Callback]: Callback stack in execution order.
    """
    monitor, direction = checkpoint_monitor(mode)
    smoothed = monitor_name(mode, config)
    callbacks: list[Callback] = []
    if smoothed != monitor:
        callbacks.append(TrailingMeanCallback(monitor, smoothed, config.monitor_smoothing_window))
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{mode}-{{epoch:02d}}",
            monitor=smoothed,
            mode=direction,
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
            enable_version_counter=False,
        )
    )
    callbacks.append(
        EarlyStopping(
            monitor=smoothed,
            mode=direction,
            patience=config.early_stopping_patience,
            min_delta=stopping_min_delta(mode, config),
            check_finite=True,
        )
    )
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    return callbacks
