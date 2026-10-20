"""Callbacks shared by the three ablation arms.

The checkpoint callback must monitor a metric that the arm actually produces.
The classifier and hybrid arms monitor validation macro-F1; the autoencoder arm
has no classifier, so it monitors validation reconstruction MSE instead.
"""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint

from eos.training.lightning_module import TrainingMode


def checkpoint_monitor(mode: TrainingMode) -> tuple[str, str]:
    """Return the monitored metric and optimisation direction for an arm.

    Args:
        mode (TrainingMode): Ablation arm selector.

    Returns:
        tuple[str, str]: Metric name and ``"max"`` or ``"min"``.
    """
    if mode == "autoencoder":
        return "val/recon_mse", "min"
    return "val/macro_f1", "max"


def build_callbacks(mode: TrainingMode, checkpoint_dir: Path) -> list[Callback]:
    """Build checkpointing and learning-rate monitoring callbacks.

    Args:
        mode (TrainingMode): Ablation arm selector.
        checkpoint_dir (Path): Directory that receives ``.ckpt`` files.

    Returns:
        list[Callback]: Checkpoint callback followed by the LR monitor.
    """
    monitor, direction = checkpoint_monitor(mode)
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{mode}-{{epoch:02d}}",
        monitor=monitor,
        mode=direction,
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    return [checkpoint, LearningRateMonitor(logging_interval="epoch")]
