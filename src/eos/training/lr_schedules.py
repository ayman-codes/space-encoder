"""Metric-driven learning-rate decay preceded by a linear warmup.

The convergence regime replaces the fixed cosine decay with
:class:`torch.optim.lr_scheduler.ReduceLROnPlateau` so the optimizer keeps
training at the peak rate while the monitored validation metric still moves and
decays only when the metric stalls. This module preserves the linear warmup of
the fixed schedule by wrapping the plateau scheduler in a ramp that owns the
first ``warmup_epochs`` steps.

The wrapper is a real :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`
subclass, so Lightning classifies it as a plateau scheduler and steps it after
validation with the monitored metric, and its warmup state round-trips through
Lightning checkpoints through the inherited ``state_dict``.
"""

from __future__ import annotations

from typing import Literal, SupportsFloat

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmupReduceOnPlateau(ReduceLROnPlateau):
    """Linear warmup followed by plateau learning-rate decay.

    The first ``warmup_epochs`` calls to :meth:`step` raise every parameter
    group's learning rate linearly from ``warmup_start_factor * base_lr`` to
    ``base_lr``, ignoring the monitored metric. Every later call delegates to
    :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`, which decays the rate
    by ``factor`` once ``patience`` consecutive checks show no improvement
    larger than the plateau threshold.

    Args:
        optimizer (Optimizer): Optimizer whose parameter groups are scheduled.
        warmup_epochs (int): Number of warmup steps. Zero starts directly in
            the plateau regime.
        warmup_start_factor (float): Fraction of the base learning rate used at
            the first warmup step, in ``(0, 1]``.
        mode (Literal["min", "max"]): Whether smaller or larger monitored values
            are better. ``"max"`` for macro-F1, ``"min"`` for reconstruction MSE.
        factor (float): Multiplicative decay applied on a plateau, in ``(0, 1)``.
        patience (int): Consecutive checks without improvement before decay.
        min_lr (float): Lower bound for every parameter group's learning rate.

    Raises:
        ValueError: If ``warmup_epochs`` is negative, ``warmup_start_factor``
            leaves ``(0, 1]``, or the plateau parameters are out of range.
    """

    def __init__(  # noqa: PLR0913
        self,
        optimizer: Optimizer,
        *,
        warmup_epochs: int = 0,
        warmup_start_factor: float = 0.1,
        mode: Literal["min", "max"] = "min",
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 0.0,
    ) -> None:
        """Validate the warmup envelope and attach the plateau regime.

        Args:
            optimizer (Optimizer): Optimizer whose parameter groups are
                scheduled.
            warmup_epochs (int): Number of warmup steps.
            warmup_start_factor (float): Fraction of the base learning rate used
                at the first warmup step.
            mode (Literal["min", "max"]): Optimization direction of the
                monitored metric.
            factor (float): Multiplicative decay applied on a plateau.
            patience (int): Checks without improvement before decay.
            min_lr (float): Lower bound for every parameter group's learning
                rate.

        Returns:
            None: The scheduler is initialized in place.

        Raises:
            ValueError: If a warmup or plateau parameter is out of range.
        """
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if not 0.0 < warmup_start_factor <= 1.0:
            raise ValueError("warmup_start_factor must lie in (0, 1]")
        super().__init__(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor
        self.warmup_base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, metrics: SupportsFloat, epoch: int | None = None) -> None:  # type: ignore[override]
        """Advance the warmup ramp or step the plateau regime.

        Args:
            metrics (SupportsFloat): Monitored validation metric. Ignored during
                warmup and required afterwards.
            epoch (int | None): Optional explicit epoch index, deprecated by
                PyTorch; omit it and let the scheduler count.

        Returns:
            None: Parameter-group learning rates are updated in place.
        """
        if self.last_epoch < self.warmup_epochs:
            self._step_warmup()
            return
        super().step(metrics, epoch=epoch)

    def _step_warmup(self) -> None:
        """Raise every parameter group's learning rate by one ramp step.

        Returns:
            None: Learning rates move from ``warmup_start_factor * base_lr``
            towards ``base_lr`` and the last rates are published.
        """
        self.last_epoch += 1
        progress = self.last_epoch / self.warmup_epochs
        factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
        for group, base_lr in zip(
            self.optimizer.param_groups,
            self.warmup_base_lrs,
            strict=True,
        ):
            group["lr"] = base_lr * factor
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
