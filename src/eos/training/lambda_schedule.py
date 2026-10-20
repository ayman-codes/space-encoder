"""Linear decay schedule for the reconstruction loss weight.

``plan.md`` bounds the reconstruction weight to ``lambda in [0.1, 0.5]`` and
requires a scheduled decay across training. The schedule is expressed in epochs
because the optimizer and learning-rate schedule are also epoch based.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_LAMBDA = 0.1
"""Lower bound of the admissible reconstruction-weight band."""

MAX_LAMBDA = 0.5
"""Upper bound of the admissible reconstruction-weight band."""


@dataclass(frozen=True)
class ReconstructionLambdaSchedule:
    """Linearly decay the reconstruction weight from ``start`` to ``final``.

    Args:
        start (float): Initial weight, in ``[0.1, 0.5]``.
        final (float): Terminal weight, in ``[0.1, 0.5]`` and not above ``start``.

    Raises:
        ValueError: If either weight leaves the band or ``final`` exceeds
            ``start``.
    """

    start: float = 0.3
    final: float = 0.1

    def __post_init__(self) -> None:
        """Validate the decay envelope.

        Returns:
            None: The dataclass is frozen after validation.

        Raises:
            ValueError: If a weight is outside the assignment band or the
                terminal weight exceeds the initial weight.
        """
        if not MIN_LAMBDA <= self.start <= MAX_LAMBDA:
            raise ValueError(f"start must lie in [{MIN_LAMBDA}, {MAX_LAMBDA}]")
        if not MIN_LAMBDA <= self.final <= MAX_LAMBDA:
            raise ValueError(f"final must lie in [{MIN_LAMBDA}, {MAX_LAMBDA}]")
        if self.final > self.start:
            raise ValueError("final must not exceed start")

    def at(self, epoch: int, max_epochs: int) -> float:
        """Return the reconstruction weight for one epoch.

        Args:
            epoch (int): Zero-based training epoch. Values outside the run are
                clamped to the nearest endpoint.
            max_epochs (int): Total scheduled epochs. A value of one or less
                returns ``start`` because no decay interval exists.

        Returns:
            float: Interpolated weight in ``[final, start]``.
        """
        if max_epochs <= 1:
            return self.start
        bounded = min(max(epoch, 0), max_epochs - 1)
        progress = bounded / (max_epochs - 1)
        return self.start + (self.final - self.start) * progress
