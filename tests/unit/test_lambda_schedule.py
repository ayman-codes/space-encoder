"""Tests for the reconstruction-weight decay schedule."""

from __future__ import annotations

import pytest

from eos.training.lambda_schedule import ReconstructionLambdaSchedule


def test_schedule_reaches_both_endpoints() -> None:
    """The first epoch uses the start weight and the last uses the final weight."""
    schedule = ReconstructionLambdaSchedule(start=0.3, final=0.1)
    assert schedule.at(0, 11) == pytest.approx(0.3)
    assert schedule.at(10, 11) == pytest.approx(0.1)


def test_schedule_is_linear_at_the_midpoint() -> None:
    """The interpolation must be linear in epoch index."""
    schedule = ReconstructionLambdaSchedule(start=0.5, final=0.1)
    assert schedule.at(2, 5) == pytest.approx(0.3)


def test_schedule_clamps_and_handles_a_single_epoch() -> None:
    """Out-of-range epochs are clamped and one epoch returns the start weight."""
    schedule = ReconstructionLambdaSchedule(start=0.4, final=0.2)
    assert schedule.at(-5, 4) == pytest.approx(0.4)
    assert schedule.at(99, 4) == pytest.approx(0.2)
    assert schedule.at(0, 1) == pytest.approx(0.4)
    assert schedule.at(0, 0) == pytest.approx(0.4)


def test_invalid_schedules_are_rejected() -> None:
    """Band violations and increasing schedules must fail at construction."""
    with pytest.raises(ValueError, match="start"):
        ReconstructionLambdaSchedule(start=0.7, final=0.1)
    with pytest.raises(ValueError, match="final"):
        ReconstructionLambdaSchedule(start=0.3, final=0.05)
    with pytest.raises(ValueError, match="exceed"):
        ReconstructionLambdaSchedule(start=0.2, final=0.3)
