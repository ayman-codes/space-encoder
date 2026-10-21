"""Tests for the warmup-plus-plateau learning-rate schedule."""

from __future__ import annotations

import pytest
import torch

from eos.training.lr_schedules import WarmupReduceOnPlateau


def _optimizer(lr: float = 1e-3) -> torch.optim.Optimizer:
    """Build a one-parameter optimizer at a known learning rate.

    Args:
        lr (float): Initial learning rate.

    Returns:
        torch.optim.Optimizer: SGD over a single parameter.
    """
    parameter = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([parameter], lr=lr)


def test_warmup_ramps_linearly_to_the_base_rate() -> None:
    """Each warmup step must move the rate towards the base rate."""
    optimizer = _optimizer()
    scheduler = WarmupReduceOnPlateau(
        optimizer,
        warmup_epochs=2,
        warmup_start_factor=0.1,
    )
    scheduler.step(0.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5.5e-4)
    scheduler.step(0.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert scheduler.get_last_lr() == [pytest.approx(1e-3)]


def test_plateau_is_ignored_during_warmup() -> None:
    """A stalled metric must not decay the rate before warmup finishes."""
    optimizer = _optimizer()
    scheduler = WarmupReduceOnPlateau(
        optimizer,
        warmup_epochs=3,
        warmup_start_factor=0.1,
        factor=0.5,
        patience=1,
    )
    for _ in range(3):
        scheduler.step(0.5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert scheduler.last_epoch == 3


def test_plateau_decays_only_after_the_patience_expires() -> None:
    """A stalled metric must decay the rate once the patience runs out."""
    optimizer = _optimizer()
    scheduler = WarmupReduceOnPlateau(
        optimizer,
        warmup_epochs=0,
        factor=0.5,
        patience=2,
        mode="max",
    )
    for _ in range(3):
        scheduler.step(0.5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    scheduler.step(0.5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)


def test_improving_metrics_reset_the_patience() -> None:
    """A metric that keeps moving must keep the peak learning rate."""
    optimizer = _optimizer()
    scheduler = WarmupReduceOnPlateau(
        optimizer,
        warmup_epochs=0,
        factor=0.5,
        patience=1,
        mode="max",
    )
    for value in (0.1, 0.2, 0.3, 0.4):
        scheduler.step(value)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_the_floor_stops_the_decay() -> None:
    """The decay must never take a parameter group below the floor."""
    optimizer = _optimizer()
    scheduler = WarmupReduceOnPlateau(
        optimizer,
        warmup_epochs=0,
        factor=0.5,
        patience=0,
        mode="max",
        min_lr=8e-4,
    )
    for _ in range(4):
        scheduler.step(0.5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(8e-4)


def test_state_round_trip_restores_the_warmup_position() -> None:
    """A resumed run must continue the ramp instead of restarting it."""
    optimizer = _optimizer()
    scheduler = WarmupReduceOnPlateau(
        optimizer,
        warmup_epochs=2,
        warmup_start_factor=0.1,
    )
    scheduler.step(0.0)
    state = scheduler.state_dict()
    resumed = WarmupReduceOnPlateau(
        _optimizer(),
        warmup_epochs=2,
        warmup_start_factor=0.1,
    )
    resumed.load_state_dict(state)
    assert resumed.last_epoch == 1
    resumed.step(0.0)
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_invalid_warmup_parameters_are_rejected() -> None:
    """A broken warmup envelope must fail at construction."""
    with pytest.raises(ValueError, match="warmup_epochs"):
        WarmupReduceOnPlateau(_optimizer(), warmup_epochs=-1)
    with pytest.raises(ValueError, match="warmup_start_factor"):
        WarmupReduceOnPlateau(_optimizer(), warmup_epochs=1, warmup_start_factor=0.0)


def test_schedule_tracks_every_parameter_group() -> None:
    """Multiple parameter groups must ramp together."""
    parameter_a = torch.nn.Parameter(torch.zeros(1))
    parameter_b = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD(
        [
            {"params": [parameter_a], "lr": 1e-3},
            {"params": [parameter_b], "lr": 2e-3},
        ],
        lr=1e-3,
    )
    scheduler = WarmupReduceOnPlateau(
        optimizer,
        warmup_epochs=2,
        warmup_start_factor=0.1,
    )
    scheduler.step(0.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5.5e-4)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1.1e-3)
