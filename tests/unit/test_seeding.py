"""Tests for deterministic seeding across Python, NumPy, and Torch."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from eos.utils.seeding import seed_everything


def test_seeding_repeats_every_generator_stream() -> None:
    """Reloading the same seed must replay identical random sequences."""
    seed_everything(123)
    first_python = random.random()
    first_numpy = float(np.random.rand())
    first_torch = torch.rand(4)
    seed_everything(123)
    assert random.random() == first_python
    assert float(np.random.rand()) == first_numpy
    assert torch.equal(torch.rand(4), first_torch)


def test_seeding_returns_seed_and_sets_torch_state() -> None:
    """The helper must return the applied seed and forward it to Torch."""
    assert seed_everything(7) == 7
    assert torch.initial_seed() == 7


def test_negative_seed_is_rejected() -> None:
    """A negative seed has no defined contract."""
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)


def test_deterministic_flags_are_enabled_and_restored() -> None:
    """Deterministic mode must request deterministic kernels."""
    previous_algorithm_state = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_deterministic = torch.backends.cudnn.deterministic
    previous_benchmark = torch.backends.cudnn.benchmark
    try:
        seed_everything(11, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.is_deterministic_algorithms_warn_only_enabled() is True
    finally:
        torch.use_deterministic_algorithms(
            previous_algorithm_state,
            warn_only=previous_warn_only,
        )
        torch.backends.cudnn.deterministic = previous_deterministic
        torch.backends.cudnn.benchmark = previous_benchmark


def test_non_deterministic_mode_leaves_kernel_flags_untouched() -> None:
    """Disabling the deterministic request must not toggle kernel flags."""
    previous = torch.backends.cudnn.deterministic
    seed_everything(5, deterministic=False)
    assert torch.backends.cudnn.deterministic is previous
