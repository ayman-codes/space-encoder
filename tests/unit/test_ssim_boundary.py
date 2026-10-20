"""Tests for the SSIM boundary benchmark utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eos.experiments.ssim_boundary import (
    PERTURBATIONS,
    BenchmarkSpec,
    apply_perturbation,
    bootstrap_interval,
    border_mask,
    run_benchmark,
)


def _batch(size: int = 16) -> torch.Tensor:
    """Build a deterministic structured batch.

    Args:
        size (int): Batch size.

    Returns:
        torch.Tensor: Batch of shape ``(size, 3, 64, 64)`` in ``[0, 1]``.
    """
    generator = torch.Generator().manual_seed(3)
    base = torch.rand(size, 3, 64, 64, generator=generator)
    ramp = torch.linspace(0.1, 0.6, 64).view(1, 1, 1, 64)
    return (0.4 * base + ramp).clamp(0.05, 0.95)


def test_border_mask_fraction() -> None:
    """The ring must cover exactly the outer frame of the patch."""
    mask = border_mask(64, 5)
    assert mask.shape == (1, 1, 64, 64)
    expected = (64 * 64 - 54 * 54) / (64 * 64)
    assert float(mask.float().mean()) == pytest.approx(expected, rel=1e-6)


def test_border_mask_rejects_degenerate_width() -> None:
    """A ring that consumes the patch must be rejected."""
    with pytest.raises(ValueError, match="border"):
        border_mask(64, 32)
    with pytest.raises(ValueError, match="border"):
        border_mask(64, 0)


def test_perturbations_are_deterministic() -> None:
    """Repeated calls with the same seed must agree exactly."""
    images = _batch(4)
    spec = BenchmarkSpec(seed=5)
    for name in PERTURBATIONS:
        first = apply_perturbation(images, name, spec)
        second = apply_perturbation(images, name, spec)
        assert torch.equal(first, second)


def test_local_perturbations_respect_their_region() -> None:
    """Border noise must leave the interior untouched and vice versa."""
    images = _batch(4)
    spec = BenchmarkSpec(seed=5)
    mask = border_mask(64, spec.border)
    border = apply_perturbation(images, "border_noise", spec)
    interior = apply_perturbation(images, "interior_noise", spec)
    assert torch.allclose(border[~mask.expand_as(border)], images[~mask.expand_as(images)])
    assert torch.allclose(interior[mask.expand_as(interior)], images[mask.expand_as(images)])
    assert not torch.allclose(border[mask.expand_as(border)], images[mask.expand_as(images)])


def test_unknown_perturbation_is_rejected() -> None:
    """An unknown name must raise instead of silently returning the input."""
    with pytest.raises(ValueError, match="unknown perturbation"):
        apply_perturbation(_batch(2), "invert", BenchmarkSpec())


def test_bootstrap_interval_wraps_a_constant() -> None:
    """A degenerate sample must yield a zero-width interval."""
    values = np.full(64, 0.25, dtype=np.float64)
    lower, upper = bootstrap_interval(values, BenchmarkSpec(bootstrap=100))
    assert lower == pytest.approx(0.25)
    assert upper == pytest.approx(0.25)


def test_bootstrap_rejects_invalid_input() -> None:
    """Empty samples and impossible confidence levels must fail."""
    with pytest.raises(ValueError, match="empty"):
        bootstrap_interval(np.asarray([], dtype=np.float64), BenchmarkSpec())
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_interval(np.ones(4), BenchmarkSpec(confidence=1.5))


def test_run_benchmark_orders_the_policies() -> None:
    """The artifact must record the expected sensitivity ordering."""
    images = _batch(16)
    spec = BenchmarkSpec(samples=16, seed=42, bootstrap=50)
    result = run_benchmark(images, spec, "0" * 64, "synthetic-v1.0.0")
    assert result["schema_version"] == 1
    policies = result["policies"]
    sensitivity = result["sensitivity"]
    assert isinstance(policies, dict)
    assert isinstance(sensitivity, dict)
    assert set(policies) == {"valid", "zero", "masked"}
    assert policies["valid"]["map_shape"] == [54, 54]
    assert policies["masked"]["map_shape"] == [64, 64]
    assert sensitivity["masked"]["border_drop"] > sensitivity["zero"]["border_drop"]
    assert sensitivity["zero"]["border_drop"] > sensitivity["valid"]["border_drop"]
    assert (
        policies["masked"]["gradient"]["border_fraction"]
        > policies["valid"]["gradient"]["border_fraction"]
    )


def test_run_benchmark_rejects_empty_batch() -> None:
    """An empty batch must fail instead of emitting an empty artifact."""
    with pytest.raises(ValueError, match="non-empty"):
        run_benchmark(torch.empty(0, 3, 64, 64), BenchmarkSpec(), "0" * 64, "v1")
