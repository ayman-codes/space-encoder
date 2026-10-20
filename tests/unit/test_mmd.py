"""Tests for the unbiased multi-scale RBF maximum mean discrepancy estimator."""

from __future__ import annotations

import json

import pytest
import torch
from torch import Tensor

from eos.config import MMDConfig
from eos.telemetry.mmd import (
    MMDBaseline,
    calibrate_baseline,
    l2_normalize,
    median_heuristic,
    mmd2_unbiased,
    rbf_kernel,
)
from eos.telemetry.mmd_drift import DriftMonitor, model_fingerprint

BANDWIDTHS: tuple[float, ...] = (1.4, 2.8, 5.6, 11.2, 22.4)
"""Fixed bandwidth grid used where calibration itself is not under test."""


def _draws(count: int, dimensions: int, seed: int, shift: float = 0.0) -> Tensor:
    """Draw a deterministic Gaussian sample, optionally shifted.

    Args:
        count (int): Number of embeddings.
        dimensions (int): Embedding width.
        seed (int): Generator seed.
        shift (float): Constant added to every coordinate.

    Returns:
        Tensor: Sample of shape ``(count, dimensions)``.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, dimensions, generator=generator) + shift


def test_l2_normalize_rows() -> None:
    """Every row must carry unit L2 norm with preserved direction."""
    values = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    normalized = l2_normalize(values)
    assert normalized[0].tolist() == pytest.approx([0.6, 0.8])
    assert float(normalized[1].norm()) == pytest.approx(1.0)


def test_median_heuristic_matches_known_distances() -> None:
    """The bandwidth base must be the median pairwise Euclidean distance."""
    values = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    assert median_heuristic(values) == pytest.approx(3.0)


def test_median_heuristic_rejects_single_sample() -> None:
    """One embedding has no pairwise distance to take a median over."""
    with pytest.raises(ValueError, match="at least two embeddings"):
        median_heuristic(torch.zeros(1, 4))


def test_rbf_kernel_is_multi_scale() -> None:
    """The kernel must sum one Gaussian term per bandwidth."""
    left = torch.zeros(1, 2)
    right = torch.tensor([[0.0, 3.0]])
    kernel = rbf_kernel(left, right, (1.0, 2.0))
    expected = float(torch.exp(torch.tensor(-9.0 / 2.0)) + torch.exp(torch.tensor(-9.0 / 8.0)))
    assert float(kernel[0, 0]) == pytest.approx(expected, rel=1.0e-5)


def test_mmd2_matches_brute_force_reference() -> None:
    """The estimator must equal the U-statistic with the diagonals removed."""
    left = _draws(6, 3, seed=1)
    right = _draws(5, 3, seed=2, shift=0.3)
    n_left, n_right = left.shape[0], right.shape[0]
    total_left = 0.0
    for i in range(n_left):
        for j in range(n_left):
            if i != j:
                total_left += float(rbf_kernel(left[i], left[j], BANDWIDTHS))
    total_right = 0.0
    for i in range(n_right):
        for j in range(n_right):
            if i != j:
                total_right += float(rbf_kernel(right[i], right[j], BANDWIDTHS))
    total_cross = 0.0
    for i in range(n_left):
        for j in range(n_right):
            total_cross += float(rbf_kernel(left[i], right[j], BANDWIDTHS))
    reference = (
        total_left / (n_left * (n_left - 1))
        + total_right / (n_right * (n_right - 1))
        - 2.0 * total_cross / (n_left * n_right)
    )
    assert mmd2_unbiased(left, right, BANDWIDTHS) == pytest.approx(reference, rel=1.0e-5)


def test_mmd2_is_symmetric() -> None:
    """Swapping the samples must not change the statistic."""
    left = _draws(16, 4, seed=3)
    right = _draws(16, 4, seed=4, shift=0.2)
    assert mmd2_unbiased(left, right, BANDWIDTHS) == pytest.approx(
        mmd2_unbiased(right, left, BANDWIDTHS), rel=1.0e-6
    )


def test_mmd2_is_near_zero_for_one_distribution() -> None:
    """Two independent draws from one distribution must score near zero."""
    same = mmd2_unbiased(_draws(256, 16, seed=5), _draws(256, 16, seed=6), BANDWIDTHS)
    assert abs(same) < 0.1


def test_mmd2_detects_mean_shift() -> None:
    """A distribution shift must inflate the statistic far beyond the null."""
    same = mmd2_unbiased(_draws(256, 16, seed=5), _draws(256, 16, seed=6), BANDWIDTHS)
    shifted = mmd2_unbiased(_draws(256, 16, seed=5), _draws(256, 16, seed=7, shift=1.5), BANDWIDTHS)
    assert shifted > 0.3
    assert shifted > 5.0 * abs(same)


def test_mmd2_rejects_mismatched_widths() -> None:
    """Samples must live in the same embedding space."""
    with pytest.raises(ValueError, match="embedding widths must agree"):
        mmd2_unbiased(torch.zeros(4, 3), torch.zeros(4, 5), BANDWIDTHS)


def test_mmd2_rejects_tiny_samples() -> None:
    """The U-statistic is undefined below two samples per side."""
    with pytest.raises(ValueError, match="at least two samples per side"):
        mmd2_unbiased(torch.zeros(1, 3), torch.zeros(4, 3), BANDWIDTHS)


def test_calibration_records_threshold_relation() -> None:
    """The alarm threshold must be mu plus the configured sigma multiples."""
    config = MMDConfig()
    baseline = calibrate_baseline(
        _draws(200, 8, seed=8),
        config,
        batch_size=16,
        draws=200,
        seed=11,
        fingerprint="probe",
    )
    assert baseline.threshold == pytest.approx(
        baseline.mu + config.sigma_threshold * baseline.sigma
    )
    sigma_med = median_heuristic(l2_normalize(_draws(200, 8, seed=8)))
    expected = tuple(sigma_med * value for value in config.bandwidth_multipliers)
    assert baseline.bandwidths == pytest.approx(expected)
    assert baseline.fingerprint == "probe"


def test_calibration_is_deterministic_under_seed() -> None:
    """The same seed must reproduce the same calibration exactly."""
    embeddings = _draws(120, 6, seed=9)
    first = calibrate_baseline(embeddings, MMDConfig(), 16, draws=100, seed=21)
    second = calibrate_baseline(embeddings, MMDConfig(), 16, draws=100, seed=21)
    third = calibrate_baseline(embeddings, MMDConfig(), 16, draws=100, seed=22)
    assert (first.mu, first.sigma, first.threshold) == (second.mu, second.sigma, second.threshold)
    assert (first.mu, first.sigma) != (third.mu, third.sigma)


def test_bootstrap_uses_the_live_batch_size() -> None:
    """Smaller live batches must widen the null distribution."""
    embeddings = _draws(256, 8, seed=10)
    narrow = calibrate_baseline(embeddings, MMDConfig(), 8, draws=300, seed=12)
    wide = calibrate_baseline(embeddings, MMDConfig(), 64, draws=300, seed=12)
    assert narrow.sigma > wide.sigma
    assert narrow.batch_size == 8
    assert wide.batch_size == 64


def test_reference_subsample_is_recorded() -> None:
    """The retained reference must respect the configured cap."""
    config = MMDConfig(max_baseline_samples=50)
    baseline = calibrate_baseline(_draws(200, 6, seed=13), config, 8, draws=50, seed=14)
    assert baseline.reference.shape == (50, 6)
    assert baseline.to_dict()["reference_count"] == 50


def test_calibration_rejects_unfittable_batches() -> None:
    """Two disjoint live batches must fit inside the reference."""
    with pytest.raises(ValueError, match="two live batches"):
        calibrate_baseline(_draws(20, 4, seed=15), MMDConfig(), 16, draws=10, seed=16)


def test_calibration_rejects_tiny_draw_count() -> None:
    """A standard deviation needs at least two bootstrap draws."""
    with pytest.raises(ValueError, match="at least two bootstrap draws"):
        calibrate_baseline(_draws(64, 4, seed=17), MMDConfig(), 8, draws=1, seed=18)


def test_baseline_round_trip_preserves_the_contract() -> None:
    """Serialization must restore every field and the reference set."""
    baseline = calibrate_baseline(_draws(80, 5, seed=19), MMDConfig(), 8, draws=50, seed=20)
    payload = json.loads(json.dumps(baseline.to_dict()))
    restored = MMDBaseline.from_dict(payload, baseline.reference, baseline.samples)
    assert restored.bandwidths == baseline.bandwidths
    assert restored.mu == baseline.mu
    assert restored.sigma == baseline.sigma
    assert restored.threshold == baseline.threshold
    assert restored.batch_size == baseline.batch_size
    assert restored.fingerprint == baseline.fingerprint
    assert torch.equal(restored.reference, baseline.reference)
    assert torch.equal(restored.samples, baseline.samples)


def test_monitor_rejects_stale_baseline() -> None:
    """A baseline calibrated on another model state must be refused."""
    baseline = calibrate_baseline(_draws(64, 5, seed=21), MMDConfig(), 8, draws=20, seed=22)
    with pytest.raises(ValueError, match="stale"):
        DriftMonitor(baseline, model_fingerprint="asvd-v2-state")


def test_monitor_scores_and_alarms() -> None:
    """In-distribution batches stay quiet while shifted batches alarm."""
    embeddings = _draws(240, 8, seed=23)
    baseline = calibrate_baseline(embeddings, MMDConfig(), 16, draws=200, seed=24)
    monitor = DriftMonitor(baseline, model_fingerprint="")
    quiet = monitor.score(_draws(16, 8, seed=25))
    alarmed = monitor.score(_draws(16, 8, seed=26, shift=1.5))
    assert not quiet.alarm
    assert quiet.batch_size == 16
    assert alarmed.alarm
    assert alarmed.mmd2 > quiet.mmd2
    assert alarmed.threshold == baseline.threshold


def test_monitor_normalizes_live_batches() -> None:
    """Live batches must be L2-normalized before kernel evaluation."""
    embeddings = l2_normalize(_draws(240, 8, seed=27))
    baseline = calibrate_baseline(embeddings, MMDConfig(), 16, draws=200, seed=28)
    monitor = DriftMonitor(baseline, model_fingerprint="")
    batch = _draws(16, 8, seed=29) * 12.0
    scaled = monitor.score(batch)
    unscaled = monitor.score(batch / 12.0)
    assert scaled.mmd2 == pytest.approx(unscaled.mmd2, rel=1.0e-3, abs=1.0e-5)


def test_score_rejects_wrong_width() -> None:
    """Live batches must match the calibrated embedding width."""
    baseline = calibrate_baseline(_draws(64, 5, seed=30), MMDConfig(), 8, draws=20, seed=31)
    monitor = DriftMonitor(baseline, model_fingerprint="")
    with pytest.raises(ValueError, match="embedding widths must agree"):
        monitor.score(torch.zeros(8, 7))


def test_model_fingerprint_binds_the_factorization_state() -> None:
    """The same checkpoint under another ASVD state must fingerprint apart."""
    base = model_fingerprint("ckpt-digest", "", "manifest-digest")
    assert base == model_fingerprint("ckpt-digest", "", "manifest-digest")
    assert base != model_fingerprint("ckpt-digest", "scope=bottleneck,rank=48", "manifest-digest")
    assert base != model_fingerprint("other-digest", "", "manifest-digest")
    assert base != model_fingerprint("ckpt-digest", "", "other-manifest")
