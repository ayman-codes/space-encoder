"""Tests for SSIM boundary handling and the multi-task objective."""

from __future__ import annotations

import pytest
import torch

from eos.model.losses import BOUNDARY_POLICIES, MultiTaskLoss, SSIMLoss


def _structured_batch(size: int = 4) -> torch.Tensor:
    """Build a deterministic structured image batch in ``[0, 1]``.

    Args:
        size (int): Batch size.

    Returns:
        torch.Tensor: Batch of shape ``(size, 3, 64, 64)``.
    """
    generator = torch.Generator().manual_seed(7)
    base = torch.rand(size, 3, 64, 64, generator=generator)
    ramp = torch.linspace(0.0, 0.4, 64).view(1, 1, 1, 64)
    return (0.3 * base + ramp).clamp(0.0, 1.0)


def _border_corruption(images: torch.Tensor, width: int = 5) -> torch.Tensor:
    """Replace the outer ring with a constant mid-grey.

    Args:
        images (torch.Tensor): Clean batch of shape ``(N, C, 64, 64)``.
        width (int): Ring width in pixels.

    Returns:
        torch.Tensor: Corrupted copy.
    """
    corrupted = images.clone()
    corrupted[:, :, :width, :] = 0.5
    corrupted[:, :, -width:, :] = 0.5
    corrupted[:, :, :, :width] = 0.5
    corrupted[:, :, :, -width:] = 0.5
    return corrupted


def test_identity_is_perfect_for_every_policy() -> None:
    """All three policies must score an exact copy as SSIM equal to one."""
    images = _structured_batch()
    for policy in BOUNDARY_POLICIES:
        loss = SSIMLoss(boundary=policy)
        assert float(loss(images, images)) == pytest.approx(0.0, abs=1e-6)


def test_map_shapes_follow_the_policy() -> None:
    """Valid crops to 54x54 while zero and masked keep 64x64."""
    images = _structured_batch()
    assert SSIMLoss(boundary="valid").ssim_map(images, images).shape == (4, 3, 54, 54)
    assert SSIMLoss(boundary="zero").ssim_map(images, images).shape == (4, 3, 64, 64)
    assert SSIMLoss(boundary="masked").ssim_map(images, images).shape == (4, 3, 64, 64)


def test_similarity_is_symmetric() -> None:
    """SSIM must be invariant to argument order."""
    images = _structured_batch()
    other = torch.clamp(images + 0.05, 0.0, 1.0)
    for policy in BOUNDARY_POLICIES:
        loss = SSIMLoss(boundary=policy)
        forward = loss.ssim_map(images, other)
        reverse = loss.ssim_map(other, images)
        assert torch.allclose(forward, reverse, atol=1e-6)


def test_border_sensitivity_ordering() -> None:
    """Masked must penalize border corruption more than zero, and zero more than valid."""
    images = _structured_batch()
    corrupted = _border_corruption(images)
    valid = float(SSIMLoss(boundary="valid")(corrupted, images))
    zero = float(SSIMLoss(boundary="zero")(corrupted, images))
    masked = float(SSIMLoss(boundary="masked")(corrupted, images))
    assert masked > zero > valid


def test_masked_survives_constant_patches() -> None:
    """Support normalization must not divide by zero at the corners."""
    constant = torch.full((2, 3, 64, 64), 0.4)
    similarity = SSIMLoss(boundary="masked").ssim_map(constant, constant)
    assert torch.isfinite(similarity).all()
    assert float(similarity.min()) == pytest.approx(1.0, abs=2e-4)


def test_autocast_guard_returns_float32() -> None:
    """SSIM must escape bfloat16 autocasting and report float32."""
    images = _structured_batch()
    loss = SSIMLoss(boundary="masked")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        similarity = loss.ssim_map(images, images)
    assert similarity.dtype == torch.float32


def test_invalid_configuration_is_rejected() -> None:
    """Even windows, non-positive sigma, and unknown policies must fail."""
    with pytest.raises(ValueError, match="odd"):
        SSIMLoss(window_size=10)
    with pytest.raises(ValueError, match="sigma"):
        SSIMLoss(sigma=0.0)
    with pytest.raises(ValueError, match="boundary"):
        SSIMLoss(boundary="reflect")  # type: ignore[arg-type]


def test_shape_mismatch_is_rejected() -> None:
    """Different shapes or ranks must fail before any convolution."""
    loss = SSIMLoss()
    with pytest.raises(ValueError, match="four-dimensional"):
        loss(torch.rand(3, 64, 64), torch.rand(3, 64, 64))
    with pytest.raises(ValueError, match="does not match"):
        loss(torch.rand(2, 3, 64, 64), torch.rand(2, 3, 32, 32))
    with pytest.raises(ValueError, match="smaller"):
        loss(torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8))


def test_multitask_weighting_is_exact() -> None:
    """Alpha must interpolate exactly between MSE and SSIM."""
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(4, 10, generator=generator)
    labels = torch.randint(0, 10, (4,), generator=generator)
    reconstruction = torch.rand(4, 3, 64, 64, generator=generator)
    target = torch.rand(4, 3, 64, 64, generator=generator)
    mse_weighted = MultiTaskLoss(alpha=1.0, recon_lambda=0.3, ssim_loss=SSIMLoss())
    breakdown = mse_weighted(logits, labels, reconstruction, target)
    expected_mse = torch.nn.functional.mse_loss(reconstruction, target)
    assert float(breakdown.reconstruction) == pytest.approx(float(0.3 * expected_mse), rel=1e-5)
    ssim_weighted = MultiTaskLoss(alpha=0.0, recon_lambda=0.3, ssim_loss=SSIMLoss())
    breakdown = ssim_weighted(logits, labels, reconstruction, target)
    expected_ssim = SSIMLoss()(reconstruction, target)
    assert float(breakdown.reconstruction) == pytest.approx(float(0.3 * expected_ssim), rel=1e-5)


def test_total_is_cross_entropy_plus_weighted_reconstruction() -> None:
    """The total must decompose without hidden terms."""
    generator = torch.Generator().manual_seed(13)
    logits = torch.randn(4, 10, generator=generator)
    labels = torch.randint(0, 10, (4,), generator=generator)
    reconstruction = torch.rand(4, 3, 64, 64, generator=generator)
    target = torch.rand(4, 3, 64, 64, generator=generator)
    criterion = MultiTaskLoss(alpha=0.85, recon_lambda=0.4, ssim_loss=SSIMLoss())
    breakdown = criterion(logits, labels, reconstruction, target)
    expected = breakdown.classification + breakdown.reconstruction
    assert float(breakdown.total) == pytest.approx(float(expected), rel=1e-6)


def test_class_weights_are_applied() -> None:
    """Weighted cross-entropy must match the manual computation."""
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(4, 10, generator=generator)
    labels = torch.tensor([0, 1, 2, 3])
    images = torch.rand(4, 3, 64, 64, generator=generator)
    weights = torch.linspace(0.5, 1.5, 10)
    criterion = MultiTaskLoss(
        alpha=1.0,
        recon_lambda=0.3,
        ssim_loss=SSIMLoss(),
        class_weights=weights,
    )
    breakdown = criterion(logits, labels, images, images)
    expected = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
    assert float(breakdown.classification) == pytest.approx(float(expected), rel=1e-6)


def test_gradients_are_finite() -> None:
    """Both branches must produce finite gradients."""
    generator = torch.Generator().manual_seed(19)
    logits = torch.randn(2, 10, generator=generator, requires_grad=True)
    reconstruction = torch.rand(2, 3, 64, 64, generator=generator, requires_grad=True)
    target = torch.rand(2, 3, 64, 64, generator=generator)
    criterion = MultiTaskLoss(alpha=0.85, recon_lambda=0.3, ssim_loss=SSIMLoss())
    breakdown = criterion(logits, torch.tensor([0, 1]), reconstruction, target)
    breakdown.total.backward()
    assert logits.grad is not None
    assert reconstruction.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(reconstruction.grad).all()


def test_multitask_rejects_invalid_weights() -> None:
    """Alpha and lambda ranges are part of the contract."""
    with pytest.raises(ValueError, match="alpha"):
        MultiTaskLoss(alpha=1.5, recon_lambda=0.3, ssim_loss=SSIMLoss())
    with pytest.raises(ValueError, match="recon_lambda"):
        MultiTaskLoss(alpha=0.5, recon_lambda=0.0, ssim_loss=SSIMLoss())


def test_scheduled_lambda_override_scales_reconstruction() -> None:
    """A scheduled weight must scale only the reconstruction term."""
    generator = torch.Generator().manual_seed(23)
    logits = torch.randn(4, 10, generator=generator)
    labels = torch.randint(0, 10, (4,), generator=generator)
    reconstruction = torch.rand(4, 3, 64, 64, generator=generator)
    target = torch.rand(4, 3, 64, 64, generator=generator)
    criterion = MultiTaskLoss(alpha=0.85, recon_lambda=0.3, ssim_loss=SSIMLoss())
    baseline = criterion(logits, labels, reconstruction, target, recon_lambda=0.3)
    scheduled = criterion(logits, labels, reconstruction, target, recon_lambda=0.1)
    assert float(scheduled.classification) == pytest.approx(
        float(baseline.classification),
        rel=1e-6,
    )
    assert float(scheduled.reconstruction) == pytest.approx(
        float(baseline.reconstruction) / 3.0,
        rel=1e-5,
    )


def test_terms_decompose_independently() -> None:
    """The public term methods must reproduce the composite exactly."""
    generator = torch.Generator().manual_seed(29)
    logits = torch.randn(4, 10, generator=generator)
    labels = torch.randint(0, 10, (4,), generator=generator)
    reconstruction = torch.rand(4, 3, 64, 64, generator=generator)
    target = torch.rand(4, 3, 64, 64, generator=generator)
    criterion = MultiTaskLoss(alpha=0.85, recon_lambda=0.3, ssim_loss=SSIMLoss())
    cross_entropy = criterion.classification_term(logits, labels)
    mse, ssim, weighted = criterion.reconstruction_term(reconstruction, target, 0.2)
    expected = 0.2 * (0.85 * mse + 0.15 * ssim)
    assert float(weighted) == pytest.approx(float(expected), rel=1e-6)
    assert float(cross_entropy) == pytest.approx(
        float(criterion(logits, labels, reconstruction, target).classification),
        rel=1e-6,
    )
    with pytest.raises(ValueError, match="strictly positive"):
        criterion.reconstruction_term(reconstruction, target, 0.0)


def test_multitask_rejects_inconsistent_batches() -> None:
    """Batch and image mismatches must fail clearly."""
    criterion = MultiTaskLoss(alpha=0.85, recon_lambda=0.3, ssim_loss=SSIMLoss())
    with pytest.raises(ValueError, match="batch dimension"):
        criterion(
            torch.randn(2, 10),
            torch.tensor([0]),
            torch.rand(2, 3, 64, 64),
            torch.rand(2, 3, 64, 64),
        )
    with pytest.raises(ValueError, match="does not match"):
        criterion(
            torch.randn(2, 10),
            torch.tensor([0, 1]),
            torch.rand(2, 3, 64, 64),
            torch.rand(2, 3, 32, 32),
        )
    with pytest.raises(ValueError, match="logits"):
        criterion(
            torch.randn(2, 10, 1),
            torch.tensor([0, 1]),
            torch.rand(2, 3, 64, 64),
            torch.rand(2, 3, 64, 64),
        )
