"""SSIM boundary-policy benchmark.

An 11x11 window cannot be evaluated at the patch border without deciding what
fills the missing pixels. This experiment measures how that decision changes the
loss: paired per-patch SSIM under controlled perturbations, the resulting
sensitivity ratios, and the fraction of gradient mass that reaches the border
ring.

The three policies are ``valid`` (crop to 54x54), ``zero`` (same-padding with
zeros), and ``masked`` (support-normalized, full 64x64 coverage).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from eos.model.losses import BOUNDARY_POLICIES, BoundaryPolicy, SSIMLoss

PERTURBATIONS: tuple[str, ...] = (
    "identity",
    "border_noise",
    "interior_noise",
    "border_constant",
    "blur",
    "shift",
    "quantize",
)

PERTURBATION_DESCRIPTIONS: dict[str, str] = {
    "identity": "unmodified prediction, sanity check",
    "border_noise": "Gaussian noise inside the five-pixel border ring",
    "interior_noise": "Gaussian noise inside the interior region",
    "border_constant": "border ring replaced by the mid-grey constant 0.5",
    "blur": "3x3 binomial blur over the whole patch",
    "shift": "three-pixel roll along both spatial axes",
    "quantize": "four-level posterization",
}

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BenchmarkSpec:
    """Deterministic settings for one benchmark run.

    Args:
        samples (int): Number of test-split patches evaluated.
        seed (int): Seed for patch sampling, noise, and bootstrap resampling.
        border (int): Border ring width in pixels.
        noise_std (float): Standard deviation of the additive noise perturbation.
        bootstrap (int): Number of paired bootstrap resamples.
        confidence (float): Confidence level of the reported interval.
    """

    samples: int = 256
    seed: int = 42
    border: int = 5
    noise_std: float = 0.10
    bootstrap: int = 2000
    confidence: float = 0.95


def border_mask(size: int, border: int, device: torch.device | None = None) -> Tensor:
    """Build a boolean mask covering the outer ring of a square patch.

    Args:
        size (int): Patch edge length in pixels.
        border (int): Ring width in pixels; must be smaller than ``size // 2``.
        device (torch.device | None): Optional allocation device.

    Returns:
        Tensor: Boolean mask of shape ``(1, 1, size, size)`` that is ``True`` on
            the ring and ``False`` in the interior.

    Raises:
        ValueError: If the ring width leaves no interior.
    """
    if border <= 0 or 2 * border >= size:
        raise ValueError("border must be positive and smaller than half the patch")
    mask = torch.ones((1, 1, size, size), dtype=torch.bool, device=device)
    mask[:, :, border : size - border, border : size - border] = False
    return mask


def apply_perturbation(
    images: Tensor,
    name: str,
    spec: BenchmarkSpec,
) -> Tensor:
    """Apply one deterministic perturbation to a batch of patches.

    Args:
        images (Tensor): Float32 batch of shape ``(N, C, H, W)`` in ``[0, 1]``.
        name (str): One of :data:`PERTURBATIONS`.
        spec (BenchmarkSpec): Run settings supplying border width, noise scale,
            and seed.

    Returns:
        Tensor: Perturbed batch with the same shape as ``images``.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name == "identity":
        return images.clone()
    if name == "blur":
        kernel = torch.tensor([1.0, 2.0, 1.0], dtype=images.dtype, device=images.device)
        kernel = torch.outer(kernel, kernel) / kernel.sum()
        channels = images.shape[1]
        weight = kernel.expand(channels, 1, 3, 3).contiguous()
        padded = torch.nn.functional.pad(images, (1, 1, 1, 1), mode="replicate")
        return torch.nn.functional.conv2d(padded, weight, groups=channels).clamp(0.0, 1.0)
    if name == "shift":
        return torch.roll(images, shifts=(3, 3), dims=(2, 3))
    if name == "quantize":
        return torch.round(images * 3.0) / 3.0
    mask = border_mask(images.shape[-1], spec.border, images.device).to(images.dtype)
    if name == "border_constant":
        return (images * (1.0 - mask) + 0.5 * mask).clamp(0.0, 1.0)
    if name in {"border_noise", "interior_noise"}:
        generator = torch.Generator(device="cpu").manual_seed(spec.seed)
        noise = torch.randn(images.shape, generator=generator, dtype=torch.float32)
        noise = noise.to(images.device, images.dtype) * spec.noise_std
        region = mask if name == "border_noise" else 1.0 - mask
        return (images + noise * region).clamp(0.0, 1.0)
    raise ValueError(f"unknown perturbation {name!r}; expected one of {PERTURBATIONS}")


def per_patch_ssim(loss: SSIMLoss, prediction: Tensor, target: Tensor) -> Tensor:
    """Return one mean SSIM value per batch element.

    Args:
        loss (SSIMLoss): Configured structural loss carrying the boundary policy.
        prediction (Tensor): Predicted batch of shape ``(N, C, H, W)``.
        target (Tensor): Reference batch of shape ``(N, C, H, W)``.

    Returns:
        Tensor: Float32 tensor of shape ``(N,)``.
    """
    with torch.no_grad():
        similarity = loss.ssim_map(prediction, target)
        return similarity.mean(dim=(1, 2, 3))


def bootstrap_interval(
    values: npt.NDArray[np.float64],
    spec: BenchmarkSpec,
) -> tuple[float, float]:
    """Estimate a percentile confidence interval for the mean.

    Args:
        values (npt.NDArray[np.float64]): Per-patch observations.
        spec (BenchmarkSpec): Run settings supplying resample count, seed, and
            confidence level.

    Returns:
        tuple[float, float]: Lower and upper confidence bounds.

    Raises:
        ValueError: If the sample is empty or the confidence is outside ``(0, 1)``.
    """
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0.0 < spec.confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    generator = np.random.default_rng(spec.seed)
    indices = generator.integers(0, values.size, size=(spec.bootstrap, values.size))
    means = values[indices].mean(axis=1)
    tail = (1.0 - spec.confidence) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return float(lower), float(upper)


def gradient_border_fraction(
    loss: SSIMLoss,
    prediction: Tensor,
    target: Tensor,
    spec: BenchmarkSpec,
) -> dict[str, float]:
    """Measure where the SSIM gradient lands across the patch.

    Args:
        loss (SSIMLoss): Configured structural loss.
        prediction (Tensor): Predicted batch of shape ``(N, C, H, W)``.
        target (Tensor): Reference batch of shape ``(N, C, H, W)``.
        spec (BenchmarkSpec): Run settings supplying the border width.

    Returns:
        dict[str, float]: Absolute gradient mass on the border ring, in the
            interior, and their totals.
    """
    candidate = prediction.detach().clone().requires_grad_(True)
    objective = loss(candidate, target)
    (gradient,) = torch.autograd.grad(objective, candidate)
    magnitude = gradient.abs()
    mask = border_mask(prediction.shape[-1], spec.border, magnitude.device).expand_as(magnitude)
    border = float(magnitude[mask].sum())
    total = float(magnitude.sum())
    return {
        "border_mass": border,
        "interior_mass": total - border,
        "total_mass": total,
        "border_fraction": border / total if total > 0.0 else 0.0,
    }


def _evaluate_policy(
    policy: BoundaryPolicy,
    images: Tensor,
    spec: BenchmarkSpec,
) -> dict[str, object]:
    """Run every perturbation for one boundary policy.

    Args:
        policy (BoundaryPolicy): Boundary handling policy.
        images (Tensor): Clean batch of shape ``(N, C, H, W)``.
        spec (BenchmarkSpec): Run settings.

    Returns:
        dict[str, object]: Per-perturbation statistics and gradient shares.
    """
    loss = SSIMLoss(window_size=11, sigma=1.5, boundary=policy)
    identity = per_patch_ssim(loss, images, images)
    baseline = float(identity.mean())
    perturbations: dict[str, object] = {}
    for name in PERTURBATIONS:
        candidate = apply_perturbation(images, name, spec)
        scores = per_patch_ssim(loss, candidate, images)
        drops = identity - scores
        mean_drop = float(drops.mean())
        lower, upper = bootstrap_interval(drops.double().numpy(), spec)
        perturbations[name] = {
            "description": PERTURBATION_DESCRIPTIONS[name],
            "mean_ssim": float(scores.mean()),
            "std_ssim": float(scores.std(unbiased=True)),
            "mean_drop": mean_drop,
            "drop_ci_low": lower,
            "drop_ci_high": upper,
        }
    border_gradients = gradient_border_fraction(
        loss,
        apply_perturbation(images, "border_noise", spec),
        images,
        spec,
    )
    return {
        "boundary": policy,
        "map_shape": list(loss.ssim_map(images, images).shape[2:]),
        "baseline_ssim": baseline,
        "perturbations": perturbations,
        "gradient": border_gradients,
    }


def _area_fractions(size: int, border: int) -> tuple[float, float]:
    """Return the border and interior area fractions.

    Args:
        size (int): Patch edge length.
        border (int): Ring width.

    Returns:
        tuple[float, float]: Border fraction and interior fraction.
    """
    total = float(size * size)
    interior = float((size - 2 * border) ** 2)
    return (total - interior) / total, interior / total


def _sensitivity(result: dict[str, object], area: tuple[float, float]) -> dict[str, float]:
    """Compute raw and area-normalized border sensitivity ratios.

    Args:
        result (dict[str, object]): One entry of :func:`_evaluate_policy`.
        area (tuple[float, float]): Border and interior area fractions.

    Returns:
        dict[str, float]: Border drop, interior drop, and both ratios.
    """
    perturbations = cast(dict[str, dict[str, float]], result["perturbations"])
    border_drop = perturbations["border_noise"]["mean_drop"]
    interior_drop = perturbations["interior_noise"]["mean_drop"]
    border_area, interior_area = area
    normalized_border = border_drop / border_area
    normalized_interior = interior_drop / interior_area
    comparable = interior_drop > 0.0
    return {
        "border_drop": border_drop,
        "interior_drop": interior_drop,
        "raw_ratio": border_drop / interior_drop if comparable else 0.0,
        "area_ratio": normalized_border / normalized_interior if comparable else 0.0,
        "comparable": comparable,
    }


def run_benchmark(
    images: Tensor,
    spec: BenchmarkSpec,
    manifest_sha256: str,
    dataset_version: str,
) -> dict[str, object]:
    """Evaluate all boundary policies over a batch of patches.

    Args:
        images (Tensor): Clean batch of shape ``(N, C, H, W)`` in ``[0, 1]``.
        spec (BenchmarkSpec): Run settings.
        manifest_sha256 (str): Dataset fingerprint recorded in the artifact.
        dataset_version (str): Curated dataset version.

    Returns:
        dict[str, object]: JSON-serializable benchmark result.

    Raises:
        ValueError: If the batch is empty or not four-dimensional.
    """
    if images.ndim != 4 or images.shape[0] == 0:
        raise ValueError("images must be a non-empty (N, C, H, W) tensor")
    area = _area_fractions(images.shape[-1], spec.border)
    policies: dict[str, object] = {}
    sensitivity: dict[str, object] = {}
    for policy in BOUNDARY_POLICIES:
        result = _evaluate_policy(policy, images, spec)
        policies[policy] = result
        sensitivity[policy] = _sensitivity(result, area)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "version": dataset_version,
            "manifest_sha256": manifest_sha256,
            "samples": int(images.shape[0]),
        },
        "config": {
            "seed": spec.seed,
            "border": spec.border,
            "noise_std": spec.noise_std,
            "window_size": 11,
            "sigma": 1.5,
            "bootstrap": spec.bootstrap,
            "confidence": spec.confidence,
            "area_fractions": {"border": area[0], "interior": area[1]},
        },
        "policies": policies,
        "sensitivity": sensitivity,
    }
