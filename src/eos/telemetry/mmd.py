r"""Unbiased multi-scale RBF maximum mean discrepancy.

The two-sample statistic compares the empirical distributions of two embedding
sets in the reproducing kernel Hilbert space of a multi-scale RBF kernel

.. math::
    k(\mathbf{z}, \mathbf{w}) = \sum_{i=1}^{M}
        \exp\left(-\frac{\lVert \mathbf{z} - \mathbf{w}\rVert_2^2}{2\sigma_i^2}\right)

with bandwidths :math:`\sigma_i = \sigma_{\text{med}} \cdot c_i` taken from the
median heuristic over validation pairwise distances. The estimator is the
unbiased U-statistic with the diagonals removed:

.. math::
    \text{MMD}^2_u = \frac{1}{n(n-1)}\sum_{i \ne j} k(\mathbf{x}_i, \mathbf{x}_j)
        + \frac{1}{m(m-1)}\sum_{i \ne j} k(\mathbf{y}_i, \mathbf{y}_j)
        - \frac{2}{nm}\sum_{i,j} k(\mathbf{x}_i, \mathbf{y}_j)

The biased V-statistic is deliberately avoided: it is inflated when the
baseline and live batch sizes differ, which would turn the three-sigma alarm
threshold into a false-alarm generator. Calibration bootstraps disjoint blocks
of the live batch size from the baseline so the null distribution matches the
statistic that live scoring computes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from eos.config import MMDConfig


def l2_normalize(embeddings: Tensor) -> Tensor:
    """Normalize every embedding row to unit L2 norm.

    Args:
        embeddings (Tensor): Embedding matrix of shape ``(N, d)``.

    Returns:
        Tensor: Float32 matrix of shape ``(N, d)`` with unit-norm rows.

    Raises:
        ValueError: If the input is not two-dimensional.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a matrix of shape (N, d)")
    values = embeddings.to(torch.float32)
    norms = values.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    return cast(Tensor, values / norms)


def median_heuristic(embeddings: Tensor) -> float:
    """Return the median pairwise Euclidean distance of an embedding set.

    Args:
        embeddings (Tensor): Embedding matrix of shape ``(N, d)`` with ``N >= 2``.

    Returns:
        float: Median of the off-diagonal pairwise distances.

    Raises:
        ValueError: If fewer than two embeddings are supplied.
    """
    if embeddings.shape[0] < 2:
        raise ValueError("at least two embeddings are required for the median heuristic")
    values = embeddings.to(torch.float32)
    distances = torch.cdist(values, values)
    off_diagonal = ~torch.eye(distances.shape[0], dtype=torch.bool)
    return float(distances[off_diagonal].median())


def rbf_kernel(left: Tensor, right: Tensor, bandwidths: Sequence[float]) -> Tensor:
    """Evaluate the multi-scale RBF kernel matrix between two embedding sets.

    Args:
        left (Tensor): Embeddings of shape ``(n, d)`` or ``(d,)``.
        right (Tensor): Embeddings of shape ``(m, d)`` or ``(d,)``.
        bandwidths (Sequence[float]): Positive bandwidths ``sigma_i``.

    Returns:
        Tensor: Kernel matrix of shape ``(n, m)``.

    Raises:
        ValueError: If the bandwidth grid is empty, non-positive, or the
            embedding widths disagree.
    """
    if not bandwidths or any(value <= 0.0 for value in bandwidths):
        raise ValueError("bandwidths must be strictly positive")
    left_matrix = left.to(torch.float32)
    right_matrix = right.to(torch.float32)
    if left_matrix.ndim == 1:
        left_matrix = left_matrix.unsqueeze(0)
    if right_matrix.ndim == 1:
        right_matrix = right_matrix.unsqueeze(0)
    if left_matrix.shape[1] != right_matrix.shape[1]:
        raise ValueError("embedding widths must agree")
    squared = torch.cdist(left_matrix, right_matrix).square()
    total = torch.zeros_like(squared)
    for sigma in bandwidths:
        total = total + torch.exp(-squared / (2.0 * sigma * sigma))
    return total


def mmd2_unbiased(left: Tensor, right: Tensor, bandwidths: Sequence[float]) -> float:
    """Estimate squared MMD between two samples with the unbiased U-statistic.

    Args:
        left (Tensor): First sample of shape ``(n, d)`` with ``n >= 2``.
        right (Tensor): Second sample of shape ``(m, d)`` with ``m >= 2``.
        bandwidths (Sequence[float]): Positive bandwidths ``sigma_i``.

    Returns:
        float: Unbiased squared maximum mean discrepancy.

    Raises:
        ValueError: If a sample has fewer than two rows or the widths disagree.
    """
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("samples must be matrices of shape (N, d)")
    if left.shape[0] < 2 or right.shape[0] < 2:
        raise ValueError("at least two samples per side are required")
    if left.shape[1] != right.shape[1]:
        raise ValueError("embedding widths must agree")
    n_left = left.shape[0]
    n_right = right.shape[0]
    kernel_left = rbf_kernel(left, left, bandwidths)
    kernel_right = rbf_kernel(right, right, bandwidths)
    kernel_cross = rbf_kernel(left, right, bandwidths)
    left_off = (kernel_left.sum() - kernel_left.diagonal().sum()) / (n_left * (n_left - 1))
    right_off = (kernel_right.sum() - kernel_right.diagonal().sum()) / (n_right * (n_right - 1))
    cross = kernel_cross.mean()
    return float(left_off + right_off - 2.0 * cross)


@dataclass(frozen=True, slots=True)
class MMDBaseline:
    """Calibrated reference distribution for latent drift detection.

    The fingerprint binds the baseline to the exact model state it was
    calibrated on. Any ASVD pass rewrites the bottleneck representation, so a
    fingerprint mismatch means the baseline is stale and must be discarded.

    Args:
        bandwidths (tuple[float, ...]): Multi-scale RBF bandwidths.
        mu (float): Mean of the bootstrapped null distribution.
        sigma (float): Standard deviation of the bootstrapped null distribution.
        threshold (float): Alarm threshold ``mu + sigma_threshold * sigma``.
        batch_size (int): Live batch size the null was bootstrapped at.
        draws (int): Number of bootstrap draws behind ``mu`` and ``sigma``.
        seed (int): Seed of the deterministic reference subsample and bootstrap.
        fingerprint (str): Provenance digest of the model state.
        reference (Tensor): L2-normalized reference embeddings of shape
            ``(N, d)``.
        samples (Tensor): Bootstrapped null statistics of shape ``(draws,)``.
    """

    bandwidths: tuple[float, ...]
    mu: float
    sigma: float
    threshold: float
    batch_size: int
    draws: int
    seed: int
    fingerprint: str
    reference: Tensor
    samples: Tensor

    def to_dict(self) -> dict[str, object]:
        """Serialize the calibration statistics without the reference set.

        Returns:
            dict[str, object]: JSON-ready payload.
        """
        return {
            "bandwidths": list(self.bandwidths),
            "mu": self.mu,
            "sigma": self.sigma,
            "threshold": self.threshold,
            "batch_size": self.batch_size,
            "draws": self.draws,
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "reference_count": int(self.reference.shape[0]),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        reference: Tensor,
        samples: Tensor,
    ) -> MMDBaseline:
        """Rebuild a baseline from its serialized form and stored tensors.

        Args:
            payload (Mapping[str, object]): Output of ``to_dict``.
            reference (Tensor): Reference embeddings of shape ``(N, d)``.
            samples (Tensor): Bootstrapped null statistics of shape ``(draws,)``.

        Returns:
            MMDBaseline: The restored baseline.

        Raises:
            ValueError: If a field is missing or malformed, or the stored tensor
                shapes disagree with the payload.
        """
        bandwidths = _as_float_tuple(payload, "bandwidths")
        count = _as_int(payload, "reference_count")
        if count != reference.shape[0]:
            raise ValueError("reference count disagrees with the supplied embeddings")
        draws = _as_int(payload, "draws")
        if draws != samples.shape[0]:
            raise ValueError("bootstrap count disagrees with the supplied samples")
        return cls(
            bandwidths=bandwidths,
            mu=_as_float(payload, "mu"),
            sigma=_as_float(payload, "sigma"),
            threshold=_as_float(payload, "threshold"),
            batch_size=_as_int(payload, "batch_size"),
            draws=draws,
            seed=_as_int(payload, "seed"),
            fingerprint=_as_str(payload, "fingerprint"),
            reference=reference,
            samples=samples,
        )


def _as_float(payload: Mapping[str, object], key: str) -> float:
    """Read one numeric field from a serialized baseline.

    Args:
        payload (Mapping[str, object]): Serialized baseline payload.
        key (str): Field name to read.

    Returns:
        float: The field value.

    Raises:
        ValueError: If the field is missing or not a number.
    """
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"baseline field {key!r} must be a number")
    return float(value)


def _as_int(payload: Mapping[str, object], key: str) -> int:
    """Read one integer field from a serialized baseline.

    Args:
        payload (Mapping[str, object]): Serialized baseline payload.
        key (str): Field name to read.

    Returns:
        int: The field value.

    Raises:
        ValueError: If the field is missing or not an integer.
    """
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"baseline field {key!r} must be an integer")
    return value


def _as_str(payload: Mapping[str, object], key: str) -> str:
    """Read one string field from a serialized baseline.

    Args:
        payload (Mapping[str, object]): Serialized baseline payload.
        key (str): Field name to read.

    Returns:
        str: The field value.

    Raises:
        ValueError: If the field is missing or not a string.
    """
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"baseline field {key!r} must be a string")
    return value


def _as_float_tuple(payload: Mapping[str, object], key: str) -> tuple[float, ...]:
    """Read one numeric sequence field from a serialized baseline.

    Args:
        payload (Mapping[str, object]): Serialized baseline payload.
        key (str): Field name to read.

    Returns:
        tuple[float, ...]: The field value.

    Raises:
        ValueError: If the field is missing or not a sequence of numbers.
    """
    value = payload.get(key)
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError(f"baseline field {key!r} must be a sequence of numbers")
    return tuple(float(item) for item in value)


def calibrate_baseline(  # noqa: PLR0913
    embeddings: Tensor,
    config: MMDConfig,
    batch_size: int,
    *,
    draws: int = 200,
    seed: int = 42,
    fingerprint: str = "",
) -> MMDBaseline:
    """Calibrate an MMD drift baseline from validation bottleneck embeddings.

    The bandwidth grid comes from the median heuristic over the full normalized
    validation set. The reference set is a deterministic subsample of at most
    ``max_baseline_samples`` rows. The null distribution is bootstrapped at the
    live batch size: every draw splits two disjoint blocks of ``batch_size``
    rows out of the reference and computes the unbiased statistic between them,
    and the alarm threshold is ``mu + sigma_threshold * sigma``.

    Args:
        embeddings (Tensor): Validation embeddings of shape ``(N, d)``.
        config (MMDConfig): Bandwidth multipliers, threshold, and reference cap.
        batch_size (int): Live batch size the null is bootstrapped at.
        draws (int): Bootstrap draws, at least two.
        seed (int): Seed for the reference subsample and the bootstrap.
        fingerprint (str): Provenance digest of the model state.

    Returns:
        MMDBaseline: The calibrated baseline.

    Raises:
        ValueError: If the embeddings are not a matrix, too few draws are
            requested, or two disjoint live batches do not fit in the reference.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a matrix of shape (N, d)")
    if embeddings.shape[0] < 2:
        raise ValueError("at least two embeddings are required for calibration")
    if draws < 2:
        raise ValueError("at least two bootstrap draws are required")
    normalized = l2_normalize(embeddings) if config.normalize else embeddings.to(torch.float32)
    sigma_med = median_heuristic(normalized)
    bandwidths = tuple(sigma_med * multiplier for multiplier in config.bandwidth_multipliers)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(normalized.shape[0], generator=generator)
    reference = normalized[order[: config.max_baseline_samples]]
    reference_count = int(reference.shape[0])
    if 2 * batch_size > reference_count:
        raise ValueError(
            "the reference must hold two live batches of "
            f"{batch_size} samples, but keeps {reference_count}"
        )
    samples: list[float] = []
    for _ in range(draws):
        block_order = torch.randperm(reference_count, generator=generator)
        left = reference[block_order[:batch_size]]
        right = reference[block_order[batch_size : 2 * batch_size]]
        samples.append(mmd2_unbiased(left, right, bandwidths))
    statistics = torch.tensor(samples, dtype=torch.float32)
    mu = float(statistics.mean())
    sigma = float(statistics.std(correction=1))
    return MMDBaseline(
        bandwidths=bandwidths,
        mu=mu,
        sigma=sigma,
        threshold=mu + config.sigma_threshold * sigma,
        batch_size=batch_size,
        draws=draws,
        seed=seed,
        fingerprint=fingerprint,
        reference=reference,
        samples=statistics,
    )
