"""Live latent-drift scoring against a calibrated MMD baseline.

A drift monitor binds to one model state through the baseline fingerprint. Any
ASVD pass rewrites the bottleneck representation, so the monitor refuses a
baseline whose fingerprint disagrees with the deployed model state instead of
quietly scoring against a distribution that no longer describes it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from torch import Tensor

from eos.telemetry.mmd import MMDBaseline, l2_normalize, mmd2_unbiased


def model_fingerprint(
    checkpoint_sha256: str,
    asvd_state: str,
    manifest_sha256: str,
) -> str:
    """Combine provenance digests into one baseline fingerprint.

    Any ASVD pass rewrites the bottleneck representation, so the serialized
    factorization state participates in the digest. A baseline calibrated before
    the pass therefore disagrees with the deployed state and is rejected as
    stale instead of silently scoring against a distribution that no longer
    describes the encoder.

    Args:
        checkpoint_sha256 (str): Digest of the checkpoint file.
        asvd_state (str): Serialized factorization state, empty when unfactorized.
        manifest_sha256 (str): Digest of the dataset manifest.

    Returns:
        str: Lowercase hexadecimal fingerprint of the combined state.
    """
    digest = hashlib.sha256()
    for part in (checkpoint_sha256, asvd_state, manifest_sha256):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DriftScore:
    """Outcome of scoring one live batch against the baseline.

    Args:
        mmd2 (float): Unbiased squared maximum mean discrepancy of the batch.
        threshold (float): Calibrated alarm threshold of the baseline.
        alarm (bool): Whether ``mmd2`` exceeds the threshold.
        batch_size (int): Number of live embeddings scored.
    """

    mmd2: float
    threshold: float
    alarm: bool
    batch_size: int

    def to_dict(self) -> dict[str, object]:
        """Serialize the score for telemetry export.

        Returns:
            dict[str, object]: JSON-ready payload.
        """
        return {
            "mmd2": self.mmd2,
            "threshold": self.threshold,
            "alarm": self.alarm,
            "batch_size": self.batch_size,
        }


class DriftMonitor:
    """Two-sample drift test for bottleneck embeddings."""

    def __init__(self, baseline: MMDBaseline, model_fingerprint: str) -> None:
        """Bind the monitor to a calibrated baseline and a model state.

        Args:
            baseline (MMDBaseline): Calibrated reference distribution.
            model_fingerprint (str): Provenance digest of the deployed model
                state; must match the baseline fingerprint.

        Raises:
            ValueError: If the baseline was calibrated on another model state
                and is therefore stale.
        """
        if model_fingerprint != baseline.fingerprint:
            raise ValueError(
                "baseline is stale for this model state; recalibrate after any ASVD pass"
            )
        self._baseline = baseline

    @property
    def baseline(self) -> MMDBaseline:
        """Return the bound baseline.

        Returns:
            MMDBaseline: The calibrated reference distribution.
        """
        return self._baseline

    def score(self, batch: Tensor) -> DriftScore:
        """Score one live batch of bottleneck embeddings.

        Live batches are L2-normalized exactly like the calibration inputs, and
        the unbiased statistic compares the batch against the full reference.

        Args:
            batch (Tensor): Live embeddings of shape ``(n, d)``.

        Returns:
            DriftScore: Statistic, threshold, and alarm decision.

        Raises:
            ValueError: If the batch width disagrees with the reference.
        """
        normalized = l2_normalize(batch)
        mmd2 = mmd2_unbiased(
            self._baseline.reference,
            normalized,
            self._baseline.bandwidths,
        )
        return DriftScore(
            mmd2=mmd2,
            threshold=self._baseline.threshold,
            alarm=mmd2 > self._baseline.threshold,
            batch_size=int(batch.shape[0]),
        )
