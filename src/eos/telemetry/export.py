"""Prometheus export of MMD drift and ASVD compression telemetry.

Gauge names are fixed by the provisioned Grafana dashboards: the drift panel
queries ``eos_mmd_score`` and ``eos_mmd_alarm`` while the compression panel
queries ``eos_asvd_param_ratio`` and ``eos_asvd_f1_delta``. Samples are pushed
through the same Pushgateway facade as the training job metrics, but without
the training run marker so telemetry groups never collide with run groups.
"""

from __future__ import annotations

import logging

from eos.config import ServiceSettings
from eos.telemetry.mmd_drift import DriftScore
from eos.training.job_metrics import (
    DEFAULT_PROBE_TIMEOUT,
    DEFAULT_PUSH_TIMEOUT,
    PushBackend,
    PushPayload,
    probe_pushgateway,
)

MMD_SCORE = "eos_mmd_score"
"""Unbiased squared MMD of the most recent live batch."""

MMD_ALARM = "eos_mmd_alarm"
"""Alarm state of the most recent live batch as 1.0 or 0.0."""

ASVD_PARAM_RATIO = "eos_asvd_param_ratio"
"""Dense parameter ratio of the factorized model against its baseline."""

ASVD_F1_DELTA = "eos_asvd_f1_delta"
"""Macro-F1 change of the factorized model against its baseline."""

DEFAULT_JOB_NAME = "eos-telemetry"
"""Pushgateway job name for telemetry groups."""

LOGGER = logging.getLogger(__name__)


class PrometheusTelemetryBackend:
    """Default backend pushing bare gauges without training run markers."""

    def push(self, url: str, payload: PushPayload) -> None:
        """Push telemetry gauges to the gateway.

        Args:
            url (str): Pushgateway base URL, including scheme.
            payload (PushPayload): Prepared gauge samples and labels.

        Returns:
            None: The registry is pushed and discarded.
        """
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway  # noqa: PLC0415

        registry = CollectorRegistry()
        for name, value in payload.samples:
            gauge = Gauge(name, f"EOS telemetry metric {name}.", registry=registry)
            gauge.set(value)
        push_to_gateway(
            url,
            job=payload.job,
            registry=registry,
            grouping_key=dict(payload.grouping_key),
            timeout=DEFAULT_PUSH_TIMEOUT,
        )


def build_mmd_payload(
    score: DriftScore,
    *,
    fingerprint: str,
    scenario: str = "live",
    job_name: str = DEFAULT_JOB_NAME,
) -> PushPayload:
    """Build the drift telemetry payload for one scored batch.

    Args:
        score (DriftScore): Outcome of scoring one live batch.
        fingerprint (str): Provenance digest of the model state.
        scenario (str): Grouping label separating live monitoring from the
            demonstration scenarios.
        job_name (str): Pushgateway job name.

    Returns:
        PushPayload: The two dashboard gauges with model grouping labels.
    """
    samples = (
        (MMD_ALARM, 1.0 if score.alarm else 0.0),
        (MMD_SCORE, score.mmd2),
    )
    return PushPayload(
        job=job_name,
        grouping_key={"kind": "mmd", "model": fingerprint, "scenario": scenario},
        samples=tuple(sorted(samples)),
        status="ok",
    )


def build_asvd_payload(
    param_ratio: float,
    f1_delta: float,
    *,
    fingerprint: str,
    job_name: str = DEFAULT_JOB_NAME,
) -> PushPayload:
    """Build the compression telemetry payload for one factorized model.

    Args:
        param_ratio (float): Dense parameter ratio ``after / before``.
        f1_delta (float): Macro-F1 change against the unfactorized model.
        fingerprint (str): Provenance digest of the factorized model state.
        job_name (str): Pushgateway job name.

    Returns:
        PushPayload: The two dashboard gauges with model grouping labels.
    """
    samples = (
        (ASVD_F1_DELTA, f1_delta),
        (ASVD_PARAM_RATIO, param_ratio),
    )
    return PushPayload(
        job=job_name,
        grouping_key={"kind": "asvd", "model": fingerprint},
        samples=tuple(sorted(samples)),
        status="ok",
    )


class TelemetryPush:
    """Guarded telemetry reporting with an injectable push backend.

    Args:
        pushgateway_url (str): Pushgateway base URL.
        job_name (str): Pushgateway job name.
        enabled (bool): Whether reporting is requested.
        backend (PushBackend | None): Client facade, injectable for tests.
        probe_timeout (float): Health-probe timeout in seconds.
    """

    def __init__(
        self,
        pushgateway_url: str,
        job_name: str = DEFAULT_JOB_NAME,
        enabled: bool = True,
        backend: PushBackend | None = None,
        probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
    ) -> None:
        """Store the reporting configuration without contacting the gateway.

        Args:
            pushgateway_url (str): Pushgateway base URL.
            job_name (str): Pushgateway job name.
            enabled (bool): Whether reporting is requested.
            backend (PushBackend | None): Client facade.
            probe_timeout (float): Health-probe timeout in seconds.

        Returns:
            None: The reporter starts inactive until :meth:`start` succeeds.
        """
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.enabled = enabled
        self.backend = backend if backend is not None else PrometheusTelemetryBackend()
        self.probe_timeout = probe_timeout
        self._active = False
        self._failure: str | None = None

    @classmethod
    def from_settings(cls, enabled: bool = True) -> TelemetryPush:
        """Build a reporter from environment-backed service settings.

        Args:
            enabled (bool): Whether reporting is requested.

        Returns:
            TelemetryPush: Reporter bound to ``ServiceSettings``.
        """
        settings = ServiceSettings()
        return cls(pushgateway_url=settings.pushgateway_url, enabled=enabled)

    @property
    def active(self) -> bool:
        """Return whether the reporter currently writes to the gateway.

        Returns:
            bool: True only after a successful start with no later failure.
        """
        return self._active

    @property
    def failure(self) -> str | None:
        """Return the reason reporting was disabled, if any.

        Returns:
            str | None: Last failure description.
        """
        return self._failure

    def start(self) -> bool:
        """Probe the gateway and arm the reporter.

        Returns:
            bool: True when the reporter is active.
        """
        if not self.enabled:
            return False
        if not probe_pushgateway(self.pushgateway_url, self.probe_timeout):
            self._disable("health probe failed")
            return False
        self._active = True
        return True

    def push(self, payload: PushPayload) -> bool:
        """Push one telemetry payload, or no-op when the reporter is inactive.

        Args:
            payload (PushPayload): Prepared gauge samples and labels.

        Returns:
            bool: True when the payload was delivered.
        """
        if not self._active:
            return False
        try:
            self.backend.push(self.pushgateway_url, payload)
        except Exception as error:
            self._disable(str(error))
            return False
        return True

    def _disable(self, reason: str) -> None:
        """Deactivate the reporter and record why.

        Args:
            reason (str): Human-readable failure description.

        Returns:
            None: Subsequent calls become no-ops.
        """
        self._active = False
        self._failure = reason
        LOGGER.warning("telemetry reporting disabled: %s", reason)
