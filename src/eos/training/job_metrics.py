"""Pushgateway job reporting that never blocks training.

Training runs publish their job details to the Prometheus Pushgateway under the
``eos-train`` job: wall time, GPU fit time, epoch and parameter counts, the run
status, and every final scalar metric. Prometheus scrapes the gateway into its
time-series store, and the Grafana training dashboard reads from Prometheus.

The reporter is a thin, injectable facade over ``prometheus_client``. Mode,
seed, and run name travel as grouping-key labels, so every sample in Grafana is
tagged with the run it came from. Metric keys such as ``val/macro_f1`` are
sanitized into Prometheus gauge names such as ``eos_train_val_macro_f1``
because Prometheus identifiers cannot contain slashes.

A gateway that is unreachable or misconfigured disables the reporter after one
failed attempt. Every subsequent call becomes a no-op, so observability can
never abort a training run. Runs that completed before this reporter existed
carry only wall time in their metrics JSON; GPU fit time is recorded going
forward and is deliberately not backfilled.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from urllib.request import urlopen

from eos.config import ServiceSettings

LOGGER = logging.getLogger(__name__)

DEFAULT_JOB_NAME = "eos-train"
"""Pushgateway job name shared by every training run."""

DEFAULT_PROBE_TIMEOUT = 2.0
"""Seconds allowed for the Pushgateway health probe."""

DEFAULT_PUSH_TIMEOUT = 5.0
"""Seconds allowed for one metric push."""

GAUGE_PREFIX = "eos_train_"
"""Prefix applied to every gauge name pushed by this module."""

INVALID_GAUGE_CHARS = re.compile(r"[^0-9a-zA-Z_]")


@dataclass(frozen=True)
class JobRecord:
    """Details of one training run as reported to Prometheus.

    Attributes:
        run_name (str): Unique run name, used as a grouping-key label.
        mode (str): Ablation arm, used as a grouping-key label.
        seed (int): Data and trainer seed, used as a grouping-key label.
        status (str): ``"finished"`` for a completed run or ``"failed"`` for a
            run that raised before completion.
        duration_seconds (float): Wall time of the whole run in seconds.
        fit_seconds (float | None): Wall time of the training loop in seconds,
            which is the window the GPU is occupied. None when the fit loop
            never ran or the value was not measured.
        epochs_total (int | None): Completed training epochs.
        parameter_count (int | None): Trainable parameter count of the model.
        metrics (Mapping[str, float]): Final scalar metrics keyed as in the
            metrics JSON, for example ``val/macro_f1``.
    """

    run_name: str
    mode: str
    seed: int
    status: str
    duration_seconds: float
    fit_seconds: float | None
    epochs_total: int | None
    parameter_count: int | None
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class PushPayload:
    """Prepared gauge sample set for one Pushgateway push.

    Attributes:
        job (str): Pushgateway job name.
        grouping_key (Mapping[str, str]): Labels identifying the run.
        samples (tuple[tuple[str, float], ...]): Sorted gauge name and value
            pairs.
        status (str): Run status exposed on the run-info gauge label.
    """

    job: str
    grouping_key: Mapping[str, str]
    samples: tuple[tuple[str, float], ...]
    status: str


class PushBackend(Protocol):
    """Minimal Pushgateway surface consumed by the reporter."""

    def push(self, url: str, payload: PushPayload) -> None:
        """Push one prepared payload to the gateway.

        Args:
            url (str): Pushgateway base URL.
            payload (PushPayload): Prepared gauge samples and labels.

        Returns:
            None: The payload is delivered to the gateway.
        """


class PrometheusClientBackend:
    """Default backend that delegates to the installed prometheus_client."""

    def push(self, url: str, payload: PushPayload) -> None:
        """Build a transient registry and push it to the gateway.

        Args:
            url (str): Pushgateway base URL, including scheme.
            payload (PushPayload): Prepared gauge samples and labels.

        Returns:
            None: The registry is pushed and discarded.
        """
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway  # noqa: PLC0415

        registry = CollectorRegistry()
        for name, value in payload.samples:
            gauge = Gauge(name, f"Training run metric {name}.", registry=registry)
            gauge.set(value)
        info = Gauge(
            f"{GAUGE_PREFIX}run_info",
            "Run identity and status marker.",
            registry=registry,
            labelnames=("status",),
        )
        info.labels(status=payload.status).set(1.0)
        push_to_gateway(
            url,
            job=payload.job,
            registry=registry,
            grouping_key=dict(payload.grouping_key),
            timeout=DEFAULT_PUSH_TIMEOUT,
        )


def sanitize_metric_name(key: str) -> str:
    """Convert a metric key into a valid Prometheus gauge name.

    Slashes and every other character outside ``[0-9A-Za-z_]`` are replaced
    with underscores, then the gauge prefix is applied.

    Args:
        key (str): Metric key such as ``val/macro_f1``.

    Returns:
        str: Valid gauge name such as ``eos_train_val_macro_f1``.

    Raises:
        ValueError: If the sanitized key is empty.
    """
    cleaned = INVALID_GAUGE_CHARS.sub("_", key).strip("_")
    if not cleaned:
        raise ValueError(f"metric key {key!r} sanitizes to an empty name")
    return f"{GAUGE_PREFIX}{cleaned}"


def build_payload(
    record: JobRecord,
    job_name: str = DEFAULT_JOB_NAME,
) -> PushPayload:
    """Turn a job record into gauge samples and grouping-key labels.

    The job fields become dedicated gauges; optional fields are omitted when
    absent. Every key in ``record.metrics`` becomes one gauge after
    sanitization, so keys that sanitize to the same name collapse into one
    sample.

    Args:
        record (JobRecord): Details of one training run.
        job_name (str): Pushgateway job name.

    Returns:
        PushPayload: Sorted samples, grouping-key labels, and run status.
    """
    samples: dict[str, float] = {
        f"{GAUGE_PREFIX}duration_seconds": float(record.duration_seconds),
        f"{GAUGE_PREFIX}push_timestamp_seconds": time.time(),
    }
    if record.fit_seconds is not None:
        samples[f"{GAUGE_PREFIX}fit_seconds"] = float(record.fit_seconds)
    if record.epochs_total is not None:
        samples[f"{GAUGE_PREFIX}epochs_total"] = float(record.epochs_total)
    if record.parameter_count is not None:
        samples[f"{GAUGE_PREFIX}parameter_count"] = float(record.parameter_count)
    for key, value in record.metrics.items():
        samples[sanitize_metric_name(key)] = float(value)
    grouping_key = {
        "mode": record.mode,
        "seed": str(record.seed),
        "run": record.run_name,
    }
    return PushPayload(
        job=job_name,
        grouping_key=grouping_key,
        samples=tuple(sorted(samples.items())),
        status=record.status,
    )


def probe_pushgateway(url: str, timeout: float = DEFAULT_PROBE_TIMEOUT) -> bool:
    """Check whether a Pushgateway answers its health endpoint.

    Args:
        url (str): Pushgateway base URL.
        timeout (float): Socket timeout in seconds.

    Returns:
        bool: True when the health endpoint returns a 2xx status.
    """
    health_url = f"{url.rstrip('/')}/-/healthy"
    try:
        with urlopen(health_url, timeout=timeout) as response:
            return 200 <= int(response.status) < 300
    except (OSError, ValueError):
        return False


class JobMetricsPush:
    """Guarded job reporting with an injectable push backend.

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
        self.backend = backend if backend is not None else PrometheusClientBackend()
        self.probe_timeout = probe_timeout
        self._active = False
        self._failure: str | None = None

    @classmethod
    def from_settings(cls, enabled: bool = True) -> JobMetricsPush:
        """Build a reporter from environment-backed service settings.

        Args:
            enabled (bool): Whether reporting is requested.

        Returns:
            JobMetricsPush: Reporter bound to ``ServiceSettings``.
        """
        settings = ServiceSettings()
        return cls(
            pushgateway_url=settings.pushgateway_url,
            enabled=enabled,
        )

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

    def push(self, record: JobRecord) -> bool:
        """Push one job record, or no-op when the reporter is inactive.

        Args:
            record (JobRecord): Details of one training run.

        Returns:
            bool: True when the payload was delivered.
        """
        if not self._active:
            return False
        try:
            self.backend.push(self.pushgateway_url, build_payload(record, self.job_name))
        except Exception as error:
            self._disable(str(error))
            return False
        return True

    def _disable(self, reason: str) -> None:
        """Deactivate the reporter and record the failure reason.

        Args:
            reason (str): Human-readable failure description.

        Returns:
            None: Subsequent calls become no-ops.
        """
        self._active = False
        self._failure = reason
        LOGGER.warning("pushgateway reporting disabled: %s", reason)
