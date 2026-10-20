"""Tests for the guarded Pushgateway job reporter and its backend."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import replace
from types import ModuleType

import pytest

from eos.training import job_metrics
from eos.training.job_metrics import (
    GAUGE_PREFIX,
    JobMetricsPush,
    JobRecord,
    PrometheusClientBackend,
    PushPayload,
    build_payload,
    probe_pushgateway,
    sanitize_metric_name,
)


def make_record(**overrides: object) -> JobRecord:
    """Build a finished job record with defaults for unrelated fields.

    Args:
        **overrides (object): Field values that replace the defaults.

    Returns:
        JobRecord: Record suitable for payload construction.
    """
    base = JobRecord(
        run_name="hybrid-seed42-20261020T091343Z",
        mode="hybrid",
        seed=42,
        status="finished",
        duration_seconds=1222.6,
        fit_seconds=1180.2,
        epochs_total=30,
        parameter_count=6453421,
        metrics={"val/macro_f1": 0.949, "test/recon_mse": 0.0063},
    )
    return replace(base, **overrides)


class FakeBackend:
    """Record pushed payloads and optionally raise on push."""

    def __init__(self, fail: bool = False) -> None:
        """Initialize an empty push log.

        Args:
            fail (bool): Whether push raises instead of recording.

        Returns:
            None: The recorder is initialized in place.
        """
        self.calls: list[tuple[str, PushPayload]] = []
        self.fail = fail

    def push(self, url: str, payload: PushPayload) -> None:
        """Record one push or raise the configured failure.

        Args:
            url (str): Pushgateway base URL.
            payload (PushPayload): Prepared push payload.

        Returns:
            None: The call is recorded.

        Raises:
            RuntimeError: If this backend is configured to fail.
        """
        if self.fail:
            raise RuntimeError("gateway rejected push")
        self.calls.append((url, payload))


class FakeResponse:
    """Minimal context-manager response for the health probe."""

    def __init__(self, status: int) -> None:
        """Store the HTTP status.

        Args:
            status (int): Status code the fake server returns.

        Returns:
            None: The status is stored.
        """
        self.status = status

    def __enter__(self) -> FakeResponse:
        """Enter the context manager.

        Returns:
            FakeResponse: This response instance.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context manager.

        Args:
            *args (object): Unused exception information.

        Returns:
            None: Nothing is suppressed.
        """


def test_sanitize_metric_name_replaces_invalid_characters() -> None:
    """Slashes and punctuation become underscores before prefixing.

    Args:
        None: This test takes no arguments.

    Returns:
        None: Assertions verify the sanitized gauge name.
    """
    assert sanitize_metric_name("val/macro_f1") == f"{GAUGE_PREFIX}val_macro_f1"
    assert sanitize_metric_name("test/ssim-valid") == f"{GAUGE_PREFIX}test_ssim_valid"


def test_sanitize_metric_name_rejects_empty_result() -> None:
    """A key made only of invalid characters raises instead of colliding.

    Args:
        None: This test takes no arguments.

    Returns:
        None: Assertions verify the exception.

    Raises:
        ValueError: Expected from the sanitizer for the ``///`` key.
    """
    with pytest.raises(ValueError, match="empty name"):
        sanitize_metric_name("///")


def test_build_payload_labels_and_samples() -> None:
    """Grouping-key labels, gauges, and status are prepared correctly.

    Args:
        None: This test takes no arguments.

    Returns:
        None: Assertions verify labels and sample values.
    """
    payload = build_payload(make_record())
    assert payload.job == "eos-train"
    assert payload.grouping_key == {
        "mode": "hybrid",
        "seed": "42",
        "run": "hybrid-seed42-20261020T091343Z",
    }
    assert payload.status == "finished"
    samples = dict(payload.samples)
    assert samples[f"{GAUGE_PREFIX}duration_seconds"] == 1222.6
    assert samples[f"{GAUGE_PREFIX}fit_seconds"] == 1180.2
    assert samples[f"{GAUGE_PREFIX}epochs_total"] == 30.0
    assert samples[f"{GAUGE_PREFIX}parameter_count"] == 6453421.0
    assert samples[f"{GAUGE_PREFIX}val_macro_f1"] == 0.949
    assert samples[f"{GAUGE_PREFIX}test_recon_mse"] == 0.0063
    assert f"{GAUGE_PREFIX}push_timestamp_seconds" in samples
    assert payload.samples == tuple(sorted(payload.samples))


def test_build_payload_omits_absent_optional_fields() -> None:
    """Optional job fields produce no gauge when they are None.

    Args:
        None: This test takes no arguments.

    Returns:
        None: Assertions verify the omitted gauges.
    """
    record = make_record(fit_seconds=None, epochs_total=None, parameter_count=None)
    samples = dict(build_payload(record).samples)
    assert f"{GAUGE_PREFIX}fit_seconds" not in samples
    assert f"{GAUGE_PREFIX}epochs_total" not in samples
    assert f"{GAUGE_PREFIX}parameter_count" not in samples
    assert f"{GAUGE_PREFIX}duration_seconds" in samples


def test_build_payload_sanitizes_metric_keys() -> None:
    """Metric keys reach the payload as valid gauge names.

    Args:
        None: This test takes no arguments.

    Returns:
        None: Assertions verify sanitized names and float values.
    """
    record = make_record(metrics={"train/loss": 0.1504, "lr-AdamW": 0.000013})
    samples = dict(build_payload(record).samples)
    assert samples[f"{GAUGE_PREFIX}train_loss"] == 0.1504
    assert samples[f"{GAUGE_PREFIX}lr_AdamW"] == 0.000013


def test_probe_pushgateway_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 2xx health response reports the gateway as reachable.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to fake the URL opener.

    Returns:
        None: Assertions verify the probe result.
    """
    monkeypatch.setattr(job_metrics, "urlopen", lambda *a, **k: FakeResponse(200))
    assert probe_pushgateway("http://localhost:9091") is True


def test_probe_pushgateway_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-2xx responses and connection errors both report unreachable.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to fake the URL opener.

    Returns:
        None: Assertions verify both failure modes.
    """
    monkeypatch.setattr(job_metrics, "urlopen", lambda *a, **k: FakeResponse(503))
    assert probe_pushgateway("http://localhost:9091") is False

    def raise_error(*args: object, **kwargs: object) -> FakeResponse:
        """Refuse the connection like an unreachable gateway.

        Args:
            *args (object): Unused positional arguments.
            **kwargs (object): Unused keyword arguments.

        Returns:
            FakeResponse: Never returned; the function always raises.

        Raises:
            OSError: Always, to simulate an unreachable gateway.
        """
        raise OSError("connection refused")

    monkeypatch.setattr(job_metrics, "urlopen", raise_error)
    assert probe_pushgateway("http://localhost:9091") is False


def test_push_delivers_payload_when_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """An active reporter delivers the prepared payload to the backend.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to fake the health probe.

    Returns:
        None: Assertions verify delivery and the active state.
    """
    monkeypatch.setattr(job_metrics, "probe_pushgateway", lambda *a, **k: True)
    backend = FakeBackend()
    reporter = JobMetricsPush("http://localhost:9091", backend=backend)
    assert reporter.start() is True
    assert reporter.push(make_record()) is True
    assert len(backend.calls) == 1
    url, payload = backend.calls[0]
    assert url == "http://localhost:9091"
    assert payload.job == "eos-train"
    assert reporter.failure is None


def test_start_disabled_reports_inactive() -> None:
    """A disabled reporter never activates and push becomes a no-op.

    Args:
        None: This test takes no arguments.

    Returns:
        None: Assertions verify the inactive reporter contract.
    """
    backend = FakeBackend()
    reporter = JobMetricsPush("http://localhost:9091", enabled=False, backend=backend)
    assert reporter.start() is False
    assert reporter.active is False
    assert reporter.push(make_record()) is False
    assert backend.calls == []


def test_probe_failure_disables_reporter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed health probe disables reporting before any push.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to fake the health probe.

    Returns:
        None: Assertions verify the recorded failure.
    """
    monkeypatch.setattr(job_metrics, "probe_pushgateway", lambda *a, **k: False)
    reporter = JobMetricsPush("http://localhost:9091", backend=FakeBackend())
    assert reporter.start() is False
    assert reporter.failure == "health probe failed"
    assert reporter.push(make_record()) is False


def test_push_failure_disables_reporter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising backend disables reporting and later pushes are no-ops.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to fake the health probe.

    Returns:
        None: Assertions verify disablement and the no-op follow-up.
    """
    monkeypatch.setattr(job_metrics, "probe_pushgateway", lambda *a, **k: True)
    backend = FakeBackend(fail=True)
    reporter = JobMetricsPush("http://localhost:9091", backend=backend)
    assert reporter.start() is True
    assert reporter.push(make_record()) is False
    assert reporter.failure == "gateway rejected push"
    assert reporter.push(make_record()) is False
    assert len(backend.calls) == 0


def test_from_settings_reads_service_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory picks the Pushgateway URL from the environment.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to set the URL variable.

    Returns:
        None: Assertions verify the bound URL.
    """
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
    reporter = JobMetricsPush.from_settings()
    assert reporter.pushgateway_url == "http://pushgateway:9091"
    assert reporter.job_name == "eos-train"


def test_prometheus_client_backend_push(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default backend builds gauges and calls push_to_gateway correctly.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to install a fake module.

    Returns:
        None: Assertions verify gauge names, values, labels, and push arguments.
    """
    created: list[FakeGauge] = []
    pushed: list[tuple[str, dict[str, object]]] = []

    class FakeGauge:
        """Gauge double that records construction, values, and labels."""

        def __init__(
            self,
            name: str,
            doc: str,
            registry: object | None = None,
            labelnames: Sequence[str] = (),
        ) -> None:
            """Record gauge construction.

            Args:
                name (str): Gauge name.
                doc (str): Gauge help text.
                registry (object | None): Registry the gauge joins.
                labelnames (Sequence[str]): Declared label names.

            Returns:
                None: The gauge is recorded in the shared log.
            """
            self.name = name
            self.doc = doc
            self.registry = registry
            self.labelnames = tuple(labelnames)
            self.values: list[float] = []
            self.labels_seen: list[dict[str, str]] = []
            created.append(self)

        def set(self, value: float) -> None:
            """Record a sample value.

            Args:
                value (float): Gauge value.

            Returns:
                None: The value is appended to the gauge log.
            """
            self.values.append(float(value))

        def labels(self, **kwargs: str) -> FakeGauge:
            """Record label values and return the gauge.

            Args:
                **kwargs (str): Label name and value pairs.

            Returns:
                FakeGauge: This gauge, so chaining works.
            """
            self.labels_seen.append(dict(kwargs))
            return self

    class FakeRegistry:
        """Registry double that accepts gauges."""

        def __init__(self) -> None:
            """Initialize an empty registry.

            Args:
                None: This constructor takes no arguments.

            Returns:
                None: The registry list is initialized.
            """
            self.gauges: list[FakeGauge] = []

    def fake_push_to_gateway(url: str, **kwargs: object) -> None:
        """Record a push_to_gateway call.

        Args:
            url (str): Pushgateway base URL.
            **kwargs (object): Job, registry, grouping key, and timeout.

        Returns:
            None: The call is recorded in the shared log.
        """
        pushed.append((url, kwargs))

    fake_module = ModuleType("prometheus_client")
    fake_module.CollectorRegistry = FakeRegistry  # type: ignore[attr-defined]
    fake_module.Gauge = FakeGauge  # type: ignore[attr-defined]
    fake_module.push_to_gateway = fake_push_to_gateway  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prometheus_client", fake_module)

    payload = build_payload(make_record())
    PrometheusClientBackend().push("http://localhost:9091", payload)

    by_name = {gauge.name: gauge for gauge in created}
    assert f"{GAUGE_PREFIX}duration_seconds" in by_name
    assert by_name[f"{GAUGE_PREFIX}duration_seconds"].values == [1222.6]
    assert by_name[f"{GAUGE_PREFIX}val_macro_f1"].values == [0.949]
    info = by_name[f"{GAUGE_PREFIX}run_info"]
    assert info.labelnames == ("status",)
    assert info.labels_seen == [{"status": "finished"}]
    assert info.values == [1.0]
    assert len(pushed) == 1
    url, kwargs = pushed[0]
    assert url == "http://localhost:9091"
    assert kwargs["job"] == "eos-train"
    assert kwargs["grouping_key"] == payload.grouping_key
    assert kwargs["timeout"] == job_metrics.DEFAULT_PUSH_TIMEOUT


def test_default_constructors_use_real_backend() -> None:
    """An unspecified backend resolves to the prometheus_client adapter.

    Args:
        None: This test takes no arguments.

    Returns:
        None: Assertions verify the default backend type.
    """
    reporter = JobMetricsPush("http://localhost:9091")
    assert isinstance(reporter.backend, PrometheusClientBackend)
