"""Tests for the Prometheus export of drift and compression telemetry."""

from __future__ import annotations

import pytest

import eos.telemetry.export as export_module
from eos.telemetry.export import (
    ASVD_F1_DELTA,
    ASVD_PARAM_RATIO,
    MMD_ALARM,
    MMD_SCORE,
    TelemetryPush,
    build_asvd_payload,
    build_mmd_payload,
)
from eos.telemetry.mmd_drift import DriftScore
from eos.training.job_metrics import PushPayload


class RecordingBackend:
    """Backend that captures payloads instead of contacting a gateway."""

    def __init__(self) -> None:
        """Initialize an empty capture list that does not fail."""
        self.payloads: list[PushPayload] = []
        self.fail = False

    def push(self, url: str, payload: PushPayload) -> None:
        """Capture one payload.

        Args:
            url (str): Gateway URL, retained for signature parity.
            payload (PushPayload): Payload to capture.

        Returns:
            None: The payload is appended to the capture list.

        Raises:
            RuntimeError: If the backend is armed to fail.
        """
        if self.fail:
            raise RuntimeError("gateway rejected the push")
        self.payloads.append(payload)


def test_build_mmd_payload_uses_dashboard_names() -> None:
    """The drift gauges must match the Grafana queries exactly."""
    assert MMD_SCORE == "eos_mmd_score"
    assert MMD_ALARM == "eos_mmd_alarm"
    alarmed = build_mmd_payload(
        DriftScore(mmd2=0.42, threshold=0.3, alarm=True, batch_size=128),
        fingerprint="abc",
    )
    quiet_score = DriftScore(mmd2=0.02, threshold=0.3, alarm=False, batch_size=128)
    quiet = build_mmd_payload(quiet_score, fingerprint="abc")
    assert dict(alarmed.samples) == {"eos_mmd_score": 0.42, "eos_mmd_alarm": 1.0}
    assert dict(quiet.samples)["eos_mmd_alarm"] == 0.0
    assert alarmed.grouping_key == {"kind": "mmd", "model": "abc", "scenario": "live"}
    assert alarmed.job == "eos-telemetry"
    demo = build_mmd_payload(quiet_score, fingerprint="abc", scenario="class_mix")
    assert demo.grouping_key["scenario"] == "class_mix"


def test_build_asvd_payload_uses_dashboard_names() -> None:
    """The compression gauges must match the Grafana queries exactly."""
    assert ASVD_PARAM_RATIO == "eos_asvd_param_ratio"
    assert ASVD_F1_DELTA == "eos_asvd_f1_delta"
    payload = build_asvd_payload(0.6085, 0.0002, fingerprint="def")
    assert dict(payload.samples) == {
        "eos_asvd_param_ratio": 0.6085,
        "eos_asvd_f1_delta": 0.0002,
    }
    assert payload.grouping_key == {"kind": "asvd", "model": "def"}


def test_push_delivers_payload_when_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """An armed reporter must deliver payloads to the backend."""
    backend = RecordingBackend()
    reporter = TelemetryPush("http://gateway:9091", backend=backend)
    monkeypatch.setattr(export_module, "probe_pushgateway", lambda url, timeout: True)
    assert reporter.start()
    payload = build_asvd_payload(0.5, 0.0, fingerprint="x")
    assert reporter.push(payload)
    assert backend.payloads == [payload]


def test_push_is_noop_when_inactive() -> None:
    """An inactive reporter must deliver nothing."""
    backend = RecordingBackend()
    reporter = TelemetryPush("http://gateway:9091", backend=backend)
    assert not reporter.push(build_asvd_payload(0.5, 0.0, fingerprint="x"))
    assert backend.payloads == []


def test_probe_failure_disables_reporter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed health probe must disarm the reporter with a reason."""
    backend = RecordingBackend()
    reporter = TelemetryPush("http://gateway:9091", backend=backend)
    monkeypatch.setattr(export_module, "probe_pushgateway", lambda url, timeout: False)
    assert not reporter.start()
    assert not reporter.active
    assert reporter.failure == "health probe failed"


def test_push_failure_disables_reporter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend failure must disarm the reporter and record the reason."""
    backend = RecordingBackend()
    reporter = TelemetryPush("http://gateway:9091", backend=backend)
    monkeypatch.setattr(export_module, "probe_pushgateway", lambda url, timeout: True)
    assert reporter.start()
    backend.fail = True
    assert not reporter.push(
        build_mmd_payload(
            DriftScore(mmd2=0.1, threshold=0.2, alarm=False, batch_size=16),
            fingerprint="x",
        )
    )
    assert not reporter.active
    assert reporter.failure == "gateway rejected the push"
