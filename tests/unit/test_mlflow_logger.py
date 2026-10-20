"""Tests for the guarded MLflow tracker and its callback."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from eos.config import ExperimentConfig, ModelConfig, TrainConfig
from eos.training import mlflow_logger
from eos.training.lightning_module import EOSTrainingModule
from eos.training.mlflow_logger import (
    MlflowModuleBackend,
    MLflowTracker,
    MLflowTrackingCallback,
    collect_scalar_metrics,
    flatten_params,
    probe_tracking_server,
    resolve_git_commit,
)


class FakeBackend:
    """Record backend calls and optionally raise on one operation."""

    def __init__(self, fail_on: str | None = None) -> None:
        """Initialize an empty call log.

        Args:
            fail_on (str | None): Operation name that raises when invoked.

        Returns:
            None: The recorder is initialized in place.
        """
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.fail_on = fail_on

    def _record(self, name: str, *args: object, **kwargs: object) -> None:
        """Record one call or raise the configured failure.

        Args:
            name (str): Operation name.
            *args (object): Positional arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            None: The call is appended to the log.

        Raises:
            RuntimeError: If this operation is the configured failure.
        """
        if self.fail_on == name:
            raise RuntimeError("boom")
        self.calls.append((name, args, kwargs))

    def set_tracking_uri(self, uri: str) -> None:
        """Record a tracking-URI assignment.

        Args:
            uri (str): Tracking URI.

        Returns:
            None: The call is recorded.
        """
        self._record("set_tracking_uri", uri)

    def set_experiment(self, name: str) -> None:
        """Record an experiment selection.

        Args:
            name (str): Experiment name.

        Returns:
            None: The call is recorded.
        """
        self._record("set_experiment", name)

    def start_run(self, run_name: str) -> None:
        """Record a run start.

        Args:
            run_name (str): Run name.

        Returns:
            None: The call is recorded.
        """
        self._record("start_run", run_name)

    def log_params(self, params: dict[str, str]) -> None:
        """Record a parameter write.

        Args:
            params (dict[str, str]): Parameter mapping.

        Returns:
            None: The call is recorded.
        """
        self._record("log_params", params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Record a metric write.

        Args:
            metrics (dict[str, float]): Metric mapping.
            step (int | None): Optional step.

        Returns:
            None: The call is recorded.
        """
        self._record("log_metrics", metrics, step=step)

    def log_artifact(self, path: str) -> None:
        """Record an artifact upload.

        Args:
            path (str): Artifact path.

        Returns:
            None: The call is recorded.
        """
        self._record("log_artifact", path)

    def end_run(self) -> None:
        """Record a run close.

        Returns:
            None: The call is recorded.
        """
        self._record("end_run")


def _active_tracker(monkeypatch: pytest.MonkeyPatch, backend: FakeBackend) -> MLflowTracker:
    """Build and start a tracker with a reachable fake server.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to bypass the probe.
        backend (FakeBackend): Recording backend.

    Returns:
        MLflowTracker: Active tracker.
    """
    monkeypatch.setattr(
        mlflow_logger,
        "probe_tracking_server",
        lambda uri, timeout=mlflow_logger.DEFAULT_PROBE_TIMEOUT: True,
    )
    tracker = MLflowTracker(
        tracking_uri="http://localhost:5001",
        experiment_name="eos-test",
        run_name="run-1",
        backend=backend,
    )
    assert tracker.start() is True
    return tracker


def _module() -> EOSTrainingModule:
    """Build a tiny classifier module for callback tests.

    Returns:
        EOSTrainingModule: Classifier arm.
    """
    model = ModelConfig(
        latent_dim=16,
        base_channels=4,
        num_classes=2,
        head_hidden=8,
        dropout=0.0,
    )
    train = TrainConfig(max_epochs=1, warmup_epochs=0, accelerator="cpu", precision="32-true")
    return EOSTrainingModule("classifier", model, train)


def test_flatten_params_handles_nested_values() -> None:
    """Nested mappings, sequences, booleans, nulls, and paths must flatten."""
    payload = {
        "alpha": 1,
        "nested": {"flag": True, "empty": None},
        "grid": [0.25, 0.5],
        "path": Path("data/processed"),
    }
    assert flatten_params(payload) == {
        "alpha": "1",
        "nested.flag": "true",
        "nested.empty": "null",
        "grid": "[0.25, 0.5]",
        "path": "data/processed",
    }


def test_probe_accepts_local_stores_and_handles_unreachable_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-HTTP stores skip the probe and unreachable servers return false."""
    assert probe_tracking_server("file:./mlruns") is True
    assert probe_tracking_server("sqlite:///mlflow.db") is True

    def _raise(*args: object, **kwargs: object) -> None:
        """Raise the socket error used by an unreachable server.

        Args:
            *args (object): Ignored positional arguments.
            **kwargs (object): Ignored keyword arguments.

        Raises:
            OSError: Always.
        """
        raise OSError("unreachable")

    monkeypatch.setattr(mlflow_logger, "urlopen", _raise)
    assert probe_tracking_server("http://localhost:5999") is False


def test_tracker_logs_config_context_metrics_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A healthy tracker must forward every documented write."""
    backend = FakeBackend()
    tracker = _active_tracker(monkeypatch, backend)
    tracker.log_config(ExperimentConfig())
    tracker.log_context("f" * 64, "abc123", "hybrid", 42)
    tracker.log_metrics({"val/loss": 0.5}, step=3)
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")
    tracker.log_artifact(artifact)
    tracker.close()
    names = [call[0] for call in backend.calls]
    assert names == [
        "set_tracking_uri",
        "set_experiment",
        "start_run",
        "log_params",
        "log_params",
        "log_metrics",
        "log_artifact",
        "end_run",
    ]
    assert tracker.active is False
    assert tracker.failure is None
    assert tracker.artifacts_enabled is True


def test_artifact_failure_keeps_metric_tracking_alive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad artifact root must disable uploads without stopping metrics."""
    backend = FakeBackend(fail_on="log_artifact")
    tracker = _active_tracker(monkeypatch, backend)
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")
    tracker.log_artifact(artifact)
    assert tracker.active is True
    assert tracker.artifacts_enabled is False
    tracker.log_metrics({"val/loss": 0.5}, step=1)
    assert any(call[0] == "log_metrics" for call in backend.calls)
    tracker.log_artifact(artifact)
    assert tracker.active is True
    tracker.close()


def test_unreachable_server_disables_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed probe must leave the tracker inactive and training unaffected."""
    monkeypatch.setattr(
        mlflow_logger,
        "probe_tracking_server",
        lambda uri, timeout=mlflow_logger.DEFAULT_PROBE_TIMEOUT: False,
    )
    tracker = MLflowTracker(
        tracking_uri="http://localhost:5999",
        experiment_name="eos-test",
        run_name="run-1",
        backend=FakeBackend(),
    )
    assert tracker.start() is False
    assert tracker.active is False
    assert tracker.failure == "health probe failed"
    tracker.log_metrics({"val/loss": 1.0})
    tracker.close()


def test_disabled_tracker_never_contacts_the_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The MLflow switch must short-circuit before the probe."""
    backend = FakeBackend()
    tracker = MLflowTracker(
        tracking_uri="http://localhost:5001",
        experiment_name="eos-test",
        run_name="run-1",
        backend=backend,
        enabled=False,
    )
    assert tracker.start() is False
    assert backend.calls == []


def test_backend_failure_disables_later_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mid-run backend failure must disable the tracker, not abort training."""
    backend = FakeBackend(fail_on="log_metrics")
    tracker = _active_tracker(monkeypatch, backend)
    tracker.log_metrics({"val/loss": 0.5})
    assert tracker.active is False
    assert tracker.failure == "boom"
    tracker.log_metrics({"val/loss": 0.6})
    tracker.close()


def test_non_finite_metrics_and_missing_artifacts_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metric and artifact contracts must fail before touching the backend."""
    tracker = _active_tracker(monkeypatch, FakeBackend())
    with pytest.raises(FloatingPointError, match="not finite"):
        tracker.log_metrics({"val/loss": float("nan")})
    with pytest.raises(FileNotFoundError):
        tracker.log_artifact(tmp_path / "missing.json")
    tracker.close()


def test_collect_scalar_metrics_rejects_invalid_values() -> None:
    """Callback metrics must be finite scalars."""
    assert collect_scalar_metrics({"a": torch.tensor(0.5), "b": 2}) == {"a": 0.5, "b": 2.0}
    with pytest.raises(FloatingPointError, match="not finite"):
        collect_scalar_metrics({"a": torch.tensor(float("nan"))})
    with pytest.raises(ValueError, match="scalar"):
        collect_scalar_metrics({"a": torch.zeros(2)})


def test_resolve_git_commit_handles_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Git provenance must degrade to ``unknown`` rather than raise."""
    monkeypatch.setattr(
        mlflow_logger.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "abc123\n", ""),
    )
    assert resolve_git_commit() == "abc123"
    monkeypatch.setattr(
        mlflow_logger.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 1, "", "fatal"),
    )
    assert resolve_git_commit() == "unknown"

    def _raise(*args: object, **kwargs: object) -> None:
        """Raise the OS error produced when git is unavailable.

        Args:
            *args (object): Ignored positional arguments.
            **kwargs (object): Ignored keyword arguments.

        Raises:
            OSError: Always.
        """
        raise OSError("git missing")

    monkeypatch.setattr(mlflow_logger.subprocess, "run", _raise)
    assert resolve_git_commit() == "unknown"


def test_callback_logs_validation_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The validation callback must log every scalar metric once per epoch."""
    backend = FakeBackend()
    tracker = _active_tracker(monkeypatch, backend)
    trainer = SimpleNamespace(
        current_epoch=2,
        callback_metrics={"val/loss": torch.tensor(0.25), "val/macro_f1": 0.75},
    )
    callback = MLflowTrackingCallback(tracker)
    callback.on_validation_epoch_end(trainer, _module())  # type: ignore[arg-type]
    metric_calls = [call for call in backend.calls if call[0] == "log_metrics"]
    assert metric_calls == [
        ("log_metrics", ({"val/loss": 0.25, "val/macro_f1": 0.75},), {"step": 2})
    ]
    tracker.close()


def test_callback_logs_test_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The test callback must log held-out metrics after evaluation."""
    backend = FakeBackend()
    tracker = _active_tracker(monkeypatch, backend)
    trainer = SimpleNamespace(
        current_epoch=0,
        callback_metrics={"test/loss": torch.tensor(0.5), "test/macro_f1": 0.4},
    )
    callback = MLflowTrackingCallback(tracker)
    callback.on_test_epoch_end(trainer, _module())  # type: ignore[arg-type]
    metric_calls = [call for call in backend.calls if call[0] == "log_metrics"]
    assert metric_calls == [
        ("log_metrics", ({"test/loss": 0.5, "test/macro_f1": 0.4},), {"step": 0})
    ]
    tracker.close()


class FakeMlflowModule:
    """Record delegations made by the default MLflow backend."""

    def __init__(self) -> None:
        """Initialize the call log.

        Returns:
            None: The recorder is initialized in place.
        """
        self.calls: list[tuple[str, object]] = []

    def set_tracking_uri(self, uri: str) -> None:
        """Record a tracking-URI call.

        Args:
            uri (str): Tracking URI.

        Returns:
            None: The call is recorded.
        """
        self.calls.append(("set_tracking_uri", uri))

    def set_experiment(self, name: str) -> None:
        """Record an experiment call.

        Args:
            name (str): Experiment name.

        Returns:
            None: The call is recorded.
        """
        self.calls.append(("set_experiment", name))

    def start_run(self, run_name: str) -> None:
        """Record a start-run call.

        Args:
            run_name (str): Run name.

        Returns:
            None: The call is recorded.
        """
        self.calls.append(("start_run", run_name))

    def log_params(self, params: dict[str, str]) -> None:
        """Record a parameter call.

        Args:
            params (dict[str, str]): Parameter mapping.

        Returns:
            None: The call is recorded.
        """
        self.calls.append(("log_params", params))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Record a metric call.

        Args:
            metrics (dict[str, float]): Metric mapping.
            step (int | None): Optional step.

        Returns:
            None: The call is recorded.
        """
        self.calls.append(("log_metrics", (metrics, step)))

    def log_artifact(self, path: str) -> None:
        """Record an artifact call.

        Args:
            path (str): Artifact path.

        Returns:
            None: The call is recorded.
        """
        self.calls.append(("log_artifact", path))

    def end_run(self) -> None:
        """Record an end-run call.

        Returns:
            None: The call is recorded.
        """
        self.calls.append(("end_run", None))


def test_module_backend_delegates_every_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default backend must forward each call to the MLflow module."""
    fake = FakeMlflowModule()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    backend = MlflowModuleBackend()
    backend.set_tracking_uri("http://localhost:5001")
    backend.set_experiment("eos")
    backend.start_run("run")
    backend.log_params({"a": "1"})
    backend.log_metrics({"loss": 0.5}, step=2)
    backend.log_artifact("metrics.json")
    backend.end_run()
    assert fake.calls == [
        ("set_tracking_uri", "http://localhost:5001"),
        ("set_experiment", "eos"),
        ("start_run", "run"),
        ("log_params", {"a": "1"}),
        ("log_metrics", ({"loss": 0.5}, 2)),
        ("log_artifact", "metrics.json"),
        ("end_run", None),
    ]


class FakeResponse:
    """Minimal context-managed HTTP response."""

    def __init__(self, status: int) -> None:
        """Store the status code.

        Args:
            status (int): HTTP status code.

        Returns:
            None: The response is initialized in place.
        """
        self.status = status

    def __enter__(self) -> FakeResponse:
        """Return the response context.

        Returns:
            FakeResponse: This response.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Leave the response context.

        Args:
            *args (object): Ignored context arguments.

        Returns:
            None: No exception is suppressed.
        """


def test_probe_accepts_healthy_and_rejects_unhealthy_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP probes must distinguish a healthy server from an error page."""
    monkeypatch.setattr(
        mlflow_logger,
        "urlopen",
        lambda url, timeout: FakeResponse(200),
    )
    assert probe_tracking_server("https://localhost:5001") is True
    monkeypatch.setattr(
        mlflow_logger,
        "urlopen",
        lambda url, timeout: FakeResponse(503),
    )
    assert probe_tracking_server("https://localhost:5001") is False


def test_start_failure_disables_the_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    """An experiment-selection failure must not escape the tracker."""
    backend = FakeBackend(fail_on="set_experiment")
    monkeypatch.setattr(
        mlflow_logger,
        "probe_tracking_server",
        lambda uri, timeout=mlflow_logger.DEFAULT_PROBE_TIMEOUT: True,
    )
    tracker = MLflowTracker(
        tracking_uri="http://localhost:5001",
        experiment_name="eos-test",
        run_name="run-1",
        backend=backend,
    )
    assert tracker.start() is False
    assert tracker.active is False
    assert tracker.failure == "boom"


def test_close_swallows_backend_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing end_run must not abort the caller's cleanup path."""
    backend = FakeBackend(fail_on="end_run")
    tracker = _active_tracker(monkeypatch, backend)
    tracker.close()
    assert tracker.active is False


class EndFailingBackend(FakeBackend):
    """Backend whose end_run always fails."""

    def end_run(self) -> None:
        """Raise instead of closing a run.

        Returns:
            None: This method never returns.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("end boom")


def test_disable_tolerates_an_end_run_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure path must not recurse or raise when end_run fails."""
    backend = EndFailingBackend(fail_on="log_metrics")
    tracker = _active_tracker(monkeypatch, backend)
    tracker.log_metrics({"val/loss": 0.5})
    assert tracker.active is False
    assert tracker.failure == "boom"
