"""MLflow tracking that never blocks training.

The tracker is a thin, injectable facade over the MLflow client. It flattens the
complete experiment configuration into scalar parameters, records the dataset
manifest fingerprint and git commit, logs per-epoch metrics, and uploads the best
checkpoint as an artifact.

A tracking server that is unreachable, slow, or misconfigured disables the
tracker after a short probe. Every subsequent call becomes a no-op, so a network
failure can never abort a training run.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse
from urllib.request import urlopen

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from eos.config import ExperimentConfig, ServiceSettings

LOGGER = logging.getLogger(__name__)

DEFAULT_PROBE_TIMEOUT = 2.0
"""Seconds allowed for the tracking-server health probe."""


class MLflowBackend(Protocol):
    """Minimal MLflow surface consumed by the tracker."""

    def set_tracking_uri(self, uri: str) -> None:
        """Set the tracking server URI.

        Args:
            uri (str): Tracking server or local store URI.

        Returns:
            None: The client is configured in place.
        """

    def set_experiment(self, name: str) -> None:
        """Select or create an experiment.

        Args:
            name (str): Experiment name.

        Returns:
            None: The experiment is resolved in place.
        """

    def start_run(self, run_name: str) -> None:
        """Start a tracked run.

        Args:
            run_name (str): Human-readable run name.

        Returns:
            None: A run is activated in place.
        """

    def log_params(self, params: Mapping[str, str]) -> None:
        """Log scalar parameters.

        Args:
            params (Mapping[str, str]): Flattened parameter mapping.

        Returns:
            None: Parameters are written to the active run.
        """

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log scalar metrics.

        Args:
            metrics (Mapping[str, float]): Metric mapping.
            step (int | None): Optional epoch or step index.

        Returns:
            None: Metrics are written to the active run.
        """

    def log_artifact(self, path: str) -> None:
        """Upload one artifact.

        Args:
            path (str): Local artifact path.

        Returns:
            None: The artifact is uploaded to the active run.
        """

    def end_run(self) -> None:
        """Close the active run.

        Returns:
            None: The run is finalized.
        """


class MlflowModuleBackend:
    """Default backend that delegates to the installed MLflow module."""

    def set_tracking_uri(self, uri: str) -> None:
        """Set the tracking server URI.

        Args:
            uri (str): Tracking server or local store URI.

        Returns:
            None: MLflow is configured in place.
        """
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri(uri)

    def set_experiment(self, name: str) -> None:
        """Select or create an experiment.

        Args:
            name (str): Experiment name.

        Returns:
            None: The experiment is resolved in place.
        """
        import mlflow  # noqa: PLC0415

        mlflow.set_experiment(name)

    def start_run(self, run_name: str) -> None:
        """Start a tracked run.

        Args:
            run_name (str): Human-readable run name.

        Returns:
            None: A run is activated in place.
        """
        import mlflow  # noqa: PLC0415

        mlflow.start_run(run_name=run_name)

    def log_params(self, params: Mapping[str, str]) -> None:
        """Log scalar parameters.

        Args:
            params (Mapping[str, str]): Flattened parameter mapping.

        Returns:
            None: Parameters are written to the active run.
        """
        import mlflow  # noqa: PLC0415

        mlflow.log_params(dict(params))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log scalar metrics.

        Args:
            metrics (Mapping[str, float]): Metric mapping.
            step (int | None): Optional epoch or step index.

        Returns:
            None: Metrics are written to the active run.
        """
        import mlflow  # noqa: PLC0415

        mlflow.log_metrics(dict(metrics), step=step)

    def log_artifact(self, path: str) -> None:
        """Upload one artifact.

        Args:
            path (str): Local artifact path.

        Returns:
            None: The artifact is uploaded to the active run.
        """
        import mlflow  # noqa: PLC0415

        mlflow.log_artifact(path)

    def end_run(self) -> None:
        """Close the active run.

        Returns:
            None: The run is finalized.
        """
        import mlflow  # noqa: PLC0415

        mlflow.end_run()


def flatten_params(payload: Mapping[str, object], prefix: str = "") -> dict[str, str]:
    """Flatten a nested configuration mapping into dotted string parameters.

    Args:
        payload (Mapping[str, object]): Nested configuration values.
        prefix (str): Optional key prefix used by recursion.

    Returns:
        dict[str, str]: Flat mapping suitable for ``mlflow.log_params``. Nested
            mappings use dotted paths, sequences are JSON encoded, ``None``
            becomes ``"null"``, and booleans become lowercase strings.
    """
    flattened: dict[str, str] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(flatten_params(value, path))
        elif isinstance(value, bool):
            flattened[path] = "true" if value else "false"
        elif value is None:
            flattened[path] = "null"
        elif isinstance(value, (list, tuple)):
            flattened[path] = json.dumps(list(value), sort_keys=True)
        elif isinstance(value, Path):
            flattened[path] = value.as_posix()
        else:
            flattened[path] = str(value)
    return flattened


def resolve_git_commit(root: Path | None = None) -> str:
    """Return the current git commit, or ``"unknown"`` when unavailable.

    Args:
        root (Path | None): Repository directory. Defaults to the process working
            directory.

    Returns:
        str: Full commit hash or ``"unknown"`` when git is missing, the command
            fails, or the directory is not a repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root or Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


def probe_tracking_server(uri: str, timeout: float = DEFAULT_PROBE_TIMEOUT) -> bool:
    """Check whether an HTTP tracking server answers its health endpoint.

    Local stores such as ``file:`` and ``sqlite:`` have no server to probe and
    are reported as reachable so MLflow can handle them directly.

    Args:
        uri (str): MLflow tracking URI.
        timeout (float): Socket timeout in seconds.

    Returns:
        bool: True when the server is reachable or the URI is not HTTP-based.
    """
    parsed = urlparse(uri)
    if parsed.scheme not in {"http", "https"}:
        return True
    health_url = f"{uri.rstrip('/')}/health"
    try:
        with urlopen(health_url, timeout=timeout) as response:
            return 200 <= int(response.status) < 300
    except (OSError, ValueError):
        return False


def collect_scalar_metrics(
    values: Mapping[str, Tensor | int | float],
) -> dict[str, float]:
    """Convert callback metrics into finite Python floats.

    Args:
        values (Mapping[str, Tensor | int | float]): Lightning callback metrics.

    Returns:
        dict[str, float]: Scalar metrics keyed by their logging names.

    Raises:
        FloatingPointError: If any scalar is NaN or infinite.
        ValueError: If a tensor metric is not scalar.
    """
    metrics: dict[str, float] = {}
    for name, value in values.items():
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise ValueError(f"metric {name!r} must be scalar, received {tuple(value.shape)}")
            number = float(value.detach().cpu())
        else:
            number = float(value)
        if not math.isfinite(number):
            raise FloatingPointError(f"metric {name!r} is not finite: {number}")
        metrics[name] = number
    return metrics


class MLflowTracker:
    """Guarded MLflow run lifecycle with an injectable backend.

    Args:
        tracking_uri (str): MLflow tracking URI.
        experiment_name (str): Experiment name.
        run_name (str): Run name.
        backend (MLflowBackend): Client facade, injectable for tests.
        enabled (bool): Whether tracking is requested. Defaults to true.
        probe_timeout (float): Health-probe timeout in seconds.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        tracking_uri: str,
        experiment_name: str,
        run_name: str,
        backend: MLflowBackend,
        enabled: bool = True,
        probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
    ) -> None:
        """Store the client configuration without contacting the server.

        Args:
            tracking_uri (str): MLflow tracking URI.
            experiment_name (str): Experiment name.
            run_name (str): Run name.
            backend (MLflowBackend): Client facade.
            enabled (bool): Whether tracking is requested.
            probe_timeout (float): Health-probe timeout in seconds.

        Returns:
            None: The tracker starts inactive until :meth:`start` succeeds.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.backend = backend
        self.enabled = enabled
        self.probe_timeout = probe_timeout
        self._active = False
        self._artifacts_enabled = True
        self._failure: str | None = None

    @classmethod
    def from_settings(
        cls,
        experiment_name: str,
        run_name: str,
        enabled: bool = True,
    ) -> MLflowTracker:
        """Build a tracker from environment-backed service settings.

        Args:
            experiment_name (str): Experiment name.
            run_name (str): Run name.
            enabled (bool): Whether tracking is requested.

        Returns:
            MLflowTracker: Tracker bound to ``ServiceSettings``.
        """
        settings = ServiceSettings()
        return cls(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name,
            backend=MlflowModuleBackend(),
            enabled=enabled,
        )

    @property
    def active(self) -> bool:
        """Return whether the tracker is currently writing to MLflow.

        Returns:
            bool: True only after a successful start with no later failure.
        """
        return self._active

    @property
    def failure(self) -> str | None:
        """Return the reason tracking was disabled, if any.

        Returns:
            str | None: Last failure description.
        """
        return self._failure

    @property
    def artifacts_enabled(self) -> bool:
        """Return whether artifact uploads are still attempted.

        Returns:
            bool: False after an artifact upload has failed.
        """
        return self._artifacts_enabled

    def start(self) -> bool:
        """Probe the server and start a run.

        Returns:
            bool: True when the tracker is active.
        """
        if not self.enabled:
            return False
        if not probe_tracking_server(self.tracking_uri, self.probe_timeout):
            self._disable("health probe failed")
            return False
        try:
            self.backend.set_tracking_uri(self.tracking_uri)
            self.backend.set_experiment(self.experiment_name)
            self.backend.start_run(self.run_name)
        except Exception as error:
            self._disable(str(error))
            return False
        self._active = True
        return True

    def log_config(self, config: ExperimentConfig) -> None:
        """Log the complete experiment configuration.

        Args:
            config (ExperimentConfig): Fully validated configuration.

        Returns:
            None: Parameters are written through the guarded backend.
        """
        params = flatten_params(config.model_dump(mode="json"))
        self._safe(lambda: self.backend.log_params(params))

    def log_context(
        self,
        manifest_fingerprint: str,
        git_commit: str,
        mode: str,
        seed: int,
    ) -> None:
        """Log dataset and provenance parameters.

        Args:
            manifest_fingerprint (str): Manifest SHA-256 fingerprint.
            git_commit (str): Repository commit or ``"unknown"``.
            mode (str): Ablation arm.
            seed (int): Global seed.

        Returns:
            None: Parameters are written through the guarded backend.
        """
        params = {
            "run/mode": mode,
            "run/seed": str(seed),
            "dataset/manifest_sha256": manifest_fingerprint,
            "git/commit": git_commit,
        }
        self._safe(lambda: self.backend.log_params(params))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Validate and log scalar metrics.

        Args:
            metrics (Mapping[str, float]): Metric mapping.
            step (int | None): Optional epoch index.

        Returns:
            None: Metrics are written through the guarded backend.

        Raises:
            FloatingPointError: If any metric is NaN or infinite.
        """
        for name, value in metrics.items():
            if not math.isfinite(value):
                raise FloatingPointError(f"metric {name!r} is not finite: {value}")
        self._safe(lambda: self.backend.log_metrics(metrics, step=step))

    def log_artifact(self, path: Path) -> None:
        """Log one local artifact.

        A failed upload disables further artifact attempts but keeps metric and
        parameter tracking alive. This matters when the tracking server accepts
        metadata but its artifact root is misconfigured: the run remains usable
        instead of being torn down by a filesystem error.

        Args:
            path (Path): Existing artifact path.

        Returns:
            None: The artifact is uploaded or artifact logging is disabled.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        if not path.is_file():
            raise FileNotFoundError(f"artifact not found: {path}")
        if not self._active or not self._artifacts_enabled:
            return
        try:
            self.backend.log_artifact(str(path))
        except Exception as error:
            self._artifacts_enabled = False
            LOGGER.warning("MLflow artifact upload disabled: %s", error)

    def close(self) -> None:
        """Close the active run, ignoring backend failures.

        Returns:
            None: The tracker becomes inactive.
        """
        if self._active:
            try:
                self.backend.end_run()
            except Exception as error:
                LOGGER.warning("MLflow end_run failed: %s", error)
        self._active = False

    def _safe(self, operation: Callable[[], None]) -> None:
        """Execute a backend operation and disable on failure.

        Args:
            operation (Callable[[], None]): Deferred backend call.

        Returns:
            None: The operation is executed or the tracker is disabled.
        """
        if not self._active:
            return
        try:
            operation()
        except Exception as error:
            self._disable(str(error))

    def _disable(self, reason: str) -> None:
        """Disable tracking and record the failure reason.

        Args:
            reason (str): Human-readable failure description.

        Returns:
            None: The tracker becomes inactive.
        """
        was_active = self._active
        self._active = False
        self._failure = reason
        LOGGER.warning("MLflow tracking disabled: %s", reason)
        if was_active:
            try:
                self.backend.end_run()
            except Exception:
                LOGGER.debug("MLflow end_run failed while disabling tracking")


class MLflowTrackingCallback(Callback):
    """Log per-epoch metrics and the best checkpoint to MLflow.

    Args:
        tracker (MLflowTracker): Guarded tracker shared with the runner.
    """

    def __init__(self, tracker: MLflowTracker) -> None:
        """Store the tracker.

        Args:
            tracker (MLflowTracker): Guarded tracker.

        Returns:
            None: The callback is initialized in place.
        """
        super().__init__()
        self.tracker = tracker

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the current epoch's scalar metrics.

        Args:
            trainer (Trainer): Active Lightning trainer.
            pl_module (LightningModule): Module being trained.

        Returns:
            None: Metrics are written through the tracker.
        """
        del pl_module
        metrics = collect_scalar_metrics(trainer.callback_metrics)
        self.tracker.log_metrics(metrics, step=trainer.current_epoch)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the held-out test metrics after evaluation.

        Args:
            trainer (Trainer): Active Lightning trainer.
            pl_module (LightningModule): Module being evaluated.

        Returns:
            None: Metrics are written through the tracker.
        """
        del pl_module
        metrics = collect_scalar_metrics(trainer.callback_metrics)
        self.tracker.log_metrics(metrics, step=trainer.current_epoch)
