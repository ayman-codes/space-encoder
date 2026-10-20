"""Integration tests: one real training epoch on a synthetic curated dataset."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from tests.helpers import build_curated_dataset

from eos.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from eos.training import job_metrics
from eos.training import runner as runner_module
from eos.training.job_metrics import JobMetricsPush, PushPayload
from eos.training.lightning_module import TrainingMode
from eos.training.runner import run_experiment

pytestmark = pytest.mark.integration


class RecordingTracker:
    """Tracker double that records lifecycle calls without MLflow."""

    def __init__(self) -> None:
        """Initialize an empty call log.

        Returns:
            None: The recorder is initialized in place.
        """
        self.calls: list[str] = []

    def start(self) -> bool:
        """Record a successful start.

        Returns:
            bool: Always true.
        """
        self.calls.append("start")
        return True

    def log_config(self, config: ExperimentConfig) -> None:
        """Record a configuration write.

        Args:
            config (ExperimentConfig): Logged configuration.

        Returns:
            None: The call is recorded.
        """
        del config
        self.calls.append("log_config")

    def log_context(
        self,
        manifest_fingerprint: str,
        git_commit: str,
        mode: str,
        seed: int,
    ) -> None:
        """Record a provenance write.

        Args:
            manifest_fingerprint (str): Manifest digest.
            git_commit (str): Repository commit.
            mode (str): Ablation arm.
            seed (int): Global seed.

        Returns:
            None: The call is recorded.
        """
        del manifest_fingerprint, git_commit, mode, seed
        self.calls.append("log_context")

    def log_artifact(self, path: Path) -> None:
        """Record an artifact upload.

        Args:
            path (Path): Artifact path.

        Returns:
            None: The call is recorded.
        """
        del path
        self.calls.append("log_artifact")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Record an epoch metric write.

        Args:
            metrics (dict[str, float]): Metric mapping.
            step (int | None): Optional epoch index.

        Returns:
            None: The call is recorded.
        """
        del metrics, step
        self.calls.append("log_metrics")

    def close(self) -> None:
        """Record a run close.

        Returns:
            None: The call is recorded.
        """
        self.calls.append("close")


def _config(dataset: Path, root: Path) -> ExperimentConfig:
    """Build a tiny CPU configuration for one arm.

    Args:
        dataset (Path): Curated dataset directory.
        root (Path): Output root for checkpoints and reports.

    Returns:
        ExperimentConfig: Validated configuration.
    """
    return ExperimentConfig(
        data=DataConfig(
            curated_root=dataset,
            image_size=64,
            channels=3,
            num_classes=2,
            batch_size=4,
            num_workers=0,
            seed=42,
        ),
        model=ModelConfig(
            input_size=64,
            in_channels=3,
            latent_dim=16,
            base_channels=4,
            downsample_stages=4,
            num_classes=2,
            head_hidden=8,
            dropout=0.0,
            recon_alpha=0.85,
            recon_lambda=0.3,
            recon_lambda_final=0.1,
        ),
        train=TrainConfig(
            max_epochs=1,
            learning_rate=1e-3,
            weight_decay=1e-4,
            warmup_epochs=0,
            gradient_clip=1.0,
            precision="32-true",
            accelerator="cpu",
            devices=1,
            seed=42,
            fast_dev_run=False,
            checkpoint_dir=root / "checkpoints",
            report_dir=root / "reports",
            mlflow_enabled=False,
            job_metrics_enabled=False,
        ),
    )


@pytest.mark.parametrize("mode", ["hybrid", "classifier", "autoencoder"])
def test_one_epoch_writes_checkpoint_and_metrics(
    mode: TrainingMode,
    tmp_path: Path,
) -> None:
    """Each arm must train one CPU epoch and emit its expected evidence."""
    dataset = build_curated_dataset(tmp_path, per_class=12)
    config = _config(dataset, tmp_path)
    result = run_experiment(mode, config)
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.is_file()
    assert result.metrics_path.is_file()
    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert payload["mode"] == mode
    assert payload["dataset"]["version"] == "synthetic-v1.0.0"
    assert payload["budget"]["max_epochs"] == 1
    assert payload["parameter_count"] > 0
    assert all(math.isfinite(value) for value in payload["metrics"].values())
    assert result.metrics == payload["metrics"]
    if mode in {"hybrid", "classifier"}:
        assert "val/macro_f1" in payload["metrics"]
        assert "val/accuracy" in payload["metrics"]
    else:
        assert "val/macro_f1" not in payload["metrics"]
        assert "val/recon_mse" in payload["metrics"]
        assert "val/ssim_valid" in payload["metrics"]
    if mode == "autoencoder":
        assert payload["metrics"]["train/recon_lambda"] == pytest.approx(1.0)
    elif mode == "hybrid":
        assert payload["metrics"]["train/recon_lambda"] == pytest.approx(0.3)


def test_active_tracker_receives_config_context_and_artifacts(tmp_path: Path) -> None:
    """A started tracker must receive provenance and both output artifacts."""
    dataset = build_curated_dataset(tmp_path, per_class=12)
    tracker = RecordingTracker()
    result = run_experiment("hybrid", _config(dataset, tmp_path), tracker=tracker)
    assert result.checkpoint_path is not None
    assert "start" in tracker.calls
    assert "log_config" in tracker.calls
    assert "log_context" in tracker.calls
    assert "log_metrics" in tracker.calls
    assert tracker.calls.count("log_artifact") >= 2
    assert tracker.calls[-1] == "close"


def test_resume_continues_from_a_saved_checkpoint(tmp_path: Path) -> None:
    """A resumed run must load the checkpoint and extend the epoch budget."""
    dataset = build_curated_dataset(tmp_path, per_class=12)
    first = run_experiment("hybrid", _config(dataset, tmp_path))
    assert first.checkpoint_path is not None
    extended = _config(dataset, tmp_path)
    extended = extended.model_copy(
        update={"train": extended.train.model_copy(update={"max_epochs": 2})}
    )
    second = run_experiment("hybrid", extended, resume=first.checkpoint_path)
    assert second.resumed_from == first.checkpoint_path
    payload = json.loads(second.metrics_path.read_text(encoding="utf-8"))
    assert payload["resumed_from"] == first.checkpoint_path.as_posix()
    assert payload["budget"]["max_epochs"] == 2


class RecordingBackend:
    """Push backend double that records payloads without a gateway."""

    def __init__(self) -> None:
        """Initialize an empty payload log.

        Args:
            None: This constructor takes no arguments.

        Returns:
            None: The recorder is initialized in place.
        """
        self.payloads: list[PushPayload] = []

    def push(self, url: str, payload: PushPayload) -> None:
        """Record one payload.

        Args:
            url (str): Pushgateway base URL.
            payload (PushPayload): Prepared push payload.

        Returns:
            None: The payload is recorded.
        """
        del url
        self.payloads.append(payload)


def test_finished_run_pushes_job_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed run must report job details to the push backend.

    Args:
        tmp_path (Path): Isolated output root.
        monkeypatch (pytest.MonkeyPatch): Fixture used to fake the health probe.

    Returns:
        None: Assertions verify the pushed payload and the metrics JSON.
    """
    monkeypatch.setattr(job_metrics, "probe_pushgateway", lambda *args, **kwargs: True)
    backend = RecordingBackend()
    dataset = build_curated_dataset(tmp_path, per_class=12)
    reporter = JobMetricsPush("http://gateway.test:9091", backend=backend)
    result = run_experiment("hybrid", _config(dataset, tmp_path), job_push=reporter)
    assert len(backend.payloads) == 1
    pushed = backend.payloads[0]
    assert pushed.status == "finished"
    assert pushed.job == "eos-train"
    assert pushed.grouping_key["mode"] == "hybrid"
    assert pushed.grouping_key["seed"] == "42"
    assert pushed.grouping_key["run"].startswith("hybrid-seed42-")
    samples = dict(pushed.samples)
    assert samples["eos_train_fit_seconds"] > 0
    assert samples["eos_train_duration_seconds"] >= samples["eos_train_fit_seconds"]
    assert samples["eos_train_epochs_total"] == 1.0
    assert samples["eos_train_parameter_count"] == float(result.parameter_count)
    metrics_payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["fit_seconds"] > 0


def test_failed_run_reports_job_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that raises must report failure before propagating the error.

    Args:
        tmp_path (Path): Isolated output root.
        monkeypatch (pytest.MonkeyPatch): Fixture used to fake the probe and the
            trainer factory.

    Returns:
        None: Assertions verify the failed payload and re-raised error.

    Raises:
        RuntimeError: Expected from the patched trainer factory.
    """
    monkeypatch.setattr(job_metrics, "probe_pushgateway", lambda *args, **kwargs: True)
    backend = RecordingBackend()
    dataset = build_curated_dataset(tmp_path, per_class=12)
    reporter = JobMetricsPush("http://gateway.test:9091", backend=backend)

    def raise_before_training(*args: object, **kwargs: object) -> object:
        """Refuse to build a trainer so the run fails mid-flight.

        Args:
            *args (object): Unused positional arguments.
            **kwargs (object): Unused keyword arguments.

        Returns:
            object: Never returned; the function always raises.

        Raises:
            RuntimeError: Always, to simulate a mid-run crash.
        """
        del args, kwargs
        raise RuntimeError("trainer boom")

    monkeypatch.setattr(runner_module, "build_trainer", raise_before_training)
    with pytest.raises(RuntimeError, match="trainer boom"):
        run_experiment("hybrid", _config(dataset, tmp_path), job_push=reporter)
    assert len(backend.payloads) == 1
    pushed = backend.payloads[0]
    assert pushed.status == "failed"
    samples = dict(pushed.samples)
    assert samples["eos_train_duration_seconds"] >= 0
    assert "eos_train_fit_seconds" not in samples
