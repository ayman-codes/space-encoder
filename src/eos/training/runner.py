"""Experiment orchestration for the three ablation arms.

The runner owns the parts that must stay identical across arms: seeding, data
setup, trainer construction, metric collection, and artifact writing. The arm
difference is confined to the objective in :class:`EOSTrainingModule`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from eos.config import ExperimentConfig
from eos.dataset.eurosat_datamodule import EuroSATDataModule
from eos.training.callbacks import build_callbacks
from eos.training.job_metrics import JobMetricsPush, JobRecord
from eos.training.lightning_module import EOSTrainingModule, TrainingMode
from eos.training.mlflow_logger import (
    MLflowTracker,
    MLflowTrackingCallback,
    collect_scalar_metrics,
    resolve_git_commit,
)
from eos.training.precision import resolve_precision
from eos.utils.seeding import seed_everything

SCHEMA_VERSION = 1
"""Version of the metrics and ablation JSON documents."""


@dataclass(frozen=True)
class RunResult:
    """Summary of one completed training arm.

    Args:
        mode (TrainingMode): Ablation arm.
        checkpoint_path (Path | None): Best checkpoint, when one was written.
        metrics_path (Path): Per-run metrics JSON path.
        metrics (dict[str, float]): Validation and test metrics.
        budget (dict[str, object]): Identical budget fields for all arms.
        parameter_count (int): Trainable parameter count.
        dataset (dict[str, str]): Dataset version and manifest fingerprint.
        git_commit (str): Repository commit or ``"unknown"``.
        resumed_from (Path | None): Checkpoint used to resume, when applicable.
    """

    mode: TrainingMode
    checkpoint_path: Path | None
    metrics_path: Path
    metrics: dict[str, float]
    budget: dict[str, object]
    parameter_count: int
    dataset: dict[str, str]
    git_commit: str
    resumed_from: Path | None = None


def normalize_fast_dev_run(config: ExperimentConfig) -> ExperimentConfig:
    """Constrain a fast-dev-run to one epoch and no warmup.

    Lightning's own ``fast_dev_run`` suppresses logging and checkpointing, so the
    budget is expressed through batch limits instead. The epoch count must also
    be normalized because the learning-rate and lambda schedules use it.

    Args:
        config (ExperimentConfig): Loaded configuration.

    Returns:
        ExperimentConfig: Configuration with a one-epoch fast budget when
            requested, otherwise the input configuration.
    """
    if not config.train.fast_dev_run:
        return config
    train = config.train.model_copy(update={"max_epochs": 1, "warmup_epochs": 0})
    return config.model_copy(update={"train": train})


def validate_consistency(config: ExperimentConfig) -> None:
    """Reject configurations whose data and model sections disagree.

    Args:
        config (ExperimentConfig): Loaded configuration.

    Returns:
        None: The sections agree.

    Raises:
        ValueError: If class count, image size, or channel count differ between
            the data and model sections.
    """
    if config.data.num_classes != config.model.num_classes:
        raise ValueError(
            f"data.num_classes={config.data.num_classes} does not match "
            f"model.num_classes={config.model.num_classes}"
        )
    if config.data.image_size != config.model.input_size:
        raise ValueError(
            f"data.image_size={config.data.image_size} does not match "
            f"model.input_size={config.model.input_size}"
        )
    if config.data.channels != config.model.in_channels:
        raise ValueError(
            f"data.channels={config.data.channels} does not match "
            f"model.in_channels={config.model.in_channels}"
        )


def budget_payload(config: ExperimentConfig) -> dict[str, object]:
    """Return the budget fields shared by every ablation arm.

    Args:
        config (ExperimentConfig): Normalized configuration.

    Returns:
        dict[str, object]: Epochs, batch size, optimizer, precision, and seed.
    """
    return {
        "max_epochs": config.train.max_epochs,
        "batch_size": config.data.batch_size,
        "learning_rate": config.train.learning_rate,
        "weight_decay": config.train.weight_decay,
        "warmup_epochs": config.train.warmup_epochs,
        "gradient_clip": config.train.gradient_clip,
        "precision": config.train.precision,
        "accelerator": config.train.accelerator,
        "devices": config.train.devices,
        "seed": config.train.seed,
        "class_weights": config.train.class_weights,
        "fast_dev_run": config.train.fast_dev_run,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON document atomically.

    Args:
        path (Path): Destination path.
        payload (dict[str, object]): JSON-serializable payload.

    Returns:
        None: The destination is replaced atomically and no temporary file
            remains.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_trainer(
    mode: TrainingMode,
    config: ExperimentConfig,
    tracker: MLflowTracker | None = None,
) -> Trainer:
    """Build the Lightning trainer shared by all arms.

    Args:
        mode (TrainingMode): Ablation arm, used for checkpoint monitoring.
        config (ExperimentConfig): Normalized configuration.
        tracker (MLflowTracker | None): Optional guarded tracker.

    Returns:
        Trainer: Configured trainer with checkpointing and metric logging.
    """
    precision = resolve_precision(config.train.precision, torch.cuda.is_available())
    callbacks = build_callbacks(mode, config.train.checkpoint_dir)
    if tracker is not None:
        callbacks.append(MLflowTrackingCallback(tracker))
    fast = config.train.fast_dev_run
    limit_train_batches: int | None = 1 if fast else None
    limit_val_batches: int | None = 1 if fast else None
    limit_test_batches: int | None = 1 if fast else None
    logger = CSVLogger(
        save_dir=config.train.report_dir / "lightning_logs",
        name=mode,
        version=0,
    )
    return Trainer(
        max_epochs=config.train.max_epochs,
        accelerator=config.train.accelerator,
        devices=config.train.devices,
        precision=precision,
        gradient_clip_val=config.train.gradient_clip,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        deterministic="warn",
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        default_root_dir=config.train.checkpoint_dir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )


def _best_checkpoint(trainer: Trainer) -> Path | None:
    """Return the best checkpoint path when a ModelCheckpoint wrote one.

    Args:
        trainer (Trainer): Fitted trainer.

    Returns:
        Path | None: Existing checkpoint path or ``None``.
    """
    callback = trainer.checkpoint_callback
    if not isinstance(callback, ModelCheckpoint):
        return None
    path = Path(callback.best_model_path)
    return path if path.is_file() else None


def _report_run(
    metrics_path: Path,
    payload: dict[str, object],
    tracker: MLflowTracker,
    jobs: JobMetricsPush,
    record: JobRecord,
) -> None:
    """Write run evidence and report the job to the observability stack.

    The metrics JSON is written atomically, uploaded to the tracking server
    together with the best checkpoint referenced in the payload, and the job
    record is pushed to the Pushgateway. Artifact failures are isolated by the
    tracker and never abort the run.

    Args:
        metrics_path (Path): Destination metrics JSON path.
        payload (dict[str, object]): Serialized run payload.
        tracker (MLflowTracker): Guarded tracking facade.
        jobs (JobMetricsPush): Guarded job reporter.
        record (JobRecord): Job details for Prometheus.

    Returns:
        None: Evidence is written and the job is reported.
    """
    write_json(metrics_path, payload)
    tracker.log_artifact(metrics_path)
    checkpoint = payload.get("best_checkpoint")
    if isinstance(checkpoint, str):
        tracker.log_artifact(Path(checkpoint))
    jobs.push(record)


def run_experiment(
    mode: TrainingMode,
    config: ExperimentConfig,
    tracker: MLflowTracker | None = None,
    resume: Path | None = None,
    job_push: JobMetricsPush | None = None,
) -> RunResult:
    """Run one training arm end to end.

    Args:
        mode (TrainingMode): Ablation arm.
        config (ExperimentConfig): Loaded configuration.
        tracker (MLflowTracker | None): Optional tracker. When omitted, a tracker
            is built from service settings and the configured experiment name.
        resume (Path | None): Optional checkpoint used to resume training.
        job_push (JobMetricsPush | None): Optional job reporter. When omitted, a
            reporter is built from service settings and the configured switch.

    Returns:
        RunResult: Checkpoint, metrics, budget, and provenance.

    Raises:
        ValueError: If the data and model sections disagree or the arm is
            unknown.
        FileNotFoundError: If a resume checkpoint was requested but is missing.
        RuntimeError: If the data module does not expose a loaded manifest.
    """
    if mode not in ("hybrid", "classifier", "autoencoder"):
        raise ValueError(f"unsupported training mode {mode!r}")
    if resume is not None and not resume.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {resume}")
    config = normalize_fast_dev_run(config)
    validate_consistency(config)
    seed = config.train.seed
    seed_everything(seed, deterministic=True)
    datamodule = EuroSATDataModule(config.data)
    datamodule.setup()
    manifest = datamodule.manifest
    if manifest is None:
        raise RuntimeError("data module did not produce a manifest")
    class_weights = datamodule.class_weights if config.train.class_weights else None
    module = EOSTrainingModule(mode, config.model, config.train, class_weights=class_weights)
    git_commit = resolve_git_commit()
    run_name = f"{mode}-seed{seed}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    active_tracker = tracker or MLflowTracker.from_settings(
        experiment_name=config.train.mlflow_experiment,
        run_name=run_name,
        enabled=config.train.mlflow_enabled,
    )
    active_jobs = job_push or JobMetricsPush.from_settings(
        enabled=config.train.job_metrics_enabled,
    )
    active_jobs.start()
    started = active_tracker.start()
    started_at = time.perf_counter()
    fit_seconds: float | None = None
    metrics: dict[str, float] = {}
    checkpoint_path: Path | None = None
    metrics_path = config.train.report_dir / mode / "metrics.json"
    dataset = {
        "version": manifest.metadata.dataset_version,
        "manifest_sha256": manifest.metadata.manifest_sha256,
    }
    try:
        if started:
            active_tracker.log_config(config)
            active_tracker.log_context(
                manifest.metadata.manifest_sha256,
                git_commit,
                mode,
                seed,
            )
        trainer = build_trainer(mode, config, active_tracker)
        fit_started = time.perf_counter()
        trainer.fit(
            module,
            datamodule=datamodule,
            ckpt_path=str(resume) if resume is not None else None,
        )
        fit_seconds = time.perf_counter() - fit_started
        metrics = collect_scalar_metrics(trainer.callback_metrics)
        checkpoint_path = _best_checkpoint(trainer)
        if checkpoint_path is not None:
            trainer.test(module, datamodule=datamodule, ckpt_path=str(checkpoint_path))
            metrics = {**metrics, **collect_scalar_metrics(trainer.callback_metrics)}
        duration = time.perf_counter() - started_at
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "mode": mode,
            "seed": seed,
            "resumed_from": resume.as_posix() if resume is not None else None,
            "dataset": dataset,
            "git_commit": git_commit,
            "budget": budget_payload(config),
            "precision_resolved": resolve_precision(
                config.train.precision,
                torch.cuda.is_available(),
            ),
            "parameter_count": module.parameter_count,
            "best_checkpoint": (
                checkpoint_path.as_posix() if checkpoint_path is not None else None
            ),
            "metrics": metrics,
            "duration_seconds": duration,
            "fit_seconds": fit_seconds,
        }
        _report_run(
            metrics_path,
            payload,
            active_tracker,
            active_jobs,
            JobRecord(
                run_name=run_name,
                mode=mode,
                seed=seed,
                status="finished",
                duration_seconds=duration,
                fit_seconds=fit_seconds,
                epochs_total=trainer.current_epoch,
                parameter_count=module.parameter_count,
                metrics=metrics,
            ),
        )
    except Exception:
        active_jobs.push(
            JobRecord(
                run_name=run_name,
                mode=mode,
                seed=seed,
                status="failed",
                duration_seconds=time.perf_counter() - started_at,
                fit_seconds=fit_seconds,
                epochs_total=None,
                parameter_count=module.parameter_count,
                metrics=metrics,
            )
        )
        raise
    finally:
        active_tracker.close()
    return RunResult(
        mode=mode,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        metrics=metrics,
        budget=budget_payload(config),
        parameter_count=module.parameter_count,
        dataset=dataset,
        git_commit=git_commit,
        resumed_from=resume,
    )
