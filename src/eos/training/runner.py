"""Experiment orchestration for the three ablation arms.

The runner owns the parts that must stay identical across arms: seeding, data
setup, trainer construction, metric collection, and artifact writing. The arm
difference is confined to the objective in :class:`EOSTrainingModule`.

Every invocation receives an isolated job workspace keyed by arm and seed, so
multi-seed campaigns never overwrite each other's checkpoints and every run has
one unambiguous continuation point (``last.ckpt`` plus the tracking run
identifier) after an interruption.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from eos.config import ExperimentConfig
from eos.dataset.eurosat_datamodule import EuroSATDataModule
from eos.training.callbacks import build_callbacks
from eos.training.job_metrics import JobMetricsPush, JobRecord
from eos.training.lightning_module import (
    EOSTrainingModule,
    TrainingMode,
    monitor_name,
    stopping_min_delta,
)
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

ABLATION_MODES: tuple[TrainingMode, ...] = ("classifier", "autoencoder", "hybrid")
"""Arm order used by the ablation document."""

RUN_ID_FILENAME = "mlflow_run_id.txt"
"""File inside a job workspace holding the tracking run identifier to resume."""


@dataclass(frozen=True)
class JobWorkspace:
    """Isolated artifact locations for one (arm, seed) training job.

    Args:
        checkpoint_dir (Path): Directory receiving the job's ``.ckpt`` files,
            its ``last.ckpt`` continuation point, and the tracking run
            identifier.
        report_dir (Path): Directory receiving the job's metrics JSON and
            Lightning CSV logs.
        metrics_path (Path): Per-job metrics JSON path; its presence marks the
            job as complete.
    """

    checkpoint_dir: Path
    report_dir: Path
    metrics_path: Path

    @property
    def run_id_path(self) -> Path:
        """Return the path holding the tracking run identifier.

        Returns:
            Path: Checkpoint-directory file used to continue one MLflow run
            across process restarts.
        """
        return self.checkpoint_dir / RUN_ID_FILENAME

    @property
    def resume_checkpoint(self) -> Path:
        """Return the continuation checkpoint for an interrupted job.

        Returns:
            Path: ``last.ckpt`` inside the job's checkpoint directory.
        """
        return self.checkpoint_dir / "last.ckpt"


def job_workspace(config: ExperimentConfig, mode: TrainingMode) -> JobWorkspace:
    """Derive the isolated artifact locations for one job.

    Args:
        config (ExperimentConfig): Normalized configuration; its seed and run
            tag select the workspace.
        mode (TrainingMode): Ablation arm.

    Returns:
        JobWorkspace: Checkpoint and report locations keyed by arm, seed, and
        variant tag.
    """
    seed = config.train.seed
    tag = config.train.run_tag
    name = f"{mode}-seed{seed}" if not tag else f"{mode}-seed{seed}-{tag}"
    report = f"seed{seed}" if not tag else f"seed{seed}-{tag}"
    checkpoint_dir = config.train.checkpoint_dir / name
    report_dir = config.train.report_dir / mode / report
    return JobWorkspace(
        checkpoint_dir=checkpoint_dir,
        report_dir=report_dir,
        metrics_path=report_dir / "metrics.json",
    )


@dataclass(frozen=True)
class RunResult:
    """Summary of one completed training arm.

    Args:
        mode (TrainingMode): Ablation arm.
        seed (int): Data and trainer seed of the run.
        checkpoint_path (Path | None): Best checkpoint, when one was written.
        metrics_path (Path): Per-run metrics JSON path.
        metrics (dict[str, float]): Validation and test metrics.
        budget (dict[str, object]): Identical budget fields for all arms.
        parameter_count (int): Trainable parameter count.
        dataset (dict[str, str]): Dataset version and manifest fingerprint.
        git_commit (str): Repository commit or ``"unknown"``.
        resumed_from (Path | None): Checkpoint used to resume, when applicable.
        epochs_consumed (int | None): Epochs actually trained. Early stopping
            makes this smaller than the declared budget.
        stopped_early (bool): Whether early stopping ended the run before the
            declared epoch ceiling.
    """

    mode: TrainingMode
    seed: int
    checkpoint_path: Path | None
    metrics_path: Path
    metrics: dict[str, float]
    budget: dict[str, object]
    parameter_count: int
    dataset: dict[str, str]
    git_commit: str
    resumed_from: Path | None = None
    epochs_consumed: int | None = None
    stopped_early: bool = False


@dataclass(frozen=True)
class FitOutcome:
    """Evidence collected from one fitted trainer.

    Args:
        checkpoint_path (Path | None): Best checkpoint, when one was written.
        best_score (float | None): Monitored value of the best checkpoint.
        metrics (dict[str, float]): Validation metrics merged with the test
            metrics of the best checkpoint.
        fit_seconds (float): Wall time spent inside the fit loop.
        epochs_consumed (int): Epochs actually trained.
        stopped_early (bool): Whether early stopping ended the run before the
            declared epoch ceiling.
    """

    checkpoint_path: Path | None
    best_score: float | None
    metrics: dict[str, float]
    fit_seconds: float
    epochs_consumed: int
    stopped_early: bool


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

    The payload covers the optimisation and stopping regime as well as the
    epoch budget, so the arm-parity assertion also proves that every arm trained
    under the same convergence rule.

    Args:
        config (ExperimentConfig): Normalized configuration.

    Returns:
        dict[str, object]: Epochs, batch size, optimizer, schedule, stopping
        rule, precision, and seed.
    """
    return {
        "max_epochs": config.train.max_epochs,
        "batch_size": config.data.batch_size,
        "learning_rate": config.train.learning_rate,
        "weight_decay": config.train.weight_decay,
        "warmup_epochs": config.train.warmup_epochs,
        "lr_scheduler": config.train.lr_scheduler,
        "plateau_factor": config.train.plateau_factor,
        "plateau_patience": config.train.plateau_patience,
        "plateau_min_lr": config.train.plateau_min_lr,
        "early_stopping_patience": config.train.early_stopping_patience,
        "early_stopping_min_delta": config.train.early_stopping_min_delta,
        "early_stopping_min_delta_recon": config.train.early_stopping_min_delta_recon,
        "monitor_smoothing_window": config.train.monitor_smoothing_window,
        "run_tag": config.train.run_tag,
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
        Trainer: Configured trainer with checkpointing and metric logging. All
        artifacts land in the job workspace derived from the arm and seed.
    """
    workspace = job_workspace(config, mode)
    precision = resolve_precision(config.train.precision, torch.cuda.is_available())
    callbacks = build_callbacks(mode, workspace.checkpoint_dir, config.train)
    if tracker is not None:
        callbacks.append(MLflowTrackingCallback(tracker))
    fast = config.train.fast_dev_run
    limit_train_batches: int | None = 1 if fast else None
    limit_val_batches: int | None = 1 if fast else None
    limit_test_batches: int | None = 1 if fast else None
    logger = CSVLogger(
        save_dir=workspace.report_dir / "lightning_logs",
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
        default_root_dir=workspace.checkpoint_dir,
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


def _fit_and_evaluate(
    trainer: Trainer,
    module: EOSTrainingModule,
    datamodule: EuroSATDataModule,
    resume: Path | None,
) -> FitOutcome:
    """Fit one arm, then re-evaluate its best checkpoint on the test split.

    Args:
        trainer (Trainer): Configured trainer with the arm's callbacks.
        module (EOSTrainingModule): Training module for the arm.
        datamodule (EuroSATDataModule): Loaded data module.
        resume (Path | None): Continuation checkpoint, when resuming.

    Returns:
        FitOutcome: Checkpoint, metrics, timing, and stopping evidence.
    """
    fit_started = time.perf_counter()
    trainer.fit(
        module,
        datamodule=datamodule,
        ckpt_path=str(resume) if resume is not None else None,
    )
    fit_seconds = time.perf_counter() - fit_started
    metrics = collect_scalar_metrics(trainer.callback_metrics)
    checkpoint_path = _best_checkpoint(trainer)
    best_score = _best_score(trainer)
    if checkpoint_path is not None:
        trainer.test(module, datamodule=datamodule, ckpt_path=str(checkpoint_path))
        metrics = {**metrics, **collect_scalar_metrics(trainer.callback_metrics)}
    epochs_consumed = int(trainer.current_epoch)
    max_epochs = trainer.max_epochs
    stopped_early = bool(
        trainer.should_stop and (max_epochs is None or epochs_consumed < max_epochs)
    )
    return FitOutcome(
        checkpoint_path=checkpoint_path,
        best_score=best_score,
        metrics=metrics,
        fit_seconds=fit_seconds,
        epochs_consumed=epochs_consumed,
        stopped_early=stopped_early,
    )


@dataclass(frozen=True)
class RunContext:
    """Everything a single-arm run needs after setup and before the fit loop.

    Args:
        datamodule (EuroSATDataModule): Data module with loaded manifests.
        module (EOSTrainingModule): Training module for the arm.
        tracker (MLflowTracker): Guarded tracker for this run.
        jobs (JobMetricsPush): Guarded job reporter.
        run_name (str): Human-readable run name.
        git_commit (str): Repository commit or ``"unknown"``.
        dataset (dict[str, str]): Dataset version and manifest fingerprint.
        workspace (JobWorkspace): Isolated artifact locations.
        started (bool): Whether tracking became active.
        started_at (float): Performance counter at run start.
    """

    datamodule: EuroSATDataModule
    module: EOSTrainingModule
    tracker: MLflowTracker
    jobs: JobMetricsPush
    run_name: str
    git_commit: str
    dataset: dict[str, str]
    workspace: JobWorkspace
    started: bool
    started_at: float


def _start_run(
    mode: TrainingMode,
    config: ExperimentConfig,
    tracker: MLflowTracker | None,
    job_push: JobMetricsPush | None,
    resume: Path | None,
) -> RunContext:
    """Prepare data, model, tracking, and reporting for one training job.

    The tracking run identifier written beside the checkpoints is what lets a
    later attempt continue the same run instead of splitting the metric history.

    Args:
        mode (TrainingMode): Ablation arm.
        config (ExperimentConfig): Normalized configuration.
        tracker (MLflowTracker | None): Optional caller-provided tracker.
        job_push (JobMetricsPush | None): Optional caller-provided job reporter.
        resume (Path | None): Continuation checkpoint, when resuming.

    Returns:
        RunContext: Prepared run state.

    Raises:
        RuntimeError: If the data module does not expose a loaded manifest.
    """
    seed = config.train.seed
    seed_everything(seed, deterministic=True)
    workspace = job_workspace(config, mode)
    datamodule = EuroSATDataModule(config.data)
    datamodule.setup()
    manifest = datamodule.manifest
    if manifest is None:
        raise RuntimeError("data module did not produce a manifest")
    class_weights = datamodule.class_weights if config.train.class_weights else None
    module = EOSTrainingModule(mode, config.model, config.train, class_weights=class_weights)
    git_commit = resolve_git_commit()
    tag = config.train.run_tag
    label = f"{mode}-seed{seed}" if not tag else f"{mode}-seed{seed}-{tag}"
    run_name = f"{label}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    active_tracker = tracker or MLflowTracker.from_settings(
        experiment_name=config.train.mlflow_experiment,
        run_name=run_name,
        enabled=config.train.mlflow_enabled,
        run_id=_resume_run_id(workspace, resume),
    )
    active_jobs = job_push or JobMetricsPush.from_settings(
        enabled=config.train.job_metrics_enabled,
    )
    active_jobs.start()
    started = active_tracker.start()
    if active_tracker.run_id is not None:
        workspace.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        workspace.run_id_path.write_text(active_tracker.run_id, encoding="utf-8")
    return RunContext(
        datamodule=datamodule,
        module=module,
        tracker=active_tracker,
        jobs=active_jobs,
        run_name=run_name,
        git_commit=git_commit,
        dataset={
            "version": manifest.metadata.dataset_version,
            "manifest_sha256": manifest.metadata.manifest_sha256,
        },
        workspace=workspace,
        started=started,
        started_at=time.perf_counter(),
    )


def run_experiment(
    mode: TrainingMode,
    config: ExperimentConfig,
    tracker: MLflowTracker | None = None,
    resume: Path | None = None,
    job_push: JobMetricsPush | None = None,
) -> RunResult:
    """Run one training arm end to end inside its isolated job workspace.

    The workspace is derived from the arm and seed, so checkpoints, the metrics
    JSON, and the tracking run identifier never collide between jobs. When
    ``resume`` points into that workspace, the stored tracking run identifier is
    reused so an interrupted job keeps one continuous per-epoch metric history.

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
    context = _start_run(mode, config, tracker, job_push, resume)
    seed = config.train.seed
    fit_seconds: float | None = None
    metrics: dict[str, float] = {}
    checkpoint_path: Path | None = None
    epochs_consumed: int | None = None
    stopped_early = False
    best_score: float | None = None
    try:
        if context.started:
            context.tracker.log_config(config)
            context.tracker.log_context(
                context.dataset["manifest_sha256"],
                context.git_commit,
                mode,
                seed,
            )
        trainer = build_trainer(mode, config, context.tracker)
        outcome = _fit_and_evaluate(trainer, context.module, context.datamodule, resume)
        metrics = outcome.metrics
        checkpoint_path = outcome.checkpoint_path
        best_score = outcome.best_score
        fit_seconds = outcome.fit_seconds
        epochs_consumed = outcome.epochs_consumed
        stopped_early = outcome.stopped_early
        duration = time.perf_counter() - context.started_at
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "mode": mode,
            "seed": seed,
            "resumed_from": resume.as_posix() if resume is not None else None,
            "mlflow_run_id": context.tracker.run_id,
            "dataset": context.dataset,
            "git_commit": context.git_commit,
            "budget": budget_payload(config),
            "monitor": monitor_name(mode, config.train),
            "stopping_min_delta": stopping_min_delta(mode, config.train),
            "epochs_consumed": epochs_consumed,
            "stopped_early": stopped_early,
            "best_model_score": best_score,
            "precision_resolved": resolve_precision(
                config.train.precision,
                torch.cuda.is_available(),
            ),
            "parameter_count": context.module.parameter_count,
            "best_checkpoint": (
                checkpoint_path.as_posix() if checkpoint_path is not None else None
            ),
            "metrics": metrics,
            "duration_seconds": duration,
            "fit_seconds": fit_seconds,
        }
        _report_run(
            context.workspace.metrics_path,
            payload,
            context.tracker,
            context.jobs,
            JobRecord(
                run_name=context.run_name,
                mode=mode,
                seed=seed,
                status="finished",
                duration_seconds=duration,
                fit_seconds=fit_seconds,
                epochs_total=epochs_consumed,
                parameter_count=context.module.parameter_count,
                metrics=metrics,
            ),
        )
    except Exception:
        context.jobs.push(
            JobRecord(
                run_name=context.run_name,
                mode=mode,
                seed=seed,
                status="failed",
                duration_seconds=time.perf_counter() - context.started_at,
                fit_seconds=fit_seconds,
                epochs_total=None,
                parameter_count=context.module.parameter_count,
                metrics=metrics,
            )
        )
        raise
    finally:
        context.tracker.close()
    return RunResult(
        mode=mode,
        seed=seed,
        checkpoint_path=checkpoint_path,
        metrics_path=context.workspace.metrics_path,
        metrics=metrics,
        budget=budget_payload(config),
        parameter_count=context.module.parameter_count,
        dataset=context.dataset,
        git_commit=context.git_commit,
        resumed_from=resume,
        epochs_consumed=epochs_consumed,
        stopped_early=stopped_early,
    )


def _best_score(trainer: Trainer) -> float | None:
    """Return the monitored score of the best checkpoint, when one exists.

    Args:
        trainer (Trainer): Fitted trainer.

    Returns:
        float | None: Best monitored value, or ``None`` when no checkpoint was
        selected.
    """
    callback = trainer.checkpoint_callback
    if not isinstance(callback, ModelCheckpoint):
        return None
    score = callback.best_model_score
    return None if score is None else float(score)


def _resume_run_id(workspace: JobWorkspace, resume: Path | None) -> str | None:
    """Return the tracking run identifier to continue for a resumed job.

    Args:
        workspace (JobWorkspace): Job artifact locations.
        resume (Path | None): Requested resume checkpoint.

    Returns:
        str | None: Stored run identifier when resuming, otherwise ``None``, so
        a resumed job keeps one continuous metric history on the tracker.
    """
    if resume is None or not workspace.run_id_path.is_file():
        return None
    return workspace.run_id_path.read_text(encoding="utf-8").strip() or None


def load_run_result(metrics_path: Path) -> RunResult:
    """Rebuild a run summary from its persisted metrics JSON.

    Completed jobs are reloaded instead of retrained, which is what makes a
    re-run campaign idempotent across process restarts.

    Args:
        metrics_path (Path): Per-job metrics JSON written by
            :func:`run_experiment`.

    Returns:
        RunResult: Summary of the completed run.

    Raises:
        FileNotFoundError: If the metrics JSON does not exist.
        ValueError: If the document does not describe a training run.
    """
    if not metrics_path.is_file():
        raise FileNotFoundError(f"run metrics not found: {metrics_path}")
    payload = cast(dict[str, object], json.loads(metrics_path.read_text(encoding="utf-8")))
    mode = payload.get("mode")
    if mode not in ("hybrid", "classifier", "autoencoder"):
        raise ValueError(f"metrics document at {metrics_path} has no valid mode")
    checkpoint = payload.get("best_checkpoint")
    resumed = payload.get("resumed_from")
    consumed = payload.get("epochs_consumed")
    return RunResult(
        mode=mode,
        seed=cast(int, payload.get("seed", 0)),
        checkpoint_path=Path(str(checkpoint)) if checkpoint is not None else None,
        metrics_path=metrics_path,
        metrics=cast(dict[str, float], payload.get("metrics", {})),
        budget=cast(dict[str, object], payload.get("budget", {})),
        parameter_count=cast(int, payload.get("parameter_count", 0)),
        dataset=cast(dict[str, str], payload.get("dataset", {})),
        git_commit=str(payload.get("git_commit", "unknown")),
        resumed_from=Path(str(resumed)) if resumed is not None else None,
        epochs_consumed=cast(int, consumed) if consumed is not None else None,
        stopped_early=cast(bool, payload.get("stopped_early", False)),
    )


def assemble_ablation(
    results: Sequence[RunResult],
    config: ExperimentConfig,
    output_path: Path,
) -> dict[str, object]:
    """Write the three-arm ablation document from completed run summaries.

    Args:
        results (Sequence[RunResult]): One completed run per arm, in any order.
        config (ExperimentConfig): Normalized configuration shared by all arms.
        output_path (Path): Destination JSON path.

    Returns:
        dict[str, object]: Written payload.

    Raises:
        RuntimeError: If an arm is missing or the arms do not report identical
            budget fields.
    """
    by_mode = {result.mode: result for result in results}
    missing = [mode for mode in ABLATION_MODES if mode not in by_mode]
    if missing:
        raise RuntimeError(f"ablation is missing arms: {missing}")
    budgets = {json.dumps(by_mode[mode].budget, sort_keys=True) for mode in ABLATION_MODES}
    if len(budgets) != 1:
        raise RuntimeError("ablation arms do not share identical budget fields")
    first = by_mode[ABLATION_MODES[0]]
    arms: list[dict[str, object]] = []
    for mode in ABLATION_MODES:
        result = by_mode[mode]
        checkpoint = result.checkpoint_path
        arms.append(
            {
                "mode": mode,
                "parameter_count": result.parameter_count,
                "budget": result.budget,
                "metrics": result.metrics,
                "checkpoint": checkpoint.as_posix() if checkpoint is not None else None,
                "metrics_path": result.metrics_path.as_posix(),
                "epochs_consumed": result.epochs_consumed,
                "stopped_early": result.stopped_early,
            }
        )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": first.dataset,
        "git_commit": first.git_commit,
        "budget": budget_payload(config),
        "arms": arms,
    }
    write_json(output_path, payload)
    return payload


def arm_payload(result: RunResult) -> dict[str, object]:
    """Serialize one completed run for a campaign ledger.

    Args:
        result (RunResult): Completed run summary.

    Returns:
        dict[str, object]: Ledger entry with provenance, budget, and metrics.
    """
    return {
        "mode": result.mode,
        "seed": result.seed,
        "budget": result.budget,
        "parameter_count": result.parameter_count,
        "dataset": result.dataset,
        "git_commit": result.git_commit,
        "metrics": result.metrics,
        "checkpoint": (
            result.checkpoint_path.as_posix() if result.checkpoint_path is not None else None
        ),
        "metrics_path": result.metrics_path.as_posix(),
        "resumed_from": (
            result.resumed_from.as_posix() if result.resumed_from is not None else None
        ),
        "epochs_consumed": result.epochs_consumed,
        "stopped_early": result.stopped_early,
    }


def assemble_ledger(
    results: Sequence[RunResult],
    config: ExperimentConfig,
    output_path: Path,
) -> dict[str, object]:
    """Assemble a campaign ledger from completed run summaries.

    The ledger records one entry per job together with the shared budget. Parity
    is asserted over the training regime with the per-job seed removed, because
    the seed is exactly what varies across a campaign grid.

    Args:
        results (Sequence[RunResult]): Completed run summaries.
        config (ExperimentConfig): Normalized configuration shared by all jobs.
        output_path (Path): Destination JSON path.

    Returns:
        dict[str, object]: Written payload.
    """
    regimes = [
        {key: value for key, value in result.budget.items() if key != "seed"} for result in results
    ]
    budgets = {json.dumps(regime, sort_keys=True) for regime in regimes}
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": results[0].dataset if results else {},
        "git_commit": results[0].git_commit if results else "unknown",
        "budget": budget_payload(config),
        "budget_parity": len(budgets) == 1,
        "jobs": [arm_payload(result) for result in results],
    }
    write_json(output_path, payload)
    return payload
