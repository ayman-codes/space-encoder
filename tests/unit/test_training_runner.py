"""Tests for experiment orchestration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eos.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from eos.training import runner as runner_module
from eos.training.runner import (
    RunResult,
    _best_checkpoint,
    assemble_ablation,
    budget_payload,
    build_trainer,
    job_workspace,
    load_run_result,
    normalize_fast_dev_run,
    run_experiment,
    validate_consistency,
    write_json,
)


def _config(tmp_path: Path | None = None) -> ExperimentConfig:
    """Build a CPU experiment configuration for orchestration tests.

    Args:
        tmp_path (Path | None): Optional checkpoint and report root.

    Returns:
        ExperimentConfig: Validated configuration.
    """
    root = tmp_path or Path(".cache/tmp/orchestration-test")
    train = TrainConfig(
        max_epochs=2,
        warmup_epochs=0,
        precision="bf16-mixed",
        accelerator="cpu",
        devices=1,
        seed=7,
        checkpoint_dir=root / "checkpoints",
        report_dir=root / "reports",
        mlflow_enabled=False,
        job_metrics_enabled=False,
    )
    return ExperimentConfig(train=train)


def test_fast_dev_run_normalizes_epochs_and_warmup() -> None:
    """A fast run must fit in one epoch without a warmup stage."""
    config = _config().model_copy(
        update={"train": _config().train.model_copy(update={"fast_dev_run": True})}
    )
    normalized = normalize_fast_dev_run(config)
    assert normalized.train.max_epochs == 1
    assert normalized.train.warmup_epochs == 0
    assert normalized.train.fast_dev_run is True


def test_normal_configuration_is_returned_unchanged() -> None:
    """The normalizer must not alter a full run."""
    config = _config()
    assert normalize_fast_dev_run(config) is config


def test_consistency_rejects_mismatched_sections() -> None:
    """Class, size, and channel mismatches must fail before training."""
    base = _config()
    mismatched_classes = base.model_copy(
        update={"data": DataConfig(num_classes=4), "model": ModelConfig(num_classes=10)}
    )
    with pytest.raises(ValueError, match="num_classes"):
        validate_consistency(mismatched_classes)
    mismatched_size = base.model_copy(
        update={"data": DataConfig(image_size=32), "model": ModelConfig(input_size=64)}
    )
    with pytest.raises(ValueError, match="image_size"):
        validate_consistency(mismatched_size)
    mismatched_channels = base.model_copy(
        update={"data": DataConfig(channels=4), "model": ModelConfig(in_channels=3)}
    )
    with pytest.raises(ValueError, match="channels"):
        validate_consistency(mismatched_channels)
    validate_consistency(base)


def test_budget_payload_contains_the_shared_fields() -> None:
    """Every field needed to prove arm parity must be present."""
    payload = budget_payload(_config())
    assert payload["max_epochs"] == 2
    assert payload["batch_size"] == 128
    assert payload["learning_rate"] == pytest.approx(1e-3)
    assert payload["warmup_epochs"] == 0
    assert payload["seed"] == 7
    assert payload["precision"] == "bf16-mixed"
    assert payload["class_weights"] is False


def test_write_json_replaces_atomically_and_leaves_no_temporary(tmp_path: Path) -> None:
    """The writer must create parent directories and clean its temp sibling."""
    destination = tmp_path / "nested" / "payload.json"
    write_json(destination, {"b": 2, "a": 1})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert list(destination.parent.iterdir()) == [destination]


def test_build_trainer_resolves_cpu_precision_and_limits(tmp_path: Path) -> None:
    """A bf16 request on CPU must resolve to float32 with one-batch limits."""
    config = _config(tmp_path).model_copy(
        update={"train": _config(tmp_path).train.model_copy(update={"fast_dev_run": True})}
    )
    trainer = build_trainer("hybrid", config)
    assert trainer.precision == "32-true"
    assert trainer.gradient_clip_val == pytest.approx(1.0)
    assert trainer.limit_train_batches == 1
    assert trainer.limit_val_batches == 1
    assert len(trainer.callbacks) >= 2
    assert _best_checkpoint(trainer) is None


def test_build_trainer_uses_autoencoder_monitor(tmp_path: Path) -> None:
    """The autoencoder arm must monitor reconstruction error."""
    trainer = build_trainer("autoencoder", _config(tmp_path))
    checkpoint = trainer.checkpoint_callback
    assert checkpoint is not None
    assert checkpoint.monitor == "val/recon_mse_smooth"
    assert checkpoint.mode == "min"


def test_best_checkpoint_ignores_non_model_checkpoint_callbacks() -> None:
    """A trainer without a ModelCheckpoint must report no checkpoint."""
    trainer = SimpleNamespace(checkpoint_callback=None)
    assert _best_checkpoint(trainer) is None  # type: ignore[arg-type]


def test_run_experiment_rejects_unknown_mode() -> None:
    """An unsupported arm must fail before reading the dataset."""
    with pytest.raises(ValueError, match="unsupported"):
        run_experiment("segmentation", _config())  # type: ignore[arg-type]


def test_run_experiment_requires_an_existing_resume_checkpoint(tmp_path: Path) -> None:
    """A missing resume checkpoint must fail before data setup."""
    with pytest.raises(FileNotFoundError, match="resume"):
        run_experiment("hybrid", _config(tmp_path), resume=tmp_path / "missing.ckpt")


def test_run_experiment_requires_a_loaded_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A data module that silently skips setup must fail loudly."""

    class FakeDataModule:
        """Data module double with no manifest."""

        def __init__(self, config: DataConfig) -> None:
            """Store the configuration and expose an empty manifest.

            Args:
                config (DataConfig): Data configuration.

            Returns:
                None: The double is initialized in place.
            """
            del config
            self.manifest = None

        def setup(self) -> None:
            """Do nothing.

            Returns:
                None: Setup is intentionally skipped.
            """

    monkeypatch.setattr(runner_module, "EuroSATDataModule", FakeDataModule)
    with pytest.raises(RuntimeError, match="manifest"):
        run_experiment("hybrid", _config())


def test_job_workspace_isolates_arms_and_seeds(tmp_path: Path) -> None:
    """Two seeds of one arm must never share artifact locations."""
    base = _config(tmp_path)
    first = job_workspace(base, "hybrid")
    other = job_workspace(
        base.model_copy(
            update={"train": base.train.model_copy(update={"seed": 9})},
        ),
        "hybrid",
    )
    assert first.checkpoint_dir == tmp_path / "checkpoints" / "hybrid-seed7"
    assert first.report_dir == tmp_path / "reports" / "hybrid" / "seed7"
    assert first.metrics_path == tmp_path / "reports" / "hybrid" / "seed7" / "metrics.json"
    assert first.resume_checkpoint == first.checkpoint_dir / "last.ckpt"
    assert first.run_id_path == first.checkpoint_dir / "mlflow_run_id.txt"
    assert other.checkpoint_dir != first.checkpoint_dir
    assert other.metrics_path != first.metrics_path
    assert job_workspace(base, "classifier").checkpoint_dir != first.checkpoint_dir


def test_budget_payload_records_the_convergence_regime(tmp_path: Path) -> None:
    """The parity record must cover the schedule and the stopping rule."""
    payload = budget_payload(_config(tmp_path))
    assert payload["max_epochs"] == 2
    assert payload["lr_scheduler"] == "cosine"
    assert payload["plateau_patience"] == 5
    assert payload["early_stopping_patience"] == 12
    assert payload["early_stopping_min_delta"] == pytest.approx(1e-3)
    assert payload["monitor_smoothing_window"] == 3


def test_load_run_result_rebuilds_a_completed_run(tmp_path: Path) -> None:
    """A completed job must be reloadable from its metrics JSON."""
    payload: dict[str, object] = {
        "schema_version": 1,
        "mode": "hybrid",
        "seed": 7,
        "resumed_from": None,
        "dataset": {"version": "v1", "manifest_sha256": "f" * 64},
        "git_commit": "abc123",
        "budget": {"max_epochs": 2},
        "parameter_count": 123,
        "best_checkpoint": (tmp_path / "hybrid-01.ckpt").as_posix(),
        "metrics": {"val/macro_f1": 0.5},
        "epochs_consumed": 2,
        "stopped_early": True,
    }
    path = tmp_path / "metrics.json"
    write_json(path, payload)
    result = load_run_result(path)
    assert result.mode == "hybrid"
    assert result.seed == 7
    assert result.checkpoint_path == tmp_path / "hybrid-01.ckpt"
    assert result.metrics_path == path
    assert result.metrics["val/macro_f1"] == pytest.approx(0.5)
    assert result.budget == {"max_epochs": 2}
    assert result.parameter_count == 123
    assert result.epochs_consumed == 2
    assert result.stopped_early is True
    assert result.resumed_from is None
    with pytest.raises(FileNotFoundError, match="metrics"):
        load_run_result(tmp_path / "absent.json")


def test_load_run_result_rejects_a_foreign_document(tmp_path: Path) -> None:
    """A JSON document without a training mode must not be trusted."""
    path = tmp_path / "metrics.json"
    write_json(path, {"mode": "seg", "seed": 1})
    with pytest.raises(ValueError, match="mode"):
        load_run_result(path)


def _result(
    mode: str, config: ExperimentConfig, budget: dict[str, object] | None = None
) -> RunResult:
    """Build a completed run summary for assembly tests.

    Args:
        mode (str): Ablation arm.
        config (ExperimentConfig): Configuration the run shares.
        budget (dict[str, object] | None): Optional budget override.

    Returns:
        RunResult: Synthetic completed run.
    """
    return RunResult(
        mode=mode,  # type: ignore[arg-type]
        seed=config.train.seed,
        checkpoint_path=None,
        metrics_path=Path("reports") / mode / "metrics.json",
        metrics={"val/loss": 0.5},
        budget=budget if budget is not None else budget_payload(config),
        parameter_count=123,
        dataset={"version": "v1", "manifest_sha256": "f" * 64},
        git_commit="abc123",
        epochs_consumed=2,
        stopped_early=True,
    )


def test_assemble_ablation_writes_three_arms_with_one_budget(tmp_path: Path) -> None:
    """The document must record every arm with its consumed epochs."""
    config = _config(tmp_path)
    results = [_result(mode, config) for mode in ("classifier", "autoencoder", "hybrid")]
    output = tmp_path / "ablation.json"
    payload = assemble_ablation(results, config, output)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert [arm["mode"] for arm in payload["arms"]] == ["classifier", "autoencoder", "hybrid"]
    assert all(arm["budget"] == payload["budget"] for arm in payload["arms"])
    assert [arm["epochs_consumed"] for arm in payload["arms"]] == [2, 2, 2]
    assert all(arm["stopped_early"] is True for arm in payload["arms"])


def test_assemble_ablation_rejects_divergent_budgets(tmp_path: Path) -> None:
    """A budget that differs between arms must fail the parity assertion."""
    config = _config(tmp_path)
    results = [_result(mode, config) for mode in ("classifier", "autoencoder", "hybrid")]
    divergent = [
        results[0],
        results[1],
        _result("hybrid", config, budget={**budget_payload(config), "seed": 999}),
    ]
    with pytest.raises(RuntimeError, match="identical budget"):
        assemble_ablation(divergent, config, tmp_path / "ablation.json")


def test_assemble_ablation_rejects_a_missing_arm(tmp_path: Path) -> None:
    """A two-arm campaign must not masquerade as an ablation."""
    config = _config(tmp_path)
    results = [_result(mode, config) for mode in ("classifier", "hybrid")]
    with pytest.raises(RuntimeError, match="missing arms"):
        assemble_ablation(results, config, tmp_path / "ablation.json")


def test_resume_run_id_is_only_adopted_when_resuming(tmp_path: Path) -> None:
    """A fresh run must not adopt a stored tracking run identifier."""
    config = _config(tmp_path)
    workspace = job_workspace(config, "hybrid")
    workspace.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    workspace.run_id_path.write_text("prior-run", encoding="utf-8")
    assert runner_module._resume_run_id(workspace, None) is None
    assert runner_module._resume_run_id(workspace, workspace.resume_checkpoint) == "prior-run"


def test_job_workspace_separates_experiment_variants(tmp_path: Path) -> None:
    """A variant tag must keep two experiments from sharing one workspace."""
    base = _config(tmp_path)
    tagged = base.model_copy(
        update={"train": base.train.model_copy(update={"run_tag": "lambda03"})}
    )
    plain = job_workspace(base, "hybrid")
    variant = job_workspace(tagged, "hybrid")
    assert plain.checkpoint_dir == tmp_path / "checkpoints" / "hybrid-seed7"
    assert variant.checkpoint_dir == tmp_path / "checkpoints" / "hybrid-seed7-lambda03"
    assert variant.metrics_path == (
        tmp_path / "reports" / "hybrid" / "seed7-lambda03" / "metrics.json"
    )
    assert variant.resume_checkpoint != plain.resume_checkpoint
    assert budget_payload(tagged)["run_tag"] == "lambda03"
