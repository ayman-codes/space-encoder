"""Tests for resumable campaign planning and execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eos.config import ExperimentConfig, TrainConfig
from eos.training import campaign as campaign_module
from eos.training.campaign import JobSpec, plan_job, run_campaign, with_seed
from eos.training.runner import RunResult, budget_payload, job_workspace, write_json


def _config(root: Path) -> ExperimentConfig:
    """Build an isolated CPU configuration for campaign tests.

    Args:
        root (Path): Artifact root isolating every job workspace.

    Returns:
        ExperimentConfig: Validated configuration.
    """
    return ExperimentConfig(
        train=TrainConfig(
            max_epochs=4,
            warmup_epochs=1,
            accelerator="cpu",
            precision="32-true",
            checkpoint_dir=root / "checkpoints",
            report_dir=root / "reports",
            mlflow_enabled=False,
            job_metrics_enabled=False,
        )
    )


def _result(spec: JobSpec, config: ExperimentConfig) -> RunResult:
    """Build a synthetic completed run for one job.

    Args:
        spec (JobSpec): Campaign job.
        config (ExperimentConfig): Seeded configuration of the job.

    Returns:
        RunResult: Synthetic completed run.
    """
    return RunResult(
        mode=spec.mode,
        seed=spec.seed,
        checkpoint_path=None,
        metrics_path=job_workspace(config, spec.mode).metrics_path,
        metrics={"val/loss": 0.5},
        budget=budget_payload(config),
        parameter_count=123,
        dataset={"version": "synthetic-v1.0.0", "manifest_sha256": "f" * 64},
        git_commit="abc123",
        epochs_consumed=3,
        stopped_early=False,
    )


def _fake_run_factory(trained: list[tuple[str, int]]) -> object:
    """Build a run double that persists its metrics document.

    Args:
        trained (list[tuple[str, int]]): Sink recording the trained jobs.

    Returns:
        object: ``run_experiment`` stand-in.
    """

    def _fake_run(
        mode: str,
        config: ExperimentConfig,
        tracker: object = None,
        resume: Path | None = None,
        job_push: object = None,
    ) -> RunResult:
        """Persist a synthetic run for one job.

        Args:
            mode (str): Ablation arm.
            config (ExperimentConfig): Seeded configuration.
            tracker (object): Ignored tracker.
            resume (Path | None): Ignored continuation checkpoint.
            job_push (object): Ignored job reporter.

        Returns:
            RunResult: Synthetic completed run.
        """
        del tracker, job_push, resume
        spec = JobSpec(mode, config.train.seed)  # type: ignore[arg-type]
        trained.append((mode, config.train.seed))
        result = _result(spec, config)
        write_json(
            result.metrics_path,
            {
                "mode": result.mode,
                "seed": result.seed,
                "budget": result.budget,
                "metrics": result.metrics,
                "parameter_count": result.parameter_count,
                "dataset": result.dataset,
                "git_commit": result.git_commit,
                "best_checkpoint": None,
                "resumed_from": None,
                "epochs_consumed": result.epochs_consumed,
                "stopped_early": result.stopped_early,
            },
        )
        return result

    return _fake_run


def test_with_seed_binds_both_sections(tmp_path: Path) -> None:
    """The seed must reach the data split and the trainer together."""
    config = with_seed(_config(tmp_path), 9)
    assert config.data.seed == 9
    assert config.train.seed == 9
    with pytest.raises(ValueError, match="non-negative"):
        with_seed(_config(tmp_path), -1)


def test_plan_job_starts_fresh_without_artifacts(tmp_path: Path) -> None:
    """An untouched job must train from scratch."""
    plan = plan_job(_config(tmp_path), JobSpec("hybrid", 42))
    assert plan.action == "fresh"
    assert plan.resume_from is None


def test_plan_job_resumes_from_the_continuation_checkpoint(tmp_path: Path) -> None:
    """An interrupted job must continue from its last checkpoint."""
    config = _config(tmp_path)
    workspace = job_workspace(config, "hybrid")
    workspace.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    workspace.resume_checkpoint.write_bytes(b"ckpt")
    plan = plan_job(config, JobSpec("hybrid", 42))
    assert plan.action == "resume"
    assert plan.resume_from == workspace.resume_checkpoint


def test_plan_job_skips_completed_jobs(tmp_path: Path) -> None:
    """A job with a metrics document must not train again."""
    config = _config(tmp_path)
    workspace = job_workspace(config, "hybrid")
    workspace.report_dir.mkdir(parents=True, exist_ok=True)
    write_json(workspace.metrics_path, {"mode": "hybrid", "seed": 42})
    plan = plan_job(config, JobSpec("hybrid", 42))
    assert plan.action == "skip"


def test_plan_job_retrains_only_when_forced(tmp_path: Path) -> None:
    """Force must override the completed marker deliberately."""
    config = _config(tmp_path)
    workspace = job_workspace(config, "hybrid")
    workspace.report_dir.mkdir(parents=True, exist_ok=True)
    write_json(workspace.metrics_path, {"mode": "hybrid", "seed": 42})
    plan = plan_job(config, JobSpec("hybrid", 42), force=True)
    assert plan.action == "fresh"
    assert plan.resume_from is None


def test_run_campaign_is_idempotent_across_restarts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second pass must reload completed jobs instead of retraining them."""
    config = _config(tmp_path)
    specs = [JobSpec(mode, seed) for seed in (42, 7) for mode in ("classifier", "hybrid")]
    trained: list[tuple[str, int]] = []
    monkeypatch.setattr(campaign_module, "run_experiment", _fake_run_factory(trained))
    first = run_campaign(config, specs, ledger_path=tmp_path / "ledger.json")
    assert trained == [
        ("classifier", 42),
        ("hybrid", 42),
        ("classifier", 7),
        ("hybrid", 7),
    ]
    trained.clear()
    second = run_campaign(config, specs, ledger_path=tmp_path / "ledger.json")
    assert trained == []
    assert [plan.action for plan in first.plans] == ["fresh"] * 4
    assert [plan.action for plan in second.plans] == ["skip"] * 4
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["budget_parity"] is True
    assert [job["seed"] for job in ledger["jobs"]] == [42, 42, 7, 7]
    assert all(job["epochs_consumed"] == 3 for job in ledger["jobs"])


def test_run_campaign_writes_the_ablation_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seed of record must produce the three-arm document."""
    config = _config(tmp_path)
    specs = [JobSpec(mode, 42) for mode in ("classifier", "autoencoder", "hybrid")]
    monkeypatch.setattr(campaign_module, "run_experiment", _fake_run_factory([]))
    run_campaign(
        config,
        specs,
        ledger_path=tmp_path / "ledger.json",
        ablation_path=tmp_path / "ablation.json",
    )
    payload = json.loads((tmp_path / "ablation.json").read_text(encoding="utf-8"))
    assert [arm["mode"] for arm in payload["arms"]] == ["classifier", "autoencoder", "hybrid"]


def test_run_campaign_skips_the_ablation_without_the_full_trio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A two-arm campaign must not write a three-arm document."""
    config = _config(tmp_path)
    specs = [JobSpec(mode, 42) for mode in ("classifier", "hybrid")]
    monkeypatch.setattr(campaign_module, "run_experiment", _fake_run_factory([]))
    run_campaign(
        config,
        specs,
        ledger_path=tmp_path / "ledger.json",
        ablation_path=tmp_path / "ablation.json",
    )
    assert not (tmp_path / "ablation.json").exists()


def test_assemble_only_never_trains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The assemble-only pass must reload artifacts and fail on gaps."""
    config = _config(tmp_path)
    specs = [JobSpec(mode, 42) for mode in ("classifier", "hybrid")]
    monkeypatch.setattr(campaign_module, "run_experiment", _fake_run_factory([]))
    run_campaign(config, specs, ledger_path=tmp_path / "ledger.json")
    trained: list[tuple[str, int]] = []
    monkeypatch.setattr(campaign_module, "run_experiment", _fake_run_factory(trained))
    run_campaign(config, specs, ledger_path=tmp_path / "ledger.json", assemble_only=True)
    assert trained == []
    with pytest.raises(FileNotFoundError, match="metrics"):
        run_campaign(
            config,
            [JobSpec("autoencoder", 42)],
            ledger_path=tmp_path / "other.json",
            assemble_only=True,
        )


def test_run_campaign_rejects_an_empty_grid(tmp_path: Path) -> None:
    """An empty campaign must fail instead of writing an empty ledger."""
    with pytest.raises(ValueError, match="at least one"):
        run_campaign(_config(tmp_path), [], ledger_path=tmp_path / "ledger.json")
