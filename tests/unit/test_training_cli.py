"""Tests for the ``eos-train`` command line and ablation writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eos.config import ExperimentConfig, TrainConfig
from eos.training import cli
from eos.training.runner import RunResult, budget_payload


def _config() -> ExperimentConfig:
    """Return a CPU configuration with observability sinks disabled.

    Returns:
        ExperimentConfig: Validated configuration.
    """
    return ExperimentConfig(
        train=TrainConfig(
            max_epochs=4,
            warmup_epochs=1,
            accelerator="cpu",
            precision="32-true",
            mlflow_enabled=False,
            job_metrics_enabled=False,
        )
    )


def _result(mode: str, config: ExperimentConfig) -> RunResult:
    """Build a fake completed arm for writer tests.

    Args:
        mode (str): Ablation arm.
        config (ExperimentConfig): Configuration that produced the arm.

    Returns:
        RunResult: Synthetic result.
    """
    return RunResult(
        mode=mode,  # type: ignore[arg-type]
        checkpoint_path=None,
        metrics_path=Path("reports/training") / mode / "metrics.json",
        metrics={"val/loss": 0.5},
        budget=budget_payload(config),
        parameter_count=123,
        dataset={"version": "synthetic-v1.0.0", "manifest_sha256": "f" * 64},
        git_commit="abc123",
    )


def test_parse_args_reads_mode_and_overrides() -> None:
    """The parser must expose mode, config, seed, budget, and resume options."""
    arguments = cli.parse_args(
        ["hybrid", "--seed", "7", "--fast-dev-run", "--max-epochs", "3", "--resume", "last.ckpt"]
    )
    assert arguments.mode == "hybrid"
    assert arguments.seed == 7
    assert arguments.fast_dev_run is True
    assert arguments.max_epochs == 3
    assert arguments.resume == Path("last.ckpt")
    assert arguments.config == Path("configs/base.yaml")


def test_apply_overrides_updates_both_seeds_and_fast_budget() -> None:
    """Seed and fast-dev-run overrides must reach data and trainer sections."""
    config = cli.apply_overrides(_config(), 9, True)
    assert config.data.seed == 9
    assert config.train.seed == 9
    assert config.train.max_epochs == 1
    assert config.train.warmup_epochs == 0
    assert config.train.fast_dev_run is True


def test_apply_overrides_sets_a_capped_epoch_budget() -> None:
    """A max-epochs override must clamp warmup when the budget is too short."""
    shortened = cli.apply_overrides(_config(), None, False, max_epochs=1)
    assert shortened.train.max_epochs == 1
    assert shortened.train.warmup_epochs == 0
    extended = cli.apply_overrides(_config(), None, False, max_epochs=10)
    assert extended.train.max_epochs == 10
    assert extended.train.warmup_epochs == 1
    with pytest.raises(ValueError, match="at least one"):
        cli.apply_overrides(_config(), None, False, max_epochs=0)


def test_apply_overrides_rejects_negative_seed() -> None:
    """A negative seed must fail before any training starts."""
    with pytest.raises(ValueError, match="non-negative"):
        cli.apply_overrides(_config(), -1, False)


def test_run_ablation_writes_three_arms_with_one_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ablation document must contain three arms and identical budgets."""
    config = _config()

    def _fake_run(
        mode: str,
        experiment: ExperimentConfig,
        tracker: object = None,
        resume: Path | None = None,
    ) -> RunResult:
        """Return a synthetic result for one arm.

        Args:
            mode (str): Ablation arm.
            experiment (ExperimentConfig): Configuration.
            tracker (object): Ignored tracker.
            resume (Path | None): Ignored resume checkpoint.

        Returns:
            RunResult: Synthetic result.
        """
        del tracker, resume
        return _result(mode, experiment)

    monkeypatch.setattr(cli, "run_experiment", _fake_run)
    output = tmp_path / "ablation.json"
    payload = cli.run_ablation(config, output)
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    arms = payload["arms"]
    assert [arm["mode"] for arm in arms] == ["classifier", "autoencoder", "hybrid"]
    budgets = {json.dumps(arm["budget"], sort_keys=True) for arm in arms}
    assert len(budgets) == 1
    assert all(arm["budget"] == payload["budget"] for arm in arms)


def test_run_ablation_rejects_divergent_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bug that changes the budget between arms must be detected."""
    config = _config()

    def _fake_run(
        mode: str,
        experiment: ExperimentConfig,
        tracker: object = None,
        resume: Path | None = None,
    ) -> RunResult:
        """Return a result whose budget depends on the arm.

        Args:
            mode (str): Ablation arm.
            experiment (ExperimentConfig): Configuration.
            tracker (object): Ignored tracker.
            resume (Path | None): Ignored resume checkpoint.

        Returns:
            RunResult: Result with a mutated budget.
        """
        del tracker, resume
        result = _result(mode, experiment)
        budget = dict(result.budget)
        budget["seed"] = 1 if mode == "classifier" else 2
        return RunResult(
            mode=result.mode,
            checkpoint_path=result.checkpoint_path,
            metrics_path=result.metrics_path,
            metrics=result.metrics,
            budget=budget,
            parameter_count=result.parameter_count,
            dataset=result.dataset,
            git_commit=result.git_commit,
        )

    monkeypatch.setattr(cli, "run_experiment", _fake_run)
    with pytest.raises(RuntimeError, match="identical budget"):
        cli.run_ablation(config, tmp_path / "ablation.json")


def test_main_dispatches_a_single_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-arm invocation must forward the overridden configuration."""
    captured: dict[str, object] = {}

    def _fake_run(
        mode: str,
        config: ExperimentConfig,
        tracker: object = None,
        resume: Path | None = None,
    ) -> RunResult:
        """Record the forwarded call.

        Args:
            mode (str): Ablation arm.
            config (ExperimentConfig): Configuration.
            tracker (object): Ignored tracker.
            resume (Path | None): Resume checkpoint.

        Returns:
            RunResult: Synthetic result.
        """
        del tracker
        captured["mode"] = mode
        captured["seed"] = config.train.seed
        captured["epochs"] = config.train.max_epochs
        captured["resume"] = resume
        return _result(mode, config)

    monkeypatch.setattr(cli, "load_config", lambda path: _config())
    monkeypatch.setattr(cli, "run_experiment", _fake_run)
    assert cli.main(["hybrid", "--seed", "5", "--fast-dev-run", "--resume", "last.ckpt"]) == 0
    assert captured == {
        "mode": "hybrid",
        "seed": 5,
        "epochs": 1,
        "resume": Path("last.ckpt"),
    }


def test_main_rejects_resume_with_ablation() -> None:
    """Resuming a sequential ablation is ambiguous and must be rejected."""
    with pytest.raises(SystemExit):
        cli.main(["ablation", "--resume", "last.ckpt"])


def test_main_dispatches_ablation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ablation mode must call the aggregate runner."""
    called: list[ExperimentConfig] = []

    def _fake_ablation(config: ExperimentConfig, output_path: Path = cli.ABLATION_PATH) -> None:
        """Record the aggregate invocation.

        Args:
            config (ExperimentConfig): Configuration.
            output_path (Path): Ignored destination.

        Returns:
            None: The call is recorded.
        """
        del output_path
        called.append(config)

    monkeypatch.setattr(cli, "load_config", lambda path: _config())
    monkeypatch.setattr(cli, "run_ablation", _fake_ablation)
    assert cli.main(["ablation"]) == 0
    assert len(called) == 1
