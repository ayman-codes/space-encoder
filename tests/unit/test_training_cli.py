"""Tests for the ``eos-train`` command line and ablation writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eos.config import ExperimentConfig, TrainConfig
from eos.training import cli
from eos.training.runner import RunResult, budget_payload, job_workspace, write_json


def _config(root: Path | None = None) -> ExperimentConfig:
    """Return a CPU configuration with observability sinks disabled.

    Args:
        root (Path | None): Optional artifact root isolating workspaces.

    Returns:
        ExperimentConfig: Validated configuration.
    """
    base = Path(".cache/tmp/cli-test") if root is None else root
    return ExperimentConfig(
        train=TrainConfig(
            max_epochs=4,
            warmup_epochs=1,
            accelerator="cpu",
            precision="32-true",
            checkpoint_dir=base / "checkpoints",
            report_dir=base / "reports",
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
        seed=config.train.seed,
        checkpoint_path=None,
        metrics_path=job_workspace(config, mode).metrics_path,  # type: ignore[arg-type]
        metrics={"val/loss": 0.5},
        budget=budget_payload(config),
        parameter_count=123,
        dataset={"version": "synthetic-v1.0.0", "manifest_sha256": "f" * 64},
        git_commit="abc123",
        epochs_consumed=3,
        stopped_early=False,
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


def test_parse_args_reads_the_campaign_grid() -> None:
    """The campaign grid must be expressible on the command line."""
    arguments = cli.parse_args(
        [
            "campaign",
            "--seeds",
            "42",
            "7",
            "--arms",
            "classifier",
            "hybrid",
            "--assemble-only",
            "--ablation-seed",
            "42",
            "--ledger-out",
            "reports/ablation/variant.json",
        ]
    )
    assert arguments.mode == "campaign"
    assert arguments.seeds == [42, 7]
    assert arguments.arms == ["classifier", "hybrid"]
    assert arguments.assemble_only is True
    assert arguments.ablation_seed == 42
    assert arguments.ledger_out == Path("reports/ablation/variant.json")
    assert arguments.force is False


def test_build_campaign_grid_orders_seeds_before_arms() -> None:
    """The grid must be seed-major so each seed completes all arms together."""
    grid = cli.build_campaign_grid([42, 7], ["classifier", "hybrid"], default_seed=1)
    assert [(spec.seed, spec.mode) for spec in grid] == [
        (42, "classifier"),
        (42, "hybrid"),
        (7, "classifier"),
        (7, "hybrid"),
    ]
    fallback = cli.build_campaign_grid(None, None, default_seed=3)
    assert [spec.seed for spec in fallback] == [3, 3, 3]
    assert {spec.mode for spec in fallback} == {"classifier", "autoencoder", "hybrid"}


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


def _fake_run_factory(
    captured: list[str] | None = None,
) -> object:
    """Build a run double that records the arms it was asked to train.

    Args:
        captured (list[str] | None): Optional sink for the trained arms.

    Returns:
        object: ``run_experiment`` stand-in returning synthetic results.
    """

    def _fake_run(
        mode: str,
        experiment: ExperimentConfig,
        tracker: object = None,
        resume: Path | None = None,
    ) -> RunResult:
        """Return a synthetic result and persist its metrics document.

        Args:
            mode (str): Ablation arm.
            experiment (ExperimentConfig): Configuration.
            tracker (object): Ignored tracker.
            resume (Path | None): Ignored resume checkpoint.

        Returns:
            RunResult: Synthetic result.
        """
        del tracker, resume
        if captured is not None:
            captured.append(mode)
        result = _result(mode, experiment)
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


def test_run_ablation_writes_three_arms_with_one_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ablation document must contain three arms and identical budgets."""
    config = _config(tmp_path)
    monkeypatch.setattr(cli, "run_experiment", _fake_run_factory())
    output = tmp_path / "ablation.json"
    payload = cli.run_ablation(config, output)
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    arms = payload["arms"]
    assert [arm["mode"] for arm in arms] == ["classifier", "autoencoder", "hybrid"]
    budgets = {json.dumps(arm["budget"], sort_keys=True) for arm in arms}
    assert len(budgets) == 1
    assert all(arm["budget"] == payload["budget"] for arm in arms)
    assert [arm["epochs_consumed"] for arm in arms] == [3, 3, 3]


def test_run_ablation_reloads_arms_that_already_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-run must skip completed arms instead of retraining them."""
    config = _config(tmp_path)
    trained: list[str] = []
    monkeypatch.setattr(cli, "run_experiment", _fake_run_factory(trained))
    cli.run_ablation(config, tmp_path / "first.json")
    assert trained == ["classifier", "autoencoder", "hybrid"]
    trained.clear()
    monkeypatch.setattr(cli, "run_experiment", _fake_run_factory(trained))
    payload = cli.run_ablation(config, tmp_path / "second.json")
    assert trained == []
    assert [arm["mode"] for arm in payload["arms"]] == ["classifier", "autoencoder", "hybrid"]


def test_run_ablation_resumes_an_interrupted_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An arm with a continuation checkpoint must resume instead of restarting."""
    config = _config(tmp_path)
    workspace = job_workspace(config, "classifier")
    workspace.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    workspace.resume_checkpoint.write_bytes(b"ckpt")
    resumed: list[Path | None] = []
    captured: list[str] = []

    def _fake_run(
        mode: str,
        experiment: ExperimentConfig,
        tracker: object = None,
        resume: Path | None = None,
    ) -> RunResult:
        """Record the resume decision and return a synthetic result.

        Args:
            mode (str): Ablation arm.
            experiment (ExperimentConfig): Configuration.
            tracker (object): Ignored tracker.
            resume (Path | None): Continuation checkpoint.

        Returns:
            RunResult: Synthetic result.
        """
        del tracker
        captured.append(mode)
        resumed.append(resume)
        return _result(mode, experiment)

    monkeypatch.setattr(cli, "run_experiment", _fake_run)
    cli.run_ablation(config, tmp_path / "ablation.json")
    assert captured == ["classifier", "autoencoder", "hybrid"]
    assert resumed[0] == workspace.resume_checkpoint
    assert resumed[1] is None


def test_run_ablation_rejects_divergent_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bug that changes the budget between arms must be detected."""
    config = _config(tmp_path)

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
            seed=result.seed,
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


def test_main_rejects_resume_with_campaign() -> None:
    """Campaign jobs resume automatically, so an explicit path is rejected."""
    with pytest.raises(SystemExit):
        cli.main(["campaign", "--resume", "last.ckpt"])


def test_main_dispatches_ablation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ablation mode must call the aggregate runner."""
    called: list[tuple[ExperimentConfig, Path]] = []

    def _fake_ablation(
        config: ExperimentConfig,
        output_path: Path = cli.ABLATION_PATH,
    ) -> dict[str, object]:
        """Record the aggregate invocation.

        Args:
            config (ExperimentConfig): Configuration.
            output_path (Path): Destination.

        Returns:
            dict[str, object]: Empty payload.
        """
        called.append((config, output_path))
        return {}

    monkeypatch.setattr(cli, "load_config", lambda path: _config())
    monkeypatch.setattr(cli, "run_ablation", _fake_ablation)
    assert cli.main(["ablation"]) == 0
    assert len(called) == 1
    assert called[0][1] == cli.ABLATION_PATH


def test_main_dispatches_campaign(monkeypatch: pytest.MonkeyPatch) -> None:
    """The campaign mode must build the grid and forward the resumable runner."""
    captured: dict[str, object] = {}

    def _fake_campaign(  # noqa: PLR0913
        config: ExperimentConfig,
        specs: list[object],
        *,
        force: bool = False,
        assemble_only: bool = False,
        ledger_path: Path,
        ablation_path: Path | None = None,
        ablation_seed: int | None = None,
    ) -> object:
        """Record the campaign invocation.

        Args:
            config (ExperimentConfig): Configuration.
            specs (list[object]): Campaign grid.
            force (bool): Whether to retrain from scratch.
            assemble_only (bool): Whether training is skipped.
            ledger_path (Path): Ledger destination.
            ablation_path (Path | None): Ablation destination.
            ablation_seed (int | None): Seed of the ablation document.

        Returns:
            object: ``None``; the call is recorded.
        """
        captured["specs"] = [(spec.seed, spec.mode) for spec in specs]  # type: ignore[attr-defined]
        captured["force"] = force
        captured["assemble_only"] = assemble_only
        captured["ablation_path"] = ablation_path
        captured["ablation_seed"] = ablation_seed
        captured["ledger_path"] = ledger_path
        del config
        return None

    monkeypatch.setattr(cli, "load_config", lambda path: _config())
    monkeypatch.setattr(cli, "run_campaign", _fake_campaign)
    assert (
        cli.main(["campaign", "--seeds", "42", "7", "--force", "--ledger-out", "ledger.json"]) == 0
    )
    assert captured["specs"] == [
        (42, "hybrid"),
        (42, "classifier"),
        (42, "autoencoder"),
        (7, "hybrid"),
        (7, "classifier"),
        (7, "autoencoder"),
    ]
    assert captured["force"] is True
    assert captured["assemble_only"] is False
    assert captured["ablation_path"] == cli.ABLATION_PATH
    assert captured["ablation_seed"] == 42
    assert captured["ledger_path"] == Path("ledger.json")
