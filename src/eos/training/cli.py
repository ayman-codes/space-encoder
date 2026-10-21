"""Command line entry point for training and ablation runs.

The ``eos-train`` command exposes one subcommand per ablation arm, an
``ablation`` command that runs all three arms with identical budgets and writes
a single JSON evidence document, and a ``campaign`` command that runs an arm by
seed grid. Both aggregate modes are resumable: completed jobs are reloaded from
their metrics JSON, interrupted jobs continue from their ``last.ckpt``, and only
untouched jobs train from scratch.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from eos.config import ExperimentConfig, load_config
from eos.device import cap_vram
from eos.training.campaign import CAMPAIGN_PATH, JobSpec, run_campaign
from eos.training.lightning_module import TRAINING_MODES, TrainingMode
from eos.training.runner import (
    ABLATION_MODES,
    RunResult,
    assemble_ablation,
    job_workspace,
    load_run_result,
    run_experiment,
)

ABLATION_PATH = Path("reports/ablation/ablation.json")
"""Deliverable path for the three-arm ablation summary."""


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        argparse.ArgumentParser: Parser with the mode, config, seed, budget,
            resume, and campaign-grid options.
    """
    parser = argparse.ArgumentParser(
        prog="eos-train",
        description=(
            "Train one ablation arm, the three-arm ablation, or a resumable arm-by-seed campaign."
        ),
    )
    parser.add_argument(
        "mode",
        choices=(*TRAINING_MODES, "ablation", "campaign"),
        help=(
            "Training arm, 'ablation' to run all three sequentially, or "
            "'campaign' to run an arm-by-seed grid."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Experiment YAML path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the data and trainer seed.",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run one batch per stage while keeping checkpointing and logging.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override the epoch budget for timing or capped runs.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume a single arm from a checkpoint such as checkpoints/last.ckpt.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seed grid for the campaign mode; defaults to the configured seed.",
    )
    parser.add_argument(
        "--arms",
        choices=TRAINING_MODES,
        nargs="+",
        default=None,
        help="Arms for the campaign mode; defaults to all three.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Campaign mode: retrain every job from scratch.",
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Campaign mode: write evidence documents from completed jobs without training.",
    )
    parser.add_argument(
        "--ledger-out",
        type=Path,
        default=CAMPAIGN_PATH,
        help="Destination of the campaign job ledger.",
    )
    parser.add_argument(
        "--ablation-out",
        type=Path,
        default=ABLATION_PATH,
        help="Destination of the three-arm ablation document.",
    )
    parser.add_argument(
        "--ablation-seed",
        type=int,
        default=None,
        help=(
            "Seed whose three arms form the ablation document; defaults to the first campaign seed."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv (Sequence[str] | None): Optional argument vector for tests.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    return build_parser().parse_args(argv)


def apply_overrides(
    config: ExperimentConfig,
    seed: int | None,
    fast_dev_run: bool,
    max_epochs: int | None = None,
) -> ExperimentConfig:
    """Apply CLI overrides to a validated configuration.

    Args:
        config (ExperimentConfig): Loaded configuration.
        seed (int | None): Optional seed override.
        fast_dev_run (bool): Whether to request a one-batch run.
        max_epochs (int | None): Optional epoch-budget override.

    Returns:
        ExperimentConfig: Configuration with overrides applied.

    Raises:
        ValueError: If the seed or epoch budget is negative or zero.
    """
    data = config.data
    train = config.train
    if seed is not None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        data = data.model_copy(update={"seed": seed})
        train = train.model_copy(update={"seed": seed})
    if max_epochs is not None:
        if max_epochs < 1:
            raise ValueError("max_epochs must be at least one")
        train = train.model_copy(update={"max_epochs": max_epochs})
        if train.warmup_epochs >= max_epochs:
            train = train.model_copy(update={"warmup_epochs": max(0, max_epochs - 1)})
    if fast_dev_run:
        train = train.model_copy(update={"fast_dev_run": True, "max_epochs": 1, "warmup_epochs": 0})
    return config.model_copy(update={"data": data, "train": train})


def run_ablation(
    config: ExperimentConfig,
    output_path: Path = ABLATION_PATH,
) -> dict[str, object]:
    """Run all three arms and write the ablation JSON.

    The aggregate is resumable: an arm that already wrote its metrics JSON is
    reloaded instead of retrained, and an interrupted arm continues from its
    ``last.ckpt``.

    Args:
        config (ExperimentConfig): Normalized configuration shared by all arms.
        output_path (Path): Destination JSON path.

    Returns:
        dict[str, object]: Written payload.

    Raises:
        RuntimeError: If the arms do not report identical budget fields.
    """
    results: list[RunResult] = []
    for mode in ABLATION_MODES:
        workspace = job_workspace(config, mode)
        if workspace.metrics_path.is_file():
            results.append(load_run_result(workspace.metrics_path))
            continue
        resume = workspace.resume_checkpoint if workspace.resume_checkpoint.is_file() else None
        results.append(run_experiment(mode, config, resume=resume))
    return assemble_ablation(results, config, output_path)


def build_campaign_grid(
    seeds: Sequence[int] | None,
    arms: Sequence[TrainingMode] | None,
    default_seed: int,
) -> list[JobSpec]:
    """Build the campaign grid in execution order.

    Args:
        seeds (Sequence[int] | None): Seed grid; ``None`` selects the default
            seed.
        arms (Sequence[TrainingMode] | None): Arms to run; ``None`` selects all
            three.
        default_seed (int): Seed used when no grid is given.

    Returns:
        list[JobSpec]: Jobs ordered seed-major, then by arm.
    """
    chosen_seeds = list(seeds) if seeds is not None else [default_seed]
    chosen_arms = tuple(arms) if arms is not None else TRAINING_MODES
    return [JobSpec(mode, seed) for seed in chosen_seeds for mode in chosen_arms]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv (Sequence[str] | None): Optional argument vector for tests.

    Returns:
        int: Process exit code.

    Raises:
        ValueError: If the configuration or override is invalid.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.mode in ("ablation", "campaign") and arguments.resume is not None:
        parser.error(
            "--resume is only supported for a single arm; ablation and campaign "
            "jobs resume automatically from their workspaces"
        )
    cap_vram()
    config = apply_overrides(
        load_config(arguments.config),
        arguments.seed,
        arguments.fast_dev_run,
        arguments.max_epochs,
    )
    if arguments.mode == "campaign":
        grid = build_campaign_grid(arguments.seeds, arguments.arms, config.train.seed)
        run_campaign(
            config,
            grid,
            force=arguments.force,
            assemble_only=arguments.assemble_only,
            ledger_path=arguments.ledger_out,
            ablation_path=arguments.ablation_out,
            ablation_seed=(
                arguments.ablation_seed if arguments.ablation_seed is not None else grid[0].seed
            ),
        )
        return 0
    if arguments.mode == "ablation":
        run_ablation(config, arguments.ablation_out)
        return 0
    run_experiment(cast(TrainingMode, arguments.mode), config, resume=arguments.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
