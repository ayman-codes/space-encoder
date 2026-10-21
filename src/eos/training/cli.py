"""Command line entry point for training and ablation runs.

The ``eos-train`` command exposes one subcommand per ablation arm plus an
``ablation`` command that runs all three arms with identical budgets and writes
a single JSON evidence document.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from eos.config import ExperimentConfig, load_config
from eos.device import cap_vram
from eos.training.lightning_module import TRAINING_MODES, TrainingMode
from eos.training.runner import SCHEMA_VERSION, budget_payload, run_experiment, write_json

ABLATION_PATH = Path("reports/ablation/ablation.json")
"""Deliverable path for the three-arm ablation summary."""

ABLATION_MODES: tuple[TrainingMode, ...] = ("classifier", "autoencoder", "hybrid")
"""Arm order used by the ablation command."""


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        argparse.ArgumentParser: Parser with the mode, config, seed,
            fast-dev-run, max-epochs, and resume options.
    """
    parser = argparse.ArgumentParser(
        prog="eos-train",
        description="Train one ablation arm or run the complete ablation.",
    )
    parser.add_argument(
        "mode",
        choices=(*TRAINING_MODES, "ablation"),
        help="Training arm, or 'ablation' to run all three sequentially.",
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

    Args:
        config (ExperimentConfig): Normalized configuration shared by all arms.
        output_path (Path): Destination JSON path.

    Returns:
        dict[str, object]: Written payload.

    Raises:
        RuntimeError: If the arms do not report identical budget fields.
    """
    results = [run_experiment(mode, config) for mode in ABLATION_MODES]
    budgets = {json.dumps(result.budget, sort_keys=True) for result in results}
    if len(budgets) != 1:
        raise RuntimeError("ablation arms do not share identical budget fields")
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": results[0].dataset,
        "git_commit": results[0].git_commit,
        "budget": budget_payload(config),
        "arms": [
            {
                "mode": result.mode,
                "parameter_count": result.parameter_count,
                "budget": result.budget,
                "metrics": result.metrics,
                "checkpoint": (
                    result.checkpoint_path.as_posix()
                    if result.checkpoint_path is not None
                    else None
                ),
                "metrics_path": result.metrics_path.as_posix(),
            }
            for result in results
        ],
    }
    write_json(output_path, payload)
    return payload


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
    if arguments.mode == "ablation" and arguments.resume is not None:
        parser.error("--resume is only supported for a single arm")
    cap_vram()
    config = apply_overrides(
        load_config(arguments.config),
        arguments.seed,
        arguments.fast_dev_run,
        arguments.max_epochs,
    )
    if arguments.mode == "ablation":
        run_ablation(config)
        return 0
    run_experiment(cast(TrainingMode, arguments.mode), config, resume=arguments.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
