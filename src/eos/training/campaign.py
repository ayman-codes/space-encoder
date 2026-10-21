"""Resumable multi-seed campaign orchestration.

A campaign is a grid of ``(arm, seed)`` jobs executed sequentially on one
machine. Every job derives its state from its workspace on disk, so a campaign
can be interrupted at any point and re-run with the same command: completed jobs
are skipped, interrupted jobs continue from their ``last.ckpt`` with the same
tracking run, and untouched jobs start fresh. The decision lives in the
artifacts themselves rather than a state file, so a crash can never leave a
ledger disagreeing with what actually trained.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from eos.config import ExperimentConfig
from eos.training.lightning_module import TrainingMode
from eos.training.runner import (
    ABLATION_MODES,
    RunResult,
    assemble_ablation,
    assemble_ledger,
    job_workspace,
    load_run_result,
    run_experiment,
)

CAMPAIGN_PATH = Path("reports/ablation/campaign.json")
"""Deliverable path for the multi-seed campaign ledger."""

JobAction = Literal["skip", "resume", "fresh"]
"""How one campaign job must proceed given its workspace on disk."""


@dataclass(frozen=True)
class JobSpec:
    """One ``(arm, seed)`` cell of a campaign grid.

    Args:
        mode (TrainingMode): Ablation arm.
        seed (int): Data split and trainer seed.
    """

    mode: TrainingMode
    seed: int


@dataclass(frozen=True)
class JobPlan:
    """Decision for one campaign job.

    Args:
        spec (JobSpec): Job the decision applies to.
        action (JobAction): ``skip`` when the job already completed, ``resume``
            when a continuation checkpoint exists, ``fresh`` otherwise.
        resume_from (Path | None): Continuation checkpoint for ``resume``.
    """

    spec: JobSpec
    action: JobAction
    resume_from: Path | None = None


@dataclass(frozen=True)
class CampaignResult:
    """Outcome of one campaign pass.

    Args:
        plans (tuple[JobPlan, ...]): Decision taken per job, in grid order.
        results (tuple[RunResult, ...]): Completed run per job, in the same
            order.
    """

    plans: tuple[JobPlan, ...]
    results: tuple[RunResult, ...]


def with_seed(config: ExperimentConfig, seed: int) -> ExperimentConfig:
    """Return a configuration variant bound to one seed.

    Args:
        config (ExperimentConfig): Base configuration.
        seed (int): Seed applied to the data split and the trainer.

    Returns:
        ExperimentConfig: Copy with the seed applied to both sections.

    Raises:
        ValueError: If the seed is negative.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    data = config.data.model_copy(update={"seed": seed})
    train = config.train.model_copy(update={"seed": seed})
    return config.model_copy(update={"data": data, "train": train})


def plan_job(config: ExperimentConfig, spec: JobSpec, force: bool = False) -> JobPlan:
    """Decide how one job must proceed given its workspace on disk.

    Args:
        config (ExperimentConfig): Base configuration.
        spec (JobSpec): Job to plan.
        force (bool): Ignore existing artifacts and retrain the job from
            scratch.

    Returns:
        JobPlan: Action and continuation checkpoint.
    """
    workspace = job_workspace(with_seed(config, spec.seed), spec.mode)
    if force:
        return JobPlan(spec, "fresh")
    if workspace.metrics_path.is_file():
        return JobPlan(spec, "skip")
    if workspace.resume_checkpoint.is_file():
        return JobPlan(spec, "resume", resume_from=workspace.resume_checkpoint)
    return JobPlan(spec, "fresh")


def run_job(config: ExperimentConfig, plan: JobPlan) -> RunResult:
    """Execute one planned job or reload its completed result.

    Args:
        config (ExperimentConfig): Base configuration.
        plan (JobPlan): Decision produced by :func:`plan_job`.

    Returns:
        RunResult: Completed run summary for the job.

    Raises:
        FileNotFoundError: If the job must be reloaded but has no metrics JSON.
    """
    seeded = with_seed(config, plan.spec.seed)
    if plan.action == "skip":
        return load_run_result(job_workspace(seeded, plan.spec.mode).metrics_path)
    return run_experiment(plan.spec.mode, seeded, resume=plan.resume_from)


def run_campaign(  # noqa: PLR0913
    config: ExperimentConfig,
    specs: Sequence[JobSpec],
    *,
    force: bool = False,
    assemble_only: bool = False,
    ledger_path: Path = CAMPAIGN_PATH,
    ablation_path: Path | None = None,
    ablation_seed: int | None = None,
) -> CampaignResult:
    """Run a campaign grid and write its evidence documents.

    Args:
        config (ExperimentConfig): Base configuration shared by every job.
        specs (Sequence[JobSpec]): Campaign grid in execution order.
        force (bool): Retrain every job from scratch, ignoring artifacts.
        assemble_only (bool): Train nothing; reload completed jobs and write
            the evidence documents. Missing jobs fail instead of training.
        ledger_path (Path): Destination of the campaign ledger.
        ablation_path (Path | None): Destination of the three-arm ablation
            document. Omit to skip it.
        ablation_seed (int | None): Seed whose three arms form the ablation
            document. Defaults to the first seed of the grid.

    Returns:
        CampaignResult: Decisions taken and completed runs.

    Raises:
        ValueError: If the grid is empty.
        FileNotFoundError: In assemble-only mode when a job has no metrics JSON.
        RuntimeError: If the ablation arms do not report identical budgets.
    """
    if not specs:
        raise ValueError("campaign needs at least one job")
    plans = tuple(plan_job(config, spec, force=force) for spec in specs)
    results = tuple(
        (
            load_run_result(
                job_workspace(with_seed(config, plan.spec.seed), plan.spec.mode).metrics_path
            )
            if assemble_only or plan.action == "skip"
            else run_job(config, plan)
        )
        for plan in plans
    )
    assemble_ledger(results, config, ledger_path)
    if ablation_path is not None:
        seed = ablation_seed if ablation_seed is not None else specs[0].seed
        trio = [result for result in results if result.seed == seed]
        if {result.mode for result in trio} == set(ABLATION_MODES):
            assemble_ablation(trio, config, ablation_path)
    return CampaignResult(plans=plans, results=results)
