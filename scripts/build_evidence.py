"""Rebuild the assignment-notebook evidence bundle from trained checkpoints.

The notebook deliverable reports real experiment results, but graded
environments have neither the training checkpoints nor the MLflow server. This
script derives a small, tracked evidence bundle once, from the checkpoints MLflow
stores for every run: it discovers the full-budget runs, downloads their best
checkpoints into a local cache, re-evaluates each one on the curated test split,
and writes JSON and NPZ files under ``reports/ablation/evidence``.

Selection is deterministic. Runs qualify when they finished with the configured
epoch budget; the showcase seed contributes one run per ablation arm (the most
recent) for embeddings and reconstruction samples. Everything is regenerable
from committed inputs, and reruns are idempotent because downloaded checkpoints
are cached.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray

from eos.config import load_config
from eos.dataset.contract import load_contract
from eos.dataset.eurosat_datamodule import EuroSATDataModule
from eos.evaluation import (
    collect_embeddings,
    collect_recon_samples,
    evaluate_classification,
    evaluate_reconstruction,
    load_module_from_checkpoint,
)
from eos.training.lightning_module import TRAINING_MODES, TrainingMode

LOGGER = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "http://localhost:5001"
"""Tracking server used when ``MLFLOW_TRACKING_URI`` is not set."""

DEFAULT_EXPERIMENT = "eos-autoencoder"
"""MLflow experiment that owns the training runs."""

DEFAULT_CONFIG = Path("configs/base.yaml")
"""Experiment configuration supplying the evaluation datamodule."""

DEFAULT_DATASET_CONTRACT = Path("data/contracts/eurosat-rgb.contract.yaml")
"""Dataset contract supplying class names in class-id order."""

DEFAULT_OUTPUT_DIR = Path("reports/ablation/evidence")
"""Tracked destination of the evidence bundle."""

DEFAULT_CACHE_DIR = Path(".cache/evidence")
"""Gitignored checkpoint cache."""

CHECKPOINT_SUFFIX = ".ckpt"
"""File suffix of checkpoint artifacts."""

HISTORY_KEYS: tuple[str, ...] = (
    "train/loss",
    "train/ce",
    "train/recon",
    "train/mse",
    "val/loss",
    "val/macro_f1",
)
"""Per-epoch metric keys exported for the loss-trajectory figures."""

SCHEMA_VERSION = 1
"""Bundle schema version written into the manifest."""

EVIDENCE_SEED = 42
"""Seed whose runs supply showcase embeddings and reconstruction samples."""

MAX_EMBEDDINGS_PER_CLASS = 120
"""Embeddings kept per class for the latent projections."""

SAMPLES_PER_CLASS = 2
"""Reconstruction samples kept per class for the patch grids."""


@dataclass(frozen=True)
class RunRecord:
    """One full-budget training run discovered in MLflow.

    Args:
        run_id (str): MLflow run identifier.
        run_name (str): Human-readable run name.
        mode (TrainingMode): Ablation arm of the run.
        seed (int): Training seed.
        max_epochs (int): Epoch budget recorded in the run parameters.
        start_time (int): Start time in milliseconds since the epoch.
    """

    run_id: str
    run_name: str
    mode: TrainingMode
    seed: int
    max_epochs: int
    start_time: int


def discover_runs(
    tracking_uri: str,
    experiment_name: str,
    min_epochs: int,
) -> tuple[RunRecord, ...]:
    """Find the finished full-budget runs of an experiment.

    Args:
        tracking_uri (str): MLflow tracking server URI.
        experiment_name (str): Experiment name.
        min_epochs (int): Minimum recorded epoch budget for a run to qualify.

    Returns:
        tuple[RunRecord, ...]: Qualified runs ordered by seed, arm, and start
            time.

    Raises:
        ValueError: If the experiment is missing or a qualifying run lacks the
            mode, seed, or epoch parameters.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"experiment {experiment_name!r} not found on {tracking_uri}")
    experiment_id = str(experiment.experiment_id)
    found = client.search_runs(
        [experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=200,
        order_by=["attributes.start_time ASC"],
    )
    records: list[RunRecord] = []
    for run in found:
        params = {str(key): str(value) for key, value in run.data.params.items()}
        mode_value = params.get("run/mode")
        if mode_value is None or mode_value not in TRAINING_MODES:
            continue
        epochs = int(params.get("train.max_epochs", "0"))
        if epochs < min_epochs:
            continue
        seed_value = params.get("run/seed")
        if seed_value is None:
            raise ValueError(f"run {run.info.run_id!s} has mode but no seed parameter")
        records.append(
            RunRecord(
                run_id=str(run.info.run_id),
                run_name=str(run.info.run_name),
                mode=mode_value,
                seed=int(seed_value),
                max_epochs=epochs,
                start_time=int(run.info.start_time),
            )
        )
    return tuple(sorted(records, key=lambda item: (item.seed, item.mode, item.start_time)))


def checkpoint_name(client: MlflowClient, run_id: str) -> str:
    """Return the checkpoint artifact name of one run.

    Args:
        client (MlflowClient): Tracking client.
        run_id (str): MLflow run identifier.

    Returns:
        str: File name of the run's checkpoint artifact.

    Raises:
        ValueError: If the run holds no checkpoint artifact.
    """
    artifacts = [str(item.path) for item in client.list_artifacts(run_id)]
    checkpoints = sorted(
        name for name in artifacts if name.endswith(CHECKPOINT_SUFFIX) and "/" not in name
    )
    if not checkpoints:
        raise ValueError(f"run {run_id} exposes no checkpoint artifact")
    return checkpoints[-1]


def download_checkpoint(
    record: RunRecord,
    artifact_name: str,
    cache_dir: Path,
) -> Path:
    """Download one run's checkpoint into the local cache.

    Args:
        record (RunRecord): Run to fetch.
        artifact_name (str): Checkpoint artifact file name.
        cache_dir (Path): Cache root; one subdirectory per run.

    Returns:
        Path: Local checkpoint path, downloaded or already cached.

    Raises:
        FileNotFoundError: If the download produced no file.
    """
    destination = cache_dir / record.run_id
    local_path = destination / artifact_name
    if local_path.is_file():
        return local_path
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = download_artifacts(
        run_id=record.run_id,
        artifact_path=artifact_name,
        dst_path=str(destination),
    )
    result = Path(str(downloaded))
    if not result.is_file():
        raise FileNotFoundError(f"download of {artifact_name} for run {record.run_id} failed")
    return result


def fetch_history(client: MlflowClient, record: RunRecord) -> dict[str, list[float]]:
    """Export per-epoch metric histories for one run.

    Args:
        client (MlflowClient): Tracking client.
        record (RunRecord): Run to read.

    Returns:
        dict[str, list[float]]: Metric key to epoch-ordered values, holding only
            the keys the run actually recorded.
    """
    history: dict[str, list[float]] = {}
    for key in HISTORY_KEYS:
        points = client.get_metric_history(record.run_id, key)
        if points:
            history[key] = [round(float(point.value), 6) for point in points]
    return history


def evaluate_run(
    record: RunRecord,
    checkpoint_path: Path,
    datamodule: EuroSATDataModule,
    class_names: tuple[str, ...],
    device: str | None,
) -> dict[str, object]:
    """Re-evaluate one run on the test split.

    Args:
        record (RunRecord): Run metadata.
        checkpoint_path (Path): Local checkpoint file.
        datamodule (EuroSATDataModule): Prepared datamodule providing the test
            split.
        class_names (tuple[str, ...]): Class names in class-id order.
        device (str | None): Inference device, or None for automatic selection.

    Returns:
        dict[str, object]: JSON-ready run payload with provenance and metrics.

    Raises:
        ValueError: If the checkpoint arm does not match the run record.
    """
    module = load_module_from_checkpoint(checkpoint_path, expected_mode=record.mode)
    loader = datamodule.test_dataloader()
    classification: dict[str, object] | None = None
    reconstruction: dict[str, object] | None = None
    if record.mode in ("hybrid", "classifier"):
        classification = evaluate_classification(module, loader, device=device).to_dict(class_names)
    if record.mode in ("hybrid", "autoencoder"):
        reconstruction = evaluate_reconstruction(module, loader, device=device).to_dict()
    return {
        "run_id": record.run_id,
        "run_name": record.run_name,
        "mode": record.mode,
        "seed": record.seed,
        "max_epochs": record.max_epochs,
        "checkpoint": checkpoint_path.name,
        "classification": classification,
        "reconstruction": reconstruction,
    }


def select_showcase(
    records: Sequence[RunRecord],
    seed: int,
) -> tuple[RunRecord, ...]:
    """Pick one run per ablation arm for the showcase artifacts.

    The most recent run of each arm at the requested seed wins, which selects
    the final campaign rather than earlier replicates.

    Args:
        records (Sequence[RunRecord]): All qualified runs.
        seed (int): Showcase seed.

    Returns:
        tuple[RunRecord, ...]: At most one run per arm, ordered by arm name.
    """
    chosen: dict[str, RunRecord] = {}
    for record in sorted(records, key=lambda item: (item.start_time, item.run_id)):
        if record.seed == seed:
            chosen[record.mode] = record
    return tuple(chosen[mode] for mode in sorted(chosen))


def collect_showcase_arrays(  # noqa: PLR0913, PLR0917
    record: RunRecord,
    checkpoint_path: Path,
    datamodule: EuroSATDataModule,
    device: str | None,
    max_per_class: int,
    samples_per_class: int,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int64],
    NDArray[np.uint8],
    NDArray[np.uint8],
    NDArray[np.int64],
]:
    """Collect embeddings and reconstruction samples for one showcase run.

    Args:
        record (RunRecord): Run metadata.
        checkpoint_path (Path): Local checkpoint file.
        datamodule (EuroSATDataModule): Prepared datamodule.
        device (str | None): Inference device, or None for automatic selection.
        max_per_class (int): Embeddings kept per class.
        samples_per_class (int): Reconstruction samples kept per class.

    Returns:
        tuple: Embeddings, embedding labels, originals, reconstructions, and
            sample labels; arrays are empty when the arm lacks the head.

    Raises:
        ValueError: If the checkpoint arm does not match the run record.
    """
    module = load_module_from_checkpoint(checkpoint_path, expected_mode=record.mode)
    loader = datamodule.test_dataloader()
    empty_images = np.zeros((0, 1, 1, 3), dtype=np.uint8)
    empty_labels = np.zeros((0,), dtype=np.int64)
    embeddings = np.zeros((0, 1), dtype=np.float32)
    embedding_labels = empty_labels
    originals = empty_images
    reconstructions = empty_images
    sample_labels = empty_labels
    if record.mode in ("hybrid", "classifier"):
        batch = collect_embeddings(module, loader, max_per_class=max_per_class, device=device)
        embeddings = batch.embeddings
        embedding_labels = batch.labels
    if record.mode in ("hybrid", "autoencoder"):
        samples = collect_recon_samples(module, loader, per_class=samples_per_class, device=device)
        originals = samples.originals
        reconstructions = samples.reconstructions
        sample_labels = samples.labels
    return embeddings, embedding_labels, originals, reconstructions, sample_labels


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface.

    Returns:
        argparse.ArgumentParser: Parser with all evidence options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-contract", type=Path, default=DEFAULT_DATASET_CONTRACT)
    parser.add_argument(
        "--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--showcase-seed", type=int, default=EVIDENCE_SEED)
    parser.add_argument("--max-embeddings-per-class", type=int, default=MAX_EMBEDDINGS_PER_CLASS)
    parser.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build and write the evidence bundle.

    Args:
        argv (Sequence[str] | None): Command-line arguments, or None to read
            the process arguments.

    Returns:
        int: Zero on success.

    Raises:
        ValueError: If discovery or evaluation inputs are inconsistent.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    contract = load_contract(args.dataset_contract)
    class_names = contract.source.class_names
    datamodule = EuroSATDataModule(config.data)
    datamodule.setup()
    manifest = datamodule.manifest
    if manifest is None:
        raise ValueError("the datamodule produced no manifest")
    records = discover_runs(args.tracking_uri, args.experiment, args.min_epochs)
    LOGGER.info("discovered %d full-budget runs", len(records))
    if not records:
        raise ValueError("no runs qualified; check the tracking URI and epoch filter")
    client = MlflowClient(tracking_uri=args.tracking_uri)
    run_payloads: list[dict[str, object]] = []
    checkpoints: dict[str, Path] = {}
    for record in records:
        artifact = checkpoint_name(client, record.run_id)
        local = download_checkpoint(record, artifact, args.cache_dir)
        checkpoints[record.run_id] = local
        payload = evaluate_run(record, local, datamodule, class_names, args.device)
        payload["history"] = fetch_history(client, record)
        run_payloads.append(payload)
        LOGGER.info("evaluated %s (%s seed %d)", record.run_name, record.mode, record.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    showcase = select_showcase(records, args.showcase_seed)
    embedding_names: list[str] = []
    for record in showcase:
        embeddings, embedding_labels, originals, reconstructions, sample_labels = (
            collect_showcase_arrays(
                record,
                checkpoints[record.run_id],
                datamodule,
                args.device,
                args.max_embeddings_per_class,
                args.samples_per_class,
            )
        )
        if embeddings.shape[0] > 0:
            np.savez_compressed(
                args.output_dir / f"embeddings_{record.mode}.npz",
                embeddings=embeddings,
                labels=embedding_labels,
                class_names=np.asarray(class_names),
            )
            embedding_names.append(f"embeddings_{record.mode}.npz")
        if originals.shape[0] > 0:
            np.savez_compressed(
                args.output_dir / f"recon_samples_{record.mode}.npz",
                originals=originals,
                reconstructions=reconstructions,
                labels=sample_labels,
                class_names=np.asarray(class_names),
            )
            embedding_names.append(f"recon_samples_{record.mode}.npz")
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "experiment": args.experiment,
        "tracking_uri": args.tracking_uri,
        "selection": {
            "min_epochs": args.min_epochs,
            "showcase_seed": args.showcase_seed,
            "max_embeddings_per_class": args.max_embeddings_per_class,
            "samples_per_class": args.samples_per_class,
        },
        "dataset": {
            "version": manifest.metadata.dataset_version,
            "manifest_sha256": manifest.metadata.manifest_sha256,
            "curated_root": str(config.data.curated_root),
        },
        "class_names": list(class_names),
        "artifacts": embedding_names,
        "runs": run_payloads,
    }
    manifest_path = args.output_dir / "run_metrics.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("evidence bundle written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
