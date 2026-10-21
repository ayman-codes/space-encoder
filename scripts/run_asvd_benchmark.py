"""Benchmark ASVD compression of trained checkpoints on the test split.

For every checkpoint the script evaluates the untouched model and each ASVD
scope on the held-out test split in float32, measures forward latency on the
resolved device, and reports dense-set and whole-model parameter reduction
separately. The milestone contract is encoded directly in the artifact: at
least ``M4_DENSE_REDUCTION`` of the dense parameters removed while macro-F1
drops by less than ``M4_F1_LOSS``. Factorization invalidates the latent
representation, so every scope is measured from a freshly reloaded checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

import torch
from torch import Tensor, nn

from eos.config import ASVDConfig, load_config
from eos.dataset.contract import load_contract
from eos.dataset.eurosat_datamodule import EuroSATDataModule
from eos.device import cap_vram
from eos.evaluation import (
    ClassificationEvaluation,
    ReconstructionEvaluation,
    evaluate_classification,
    evaluate_reconstruction,
    load_module_from_checkpoint,
)
from eos.model.compression import (
    SCOPE_PATHS,
    ASVDResult,
    LayerCompression,
    factorize_module,
    state_signature,
)
from eos.model.hybrid_cae import ConvAE, PureCNN
from eos.telemetry.export import TelemetryPush, build_asvd_payload
from eos.telemetry.mmd_drift import model_fingerprint
from eos.tempdir import use_repo_scratch_space
from eos.training.lightning_module import EOSTrainingModule

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/base.yaml")
"""Experiment configuration supplying the evaluation datamodule."""

DEFAULT_DATASET_CONTRACT = Path("data/contracts/eurosat-rgb.contract.yaml")
"""Dataset contract supplying class names in class-id order."""

DEFAULT_OUTPUT_DIR = Path("reports/compression")
"""Tracked destination of the compression benchmark artifact."""

DEFAULT_SCOPES: tuple[str, ...] = ("classifier", "bottleneck", "all_intermediate")
"""Scopes benchmarked unless overridden on the command line."""

DEFAULT_ENERGIES: tuple[float, ...] = (0.90, 0.95)
"""Admissible energy-retention band endpoints swept by the benchmark."""

WARMUP_PASSES = 3
"""Forward passes executed before latency sampling starts."""

TIMED_PASSES = 10
"""Forward passes sampled for the latency statistics."""

M4_DENSE_REDUCTION = 0.30
"""Milestone floor for dense-parameter reduction."""

M4_F1_LOSS = 0.015
"""Milestone ceiling for macro-F1 degradation."""

SCHEMA_VERSION = 1
"""Benchmark artifact schema version."""


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Forward-pass latency sampled over a fixed batch.

    Args:
        median_ms (float): Median pass duration in milliseconds.
        min_ms (float): Fastest sampled pass in milliseconds.
        max_ms (float): Slowest sampled pass in milliseconds.
        passes (int): Number of sampled passes.
        batch_size (int): Batch size used for every pass.
    """

    median_ms: float
    min_ms: float
    max_ms: float
    passes: int
    batch_size: int

    def to_dict(self) -> dict[str, object]:
        """Serialize the latency statistics.

        Returns:
            dict[str, object]: JSON-ready payload.
        """
        return {
            "median_ms": self.median_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "passes": self.passes,
            "batch_size": self.batch_size,
        }


def measure_latency(
    model: nn.Module,
    batch: Tensor,
    warmup: int = WARMUP_PASSES,
    passes: int = TIMED_PASSES,
) -> LatencyStats:
    """Measure forward-pass latency for one model on one batch.

    The batch is moved to the device of the model's parameters, and CUDA
    dispatch is synchronized around every timed pass so each sample measures
    real completion time rather than kernel launch time.

    Args:
        model (nn.Module): Model evaluated in inference mode.
        batch (Tensor): Input batch of shape ``(B, 3, 64, 64)``.
        warmup (int): Untimed passes run before sampling.
        passes (int): Timed passes sampled for the statistics.

    Returns:
        LatencyStats: Median, minimum, and maximum pass durations.

    Raises:
        ValueError: If no pass is requested or the batch is empty.
    """
    if passes < 1:
        raise ValueError("at least one timed pass is required")
    if batch.shape[0] == 0:
        raise ValueError("the latency batch must not be empty")
    parameter = next(model.parameters())
    device = parameter.device
    batch = batch.to(device)
    model.eval()
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(batch)
        for _ in range(passes):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return LatencyStats(
        median_ms=float(median(samples)),
        min_ms=samples[0],
        max_ms=samples[-1],
        passes=passes,
        batch_size=int(batch.shape[0]),
    )


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    Args:
        path (Path): File to digest.

    Returns:
        str: Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _layer_payload(layer: LayerCompression) -> dict[str, object]:
    """Serialize one compression record.

    Args:
        layer (LayerCompression): Per-layer accounting entry.

    Returns:
        dict[str, object]: JSON-ready payload including the measured gain.
    """
    return {
        "name": layer.name,
        "shape": list(layer.shape),
        "rank": layer.rank,
        "energy": layer.energy,
        "gain": layer.gain,
        "params_before": layer.params_before,
        "params_after": layer.params_after,
        "skipped": layer.skipped,
        "reason": layer.reason,
    }


def _evaluate(
    module: EOSTrainingModule,
    datamodule: EuroSATDataModule,
    device: str | None,
) -> tuple[ClassificationEvaluation | None, ReconstructionEvaluation | None]:
    """Evaluate a loaded module on the test split in the evaluation regime.

    Args:
        module (EOSTrainingModule): Reloaded training module under test.
        datamodule (EuroSATDataModule): Prepared datamodule providing the test
            loader.
        device (str | None): Inference device, or None for automatic selection.

    Returns:
        tuple[ClassificationEvaluation | None, ReconstructionEvaluation | None]:
        Classification metrics when the arm has a head and reconstruction
        metrics when the arm has a decoder; both are absent when missing.
    """
    loader = datamodule.test_dataloader()
    classification = (
        None
        if isinstance(module.model, ConvAE)
        else evaluate_classification(module, loader, device=device)
    )
    reconstruction = (
        None
        if isinstance(module.model, PureCNN)
        else evaluate_reconstruction(module, loader, device=device)
    )
    return classification, reconstruction


def benchmark_checkpoint(  # noqa: PLR0913, PLR0917
    checkpoint: Path,
    asvd: ASVDConfig,
    scopes: Sequence[str],
    energies: Sequence[float],
    datamodule: EuroSATDataModule,
    class_names: Sequence[str],
    latency_batch: Tensor,
    device: str | None,
    reporter: TelemetryPush | None = None,
) -> dict[str, object]:
    """Benchmark one checkpoint across the requested ASVD scopes and energies.

    Each scope and energy starts from a freshly reloaded checkpoint because
    factorization rewrites the model weights in place.

    Args:
        checkpoint (Path): Checkpoint file produced by the eos trainer.
        asvd (ASVDConfig): Minimum rank and compression-gain floor.
        scopes (Sequence[str]): Scope names to benchmark.
        energies (Sequence[float]): Energy-retention values to sweep.
        datamodule (EuroSATDataModule): Prepared datamodule providing the test
            loader.
        class_names (Sequence[str]): Class names in class-id order.
        latency_batch (Tensor): Batch used for latency measurement.
        device (str | None): Inference device, or None for automatic selection.
        reporter (TelemetryPush | None): Armed gauge exporter, or None to skip
            telemetry export.

    Returns:
        dict[str, object]: Baseline and per-scope measurements.

    Raises:
        ValueError: If a requested scope is unknown or an energy leaves the
            admissible band enforced by ``ASVDConfig``.
    """
    checkpoint_sha = _sha256(checkpoint)
    manifest = datamodule.manifest
    manifest_sha = manifest.metadata.manifest_sha256 if manifest is not None else ""
    baseline_module = load_module_from_checkpoint(checkpoint, map_location="cpu")
    baseline_classification, baseline_reconstruction = _evaluate(
        baseline_module, datamodule, device
    )
    baseline_latency = measure_latency(baseline_module.model, latency_batch)
    baseline_params = sum(parameter.numel() for parameter in baseline_module.model.parameters())
    baseline_payload: dict[str, object] = {
        "params": baseline_params,
        "classification": (
            baseline_classification.to_dict(class_names)
            if baseline_classification is not None
            else None
        ),
        "reconstruction": (
            baseline_reconstruction.to_dict() if baseline_reconstruction is not None else None
        ),
        "latency": baseline_latency.to_dict(),
    }
    scope_payloads: list[dict[str, object]] = []
    for scope in scopes:
        if scope not in SCOPE_PATHS:
            raise ValueError(f"unknown ASVD scope {scope!r}")
        for energy in energies:
            scope_config = ASVDConfig.model_validate(
                {
                    "scope": scope,
                    "preserve_energy": energy,
                    "min_rank": asvd.min_rank,
                    "min_compression_gain": asvd.min_compression_gain,
                }
            )
            module = load_module_from_checkpoint(checkpoint, map_location="cpu")
            result: ASVDResult = factorize_module(module.model, scope_config)
            classification, reconstruction = _evaluate(module, datamodule, device)
            latency = measure_latency(module.model, latency_batch)
            macro_f1_delta = (
                None
                if classification is None or baseline_classification is None
                else classification.macro_f1 - baseline_classification.macro_f1
            )
            f1_loss_ok = None if macro_f1_delta is None else -macro_f1_delta < M4_F1_LOSS
            scope_payloads.append(
                {
                    "scope": scope,
                    "preserve_energy": energy,
                    "layers": [_layer_payload(layer) for layer in result.layers],
                    "dense_before": result.dense_before,
                    "dense_after": result.dense_after,
                    "dense_reduction": result.dense_reduction,
                    "model_params_before": result.model_params_before,
                    "model_params_after": result.model_params_after,
                    "model_reduction": result.model_reduction,
                    "classification": (
                        classification.to_dict(class_names) if classification is not None else None
                    ),
                    "reconstruction": (
                        reconstruction.to_dict() if reconstruction is not None else None
                    ),
                    "latency": latency.to_dict(),
                    "macro_f1_delta": macro_f1_delta,
                    "m4_dense_reduction_ok": result.dense_reduction >= M4_DENSE_REDUCTION,
                    "m4_f1_loss_ok": f1_loss_ok,
                    "m4_passes": (
                        None
                        if f1_loss_ok is None
                        else result.dense_reduction >= M4_DENSE_REDUCTION and f1_loss_ok
                    ),
                }
            )
            LOGGER.info(
                "%s scope=%s energy=%.2f dense_reduction=%.4f macro_f1_delta=%s",
                checkpoint.name,
                scope,
                energy,
                result.dense_reduction,
                "n/a" if macro_f1_delta is None else f"{macro_f1_delta:+.4f}",
            )
            if reporter is not None and reporter.active:
                reporter.push(
                    build_asvd_payload(
                        result.dense_after / result.dense_before
                        if result.dense_before > 0
                        else 1.0,
                        macro_f1_delta if macro_f1_delta is not None else 0.0,
                        fingerprint=model_fingerprint(
                            checkpoint_sha, state_signature(result), manifest_sha
                        ),
                    )
                )
    return {
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "mode": baseline_module.mode,
        "baseline": baseline_payload,
        "scopes": scope_payloads,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: Parser with all benchmark options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-contract", type=Path, default=DEFAULT_DATASET_CONTRACT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        dest="checkpoints",
        required=True,
        help="Checkpoint to benchmark; repeat for several arms.",
    )
    parser.add_argument("--scopes", nargs="+", default=list(DEFAULT_SCOPES))
    parser.add_argument(
        "--energies",
        nargs="+",
        type=float,
        default=list(DEFAULT_ENERGIES),
        help="Energy-retention values to sweep within the admissible band.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "benchmark.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--push", action="store_true", help="Export ASVD gauges to the gateway.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the compression benchmark and write the artifact.

    Args:
        argv (Sequence[str] | None): Command-line arguments, or None to read
            the process arguments.

    Returns:
        int: Zero on success.

    Raises:
        FileNotFoundError: If a checkpoint does not exist.
        ValueError: If the datamodule produces no manifest or a scope is unknown.
    """
    use_repo_scratch_space()
    cap_vram()
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
    latency_images, _ = next(iter(datamodule.test_dataloader()))
    reporter: TelemetryPush | None = None
    if args.push:
        reporter = TelemetryPush.from_settings()
        if not reporter.start():
            LOGGER.warning("telemetry push unavailable: %s", reporter.failure)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "version": manifest.metadata.dataset_version,
            "manifest_sha256": manifest.metadata.manifest_sha256,
        },
        "asvd": config.asvd.model_dump(mode="json"),
        "milestone": {
            "min_dense_reduction": M4_DENSE_REDUCTION,
            "max_f1_loss": M4_F1_LOSS,
        },
        "latency": {
            "warmup_passes": WARMUP_PASSES,
            "timed_passes": TIMED_PASSES,
            "batch_size": int(latency_images.shape[0]),
            "device": args.device or "auto",
        },
        "benchmarks": [
            benchmark_checkpoint(
                checkpoint,
                config.asvd,
                args.scopes,
                args.energies,
                datamodule,
                class_names,
                latency_images,
                args.device,
                reporter,
            )
            for checkpoint in args.checkpoints
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("compression benchmark written to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
