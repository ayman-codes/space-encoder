"""Calibrate the MMD latent-drift baseline for one model state.

The script collects bottleneck embeddings over the full validation split,
calibrates the multi-scale RBF bandwidth grid with the median heuristic,
bootstraps the null distribution at the live batch size, and writes the baseline
artifact together with its reference set. The baseline fingerprint binds it to
the checkpoint, the dataset manifest, and the ASVD factorization state, so a
baseline calibrated before an ASVD pass is provably stale afterwards. Optional
flags score an in-distribution batch and a radiometrically shifted batch against
the baseline, demonstrate the stale-baseline rejection, and export the drift
gauges consumed by the Grafana drift panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from eos.config import ASVDConfig, load_config
from eos.dataset.contract import load_contract
from eos.dataset.eurosat_datamodule import EuroSATDataModule
from eos.evaluation import collect_embeddings, load_module_from_checkpoint
from eos.model.compression import SCOPE_PATHS, ASVDResult, factorize_module, state_signature
from eos.telemetry.export import TelemetryPush, build_mmd_payload
from eos.telemetry.mmd import MMDBaseline, calibrate_baseline
from eos.telemetry.mmd_drift import DriftMonitor, model_fingerprint
from eos.training.lightning_module import EOSTrainingModule

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/base.yaml")
"""Experiment configuration supplying the evaluation datamodule."""

DEFAULT_DATASET_CONTRACT = Path("data/contracts/eurosat-rgb.contract.yaml")
"""Dataset contract supplying class names in class-id order."""

DEFAULT_OUTPUT_DIR = Path("reports/telemetry")
"""Tracked destination of the drift baseline artifacts."""

DEFAULT_LIVE_BATCH = 128
"""Live batch size the null distribution is bootstrapped at."""

DEFAULT_DRAWS = 500
"""Bootstrap draws behind the calibrated mean and deviation."""

DEFAULT_SEED = 42
"""Seed for the reference subsample and the bootstrap."""

DEFAULT_MAX_PER_CLASS = 500
"""Embedding cap per class; the validation split holds 405 per class."""

DEFAULT_DEMO_SHIFT = 0.25
"""Radiometric shift applied to the demonstration batch in pixel space."""

SCHEMA_VERSION = 1
"""Baseline artifact schema version."""


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


def _forward_embeddings(model: torch.nn.Module, images: Tensor) -> Tensor:
    """Extract bottleneck embeddings for one image batch.

    Args:
        model (torch.nn.Module): Model exposing the shared encoder.
        images (Tensor): Image batch of shape ``(B, 3, 64, 64)``.

    Returns:
        Tensor: Embeddings of shape ``(B, latent_dim)`` on the CPU.

    Raises:
        ValueError: If the model produces no bottleneck embedding.
    """
    with torch.inference_mode():
        output = model(images)
    embedding = output.embedding
    if not isinstance(embedding, Tensor):
        raise ValueError("the model produced no bottleneck embedding")
    return embedding.detach().cpu()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: Parser with all calibration options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-contract", type=Path, default=DEFAULT_DATASET_CONTRACT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--live-batch-size", type=int, default=DEFAULT_LIVE_BATCH)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS)
    parser.add_argument("--asvd-scope", default=None)
    parser.add_argument("--asvd-energy", type=float, default=0.90)
    parser.add_argument("--demo-shift", type=float, default=DEFAULT_DEMO_SHIFT)
    parser.add_argument("--demonstrate", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Calibrate the drift baseline and write the artifacts.

    Args:
        argv (Sequence[str] | None): Command-line arguments, or None to read
            the process arguments.

    Returns:
        int: Zero on success.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If the datamodule produces no manifest or the requested
            ASVD scope is unknown.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    contract = load_contract(args.dataset_contract)
    datamodule = EuroSATDataModule(config.data)
    datamodule.setup()
    manifest = datamodule.manifest
    if manifest is None:
        raise ValueError("the datamodule produced no manifest")
    module = load_module_from_checkpoint(args.checkpoint, map_location="cpu")
    asvd_result: ASVDResult | None = None
    if args.asvd_scope is not None:
        if args.asvd_scope not in SCOPE_PATHS:
            raise ValueError(f"unknown ASVD scope {args.asvd_scope!r}")
        asvd_result = factorize_module(
            module.model,
            ASVDConfig.model_validate(
                {"scope": args.asvd_scope, "preserve_energy": args.asvd_energy}
            ),
        )
        LOGGER.info(
            "factorized %s at energy %.2f: dense reduction %.4f",
            args.asvd_scope,
            args.asvd_energy,
            asvd_result.dense_reduction,
        )
    collected = collect_embeddings(
        module,
        datamodule.val_dataloader(),
        max_per_class=args.max_per_class,
        device=args.device,
    )
    embeddings = torch.as_tensor(collected.embeddings)
    checkpoint_sha = _sha256(args.checkpoint)
    asvd_state = state_signature(asvd_result) if asvd_result is not None else ""
    fingerprint = model_fingerprint(checkpoint_sha, asvd_state, manifest.metadata.manifest_sha256)
    baseline = calibrate_baseline(
        embeddings,
        config.mmd,
        args.live_batch_size,
        draws=args.draws,
        seed=args.seed,
        fingerprint=fingerprint,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = args.output_dir / "mmd_reference.npz"
    np.savez_compressed(
        reference_path,
        reference=baseline.reference.numpy(),
        bootstrap=baseline.samples.numpy(),
    )
    baseline_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "asvd_state": asvd_state,
        "fingerprint": fingerprint,
        "dataset": {
            "version": manifest.metadata.dataset_version,
            "manifest_sha256": manifest.metadata.manifest_sha256,
        },
        "class_names": list(contract.source.class_names),
        "collected_embeddings": int(embeddings.shape[0]),
        "live_batch_size": args.live_batch_size,
        "mmd": baseline.to_dict(),
        "artifacts": [reference_path.name],
    }
    (args.output_dir / "mmd_baseline.json").write_text(
        json.dumps(baseline_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "baseline calibrated: %d embeddings, mu=%.5f sigma=%.5f threshold=%.5f",
        embeddings.shape[0],
        baseline.mu,
        baseline.sigma,
        baseline.threshold,
    )
    if args.demonstrate:
        _demonstrate(module, datamodule, baseline, fingerprint, args)
    return 0


def _gather_images(datamodule: EuroSATDataModule) -> Tensor:
    """Collect the full validation image set into one tensor.

    The curated loader streams class-ordered batches, so demonstration batches
    are drawn from the gathered set with a seeded permutation to stay
    representative of the class mix.

    Args:
        datamodule (EuroSATDataModule): Prepared datamodule.

    Returns:
        Tensor: Images of shape ``(N, 3, 64, 64)`` in loader order.
    """
    batches: list[Tensor] = []
    for images, _ in datamodule.val_dataloader():
        batches.append(images)
    return torch.cat(batches, dim=0)


def _demonstrate(
    module: EOSTrainingModule,
    datamodule: EuroSATDataModule,
    baseline: MMDBaseline,
    fingerprint: str,
    args: argparse.Namespace,
) -> None:
    """Score demonstration batches and prove the stale-baseline rejection.

    Three live scenarios are scored against the baseline: a seeded random
    cross-class validation batch that must stay quiet, the same batch after a
    radiometric pixel shift that must alarm, and a single-class slice of the
    class-ordered loader that must alarm as class-mix drift. A monitor bound to
    a post-ASVD model state must additionally refuse the pre-ASVD baseline.

    Args:
        module (EOSTrainingModule): Reloaded training module.
        datamodule (EuroSATDataModule): Prepared datamodule.
        baseline (MMDBaseline): Calibrated baseline.
        fingerprint (str): Fingerprint of the current model state.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None: The demonstration artifact is written and gauges are pushed.
    """
    images = _gather_images(datamodule)
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(images.shape[0], generator=generator)
    batch = images[order[: args.live_batch_size]]
    monitor = DriftMonitor(baseline, model_fingerprint=fingerprint)
    null_score = monitor.score(_forward_embeddings(module.model, batch))
    shifted_images = (batch + args.demo_shift).clamp(0.0, 1.0)
    shifted_score = monitor.score(_forward_embeddings(module.model, shifted_images))
    mix_score = monitor.score(_forward_embeddings(module.model, images[: args.live_batch_size]))
    stale_rejected = True
    stale_message = ""
    try:
        DriftMonitor(
            baseline,
            model_fingerprint=model_fingerprint(fingerprint, "asvd-passed", ""),
        )
    except ValueError as error:
        stale_message = str(error)
    else:
        stale_rejected = False
    demo_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "demo_shift": args.demo_shift,
        "null": {
            "description": "seeded random cross-class validation batch",
            **null_score.to_dict(),
        },
        "shifted": {
            "description": f"same batch with +{args.demo_shift} radiometric pixel shift",
            **shifted_score.to_dict(),
        },
        "class_mix": {
            "description": "single-class slice of the class-ordered validation loader",
            **mix_score.to_dict(),
        },
        "stale_baseline_rejected": stale_rejected,
        "stale_message": stale_message,
    }
    (args.output_dir / "drift_demo.json").write_text(
        json.dumps(demo_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "drift demo: null mmd2=%.5f alarm=%s, shifted mmd2=%.5f alarm=%s, "
        "class-mix mmd2=%.5f alarm=%s",
        null_score.mmd2,
        null_score.alarm,
        shifted_score.mmd2,
        shifted_score.alarm,
        mix_score.mmd2,
        mix_score.alarm,
    )
    if args.push:
        reporter = TelemetryPush.from_settings()
        if reporter.start():
            reporter.push(build_mmd_payload(null_score, fingerprint=fingerprint, scenario="null"))
            reporter.push(
                build_mmd_payload(shifted_score, fingerprint=fingerprint, scenario="shifted")
            )
            reporter.push(
                build_mmd_payload(mix_score, fingerprint=fingerprint, scenario="class_mix")
            )


if __name__ == "__main__":
    raise SystemExit(main())
