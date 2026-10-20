"""Run the SSIM boundary-policy benchmark and write evidence artifacts.

The script samples deterministic test-split patches, evaluates the three SSIM
boundary policies under a fixed perturbation suite, and writes a JSON result,
a markdown summary, and three figures. It is the reproducible source of the
finding recorded in `.opencode/progress/phase2-report.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from eos.config import ExperimentConfig, load_config
from eos.experiments.ssim_boundary import (
    PERTURBATIONS,
    BenchmarkSpec,
    apply_perturbation,
    run_benchmark,
)
from eos.model.losses import BOUNDARY_POLICIES, SSIMLoss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/base.yaml")
DEFAULT_OUTPUT = Path("reports/ssim-boundary")

matplotlib.use("Agg")


def _resolve(path: Path) -> Path:
    """Resolve a repository-relative path.

    Args:
        path (Path): Candidate path.

    Returns:
        Path: Absolute path rooted at the repository when relative.
    """
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_patches(
    config: ExperimentConfig,
    split: str,
    samples: int,
    seed: int,
) -> tuple[Tensor, str, str]:
    """Load a deterministic subset of curated patches.

    Args:
        config (ExperimentConfig): Experiment configuration.
        split (str): Manifest split name.
        samples (int): Number of patches to sample.
        seed (int): Sampling seed.

    Returns:
        tuple[Tensor, str, str]: Patches of shape ``(N, C, H, W)`` in ``[0, 1]``,
            dataset version, and manifest fingerprint.

    Raises:
        FileNotFoundError: If the curated arrays or metadata are missing.
        ValueError: If the split has no patches.
    """
    curated = _resolve(config.data.curated_root)
    images_path = curated / "images_uint8.npy"
    indices_path = curated / f"{split}_indices.npy"
    metadata_path = curated / "manifest.json"
    for path in (images_path, indices_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"curated artifact missing: {path}")
    images = np.load(images_path, mmap_mode="r")
    indices = np.load(indices_path)
    if indices.size == 0:
        raise ValueError(f"split {split!r} contains no patches")
    generator = np.random.default_rng(seed)
    chosen = np.sort(generator.choice(indices, size=min(samples, indices.size), replace=False))
    batch = np.asarray(images[chosen], dtype=np.float32) / 255.0
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return tensor, str(metadata["dataset_version"]), str(metadata["manifest_sha256"])


def _embed_map(similarity: Tensor) -> np.ndarray:
    """Place a cropped SSIM map inside a full-size NaN frame.

    Args:
        similarity (Tensor): Map of shape ``(H, W)``; smaller maps are centered.

    Returns:
        np.ndarray: Array of shape ``(size, size)`` with NaN outside the map.
    """
    values = similarity.detach().cpu().numpy()
    if values.shape[0] == values.shape[1] and values.shape[0] == 64:
        return values
    frame = np.full((64, 64), np.nan, dtype=np.float64)
    offset = (64 - values.shape[0]) // 2
    frame[offset : offset + values.shape[0], offset : offset + values.shape[1]] = values
    return frame


def _render_maps(images: Tensor, spec: BenchmarkSpec, output: Path) -> None:
    """Render SSIM maps per policy for three sample patches.

    Args:
        images (Tensor): Clean batch of shape ``(N, C, H, W)``.
        spec (BenchmarkSpec): Run settings.
        output (Path): Destination figure path.

    Returns:
        None: The figure is written to disk.
    """
    count = min(3, images.shape[0])
    clean = images[:count]
    corrupted = apply_perturbation(clean, "border_noise", spec)
    losses = {
        policy: SSIMLoss(window_size=11, sigma=1.5, boundary=policy) for policy in BOUNDARY_POLICIES
    }
    columns = 2 + len(BOUNDARY_POLICIES)
    figure, axes = plt.subplots(count, columns, figsize=(2.4 * columns, 2.4 * count))
    colormap = plt.get_cmap("inferno").copy()
    colormap.set_bad("black")
    for row in range(count):
        axes[row, 0].imshow(clean[row].permute(1, 2, 0).numpy())
        axes[row, 0].set_title("target" if row == 0 else "")
        axes[row, 1].imshow(corrupted[row].permute(1, 2, 0).numpy())
        axes[row, 1].set_title("border noise" if row == 0 else "")
        for column, policy in enumerate(BOUNDARY_POLICIES, start=2):
            similarity = losses[policy].ssim_map(corrupted, clean)[row].mean(dim=0)
            axes[row, column].imshow(
                _embed_map(similarity),
                cmap=colormap,
                vmin=0.0,
                vmax=1.0,
            )
            suffix = " (cropped)" if policy == "valid" else ""
            axes[row, column].set_title(f"{policy} SSIM{suffix}" if row == 0 else "")
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _render_gradients(images: Tensor, spec: BenchmarkSpec, output: Path) -> None:
    """Render absolute SSIM gradient mass per policy.

    Args:
        images (Tensor): Clean batch of shape ``(N, C, H, W)``.
        spec (BenchmarkSpec): Run settings.
        output (Path): Destination figure path.

    Returns:
        None: The figure is written to disk.
    """
    corrupted = apply_perturbation(images[:1], "border_noise", spec)
    clean = images[:1]
    figure, axes = plt.subplots(1, len(BOUNDARY_POLICIES), figsize=(4.2 * 3, 4.0))
    for axis, policy in zip(axes, BOUNDARY_POLICIES, strict=True):
        loss = SSIMLoss(window_size=11, sigma=1.5, boundary=policy)
        candidate = corrupted.detach().clone().requires_grad_(True)
        (gradient,) = torch.autograd.grad(loss(candidate, clean), candidate)
        magnitude = gradient.abs().sum(dim=1)[0].detach().cpu().numpy()
        peak = float(magnitude.max()) or 1.0
        axis.imshow(magnitude / peak, cmap="magma")
        axis.set_title(f"{policy}: |dL/dx| / max")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _render_sensitivity(result: dict[str, object], output: Path) -> None:
    """Render border and interior SSIM drops per policy.

    Args:
        result (dict[str, object]): Benchmark result.
        output (Path): Destination figure path.

    Returns:
        None: The figure is written to disk.
    """
    policies = cast(dict[str, dict[str, object]], result["policies"])
    names = list(BOUNDARY_POLICIES)
    border = []
    interior = []
    border_error = []
    interior_error = []
    for policy in names:
        perturbations = cast(dict[str, dict[str, float]], policies[policy]["perturbations"])
        border_stats = perturbations["border_noise"]
        interior_stats = perturbations["interior_noise"]
        border.append(border_stats["mean_drop"])
        interior.append(interior_stats["mean_drop"])
        border_error.append((border_stats["drop_ci_high"] - border_stats["drop_ci_low"]) / 2.0)
        interior_error.append(
            (interior_stats["drop_ci_high"] - interior_stats["drop_ci_low"]) / 2.0
        )
    positions = np.arange(len(names), dtype=np.float64)
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(positions - width / 2, border, width, yerr=border_error, label="border noise")
    axis.bar(positions + width / 2, interior, width, yerr=interior_error, label="interior noise")
    axis.set_xticks(positions, names)
    axis.set_ylabel("mean SSIM drop vs identity")
    axis.set_title("SSIM sensitivity by boundary policy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _fmt(value: float, digits: int = 5) -> str:
    """Format a float for a markdown table.

    Args:
        value (float): Value to format.
        digits (int): Decimal places.

    Returns:
        str: Fixed-point representation.
    """
    return f"{value:.{digits}f}"


def _write_summary(result: dict[str, object], output: Path) -> None:
    """Write a markdown summary of the benchmark.

    Args:
        result (dict[str, object]): Benchmark result.
        output (Path): Destination markdown path.

    Returns:
        None: The summary is written to disk.
    """
    policies = cast(dict[str, dict[str, object]], result["policies"])
    sensitivity = cast(dict[str, dict[str, object]], result["sensitivity"])
    dataset = cast(dict[str, object], result["dataset"])
    config = cast(dict[str, object], result["config"])
    lines = [
        "# SSIM Boundary Policy Benchmark",
        "",
        f"Generated: {result['generated_at']}",
        f"Dataset: `{dataset['version']}` · {dataset['samples']} patches · "
        f"manifest `{dataset['manifest_sha256']}`",
        f"Window {config['window_size']}x{config['window_size']}, sigma={config['sigma']}, "
        f"border {config['border']} px, {config['bootstrap']} bootstrap resamples.",
        "",
        "## Headline",
        "",
        "| Policy | Map | Border drop | Interior drop | Area-normalized ratio | "
        "Border gradient fraction |",
        "| :--- | :---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in BOUNDARY_POLICIES:
        stats = cast(dict[str, float], sensitivity[policy])
        gradient = cast(dict[str, float], policies[policy]["gradient"])
        lines.append(
            f"| `{policy}` | {policies[policy]['map_shape']} | "
            f"{_fmt(float(stats['border_drop']))} | {_fmt(float(stats['interior_drop']))} | "
            f"{_fmt(float(stats['area_ratio']), 3)} | "
            f"{_fmt(gradient['border_fraction'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## Per-perturbation SSIM",
            "",
            "| Policy | Perturbation | Mean SSIM | Drop vs identity | 95% CI |",
            "| :--- | :--- | ---: | ---: | :--- |",
        ]
    )
    for policy in BOUNDARY_POLICIES:
        perturbations = cast(dict[str, dict[str, float]], policies[policy]["perturbations"])
        for name in PERTURBATIONS:
            entry = perturbations[name]
            lines.append(
                f"| `{policy}` | `{name}` | {_fmt(entry['mean_ssim'])} | "
                f"{_fmt(entry['mean_drop'])} | "
                f"[{_fmt(entry['drop_ci_low'])}, {_fmt(entry['drop_ci_high'])}] |"
            )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "uv run python scripts/benchmark_ssim.py",
            "```",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write artifacts.

    Args:
        argv (list[str] | None): Optional argument override.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description="SSIM boundary-policy benchmark")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)

    config = load_config(_resolve(arguments.config))
    spec = BenchmarkSpec(
        samples=arguments.samples,
        seed=arguments.seed,
        bootstrap=arguments.bootstrap,
    )
    images, version, fingerprint = _load_patches(
        config,
        arguments.split,
        arguments.samples,
        arguments.seed,
    )
    result = run_benchmark(images, spec, fingerprint, version)

    output = _resolve(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary(result, output / "summary.md")
    _render_maps(images, spec, output / "ssim_maps.png")
    _render_gradients(images, spec, output / "gradient_maps.png")
    _render_sensitivity(result, output / "sensitivity.png")
    print(f"benchmark written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
