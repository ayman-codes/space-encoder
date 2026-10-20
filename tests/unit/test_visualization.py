"""Headless unit tests for the notebook visualisation helpers.

The Agg backend is forced at module import before any figure is created. Neither this
module nor ``eos.utils.visualization`` imports ``matplotlib.pyplot``, so the notebook
deliverable keeps its interactive backend while CI renders safely headless. All
synthetic data comes from seeded generators and every file write stays inside pytest's
``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D

from eos.utils import visualization as viz

matplotlib.use("Agg")


def test_apply_style_is_idempotent_and_transparent() -> None:
    """The style must be repeatable and leave transparent backgrounds."""
    viz.apply_style()
    viz.apply_style()
    assert matplotlib.rcParams["figure.facecolor"] == "none"
    assert matplotlib.rcParams["axes.facecolor"] == "none"
    assert matplotlib.rcParams["savefig.transparent"] is True
    assert matplotlib.rcParams["figure.dpi"] == 120
    assert matplotlib.rcParams["axes.spines.top"] is False
    assert matplotlib.rcParams["axes.spines.right"] is False
    assert matplotlib.rcParams["axes.grid"] is True


def test_to_array_converts_numpy_lists_and_tensors() -> None:
    """Tensors, arrays, and nested lists must become floating arrays."""
    from_numpy = viz.to_array(np.array([[1.0, 2.0]], dtype=np.float32))
    assert np.issubdtype(from_numpy.dtype, np.floating)
    assert np.allclose(from_numpy, [[1.0, 2.0]])
    from_list = viz.to_array([[1, 2], [3, 4]])
    assert from_list.dtype == np.float64
    assert from_list.shape == (2, 2)
    from_tensor = viz.to_array(torch.tensor([[1.0, 2.0]]))
    assert np.issubdtype(from_tensor.dtype, np.floating)
    assert np.allclose(from_tensor, [[1.0, 2.0]])


def test_to_array_rejects_unsupported_input() -> None:
    """Unsupported containers and non-numeric payloads must raise TypeError."""
    with pytest.raises(TypeError, match="array-like"):
        viz.to_array({"a": 1})
    with pytest.raises(TypeError, match="numeric"):
        viz.to_array(["not", "numbers"])


def test_save_figure_round_trip(tmp_path: Path) -> None:
    """Saving must create directories and write a non-empty file."""
    fig, _ax = viz.plot_class_metrics_bar([0.2, 0.4], class_names=("a", "b"))
    target = tmp_path / "nested" / "metrics.png"
    saved = viz.save_figure(fig, target, dpi=60)
    assert saved == target.resolve()
    assert saved.is_file()
    assert saved.stat().st_size > 0


def test_save_figure_rejects_bad_dpi(tmp_path: Path) -> None:
    """A dpi below one has no defined rendering contract."""
    fig, _ax = viz.plot_class_metrics_bar([0.5], class_names=("only",))
    with pytest.raises(ValueError, match="dpi"):
        viz.save_figure(fig, tmp_path / "bad.png", dpi=0)


def test_patch_grid_supports_channels_first_and_channels_last() -> None:
    """Both (N, C, H, W) and (N, H, W, C) batches must produce the same grid."""
    rng = np.random.default_rng(42)
    originals = rng.random((6, 3, 8, 8)).astype(np.float32)
    reconstructions = np.clip(originals + 0.05, 0.0, 1.0)
    labels = [0, 1, 0, 2, 1, 2]
    fig_first, axes_first = viz.plot_patch_grid(originals, reconstructions, labels, max_per_class=2)
    _fig_last, axes_last = viz.plot_patch_grid(
        np.moveaxis(originals, 1, -1),
        np.moveaxis(reconstructions, 1, -1),
        labels,
        max_per_class=2,
    )
    assert isinstance(fig_first, Figure)
    assert axes_first.shape == (6, 2)
    assert axes_last.shape == axes_first.shape
    assert isinstance(axes_first[0, 0], Axes)
    assert axes_first[2, 0].get_ylabel() == "original"
    assert axes_first[3, 0].get_ylabel() == "reconstruction"


def test_patch_grid_titles_carry_palette_colors() -> None:
    """Class row titles must act as palette-coloured swatches."""
    rng = np.random.default_rng(42)
    originals = rng.random((4, 3, 8, 8))
    fig, axes = viz.plot_patch_grid(
        originals, originals, [0, 0, 1, 1], max_per_class=2, title="grid"
    )
    assert fig.texts[0].get_text() == "grid"
    assert axes[0, 0].get_title() == "AnnualCrop"
    assert axes[0, 0].title.get_color() == viz.PALETTE[0]
    assert axes[2, 0].get_title() == "Forest"
    assert axes[2, 0].title.get_color() == viz.PALETTE[1]
    assert axes[0, 1].get_title() == ""


def test_residual_heatmaps_are_per_sample_with_colorbars() -> None:
    """Each sample heatmap must use the sequential colormap with its own colorbar."""
    rng = np.random.default_rng(42)
    originals = rng.random((5, 3, 8, 8))
    reconstructions = np.clip(originals + 0.1, 0.0, 1.0)
    labels = [0, 1, 2, 3, 4]
    fig, axes = viz.plot_residual_heatmaps(originals, reconstructions, labels, max_samples=3)
    assert axes.shape == (1, 3)
    for ax in axes.flat:
        assert len(ax.images) == 1
        assert ax.images[0].get_cmap().name == viz.SEQUENTIAL
        assert ax.get_title().startswith("#")
    assert len(fig.axes) == 6


def test_latent_projection_legend_aspect_and_transparency() -> None:
    """The 2D projection must be transparent, equal-aspect, and class-legended."""
    rng = np.random.default_rng(42)
    points = rng.normal(size=(30, 2))
    labels = np.array([0, 1, 2] * 10)
    fig, ax = viz.plot_latent_projection(points, labels)
    assert isinstance(fig, Figure)
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
    ]
    assert ax.get_aspect() in (1.0, "equal")
    assert ax.get_xlabel() == "z1"
    assert ax.get_ylabel() == "z2"
    assert ax.get_title() == "Latent space projection"
    assert fig.get_facecolor()[3] == 0.0
    assert ax.get_facecolor()[3] == 0.0


def test_latent_projection_composes_with_existing_axes() -> None:
    """A caller supplied axes must be reused instead of allocating a figure."""
    rng = np.random.default_rng(42)
    points = rng.normal(size=(10, 2))
    labels = np.zeros(10, dtype=int)
    figure = Figure()
    existing = figure.add_subplot(111)
    returned_fig, returned_ax = viz.plot_latent_projection(
        points, labels, ax=existing, legend=False, title="custom", axis_labels=("a", "b")
    )
    assert returned_fig is figure
    assert returned_ax is existing
    assert existing.get_legend() is None
    assert existing.get_title() == "custom"
    assert existing.get_xlabel() == "a"
    assert existing.get_ylabel() == "b"


def test_latent_scatter_3d_axis_labels_and_legend() -> None:
    """The 3D scatter must expose overridable axis labels and a class legend."""
    rng = np.random.default_rng(42)
    points = rng.normal(size=(20, 3))
    labels = np.array([0, 1] * 10)
    fig, ax = viz.plot_latent_scatter_3d(points, labels, axis_labels=("a", "b", "c"))
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes3D)
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()) == ("a", "b", "c")
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["AnnualCrop", "Forest"]
    _fig_default, ax_default = viz.plot_latent_scatter_3d(points, labels, legend=False)
    assert (ax_default.get_xlabel(), ax_default.get_zlabel()) == ("z1", "z3")
    assert ax_default.get_legend() is None


def test_histogram_3d_draws_bar3d_collections() -> None:
    """The 3D histogram must build genuine bar3d collections with group legend."""
    rng = np.random.default_rng(42)
    groups = [rng.normal(loc=0.0, size=50), rng.normal(loc=1.0, size=50)]
    fig, ax = viz.plot_histogram_3d(groups, group_names=("baseline", "hybrid"), bins=6)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes3D)
    bars = [collection for collection in ax.collections if isinstance(collection, Poly3DCollection)]
    assert len(bars) == 2
    assert all(bar.get_facecolors().size > 0 for bar in bars)
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["baseline", "hybrid"]
    assert ax.get_xlabel() == "group"
    assert ax.get_ylabel() == "value"
    assert ax.get_zlabel() == "count"


def test_loss_trajectories_lines_legend_and_markers() -> None:
    """Trajectory plots must label axes, legend every series, and honour markers."""
    history = {"train/ce": [1.0, 0.5, 0.25], "val/loss": [1.2, 0.7, 0.4]}
    fig, ax = viz.plot_loss_trajectories(history, title="curves")
    assert isinstance(fig, Figure)
    assert ax.get_title() == "curves"
    assert ax.get_xlabel() == "epoch"
    assert ax.get_ylabel() == "value"
    assert len(ax.get_lines()) == 2
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["train/ce", "val/loss"]
    assert all(line.get_marker() in ("None", "") for line in ax.get_lines())
    _fig_marked, marked = viz.plot_loss_trajectories(history, marker=True, legend=False)
    assert marked.get_legend() is None
    assert all(line.get_marker() == "o" for line in marked.get_lines())


def test_confusion_matrix_normalizes_rows_and_emphasizes_diagonal() -> None:
    """Normalization must divide each row and bold the diagonal annotations."""
    matrix = np.array([[8.0, 2.0, 0.0], [1.0, 9.0, 0.0], [0.0, 0.0, 0.0]])
    fig, ax = viz.plot_confusion_matrix(matrix, normalize=True, class_names=("a", "b", "c"))
    displayed = np.asarray(ax.images[0].get_array())
    row_sums = matrix.sum(axis=1, keepdims=True)
    expected = matrix / np.where(row_sums > 0.0, row_sums, 1.0)
    assert np.allclose(displayed, expected)
    assert np.allclose(displayed.sum(axis=1), [1.0, 1.0, 0.0])
    assert ax.texts[0].get_text() == "0.80"
    assert ax.texts[0].get_fontweight() == "bold"
    assert ax.texts[1].get_fontweight() == "normal"
    assert [text.get_text() for text in ax.get_xticklabels()] == ["a", "b", "c"]
    assert ax.get_xlabel() == "predicted label"
    assert ax.get_ylabel() == "true label"
    assert len(fig.axes) == 2


def test_confusion_matrix_formats_counts_when_not_normalized() -> None:
    """Raw count matrices must annotate integer values compactly."""
    matrix = np.array([[8, 2], [1, 9]], dtype=float)
    _fig, ax = viz.plot_confusion_matrix(matrix, class_names=("a", "b"))
    assert [text.get_text() for text in ax.texts] == ["8", "2", "1", "9"]
    assert ax.texts[0].get_fontweight() == "normal"
    assert ax.get_title() == "Confusion matrix"


def test_class_metrics_bar_annotations_and_legend() -> None:
    """Per-class bars must annotate values and name all ten EuroSAT classes."""
    values = np.linspace(0.1, 0.9, 10)
    fig, ax = viz.plot_class_metrics_bar(values, metric_name="macro-F1")
    assert isinstance(fig, Figure)
    assert len(ax.patches) == 10
    assert [text.get_text() for text in ax.get_yticklabels()] == list(viz.CLASS_NAMES)
    assert [text.get_text() for text in ax.texts] == [f"{value:.3f}" for value in values]
    assert ax.get_xlabel() == "macro-F1"
    assert ax.get_legend() is None
    _fig, labelled = viz.plot_class_metrics_bar(
        [0.5, 0.7], class_names=("a", "b"), metric_name="accuracy", legend=True
    )
    assert labelled.get_xlabel() == "accuracy"
    assert labelled.get_legend() is not None


def test_metric_distributions_violins_with_deterministic_jitter() -> None:
    """Violin bodies, optional jittered points, and deterministic jitter are required."""
    rng = np.random.default_rng(42)
    arms = [rng.normal(loc=0.5, size=6), rng.normal(loc=0.6, size=6)]
    fig, ax = viz.plot_metric_distributions(arms, arm_names=("baseline", "hybrid"))
    assert isinstance(fig, Figure)
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["baseline", "hybrid"]
    assert ax.get_ylabel() == "macro-F1"
    assert any(isinstance(collection, PolyCollection) for collection in ax.collections)
    scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(scatters) == 2
    _fig, twin = viz.plot_metric_distributions(arms, arm_names=("baseline", "hybrid"))
    scatters_twin = [c for c in twin.collections if isinstance(c, PathCollection)]
    for first, second in zip(scatters, scatters_twin, strict=True):
        assert np.allclose(first.get_offsets(), second.get_offsets())
    _fig_plain, plain = viz.plot_metric_distributions(arms, show_points=False, legend=False)
    assert plain.get_legend() is None
    assert not [c for c in plain.collections if isinstance(c, PathCollection)]


def test_ablation_bars_groups_arms_per_metric() -> None:
    """Ablation bars must group arms per metric with legend and value labels."""
    arm_metrics = {
        "classifier": {"macro-F1": 0.7, "accuracy": 0.75},
        "hybrid": {"macro-F1": 0.8, "accuracy": 0.82},
    }
    fig, ax = viz.plot_ablation_bars(arm_metrics)
    assert isinstance(fig, Figure)
    assert len(ax.patches) == 4
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["classifier", "hybrid"]
    assert [text.get_text() for text in ax.get_xticklabels()] == ["macro-F1", "accuracy"]
    assert [text.get_text() for text in ax.texts] == ["0.700", "0.750", "0.800", "0.820"]
    assert ax.get_title() == "Ablation comparison"
    _fig, subset = viz.plot_ablation_bars(arm_metrics, metric_names=("accuracy",))
    assert [text.get_text() for text in subset.get_xticklabels()] == ["accuracy"]


def test_invalid_shapes_and_empty_inputs_raise_value_errors() -> None:
    """Malformed tensor shapes and empty containers must fail with ValueError."""
    flat = np.zeros((3, 8, 8))
    with pytest.raises(ValueError, match="4D"):
        viz.plot_patch_grid(flat, flat, [0, 1, 2])
    with pytest.raises(ValueError, match="same shape"):
        viz.plot_patch_grid(np.zeros((2, 3, 8, 8)), np.zeros((2, 3, 8, 4)), [0, 1])
    with pytest.raises(ValueError, match="sample count"):
        viz.plot_residual_heatmaps(np.zeros((2, 3, 8, 8)), np.zeros((2, 3, 8, 8)), [0])
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        viz.plot_latent_projection(np.zeros((4, 3)), np.zeros(4, dtype=int))
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        viz.plot_latent_scatter_3d(np.zeros((4, 2)), np.zeros(4, dtype=int))
    with pytest.raises(ValueError, match="axis_labels"):
        viz.plot_latent_projection(np.zeros((4, 2)), np.zeros(4, dtype=int), axis_labels=("a",))
    with pytest.raises(ValueError, match="history"):
        viz.plot_loss_trajectories({})
    with pytest.raises(ValueError, match="at least one group"):
        viz.plot_histogram_3d([])
    with pytest.raises(ValueError, match="square"):
        viz.plot_confusion_matrix(np.zeros((2, 3)))


def test_torch_tensors_flow_through_plot_helpers() -> None:
    """Torch tensors must be accepted end to end without manual conversion."""
    generator = torch.Generator().manual_seed(42)
    originals = torch.rand(4, 3, 8, 8, generator=generator)
    reconstructions = originals + 0.05
    labels = torch.tensor([0, 1, 2, 3])
    fig, axes = viz.plot_patch_grid(originals, reconstructions, labels, max_per_class=2)
    assert isinstance(fig, Figure)
    assert axes.shape == (8, 2)
    points = torch.tensor([[0.0, 0.1], [1.0, 0.9], [0.2, 0.3], [0.8, 0.7]])
    _fig, ax = viz.plot_latent_projection(points, labels)
    assert len(ax.collections) == 4
