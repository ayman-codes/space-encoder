"""Notebook visualisation helpers for the EuroSAT hybrid CNN-autoencoder deliverable.

The module renders every figure required by ``notebooks/01_hybrid_autoencoder.ipynb``:
original-versus-reconstructed patch grids, residual error heatmaps, 2D and 3D latent
projections, genuine 3D histograms, loss trajectories, confusion matrices, per-class
metric bars, per-arm metric distributions, and ablation comparisons.

The implementation never imports :mod:`matplotlib.pyplot`, never selects a backend, and
never calls ``plt.show`` so notebooks keep an interactive backend while CI runs headless.
Every figure is a standalone :class:`matplotlib.figure.Figure` with a transparent
background and can be persisted through :func:`eos.utils.visualization.save_figure`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore[import-untyped]
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch

CLASS_NAMES: tuple[str, ...] = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)

PALETTE: tuple[str, ...] = (
    "#4C6EF5",
    "#7048E8",
    "#12B886",
    "#2F9E44",
    "#F59F00",
    "#E8590C",
    "#E03131",
    "#D6336C",
    "#22B8CF",
    "#868E96",
)

SERIES_PALETTE: tuple[str, ...] = (
    "#3B5BDB",
    "#0CA678",
    "#E8590C",
    "#AE3EC9",
    "#1098AD",
    "#F08C00",
    "#C2255C",
    "#5F3DC4",
)

DIVERGING: str = "coolwarm"

SEQUENTIAL: str = "magma"

type ArrayLike = (
    torch.Tensor
    | NDArray[Any]
    | Sequence[float]
    | Sequence[Sequence[float]]
    | Sequence[Sequence[Sequence[float]]]
    | Sequence[Sequence[Sequence[Sequence[float]]]]
)

type LabelLike = torch.Tensor | NDArray[Any] | Sequence[int]


def apply_style() -> None:
    """Install the modern transparent rcParams shared by every figure.

    The style sets transparent figure and axes backgrounds, hides the top and right
    spines, enables a subtle dashed grid, and selects readable font sizes. Applying
    the style repeatedly is idempotent because the same values are rewritten.

    Returns:
        None: Matplotlib rcParams are updated in place.
    """
    matplotlib.rcParams["figure.facecolor"] = "none"
    matplotlib.rcParams["axes.facecolor"] = "none"
    matplotlib.rcParams["savefig.transparent"] = True
    matplotlib.rcParams["figure.dpi"] = 120
    matplotlib.rcParams["axes.spines.top"] = False
    matplotlib.rcParams["axes.spines.right"] = False
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["grid.alpha"] = 0.3
    matplotlib.rcParams["grid.linestyle"] = "--"
    matplotlib.rcParams["grid.color"] = "#ADB5BD"
    matplotlib.rcParams["font.size"] = 11
    matplotlib.rcParams["axes.titlesize"] = 13
    matplotlib.rcParams["axes.titleweight"] = "bold"
    matplotlib.rcParams["axes.labelsize"] = 11
    matplotlib.rcParams["axes.edgecolor"] = "#495057"
    matplotlib.rcParams["axes.labelcolor"] = "#212529"
    matplotlib.rcParams["xtick.color"] = "#495057"
    matplotlib.rcParams["ytick.color"] = "#495057"
    matplotlib.rcParams["legend.frameon"] = False


def _resolve_torch() -> ModuleType | None:
    """Import torch lazily so plotting works when torch is unavailable.

    Returns:
        types.ModuleType | None: The torch module when it is installed, otherwise None.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    return torch


def to_array(data: ArrayLike) -> NDArray[np.floating[Any]]:
    """Convert a tensor or array-like object to a floating NumPy array.

    Torch tensors are detached, moved to CPU, and exported through ``.numpy()``.
    Integer, unsigned, and boolean inputs are promoted to ``float64`` so downstream
    plotting arithmetic is well defined, while existing floating dtypes are kept.

    Args:
        data (ArrayLike): Torch tensor, NumPy array, or nested numeric sequence of
            up to four dimensions.

    Returns:
        NDArray[np.floating[Any]]: Floating array sharing the values of ``data``.

    Raises:
        TypeError: If ``data`` is not a torch tensor, NumPy array, list, or tuple,
            or if its dtype is not numeric.
    """
    torch_module = _resolve_torch()
    if torch_module is not None and isinstance(data, torch_module.Tensor):
        array: NDArray[Any] = np.asarray(cast("torch.Tensor", data).detach().cpu().numpy())
    elif isinstance(data, np.ndarray):
        array = data
    elif isinstance(data, (list, tuple)):
        array = np.asarray(data)
    else:
        raise TypeError(f"to_array expects a torch.Tensor or array-like, got {type(data).__name__}")
    if array.dtype.kind not in "biuf":
        raise TypeError(f"to_array expects numeric values, got dtype {array.dtype}")
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64)
    return array


def save_figure(fig: Figure, path: str | Path, *, dpi: int = 200, transparent: bool = True) -> Path:
    """Render a figure to disk with a transparent background.

    Parent directories are created on demand and the output is written through
    :meth:`matplotlib.figure.Figure.savefig`, so the global backend choice is
    irrelevant to the produced file.

    Args:
        fig (Figure): Figure to render.
        path (str | Path): Destination file path.
        dpi (int): Output resolution in dots per inch; must be >= 1.
        transparent (bool): When True, omit figure and axes background fills.

    Returns:
        Path: Resolved absolute path of the written file.

    Raises:
        ValueError: If ``dpi`` is smaller than 1.
    """
    if dpi < 1:
        raise ValueError(f"dpi must be >= 1, got {dpi}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, transparent=transparent, bbox_inches="tight")
    return target.resolve()


def _new_axes(figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Create a standalone transparent figure with a single 2D axes.

    Args:
        figsize (tuple[float, float]): Figure width and height in inches.

    Returns:
        tuple[Figure, Axes]: The figure and its single subplot.
    """
    figure = Figure(figsize=figsize)
    figure.patch.set_facecolor("none")
    ax = figure.add_subplot(111)
    ax.set_facecolor("none")
    return figure, ax


def _new_axes_3d(figsize: tuple[float, float]) -> tuple[Figure, Axes3D]:
    """Create a standalone transparent figure with a single 3D axes.

    Args:
        figsize (tuple[float, float]): Figure width and height in inches.

    Returns:
        tuple[Figure, Axes3D]: The figure and its single 3D subplot.
    """
    figure = Figure(figsize=figsize)
    figure.patch.set_facecolor("none")
    ax = cast("Axes3D", figure.add_subplot(111, projection="3d"))
    ax.set_facecolor("none")
    return figure, ax


def _resolve_axes(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Return the caller supplied axes or create a fresh figure with one axes.

    Args:
        ax (Axes | None): Existing axes to draw into, or None to allocate a figure.
        figsize (tuple[float, float]): Size used when a new figure is allocated.

    Returns:
        tuple[Figure, Axes]: The owning figure and the axes used for drawing.
    """
    if ax is None:
        return _new_axes(figsize)
    return cast("Figure", ax.figure), ax


def _channels_last(array: NDArray[Any]) -> NDArray[Any]:
    """Reorder a 4D image batch into channels-last layout.

    A leading channel axis of size 1, 3, or 4 is treated as channels-first and
    moved to the end; ambiguous shapes such as ``(N, 3, 3, 3)`` prefer the
    channels-first interpretation.

    Args:
        array (NDArray[Any]): Batch of shape (N, C, H, W) or (N, H, W, C).

    Returns:
        NDArray[Any]: Batch of shape (N, H, W, C).

    Raises:
        ValueError: If the batch is not 4D or exposes no plausible channel axis.
    """
    if array.ndim != 4:
        raise ValueError(f"expected a 4D batch, got shape {array.shape}")
    first_axis, last_axis = array.shape[1], array.shape[-1]
    if last_axis in (1, 3) and first_axis not in (1, 3):
        return array
    if first_axis in (1, 3, 4):
        return np.moveaxis(array, 1, -1)
    if last_axis in (1, 3, 4):
        return array
    raise ValueError(f"no channel axis of size 1, 3, or 4 in shape {array.shape}")


def _display_image(image: NDArray[Any]) -> NDArray[Any]:
    """Normalize a single channels-last patch for display.

    Grayscale patches are squeezed to 2D, wide multispectral patches keep their first
    three bands, and out-of-range intensities are min-max scaled before clipping to
    the unit interval expected by :meth:`matplotlib.axes.Axes.imshow`.

    Args:
        image (NDArray[Any]): Patch of shape (H, W, C) or (H, W, 1).

    Returns:
        NDArray[Any]: Displayable array of shape (H, W) or (H, W, 3) inside [0, 1].
    """
    displayed: NDArray[Any] = image
    if displayed.ndim == 3 and displayed.shape[-1] == 1:
        displayed = displayed[..., 0]
    if displayed.ndim == 3 and displayed.shape[-1] > 3:
        displayed = displayed[..., :3]
    low, high = float(displayed.min()), float(displayed.max())
    if low < 0.0 or high > 1.0:
        span = high - low
        displayed = (displayed - low) / span if span > 0.0 else np.zeros_like(displayed)
    clipped: NDArray[Any] = np.clip(displayed, 0.0, 1.0)
    return clipped


def _label_ids(labels: LabelLike, sample_count: int) -> NDArray[np.int64]:
    """Convert label input to a flat integer id array validated against a batch.

    Args:
        labels (LabelLike): Torch tensor, NumPy array, or integer sequence of shape (N,).
        sample_count (int): Expected number of samples in the paired image batch.

    Returns:
        NDArray[np.int64]: Integer class ids of shape (sample_count,).

    Raises:
        ValueError: If the number of labels differs from ``sample_count``.
    """
    values = to_array(labels).reshape(-1)
    if values.shape[0] != sample_count:
        raise ValueError(
            f"labels length {values.shape[0]} does not match sample count {sample_count}"
        )
    return values.astype(np.int64)


def _class_color(class_id: int) -> str:
    """Return the palette colour bound to a class id.

    Args:
        class_id (int): Integer class id; ids beyond the palette wrap around.

    Returns:
        str: Hex colour string from :data:`PALETTE`.
    """
    return PALETTE[class_id % len(PALETTE)]


def _class_name(class_id: int, class_names: Sequence[str] | None) -> str:
    """Resolve the display name of a single class id.

    Args:
        class_id (int): Integer class id.
        class_names (Sequence[str] | None): Caller supplied names indexed by class id.

    Returns:
        str: Caller supplied name, EuroSAT default name, or ``class_<id>`` fallback.
    """
    if class_names is not None and 0 <= class_id < len(class_names):
        return str(class_names[class_id])
    if class_names is None and 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"class_{class_id}"


def _class_display_names(count: int, class_names: Sequence[str] | None) -> list[str]:
    """Resolve a contiguous block of class display names.

    Args:
        count (int): Number of leading class names required.
        class_names (Sequence[str] | None): Caller supplied names indexed by class id.

    Returns:
        list[str]: Names for class ids 0 through count - 1.

    Raises:
        ValueError: If ``class_names`` is shorter than ``count``.
    """
    if class_names is not None:
        resolved = [str(name) for name in class_names]
        if len(resolved) < count:
            raise ValueError(f"expected at least {count} class names, got {len(resolved)}")
        return resolved[:count]
    if count == len(CLASS_NAMES):
        return list(CLASS_NAMES)
    return [f"class_{index}" for index in range(count)]


def _display_names(count: int, names: Sequence[str] | None, prefix: str) -> list[str]:
    """Resolve group, arm, or series display names with an indexed fallback.

    Args:
        count (int): Number of names required.
        names (Sequence[str] | None): Caller supplied names.
        prefix (str): Fallback prefix used to synthesise ``prefix_<index>`` names.

    Returns:
        list[str]: Exactly ``count`` display names.

    Raises:
        ValueError: If ``names`` is shorter than ``count``.
    """
    if names is not None:
        resolved = [str(name) for name in names]
        if len(resolved) < count:
            raise ValueError(f"expected at least {count} names, got {len(resolved)}")
        return resolved[:count]
    return [f"{prefix}_{index}" for index in range(count)]


def _axis_labels(axis_labels: Sequence[str] | None, defaults: tuple[str, ...]) -> list[str]:
    """Resolve axis labels against a fixed default set.

    Args:
        axis_labels (Sequence[str] | None): Caller supplied labels, or None for defaults.
        defaults (tuple[str, ...]): Labels used when the caller supplies nothing.

    Returns:
        list[str]: Resolved labels with the same length as ``defaults``.

    Raises:
        ValueError: If ``axis_labels`` length differs from ``defaults``.
    """
    if axis_labels is None:
        return list(defaults)
    resolved = [str(label) for label in axis_labels]
    if len(resolved) != len(defaults):
        raise ValueError(f"axis_labels must contain {len(defaults)} entries, got {len(resolved)}")
    return resolved


def _text_color(value: float, max_value: float, cmap: str) -> str:
    """Choose a readable annotation colour for a heatmap cell.

    The colormap is sampled at the cell intensity and its luminance decides between
    dark and light text so annotations stay legible on every colormap.

    Args:
        value (float): Cell value being annotated.
        max_value (float): Largest cell value in the displayed matrix.
        cmap (str): Matplotlib colormap name backing the heatmap.

    Returns:
        str: "black" on bright colormap regions and "white" on dark ones.
    """
    fraction = value / max_value if max_value > 0.0 else 0.0
    rgba = matplotlib.colormaps[cmap](fraction)
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "black" if luminance > 0.5 else "white"


def _format_cell(value: float, normalize: bool) -> str:
    """Format one confusion matrix cell for annotation.

    Args:
        value (float): Displayed cell value.
        normalize (bool): Whether the matrix is row-normalized.

    Returns:
        str: Two-decimal text for normalized matrices and compact text otherwise.
    """
    if normalize:
        return f"{value:.2f}"
    return f"{int(value)}" if float(value).is_integer() else f"{value:.2f}"


def plot_patch_grid(  # noqa: PLR0913
    originals: ArrayLike,
    reconstructions: ArrayLike,
    labels: LabelLike,
    *,
    class_names: Sequence[str] | None = None,
    max_per_class: int = 3,
    title: str | None = None,
) -> tuple[Figure, NDArray[Any]]:
    """Render originals against reconstructions in per-class row pairs.

    For every class present in ``labels`` two rows are drawn: the upper row shows up
    to ``max_per_class`` original patches and the lower row shows the matching
    reconstructions. The first column carries a coloured class title using
    :data:`PALETTE` as a legend-style swatch for the row pair.

    Args:
        originals (ArrayLike): Ground-truth batch of shape (N, C, H, W) or (N, H, W, C).
        reconstructions (ArrayLike): Reconstructed batch with the same shape.
        labels (LabelLike): Integer class ids of shape (N,).
        class_names (Sequence[str] | None): Class names indexed by class id.
        max_per_class (int): Maximum samples drawn per class; must be >= 1.
        title (str | None): Figure title; a descriptive default is used when omitted.

    Returns:
        tuple[Figure, NDArray[Any]]: Figure and an axes array of shape
            (2 * classes_present, max_per_class).

    Raises:
        ValueError: If ``max_per_class`` is below 1, the batches are not 4D with
            matching shapes, or the label count mismatches the sample count.
    """
    apply_style()
    if max_per_class < 1:
        raise ValueError(f"max_per_class must be >= 1, got {max_per_class}")
    original_array, reconstruction_array, label_ids = _paired_images(
        originals, reconstructions, labels
    )
    class_ids = sorted({int(value) for value in np.unique(label_ids)})
    row_count = 2 * len(class_ids)
    figure = Figure(figsize=(2.4 * max_per_class + 1.0, 2.4 * row_count))
    figure.patch.set_facecolor("none")
    axes: NDArray[Any] = np.empty((row_count, max_per_class), dtype=object)
    for class_row, class_id in enumerate(class_ids):
        sample_indices = np.flatnonzero(label_ids == class_id)[:max_per_class]
        colour = _class_color(class_id)
        name = _class_name(class_id, class_names)
        for column_index in range(max_per_class):
            original_ax = figure.add_subplot(
                row_count, max_per_class, 2 * class_row * max_per_class + column_index + 1
            )
            reconstruction_ax = figure.add_subplot(
                row_count, max_per_class, (2 * class_row + 1) * max_per_class + column_index + 1
            )
            axes[2 * class_row, column_index] = original_ax
            axes[2 * class_row + 1, column_index] = reconstruction_ax
            for target in (original_ax, reconstruction_ax):
                target.set_xticks([])
                target.set_yticks([])
                target.set_facecolor("none")
            if column_index < sample_indices.size:
                sample_index = int(sample_indices[column_index])
                original_ax.imshow(
                    _display_image(original_array[sample_index]), interpolation="nearest"
                )
                reconstruction_ax.imshow(
                    _display_image(reconstruction_array[sample_index]), interpolation="nearest"
                )
                if column_index == 0:
                    original_ax.set_title(name, color=colour, fontweight="bold")
            else:
                original_ax.set_axis_off()
                reconstruction_ax.set_axis_off()
        cast("Axes", axes[2 * class_row, 0]).set_ylabel("original", fontsize=9, color="#868E96")
        cast("Axes", axes[2 * class_row + 1, 0]).set_ylabel(
            "reconstruction", fontsize=9, color="#868E96"
        )
    figure.suptitle(title if title is not None else "Original vs reconstructed patches")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return figure, axes


def _paired_images(
    originals: ArrayLike,
    reconstructions: ArrayLike,
    labels: LabelLike,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[np.int64]]:
    """Convert a labeled image pair batch into channels-last arrays.

    Args:
        originals (ArrayLike): Ground-truth batch of shape (N, C, H, W) or (N, H, W, C).
        reconstructions (ArrayLike): Reconstructed batch with the same shape as originals.
        labels (LabelLike): Integer class ids of shape (N,).

    Returns:
        tuple[NDArray[Any], NDArray[Any], NDArray[np.int64]]: Channels-last originals,
            channels-last reconstructions, and integer label ids.

    Raises:
        ValueError: If either batch is not 4D, if the two shapes differ, or if the
            label count does not match the sample count.
    """
    original_array = _channels_last(to_array(originals))
    reconstruction_array = _channels_last(to_array(reconstructions))
    if original_array.shape != reconstruction_array.shape:
        raise ValueError(
            f"originals {original_array.shape} and reconstructions "
            f"{reconstruction_array.shape} must share the same shape"
        )
    return original_array, reconstruction_array, _label_ids(labels, original_array.shape[0])


def plot_residual_heatmaps(  # noqa: PLR0913
    originals: ArrayLike,
    reconstructions: ArrayLike,
    labels: LabelLike,
    *,
    class_names: Sequence[str] | None = None,
    max_samples: int = 8,
    title: str | None = None,
) -> tuple[Figure, NDArray[Any]]:
    """Render per-sample residual magnitude heatmaps.

    Each heatmap shows the channel-averaged absolute error ``|X - X_hat|`` for one of
    the first ``max_samples`` samples using the :data:`SEQUENTIAL` colormap and a
    dedicated colorbar. Titles carry the sample's class name in its palette colour.

    Args:
        originals (ArrayLike): Ground-truth batch of shape (N, C, H, W) or (N, H, W, C).
        reconstructions (ArrayLike): Reconstructed batch with the same shape.
        labels (LabelLike): Integer class ids of shape (N,).
        class_names (Sequence[str] | None): Class names indexed by class id.
        max_samples (int): Upper bound on displayed samples; must be >= 1.
        title (str | None): Figure title; a descriptive default is used when omitted.

    Returns:
        tuple[Figure, NDArray[Any]]: Figure and a 2D axes array covering the grid.

    Raises:
        ValueError: If ``max_samples`` is below 1, the batch is empty or not 4D with
            matching shapes, or the label count mismatches the sample count.
    """
    apply_style()
    if max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")
    original_array, reconstruction_array, label_ids = _paired_images(
        originals, reconstructions, labels
    )
    sample_count = original_array.shape[0]
    if sample_count == 0:
        raise ValueError("at least one sample is required")
    displayed_count = min(sample_count, max_samples)
    residuals = np.abs(original_array[:displayed_count] - reconstruction_array[:displayed_count])
    residual_maps = residuals.mean(axis=-1)
    column_count = min(displayed_count, 4)
    row_count = int(np.ceil(displayed_count / column_count))
    figure = Figure(figsize=(3.0 * column_count + 0.8, 2.8 * row_count))
    figure.patch.set_facecolor("none")
    axes: NDArray[Any] = np.empty((row_count, column_count), dtype=object)
    for sample_index in range(row_count * column_count):
        row_index, column_index = divmod(sample_index, column_count)
        ax = figure.add_subplot(row_count, column_count, sample_index + 1)
        ax.set_facecolor("none")
        axes[row_index, column_index] = ax
        if sample_index >= displayed_count:
            ax.set_axis_off()
            continue
        image = ax.imshow(residual_maps[sample_index], cmap=SEQUENTIAL, interpolation="nearest")
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        class_id = int(label_ids[sample_index])
        ax.set_title(
            f"#{sample_index} {_class_name(class_id, class_names)}",
            color=_class_color(class_id),
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    figure.suptitle(title if title is not None else "Reconstruction residual heatmaps")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return figure, axes


def plot_latent_projection(  # noqa: PLR0913
    points_2d: ArrayLike,
    labels: LabelLike,
    *,
    class_names: Sequence[str] | None = None,
    title: str | None = None,
    legend: bool = True,
    alpha: float = 0.85,
    axis_labels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Render a 2D latent manifold projection coloured by class.

    Points are drawn in :data:`PALETTE` colours on an equal-aspect axes so t-SNE or
    PCA maps keep their geometry, and a legend lists every class present.

    Args:
        points_2d (ArrayLike): Latent coordinates of shape (N, 2).
        labels (LabelLike): Integer class ids of shape (N,).
        class_names (Sequence[str] | None): Class names indexed by class id.
        title (str | None): Axes title; a descriptive default is used when omitted.
        legend (bool): Draw a class legend when True.
        alpha (float): Marker opacity in the interval (0, 1].
        axis_labels (Sequence[str] | None): Two axis labels overriding "z1" and "z2".
        ax (Axes | None): Existing axes to draw into; a new figure is created when None.

    Returns:
        tuple[Figure, Axes]: The owning figure and the drawn axes.

    Raises:
        ValueError: If ``points_2d`` is not shaped (N, 2), the label count mismatches
            the point count, or ``axis_labels`` does not contain exactly two entries.
    """
    apply_style()
    points = to_array(points_2d)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_2d must have shape (N, 2), got {points.shape}")
    label_ids = _label_ids(labels, points.shape[0])
    figure, ax = _resolve_axes(ax, figsize=(7.5, 5.5))
    for class_id in sorted({int(value) for value in np.unique(label_ids)}):
        mask = label_ids == class_id
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=36,
            alpha=alpha,
            color=_class_color(class_id),
            label=_class_name(class_id, class_names),
            edgecolors="none",
        )
    resolved_labels = _axis_labels(axis_labels, ("z1", "z2"))
    ax.set_xlabel(resolved_labels[0])
    ax.set_ylabel(resolved_labels[1])
    ax.set_aspect("equal", adjustable="datalim")
    if legend:
        ax.legend(loc="best")
    ax.set_title(title if title is not None else "Latent space projection")
    return figure, ax


def plot_latent_scatter_3d(  # noqa: PLR0913
    points_3d: ArrayLike,
    labels: LabelLike,
    *,
    class_names: Sequence[str] | None = None,
    title: str | None = None,
    legend: bool = True,
    alpha: float = 0.8,
    axis_labels: Sequence[str] | None = None,
) -> tuple[Figure, Axes3D]:
    """Render a 3D latent space scatter coloured by class.

    One scatter series per class is drawn in :data:`PALETTE` colours with shaded
    markers so depth is perceivable, and the default axes are labelled ``z1``, ``z2``,
    and ``z3``.

    Args:
        points_3d (ArrayLike): Latent coordinates of shape (N, 3).
        labels (LabelLike): Integer class ids of shape (N,).
        class_names (Sequence[str] | None): Class names indexed by class id.
        title (str | None): Axes title; a descriptive default is used when omitted.
        legend (bool): Draw a class legend when True.
        alpha (float): Marker opacity in the interval (0, 1].
        axis_labels (Sequence[str] | None): Three axis labels overriding the defaults.

    Returns:
        tuple[Figure, Axes3D]: The owning figure and the drawn 3D axes.

    Raises:
        ValueError: If ``points_3d`` is not shaped (N, 3), the label count mismatches
            the point count, or ``axis_labels`` does not contain exactly three entries.
    """
    apply_style()
    points = to_array(points_3d)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_3d must have shape (N, 3), got {points.shape}")
    label_ids = _label_ids(labels, points.shape[0])
    figure, ax = _new_axes_3d(figsize=(11.0, 6.5))
    for class_id in sorted({int(value) for value in np.unique(label_ids)}):
        mask = label_ids == class_id
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            points[mask, 2],
            s=32,
            alpha=alpha,
            color=_class_color(class_id),
            label=_class_name(class_id, class_names),
            depthshade=True,
            edgecolors="none",
        )
    resolved_labels = _axis_labels(axis_labels, ("z1", "z2", "z3"))
    ax.set_xlabel(resolved_labels[0])
    ax.set_ylabel(resolved_labels[1])
    ax.set_zlabel(resolved_labels[2])
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1.0), borderaxespad=0.0)
    ax.set_title(title if title is not None else "Latent space (3D)")
    return figure, ax


def plot_histogram_3d(
    values_by_group: Sequence[ArrayLike],
    *,
    group_names: Sequence[str] | None = None,
    bins: int = 20,
    title: str | None = None,
    legend: bool = True,
) -> tuple[Figure, Axes3D]:
    """Render a genuine 3D histogram with one depth-aligned row of bars per group.

    All groups share bin edges derived from the pooled value range so bars line up
    along the depth axis; each group's counts come from :func:`numpy.histogram`. The
    horizontal axis holds the group index, the depth axis holds bin edges, and the
    vertical axis holds counts. Bars are drawn with :meth:`mpl_toolkits.mplot3d.axes3d.Axes3D.bar3d`
    in :data:`PALETTE` colours with legend proxy artists per group.

    Args:
        values_by_group (Sequence[ArrayLike]): One 1D value array per group.
        group_names (Sequence[str] | None): Group names for ticks and legend entries.
        bins (int): Number of shared histogram bins; must be >= 1.
        title (str | None): Axes title; a descriptive default is used when omitted.
        legend (bool): Draw a legend of group proxy artists when True.

    Returns:
        tuple[Figure, Axes3D]: The owning figure and the drawn 3D axes.

    Raises:
        ValueError: If no groups are given, any group is empty, or ``bins`` is below 1.
    """
    apply_style()
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    groups = [to_array(group).reshape(-1) for group in values_by_group]
    if not groups:
        raise ValueError("values_by_group must contain at least one group")
    if any(group.size == 0 for group in groups):
        raise ValueError("each group must contain at least one value")
    pooled = np.concatenate(groups)
    low, high = float(pooled.min()), float(pooled.max())
    if high <= low:
        high = low + 1.0
    edges = np.linspace(low, high, bins + 1)
    names = _display_names(len(groups), group_names, "group")
    figure, ax = _new_axes_3d(figsize=(9.5, 6.5))
    bar_width = 0.6
    bin_width = float(edges[1] - edges[0])
    handles: list[Patch] = []
    for group_index, group in enumerate(groups):
        counts, _ = np.histogram(group, bins=edges)
        colour = PALETTE[group_index % len(PALETTE)]
        offsets_x = np.full(counts.size, group_index - bar_width / 2.0)
        offsets_y = edges[:-1]
        offsets_z = np.zeros(counts.size)
        ax.bar3d(
            offsets_x,
            offsets_y,
            offsets_z,
            bar_width,
            bin_width,
            counts,
            color=colour,
            alpha=0.85,
            shade=True,
        )
        handles.append(Patch(facecolor=colour, alpha=0.85, label=names[group_index]))
    ax.set_xticks(np.arange(len(groups), dtype=np.float64), labels=names)
    ax.set_xlabel("group")
    ax.set_ylabel("value")
    ax.set_zlabel("count")
    if legend:
        ax.legend(handles=handles, loc="best")
    ax.set_title(title if title is not None else "3D histogram by group")
    return figure, ax


def plot_loss_trajectories(
    history: Mapping[str, ArrayLike],
    *,
    title: str | None = None,
    legend: bool = True,
    marker: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Render one loss or metric trajectory per history key.

    Every series is drawn against its epoch index in :data:`SERIES_PALETTE` colours
    so training, reconstruction, and validation curves stay distinguishable.

    Args:
        history (Mapping[str, ArrayLike]): Series name to per-epoch values; values may
            be sequences or tensors.
        title (str | None): Axes title; a descriptive default is used when omitted.
        legend (bool): Draw a legend of series names when True.
        marker (bool): Draw circular markers on each epoch when True.
        ax (Axes | None): Existing axes to draw into; a new figure is created when None.

    Returns:
        tuple[Figure, Axes]: The owning figure and the drawn axes.

    Raises:
        ValueError: If ``history`` is empty or any series holds no values.
    """
    apply_style()
    if not history:
        raise ValueError("history must contain at least one series")
    figure, ax = _resolve_axes(ax, figsize=(8.5, 5.2))
    for series_index, (series_name, raw_values) in enumerate(history.items()):
        values = to_array(raw_values).reshape(-1)
        if values.size == 0:
            raise ValueError(f"series {series_name!r} holds no values")
        epochs = np.arange(values.size, dtype=np.float64)
        ax.plot(
            epochs,
            values,
            color=SERIES_PALETTE[series_index % len(SERIES_PALETTE)],
            linewidth=2.0,
            marker="o" if marker else None,
            markersize=4,
            label=series_name,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    if legend:
        ax.legend(loc="best")
    ax.set_title(title if title is not None else "Loss trajectories")
    return figure, ax


def plot_confusion_matrix(  # noqa: PLR0913
    matrix: ArrayLike,
    *,
    class_names: Sequence[str] | None = None,
    normalize: bool = False,
    title: str | None = None,
    cmap: str = SEQUENTIAL,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Render an annotated confusion matrix heatmap.

    Every cell is annotated with its value using text colour sampled from the
    colormap luminance, and normalized matrices bold the diagonal so per-class recall
    is immediately readable.

    Args:
        matrix (ArrayLike): Square confusion counts of shape (C, C) with true labels on
            rows and predicted labels on columns.
        class_names (Sequence[str] | None): Class names for the tick labels.
        normalize (bool): Divide each row by its sum; zero rows stay all-zero.
        title (str | None): Axes title; a descriptive default is used when omitted.
        cmap (str): Matplotlib colormap name for the heatmap.
        ax (Axes | None): Existing axes to draw into; a new figure is created when None.

    Returns:
        tuple[Figure, Axes]: The owning figure and the drawn axes.

    Raises:
        ValueError: If ``matrix`` is not a 2D square array or ``class_names`` is
            shorter than the matrix dimension.
    """
    apply_style()
    values = to_array(matrix)
    if values.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {values.shape}")
    if values.shape[0] != values.shape[1]:
        raise ValueError(f"matrix must be square, got shape {values.shape}")
    class_count = values.shape[0]
    names = _class_display_names(class_count, class_names)
    displayed = values.astype(np.float64, copy=True)
    if normalize:
        row_sums = displayed.sum(axis=1, keepdims=True)
        displayed = displayed / np.where(row_sums > 0.0, row_sums, 1.0)
    figure, ax = _resolve_axes(ax, figsize=(7.5, 6.2))
    image = ax.imshow(displayed, cmap=cmap, interpolation="nearest")
    figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    positions = np.arange(class_count, dtype=np.float64)
    ax.set_xticks(positions, labels=names)
    ax.set_yticks(positions, labels=names)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")
    max_value = float(displayed.max())
    for row_index in range(class_count):
        for column_index in range(class_count):
            value = float(displayed[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                _format_cell(value, normalize),
                ha="center",
                va="center",
                fontsize=9,
                color=_text_color(value, max_value, cmap),
                fontweight="bold" if normalize and row_index == column_index else "normal",
            )
    ax.set_xlabel("predicted label")
    ax.set_ylabel("true label")
    default_title = "Normalized confusion matrix" if normalize else "Confusion matrix"
    ax.set_title(title if title is not None else default_title)
    return figure, ax


def plot_class_metrics_bar(  # noqa: PLR0913
    values: ArrayLike,
    *,
    metric_name: str = "macro-F1",
    class_names: Sequence[str] | None = None,
    title: str | None = None,
    legend: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Render per-class metric values as annotated horizontal bars.

    Bars are coloured per class from :data:`PALETTE` and each bar carries its exact
    value so the figure can be quoted directly in the notebook report.

    Args:
        values (ArrayLike): One metric value per class of shape (C,).
        metric_name (str): Metric name used for the value axis label.
        class_names (Sequence[str] | None): Class names for the category axis.
        title (str | None): Axes title; a descriptive default is used when omitted.
        legend (bool): Draw a per-class colour legend when True.
        ax (Axes | None): Existing axes to draw into; a new figure is created when None.

    Returns:
        tuple[Figure, Axes]: The owning figure and the drawn axes.

    Raises:
        ValueError: If ``values`` is empty or ``class_names`` is shorter than the
            number of values.
    """
    apply_style()
    scores = to_array(values).reshape(-1)
    if scores.size == 0:
        raise ValueError("values must contain at least one metric value")
    class_count = int(scores.size)
    names = _class_display_names(class_count, class_names)
    figure, ax = _resolve_axes(ax, figsize=(8.5, 0.5 * class_count + 1.5))
    positions = np.arange(class_count, dtype=np.float64)
    colours = [PALETTE[index % len(PALETTE)] for index in range(class_count)]
    ax.barh(positions, scores, color=colours, alpha=0.9, height=0.65)
    span = float(scores.max() - scores.min())
    offset = 0.02 * (span if span > 0.0 else max(abs(float(scores.max())), 1.0))
    for position, score in zip(positions, scores, strict=True):
        ax.text(float(score) + offset, position, f"{float(score):.3f}", va="center", fontsize=9)
    ax.set_yticks(positions, labels=names)
    ax.invert_yaxis()
    ax.set_xlabel(metric_name)
    if legend:
        handles = [
            Patch(facecolor=colour, label=name) for colour, name in zip(colours, names, strict=True)
        ]
        ax.legend(handles=handles, loc="best")
    ax.set_title(title if title is not None else f"{metric_name} per class")
    return figure, ax


def plot_metric_distributions(  # noqa: PLR0913
    values_by_arm: Sequence[ArrayLike],
    *,
    arm_names: Sequence[str] | None = None,
    metric_name: str = "macro-F1",
    title: str | None = None,
    legend: bool = True,
    show_points: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Render per-arm metric distributions as violins with optional raw points.

    Each arm becomes one violin in :data:`SERIES_PALETTE` colours with a transparent
    face so overlapping arms remain readable. Jittered raw points are overlaid from a
    fixed internal seed, so repeated calls produce identical scatter positions.

    Args:
        values_by_arm (Sequence[ArrayLike]): One 1D sample array per arm, typically
            per-seed metric values.
        arm_names (Sequence[str] | None): Arm names for the category axis and legend.
        metric_name (str): Metric name used for the value axis label.
        title (str | None): Axes title; a descriptive default is used when omitted.
        legend (bool): Draw a legend of arm proxy artists when True.
        show_points (bool): Overlay jittered raw sample points when True.
        ax (Axes | None): Existing axes to draw into; a new figure is created when None.

    Returns:
        tuple[Figure, Axes]: The owning figure and the drawn axes.

    Raises:
        ValueError: If no arms are given or any arm holds no values.
    """
    apply_style()
    datasets = [to_array(values).reshape(-1) for values in values_by_arm]
    if not datasets:
        raise ValueError("values_by_arm must contain at least one arm")
    if any(dataset.size == 0 for dataset in datasets):
        raise ValueError("each arm must contain at least one value")
    arm_count = len(datasets)
    names = _display_names(arm_count, arm_names, "arm")
    figure, ax = _resolve_axes(ax, figsize=(8.5, 5.2))
    positions = np.arange(1, arm_count + 1, dtype=np.float64)
    colours = [SERIES_PALETTE[index % len(SERIES_PALETTE)] for index in range(arm_count)]
    parts = ax.violinplot(datasets, positions=positions, showmeans=True, showextrema=False)
    bodies = cast("list[PolyCollection]", parts["bodies"])
    for body, colour in zip(bodies, colours, strict=True):
        body.set_facecolor(colour)
        body.set_alpha(0.45)
        body.set_edgecolor(colour)
    if show_points:
        jitter_generator = np.random.default_rng(20240517)
        for position, dataset, colour in zip(positions, datasets, colours, strict=True):
            jitter = jitter_generator.uniform(-0.12, 0.12, size=dataset.size)
            ax.scatter(
                np.full(dataset.size, position) + jitter,
                dataset,
                s=18,
                color=colour,
                alpha=0.8,
                edgecolors="none",
            )
    ax.set_xticks(positions, labels=names)
    ax.set_ylabel(metric_name)
    if legend:
        handles = [
            Patch(facecolor=colour, label=name) for colour, name in zip(colours, names, strict=True)
        ]
        ax.legend(handles=handles, loc="best")
    ax.set_title(title if title is not None else f"{metric_name} distribution per arm")
    return figure, ax


def plot_ablation_bars(
    arm_metrics: Mapping[str, Mapping[str, float]],
    *,
    metric_names: Sequence[str] | None = None,
    title: str | None = None,
    legend: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Render a grouped bar chart comparing experiment arms across metrics.

    Each metric forms one bar group and each arm one series inside the group, so
    ablations such as classifier, autoencoder, hybrid, and compressed hybrid can be
    compared across several metrics at once. Arms missing a metric are drawn as zero.

    Args:
        arm_metrics (Mapping[str, Mapping[str, float]]): Arm name to metric values.
        metric_names (Sequence[str] | None): Metric order to plot; the first-seen union
            across arms is used when omitted.
        title (str | None): Axes title; a descriptive default is used when omitted.
        legend (bool): Draw a legend of arm names when True.
        ax (Axes | None): Existing axes to draw into; a new figure is created when None.

    Returns:
        tuple[Figure, Axes]: The owning figure and the drawn axes.

    Raises:
        ValueError: If ``arm_metrics`` is empty or resolves to zero metrics.
    """
    apply_style()
    if not arm_metrics:
        raise ValueError("arm_metrics must contain at least one arm")
    if metric_names is not None:
        resolved_metrics = [str(name) for name in metric_names]
    else:
        resolved_metrics = list(
            dict.fromkeys(name for metrics in arm_metrics.values() for name in metrics)
        )
    if not resolved_metrics:
        raise ValueError("at least one metric is required")
    metric_count = len(resolved_metrics)
    arm_names = list(arm_metrics.keys())
    arm_count = len(arm_names)
    figure, ax = _resolve_axes(ax, figsize=(8.5, 5.2))
    group_width = 0.8
    bar_width = group_width / arm_count
    metric_positions = np.arange(metric_count, dtype=np.float64)
    for arm_index, arm_name in enumerate(arm_names):
        metrics = arm_metrics[arm_name]
        heights = [float(metrics.get(name, 0.0)) for name in resolved_metrics]
        offsets = metric_positions - group_width / 2.0 + (arm_index + 0.5) * bar_width
        bars = ax.bar(
            offsets,
            heights,
            width=bar_width,
            color=SERIES_PALETTE[arm_index % len(SERIES_PALETTE)],
            alpha=0.9,
            label=arm_name,
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(metric_positions, labels=resolved_metrics)
    ax.set_ylabel("value")
    if legend:
        ax.legend(loc="best")
    ax.set_title(title if title is not None else "Ablation comparison")
    return figure, ax
