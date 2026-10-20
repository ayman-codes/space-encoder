"""Offline re-evaluation of trained checkpoints.

The notebook deliverable needs per-class metrics, confusion matrices, latent
embeddings, and reconstruction samples from the runs that already exist. This
module derives them from stored checkpoints instead of retraining: it rebuilds
an :class:`~eos.training.lightning_module.EOSTrainingModule` from the provenance
the trainer writes into every checkpoint, then evaluates the model on a
dataloader.

Metric definitions match the training telemetry. Macro-F1 is the unweighted
mean of per-class F1 scores derived from the aggregated confusion matrix, with
absent classes contributing zero. The two structural-similarity numbers are the
same pair the trainer logs: the masked-window value is the training surrogate
and the valid-window value is the canonical reporting quantity. They are
returned under distinct field names and must never be compared to each other
(findings entry F-001).

Evaluation runs in float32 regardless of the precision the model was trained
with, so re-evaluated numbers are comparable across arms.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader

from eos.config import ModelConfig, TrainConfig
from eos.model.hybrid_cae import ConvAE, HybridCAE, PureCNN
from eos.training.lightning_module import (
    TRAINING_MODES,
    EOSTrainingModule,
    TrainingMode,
)

PROVENANCE_KEYS: tuple[str, ...] = ("eos_mode", "eos_model_config", "eos_train_config")
"""Checkpoint keys written by the trainer's checkpoint hook."""

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ImageArray = NDArray[np.uint8]
EmbeddingArray = NDArray[np.float32]


@dataclass(frozen=True)
class ClassificationEvaluation:
    """Classification metrics computed on one split.

    Args:
        accuracy (float): Micro accuracy in ``[0, 1]``.
        macro_f1 (float): Unweighted mean of per-class F1 in ``[0, 1]``.
        per_class_f1 (tuple[float, ...]): F1 per class id.
        confusion (tuple[tuple[int, ...], ...]): Confusion counts with true
            labels on rows and predictions on columns.
        sample_count (int): Number of evaluated samples.
    """

    accuracy: float
    macro_f1: float
    per_class_f1: tuple[float, ...]
    confusion: tuple[tuple[int, ...], ...]
    sample_count: int

    def to_dict(self, class_names: Sequence[str]) -> dict[str, object]:
        """Serialize the evaluation with human-readable class names.

        Args:
            class_names (Sequence[str]): One name per class id.

        Returns:
            dict[str, object]: JSON-ready payload.

        Raises:
            ValueError: If the name count does not match the class count.
        """
        if len(class_names) != len(self.per_class_f1):
            raise ValueError(
                f"expected {len(self.per_class_f1)} class names, received {len(class_names)}"
            )
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class_f1": {
                name: value for name, value in zip(class_names, self.per_class_f1, strict=True)
            },
            "confusion": [list(row) for row in self.confusion],
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class ReconstructionEvaluation:
    """Reconstruction metrics computed on one split.

    Args:
        mse (float): Mean squared error over all pixels.
        ssim_masked (float): Masked-window structural similarity, the training
            surrogate. Comparable only against other masked values.
        ssim_valid (float): Valid-window structural similarity, the canonical
            reporting quantity. Comparable only against other valid values.
        sample_count (int): Number of evaluated samples.
    """

    mse: float
    ssim_masked: float
    ssim_valid: float
    sample_count: int

    def to_dict(self) -> dict[str, object]:
        """Serialize the evaluation.

        Returns:
            dict[str, object]: JSON-ready payload.
        """
        return {
            "mse": self.mse,
            "ssim_masked": self.ssim_masked,
            "ssim_valid": self.ssim_valid,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class EmbeddingBatch:
    """Bottleneck embeddings collected from a split.

    Args:
        embeddings (EmbeddingArray): Array of shape ``(N, latent_dim)`` as
            float32.
        labels (IntArray): Class ids of shape ``(N,)``.
    """

    embeddings: EmbeddingArray
    labels: IntArray


@dataclass(frozen=True)
class ReconstructionSamples:
    """Image pairs collected for rendering.

    Args:
        originals (ImageArray): Reference patches of shape ``(N, H, W, C)``
            as uint8.
        reconstructions (ImageArray): Model outputs in the same layout.
        labels (IntArray): Class ids of shape ``(N,)``.
    """

    originals: ImageArray
    reconstructions: ImageArray
    labels: IntArray


@dataclass(frozen=True)
class _BatchForward:
    """Per-batch forward outputs shared across ablation arms.

    Args:
        logits (Tensor | None): Class logits of shape ``(B, num_classes)`` when
            the arm has a classification head.
        reconstruction (Tensor | None): Reconstruction of shape ``(B, C, H, W)``
            when the arm has a decoder.
        embedding (Tensor): Bottleneck of shape ``(B, latent_dim)``.
    """

    logits: Tensor | None
    reconstruction: Tensor | None
    embedding: Tensor


def _resolve_device(device: torch.device | str | None) -> torch.device:
    """Resolve the evaluation device.

    Args:
        device (torch.device | str | None): Explicit device, or None to pick
            CUDA when available and the CPU otherwise.

    Returns:
        torch.device: Device used for inference.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def _read_checkpoint(path: Path, map_location: str) -> dict[str, object]:
    """Load a checkpoint payload, trusting the local artifact store.

    Args:
        path (Path): Checkpoint file.
        map_location (str): Torch load location mapping.

    Returns:
        dict[str, object]: Raw checkpoint payload.

    Raises:
        ValueError: If the file does not contain a mapping.
    """
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} does not contain a payload mapping")
    return cast(dict[str, object], payload)


def load_module_from_checkpoint(
    path: Path | str,
    *,
    expected_mode: TrainingMode | None = None,
    map_location: str = "cpu",
) -> EOSTrainingModule:
    """Rebuild a training module from a stored checkpoint.

    The trainer writes the ablation arm and the full configuration into every
    checkpoint, so the module is rebuilt from the checkpoint itself rather than
    from an external configuration file.

    Args:
        path (Path | str): Checkpoint file to load.
        expected_mode (TrainingMode | None): When given, refuse checkpoints
            produced by a different arm.
        map_location (str): Torch load location mapping.

    Returns:
        EOSTrainingModule: Module with restored weights in evaluation mode.

    Raises:
        FileNotFoundError: If the checkpoint path does not exist.
        ValueError: If provenance keys are missing, the stored arm is unknown,
            the expected arm does not match, or the payload is not a mapping.
        RuntimeError: If the stored weights do not fit the rebuilt module.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = _read_checkpoint(checkpoint_path, map_location)
    missing = [key for key in PROVENANCE_KEYS if key not in payload]
    if missing:
        raise ValueError(
            f"checkpoint {checkpoint_path} is missing provenance keys {missing}; "
            "it was not produced by the eos trainer"
        )
    stored_mode = payload["eos_mode"]
    if not isinstance(stored_mode, str) or stored_mode not in TRAINING_MODES:
        raise ValueError(f"checkpoint {checkpoint_path} stores unknown arm {stored_mode!r}")
    if expected_mode is not None and stored_mode != expected_mode:
        raise ValueError(
            f"checkpoint was produced by mode {stored_mode!r}, "
            f"but mode {expected_mode!r} was expected"
        )
    mode = stored_mode
    model_config = ModelConfig.model_validate(payload["eos_model_config"])
    train_config = TrainConfig.model_validate(payload["eos_train_config"])
    module = EOSTrainingModule(mode, model_config, train_config)
    state_dict = cast(dict[str, Tensor], payload["state_dict"])
    module.load_state_dict(state_dict)
    module.eval()
    return module


def _metrics_from_confusion(confusion: NDArray[np.int64]) -> tuple[float, float, FloatArray]:
    """Derive accuracy and per-class F1 from a confusion matrix.

    Args:
        confusion (NDArray[np.int64]): Square matrix of shape ``(C, C)`` with
            true labels on rows and predictions on columns.

    Returns:
        tuple[float, float, FloatArray]: Micro accuracy, macro F1, and per-class
            F1 of shape ``(C,)``. Classes without support contribute zero.

    Raises:
        ValueError: If the matrix is empty or not square.
    """
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1] or confusion.shape[0] == 0:
        raise ValueError(f"confusion must be a non-empty square matrix, got {confusion.shape}")
    total = int(confusion.sum())
    if total == 0:
        raise ValueError("confusion matrix contains no samples")
    true_positive = np.diag(confusion).astype(np.float64)
    false_positive = confusion.sum(axis=0).astype(np.float64) - true_positive
    false_negative = confusion.sum(axis=1).astype(np.float64) - true_positive
    denominator = 2.0 * true_positive + false_positive + false_negative
    per_class = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0.0,
    )
    accuracy = float(true_positive.sum() / total)
    return accuracy, float(per_class.mean()), cast(FloatArray, per_class)


def classification_metrics(
    predictions: IntArray,
    targets: IntArray,
    num_classes: int,
) -> ClassificationEvaluation:
    """Compute classification metrics from prediction arrays.

    Args:
        predictions (IntArray): Predicted class ids of shape ``(N,)``.
        targets (IntArray): Ground-truth class ids of shape ``(N,)``.
        num_classes (int): Number of declared classes.

    Returns:
        ClassificationEvaluation: Accuracy, macro F1, per-class F1, and the
            confusion matrix.

    Raises:
        ValueError: If the arrays differ in length, are empty, ``num_classes``
            is not positive, or any label falls outside the class count.
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions and targets must have the same length, "
            f"got {predictions.shape} and {targets.shape}"
        )
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, received {num_classes}")
    if predictions.size == 0:
        raise ValueError("cannot evaluate an empty prediction set")
    for name, values in (("predictions", predictions), ("targets", targets)):
        if int(values.min()) < 0 or int(values.max()) >= num_classes:
            raise ValueError(f"{name} contain labels outside 0..{num_classes - 1}")
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    accuracy, macro_f1, per_class = _metrics_from_confusion(confusion)
    return ClassificationEvaluation(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_f1=tuple(float(value) for value in per_class),
        confusion=tuple(tuple(int(value) for value in row) for row in confusion),
        sample_count=int(targets.size),
    )


def _forward_batch(model: HybridCAE | PureCNN | ConvAE, images: Tensor) -> _BatchForward:
    """Run one batch through any ablation arm.

    Args:
        model (HybridCAE | PureCNN | ConvAE): Architecture of the arm.
        images (Tensor): Input batch of shape ``(B, C, H, W)`` in ``[0, 1]``.

    Returns:
        _BatchForward: Logits and reconstructions where the arm provides them,
            always including the bottleneck embedding.
    """
    if isinstance(model, HybridCAE):
        output = model(images)
        return _BatchForward(output.logits, output.reconstruction, output.embedding)
    embedding = model.encoder(images)
    if isinstance(model, PureCNN):
        return _BatchForward(model.classifier(embedding), None, embedding)
    return _BatchForward(None, model.decoder(embedding), embedding)


def evaluate_classification(
    module: EOSTrainingModule,
    loader: DataLoader[tuple[Tensor, Tensor]],
    *,
    device: torch.device | str | None = None,
) -> ClassificationEvaluation:
    """Evaluate the classification head on a split.

    Args:
        module (EOSTrainingModule): Reloaded training module.
        loader (DataLoader[tuple[Tensor, Tensor]]): Batches of images and labels.
        device (torch.device | str | None): Inference device, or None for
            automatic selection.

    Returns:
        ClassificationEvaluation: Accuracy, macro F1, per-class F1, and
            confusion matrix over the whole split.

    Raises:
        ValueError: If the arm has no classification head or the loader yields
            no samples.
    """
    if isinstance(module.model, ConvAE):
        raise ValueError("the autoencoder arm has no classification head to evaluate")
    resolved = _resolve_device(device)
    module.to(resolved)
    module.eval()
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            forward = _forward_batch(module.model, images.to(resolved))
            if forward.logits is None:
                raise ValueError("the autoencoder arm has no classification head to evaluate")
            predictions.append(forward.logits.argmax(dim=1).detach().cpu())
            targets.append(labels.detach().cpu())
    if not predictions:
        raise ValueError("the loader produced no samples")
    prediction_array = cast(IntArray, torch.cat(predictions).numpy().astype(np.int64))
    target_array = cast(IntArray, torch.cat(targets).numpy().astype(np.int64))
    return classification_metrics(prediction_array, target_array, module.model_config.num_classes)


def evaluate_reconstruction(
    module: EOSTrainingModule,
    loader: DataLoader[tuple[Tensor, Tensor]],
    *,
    device: torch.device | str | None = None,
) -> ReconstructionEvaluation:
    """Evaluate the reconstruction head on a split.

    The structural similarities come from the two loss instances the trainer
    configured, so the values are definitionally identical to the logged
    ``test/ssim_masked`` and ``test/ssim_valid`` quantities.

    Args:
        module (EOSTrainingModule): Reloaded training module.
        loader (DataLoader[tuple[Tensor, Tensor]]): Batches of images and labels.
        device (torch.device | str | None): Inference device, or None for
            automatic selection.

    Returns:
        ReconstructionEvaluation: MSE, masked SSIM, valid SSIM, and the sample
            count.

    Raises:
        ValueError: If the arm has no decoder or the loader yields no samples.
    """
    if isinstance(module.model, PureCNN):
        raise ValueError("the classifier arm has no reconstruction head to evaluate")
    resolved = _resolve_device(device)
    module.to(resolved)
    module.eval()
    squared_error = 0.0
    pixel_count = 0
    masked_total = 0.0
    valid_total = 0.0
    sample_count = 0
    with torch.no_grad():
        for images, _ in loader:
            batch = images.to(resolved)
            forward = _forward_batch(module.model, batch)
            if forward.reconstruction is None:
                raise ValueError("the classifier arm has no reconstruction head to evaluate")
            reconstruction = forward.reconstruction
            batch_size = int(batch.shape[0])
            squared_error += float((reconstruction - batch).pow(2).sum().item())
            pixel_count += int(batch.numel())
            masked_total += (
                float(module.loss_fn.ssim_loss(reconstruction, batch).item()) * batch_size
            )
            valid_total += float(module.reporting_ssim(reconstruction, batch).item()) * batch_size
            sample_count += batch_size
    if sample_count == 0:
        raise ValueError("the loader produced no samples")
    return ReconstructionEvaluation(
        mse=squared_error / pixel_count,
        ssim_masked=1.0 - masked_total / sample_count,
        ssim_valid=1.0 - valid_total / sample_count,
        sample_count=sample_count,
    )


def collect_embeddings(
    module: EOSTrainingModule,
    loader: DataLoader[tuple[Tensor, Tensor]],
    *,
    max_per_class: int,
    device: torch.device | str | None = None,
) -> EmbeddingBatch:
    """Collect bottleneck embeddings, balanced across classes.

    Samples are taken in loader order until each class reaches the cap, which
    makes the selection deterministic for a deterministic loader.

    Args:
        module (EOSTrainingModule): Reloaded training module.
        loader (DataLoader[tuple[Tensor, Tensor]]): Batches of images and labels.
        max_per_class (int): Maximum samples kept per class.
        device (torch.device | str | None): Inference device, or None for
            automatic selection.

    Returns:
        EmbeddingBatch: Float32 embeddings of shape ``(N, latent_dim)`` and
            class ids of shape ``(N,)``.

    Raises:
        ValueError: If ``max_per_class`` is not positive or the loader yields
            no samples.
    """
    if max_per_class < 1:
        raise ValueError(f"max_per_class must be positive, received {max_per_class}")
    resolved = _resolve_device(device)
    module.to(resolved)
    module.eval()
    kept_embeddings: list[Tensor] = []
    kept_labels: list[Tensor] = []
    counts: dict[int, int] = {}
    with torch.no_grad():
        for images, labels in loader:
            forward = _forward_batch(module.model, images.to(resolved))
            cpu_labels = labels.detach().cpu()
            keep: list[int] = []
            for index, class_id in enumerate(cpu_labels.tolist()):
                seen = counts.get(int(class_id), 0)
                if seen < max_per_class:
                    counts[int(class_id)] = seen + 1
                    keep.append(index)
            if keep:
                kept_embeddings.append(forward.embedding[keep].detach().cpu())
                kept_labels.append(cpu_labels[keep])
    if not kept_embeddings:
        raise ValueError("the loader produced no samples")
    embeddings = cast(EmbeddingArray, torch.cat(kept_embeddings).numpy().astype(np.float32))
    collected = cast(IntArray, torch.cat(kept_labels).numpy().astype(np.int64))
    return EmbeddingBatch(embeddings=embeddings, labels=collected)


def _to_uint8(images: Tensor) -> ImageArray:
    """Convert a float image batch in ``[0, 1]`` to NHWC uint8.

    Args:
        images (Tensor): Batch of shape ``(B, C, H, W)``.

    Returns:
        ImageArray: Array of shape ``(B, H, W, C)`` as uint8.
    """
    scaled = (images.detach().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return cast(ImageArray, scaled.numpy().transpose(0, 2, 3, 1))


def collect_recon_samples(
    module: EOSTrainingModule,
    loader: DataLoader[tuple[Tensor, Tensor]],
    *,
    per_class: int,
    device: torch.device | str | None = None,
) -> ReconstructionSamples:
    """Collect original and reconstructed patches, balanced across classes.

    Args:
        module (EOSTrainingModule): Reloaded training module.
        loader (DataLoader[tuple[Tensor, Tensor]]): Batches of images and labels.
        per_class (int): Maximum samples kept per class.
        device (torch.device | str | None): Inference device, or None for
            automatic selection.

    Returns:
        ReconstructionSamples: Original and reconstructed uint8 patches of
            shape ``(N, H, W, C)`` with class ids of shape ``(N,)``.

    Raises:
        ValueError: If the arm has no decoder, ``per_class`` is not positive,
            or the loader yields no samples.
    """
    if isinstance(module.model, PureCNN):
        raise ValueError("the classifier arm has no reconstruction head to evaluate")
    if per_class < 1:
        raise ValueError(f"per_class must be positive, received {per_class}")
    resolved = _resolve_device(device)
    module.to(resolved)
    module.eval()
    kept_originals: list[Tensor] = []
    kept_reconstructions: list[Tensor] = []
    kept_labels: list[Tensor] = []
    counts: dict[int, int] = {}
    with torch.no_grad():
        for images, labels in loader:
            batch = images.to(resolved)
            forward = _forward_batch(module.model, batch)
            if forward.reconstruction is None:
                raise ValueError("the classifier arm has no reconstruction head to evaluate")
            cpu_labels = labels.detach().cpu()
            keep: list[int] = []
            for index, class_id in enumerate(cpu_labels.tolist()):
                seen = counts.get(int(class_id), 0)
                if seen < per_class:
                    counts[int(class_id)] = seen + 1
                    keep.append(index)
            if keep:
                kept_originals.append(batch[keep].detach().cpu())
                kept_reconstructions.append(forward.reconstruction[keep].detach().cpu())
                kept_labels.append(cpu_labels[keep])
    if not kept_originals:
        raise ValueError("the loader produced no samples")
    originals = _to_uint8(torch.cat(kept_originals))
    reconstructions = _to_uint8(torch.cat(kept_reconstructions))
    collected = cast(IntArray, torch.cat(kept_labels).numpy().astype(np.int64))
    return ReconstructionSamples(
        originals=originals,
        reconstructions=reconstructions,
        labels=collected,
    )
