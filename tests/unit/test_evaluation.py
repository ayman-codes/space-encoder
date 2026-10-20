"""Tests for checkpoint reload and offline re-evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from tests.helpers import build_curated_dataset

from eos.config import DataConfig, ModelConfig, TrainConfig
from eos.dataset.eurosat_datamodule import EuroSATDataModule
from eos.evaluation import (
    ClassificationEvaluation,
    EmbeddingBatch,
    ReconstructionEvaluation,
    ReconstructionSamples,
    classification_metrics,
    collect_embeddings,
    collect_recon_samples,
    evaluate_classification,
    evaluate_reconstruction,
    load_module_from_checkpoint,
)
from eos.training.lightning_module import EOSTrainingModule, TrainingMode


def _model_config() -> ModelConfig:
    """Build the miniature architecture used by orchestration tests.

    Returns:
        ModelConfig: Small validated architecture configuration.
    """
    return ModelConfig(
        input_size=64,
        in_channels=3,
        latent_dim=16,
        base_channels=4,
        downsample_stages=4,
        num_classes=2,
        head_hidden=8,
        dropout=0.0,
        recon_alpha=0.85,
        recon_lambda=0.3,
        recon_lambda_final=0.1,
    )


def _train_config() -> TrainConfig:
    """Build a CPU training configuration for evaluation tests.

    Returns:
        TrainConfig: Validated training configuration.
    """
    return TrainConfig(
        max_epochs=1,
        warmup_epochs=0,
        accelerator="cpu",
        precision="32-true",
        seed=42,
    )


def _build_module(mode: TrainingMode = "hybrid") -> EOSTrainingModule:
    """Assemble one miniature training module.

    Args:
        mode (TrainingMode): Ablation arm to build.

    Returns:
        EOSTrainingModule: Freshly initialized module.
    """
    torch.manual_seed(0)
    return EOSTrainingModule(mode, _model_config(), _train_config())


def _save_checkpoint(module: EOSTrainingModule, path: Path) -> None:
    """Persist a checkpoint exactly as the trainer does.

    Args:
        module (EOSTrainingModule): Module to serialize.
        path (Path): Destination checkpoint path.

    Returns:
        None: The checkpoint file is written.
    """
    payload: dict[str, object] = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _test_loader(root: Path) -> torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build the test dataloader of a tiny curated dataset.

    Args:
        root (Path): Directory that receives the synthetic dataset.

    Returns:
        DataLoader[tuple[Tensor, Tensor]]: Test split batches of images and
            labels.
    """
    dataset = build_curated_dataset(root, per_class=24)
    datamodule = EuroSATDataModule(
        DataConfig(
            curated_root=dataset,
            image_size=64,
            channels=3,
            num_classes=2,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
    )
    datamodule.setup()
    return datamodule.test_dataloader()


def test_load_module_round_trip(tmp_path: Path) -> None:
    """A saved checkpoint must reload into an equivalent module.

    Args:
        tmp_path (Path): Isolated output root.

    Returns:
        None: Assertions verify mode, provenance, and weight equality.
    """
    module = _build_module("hybrid")
    checkpoint = tmp_path / "hybrid-0.ckpt"
    _save_checkpoint(module, checkpoint)
    restored = load_module_from_checkpoint(checkpoint)
    assert isinstance(restored, EOSTrainingModule)
    assert restored.mode == "hybrid"
    original = dict(module.model.state_dict())
    reloaded = dict(restored.model.state_dict())
    assert original.keys() == reloaded.keys()
    for name, value in original.items():
        assert torch.equal(value, reloaded[name])


def test_load_missing_checkpoint_raises(tmp_path: Path) -> None:
    """A missing checkpoint must fail with a clear error.

    Args:
        tmp_path (Path): Isolated output root.

    Returns:
        None: Assertions verify the exception type.

    Raises:
        FileNotFoundError: Expected from the loader.
    """
    with pytest.raises(FileNotFoundError):
        load_module_from_checkpoint(tmp_path / "absent.ckpt")


def test_load_rejects_checkpoint_without_provenance(tmp_path: Path) -> None:
    """A checkpoint without stored provenance cannot be rebuilt.

    Args:
        tmp_path (Path): Isolated output root.

    Returns:
        None: Assertions verify the exception type.

    Raises:
        ValueError: Expected from the loader.
    """
    path = tmp_path / "bare.ckpt"
    torch.save({"state_dict": _build_module().model.state_dict()}, path)
    with pytest.raises(ValueError, match="provenance"):
        load_module_from_checkpoint(path)


def test_load_rejects_mismatched_mode(tmp_path: Path) -> None:
    """A checkpoint from another arm must be refused when a mode is expected.

    Args:
        tmp_path (Path): Isolated output root.

    Returns:
        None: Assertions verify the exception type.

    Raises:
        ValueError: Expected from the loader.
    """
    checkpoint = tmp_path / "hybrid-0.ckpt"
    _save_checkpoint(_build_module("hybrid"), checkpoint)
    with pytest.raises(ValueError, match="produced by mode"):
        load_module_from_checkpoint(checkpoint, expected_mode="classifier")


def test_classification_metrics_perfect_predictions() -> None:
    """Perfect predictions must yield unit metrics and a diagonal confusion.

    Args:
        None: This test takes no arguments.

    Returns:
        None: Assertions verify the metric values.
    """
    targets = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    result = classification_metrics(targets.copy(), targets, num_classes=3)
    assert result.accuracy == pytest.approx(1.0)
    assert result.macro_f1 == pytest.approx(1.0)
    assert result.per_class_f1 == (1.0, 1.0, 1.0)
    assert result.confusion == ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    assert result.sample_count == 6


def test_classification_metrics_known_answer() -> None:
    """Per-class F1 must match a hand-computed confusion case.

    Args:
        None: This test takes no arguments.

    Returns:
        None: Assertions verify hand-computed per-class values.
    """
    targets = np.array([0, 0, 1, 1], dtype=np.int64)
    predictions = np.array([0, 1, 1, 1], dtype=np.int64)
    result = classification_metrics(predictions, targets, num_classes=2)
    assert result.per_class_f1[0] == pytest.approx(2 * (0.5 * 1.0) / 1.5)
    assert result.per_class_f1[1] == pytest.approx(2 * (2 / 3 * 1.0) / (5 / 3))
    assert result.macro_f1 == pytest.approx(sum(result.per_class_f1) / 2)
    assert result.confusion == ((1, 1), (0, 2))
    assert result.accuracy == pytest.approx(0.75)


def test_classification_metrics_rejects_mismatched_shapes() -> None:
    """Prediction and target arrays must have the same length.

    Args:
        None: This test takes no arguments.

    Returns:
        None: Assertions verify the exception.

    Raises:
        ValueError: Expected from the metrics helper.
    """
    with pytest.raises(ValueError, match="same length"):
        classification_metrics(np.array([0], dtype=np.int64), np.zeros(2, dtype=np.int64), 2)


def test_classification_metrics_rejects_unknown_class() -> None:
    """Labels outside the declared class count must be refused.

    Args:
        None: This test takes no arguments.

    Returns:
        None: Assertions verify the exception.

    Raises:
        ValueError: Expected from the metrics helper.
    """
    with pytest.raises(ValueError, match="outside"):
        classification_metrics(np.array([3], dtype=np.int64), np.array([0], dtype=np.int64), 2)


def test_evaluate_classification_end_to_end(tmp_path: Path) -> None:
    """A real module must produce complete classification metrics.

    Args:
        tmp_path (Path): Isolated output root.

    Returns:
        None: Assertions verify counts, ranges, and shapes.
    """
    loader = _test_loader(tmp_path)
    result = evaluate_classification(_build_module("hybrid"), loader)
    assert isinstance(result, ClassificationEvaluation)
    assert result.sample_count > 0
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
    assert len(result.per_class_f1) == 2
    assert len(result.confusion) == 2
    assert sum(sum(row) for row in result.confusion) == result.sample_count


def test_evaluate_classification_rejects_autoencoder(tmp_path: Path) -> None:
    """The autoencoder arm has no logits and cannot be classified.

    Args:
        tmp_path (Path): Isolated output root.

    Returns:
        None: Assertions verify the exception.

    Raises:
        ValueError: Expected from the evaluator.
    """
    loader = _test_loader(tmp_path)
    with pytest.raises(ValueError, match="no classification"):
        evaluate_classification(_build_module("autoencoder"), loader)


def test_evaluate_reconstruction_end_to_end(tmp_path: Path) -> None:
    """A real module must produce reconstruction metrics per policy.

    Args:
        tmp_path (Path): Isolated output root.

    Returns:
        None: Assertions verify metric ranges and counts.
    """
    loader = _test_loader(tmp_path)
    result = evaluate_reconstruction(_build_module("hybrid"), loader)
    assert isinstance(result, ReconstructionEvaluation)
    assert result.mse > 0.0
    assert np.isfinite(result.mse)
    assert result.ssim_masked <= 1.0
    assert result.ssim_valid <= 1.0
    assert np.isfinite(result.ssim_masked)
    assert np.isfinite(result.ssim_valid)
    assert result.sample_count > 0


def test_evaluate_reconstruction_rejects_classifier(tmp_path: Path) -> None:
    """The classifier arm has no decoder and cannot reconstruct.

    Args:
        tmp_path (Path): Isolated output root.

    Returns:
        None: Assertions verify the exception.

    Raises:
        ValueError: Expected from the evaluator.
    """
    loader = _test_loader(tmp_path)
    with pytest.raises(ValueError, match="no reconstruction"):
        evaluate_reconstruction(_build_module("classifier"), loader)


def test_collect_embeddings_shapes_and_determinism(tmp_path: Path) -> None:
    """Embedding collection must be balanced, shaped, and deterministic.

    Args:
        tmp_path (Path): Isolated output root.

    Returns:
        None: Assertions verify shapes, per-class caps, and repeatability.
    """
    loader = _test_loader(tmp_path)
    module = _build_module("hybrid")
    first = collect_embeddings(module, loader, max_per_class=2)
    second = collect_embeddings(module, loader, max_per_class=2)
    assert isinstance(first, EmbeddingBatch)
    assert first.embeddings.shape[1] == 16
    assert first.embeddings.shape[0] == first.labels.shape[0]
    assert first.embeddings.dtype == np.float32
    counts = np.bincount(first.labels, minlength=2)
    assert counts.max() <= 2
    assert np.allclose(first.embeddings, second.embeddings)
    assert np.array_equal(first.labels, second.labels)


def test_collect_embeddings_rejects_nonpositive_cap() -> None:
    """A non-positive per-class cap must be refused.

    Args:
        None: This test takes no arguments.

    Returns:
        None: Assertions verify the exception.

    Raises:
        ValueError: Expected from the collector.
    """
    with pytest.raises(ValueError, match="positive"):
        collect_embeddings(_build_module("hybrid"), [], max_per_class=0)


def test_collect_recon_samples_shapes_and_cap(tmp_path: Path) -> None:
    """Sample collection must respect the per-class cap and store uint8.

    Args:
        tmp_path (Path): Isolated output root.

    Returns:
        None: Assertions verify shapes, caps, and dtypes.
    """
    loader = _test_loader(tmp_path)
    module = _build_module("hybrid")
    samples = collect_recon_samples(module, loader, per_class=1)
    assert isinstance(samples, ReconstructionSamples)
    assert samples.originals.dtype == np.uint8
    assert samples.reconstructions.dtype == np.uint8
    assert samples.originals.shape == samples.reconstructions.shape
    assert samples.originals.ndim == 4
    assert samples.originals.shape[0] == samples.labels.shape[0]
    counts = np.bincount(samples.labels, minlength=2)
    assert counts.max() <= 1
    assert samples.originals.max() <= 255


def test_collect_recon_samples_rejects_classifier(tmp_path: Path) -> None:
    """The classifier arm cannot provide reconstructions.

    Args:
        tmp_path (Path): Isolated output root.

    Returns:
        None: Assertions verify the exception.

    Raises:
        ValueError: Expected from the collector.
    """
    loader = _test_loader(tmp_path)
    with pytest.raises(ValueError, match="no reconstruction"):
        collect_recon_samples(_build_module("classifier"), loader, per_class=1)


def test_evaluation_serialization() -> None:
    """Serializers must emit stable JSON-ready dictionaries.

    Args:
        None: This test takes no arguments.

    Returns:
        None: Assertions verify keys and values.
    """
    classification = ClassificationEvaluation(
        accuracy=0.9,
        macro_f1=0.8,
        per_class_f1=(1.0, 0.6),
        confusion=((1, 0), (1, 2)),
        sample_count=4,
    )
    payload = classification.to_dict(("Alpha", "Beta"))
    assert payload["accuracy"] == 0.9
    assert payload["macro_f1"] == 0.8
    assert payload["per_class_f1"] == {"Alpha": 1.0, "Beta": 0.6}
    assert payload["confusion"] == [[1, 0], [1, 2]]
    reconstruction = ReconstructionEvaluation(
        mse=0.01,
        ssim_masked=0.7,
        ssim_valid=0.8,
        sample_count=4,
    )
    assert reconstruction.to_dict() == {
        "mse": 0.01,
        "ssim_masked": 0.7,
        "ssim_valid": 0.8,
        "sample_count": 4,
    }
