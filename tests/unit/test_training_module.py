"""Tests for the mode-aware training module and its objective dispatch."""

from __future__ import annotations

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
from torchmetrics import Metric

from eos.config import ModelConfig, TrainConfig
from eos.model.hybrid_cae import ConvAE, HybridCAE, ModelOutput, PureCNN
from eos.training.lightning_module import (
    TRAINING_MODES,
    EOSTrainingModule,
    TrainingMode,
    build_model,
)


def _configs() -> tuple[ModelConfig, TrainConfig]:
    """Return a small architecture and a short optimizer schedule.

    Returns:
        tuple[ModelConfig, TrainConfig]: Tiny model and trainer configurations.
    """
    model = ModelConfig(
        input_size=64,
        in_channels=3,
        latent_dim=16,
        base_channels=4,
        downsample_stages=4,
        num_classes=3,
        head_hidden=8,
        dropout=0.0,
        recon_alpha=0.85,
        recon_lambda=0.3,
        recon_lambda_final=0.1,
        ssim_window=11,
        ssim_sigma=1.5,
        ssim_boundary="masked",
    )
    train = TrainConfig(
        max_epochs=3,
        learning_rate=1e-3,
        weight_decay=1e-4,
        warmup_epochs=1,
        gradient_clip=1.0,
        precision="32-true",
        accelerator="cpu",
        devices=1,
        seed=0,
    )
    return model, train


def _module(mode: TrainingMode, **train_updates: object) -> EOSTrainingModule:
    """Build a tiny training module for one arm.

    Args:
        mode (TrainingMode): Ablation arm.
        **train_updates (object): Optional ``TrainConfig`` replacements.

    Returns:
        EOSTrainingModule: Configured module.
    """
    model, train = _configs()
    if train_updates:
        train = train.model_copy(update=train_updates)
    return EOSTrainingModule(mode, model, train)


@pytest.mark.parametrize("mode", TRAINING_MODES)
def test_build_model_selects_the_matching_architecture(mode: TrainingMode) -> None:
    """Each arm must own the architecture that exposes its heads."""
    model, _ = _configs()
    built = build_model(mode, model)
    if mode == "hybrid":
        assert isinstance(built, HybridCAE)
    elif mode == "classifier":
        assert isinstance(built, PureCNN)
    else:
        assert isinstance(built, ConvAE)


@pytest.mark.parametrize("mode", TRAINING_MODES)
def test_forward_shapes_follow_the_arm(mode: TrainingMode) -> None:
    """Forward output must match the heads present in the arm."""
    module = _module(mode)
    images = torch.rand(2, 3, 64, 64)
    output = module(images)
    if mode == "hybrid":
        assert isinstance(output, ModelOutput)
        assert output.logits.shape == (2, 3)
        assert output.reconstruction.shape == (2, 3, 64, 64)
        assert output.embedding.shape == (2, 16)
    elif mode == "classifier":
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 3)
    else:
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 3, 64, 64)


@pytest.mark.parametrize("mode", TRAINING_MODES)
def test_training_objective_reaches_the_shared_encoder(mode: TrainingMode) -> None:
    """One batch must produce a finite loss and encoder gradients."""
    module = _module(mode)
    images = torch.rand(2, 3, 64, 64)
    labels = torch.tensor([0, 1])
    total, components = module._training_objective((images, labels))
    assert torch.isfinite(total)
    assert "loss" in components
    total.backward()
    gradient = module.model.encoder.projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


def test_each_arm_logs_only_its_objective_components() -> None:
    """Hybrid, classifier, and autoencoder must expose different components."""
    model, _ = _configs()
    images = torch.rand(2, 3, 64, 64)
    labels = torch.tensor([0, 1])
    hybrid = EOSTrainingModule("hybrid", model, _configs()[1])._training_objective(
        (images, labels)
    )[1]
    assert {"loss", "ce", "recon", "mse", "ssim_loss", "ssim_surrogate"} <= set(hybrid)
    classifier = EOSTrainingModule("classifier", model, _configs()[1])._training_objective(
        (images, labels)
    )[1]
    assert set(classifier) == {"loss", "ce"}
    autoencoder = EOSTrainingModule("autoencoder", model, _configs()[1])._training_objective(
        (images, labels)
    )[1]
    assert {"loss", "recon", "mse", "ssim_loss", "ssim_surrogate"} <= set(autoencoder)


def test_hybrid_reconstruction_is_lambda_weighted() -> None:
    """The hybrid term must equal lambda times the alpha composite."""
    model, train = _configs()
    module = EOSTrainingModule("hybrid", model, train)
    images = torch.rand(2, 3, 64, 64)
    labels = torch.tensor([0, 1])
    _, components = module._training_objective((images, labels))
    expected = model.recon_lambda * (
        model.recon_alpha * components["mse"] + (1.0 - model.recon_alpha) * components["ssim_loss"]
    )
    assert torch.allclose(components["recon"], expected)


def test_autoencoder_objective_is_unweighted() -> None:
    """The autoencoder arm must use lambda equal to one."""
    model, train = _configs()
    module = EOSTrainingModule("autoencoder", model, train)
    images = torch.rand(2, 3, 64, 64)
    labels = torch.tensor([0, 1])
    _, components = module._training_objective((images, labels))
    expected = (
        model.recon_alpha * components["mse"] + (1.0 - model.recon_alpha) * components["ssim_loss"]
    )
    assert torch.allclose(components["recon"], expected)
    assert torch.allclose(components["loss"], components["recon"])


@pytest.mark.parametrize("mode", TRAINING_MODES)
def test_metrics_are_registered_on_the_module(mode: TrainingMode) -> None:
    """Torchmetrics objects must be discoverable as submodules."""
    module = _module(mode)
    names = dict(module.named_modules())
    assert "val_accuracy" in names
    assert "val_f1" in names
    assert "val_recon_mse" in names
    assert "test_accuracy" in names
    assert "test_f1" in names
    assert "test_recon_mse" in names
    assert isinstance(module.val_f1, Metric)
    assert isinstance(module.val_recon_mse, Metric)


def test_optimizer_uses_warmup_then_cosine() -> None:
    """A positive warmup must build a sequential schedule."""
    module = _module("hybrid")
    settings = module.configure_optimizers()
    assert isinstance(settings["optimizer"], AdamW)
    scheduler = settings["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, SequentialLR)


def test_optimizer_without_warmup_uses_cosine_directly() -> None:
    """A zero warmup must skip the linear stage."""
    module = _module("hybrid", warmup_epochs=0)
    settings = module.configure_optimizers()
    scheduler = settings["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, CosineAnnealingLR)


def test_empty_decay_interval_is_rejected() -> None:
    """A bypassed config that leaves no decay epoch must fail clearly."""
    module = _module("hybrid")
    module.train_config = module.train_config.model_copy(
        update={"max_epochs": 1, "warmup_epochs": 1}
    )
    with pytest.raises(ValueError, match="decay epoch"):
        module.configure_optimizers()


def test_schedule_progression_matches_the_epoch_index() -> None:
    """The stored schedule must decay linearly across the configured run."""
    module = _module("hybrid")
    assert module.lambda_schedule.at(0, 3) == pytest.approx(0.3)
    assert module.lambda_schedule.at(1, 3) == pytest.approx(0.2)
    assert module.lambda_schedule.at(2, 3) == pytest.approx(0.1)


def test_non_finite_objective_is_rejected() -> None:
    """NaN or infinite objectives must abort rather than train silently."""
    with pytest.raises(FloatingPointError):
        EOSTrainingModule._assert_finite("loss", torch.tensor(float("nan")))
    with pytest.raises(FloatingPointError):
        EOSTrainingModule._assert_finite("loss", torch.tensor(float("inf")))
    with pytest.raises(ValueError, match="scalar"):
        EOSTrainingModule._assert_finite("loss", torch.zeros(2))


def test_checkpoint_metadata_and_mode_guard() -> None:
    """Checkpoints must record the arm and reject a cross-arm load."""
    module = _module("hybrid")
    checkpoint: dict[str, object] = {}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint["eos_mode"] == "hybrid"
    assert checkpoint["eos_model_config"]["num_classes"] == 3
    classifier = _module("classifier")
    with pytest.raises(ValueError, match="mode"):
        classifier.on_load_checkpoint(checkpoint)
    module.on_load_checkpoint({"eos_mode": "hybrid"})


def test_parameter_count_and_class_weight_validation() -> None:
    """Parameter counts are positive and class weights must match classes."""
    module = _module("hybrid")
    assert module.parameter_count > 0
    with pytest.raises(ValueError, match="class_weights"):
        EOSTrainingModule("hybrid", module.model_config, module.train_config, torch.ones(4))


def test_unknown_mode_is_rejected() -> None:
    """Only the three documented arms are admissible."""
    model, train = _configs()
    with pytest.raises(ValueError, match="mode"):
        EOSTrainingModule("segmentation", model, train)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode"):
        build_model("segmentation", model)  # type: ignore[arg-type]


def test_head_properties_reject_foreign_architectures() -> None:
    """A hybrid module must not expose classifier or autoencoder properties."""
    hybrid = _module("hybrid")
    with pytest.raises(RuntimeError, match="PureCNN"):
        _ = hybrid.classifier
    with pytest.raises(RuntimeError, match="ConvAE"):
        _ = hybrid.autoencoder
    classifier = _module("classifier")
    with pytest.raises(RuntimeError, match="HybridCAE"):
        _ = classifier.hybrid
    with pytest.raises(RuntimeError, match="ConvAE"):
        _ = classifier.autoencoder
    autoencoder = _module("autoencoder")
    with pytest.raises(RuntimeError, match="HybridCAE"):
        _ = autoencoder.hybrid
    with pytest.raises(RuntimeError, match="PureCNN"):
        _ = autoencoder.classifier
