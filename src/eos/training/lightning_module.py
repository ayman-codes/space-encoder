"""Single LightningModule shared by the three ablation arms.

The module owns the optimisation lifecycle, metrics, and the scheduled
reconstruction weight. The only difference between ``hybrid``, ``classifier``,
and ``autoencoder`` is the objective that produces the backward signal:

- ``hybrid``: cross-entropy plus the scheduled reconstruction composite;
- ``classifier``: cross-entropy only;
- ``autoencoder``: the unweighted reconstruction composite, because there is no
  cross-entropy term for ``lambda`` to balance.

The architecture itself stays in :mod:`eos.model`; this module never changes the
layers.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR
from torchmetrics.classification import F1Score, MulticlassAccuracy
from torchmetrics.regression import MeanSquaredError

from eos.config import ModelConfig, TrainConfig
from eos.model.hybrid_cae import ConvAE, HybridCAE, ModelOutput, PureCNN
from eos.model.losses import MultiTaskLoss, SSIMLoss
from eos.training.lambda_schedule import ReconstructionLambdaSchedule

TrainingMode = Literal["hybrid", "classifier", "autoencoder"]

TRAINING_MODES: tuple[TrainingMode, ...] = ("hybrid", "classifier", "autoencoder")

WARMUP_START_FACTOR = 0.1
"""Fraction of the peak learning rate used at the first warmup epoch."""

MINIMUM_LR_FACTOR = 0.01
"""Cosine floor as a fraction of the peak learning rate."""


def build_model(mode: TrainingMode, config: ModelConfig) -> HybridCAE | PureCNN | ConvAE:
    """Instantiate the architecture that matches an ablation arm.

    Args:
        mode (TrainingMode): Ablation arm selector.
        config (ModelConfig): Architecture configuration.

    Returns:
        HybridCAE | PureCNN | ConvAE: Model with the heads required by the mode.

    Raises:
        ValueError: If ``mode`` is not a supported arm.
    """
    if mode == "hybrid":
        return HybridCAE(config)
    if mode == "classifier":
        return PureCNN(config)
    if mode == "autoencoder":
        return ConvAE(config)
    raise ValueError(f"unsupported training mode {mode!r}")


class EOSTrainingModule(LightningModule):
    """Train one ablation arm with a shared optimisation and metric contract.

    Args:
        mode (TrainingMode): One of ``hybrid``, ``classifier``, or ``autoencoder``.
        model_config (ModelConfig): Architecture and objective configuration.
        train_config (TrainConfig): Optimizer, schedule, and output settings.
        class_weights (Tensor | None): Optional per-class weights of shape
            ``(num_classes,)`` for inverse-frequency cross-entropy. Ignored by
            the autoencoder arm.

    Raises:
        ValueError: If ``mode`` is unknown or the class-weight shape does not
            match ``num_classes``.
    """

    def __init__(
        self,
        mode: TrainingMode,
        model_config: ModelConfig,
        train_config: TrainConfig,
        class_weights: Tensor | None = None,
    ) -> None:
        """Build the model, objective, metrics, and weight schedule.

        Args:
            mode (TrainingMode): Ablation arm selector.
            model_config (ModelConfig): Architecture and objective configuration.
            train_config (TrainConfig): Optimizer and output configuration.
            class_weights (Tensor | None): Optional class weights of shape
                ``(num_classes,)``.

        Returns:
            None: Components are registered on the module.

        Raises:
            ValueError: If the mode or class-weight geometry is invalid.
        """
        super().__init__()
        if mode not in TRAINING_MODES:
            raise ValueError(f"mode must be one of {TRAINING_MODES}, received {mode!r}")
        if class_weights is not None and class_weights.shape != (model_config.num_classes,):
            raise ValueError(
                f"class_weights must have shape ({model_config.num_classes},), "
                f"received {tuple(class_weights.shape)}"
            )
        self.mode: TrainingMode = mode
        self.model_config = model_config
        self.train_config = train_config
        self.model = build_model(mode, model_config)
        training_ssim = SSIMLoss(
            window_size=model_config.ssim_window,
            sigma=model_config.ssim_sigma,
            boundary=model_config.ssim_boundary,
        )
        weights = None if mode == "autoencoder" else class_weights
        self.loss_fn = MultiTaskLoss(
            alpha=model_config.recon_alpha,
            recon_lambda=model_config.recon_lambda,
            ssim_loss=training_ssim,
            class_weights=weights,
        )
        self.reporting_ssim = SSIMLoss(
            window_size=model_config.ssim_window,
            sigma=model_config.ssim_sigma,
            boundary="valid",
        )
        self.lambda_schedule = ReconstructionLambdaSchedule(
            start=model_config.recon_lambda,
            final=model_config.recon_lambda_final,
        )
        self._current_lambda = model_config.recon_lambda
        self.val_accuracy = MulticlassAccuracy(
            num_classes=model_config.num_classes,
            average="micro",
        )
        self.val_f1 = F1Score(
            task="multiclass",
            num_classes=model_config.num_classes,
            average="macro",
        )
        self.test_accuracy = MulticlassAccuracy(
            num_classes=model_config.num_classes,
            average="micro",
        )
        self.test_f1 = F1Score(
            task="multiclass",
            num_classes=model_config.num_classes,
            average="macro",
        )
        self.val_recon_mse = MeanSquaredError()
        self.test_recon_mse = MeanSquaredError()

    @property
    def hybrid(self) -> HybridCAE:
        """Return the hybrid model.

        Returns:
            HybridCAE: Shared-encoder multi-task model.

        Raises:
            RuntimeError: If the module was built for another arm.
        """
        if not isinstance(self.model, HybridCAE):
            raise RuntimeError("this module does not own a HybridCAE")
        return self.model

    @property
    def classifier(self) -> PureCNN:
        """Return the pure classifier model.

        Returns:
            PureCNN: Encoder plus classification head.

        Raises:
            RuntimeError: If the module was built for another arm.
        """
        if not isinstance(self.model, PureCNN):
            raise RuntimeError("this module does not own a PureCNN")
        return self.model

    @property
    def autoencoder(self) -> ConvAE:
        """Return the pure convolutional autoencoder.

        Returns:
            ConvAE: Encoder plus reconstruction decoder.

        Raises:
            RuntimeError: If the module was built for another arm.
        """
        if not isinstance(self.model, ConvAE):
            raise RuntimeError("this module does not own a ConvAE")
        return self.model

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable parameters.

        Returns:
            int: Sum of ``numel()`` over parameters with gradients enabled.
        """
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, x: Tensor) -> ModelOutput | Tensor:
        """Run the underlying model for one batch.

        Args:
            x (Tensor): Input batch of shape ``(B, C, H, W)`` in ``[0, 1]``.

        Returns:
            ModelOutput | Tensor: Hybrid output for ``hybrid``, logits for
                ``classifier``, and a reconstruction for ``autoencoder``.
        """
        return cast(ModelOutput | Tensor, self.model(x))

    @staticmethod
    def _assert_finite(name: str, value: Tensor) -> None:
        """Reject non-finite objective values.

        Args:
            name (str): Metric or component name used in the error message.
            value (Tensor): Candidate scalar tensor.

        Returns:
            None: The check passes silently.

        Raises:
            ValueError: If the tensor is not scalar.
            FloatingPointError: If any element is NaN or infinite.
        """
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar tensor, received shape {tuple(value.shape)}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} is not finite: {float(value.detach().cpu())}")

    def _training_objective(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the forward objective for one training batch.

        Args:
            batch (tuple[Tensor, Tensor]): Image batch of shape ``(B, C, H, W)``
                in ``[0, 1]`` and labels of shape ``(B,)``.

        Returns:
            tuple[Tensor, dict[str, Tensor]]: Scalar objective and its named
                additive components.

        Raises:
            FloatingPointError: If the objective is not finite.
        """
        images, labels = batch
        components: dict[str, Tensor]
        if self.mode == "hybrid":
            output = self.hybrid(images)
            breakdown = self.loss_fn(
                output.logits,
                labels,
                output.reconstruction,
                images,
                recon_lambda=self._current_lambda,
            )
            total = breakdown.total
            components = {
                "loss": breakdown.total,
                "ce": breakdown.classification,
                "recon": breakdown.reconstruction,
                "mse": breakdown.mse,
                "ssim_loss": breakdown.ssim,
                "ssim_surrogate": 1.0 - breakdown.ssim,
            }
        elif self.mode == "classifier":
            logits = self.classifier(images)
            cross_entropy = self.loss_fn.classification_term(logits, labels)
            total = cross_entropy
            components = {"loss": cross_entropy, "ce": cross_entropy}
        else:
            reconstruction = self.autoencoder(images)
            mse, ssim, weighted = self.loss_fn.reconstruction_term(
                reconstruction,
                images,
                recon_lambda=1.0,
            )
            total = weighted
            components = {
                "loss": weighted,
                "recon": weighted,
                "mse": mse,
                "ssim_loss": ssim,
                "ssim_surrogate": 1.0 - ssim,
            }
        self._assert_finite("total loss", total)
        return total, components

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Run one optimisation step.

        Args:
            batch (tuple[Tensor, Tensor]): Image batch of shape ``(B, C, H, W)``
                in ``[0, 1]`` and labels of shape ``(B,)``.
            batch_idx (int): Batch index supplied by Lightning.

        Returns:
            Tensor: Scalar objective used for backpropagation.

        Raises:
            FloatingPointError: If the objective is not finite.
        """
        del batch_idx
        total, components = self._training_objective(batch)
        batch_size = int(batch[0].shape[0])
        for name, value in components.items():
            self.log(
                f"train/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name == "loss",
                batch_size=batch_size,
            )
        if self.mode != "classifier":
            scheduled = self._current_lambda if self.mode == "hybrid" else 1.0
            self.log(
                "train/recon_lambda",
                float(scheduled),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        return total

    def _evaluation_objective(
        self,
        batch: tuple[Tensor, Tensor],
        stage: Literal["val", "test"],
    ) -> None:
        """Evaluate one batch and log every applicable metric.

        The masked SSIM is the training surrogate; the valid-window SSIM is the
        canonical reporting quantity. They are logged under distinct names and
        must never be compared directly.

        Args:
            batch (tuple[Tensor, Tensor]): Image batch of shape ``(B, C, H, W)``
                in ``[0, 1]`` and labels of shape ``(B,)``.
            stage (Literal["val", "test"]): Metric namespace.

        Returns:
            None: Metrics are logged on the module.

        Raises:
            FloatingPointError: If the evaluation objective is not finite.
        """
        images, labels = batch
        batch_size = int(images.shape[0])
        logits: Tensor | None = None
        reconstruction: Tensor | None = None
        if self.mode == "hybrid":
            output = self.hybrid(images)
            breakdown = self.loss_fn(
                output.logits,
                labels,
                output.reconstruction,
                images,
                recon_lambda=self._current_lambda,
            )
            total = breakdown.total
            logits = output.logits
            reconstruction = output.reconstruction
            self.log(
                f"{stage}/ce",
                breakdown.classification,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        elif self.mode == "classifier":
            logits = self.classifier(images)
            cross_entropy = self.loss_fn.classification_term(logits, labels)
            total = cross_entropy
            self.log(
                f"{stage}/ce",
                cross_entropy,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        else:
            reconstruction = self.autoencoder(images)
            mse, _, weighted = self.loss_fn.reconstruction_term(
                reconstruction,
                images,
                recon_lambda=1.0,
            )
            total = weighted
            self.log(
                f"{stage}/mse",
                mse,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        self._assert_finite(f"{stage} loss", total)
        self.log(
            f"{stage}/loss",
            total,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        accuracy = self.val_accuracy if stage == "val" else self.test_accuracy
        f1_score = self.val_f1 if stage == "val" else self.test_f1
        if logits is not None:
            accuracy(logits.detach(), labels)
            f1_score(logits.detach(), labels)
            self.log(
                f"{stage}/accuracy",
                accuracy,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/macro_f1",
                f1_score,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        if reconstruction is not None:
            mse_metric = self.val_recon_mse if stage == "val" else self.test_recon_mse
            mse_metric(reconstruction.detach(), images.detach())
            self.log(
                f"{stage}/recon_mse",
                mse_metric,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            masked = self.loss_fn.ssim_loss(reconstruction, images)
            valid = self.reporting_ssim(reconstruction, images)
            self.log(
                f"{stage}/ssim_masked",
                1.0 - masked,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/ssim_valid",
                1.0 - valid,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Evaluate one validation batch.

        Args:
            batch (tuple[Tensor, Tensor]): Image batch of shape ``(B, C, H, W)``
                in ``[0, 1]`` and labels of shape ``(B,)``.
            batch_idx (int): Batch index supplied by Lightning.

        Returns:
            None: Metrics are logged on the module.
        """
        del batch_idx
        self._evaluation_objective(batch, "val")

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Evaluate one test batch.

        Args:
            batch (tuple[Tensor, Tensor]): Image batch of shape ``(B, C, H, W)``
                in ``[0, 1]`` and labels of shape ``(B,)``.
            batch_idx (int): Batch index supplied by Lightning.

        Returns:
            None: Metrics are logged on the module.
        """
        del batch_idx
        self._evaluation_objective(batch, "test")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Build AdamW with linear warmup followed by cosine decay.

        Returns:
            OptimizerLRSchedulerConfig: Optimizer and epoch-stepped scheduler.

        Raises:
            ValueError: If the configured schedule leaves no decay interval.
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        total_epochs = self.train_config.max_epochs
        warmup_epochs = self.train_config.warmup_epochs
        decay_epochs = total_epochs - warmup_epochs
        if decay_epochs < 1:
            raise ValueError("the cosine schedule requires at least one decay epoch")
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=decay_epochs,
            eta_min=self.train_config.learning_rate * MINIMUM_LR_FACTOR,
        )
        scheduler: LRScheduler
        if warmup_epochs > 0:
            linear = LinearLR(
                optimizer,
                start_factor=WARMUP_START_FACTOR,
                total_iters=warmup_epochs,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[linear, cosine],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = cosine
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self) -> None:
        """Update the reconstruction weight for the new epoch.

        Returns:
            None: The scheduled weight is stored on the module.
        """
        self._current_lambda = self.lambda_schedule.at(
            self.current_epoch,
            self.train_config.max_epochs,
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Attach the arm and configuration to a checkpoint.

        Args:
            checkpoint (dict[str, Any]): Lightning checkpoint payload.

        Returns:
            None: The payload is extended in place.
        """
        checkpoint["eos_mode"] = self.mode
        checkpoint["eos_model_config"] = self.model_config.model_dump(mode="json")
        checkpoint["eos_train_config"] = self.train_config.model_dump(mode="json")

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject a checkpoint produced by a different ablation arm.

        Args:
            checkpoint (dict[str, Any]): Lightning checkpoint payload.

        Returns:
            None: The checkpoint belongs to this arm.

        Raises:
            ValueError: If the stored mode is present and differs from this arm.
        """
        stored = checkpoint.get("eos_mode")
        if stored is not None and stored != self.mode:
            raise ValueError(
                f"checkpoint was produced by mode {stored!r}, "
                f"but this module is configured for {self.mode!r}"
            )
