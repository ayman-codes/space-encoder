"""Typed, immutable configuration contracts for the earth observation engine.

Every section rejects unknown keys so a typo in YAML fails at load time rather
than silently training with a default value. Cross-field invariants that encode
the architecture guardrails live in ``model_validator`` methods.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrozenModel(BaseModel):
    """Base class for immutable, strictly validated configuration sections.

    Unknown keys are rejected and instances are hashable, which keeps config
    objects safe to cache and to share across dataloader workers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class DataConfig(FrozenModel):
    """Dataset, split, and dataloader configuration for EuroSAT.

    Args:
        root (Path): Directory holding the extracted ``EuroSAT_RGB`` class
            folders. Relative paths resolve from the repository root.
        image_size (int): Square patch edge length in pixels. Must be divisible
            by ``2 ** 4`` to preserve the encoder's exact spatial symmetry.
        channels (int): Number of spectral channels in ``(B, C, H, W)`` inputs.
            EuroSAT RGB uses 3.
        num_classes (int): Number of land-use/land-cover classes.
        batch_size (int): Mini-batch size.
        num_workers (int): DataLoader worker processes. Use 0 on constrained
            shared-memory environments.
        val_fraction (float): Validation share of the full dataset.
        test_fraction (float): Test share of the full dataset.
        seed (int): Deterministic split and augmentation seed.
        mean (tuple[float, ...]): Per-channel means used for radiometric
            centering. Must match ``channels``.
        std (tuple[float, ...]): Per-channel standard deviations. Must be
            strictly positive and match ``channels``.

    Raises:
        ValueError: If the split fractions do not leave a positive training
            share, if the normalization tuples do not match ``channels``, or if
            any standard deviation is not positive.
    """

    root: Path = Path("data/raw/eurosat")
    image_size: int = Field(default=64, ge=16, le=512)
    channels: int = Field(default=3, ge=1, le=13)
    num_classes: int = Field(default=10, ge=2)
    batch_size: int = Field(default=128, ge=1)
    num_workers: int = Field(default=4, ge=0)
    val_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    seed: int = Field(default=42, ge=0)
    mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    std: tuple[float, ...] = (0.5, 0.5, 0.5)

    @model_validator(mode="after")
    def _check_splits(self) -> Self:
        """Ensure a positive training portion remains after the splits.

        Returns:
            Self: The validated configuration instance.

        Raises:
            ValueError: If validation and test fractions consume the dataset.
        """
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be strictly below 1.0")
        return self

    @model_validator(mode="after")
    def _check_normalization(self) -> Self:
        """Ensure the normalization statistics match the channel count.

        Returns:
            Self: The validated configuration instance.

        Raises:
            ValueError: If the tuples are misaligned or a standard deviation is
                not strictly positive.
        """
        if len(self.mean) != self.channels or len(self.std) != self.channels:
            raise ValueError("mean and std must provide exactly one value per channel")
        if any(value <= 0.0 for value in self.std):
            raise ValueError("every std value must be strictly positive")
        return self


class ModelConfig(FrozenModel):
    """Architecture configuration for the hybrid convolutional autoencoder.

    Args:
        input_size (int): Square input edge length. The encoder halves the
            spatial extent once per stage.
        latent_dim (int): Bottleneck dimensionality of ``z``.
        base_channels (int): Width of the first convolutional block. Each stage
            doubles it.
        downsample_stages (int): Number of stride-2 encoder stages and matching
            transposed-convolution decoder stages.
        num_classes (int): Number of classification logits.
        dropout (float): Dropout probability in the classification head.
        recon_alpha (float): Weight of the MSE term inside the reconstruction
            loss; ``1 - recon_alpha`` weights the SSIM term.
        recon_lambda (float): Reconstrution loss weight relative to the
            cross-entropy term.
        kernel_size (int): Convolution kernel size for the projection layers.
        output_activation (Literal["sigmoid"]): Terminal decoder activation.
            The decoder must remain sigmoid-bounded because targets are scaled
            to ``[0, 1]``.

    Raises:
        ValueError: If the spatial extent is not divisible by the number of
            downsampling stages, or if the classification head cannot produce
            one logit per class.
    """

    input_size: int = Field(default=64, ge=16, le=512)
    latent_dim: int = Field(default=128, ge=8, le=4096)
    base_channels: int = Field(default=32, ge=4, le=512)
    downsample_stages: int = Field(default=4, ge=1, le=6)
    num_classes: int = Field(default=10, ge=2)
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    recon_alpha: float = Field(default=0.85, ge=0.0, le=1.0)
    recon_lambda: float = Field(default=0.3, ge=0.1, le=0.5)
    kernel_size: int = Field(default=3, ge=1, le=7)
    output_activation: Literal["sigmoid"] = "sigmoid"

    @model_validator(mode="after")
    def _check_spatial_symmetry(self) -> Self:
        """Verify that strided downsampling lands on an integer extent.

        Returns:
            Self: The validated configuration instance.

        Raises:
            ValueError: If the input size is not divisible by ``2 ** stages``.
        """
        divisor = 2**self.downsample_stages
        if self.input_size % divisor != 0:
            raise ValueError(
                f"input_size={self.input_size} must be divisible by 2**{self.downsample_stages}"
            )
        return self


class TrainConfig(FrozenModel):
    """Lightning trainer and optimizer configuration.

    Args:
        max_epochs (int): Upper bound on training epochs.
        learning_rate (float): Peak AdamW learning rate.
        weight_decay (float): AdamW decoupled weight decay.
        warmup_epochs (int): Linear warmup duration before cosine decay.
        gradient_clip (float): Global gradient norm clipping value.
        precision (Literal["bf16-mixed", "32-true"]): Numerical precision. FP16
            is intentionally excluded because SSIM variance denominators can
            underflow.
        accelerator (Literal["auto", "gpu", "cpu"]): Lightning accelerator.
        devices (int): Number of accelerator devices.
        seed (int): Global seed applied to Python, NumPy, and Torch.
        fast_dev_run (bool): Run a single batch for smoke testing.

    Raises:
        ValueError: If the warmup does not fit inside the epoch budget.
    """

    max_epochs: int = Field(default=30, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    warmup_epochs: int = Field(default=2, ge=0)
    gradient_clip: float = Field(default=1.0, gt=0.0)
    precision: Literal["bf16-mixed", "32-true"] = "bf16-mixed"
    accelerator: Literal["auto", "gpu", "cpu"] = "auto"
    devices: int = Field(default=1, ge=1)
    seed: int = Field(default=42, ge=0)
    fast_dev_run: bool = False

    @model_validator(mode="after")
    def _check_warmup(self) -> Self:
        """Ensure the warmup schedule fits inside the epoch budget.

        Returns:
            Self: The validated configuration instance.

        Raises:
            ValueError: If ``warmup_epochs`` is not below ``max_epochs``.
        """
        if self.warmup_epochs >= self.max_epochs:
            raise ValueError("warmup_epochs must be smaller than max_epochs")
        return self


class ASVDConfig(FrozenModel):
    """Adaptive singular value decomposition configuration.

    Args:
        preserve_energy (float): Fraction of cumulative spectral energy to
            retain. Constrained to the assignment's ``[0.90, 0.95]`` band.
        min_rank (int): Lowest admissible truncation rank.
        skip_last_layer (Literal[True]): The final classifier projection must
            never be factorized because its logits are calibrated.
        min_compression_gain (float): Only factorize a layer of shape ``(m, n)``
            when ``k * (m + n) < m * n``.
    """

    preserve_energy: float = Field(default=0.95, ge=0.90, le=0.95)
    min_rank: int = Field(default=1, ge=1)
    skip_last_layer: Literal[True] = True
    min_compression_gain: float = Field(default=1.0, ge=1.0)


class MMDConfig(FrozenModel):
    """Maximum mean discrepancy drift telemetry configuration.

    Args:
        bandwidth_multipliers (tuple[float, ...]): Scale factors applied to the
            median pairwise distance to build the multi-scale RBF kernel.
        sigma_threshold (float): Alarm threshold expressed in baseline standard
            deviations above the calibrated mean.
        max_baseline_samples (int): Number of validation embeddings retained for
            the reference distribution.
        normalize (Literal[True]): Whether to L2-normalize bottleneck embeddings
            before kernel evaluation.

    Raises:
        ValueError: If the bandwidth multipliers are not strictly increasing
            positive numbers.
    """

    bandwidth_multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    sigma_threshold: float = Field(default=3.0, gt=0.0)
    max_baseline_samples: int = Field(default=500, ge=10)
    normalize: Literal[True] = True

    @model_validator(mode="after")
    def _check_bandwidths(self) -> Self:
        """Validate the multi-scale bandwidth grid.

        Returns:
            Self: The validated configuration instance.

        Raises:
            ValueError: If any multiplier is non-positive or the sequence is not
                strictly increasing.
        """
        multipliers = self.bandwidth_multipliers
        if not multipliers or any(value <= 0.0 for value in multipliers):
            raise ValueError("bandwidth multipliers must be strictly positive")
        if any(later <= earlier for earlier, later in itertools.pairwise(multipliers)):
            raise ValueError("bandwidth multipliers must be strictly increasing")
        return self


class TelemetryConfig(FrozenModel):
    """Service endpoints used by experiment tracking and metrics export.

    Args:
        service_name (str): Prometheus job label.
        pushgateway_url (str): Pushgateway base URL.
        prometheus_url (str): Prometheus base URL used for read-back checks.
    """

    service_name: str = Field(default="eos-telemetry", min_length=1)
    pushgateway_url: str = "http://localhost:9091"
    prometheus_url: str = "http://localhost:9090"


class ExperimentConfig(FrozenModel):
    """Aggregate configuration for a single experiment run.

    Args:
        data (DataConfig): Dataset and dataloader settings.
        model (ModelConfig): Architecture settings.
        train (TrainConfig): Trainer and optimizer settings.
        asvd (ASVDConfig): Compression settings.
        mmd (MMDConfig): Drift telemetry settings.
        telemetry (TelemetryConfig): Metrics service endpoints.
    """

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    asvd: ASVDConfig = Field(default_factory=ASVDConfig)
    mmd: MMDConfig = Field(default_factory=MMDConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)


class ServiceSettings(BaseSettings):
    """Runtime secrets and endpoints loaded from the environment and ``.env``.

    Args:
        mlflow_tracking_uri (str): MLflow tracking server URL.
        pushgateway_url (str): Pushgateway base URL.
        prometheus_url (str): Prometheus base URL.
        minio_endpoint (str): MinIO S3 endpoint.
        minio_access_key (SecretStr | None): MinIO access key.
        minio_secret_key (SecretStr | None): MinIO secret key.
        grafana_admin_user (str): Grafana administrator login.
        grafana_admin_password (SecretStr | None): Grafana administrator password.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    mlflow_tracking_uri: str = "http://localhost:5000"
    pushgateway_url: str = "http://localhost:9091"
    prometheus_url: str = "http://localhost:9090"
    minio_endpoint: str = "http://localhost:9000"
    minio_access_key: SecretStr | None = None
    minio_secret_key: SecretStr | None = None
    grafana_admin_user: str = "admin"
    grafana_admin_password: SecretStr | None = None


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from YAML.

    Args:
        path (Path): YAML file containing one or more configuration sections.

    Returns:
        ExperimentConfig: Fully validated, immutable configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file does not contain a YAML mapping.
        pydantic.ValidationError: If any value violates its contract.
    """
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping, received {type(payload).__name__}")
    return ExperimentConfig.model_validate(payload)
