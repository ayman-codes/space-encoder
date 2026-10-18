"""Verify ROCm visibility and Lightning bf16 training inside the GPU container.

The checks run from the weakest assumption to the strongest: the HIP build is
present, the expected device is enumerable, bf16 matrix multiplication and a
convolution backward pass are numerically stable, and a one-epoch Lightning
fit completes with ``bf16-mixed`` precision.
"""

from __future__ import annotations

import sys

import lightning
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EXPECTED_DEVICE = "AMD Radeon RX 9070 XT"


class SmokeModule(lightning.LightningModule):
    """Minimal convolutional classifier used for the GPU smoke test.

    Args:
        learning_rate (float): AdamW learning rate.
    """

    def __init__(self, learning_rate: float = 1e-3) -> None:
        """Initialize the convolutional stack and classification head.

        Args:
            learning_rate (float): AdamW learning rate.

        Returns:
            None: The module is configured in place.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(16, 10)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of patches to class logits.

        Args:
            x (torch.Tensor): Input batch of shape ``(B, 3, 64, 64)``.

        Returns:
            torch.Tensor: Logits of shape ``(B, 10)``.
        """
        features = self.encoder(x).flatten(1)
        logits: torch.Tensor = self.head(features)
        return logits

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Run one optimisation step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): Inputs and targets.
            batch_idx (int): Index of the batch inside the epoch.

        Returns:
            torch.Tensor: Scalar cross-entropy loss.
        """
        inputs, targets = batch
        loss: torch.Tensor = self.loss(self(inputs), targets)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Build the optimizer.

        Returns:
            torch.optim.Optimizer: AdamW over all parameters.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


def check_hip_build() -> str:
    """Assert that PyTorch was compiled against ROCm.

    Returns:
        str: The HIP runtime version string.

    Raises:
        RuntimeError: If the installed build is not a HIP build.
    """
    hip_version = getattr(torch.version, "hip", None)
    if not hip_version:
        raise RuntimeError("torch was not built against ROCm")
    return str(hip_version)


def check_device() -> torch.device:
    """Assert that the expected GPU is visible.

    Returns:
        torch.device: The first CUDA-compatible device, which HIP aliases.

    Raises:
        RuntimeError: If no GPU is visible or the device name is unexpected.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() returned False")
    name = torch.cuda.get_device_name(0)
    if EXPECTED_DEVICE not in name:
        raise RuntimeError(f"unexpected device name: {name}")
    return torch.device("cuda", 0)


def check_bf16_matmul(device: torch.device) -> None:
    """Exercise a bf16 matrix multiplication and verify finiteness.

    Args:
        device (torch.device): Device used for the computation.

    Raises:
        RuntimeError: If the result contains non-finite values.
    """
    left = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
    product = left @ left
    if not bool(torch.isfinite(product.float()).all()):
        raise RuntimeError("bf16 matmul produced non-finite values")


def check_convolution(device: torch.device) -> None:
    """Exercise a convolutional forward and backward pass.

    Args:
        device (torch.device): Device used for the computation.

    Raises:
        RuntimeError: If gradients are missing or non-finite.
    """
    module = nn.Conv2d(3, 8, kernel_size=3, padding=1).to(device)
    inputs = torch.randn(2, 3, 64, 64, device=device, requires_grad=True)
    module(inputs).sum().backward()
    if inputs.grad is None or not bool(torch.isfinite(inputs.grad).all()):
        raise RuntimeError("convolution backward pass produced invalid gradients")


def check_lightning_fit(device: torch.device) -> None:
    """Run a single bf16-mixed Lightning epoch on synthetic data.

    Args:
        device (torch.device): Device used for training.
    """
    generator = torch.Generator().manual_seed(42)
    inputs = torch.randn(32, 3, 64, 64, generator=generator)
    targets = torch.randint(0, 10, (32,), generator=generator)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    trainer = lightning.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(SmokeModule(), loader)


def main() -> int:
    """Run the full ROCm verification sequence.

    Returns:
        int: Process exit code, zero when every check passes.
    """
    hip_version = check_hip_build()
    device = check_device()
    properties = torch.cuda.get_device_properties(device)
    vram_gib = properties.total_memory / 1024**3
    print(f"torch        {torch.__version__}")
    print(f"hip          {hip_version}")
    print(f"device       {properties.name}")
    print(f"vram         {vram_gib:.1f} GiB")
    check_bf16_matmul(device)
    print("bf16 matmul  OK")
    check_convolution(device)
    print("conv bwd     OK")
    check_lightning_fit(device)
    print("lightning    bf16-mixed epoch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
