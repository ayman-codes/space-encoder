"""Radiometric scaling and geometry-only augmentation for EuroSAT patches.

Targets remain in ``[0, 1]`` because the decoder terminates with a sigmoid.
ImageNet mean/std normalization is deliberately absent: it would break the
radiometric contract shared by the reconstruction loss.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import Tensor, nn


class ToCHWFloat(nn.Module):
    """Convert an HWC uint8 array to a CHW float32 tensor in ``[0, 1]``."""

    def forward(self, image: np.ndarray) -> Tensor:
        """Convert one image.

        Args:
            image (np.ndarray): Array of shape ``(H, W, C)`` with uint8 values.

        Returns:
            Tensor: Float32 tensor of shape ``(C, H, W)`` in ``[0, 1]``.

        Raises:
            ValueError: If the input is not a three-dimensional uint8 array.
        """
        if image.ndim != 3 or image.dtype != np.uint8:
            raise ValueError("image must be a three-dimensional uint8 array")
        tensor = torch.from_numpy(np.ascontiguousarray(image))
        return tensor.permute(2, 0, 1).to(torch.float32).div_(255.0)


class RandomFlip(nn.Module):
    """Apply independent horizontal and vertical flips with probability 0.5."""

    def forward(self, image: Tensor) -> Tensor:
        """Flip a CHW tensor.

        Args:
            image (Tensor): Tensor of shape ``(C, H, W)``.

        Returns:
            Tensor: Flipped tensor.
        """
        if bool(torch.rand(()) < 0.5):
            image = torch.flip(image, dims=(2,))
        if bool(torch.rand(()) < 0.5):
            image = torch.flip(image, dims=(1,))
        return image


class RandomQuarterTurn(nn.Module):
    """Rotate by a random multiple of 90 degrees without interpolation."""

    def forward(self, image: Tensor) -> Tensor:
        """Rotate a CHW tensor.

        Args:
            image (Tensor): Tensor of shape ``(C, H, W)``.

        Returns:
            Tensor: Rotated tensor.
        """
        turns = int(torch.randint(0, 4, ()).item())
        return torch.rot90(image, turns, dims=(1, 2)) if turns else image


class TrainTransform(nn.Module):
    """Training transform with radiometric scaling and geometry augmentation."""

    def __init__(self) -> None:
        """Create the training transform pipeline.

        Returns:
            None: Submodules are attached to the instance.
        """
        super().__init__()
        self.scale = ToCHWFloat()
        self.flip = RandomFlip()
        self.turn = RandomQuarterTurn()

    def forward(self, image: np.ndarray) -> Tensor:
        """Transform a training image.

        Args:
            image (np.ndarray): HWC uint8 image.

        Returns:
            Tensor: Augmented CHW float32 tensor in ``[0, 1]``.
        """
        return cast(Tensor, self.turn(self.flip(self.scale(image))))


class EvalTransform(nn.Module):
    """Deterministic evaluation transform."""

    def __init__(self) -> None:
        """Create the deterministic evaluation transform.

        Returns:
            None: The scaling submodule is attached to the instance.
        """
        super().__init__()
        self.scale = ToCHWFloat()

    def forward(self, image: np.ndarray) -> Tensor:
        """Transform an evaluation image.

        Args:
            image (np.ndarray): HWC uint8 image.

        Returns:
            Tensor: CHW float32 tensor in ``[0, 1]``.
        """
        return cast(Tensor, self.scale(image))


def build_train_transform() -> nn.Module:
    """Return the training transform.

    Returns:
        nn.Module: Training transform module.
    """
    return TrainTransform()


def build_eval_transform() -> nn.Module:
    """Return the deterministic evaluation transform.

    Returns:
        nn.Module: Evaluation transform module.
    """
    return EvalTransform()
