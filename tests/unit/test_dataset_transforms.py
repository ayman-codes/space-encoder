"""Tests for radiometric scaling and geometry-only augmentation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eos.dataset.transforms import (
    EvalTransform,
    RandomFlip,
    RandomQuarterTurn,
    ToCHWFloat,
    TrainTransform,
    build_eval_transform,
    build_train_transform,
)


def sample_image() -> np.ndarray:
    """Return a deterministic HWC uint8 image."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


def test_scaling_produces_chw_float_in_unit_range() -> None:
    """Scaling must produce float32 CHW values in ``[0, 1]``."""
    tensor = ToCHWFloat()(sample_image())
    assert tensor.shape == (3, 64, 64)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_scaling_rejects_invalid_input() -> None:
    """A non-uint8 or non-3D input must fail clearly."""
    with pytest.raises(ValueError, match="image must be"):
        ToCHWFloat()(np.zeros((64, 64, 3), dtype=np.float32))


def test_train_and_eval_transforms_keep_shape() -> None:
    """Both transforms must preserve tensor geometry."""
    image = sample_image()
    train_output = TrainTransform()(image)
    eval_output = EvalTransform()(image)
    assert train_output.shape == (3, 64, 64)
    assert eval_output.shape == (3, 64, 64)
    assert build_train_transform()(image).shape == (3, 64, 64)
    assert build_eval_transform()(image).shape == (3, 64, 64)


def test_flip_and_quarter_turn_are_geometry_only() -> None:
    """Augmentations must not change value range or channel count."""
    tensor = ToCHWFloat()(sample_image())
    flipped = RandomFlip()(tensor)
    turned = RandomQuarterTurn()(tensor)
    assert flipped.shape == tensor.shape
    assert turned.shape == tensor.shape
    assert float(flipped.min()) >= 0.0
    assert float(turned.max()) <= 1.0
