"""Tests for the encoder spatial trace and bottleneck projection."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from eos.config import ModelConfig
from eos.model.encoder import Encoder


def test_encoder_spatial_trace() -> None:
    """Four stride-2 blocks must halve 64 to 4 without remainder."""
    model = Encoder(ModelConfig()).eval()
    batch = torch.rand(2, 3, 64, 64)
    shapes: list[tuple[int, ...]] = []
    with torch.no_grad():
        values = batch
        for block in model.blocks:
            values = block(values)
            shapes.append(tuple(values.shape))
    assert [shape[2] for shape in shapes] == [32, 16, 8, 4]
    assert [shape[1] for shape in shapes] == [64, 128, 256, 512]


def test_encoder_output_and_features() -> None:
    """The embedding and flattened features must match the configuration."""
    model = Encoder(ModelConfig()).eval()
    batch = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        embedding = model(batch)
        features = model.features(batch)
    assert embedding.shape == (2, 128)
    assert features.shape == (2, 8192)
    assert model.latent_shape == (512, 4, 4)


def test_encoder_rejects_wrong_geometry() -> None:
    """Channel and spatial mismatches must fail before the convolution runs."""
    model = Encoder(ModelConfig())
    with pytest.raises(ValueError, match="channels"):
        model(torch.rand(2, 1, 64, 64))
    with pytest.raises(ValueError, match="patches"):
        model(torch.rand(2, 3, 32, 32))
    with pytest.raises(ValueError, match="4-D"):
        model(torch.rand(3, 64, 64))


def test_encoder_gradient_reaches_projection() -> None:
    """A backward pass must populate the bottleneck projection gradient."""
    model = Encoder(ModelConfig())
    model.train()
    embedding = model(torch.rand(2, 3, 64, 64))
    embedding.square().mean().backward()
    assert model.projection.weight.grad is not None
    assert float(model.projection.weight.grad.abs().sum()) > 0.0
    assert isinstance(model.projection, nn.Linear)
