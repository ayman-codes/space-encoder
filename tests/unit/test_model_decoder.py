"""Tests for the symmetric decoder and its sigmoid output contract."""

from __future__ import annotations

import pytest
import torch

from eos.config import ModelConfig
from eos.model.decoder import Decoder


def test_decoder_spatial_trace() -> None:
    """Four transposed blocks must double 4 back to 64 exactly."""
    model = Decoder(ModelConfig()).eval()
    config = ModelConfig()
    with torch.no_grad():
        values = model.projection(torch.rand(2, config.latent_dim))
        values = values.reshape(2, 512, 4, 4)
        shapes: list[tuple[int, ...]] = []
        for block in model.blocks:
            values = block(values)
            shapes.append(tuple(values.shape))
    assert [shape[2] for shape in shapes] == [8, 16, 32, 64]
    assert [shape[1] for shape in shapes] == [256, 128, 64, 32]


def test_decoder_output_contract() -> None:
    """Reconstruction must be float32 in ``[0, 1]`` with the input geometry."""
    model = Decoder(ModelConfig()).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 128))
    assert output.shape == (2, 3, 64, 64)
    assert output.dtype == torch.float32
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0


def test_decoder_accepts_alternate_geometry() -> None:
    """A 32x32 configuration must reconstruct 32x32 through a 2x2 bottleneck."""
    model = Decoder(ModelConfig(input_size=32)).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 128))
    assert output.shape == (2, 3, 32, 32)


def test_decoder_rejects_wrong_latent() -> None:
    """The embedding rank and width are part of the contract."""
    model = Decoder(ModelConfig())
    with pytest.raises(ValueError, match="latent_dim"):
        model(torch.randn(2, 64))
    with pytest.raises(ValueError, match="2-D"):
        model(torch.randn(2, 128, 1))


def test_decoder_gradient_reaches_projection() -> None:
    """The reconstruction must differentiate through the linear projection."""
    model = Decoder(ModelConfig())
    output = model(torch.randn(2, 128))
    output.mean().backward()
    assert model.projection.weight.grad is not None
    assert float(model.projection.weight.grad.abs().sum()) > 0.0
