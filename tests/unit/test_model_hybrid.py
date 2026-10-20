"""Tests for the composition models and the parameter budget."""

from __future__ import annotations

import torch

from eos.config import ModelConfig
from eos.model.hybrid_cae import ConvAE, HybridCAE, PureCNN


def test_hybrid_forward_contract() -> None:
    """One forward pass must yield logits, reconstruction, and embedding."""
    model = HybridCAE(ModelConfig()).eval()
    with torch.no_grad():
        output = model(torch.rand(2, 3, 64, 64))
    assert output.logits.shape == (2, 10)
    assert output.reconstruction.shape == (2, 3, 64, 64)
    assert output.embedding.shape == (2, 128)
    assert float(output.reconstruction.min()) >= 0.0
    assert float(output.reconstruction.max()) <= 1.0


def test_parameter_budget_is_exact() -> None:
    """The budget must equal the closed-form layer arithmetic."""
    counts = HybridCAE(ModelConfig()).parameter_count()
    assert counts["encoder"] == 2_600_640
    assert counts["decoder"] == 3_843_875
    assert counts["classifier"] == 8_906
    assert counts["total"] == 6_453_421


def test_both_branches_reach_the_shared_encoder() -> None:
    """Classification and reconstruction must both train the encoder."""
    model = HybridCAE(ModelConfig())
    batch = torch.rand(2, 3, 64, 64)
    model.zero_grad()
    model(batch).logits.square().mean().backward()
    classification_gradient = model.encoder.projection.weight.grad
    assert classification_gradient is not None
    assert float(classification_gradient.abs().sum()) > 0.0

    model.zero_grad()
    model(batch).reconstruction.square().mean().backward()
    reconstruction_gradient = model.encoder.projection.weight.grad
    assert reconstruction_gradient is not None
    assert float(reconstruction_gradient.abs().sum()) > 0.0


def test_eval_is_deterministic() -> None:
    """Dropout must be inactive during evaluation."""
    model = HybridCAE(ModelConfig()).eval()
    batch = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        first = model(batch)
        second = model(batch)
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first.reconstruction, second.reconstruction)


def test_ablation_wrappers_share_components() -> None:
    """PureCNN and ConvAE must reuse the encoder and expose their head only."""
    config = ModelConfig()
    cnn = PureCNN(config).eval()
    autoencoder = ConvAE(config).eval()
    batch = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        assert cnn(batch).shape == (2, 10)
        assert autoencoder(batch).shape == (2, 3, 64, 64)
    assert cnn.parameter_count()["total"] == 2_600_640 + 8_906
    assert autoencoder.parameter_count()["total"] == 2_600_640 + 3_843_875
