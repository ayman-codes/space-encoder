"""Tests for the classification head."""

from __future__ import annotations

import pytest
import torch

from eos.config import ModelConfig
from eos.model.classifier import ClassificationHead


def test_head_output_shape() -> None:
    """The head must emit one logit per configured class."""
    model = ClassificationHead(ModelConfig()).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 128))
    assert logits.shape == (2, 10)


def test_head_respects_hidden_width() -> None:
    """The hidden projection is the ASVD target and must follow the config."""
    config = ModelConfig(head_hidden=32, num_classes=5)
    model = ClassificationHead(config)
    assert model.hidden.out_features == 32
    assert model.logits.out_features == 5
    with torch.no_grad():
        assert model(torch.randn(2, 128)).shape == (2, 5)


def test_dropout_follows_module_mode() -> None:
    """Dropout probability comes from config and tracks train/eval mode."""
    model = ClassificationHead(ModelConfig(dropout=0.4))
    assert model.dropout.p == pytest.approx(0.4)
    model.train()
    assert model.dropout.training
    model.eval()
    assert not model.dropout.training


def test_head_rejects_wrong_embedding() -> None:
    """A non-2-D or wrongly sized embedding must fail clearly."""
    model = ClassificationHead(ModelConfig())
    with pytest.raises(ValueError, match="classification head"):
        model(torch.randn(2, 64))
    with pytest.raises(ValueError, match="classification head"):
        model(torch.randn(2, 128, 1))
