"""Tests for adaptive SVD rank truncation of dense layers."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from eos.config import ASVDConfig, ModelConfig
from eos.model.compression import (
    FactorizedLinear,
    factorize_linear,
    factorize_module,
    select_rank,
    state_signature,
)
from eos.model.hybrid_cae import ConvAE, HybridCAE, PureCNN


def _equal_spectrum_layer(
    out_features: int,
    in_features: int,
    rank: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> nn.Linear:
    """Build a linear layer whose nonzero singular values are all equal.

    An equal spectrum makes the energy rank fully deterministic: rank ``k``
    covers exactly ``k / rank`` of the spectral energy.

    Args:
        out_features (int): Number of output features ``m``.
        in_features (int): Number of input features ``n``.
        rank (int): Exact algebraic rank of the weight matrix.
        dtype (torch.dtype): Parameter dtype of the returned layer.

    Returns:
        nn.Linear: A linear layer with the requested exact rank.
    """
    generator = torch.Generator().manual_seed(sum([out_features, in_features, rank]))
    left = torch.linalg.qr(torch.randn(out_features, rank, generator=generator)).Q
    right = torch.linalg.qr(torch.randn(in_features, rank, generator=generator)).Q
    weight = left @ right.T
    layer = nn.Linear(in_features, out_features, bias=True, dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(weight.to(dtype))
        layer.bias.copy_(torch.linspace(-1.0, 1.0, out_features).to(dtype))
    return layer


def _spectrum_layer(
    out_features: int,
    in_features: int,
    spectrum: Tensor,
) -> nn.Linear:
    """Build a linear layer with a prescribed singular value spectrum.

    Args:
        out_features (int): Number of output features ``m``.
        in_features (int): Number of input features ``n``.
        spectrum (Tensor): Descending nonzero singular values, shape ``(r,)``.

    Returns:
        nn.Linear: A linear layer whose weight has exactly that spectrum.
    """
    rank = spectrum.numel()
    generator = torch.Generator().manual_seed(rank * 1000 + out_features)
    left = torch.linalg.qr(torch.randn(out_features, rank, generator=generator)).Q
    right = torch.linalg.qr(torch.randn(in_features, rank, generator=generator)).Q
    weight = left @ torch.diag(spectrum) @ right.T
    layer = nn.Linear(in_features, out_features)
    with torch.no_grad():
        layer.weight.copy_(weight)
    return layer


def _patch_low_rank(model: nn.Module, rank: int = 8) -> None:
    """Replace every eligible projection of a model with exact low-rank weights.

    Args:
        model (nn.Module): Model whose projections are replaced in place.
        rank (int): Exact algebraic rank imposed on each projection.

    Returns:
        None: The model is mutated in place.
    """
    replacements = {
        "encoder.projection": (128, 8192),
        "decoder.projection": (8192, 128),
        "classifier.hidden": (64, 128),
    }
    modules = dict(model.named_modules())
    for name, (out_features, in_features) in replacements.items():
        module = modules.get(name)
        if isinstance(module, nn.Linear):
            fresh = _equal_spectrum_layer(out_features, in_features, rank)
            with torch.no_grad():
                module.weight.copy_(fresh.weight)
                module.bias.copy_(fresh.bias)


def test_select_rank_respects_energy_threshold() -> None:
    """The chosen rank must be the smallest one meeting the energy target."""
    singular = torch.tensor([4.0, 3.0, 2.0, 1.0])
    rank, energy = select_rank(singular, preserve_energy=0.90, min_rank=1)
    assert rank == 3
    assert energy == pytest.approx((16 + 9 + 4) / 30.0)


def test_select_rank_applies_min_rank_floor() -> None:
    """The minimum rank must override an earlier energy crossing."""
    singular = torch.tensor([4.0, 3.0, 2.0, 1.0])
    rank, energy = select_rank(singular, preserve_energy=0.50, min_rank=3)
    assert rank == 3
    assert energy == pytest.approx((16 + 9 + 4) / 30.0)


def test_select_rank_clamps_to_available_components() -> None:
    """The rank cannot exceed the number of singular values."""
    singular = torch.tensor([4.0, 3.0])
    rank, energy = select_rank(singular, preserve_energy=0.95, min_rank=8)
    assert rank == 2
    assert energy == pytest.approx(1.0)


def test_select_rank_rejects_empty_spectrum() -> None:
    """An empty spectrum is a programming error, not a compression decision."""
    with pytest.raises(ValueError, match="singular spectrum must not be empty"):
        select_rank(torch.tensor([]), preserve_energy=0.95, min_rank=1)


def test_factorization_meets_the_frobenius_truncation_bound() -> None:
    """The reconstruction error must match the optimal rank-k truncation error."""
    spectrum = torch.tensor([1.0, 0.9, 0.81, 0.729, 0.6561, 0.59049, 0.531441, 0.4782969])
    layer = _spectrum_layer(32, 128, spectrum)
    config = ASVDConfig(preserve_energy=0.90)
    record, replacement = factorize_linear("probe", layer, config)
    assert not record.skipped
    assert isinstance(replacement, FactorizedLinear)
    full = torch.linalg.svdvals(layer.weight.detach().to(torch.float32))
    tail = float(full[record.rank :].square().sum().sqrt())
    error = float(
        torch.norm(layer.weight.detach().to(torch.float32) - replacement.rebuild().detach())
    )
    assert error > 0.0
    assert error <= tail * (1.0 + 1.0e-4) + 1.0e-5


def test_factorized_linear_preserves_bias_and_forward() -> None:
    """A rank-exact factorization must reproduce the original layer exactly."""
    layer = _equal_spectrum_layer(32, 64, rank=4)
    record, replacement = factorize_linear("probe", layer, ASVDConfig())
    assert not record.skipped
    assert isinstance(replacement, FactorizedLinear)
    assert replacement.down.bias is None
    assert replacement.up.bias is not None
    assert torch.allclose(replacement.up.bias, layer.bias)
    inputs = torch.randn(5, 64)
    with torch.no_grad():
        expected = layer(inputs)
        actual = replacement(inputs)
    assert torch.allclose(actual, expected, atol=1.0e-5)


def test_gain_guard_skips_layers_and_records_the_measured_rank() -> None:
    """A layer without real compression gain must be skipped with evidence."""
    layer = _equal_spectrum_layer(64, 64, rank=64)
    record, replacement = factorize_linear("probe", layer, ASVDConfig())
    assert record.skipped
    assert record.rank == 61
    assert "gain" in record.reason
    assert record.params_before == record.params_after
    assert replacement is layer


def test_accounting_counts_are_exact() -> None:
    """Before and after parameter counts must equal the closed-form arithmetic."""
    layer = _equal_spectrum_layer(32, 64, rank=4)
    record, _ = factorize_linear("probe", layer, ASVDConfig())
    assert record.shape == (32, 64)
    assert record.rank == 4
    assert record.params_before == 32 * 64 + 32
    assert record.params_after == 4 * 64 + 4 * 32 + 32
    assert record.energy == pytest.approx(1.0)


def test_factorization_preserves_parameter_dtype() -> None:
    """The SVD runs in float32 while the factors keep the original dtype."""
    layer = _equal_spectrum_layer(32, 64, rank=4, dtype=torch.float64)
    record, replacement = factorize_linear("probe", layer, ASVDConfig())
    assert not record.skipped
    assert isinstance(replacement, FactorizedLinear)
    assert replacement.up.weight.dtype == torch.float64
    assert replacement.down.weight.dtype == torch.float64


@pytest.mark.parametrize(
    ("scope", "factorized", "untouched"),
    [
        ("classifier", {"classifier.hidden"}, {"encoder.projection", "decoder.projection"}),
        ("bottleneck", {"encoder.projection", "decoder.projection"}, {"classifier.hidden"}),
        (
            "all_intermediate",
            {"encoder.projection", "decoder.projection", "classifier.hidden"},
            set(),
        ),
    ],
)
def test_scope_dispatch_selects_the_documented_layers(
    scope: str,
    factorized: set[str],
    untouched: set[str],
) -> None:
    """Each scope must factorize exactly its eligible projection set.

    Args:
        scope (str): ``ASVDConfig.scope`` value under test.
        factorized (set[str]): Module paths expected to become factorized.
        untouched (set[str]): Module paths expected to remain plain linear.

    Returns:
        None: Assertions verify the dispatch table.
    """
    model = HybridCAE(ModelConfig())
    _patch_low_rank(model)
    factorize_module(model, ASVDConfig(scope=scope))  # type: ignore[arg-type]
    modules = dict(model.named_modules())
    for name in factorized:
        assert isinstance(modules[name], FactorizedLinear)
    for name in untouched:
        assert isinstance(modules[name], nn.Linear)
    assert not isinstance(modules["classifier.logits"], FactorizedLinear)


def test_state_signature_is_deterministic() -> None:
    """Equal passes must serialize identically and scopes must differ."""
    model = HybridCAE(ModelConfig())
    _patch_low_rank(model)
    first = factorize_module(model, ASVDConfig(scope="bottleneck"))
    assert state_signature(first) == "decoder.projection:8:0;encoder.projection:8:0"
    other = HybridCAE(ModelConfig())
    _patch_low_rank(other)
    second = factorize_module(other, ASVDConfig(scope="all_intermediate"))
    assert state_signature(second) != state_signature(first)


def test_logits_projection_is_never_factorized() -> None:
    """The calibrated logits layer must survive every scope untouched."""
    model = HybridCAE(ModelConfig())
    _patch_low_rank(model)
    logits_before = model.classifier.logits
    for scope in ("classifier", "bottleneck", "all_intermediate"):
        factorize_module(model, ASVDConfig(scope=scope))  # type: ignore[arg-type]
    assert model.classifier.logits is logits_before
    assert isinstance(model.classifier.logits, nn.Linear)


def test_model_forward_contract_survives_factorization() -> None:
    """The factorized model must reproduce the original forward pass."""
    model = HybridCAE(ModelConfig()).eval()
    _patch_low_rank(model)
    batch = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        before = model(batch)
    result = factorize_module(model, ASVDConfig(scope="all_intermediate"))
    model.eval()
    with torch.no_grad():
        after = model(batch)
    assert after.logits.shape == (2, 10)
    assert after.reconstruction.shape == (2, 3, 64, 64)
    assert after.embedding.shape == (2, 128)
    assert torch.allclose(after.logits, before.logits, atol=1.0e-4)
    assert torch.allclose(after.reconstruction, before.reconstruction, atol=1.0e-4)
    assert result.dense_reduction > 0.0


def test_result_reports_dense_and_model_separately() -> None:
    """Dense-set and whole-model compression must be reported apart."""
    model = HybridCAE(ModelConfig())
    _patch_low_rank(model)
    before = sum(parameter.numel() for parameter in model.parameters())
    result = factorize_module(model, ASVDConfig(scope="bottleneck"))
    after = sum(parameter.numel() for parameter in model.parameters())
    assert result.model_params_before == before
    assert result.model_params_after == after
    assert result.model_params_before == 6_453_421
    assert result.dense_before == 2 * (8192 * 128) + 128 + 8192
    assert result.dense_after < result.dense_before
    assert result.model_reduction < result.dense_reduction


def test_single_branch_models_are_handled() -> None:
    """Models missing a branch must factorize only the projections they own."""
    classifier = PureCNN(ModelConfig())
    _patch_low_rank(classifier)
    result = factorize_module(classifier, ASVDConfig(scope="bottleneck"))
    assert [layer.name for layer in result.layers] == ["encoder.projection"]
    autoencoder = ConvAE(ModelConfig())
    _patch_low_rank(autoencoder)
    result = factorize_module(autoencoder, ASVDConfig(scope="all_intermediate"))
    assert [layer.name for layer in result.layers] == [
        "decoder.projection",
        "encoder.projection",
    ]
