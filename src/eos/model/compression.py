r"""Adaptive singular value decomposition of dense projections.

A dense weight :math:`W \\in \\mathbb{R}^{m \\times n}` is decomposed with the
economy SVD :math:`W = U \\Sigma V^T` and truncated at the smallest rank ``k``
whose cumulative spectral energy meets ``preserve_energy``:

.. math::
    k = \\arg\\min_r \\left(
        \\frac{\\sum_{i=1}^{r} \\sigma_i^2}{\\sum_{i=1}^{\\min(m, n)} \\sigma_i^2}
        \\ge E \\right)

The layer is then replaced by the two-bank factorization
:math:`W_A = U_k \\sqrt{\\Sigma_k}` and
:math:`W_B = \\sqrt{\\Sigma_k} V_k^T` with no bias on the inner projection, so
the parameter cost drops from :math:`m n` to :math:`k(m + n)`. Factorization is
applied only when the multiplicative gain ``m n / (k (m + n))`` reaches
``min_compression_gain``; every rejected layer is recorded with its measured
rank and the reason, because a silent no-op is indistinguishable from a bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from eos.config import ASVDConfig

SCOPE_PATHS: dict[str, tuple[str, ...]] = {
    "classifier": ("classifier.hidden",),
    "bottleneck": ("encoder.projection", "decoder.projection"),
    "all_intermediate": (
        "encoder.projection",
        "decoder.projection",
        "classifier.hidden",
    ),
}


@dataclass(frozen=True, slots=True)
class LayerCompression:
    """Compression record for a single dense layer.

    Args:
        name (str): Module path of the layer inside the model.
        shape (tuple[int, int]): Weight shape ``(out_features, in_features)``.
        rank (int): Truncation rank selected by the energy rule.
        energy (float): Cumulative spectral energy retained at that rank.
        params_before (int): Parameters of the layer before factorization,
            including bias.
        params_after (int): Parameters of the layer after factorization,
            including bias. Equal to ``params_before`` when skipped.
        skipped (bool): Whether the layer was left untouched.
        reason (str): Why the layer was skipped; empty when factorized.
    """

    name: str
    shape: tuple[int, int]
    rank: int
    energy: float
    params_before: int
    params_after: int
    skipped: bool
    reason: str

    @property
    def gain(self) -> float:
        """Return the multiplicative parameter reduction of this layer.

        Returns:
            float: Ratio ``m n / (k (m + n))`` at the selected rank.
        """
        out_features, in_features = self.shape
        return (out_features * in_features) / (self.rank * (out_features + in_features))


@dataclass(frozen=True, slots=True)
class ASVDResult:
    """Aggregate accounting for one factorization pass.

    Dense-set and whole-model compression are reported separately because the
    convolutional backbone dominates the model while the factorized projections
    dominate the dense set.

    Args:
        layers (tuple[LayerCompression, ...]): Per-layer records, sorted by name.
        dense_before (int): Parameters of the eligible layers before the pass.
        dense_after (int): Parameters of the eligible layers after the pass.
        model_params_before (int): Whole-model parameters before the pass.
        model_params_after (int): Whole-model parameters after the pass.
    """

    layers: tuple[LayerCompression, ...]
    dense_before: int
    dense_after: int
    model_params_before: int
    model_params_after: int

    @property
    def dense_reduction(self) -> float:
        """Return the fraction of dense parameters removed.

        Returns:
            float: ``1 - dense_after / dense_before`` in ``[0, 1]``.
        """
        if self.dense_before == 0:
            return 0.0
        return 1.0 - self.dense_after / self.dense_before

    @property
    def model_reduction(self) -> float:
        """Return the fraction of whole-model parameters removed.

        Returns:
            float: ``1 - model_params_after / model_params_before`` in ``[0, 1]``.
        """
        if self.model_params_before == 0:
            return 0.0
        return 1.0 - self.model_params_after / self.model_params_before


class FactorizedLinear(nn.Module):
    """Two-bank low-rank replacement for a dense linear layer.

    The forward pass computes ``up(down(x))``, so the inner projection carries
    no bias and the outer projection owns the original bias.
    """

    def __init__(
        self,
        down_weight: Tensor,
        up_weight: Tensor,
        bias: Tensor | None,
    ) -> None:
        """Build the factor pair from truncated SVD factors.

        Args:
            down_weight (Tensor): Inner projection weight of shape ``(k, n)``.
            up_weight (Tensor): Outer projection weight of shape ``(m, k)``.
            bias (Tensor | None): Original bias of shape ``(m,)`` or ``None``.

        Raises:
            ValueError: If the factor shapes do not share the rank dimension.
        """
        super().__init__()
        rank, in_features = down_weight.shape
        out_features, up_rank = up_weight.shape
        if up_rank != rank:
            raise ValueError("factor rank dimensions must agree")
        self.down = nn.Linear(
            in_features,
            rank,
            bias=False,
            dtype=down_weight.dtype,
            device=down_weight.device,
        )
        self.up = nn.Linear(
            rank,
            out_features,
            bias=bias is not None,
            dtype=up_weight.dtype,
            device=up_weight.device,
        )
        with torch.no_grad():
            self.down.weight.copy_(down_weight)
            self.up.weight.copy_(up_weight)
            if bias is not None and self.up.bias is not None:
                self.up.bias.copy_(bias)

    @property
    def rank(self) -> int:
        """Return the truncation rank of the factorization.

        Returns:
            int: Shared dimension ``k`` of the factor pair.
        """
        return self.down.out_features

    @property
    def in_features(self) -> int:
        """Return the input width of the replaced layer.

        Returns:
            int: Number of input features ``n``.
        """
        return self.down.in_features

    @property
    def out_features(self) -> int:
        """Return the output width of the replaced layer.

        Returns:
            int: Number of output features ``m``.
        """
        return self.up.out_features

    def rebuild(self) -> Tensor:
        """Return the dense weight matrix approximated by the factors.

        Returns:
            Tensor: Product ``W_A W_B`` of shape ``(m, n)``.
        """
        return self.up.weight @ self.down.weight

    def forward(self, x: Tensor) -> Tensor:
        """Apply the factorized projection to a batch of inputs.

        Args:
            x (Tensor): Input tensor of shape ``(B, n)``.

        Returns:
            Tensor: Output tensor of shape ``(B, m)``.
        """
        return cast(Tensor, self.up(self.down(x)))


def select_rank(
    singular_values: Tensor,
    preserve_energy: float,
    min_rank: int,
) -> tuple[int, float]:
    """Select the smallest truncation rank meeting the energy target.

    Args:
        singular_values (Tensor): Singular values in descending order, shape
            ``(q,)`` with ``q = min(m, n)``.
        preserve_energy (float): Cumulative energy target ``E`` in ``(0, 1]``.
        min_rank (int): Lowest admissible rank; clamped to ``q``.

    Returns:
        tuple[int, float]: The selected rank and the energy it retains.

    Raises:
        ValueError: If the spectrum is empty.
    """
    if singular_values.numel() == 0:
        raise ValueError("singular spectrum must not be empty")
    energy = singular_values.detach().to(torch.float32).square()
    total = float(energy.sum())
    components = singular_values.numel()
    if total <= 0.0:
        rank = min(max(min_rank, 1), components)
        return rank, 1.0
    cumulative = energy.cumsum(0) / total
    target = torch.tensor(preserve_energy, dtype=cumulative.dtype)
    crossing = int(torch.searchsorted(cumulative, target).item()) + 1
    rank = min(max(crossing, min_rank), components)
    return rank, float(cumulative[rank - 1])


def state_signature(result: ASVDResult) -> str:
    """Serialize the factorization state of a pass deterministically.

    The signature participates in telemetry fingerprints so that a baseline
    calibrated before a factorization pass provably disagrees with the state
    after it.

    Args:
        result (ASVDResult): Accounting record of one factorization pass.

    Returns:
        str: Canonical text listing every layer as ``name:rank:skipped``.
    """
    return ";".join(f"{layer.name}:{layer.rank}:{int(layer.skipped)}" for layer in result.layers)


def factorize_linear(
    name: str,
    linear: nn.Linear,
    config: ASVDConfig,
) -> tuple[LayerCompression, nn.Module]:
    """Factorize one dense layer or record why it was left untouched.

    The SVD is computed in float32 regardless of the parameter dtype; the
    resulting factors are cast back to the original dtype.

    Args:
        name (str): Module path used in the compression record.
        linear (nn.Linear): Layer to factorize, weight of shape ``(m, n)``.
        config (ASVDConfig): Energy band, minimum rank, and gain floor.

    Returns:
        tuple[LayerCompression, nn.Module]: The record and the module to install,
        which is the original layer when the gain guard rejects the factorization.
    """
    weight = linear.weight
    out_features, in_features = weight.shape
    left, singular, right = torch.linalg.svd(weight.to(torch.float32), full_matrices=False)
    rank, energy = select_rank(singular, config.preserve_energy, config.min_rank)
    bias_present = linear.bias is not None
    params_before = weight.numel() + (out_features if bias_present else 0)
    gain = (out_features * in_features) / (rank * (out_features + in_features))
    if gain < config.min_compression_gain:
        reason = (
            f"compression gain {gain:.3f} below the "
            f"{config.min_compression_gain:.3f} floor at rank {rank}"
        )
        record = LayerCompression(
            name=name,
            shape=(out_features, in_features),
            rank=rank,
            energy=energy,
            params_before=params_before,
            params_after=params_before,
            skipped=True,
            reason=reason,
        )
        return record, linear
    scale = singular[:rank].sqrt()
    down_weight = (right[:rank, :] * scale.unsqueeze(1)).to(weight.dtype)
    up_weight = (left[:, :rank] * scale.unsqueeze(0)).to(weight.dtype)
    bias = linear.bias.detach().clone() if linear.bias is not None else None
    replacement = FactorizedLinear(down_weight, up_weight, bias)
    params_after = down_weight.numel() + up_weight.numel() + (out_features if bias_present else 0)
    record = LayerCompression(
        name=name,
        shape=(out_features, in_features),
        rank=rank,
        energy=energy,
        params_before=params_before,
        params_after=params_after,
        skipped=False,
        reason="",
    )
    return record, replacement


def _replace_module(root: nn.Module, path: str, replacement: nn.Module) -> None:
    """Install a replacement module at a dotted attribute path.

    Args:
        root (nn.Module): Model owning the path.
        path (str): Dotted module path such as ``encoder.projection``.
        replacement (nn.Module): Module to install at that path.

    Returns:
        None: The model is mutated in place.
    """
    parent: nn.Module = root
    parts = path.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def factorize_module(model: nn.Module, config: ASVDConfig) -> ASVDResult:
    """Factorize every eligible projection of a model according to the scope.

    The calibrated logits projection is never eligible, whatever the scope.
    Layers that are already factorized are recorded as skipped so repeat passes
    stay explainable, and any other module type at an eligible path is a
    contract violation.

    Args:
        model (nn.Module): Model mutated in place.
        config (ASVDConfig): Scope, energy band, and compression-gain floor.

    Returns:
        ASVDResult: Per-layer records plus dense-set and whole-model accounting.

    Raises:
        TypeError: If an eligible path holds something other than a linear layer.
    """
    modules = dict(model.named_modules())
    model_params_before = sum(parameter.numel() for parameter in model.parameters())
    records: list[LayerCompression] = []
    for path in SCOPE_PATHS[config.scope]:
        module = modules.get(path)
        if module is None:
            continue
        if isinstance(module, FactorizedLinear):
            count = sum(parameter.numel() for parameter in module.parameters())
            records.append(
                LayerCompression(
                    name=path,
                    shape=(module.out_features, module.in_features),
                    rank=module.rank,
                    energy=1.0,
                    params_before=count,
                    params_after=count,
                    skipped=True,
                    reason="already factorized",
                )
            )
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"eligible projection {path} must be nn.Linear, got {type(module).__name__}"
            )
        record, replacement = factorize_linear(path, module, config)
        records.append(record)
        if not record.skipped:
            _replace_module(model, path, replacement)
    records.sort(key=lambda item: item.name)
    model_params_after = sum(parameter.numel() for parameter in model.parameters())
    return ASVDResult(
        layers=tuple(records),
        dense_before=sum(record.params_before for record in records),
        dense_after=sum(record.params_after for record in records),
        model_params_before=model_params_before,
        model_params_after=model_params_after,
    )
