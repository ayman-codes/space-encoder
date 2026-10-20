"""Numerical objectives for the hybrid autoencoder.

The reconstruction term mixes pixel fidelity (MSE) with structural similarity
(SSIM). SSIM is evaluated in float32 with autocasting disabled because the local
variance denominators underflow in bfloat16.

Three boundary policies are supported because an 11x11 window cannot be
evaluated at the patch border without deciding what fills the missing pixels:

- ``valid`` crops the map to 54x54: border pixels never serve as a window
  center, so their influence on the objective is attenuated;
- ``zero`` pads both tensors with zeros, which dilutes local covariance and
  under-penalizes real border errors;
- ``masked`` normalizes every local statistic by the weight mass that actually
  lands inside the patch, preserving full 64x64 coverage without synthetic
  pixels.

``masked`` is the default. ``valid`` remains available for reporting because it
is the canonical definition used in the literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

BoundaryPolicy = Literal["valid", "zero", "masked"]

BOUNDARY_POLICIES: tuple[BoundaryPolicy, ...] = ("valid", "zero", "masked")


def _gaussian_kernel(
    window_size: int,
    sigma: float,
    channels: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Build a depthwise Gaussian window normalized to unit mass.

    Args:
        window_size (int): Odd window edge length in pixels.
        sigma (float): Standard deviation of the Gaussian in pixels.
        channels (int): Number of image channels; the kernel is repeated once per
            channel so the convolution is depthwise.
        dtype (torch.dtype): Target dtype of the kernel.
        device (torch.device): Device on which the kernel is allocated.

    Returns:
        Tensor: Kernel of shape ``(channels, 1, window_size, window_size)`` whose
            values sum to one over the full window.
    """
    coordinates = torch.arange(window_size, dtype=torch.float32, device=device)
    coordinates = coordinates - (window_size - 1) / 2.0
    gaussian = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    kernel = torch.outer(gaussian, gaussian)
    return kernel.expand(channels, 1, window_size, window_size).contiguous().to(dtype=dtype)


class SSIMLoss(nn.Module):
    """Structural similarity loss with an explicit boundary policy.

    Args:
        window_size (int): Odd Gaussian window edge length. Defaults to 11.
        sigma (float): Gaussian standard deviation in pixels. Defaults to 1.5.
        boundary (BoundaryPolicy): Boundary handling policy. Defaults to
            ``masked``.
        data_range (float): Dynamic range of the inputs. Defaults to 1.0 because
            inputs live in ``[0, 1]``.

    Raises:
        ValueError: If the window is even or below three, if ``sigma`` or
            ``data_range`` is not positive, or if the boundary policy is unknown.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        boundary: BoundaryPolicy = "masked",
        data_range: float = 1.0,
    ) -> None:
        """Validate and store the SSIM configuration.

        Args:
            window_size (int): Odd Gaussian window edge length.
            sigma (float): Gaussian standard deviation in pixels.
            boundary (BoundaryPolicy): Boundary handling policy.
            data_range (float): Dynamic range of the inputs.

        Returns:
            None: Parameters are stored on the module.

        Raises:
            ValueError: If the window, sigma, range, or policy is invalid.
        """
        super().__init__()
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer of at least three")
        if sigma <= 0.0:
            raise ValueError("sigma must be strictly positive")
        if data_range <= 0.0:
            raise ValueError("data_range must be strictly positive")
        if boundary not in BOUNDARY_POLICIES:
            raise ValueError(f"boundary must be one of {BOUNDARY_POLICIES}, received {boundary!r}")
        self.window_size = window_size
        self.sigma = sigma
        self.boundary: BoundaryPolicy = boundary
        self.data_range = data_range
        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2

    def _validate(self, prediction: Tensor, target: Tensor) -> None:
        """Check that both tensors form a comparable image batch.

        Args:
            prediction (Tensor): Reconstructed batch of shape ``(B, C, H, W)``.
            target (Tensor): Reference batch of shape ``(B, C, H, W)``.

        Returns:
            None: Validation passes silently.

        Raises:
            ValueError: If the ranks, shapes, or spatial extent are invalid.
        """
        if prediction.ndim != 4 or target.ndim != 4:
            raise ValueError("SSIM expects two four-dimensional (B, C, H, W) tensors")
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction shape {tuple(prediction.shape)} does not match "
                f"target shape {tuple(target.shape)}"
            )
        height, width = prediction.shape[2], prediction.shape[3]
        if height < self.window_size or width < self.window_size:
            raise ValueError(
                f"spatial extent {(height, width)} is smaller than the "
                f"{self.window_size}x{self.window_size} window"
            )

    def _local_statistics(self, prediction: Tensor, target: Tensor) -> tuple[Tensor, ...]:
        """Compute local means, variances, and covariance under the policy.

        Args:
            prediction (Tensor): Float32 reconstructed batch of shape
                ``(B, C, H, W)``.
            target (Tensor): Float32 reference batch of shape ``(B, C, H, W)``.

        Returns:
            tuple[Tensor, ...]: ``(mean_x, mean_y, var_x, var_y, covariance)``,
                each with the policy-dependent spatial extent.
        """
        channels = prediction.shape[1]
        kernel = _gaussian_kernel(
            self.window_size,
            self.sigma,
            channels,
            prediction.dtype,
            prediction.device,
        )
        padding = 0 if self.boundary == "valid" else self.window_size // 2

        def conv(values: Tensor) -> Tensor:
            """Apply the depthwise Gaussian window under the active padding policy.

            Args:
                values (Tensor): Image batch of shape ``(B, C, H, W)``.

            Returns:
                Tensor: Filtered tensor with the policy-dependent extent.
            """
            return functional.conv2d(values, kernel, padding=padding, groups=channels)

        mean_x = conv(prediction)
        mean_y = conv(target)
        second_x = conv(prediction * prediction)
        second_y = conv(target * target)
        cross = conv(prediction * target)

        if self.boundary == "masked":
            ones = torch.ones_like(prediction[:, :1])
            mass = functional.conv2d(ones, kernel[:1], padding=padding).clamp_min(
                torch.finfo(ones.dtype).eps
            )
            mean_x = mean_x / mass
            mean_y = mean_y / mass
            second_x = second_x / mass
            second_y = second_y / mass
            cross = cross / mass

        var_x = (second_x - mean_x * mean_x).clamp_min(0.0)
        var_y = (second_y - mean_y * mean_y).clamp_min(0.0)
        covariance = cross - mean_x * mean_y
        return mean_x, mean_y, var_x, var_y, covariance

    def ssim_map(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Return the per-position SSIM map.

        Args:
            prediction (Tensor): Reconstructed batch of shape ``(B, C, H, W)`` in
                ``[0, 1]``.
            target (Tensor): Reference batch of shape ``(B, C, H, W)`` in
                ``[0, 1]``.

        Returns:
            Tensor: Float32 SSIM map. Shape is ``(B, C, H - k + 1, W - k + 1)``
                for ``valid`` and ``(B, C, H, W)`` for ``zero`` and ``masked``.

        Raises:
            ValueError: If the tensors do not form a comparable image batch.
        """
        self._validate(prediction, target)
        with torch.amp.autocast(device_type=prediction.device.type, enabled=False):
            x = prediction.float()
            y = target.float()
            mean_x, mean_y, var_x, var_y, covariance = self._local_statistics(x, y)
            numerator = (2.0 * mean_x * mean_y + self.c1) * (2.0 * covariance + self.c2)
            denominator = (mean_x * mean_x + mean_y * mean_y + self.c1) * (var_x + var_y + self.c2)
            return numerator / denominator

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Return the scalar SSIM loss ``1 - mean(SSIM)``.

        Args:
            prediction (Tensor): Reconstructed batch of shape ``(B, C, H, W)`` in
                ``[0, 1]``.
            target (Tensor): Reference batch of shape ``(B, C, H, W)`` in
                ``[0, 1]``.

        Returns:
            Tensor: Scalar float32 loss.
        """
        return 1.0 - self.ssim_map(prediction, target).mean()


@dataclass(frozen=True)
class LossBreakdown:
    """Decomposed multi-task loss used for logging and diagnostics.

    Args:
        total (Tensor): Scalar objective optimized by the trainer.
        classification (Tensor): Cross-entropy term.
        reconstruction (Tensor): Weighted reconstruction term.
        mse (Tensor): Pixel mean squared error.
        ssim (Tensor): Structural dissimilarity ``1 - SSIM``.
    """

    total: Tensor
    classification: Tensor
    reconstruction: Tensor
    mse: Tensor
    ssim: Tensor


class MultiTaskLoss(nn.Module):
    """Composite objective ``CE + lambda * (alpha * MSE + (1 - alpha) * SSIM)``.

    Args:
        alpha (float): Weight of MSE inside the reconstruction term. Must lie in
            ``[0, 1]``.
        recon_lambda (float): Weight of the reconstruction term relative to
            cross-entropy. Must be positive.
        ssim_loss (SSIMLoss): Structural loss carrying the boundary policy.
        class_weights (Tensor | None): Optional per-class weights of shape
            ``(num_classes,)`` for inverse-frequency cross-entropy.

    Raises:
        ValueError: If ``alpha`` or ``recon_lambda`` is out of range.
    """

    def __init__(
        self,
        alpha: float,
        recon_lambda: float,
        ssim_loss: SSIMLoss,
        class_weights: Tensor | None = None,
    ) -> None:
        """Validate and store the loss weights.

        Args:
            alpha (float): Weight of MSE inside the reconstruction term.
            recon_lambda (float): Weight of the reconstruction term.
            ssim_loss (SSIMLoss): Structural loss instance.
            class_weights (Tensor | None): Optional per-class weights.

        Returns:
            None: Weights and the buffer are stored on the module.

        Raises:
            ValueError: If a weight is outside its admissible range.
        """
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if recon_lambda <= 0.0:
            raise ValueError("recon_lambda must be strictly positive")
        self.alpha = alpha
        self.recon_lambda = recon_lambda
        self.ssim_loss = ssim_loss
        self.register_buffer("class_weights", class_weights)

    def classification_term(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute the cross-entropy term for one batch.

        Args:
            logits (Tensor): Class logits of shape ``(B, num_classes)``.
            labels (Tensor): Integer class targets of shape ``(B,)``.

        Returns:
            Tensor: Scalar float32 cross-entropy, optionally class weighted.

        Raises:
            ValueError: If the ranks or batch dimensions are inconsistent.
        """
        if logits.ndim != 2 or labels.ndim != 1:
            raise ValueError("logits must be (B, K) and labels must be (B,)")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("logits and labels must share the batch dimension")
        with torch.amp.autocast(device_type=logits.device.type, enabled=False):
            return functional.cross_entropy(
                logits.float(),
                labels,
                weight=cast("Tensor | None", self.class_weights),
            )

    def reconstruction_term(
        self,
        reconstruction: Tensor,
        target: Tensor,
        recon_lambda: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the reconstruction components for one batch.

        Args:
            reconstruction (Tensor): Decoder output of shape ``(B, C, H, W)`` in
                ``[0, 1]``.
            target (Tensor): Reference batch of shape ``(B, C, H, W)`` in
                ``[0, 1]``.
            recon_lambda (float | None): Optional weight override for a scheduled
                decay. When omitted the configured weight is used.

        Returns:
            tuple[Tensor, Tensor, Tensor]: ``(mse, ssim_dissimilarity,
                weighted_reconstruction)``, each a scalar.

        Raises:
            ValueError: If the shapes differ or the effective weight is not
                strictly positive.
        """
        if reconstruction.shape != target.shape:
            raise ValueError(
                f"reconstruction shape {tuple(reconstruction.shape)} does not match "
                f"target shape {tuple(target.shape)}"
            )
        weight = self.recon_lambda if recon_lambda is None else recon_lambda
        if weight <= 0.0:
            raise ValueError("recon_lambda must be strictly positive")
        with torch.amp.autocast(device_type=reconstruction.device.type, enabled=False):
            mse = functional.mse_loss(reconstruction.float(), target.float())
            ssim = self.ssim_loss(reconstruction, target)
            composite = self.alpha * mse + (1.0 - self.alpha) * ssim
            weighted = weight * composite
        return mse, ssim, weighted

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        reconstruction: Tensor,
        target: Tensor,
        recon_lambda: float | None = None,
    ) -> LossBreakdown:
        """Compute every loss component for one batch.

        Args:
            logits (Tensor): Class logits of shape ``(B, num_classes)``.
            labels (Tensor): Integer class targets of shape ``(B,)``.
            reconstruction (Tensor): Decoder output of shape ``(B, C, H, W)`` in
                ``[0, 1]``.
            target (Tensor): Reference batch of shape ``(B, C, H, W)`` in
                ``[0, 1]``.
            recon_lambda (float | None): Optional scheduled weight override.

        Returns:
            LossBreakdown: Scalar total plus each additive component.

        Raises:
            ValueError: If the batch dimensions, image shapes, or weight are
                invalid.
        """
        classification = self.classification_term(logits, labels)
        mse, ssim, reconstruction_loss = self.reconstruction_term(
            reconstruction,
            target,
            recon_lambda,
        )
        total = classification + reconstruction_loss
        return LossBreakdown(
            total=total,
            classification=classification,
            reconstruction=reconstruction_loss,
            mse=mse,
            ssim=ssim,
        )
