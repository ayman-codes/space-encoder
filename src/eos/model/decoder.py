"""Symmetric transposed-convolution decoder.

The decoder mirrors the encoder's spatial trace exactly. Each
``ConvTranspose2d(kernel_size=4, stride=2, padding=1)`` doubles the feature map,
so a 4x4 bottleneck returns to 64x64 without interpolation artifacts. The final
1x1-style projection is a 3x3 convolution followed by a sigmoid, keeping outputs
inside the radiometric contract ``[0, 1]``.
"""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from eos.config import ModelConfig
from eos.model.encoder import NEGATIVE_SLOPE


class Decoder(nn.Module):
    """Transposed-convolution decoder with a sigmoid output.

    Args:
        config (ModelConfig): Architecture configuration. It must match the
            encoder that produced the embedding.

    Raises:
        ValueError: If the configured spatial trace cannot reproduce
            ``input_size``.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the projection and transposed-convolution stages.

        Args:
            config (ModelConfig): Architecture configuration.

        Returns:
            None: Layers are registered on the module.

        Raises:
            ValueError: If the decoder output extent differs from ``input_size``.
        """
        super().__init__()
        self.config = config
        divisor = 2**config.downsample_stages
        latent_size = config.input_size // divisor
        if latent_size * divisor != config.input_size:
            raise ValueError(
                f"decoder cannot reconstruct {config.input_size} from a {latent_size} bottleneck"
            )
        self.latent_size = latent_size
        encoder_channels = [
            config.base_channels * 2 ** (index + 1) for index in range(config.downsample_stages)
        ]
        self.latent_channels = encoder_channels[-1]
        self.flattened_dim = self.latent_channels * latent_size * latent_size
        self.projection = nn.Linear(config.latent_dim, self.flattened_dim)

        blocks: list[nn.Sequential] = []
        in_channels = self.latent_channels
        stride = 2
        kernel = 2 * stride
        for out_channels in [channels // 2 for channels in reversed(encoder_channels)]:
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel,
                        stride=2,
                        padding=(kernel - 2) // 2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(NEGATIVE_SLOPE),
                )
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Conv2d(
            in_channels,
            config.in_channels,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
        )
        self.activation = nn.Sigmoid()

    def _validate_latent(self, z: Tensor) -> None:
        """Check the embedding geometry.

        Args:
            z (Tensor): Candidate embedding of shape ``(B, latent_dim)``.

        Returns:
            None: Validation passes silently.

        Raises:
            ValueError: If the rank or width is wrong.
        """
        if z.ndim != 2:
            raise ValueError(f"decoder expects a 2-D (B, latent_dim) tensor, received {z.ndim}-D")
        if z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"decoder expects latent_dim={self.config.latent_dim}, received {z.shape[1]}"
            )

    def forward(self, z: Tensor) -> Tensor:
        """Reconstruct a batch of patches from embeddings.

        Args:
            z (Tensor): Embedding of shape ``(B, latent_dim)``.

        Returns:
            Tensor: Reconstruction of shape ``(B, in_channels, input_size,
                input_size)`` in ``[0, 1]``.

        Raises:
            ValueError: If the embedding geometry is invalid.
        """
        self._validate_latent(z)
        values = self.projection(z)
        values = values.reshape(
            z.shape[0],
            self.latent_channels,
            self.latent_size,
            self.latent_size,
        )
        for block in self.blocks:
            values = block(values)
        return cast(Tensor, self.activation(self.output(values)))
