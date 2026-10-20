"""Shared convolutional encoder producing the bottleneck embedding.

The spatial trace is fixed by the configuration: four stride-2 convolutions take
64x64 to 4x4, with stage widths ``base_channels * 2 ** (stage + 1)``. With the
default base of 32 that is ``64, 128, 256, 512``, so the flattened feature map is
8192-wide and matches the decoder's projection.
"""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from eos.config import ModelConfig

NEGATIVE_SLOPE = 0.2
"""LeakyReLU slope shared by the encoder and decoder blocks."""


class Encoder(nn.Module):
    """Strided convolutional encoder with a linear bottleneck projection.

    Args:
        config (ModelConfig): Architecture configuration. ``input_size`` must be
            divisible by ``2 ** downsample_stages``.

    Raises:
        ValueError: If the configured spatial trace does not land on an integer
            feature map.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the convolutional stages and the projection.

        Args:
            config (ModelConfig): Architecture configuration.

        Returns:
            None: Layers are registered on the module.

        Raises:
            ValueError: If the spatial trace is not integral or the flattened
                dimension is not positive.
        """
        super().__init__()
        divisor = 2**config.downsample_stages
        if config.input_size % divisor != 0:
            raise ValueError(
                f"input_size={config.input_size} is not divisible by {divisor} encoder stages"
            )
        self.config = config
        channels = [
            config.base_channels * 2 ** (index + 1) for index in range(config.downsample_stages)
        ]
        blocks: list[nn.Sequential] = []
        in_channels = config.in_channels
        for out_channels in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=config.kernel_size,
                        stride=2,
                        padding=config.kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(NEGATIVE_SLOPE),
                )
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.latent_size = config.input_size // divisor
        self.latent_channels = channels[-1]
        self.flattened_dim = self.latent_channels * self.latent_size * self.latent_size
        if self.flattened_dim <= 0:
            raise ValueError("configured encoder produces an empty feature map")
        self.projection = nn.Linear(self.flattened_dim, config.latent_dim)

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        """Return the pre-projection feature-map shape.

        Returns:
            tuple[int, int, int]: ``(channels, height, width)`` at the bottleneck.
        """
        return self.latent_channels, self.latent_size, self.latent_size

    def _validate_input(self, x: Tensor) -> None:
        """Check the input batch geometry.

        Args:
            x (Tensor): Candidate input batch of shape ``(B, C, H, W)``.

        Returns:
            None: Validation passes silently.

        Raises:
            ValueError: If the rank, channels, or spatial extent is wrong.
        """
        if x.ndim != 4:
            raise ValueError(f"encoder expects a 4-D (B, C, H, W) tensor, received {x.ndim}-D")
        _, channels, height, width = x.shape
        if channels != self.config.in_channels:
            raise ValueError(
                f"encoder expects {self.config.in_channels} channels, received {channels}"
            )
        if height != self.config.input_size or width != self.config.input_size:
            raise ValueError(
                f"encoder expects {self.config.input_size}x{self.config.input_size} patches, "
                f"received {height}x{width}"
            )

    def features(self, x: Tensor) -> Tensor:
        """Return the flattened feature map before the bottleneck projection.

        Args:
            x (Tensor): Input batch of shape ``(B, C, H, W)``.

        Returns:
            Tensor: Flattened features of shape
                ``(B, channels * latent_size * latent_size)``.

        Raises:
            ValueError: If the input geometry is invalid.
        """
        self._validate_input(x)
        values = x
        for block in self.blocks:
            values = block(values)
        return values.flatten(start_dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a batch into the bottleneck embedding.

        Args:
            x (Tensor): Input batch of shape ``(B, C, H, W)`` in ``[0, 1]``.

        Returns:
            Tensor: Embedding of shape ``(B, latent_dim)``.

        Raises:
            ValueError: If the input geometry is invalid.
        """
        return cast(Tensor, self.projection(self.features(x)))
