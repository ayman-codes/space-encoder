"""Model compositions used by the ablation arms.

``HybridCAE`` is the multi-task model: one shared encoder feeding a
classification head and a reconstruction decoder. ``PureCNN`` and ``ConvAE``
reuse the exact same component classes so the ablation isolates the objective,
not the architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from torch import Tensor, nn

from eos.config import ModelConfig
from eos.model.classifier import ClassificationHead
from eos.model.decoder import Decoder
from eos.model.encoder import Encoder


@dataclass(frozen=True)
class ModelOutput:
    """Forward-pass result of the hybrid model.

    Args:
        logits (Tensor): Class logits of shape ``(B, num_classes)``.
        reconstruction (Tensor): Decoder output of shape
            ``(B, in_channels, input_size, input_size)`` in ``[0, 1]``.
        embedding (Tensor): Bottleneck of shape ``(B, latent_dim)``.
    """

    logits: Tensor
    reconstruction: Tensor
    embedding: Tensor


def _count(module: nn.Module) -> int:
    """Count trainable parameters in a module.

    Args:
        module (nn.Module): Any module.

    Returns:
        int: Number of parameters with ``requires_grad`` enabled.
    """
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class HybridCAE(nn.Module):
    """Shared encoder with classification and reconstruction heads.

    Args:
        config (ModelConfig): Architecture configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the encoder, decoder, and classification head.

        Args:
            config (ModelConfig): Architecture configuration.

        Returns:
            None: Components are registered on the module.
        """
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.classifier = ClassificationHead(config)

    def encode(self, x: Tensor) -> Tensor:
        """Return the bottleneck embedding.

        Args:
            x (Tensor): Input batch of shape ``(B, C, H, W)``.

        Returns:
            Tensor: Embedding of shape ``(B, latent_dim)``.
        """
        return cast(Tensor, self.encoder(x))

    def reconstruct(self, z: Tensor) -> Tensor:
        """Reconstruct patches from embeddings.

        Args:
            z (Tensor): Embedding of shape ``(B, latent_dim)``.

        Returns:
            Tensor: Reconstruction of shape ``(B, C, H, W)`` in ``[0, 1]``.
        """
        return cast(Tensor, self.decoder(z))

    def classify(self, z: Tensor) -> Tensor:
        """Classify embeddings.

        Args:
            z (Tensor): Embedding of shape ``(B, latent_dim)``.

        Returns:
            Tensor: Logits of shape ``(B, num_classes)``.
        """
        return cast(Tensor, self.classifier(z))

    def forward(self, x: Tensor) -> ModelOutput:
        """Run both heads from one shared encoding.

        Args:
            x (Tensor): Input batch of shape ``(B, C, H, W)`` in ``[0, 1]``.

        Returns:
            ModelOutput: Logits, reconstruction, and the shared embedding.
        """
        embedding = self.encode(x)
        return ModelOutput(
            logits=self.classify(embedding),
            reconstruction=self.reconstruct(embedding),
            embedding=embedding,
        )

    def parameter_count(self) -> dict[str, int]:
        """Return a parameter budget per component.

        Returns:
            dict[str, int]: Counts for ``encoder``, ``decoder``, ``classifier``,
                and ``total``.
        """
        encoder = _count(self.encoder)
        decoder = _count(self.decoder)
        classifier = _count(self.classifier)
        return {
            "encoder": encoder,
            "decoder": decoder,
            "classifier": classifier,
            "total": encoder + decoder + classifier,
        }


class PureCNN(nn.Module):
    """Encoder plus classification head, without reconstruction.

    Args:
        config (ModelConfig): Architecture configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the shared encoder and the classification head.

        Args:
            config (ModelConfig): Architecture configuration.

        Returns:
            None: Components are registered on the module.
        """
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.classifier = ClassificationHead(config)

    def forward(self, x: Tensor) -> Tensor:
        """Return class logits for a batch.

        Args:
            x (Tensor): Input batch of shape ``(B, C, H, W)``.

        Returns:
            Tensor: Logits of shape ``(B, num_classes)``.
        """
        return cast(Tensor, self.classifier(self.encoder(x)))

    def parameter_count(self) -> dict[str, int]:
        """Return a parameter budget per component.

        Returns:
            dict[str, int]: Counts for ``encoder``, ``classifier``, and ``total``.
        """
        encoder = _count(self.encoder)
        classifier = _count(self.classifier)
        return {"encoder": encoder, "classifier": classifier, "total": encoder + classifier}


class ConvAE(nn.Module):
    """Encoder plus decoder, without classification.

    Args:
        config (ModelConfig): Architecture configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the shared encoder and the reconstruction decoder.

        Args:
            config (ModelConfig): Architecture configuration.

        Returns:
            None: Components are registered on the module.
        """
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct a batch.

        Args:
            x (Tensor): Input batch of shape ``(B, C, H, W)``.

        Returns:
            Tensor: Reconstruction of shape ``(B, C, H, W)`` in ``[0, 1]``.
        """
        return cast(Tensor, self.decoder(self.encoder(x)))

    def parameter_count(self) -> dict[str, int]:
        """Return a parameter budget per component.

        Returns:
            dict[str, int]: Counts for ``encoder``, ``decoder``, and ``total``.
        """
        encoder = _count(self.encoder)
        decoder = _count(self.decoder)
        return {"encoder": encoder, "decoder": decoder, "total": encoder + decoder}
