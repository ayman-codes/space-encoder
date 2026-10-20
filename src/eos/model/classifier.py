"""Classification head over the shared bottleneck embedding.

The head is deliberately small: ``latent_dim -> head_hidden -> num_classes`` with
ReLU and dropout. Both linear layers are named attributes because the hidden
projection is the ASVD target and the logits projection is permanently excluded
from factorization.
"""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from eos.config import ModelConfig


class ClassificationHead(nn.Module):
    """Two-layer perceptron over the bottleneck embedding.

    Args:
        config (ModelConfig): Architecture configuration providing
            ``latent_dim``, ``head_hidden``, ``num_classes``, and ``dropout``.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the hidden projection and the logits projection.

        Args:
            config (ModelConfig): Architecture configuration.

        Returns:
            None: Layers are registered on the module.
        """
        super().__init__()
        self.config = config
        self.hidden = nn.Linear(config.latent_dim, config.head_hidden)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.logits = nn.Linear(config.head_hidden, config.num_classes)

    def forward(self, z: Tensor) -> Tensor:
        """Classify a batch of embeddings.

        Args:
            z (Tensor): Embedding of shape ``(B, latent_dim)``.

        Returns:
            Tensor: Unnormalized logits of shape ``(B, num_classes)``.

        Raises:
            ValueError: If the embedding geometry is invalid.
        """
        if z.ndim != 2 or z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"classification head expects (B, {self.config.latent_dim}), "
                f"received {tuple(z.shape)}"
            )
        return cast(Tensor, self.logits(self.dropout(self.activation(self.hidden(z)))))
