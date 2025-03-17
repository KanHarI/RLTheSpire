import dataclasses

import torch

from rl_the_spire.models.position_encodings.positional_sequence_encoder import (
    PositionalSequenceEncoder,
)


@dataclasses.dataclass
class DirectProbabilityDistributionEmbedderConfig:
    n_symbols: int
    n_embed: int
    device: torch.device
    dtype: torch.dtype
    init_std: float


class DirectProbabilityDistributionEmbedder(torch.nn.Module):
    def __init__(self, config: DirectProbabilityDistributionEmbedderConfig):
        super().__init__()
        self.config = config

        self.symbol_embeddings = torch.nn.Parameter(
            torch.zeros(
                self.config.n_symbols + 1,
                self.config.n_embed - 1,  # Last dimension is for the distribution
            ),
            requires_grad=True,
        )

    def init_weights(self) -> None:
        torch.nn.init.normal_(self.symbol_embeddings, 0.0, self.config.init_std)

    def forward(
        self,
        used_symbols: torch.Tensor,
        distribution: torch.Tensor,
        positional_sequence_encoder: PositionalSequenceEncoder,
    ) -> torch.Tensor:
        """
        Args:
            used_symbols: [batch_size, n_symbols + 1] tensor of integers
            distribution: [batch_size, n_symbols + 1] tensor of floats

        Returns:
            [batch_size, n_symbols + 1, n_embed] tensor of floats
        """
        B, _ = used_symbols.shape

        # Unsqueeze the positional embeddings to [1, n_symbols + 1, n_embed]
        x = torch.zeros(
            B,
            self.config.n_symbols + 1,
            self.config.n_embed,
            device=self.config.device,
            dtype=self.config.dtype,
        )

        x = positional_sequence_encoder(x)

        # Add the distribution to the last dimension
        x[:, :, -1] += distribution

        # Add the symbol embeddings to the first n_symbols dimensions
        x[:, :, :-1] += self.symbol_embeddings[used_symbols[:, :-1]]

        return x
