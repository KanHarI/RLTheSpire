import dataclasses
from typing import Callable

import torch

from rl_the_spire.models.position_encodings.positional_sequence_encoder import (
    PositionalSequenceEncoder,
)
from rl_the_spire.models.transformers.transformer_body import (
    TransformerBody,
    TransformerBodyConfig,
)


@dataclasses.dataclass
class DirectProbabilityDistributionEmbedderConfig:
    n_symbols: int
    distribution_n_tokens: int
    n_embed: int
    n_layers: int
    n_heads: int
    attn_dropout: float
    resid_dropout: float
    mlp_dropout: float
    init_std: float
    ln_eps: float
    device: torch.device
    dtype: torch.dtype
    activation: Callable[[torch.Tensor], torch.Tensor]


class DirectProbabilityDistributionEmbedder(torch.nn.Module):
    def __init__(self, config: DirectProbabilityDistributionEmbedderConfig):
        super().__init__()
        self.config = config

        self.symbol_embeddings = torch.nn.Parameter(
            torch.zeros(
                self.config.n_symbols + self.config.distribution_n_tokens,
                self.config.n_embed - 1,  # Last dimension is for the distribution
            ),
            requires_grad=True,
        )

        transformer_body_config = TransformerBodyConfig(
            n_layers=config.n_layers,
            n_embed=config.n_embed,
            n_extra_embed=0,
            n_heads=config.n_heads,
            attn_dropout=config.attn_dropout,
            resid_dropout=config.resid_dropout,
            dtype=config.dtype,
            device=config.device,
            linear_size_multiplier=1,
            activation=config.activation,
            mlp_dropout=config.mlp_dropout,
            init_std=config.init_std,
            ln_eps=config.ln_eps,
        )
        self.transformer_body = TransformerBody(transformer_body_config)

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

        x = torch.zeros(
            B,
            self.config.n_symbols + self.config.distribution_n_tokens,
            self.config.n_embed,
            device=self.config.device,
            dtype=self.config.dtype,
        )

        x = positional_sequence_encoder(x)

        # Add the distribution to the last dimension
        x[:, :, -1] += distribution

        # Add the symbol embeddings to the first n_symbols dimensions
        x[:, :, :-1] += self.symbol_embeddings[used_symbols[:, :-1]]

        x = self.transformer_body(x)

        return x
