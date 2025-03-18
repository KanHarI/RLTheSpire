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
class DistributionSamplerConfig:
    n_layers: int
    n_heads: int
    n_symbols: int
    n_distribution_tokens: int
    n_samples_per_distribution: int
    n_embed: int
    dtype: torch.dtype
    device: torch.device
    attn_dropout: float
    resid_dropout: float
    mlp_dropout: float
    init_std: float
    ln_eps: float
    activation: Callable[[torch.Tensor], torch.Tensor]
    linear_size_multiplier: int


class DistributionSampler(torch.nn.Module):
    def __init__(self, config: DistributionSamplerConfig):
        super().__init__()
        self.config = config
        transformer_body_config = TransformerBodyConfig(
            n_layers=config.n_layers,
            n_embed=config.n_embed,
            n_extra_embed=config.n_embed,
            n_heads=config.n_heads,
            attn_dropout=config.attn_dropout,
            resid_dropout=config.resid_dropout,
            mlp_dropout=config.mlp_dropout,
            init_std=config.init_std,
            ln_eps=config.ln_eps,
            activation=config.activation,
            dtype=config.dtype,
            device=config.device,
            linear_size_multiplier=config.linear_size_multiplier,
        )
        self.transformer_body = TransformerBody(transformer_body_config)

    def init_weights(self) -> None:
        self.transformer_body.init_weights()

    def forward(
        self,
        distribution_embeddings: torch.Tensor,
        positional_sequence_encoder: PositionalSequenceEncoder,
    ) -> torch.Tensor:
        """
        Args:
            distribution_embeddings: (batch_size, n_distribution_tokens, n_embed)
        """
        B, _, _ = distribution_embeddings.shape
        randn_samples = torch.randn(
            B,
            self.config.n_samples_per_distribution,
            self.config.n_symbols + self.config.n_distribution_tokens,
            self.config.n_embed,
            device=self.config.device,
            dtype=self.config.dtype,
        )

        x = torch.zeros_like(randn_samples)

        # Duplicate the distribution embeddings for each sample
        distribution_embeddings = distribution_embeddings.unsqueeze(1).expand(
            -1, self.config.n_samples_per_distribution, -1, -1
        )

        # Add the distribution embeddings to the projected samples
        x[:, :, -self.config.n_distribution_tokens :, :] += distribution_embeddings

        x = positional_sequence_encoder(x)
        x = self.transformer_body(x)

        # Distribution logits are the last n_symbols tokens in the last layer, taking the last dimension
        distribution_logits = x[:, :, : self.config.n_symbols, -1]

        # Now softmax over the last dimension
        distribution_logits = torch.nn.functional.softmax(distribution_logits, dim=-1)

        return distribution_logits
