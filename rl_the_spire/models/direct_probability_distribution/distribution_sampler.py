import dataclasses

import torch

from rl_the_spire.models.position_encodings.positional_sequence_encoder import (
    PositionalSequenceEncoder,
)


@dataclasses.dataclass
class DistributionSamplerConfig:
    n_symbols: int
    n_distribution_tokens: int
    n_samples_per_distribution: int
    n_embed: int
    dtype: torch.dtype
    device: torch.device


class DistributionSampler(torch.nn.Module):
    def __init__(self, config: DistributionSamplerConfig):
        super().__init__()
        self.config = config
        self.randn_projection = torch.nn.Parameter(
            torch.zeros(
                config.n_embed,
                config.n_embed,
                device=config.device,
                dtype=config.dtype,
            ),
            requires_grad=True,
        )

    def init_weights(self):
        eye = torch.eye(
            self.config.n_embed, device=self.config.device, dtype=self.config.dtype
        )
        normal = torch.randn(
            self.config.n_embed,
            self.config.n_embed,
            device=self.config.device,
            dtype=self.config.dtype,
        )
        self.randn_projection.data.copy_(eye + normal)

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

        x = torch.einsum("bnli,ij->bnlj", randn_samples, self.randn_projection)

        # Unsqueeze the distribution embeddings to [batch_size, 1, n_distribution_tokens, n_embed]
        distribution_embeddings = distribution_embeddings.unsqueeze(1)

        # Add the distribution embeddings to the projected samples
        x[:, :, -self.config.n_distribution_tokens :, :] += distribution_embeddings

        x = positional_sequence_encoder(x)
