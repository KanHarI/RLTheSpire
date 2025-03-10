import dataclasses
from typing import Callable

import torch

from rl_the_spire.models.transformers.transformer_body import (
    TransformerBody,
    TransformerBodyConfig,
)


@dataclasses.dataclass
class PermutationSequenceDecoderConfig:
    n_embed: int
    n_max_permutation_size: int
    n_layers: int
    n_heads: int
    linear_size_multiplier: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    dtype: torch.dtype
    device: torch.device
    init_std: float
    ln_eps: float
    attn_dropout: float
    resid_dropout: float
    mlp_dropout: float


class PermutationSequenceDecoder(torch.nn.Module):
    def __init__(self, config: PermutationSequenceDecoderConfig):
        super().__init__()
        self.config = config

        # Transformer for sequence processing
        transformer_config = TransformerBodyConfig(
            n_layers=config.n_layers,
            n_embed=config.n_embed,
            n_extra_embed=0,
            n_heads=config.n_heads,
            attn_dropout=config.attn_dropout,
            resid_dropout=config.resid_dropout,
            dtype=config.dtype,
            device=config.device,
            linear_size_multiplier=config.linear_size_multiplier,
            activation=config.activation,
            mlp_dropout=config.mlp_dropout,
            init_std=config.init_std,
            ln_eps=config.ln_eps,
        )

        self.transformer = TransformerBody(transformer_config)

    def init_weights(self) -> None:
        self.transformer.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PermutationSequenceDecoder.

        Args:
            x: Tensor of shape (batch_size, sequence_length, n_embed)
                The input sequence representation

        Returns:
            Tensor of shape (batch_size, n_max_permutation_size, n_embed)
                The decoded permutation sequence
        """
        # Ensure the sequence length is correct
        if x.size(1) < self.config.n_max_permutation_size:
            # Pad sequence to match max permutation size
            padding = torch.zeros(
                (x.size(0), self.config.n_max_permutation_size - x.size(1), x.size(2)),
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, padding], dim=1)
        elif x.size(1) > self.config.n_max_permutation_size:
            raise ValueError(
                f"Sequence length {x.size(1)} is greater than the maximum permutation size {self.config.n_max_permutation_size}"
            )

        # Pass through transformer
        output = self.transformer(x, extra_embed=torch.zeros_like(x[:, :, :0]))

        return output  # type: ignore
