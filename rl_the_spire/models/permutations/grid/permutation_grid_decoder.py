import dataclasses
from typing import Callable

import torch

from rl_the_spire.models.position_encodings.positional_sequence_encoder import (
    PositionalSequenceEncoder,
)
from rl_the_spire.models.transformers.conv_transformer_block import (
    ConvTransformerBlock,
    ConvTransformerBlockConfig,
)
from rl_the_spire.models.transformers.grid_to_sequence import (
    GridToSequence,
    GridToSequenceConfig,
)
from rl_the_spire.models.transformers.transformer_body import (
    TransformerBody,
    TransformerBodyConfig,
)


@dataclasses.dataclass
class PermutationGridDecoderConfig:
    n_embed_grid: int
    n_embed_sequence: int
    n_grid_rows: int
    n_grid_columns: int
    n_max_permutation_size: int
    n_sequence_layers: int
    conv_transformer_n_heads: int
    sequence_n_heads: int
    linear_size_multiplier: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    dtype: torch.dtype
    device: torch.device
    init_std: float
    ln_eps: float
    attn_dropout: float
    resid_dropout: float
    mlp_dropout: float


class PermutationGridDecoder(torch.nn.Module):
    def __init__(self, config: PermutationGridDecoderConfig):
        super().__init__()
        self.config = config

        self.n_embed_row_pos = torch.nn.Parameter(
            torch.zeros(
                config.n_grid_rows,
                config.n_embed_grid,
                dtype=config.dtype,
                device=config.device,
                requires_grad=True,
            )
        )

        self.n_embed_col_pos = torch.nn.Parameter(
            torch.zeros(
                config.n_grid_columns,
                config.n_embed_grid,
                dtype=config.dtype,
                device=config.device,
                requires_grad=True,
            )
        )

        grid_transformer_config = ConvTransformerBlockConfig(
            n_embed=config.n_embed_grid,
            n_heads=config.conv_transformer_n_heads,
            attn_dropout=config.attn_dropout,
            resid_dropout=config.resid_dropout,
            mlp_dropout=config.mlp_dropout,
            linear_size_multiplier=config.linear_size_multiplier,
            activation=config.activation,
            dtype=config.dtype,
            device=config.device,
            init_std=config.init_std,
            ln_eps=config.ln_eps,
        )

        self.grid_transformer = ConvTransformerBlock(grid_transformer_config)

        grid_to_sequence_config = GridToSequenceConfig(
            grid_n_embed=config.n_embed_grid,
            seq_n_embed=config.n_embed_sequence,
            n_heads=config.conv_transformer_n_heads,
            attn_dropout=config.attn_dropout,
            resid_dropout=config.resid_dropout,
            mlp_dropout=config.mlp_dropout,
            linear_size_multiplier=config.linear_size_multiplier,
            activation=config.activation,
            dtype=config.dtype,
            device=config.device,
            init_std=config.init_std,
            ln_eps=config.ln_eps,
        )

        self.grid_to_sequence = GridToSequence(grid_to_sequence_config)

        sequence_transformer_config = TransformerBodyConfig(
            n_layers=config.n_sequence_layers,
            n_embed=config.n_embed_sequence,
            n_extra_embed=0,
            n_heads=config.sequence_n_heads,
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

        self.sequence_transformer = TransformerBody(sequence_transformer_config)

    def init_weights(self) -> None:
        self.grid_transformer.init_weights()
        self.grid_to_sequence.init_weights()
        self.sequence_transformer.init_weights()

    def forward(
        self, pos_encoder: PositionalSequenceEncoder, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the PermutationDecoder.

        Args:
            pos_encoder: Positional sequence encoder to use for encoding position information
            x: Tensor of shape (batch_size, n_grid_rows, n_grid_columns, n_embed_grid)

        Returns:
            Tensor of shape (batch_size, n_max_permutation_size, n_embed_sequence)
        """
        x = self.grid_transformer(x)

        # Create a dummy sequence tensor to apply positional encoding
        batch_size = x.shape[0]
        dummy_seq = torch.zeros(
            (
                batch_size,
                self.config.n_max_permutation_size,
                self.config.n_embed_sequence,
            ),
            device=x.device,
            dtype=x.dtype,
        )

        # Get positional encodings by applying the encoder to the dummy sequence
        # This will extract just the positional component
        pos_encodings = pos_encoder(dummy_seq) - dummy_seq

        # Apply grid to sequence transformation with positional encodings
        x = self.grid_to_sequence(x, pos_encodings)

        return self.sequence_transformer(x, extra_embed=torch.zeros_like(x[:, :, :0]))  # type: ignore
