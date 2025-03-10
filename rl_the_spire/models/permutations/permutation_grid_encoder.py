import dataclasses
from typing import Callable

import torch

from rl_the_spire.models.permutations.permutation_embedder import (
    PermutationEmbedder,
    PermutationEmbedderConfig,
)
from rl_the_spire.models.position_encodings.positional_sequence_encoder import (
    PositionalSequenceEncoder,
)
from rl_the_spire.models.transformers.attention_to_tensor import (
    AttentionToTensor,
    AttentionToTensorConfig,
)
from rl_the_spire.models.transformers.conv_transformer_body import (
    ConvTransformerBody,
    ConvTransformerBodyConfig,
)
from rl_the_spire.models.transformers.transformer_body import (
    TransformerBody,
    TransformerBodyConfig,
)


@dataclasses.dataclass
class PermutationGridEncoderConfig:
    n_max_permutation_size: int
    n_embed: int
    n_heads: int
    n_layers: int
    attn_dropout: float
    resid_dropout: float
    dtype: torch.dtype
    device: torch.device
    init_std: float
    mlp_dropout: float
    ln_eps: float
    n_output_heads: int
    n_output_embed: int
    n_output_rows: int
    n_output_columns: int
    conv_transformer_n_heads: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    linear_size_multiplier: int
    conv_transformer_blocks: int
    sigma_output: bool


class PermutationGridEncoder(torch.nn.Module):
    def __init__(self, config: PermutationGridEncoderConfig):
        super().__init__()
        self.config = config
        embedder_config = PermutationEmbedderConfig(
            n_max_permutation_size=config.n_max_permutation_size,
            n_embed=config.n_embed,
            dtype=config.dtype,
            device=config.device,
            init_std=config.init_std,
        )
        self.embedder = PermutationEmbedder(embedder_config)
        transformer_body_config = TransformerBodyConfig(
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
        self.transformer_body = TransformerBody(transformer_body_config)
        attention_to_tensor_config = AttentionToTensorConfig(
            n_embed=config.n_embed,
            n_output_embed=config.n_output_embed
            * (
                2 if config.sigma_output else 1
            ),  # For mus and logvars if sigma_output else just mus
            n_output_rows=config.n_output_rows,
            n_output_columns=config.n_output_columns,
            n_heads=config.n_output_heads,
            activation=config.activation,
            linear_size_multiplier=config.linear_size_multiplier,
            dtype=config.dtype,
            device=config.device,
            init_std=config.init_std,
            mlp_dropout=config.mlp_dropout,
        )
        self.attention_to_tensor = AttentionToTensor(attention_to_tensor_config)

        # Replace single ConvTransformerBlock with ConvTransformerBody
        conv_transformer_body_config = ConvTransformerBodyConfig(
            n_blocks=config.conv_transformer_blocks,
            n_embed=config.n_output_embed
            * (
                2 if config.sigma_output else 1
            ),  # For mus and logvars if sigma_output else just mus
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
        self.conv_transformer_body = ConvTransformerBody(conv_transformer_body_config)

    def init_weights(self) -> None:
        self.embedder.init_weights()
        self.transformer_body.init_weights()
        self.attention_to_tensor.init_weights()
        self.conv_transformer_body.init_weights()

    def forward(
        self, pos_encoder: PositionalSequenceEncoder, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of PermutationEncoder.

        Args:
            pos_encoder: Positional sequence encoder to use for encoding position information
            x: [batch_size, permutation_size] tensor of integers

        Returns:
            mus: [batch_size, n_output_rows, n_output_columns, n_output_embed] tensor of floats
            logvars: [batch_size, n_output_rows, n_output_columns, n_output_embed] tensor of floats
        """
        # First, use the embedder to convert the permutation indices to embeddings
        embeds = self.embedder(pos_encoder, x)

        # Pass through the transformer
        transformer_out = self.transformer_body(
            embeds, extra_embed=torch.zeros_like(embeds[:, :, :0])
        )

        # Convert to tensor
        tensor_out = self.attention_to_tensor(transformer_out)

        # shape: [batch_size, n_output_rows, n_output_columns, n_output_embed * (2 if sigma_output else 1)]

        # Apply ConvTransformerBody
        tensor_out = self.conv_transformer_body(tensor_out)

        if self.config.sigma_output:
            n_embed = self.config.n_output_embed
            mus = tensor_out[..., :n_embed]
            logvars = tensor_out[..., n_embed:]
            return mus, logvars
        else:
            return tensor_out, torch.ones_like(tensor_out)
