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
from rl_the_spire.models.transformers.transformer_body import (
    TransformerBody,
    TransformerBodyConfig,
)


@dataclasses.dataclass
class PermutationSequenceEncoderConfig:
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
    linear_size_multiplier: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    sigma_output: bool


class PermutationSequenceEncoder(torch.nn.Module):
    def __init__(self, config: PermutationSequenceEncoderConfig):
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

    def init_weights(self) -> None:
        self.embedder.init_weights()
        self.transformer_body.init_weights()

    def forward(
        self, pos_encoder: PositionalSequenceEncoder, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Forward pass of PermutationSequenceEncoder.

        Args:
            pos_encoder: Positional sequence encoder to use for encoding position information
            x: [batch_size, permutation_size] tensor of integers

        Returns:
            If sigma_output is True:
                mus: tensor of shape [batch_size, sequence_length, embed_dim] or
                     [batch_size, output_projection_size] if output_projection is used
                logvars: tensor of the same shape as mus
            Otherwise:
                outputs: tensor of shape [batch_size, sequence_length, embed_dim] or
                        [batch_size, output_projection_size] if output_projection is used
        """
        # First, use the embedder to convert the permutation indices to embeddings
        embeds = self.embedder(pos_encoder, x)

        # Pass through the transformer
        sequence_outputs = self.transformer_body(
            embeds, extra_embed=torch.zeros_like(embeds[:, :, :0])
        )
        if self.config.sigma_output:
            # Split the outputs into mu and logvar
            n_embed = sequence_outputs.size(-1) // 2
            mus = sequence_outputs[..., :n_embed]
            logvars = sequence_outputs[..., n_embed:]
            return mus, logvars
        else:
            return sequence_outputs, torch.ones_like(sequence_outputs)
