import dataclasses
from typing import Callable

import torch

from rl_the_spire.models.permutations.permutation_embedder import (
    PermutationEmbedder,
    PermutationEmbedderConfig,
)
from rl_the_spire.models.transformers.attention_to_tensor import (
    AttentionToTensor,
    AttentionToTensorConfig,
)
from rl_the_spire.models.transformers.conv_transformer_block import (
    ConvTransformerBlock,
    ConvTransformerBlockConfig,
)
from rl_the_spire.models.transformers.transformer_body import (
    TransformerBody,
    TransformerBodyConfig,
)


@dataclasses.dataclass
class PermutationEncoderConfig:
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


class PermutationEncoder(torch.nn.Module):
    def __init__(self, config: PermutationEncoderConfig):
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
            n_output_embed=config.n_output_embed * 2,  # For mus and logvars
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
        conv_transfomer_block_config = ConvTransformerBlockConfig(
            n_embed=config.n_output_embed * 2,  # For mus and logvars
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
        self.conv_transformer_block = ConvTransformerBlock(conv_transfomer_block_config)

    def init_weights(self) -> None:
        self.embedder.init_weights()
        self.transformer_body.init_weights()
        self.attention_to_tensor.init_weights()
        self.conv_transformer_block.init_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedder(x)
        x = self.transformer_body(x, extra_embed=torch.zeros_like(x[:, :, :1]))
        x = self.attention_to_tensor(x)
        x = self.conv_transformer_block(x)
        # x: (batch_size, n_output_heads, n_output_rows, n_output_columns, n_output_embed)
        # Turn into (batch_size, n_output_rows, n_output_columns, n_output_embed // 2, 2)
        # For mus, logvars
        x = x.view(
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3] // 2,
            2,
        )
        mus = x[:, :, :, 0]
        logsigmas = x[:, :, :, 1]
        return mus, logsigmas
