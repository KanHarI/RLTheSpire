import dataclasses
from typing import Callable

import torch

from rl_the_spire.models.transformers.conv_transformer_block import (
    ConvTransformerBlock,
    ConvTransformerBlockConfig,
)


@dataclasses.dataclass
class ConvTransformerBodyConfig:
    """
    Configuration for the ConvTransformerBody and its ConvTransformerBlocks.

    Attributes:
        n_blocks (int): Number of ConvTransformerBlock instances to stack.
        n_embed (int): Dimensionality of the token embeddings.
        n_heads (int): Number of attention heads.
        attn_dropout (float): Dropout probability for the attention modules.
        resid_dropout (float): Dropout probability for the residual connections.
        mlp_dropout (float): Dropout probability for the MLP.
        linear_size_multiplier (int): Multiplier for the hidden layer size in the MLP.
        activation (Callable[[torch.Tensor], torch.Tensor]): Activation function used in the MLP.
        dtype (torch.dtype): Data type for parameters.
        device (torch.device): Device to store the parameters.
        init_std (float): Standard deviation for weight initialization.
        ln_eps (float): Epsilon value for layer normalization.
    """

    n_blocks: int
    n_embed: int
    n_heads: int
    attn_dropout: float
    resid_dropout: float
    mlp_dropout: float
    linear_size_multiplier: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    dtype: torch.dtype
    device: torch.device
    init_std: float
    ln_eps: float = 1e-5


class ConvTransformerBody(torch.nn.Module):
    """
    A body that stacks multiple ConvTransformerBlocks.

    The configuration unpacks all necessary parameters for each ConvTransformerBlock,
    which is then applied sequentially to the input tensor.
    """

    def __init__(self, config: ConvTransformerBodyConfig):
        super().__init__()
        self.config = config

        # Create a block configuration from the unpacked parameters.
        block_config = ConvTransformerBlockConfig(
            n_embed=config.n_embed,
            n_heads=config.n_heads,
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

        # Stack the ConvTransformerBlocks.
        self.blocks = torch.nn.ModuleList(
            [ConvTransformerBlock(block_config) for _ in range(config.n_blocks)]
        )

    def init_weights(self) -> None:
        """
        Initialize weights for all ConvTransformerBlocks.
        """
        for block in self.blocks:
            block.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the stacked ConvTransformerBlocks.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, M, EMBED).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, M, EMBED) after processing by all blocks.
        """
        for block in self.blocks:
            x = block(x)
        return x
