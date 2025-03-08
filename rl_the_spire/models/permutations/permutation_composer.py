import dataclasses
from typing import Callable

import torch

from rl_the_spire.models.transformers.conv_transformer_block import (
    ConvTransformerBlock,
    ConvTransformerBlockConfig,
)
from rl_the_spire.models.transformers.conv_transformer_body import (
    ConvTransformerBody,
    ConvTransformerBodyConfig,
)


@dataclasses.dataclass
class PermutationComposerConfig:
    """
    Configuration for the GridComposer module.

    Attributes:
        n_embed (int): Dimensionality of the token embeddings.
        n_heads (int): Number of attention heads.
        attn_dropout (float): Dropout probability for the attention modules.
        resid_dropout (float): Dropout probability for the residual connections.
        mlp_dropout (float): Dropout probability for the MLP.
        linear_size_multiplier (int): Multiplier for the hidden layer size in the MLP.
        activation (Callable[[torch.Tensor], torch.Tensor]): Activation function used in the MLP.
        dtype (torch.dtype): Data type for parameters.
        device (torch.device): Device on which the parameters will be stored.
        init_std (float): Standard deviation for weight initialization.
        n_layers (int): Number of layers after merging the grid.
        ln_eps (float): Epsilon value for layer normalization.
    """

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
    n_layers: int
    ln_eps: float = 1e-5


class PermutationComposer(torch.nn.Module):
    def __init__(self, config: PermutationComposerConfig) -> None:
        super().__init__()
        self.config = config

        # --- Positional encodings ---
        # Shape (2, n_embed). We'll keep them constant (not learned).
        # If you prefer them trainable, use nn.Parameter instead.
        self.pos_enc = torch.nn.Parameter(
            torch.zeros(
                2,
                config.n_embed,
                device=config.device,
                dtype=config.dtype,
                requires_grad=True,
            )
        )

        first_block_config = ConvTransformerBlockConfig(
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

        self.first_block = ConvTransformerBlock(first_block_config)

        # --- Linear projections for each stream ---
        # We will initialize these to half the identity plus normal noise.
        # self.proj_x = nn.Linear(config.n_embed, config.n_embed, bias=False,
        #                         device=config.device, dtype=config.dtype)
        # self.proj_y = nn.Linear(config.n_embed, config.n_embed, bias=False,
        #                         device=config.device, dtype=config.dtype)
        self.proj_x = torch.nn.Parameter(
            torch.zeros(
                config.n_embed,
                config.n_embed,
                device=config.device,
                dtype=config.dtype,
                requires_grad=True,
            )
        )
        self.proj_y = torch.nn.Parameter(
            torch.zeros(
                config.n_embed,
                config.n_embed,
                device=config.device,
                dtype=config.dtype,
                requires_grad=True,
            )
        )

        # --- Second ConvTransformerBlock for the merged outputs (B, N, M, E) ---
        body_after_merge_config = ConvTransformerBodyConfig(
            n_blocks=config.n_layers,
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
        self.body_after_merge = ConvTransformerBody(body_after_merge_config)

    def init_weights(self) -> None:
        """Initialize weights of the transformer blocks (if desired)."""
        torch.nn.init.normal_(self.pos_enc, mean=0.0, std=self.config.init_std)
        self.first_block.init_weights()
        self.body_after_merge.init_weights()
        # Initialize x_proj and y_proj to half the identity plus normal noise.
        torch.nn.init.normal_(self.proj_x, mean=0.0, std=self.config.init_std)
        torch.nn.init.normal_(self.proj_y, mean=0.0, std=self.config.init_std)
        self.proj_x.data.copy_(
            0.5
            * torch.eye(
                self.config.n_embed, device=self.config.device, dtype=self.config.dtype
            )
            + self.proj_x.data
        )
        self.proj_y.data.copy_(
            0.5
            * torch.eye(
                self.config.n_embed, device=self.config.device, dtype=self.config.dtype
            )
            + self.proj_y.data
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:

        1. x, y are each (B, N, M, E).
        2. Add constant positional encodings to distinguish them.
        3. Concat along the row dimension -> shape (B, 2N, M, E).
        4. Pass through a ConvTransformerBlock.
        5. Split back into (B, N, M, E) chunks.
        6. Apply separate linear projections and sum -> shape (B, N, M, E).
        7. Pass through a second ConvTransformerBlock.
        8. Return the result.
        """
        B, N, M, E = x.shape

        # (1) Add positional encodings so that the block knows which 'slot' is x vs y.
        #     pos_enc has shape (2, E), so we broadcast across (B, N, M).
        x = x + self.pos_enc[0].view(1, 1, 1, E)
        y = y + self.pos_enc[1].view(1, 1, 1, E)

        # (2) Concatenate along the row dimension: shape = (B, 2N, M, E)
        z = torch.cat([x, y], dim=1)

        # (3) Pass through first transformer block
        z = self.first_block(z)  # still (B, 2N, M, E)

        # (4) Split into two (B, N, M, E) tensors
        xz, yz = z[:, :N, :, :], z[:, N:, :, :]

        # (5) Linear projections on each, then sum
        xz = torch.einsum("...i,ij->...j", xz, self.proj_x)
        yz = torch.einsum("...i,ij->...j", yz, self.proj_y)
        merged = xz + yz  # (B, N, M, E)

        # (6) Pass the merged result through second transformer block
        out = self.body_after_merge(merged)

        return out  # type: ignore
