import dataclasses
import math
from typing import Callable

import torch
import torch.nn as nn

# Assuming MLP and MLPConfig are defined elsewhere following similar conventions.
from rl_the_spire.models.common.mlp import MLP, MLPConfig


@dataclasses.dataclass
class GridToSequenceConfig:
    """
    Configuration for the GridToSequenceTransformerBlock.

    Attributes:
        n_embed (int): Dimensionality of grid token embeddings (EMBED).
        n_pos_embed (int): Dimensionality of the positional embeddings provided as query.
        n_heads (int): Number of attention heads.
        attn_dropout (float): Dropout probability for the attention weights.
        resid_dropout (float): Dropout probability for the residual connections.
        mlp_dropout (float): Dropout probability in the MLP.
        linear_size_multiplier (int): Multiplier for the hidden layer size in the MLP.
        activation (Callable[[torch.Tensor], torch.Tensor]): Activation function used in the MLP.
        dtype (torch.dtype): Data type for parameters.
        device (str): Device to store the parameters.
        init_std (float): Standard deviation for weight initialization.
    """

    n_embed: int
    n_pos_embed: int
    n_heads: int
    attn_dropout: float
    resid_dropout: float
    mlp_dropout: float
    linear_size_multiplier: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    dtype: torch.dtype
    device: str
    init_std: float
    ln_eps: float = 1e-5


class GridToSequence(nn.Module):
    """
    Transforms grid features of shape (B, N, M, EMBED) together with query positional embeddings
    of shape (B, L, POSITIONAL_EMBEDDING) into a sequence representation of shape (B, L, EMBED)
    using multi-head cross-attention followed by an MLP, each with residual connections.

    Linear projections are implemented using learned parameters (via torch.zeros and nn.Parameter)
    and all linear operations are performed with einsum.
    """

    def __init__(self, config: GridToSequenceConfig):
        super().__init__()
        self.config = config
        d = config.n_embed
        H = config.n_heads
        self.head_size = d // H

        # Query projection: from (n_pos_embed) -> (n_embed)
        self.query_proj = nn.Parameter(
            torch.zeros(
                (config.n_pos_embed, d),
                dtype=config.dtype,
                device=config.device,
                requires_grad=True,
            )
        )
        # Key projection: from (n_embed) -> (n_embed)
        self.key_proj = nn.Parameter(
            torch.zeros(
                (d, d), dtype=config.dtype, device=config.device, requires_grad=True
            )
        )
        # Value projection: from (n_embed) -> (n_embed)
        self.value_proj = nn.Parameter(
            torch.zeros(
                (d, d), dtype=config.dtype, device=config.device, requires_grad=True
            )
        )
        # Output projection: from (n_embed) -> (n_embed)
        self.out_proj = nn.Parameter(
            torch.zeros(
                (d, d), dtype=config.dtype, device=config.device, requires_grad=True
            )
        )

        # MLP for post-attention processing.
        self.mlp_config = MLPConfig(
            n_in=d,
            n_out=d,
            linear_size_multiplier=config.linear_size_multiplier,
            activation=config.activation,
            dropout=config.mlp_dropout,
            dtype=config.dtype,
            device=config.device,
            init_std=config.init_std,
        )
        self.mlp = MLP(self.mlp_config)

        # Dropout modules.
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        # Layer normalizations.
        self.ln_grid = nn.LayerNorm(
            d, eps=config.ln_eps, device=config.device, dtype=config.dtype
        )
        self.ln_query = nn.LayerNorm(
            config.n_pos_embed,
            eps=config.ln_eps,
            device=config.device,
            dtype=config.dtype,
        )
        self.ln_mlp = nn.LayerNorm(
            d, eps=config.ln_eps, device=config.device, dtype=config.dtype
        )

    def init_weights(self) -> None:
        """
        Initialize learned parameters with a normal distribution using init_std.
        """
        torch.nn.init.normal_(self.query_proj, std=self.config.init_std)
        torch.nn.init.normal_(self.key_proj, std=self.config.init_std)
        torch.nn.init.normal_(self.value_proj, std=self.config.init_std)
        torch.nn.init.normal_(self.out_proj, std=self.config.init_std)
        self.mlp.init_weights()

    def forward(self, grid: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid (torch.Tensor): Grid features of shape (B, N, M, EMBED).
            query_pos (torch.Tensor): Positional embeddings of shape (B, L, POSITIONAL_EMBEDDING).

        Returns:
            torch.Tensor: Output sequence of shape (B, L, EMBED).
        """
        B, N, M, d = grid.shape
        # Flatten grid to shape (B, N*M, d) and normalize.
        grid_flat = grid.view(B, N * M, d)
        grid_norm = self.ln_grid(grid_flat)

        # Compute keys and values from the grid.
        # Keys: (B, N*M, d) = einsum("btd,de->bte", grid_norm, key_proj)
        K = torch.einsum("btd,de->bte", grid_norm, self.key_proj)
        # Values: (B, N*M, d)
        V = torch.einsum("btd,de->bte", grid_norm, self.value_proj)

        # Process queries: normalize and project.
        # query_pos: (B, L, n_pos_embed) -> (B, L, n_pos_embed) normalized.
        query_norm = self.ln_query(query_pos)
        # Project queries: (B, L, d) = einsum("bld,de->ble", query_norm, query_proj)
        Q = torch.einsum("bld,de->ble", query_norm, self.query_proj)

        # Reshape Q, K, V to multi-head format.
        H = self.config.n_heads
        head_size = self.head_size
        # Q: (B, L, d) -> (B, L, H, head_size) -> (B, H, L, head_size)
        Q = Q.view(B, -1, H, head_size).transpose(1, 2)
        # K: (B, N*M, d) -> (B, N*M, H, head_size) -> (B, H, N*M, head_size)
        K = K.view(B, -1, H, head_size).transpose(1, 2)
        # V: (B, N*M, d) -> (B, N*M, H, head_size) -> (B, H, N*M, head_size)
        V = V.view(B, -1, H, head_size).transpose(1, 2)

        # Scaled dot-product attention.
        scale = 1.0 / math.sqrt(head_size)
        attn_scores = torch.einsum("bhld,bhmd->bhlm", Q, K) * scale  # (B, H, L, N*M)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        # Compute attention output.
        attn_output = torch.einsum(
            "bhlm,bhmd->bhld", attn_probs, V
        )  # (B, H, L, head_size)
        # Merge heads: (B, H, L, head_size) -> (B, L, d)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, d)

        # Apply output projection.
        attn_output = torch.einsum("bld,de->ble", attn_output, self.out_proj)
        attn_output = self.resid_dropout(attn_output)

        # Residual connection: add the projected queries.
        # Recompute projected queries for residual connection.
        Q_proj = torch.einsum("bld,de->ble", query_norm, self.query_proj)
        x = Q_proj + attn_output

        # MLP stage.
        x_norm = self.ln_mlp(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.resid_dropout(mlp_out)

        return x
