import dataclasses
from typing import Callable, Optional

import torch
import torch.nn as nn

# Import your MLP and MLPConfig; adjust the import path as needed.
from rl_the_spire.models.common.mlp import MLP, MLPConfig


@dataclasses.dataclass
class AttentionToTensorConfig:
    """
    Configuration for the AttentionToTensor module.

    Attributes:
        n_embed (int): Dimensionality of the input sequence embeddings.
        n_output_embed (int): Dimensionality of the output embedding for each grid cell.
                              Must be divisible by n_heads.
        n_output_rows (int): The number of rows in the output grid.
        n_output_columns (int): The number of columns in the output grid.
        n_heads (int): Number of attention heads.
        activation (Callable[[torch.Tensor], torch.Tensor]): Activation function for the MLP.
        linear_size_multiplier (int): Multiplier for the hidden size in the MLP.
        dtype (torch.dtype): Data type of the parameters.
        device (str): Device on which to place the parameters.
        init_std (float): Standard deviation for weight initialization.
        mlp_dropout (float): Dropout probability in the MLP.
    """

    n_embed: int
    n_output_embed: int
    n_output_rows: int
    n_output_columns: int
    n_heads: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    linear_size_multiplier: int
    dtype: torch.dtype
    device: torch.device
    init_std: float
    mlp_dropout: float


class AttentionToTensor(nn.Module):
    """
    Encodes an unordered sequence of vectors into a grid tensor of shape
    (batch_size, n_output_rows, n_output_columns, n_output_embed) using attention.

    Instead of a single query parameter, this module learns two query tensors:
      - row_query of shape (n_output_rows, n_output_embed // 2)
      - col_query of shape (n_output_columns, n_output_embed // 2)
    For a grid cell at position (m, n), the query is formed by concatenating the mth entry of row_query
    with the nth entry of col_query, then projecting it with a learned linear map.

    Keys and values are computed from the input sequence. For each grid cell, attention is computed over the sequence
    and the attended values are aggregated. An MLP (with a residual connection) is applied to the aggregated output.
    """

    def __init__(self, config: AttentionToTensorConfig):
        super().__init__()
        assert (
            config.n_output_embed % config.n_heads == 0
        ), "n_output_embed must be divisible by n_heads"
        self.config = config

        # Map input embeddings to keys and values.
        # The projection outputs 2 * n_output_embed so that we can split into keys and values.
        self.transformer_to_kv = nn.Parameter(
            torch.zeros(
                (self.config.n_embed, self.config.n_output_embed * 2),
                dtype=self.config.dtype,
                device=self.config.device,
                requires_grad=True,
            )
        )
        self.transformer_to_kv_bias = nn.Parameter(
            torch.zeros(
                (self.config.n_output_embed * 2,),
                dtype=self.config.dtype,
                device=self.config.device,
                requires_grad=True,
            )
        )
        # Learn two query tensors.
        # row_query: (n_output_rows, n_output_embed // 2)
        self.row_query = nn.Parameter(
            torch.zeros(
                (self.config.n_output_rows, self.config.n_output_embed // 2),
                dtype=self.config.dtype,
                device=self.config.device,
                requires_grad=True,
            )
        )
        # col_query: (n_output_columns, n_output_embed // 2)
        self.col_query = nn.Parameter(
            torch.zeros(
                (self.config.n_output_columns, self.config.n_output_embed // 2),
                dtype=self.config.dtype,
                device=self.config.device,
                requires_grad=True,
            )
        )
        # Learned linear projection to map the concatenated query to the proper query space.
        self.query_projection = nn.Parameter(
            torch.zeros(
                (self.config.n_output_embed, self.config.n_output_embed),
                dtype=self.config.dtype,
                device=self.config.device,
                requires_grad=True,
            )
        )
        # MLP for post-processing the aggregated grid.
        self.mlp_config = MLPConfig(
            n_in=self.config.n_output_embed,
            n_out=self.config.n_output_embed,
            linear_size_multiplier=self.config.linear_size_multiplier,
            activation=self.config.activation,
            dtype=self.config.dtype,
            device=self.config.device,
            init_std=self.config.init_std,
            dropout=self.config.mlp_dropout,
        )
        self.mlp = MLP(self.mlp_config)

    def init_weights(self) -> None:
        """
        Initialize weights with a normal distribution using init_std.
        """
        torch.nn.init.normal_(
            self.transformer_to_kv, mean=0.0, std=self.config.init_std
        )
        torch.nn.init.normal_(
            self.transformer_to_kv_bias, mean=0.0, std=self.config.init_std
        )
        torch.nn.init.normal_(self.row_query, mean=0.0, std=self.config.init_std)
        torch.nn.init.normal_(self.col_query, mean=0.0, std=self.config.init_std)
        torch.nn.init.normal_(self.query_projection, mean=0.0, std=self.config.init_std)
        self.mlp.init_weights()

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_len, n_embed).
            mask (Optional[torch.Tensor]): Optional boolean mask of shape (batch_size, sequence_len)
                                           where False indicates positions to be masked.

        Returns:
            torch.Tensor: Output tensor of shape
                          (batch_size, n_output_rows, n_output_columns, n_output_embed).
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        R = self.config.n_output_rows
        C = self.config.n_output_columns
        d_total = self.config.n_output_embed
        H = self.config.n_heads
        d = d_total // H

        # Compute keys and values from the input.
        # x: (B, seq_len, n_embed)
        # transformer_to_kv: (n_embed, 2*n_output_embed)
        # kv: (B, seq_len, 2*n_output_embed)
        kv = (
            torch.einsum("ij,bnj->bnj", self.transformer_to_kv, x)
            + self.transformer_to_kv_bias
        )
        k, v = torch.split(kv, d_total, dim=2)
        # Reshape to (B, seq_len, H, d)
        k = k.view(batch_size, seq_len, H, d)
        v = v.view(batch_size, seq_len, H, d)

        # Build the query grid.
        # row_query: (R, d_total//2) and col_query: (C, d_total//2)
        # Create a grid of shape (R, C, d_total) by concatenating row and column queries.
        row_query_exp = self.row_query.unsqueeze(1)  # (R, 1, d_total//2)
        col_query_exp = self.col_query.unsqueeze(0)  # (1, C, d_total//2)
        query_grid = torch.cat(
            [row_query_exp.expand(-1, C, -1), col_query_exp.expand(R, -1, -1)],
            dim=2,
        )  # (R, C, d_total)
        # Flatten the grid to shape (R*C, d_total)
        query_grid = query_grid.view(R * C, d_total)
        # Project the queries.
        query_grid = torch.matmul(query_grid, self.query_projection)  # (R*C, d_total)
        # Reshape to (R*C, H, d)
        query_grid = query_grid.view(R * C, H, d)

        # Compute attention scores.
        # k: (B, seq_len, H, d), query_grid: (R*C, H, d)
        # scores: (B, seq_len, H, R*C)
        scores = torch.einsum("bshd,qhd->bshq", k, query_grid)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(2).unsqueeze(3), float("-inf"))
        # Softmax over the sequence dimension.
        att = torch.softmax(scores, dim=1)  # (B, seq_len, H, R*C)

        # Aggregate values with attention.
        # For each grid cell, sum over the sequence:
        # aggregated: (B, R*C, H, d)
        aggregated = torch.einsum("bshd,bshq->bqhd", v, att)
        # Reshape to grid: (B, R, C, d_total)
        aggregated = aggregated.view(batch_size, R, C, d_total)

        # Apply MLP (with a residual connection) to each grid cell.
        aggregated_flat = aggregated.view(batch_size, R * C, d_total)
        out_flat = aggregated_flat + self.mlp(aggregated_flat)
        out = out_flat.view(batch_size, R, C, d_total)
        return out  # type: ignore
