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
        n_output_size (int): The spatial size N of the output grid (output shape is N x N x n_output_embed).
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
    n_output_size: int
    n_heads: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    linear_size_multiplier: int
    dtype: torch.dtype
    device: str
    init_std: float
    mlp_dropout: float


class AttentionToTensor(nn.Module):
    """
    Encodes an unordered sequence of vectors into a grid tensor of shape
    (batch_size, n_output_size, n_output_size, n_output_embed) using attention.

    Instead of a single query parameter, this module learns two query tensors
    (row_query and col_query), each of length n_output_size. For a grid cell at
    position (m, n), the query is formed by concatenating the mth entry of row_query
    with the nth entry of col_query, then projecting it with a learned linear map.

    Keys and values are computed from the input sequence. For each grid cell, attention
    is computed over the sequence and the attended values are aggregated. An MLP (with
    a residual connection) is applied to the aggregated output.
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
        # Learn two query tensors. They will be concatenated to form a query vector.
        # Each has shape (n_output_size, n_output_embed // 2), so concatenation gives n_output_embed.
        self.row_query = nn.Parameter(
            torch.zeros(
                (self.config.n_output_size, self.config.n_output_embed // 2),
                dtype=self.config.dtype,
                device=self.config.device,
                requires_grad=True,
            )
        )
        self.col_query = nn.Parameter(
            torch.zeros(
                (self.config.n_output_size, self.config.n_output_embed // 2),
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
                          (batch_size, n_output_size, n_output_size, n_output_embed).
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        N = self.config.n_output_size
        d_total = self.config.n_output_embed
        H = self.config.n_heads
        d = d_total // H

        # Compute keys and values from the input.
        # x: (B, seq_len, n_embed)
        # transformer_to_kv: (n_embed, 2*n_output_embed)
        # Resulting kv: (B, seq_len, 2*n_output_embed)
        kv = (
            torch.einsum("ij,bnj->bnj", self.transformer_to_kv, x)
            + self.transformer_to_kv_bias
        )
        k, v = torch.split(kv, d_total, dim=2)
        # Reshape to (B, seq_len, H, d)
        k = k.view(batch_size, seq_len, H, d)
        v = v.view(batch_size, seq_len, H, d)

        # Build the query grid.
        # row_query: (N, d_total//2) and col_query: (N, d_total//2)
        # We use broadcasting to create a grid of shape (N, N, d_total) where:
        #   grid[m, n] = concat(row_query[m], col_query[n])
        row_query_exp = self.row_query.unsqueeze(1)  # (N, 1, d_total//2)
        col_query_exp = self.col_query.unsqueeze(0)  # (1, N, d_total//2)
        query_grid = torch.cat(
            [row_query_exp.expand(-1, N, -1), col_query_exp.expand(N, -1, -1)], dim=2
        )  # (N, N, d_total)
        # Flatten the grid to shape (N*N, d_total)
        query_grid = query_grid.view(N * N, d_total)
        # Project the queries.
        query_grid = torch.matmul(query_grid, self.query_projection)  # (N*N, d_total)
        # Reshape to (N*N, H, d)
        query_grid = query_grid.view(N * N, H, d)

        # Compute attention scores.
        # k: (B, seq_len, H, d), query_grid: (N*N, H, d)
        # scores: (B, seq_len, H, N*N)
        scores = torch.einsum("bshd,qhd->bshq", k, query_grid)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(2).unsqueeze(3), float("-inf"))
        # Softmax over the sequence dimension.
        att = torch.softmax(scores, dim=1)  # (B, seq_len, H, N*N)

        # Aggregate values with attention.
        # For each grid cell, sum over the sequence: (B, N*N, H, d)
        aggregated = torch.einsum("bshd,bshq->bqhd", v, att)
        # Reshape to grid: (B, N, N, d_total)
        aggregated = aggregated.view(batch_size, N, N, d_total)

        # Apply MLP (with a residual connection) to each grid cell.
        aggregated_flat = aggregated.view(batch_size, N * N, d_total)
        out_flat = aggregated_flat + self.mlp(aggregated_flat)
        out = out_flat.view(batch_size, N, N, d_total)
        return out  # type: ignore
