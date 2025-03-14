"""Positional grid encoder module."""

import dataclasses

import torch


@dataclasses.dataclass
class PositionalGridEncoderConfig:
    """
    Configuration for the PositionalGridEncoder.

    Attributes:
        n_embed (int): Dimensionality of the embeddings.
        n_rows (int): Number of rows in the grid.
        n_cols (int): Number of columns in the grid.
        device (torch.device): Device to store the parameters.
        dtype (torch.dtype): Data type for parameters.
        init_std (float): Standard deviation for weight initialization.
    """

    n_embed: int
    n_rows: int
    n_cols: int
    device: torch.device
    dtype: torch.dtype
    init_std: float = 0.02


class PositionalGridEncoder(torch.nn.Module):
    def __init__(self, config: PositionalGridEncoderConfig):
        super().__init__()
        self.config = config
        self.n_rows = config.n_rows
        self.n_cols = config.n_cols
        self.n_embed = config.n_embed

        # Create learnable row and column embeddings
        self.row_embeddings = torch.nn.Parameter(
            torch.zeros(
                config.n_rows,
                config.n_embed,
                device=config.device,
                dtype=config.dtype,
            ),
            requires_grad=True,
        )
        self.col_embeddings = torch.nn.Parameter(
            torch.zeros(
                config.n_cols,
                config.n_embed,
                device=config.device,
                dtype=config.dtype,
            ),
            requires_grad=True,
        )

    def init_weights(self) -> None:
        # Initialize with small random values
        torch.nn.init.normal_(self.row_embeddings, std=self.config.init_std)
        torch.nn.init.normal_(self.col_embeddings, std=self.config.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encodings to the grid of embeddings.

        Args:
            x: Tensor with shape [..., n_rows, n_cols, n_embed]
               Supports shapes [batch_size, n_rows, n_cols, n_embed],
               [seq_len, batch_size, n_rows, n_cols, n_embed], or just [n_rows, n_cols, n_embed]

        Returns:
            x with positional encodings added: Same shape as input
        """
        # Extract shape components, with *B capturing any leading dimensions
        *B, R, C, E = x.shape

        # Validate the trailing dimensions match our configuration
        assert R == self.n_rows, f"Expected {self.n_rows} rows, got {R}"
        assert C == self.n_cols, f"Expected {self.n_cols} columns, got {C}"
        assert E == self.n_embed, f"Expected {self.n_embed} embedding dim, got {E}"

        # Create row positional encodings and reshape for broadcasting with any input shape
        # For any leading dimensions, add singleton dimensions
        leading_dims = len(B)
        row_pos = self.row_embeddings.view(
            *([1] * leading_dims), self.n_rows, 1, self.n_embed
        )

        # Create column positional encodings with singleton dimensions for broadcasting
        col_pos = self.col_embeddings.view(
            *([1] * leading_dims), 1, self.n_cols, self.n_embed
        )

        # Add positional encodings to the input
        # Broadcasting handles adding row_pos to each column and col_pos to each row
        return x + row_pos + col_pos
