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
                config.n_rows, config.n_embed, device=config.device, dtype=config.dtype
            )
        )
        self.col_embeddings = torch.nn.Parameter(
            torch.zeros(
                config.n_cols, config.n_embed, device=config.device, dtype=config.dtype
            )
        )

        # Initialize with small random values
        torch.nn.init.normal_(self.row_embeddings, std=config.init_std)
        torch.nn.init.normal_(self.col_embeddings, std=config.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encodings to the grid of embeddings.

        Args:
            x: [batch_size, n_rows, n_cols, n_embed] tensor

        Returns:
            x with positional encodings added: [batch_size, n_rows, n_cols, n_embed]
        """
        # Create row positional encodings [1, n_rows, 1, n_embed]
        row_pos = self.row_embeddings.unsqueeze(0).unsqueeze(2)

        # Create column positional encodings [1, 1, n_cols, n_embed]
        col_pos = self.col_embeddings.unsqueeze(0).unsqueeze(1)

        # Add positional encodings to the input
        # The broadcasting will ensure row_pos is added to each column and col_pos to each row
        return x + row_pos + col_pos

    def init_weights(self) -> None:
        # Weights already initialized in __init__
        pass
