"""Positional sequence encoder module."""

import dataclasses

import torch


@dataclasses.dataclass
class PositionalSequenceEncoderConfig:
    """
    Configuration for the PositionalSequenceEncoder.

    Attributes:
        n_embed (int): Dimensionality of the embeddings.
        max_seq_len (int): Maximum sequence length.
        device (torch.device): Device to store the parameters.
        dtype (torch.dtype): Data type for parameters.
        init_std (float): Standard deviation for weight initialization.
    """

    n_embed: int
    max_seq_len: int
    device: torch.device
    dtype: torch.dtype
    init_std: float = 0.02


class PositionalSequenceEncoder(torch.nn.Module):
    def __init__(self, config: PositionalSequenceEncoderConfig):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.n_embed = config.n_embed

        # Create learnable sequence position embeddings
        self.pos_embeddings = torch.nn.Parameter(
            torch.zeros(
                config.max_seq_len,
                config.n_embed,
                device=config.device,
                dtype=config.dtype,
            ),
            requires_grad=True,
        )

    def init_weights(self) -> None:
        # Initialize with small random values
        torch.nn.init.normal_(self.pos_embeddings, std=self.config.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encodings to the sequence of embeddings.

        Args:
            x: [batch_size, seq_len, n_embed] tensor

        Returns:
            x with positional encodings added: [batch_size, seq_len, n_embed]
        """
        # Get sequence length from input
        seq_len = x.shape[1]

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds maximum sequence "
                f"length ({self.max_seq_len})"
            )

        # Create position encodings [1, seq_len, n_embed]
        pos = self.pos_embeddings[:seq_len].unsqueeze(0)

        # Add positional encodings to the input
        return x + pos
