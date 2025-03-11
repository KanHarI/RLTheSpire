import dataclasses

import torch

from rl_the_spire.models.position_encodings.positional_sequence_encoder import (
    PositionalSequenceEncoder,
)


@dataclasses.dataclass
class PermutationEmbedderConfig:
    n_max_permutation_size: int
    n_embed: int
    dtype: torch.dtype
    device: torch.device
    init_std: float


class PermutationEmbedder(torch.nn.Module):
    def __init__(self, config: PermutationEmbedderConfig) -> None:
        super().__init__()
        self.config = config
        self.c_perm = torch.nn.Parameter(
            torch.zeros(
                config.n_max_permutation_size,
                config.n_embed,
                dtype=config.dtype,
                device=config.device,
                requires_grad=True,
            )
        )

    def init_weights(self) -> None:
        torch.nn.init.normal_(self.c_perm, std=self.config.init_std)

    def forward(
        self, pos_encoder: PositionalSequenceEncoder, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pos_encoder (PositionalSequenceEncoder): Positional sequence encoder to use instead of internal pos_embedding
            x (torch.Tensor): (batch_size, n_max_permutation_size) - indices of the permutation.

        Returns:
            torch.Tensor: (batch_size, n_max_permutation_size, n_embed) - candidate embeddings with added
                          positional encodings.
        """
        # Lookup candidate embeddings based on the permutation indices.
        embeddings = self.c_perm[x]  # shape: (B, n_max_permutation_size, n_embed)

        # Apply the provided positional encoder instead of using our internal pos_embedding
        return pos_encoder(embeddings)  # type: ignore

    def get_logprobs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Can be one of:
                - (batch_size, n_max_permutation_size, n_embed) - reconstructed embeddings.
                - (S, batch_size, n_max_permutation_size, n_embed) - reconstructed embeddings with sequence dimension.
                - (n_max_permutation_size, n_embed) - single example without batch dimension.

        Returns:
            torch.Tensor: Log probabilities with shape matching input:
                - (batch_size, n_max_permutation_size, n_max_permutation_size)
                - (S, batch_size, n_max_permutation_size, n_max_permutation_size)
                - (n_max_permutation_size, n_max_permutation_size)
        """
        original_shape = x.shape
        
        # Handle single example without batch dimension
        if len(original_shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Handle sequence dimension
        has_seq_dim = len(x.shape) == 4
        if has_seq_dim:
            S, B, P, E = x.shape
            x = x.reshape(S * B, P, E)
        
        # Original computation
        logits = torch.matmul(x, self.c_perm.t())
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Reshape back to original dimensions
        if has_seq_dim:
            log_probs = log_probs.reshape(S, B, P, -1)
        elif len(original_shape) == 2:
            log_probs = log_probs.squeeze(0)  # Remove batch dimension
            
        return log_probs

    def nll_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Can be one of:
                - (batch_size, n_max_permutation_size, n_embed) - reconstructed embeddings.
                - (S, batch_size, n_max_permutation_size, n_embed) - reconstructed embeddings with sequence dimension.
                - (n_max_permutation_size, n_embed) - single example without batch dimension.
            y (torch.Tensor): Target permutation with shape matching x:
                - (batch_size, n_max_permutation_size)
                - (S, batch_size, n_max_permutation_size)
                - (n_max_permutation_size)

        Returns:
            torch.Tensor: Negative log likelihood loss with shape:
                - (batch_size,)
                - (S, batch_size)
                - scalar
        """
        original_x_shape = x.shape
        original_y_shape = y.shape
        
        # Handle single example without batch dimension
        if len(original_x_shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension
            y = y.unsqueeze(0)  # Add batch dimension
            
        # Handle sequence dimension
        has_seq_dim = len(x.shape) == 4
        if has_seq_dim:
            S, B, P, E = x.shape
            x = x.reshape(S * B, P, E)
            y = y.reshape(S * B, P)
        
        # Get log probabilities
        log_probs = self.get_logprobs(x)
        
        # Gather target log probabilities
        target_log_probs = torch.gather(log_probs, dim=2, index=y.unsqueeze(2)).squeeze(2)
        
        # Sum across the permutation dimension to get NLL per example
        nll = -torch.sum(target_log_probs, dim=1)
        
        # Reshape back to match original dimensions
        if has_seq_dim:
            nll = nll.reshape(S, B)
        elif len(original_x_shape) == 2:
            nll = nll.squeeze(0)  # Remove batch dimension
            
        return nll
