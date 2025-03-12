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
        # Ensure x has at least 3 dimensions (add batch dim if needed)
        if x.dim() == 2:
            x = x.unsqueeze(-3)
            
        # Compute logits and log probabilities
        logits = torch.matmul(x, self.c_perm.t())
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # If input was 2D, output should be 2D as well
        if x.dim() == 3 and x.size(-3) == 1:
            log_probs = log_probs.squeeze(-3)
            
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
        # Track if we need to squeeze the output at the end
        needs_squeeze = (x.dim() == 2)
        
        # Ensure x has at least 3 dimensions (add batch dim if needed)
        if needs_squeeze:
            x = x.unsqueeze(-3)
            y = y.unsqueeze(-2)
        
        # Get log probabilities
        log_probs = self.get_logprobs(x)
        
        # Gather target log probabilities
        target_log_probs = torch.gather(log_probs, dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
        
        # Sum across the permutation dimension to get NLL per example
        nll = -torch.sum(target_log_probs, dim=-1)
        
        # If input was 2D, output should be a scalar
        if needs_squeeze:
            nll = nll.squeeze(-1)
            
        return nll
