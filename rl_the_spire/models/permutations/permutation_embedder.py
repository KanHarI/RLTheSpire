import dataclasses

import torch


@dataclasses.dataclass
class PermutationEmbedderConfig:
    n_max_permutation_size: int
    n_embed: int
    dtype: torch.dtype
    device: str
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, n_max_permutation_size) - indices of the permutation
        returns: (batch_size, n_max_permutation_size, n_embed)
        """
        return self.c_perm[x]

    def get_logprobs(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, n_max_permutation_size, n_embed) - reconstructed embeddings
        returns: (batch_size, n_max_permutation_size, n_max_permutation_size) - log probabilities of the permutation
        """
        # Compute logits by taking the dot product between each reconstructed embedding and all candidate embeddings
        # Resulting shape: (batch_size, n_max_permutation_size, n_max_permutation_size)
        logits = torch.matmul(x, self.c_perm.t())

        # Convert logits to log probabilities using softmax over the candidate dimension (last dimension)
        return torch.nn.functional.log_softmax(logits, dim=-1)

    def nll_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, n_max_permutation_size, n_embed) - reconstructed embeddings
        y: (batch_size, n_max_permutation_size) - target permutation
        returns: (batch_size,) - negative log likelihood loss
        """
        # Get the log probabilities from the reconstructed embeddings
        # log_probs shape: (batch_size, n_max_permutation_size, n_max_permutation_size)
        log_probs = self.get_logprobs(x)

        # Gather the log probability corresponding to the target indices for each position.
        # y.unsqueeze(2) has shape: (batch_size, n_max_permutation_size, 1)
        # target_log_probs shape: (batch_size, n_max_permutation_size, 1) -> squeeze to (batch_size, n_max_permutation_size)
        target_log_probs = torch.gather(log_probs, dim=2, index=y.unsqueeze(2)).squeeze(
            2
        )

        # Compute the negative log likelihood loss per batch by summing over positions.
        loss = -target_log_probs.sum(dim=1)
        return loss
