import dataclasses
import torch

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
        # Learned positional encodings for positions [0, ..., n_max_permutation_size-1].
        self.pos_embedding = torch.nn.Parameter(
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
        torch.nn.init.normal_(self.pos_embedding, std=self.config.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (batch_size, n_max_permutation_size) - indices of the permutation.
        
        Returns:
            torch.Tensor: (batch_size, n_max_permutation_size, n_embed) - candidate embeddings with added
                          learned positional encodings.
        """
        # Lookup candidate embeddings based on the permutation indices.
        embeddings = self.c_perm[x]  # shape: (B, n_max_permutation_size, n_embed)
        # Add the learned positional encoding (broadcast along batch dimension).
        pos_enc = self.pos_embedding.unsqueeze(0)  # shape: (1, n_max_permutation_size, n_embed)
        return embeddings + pos_enc

    def get_logprobs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (batch_size, n_max_permutation_size, n_embed) - reconstructed embeddings.
        
        Returns:
            torch.Tensor: (batch_size, n_max_permutation_size, n_max_permutation_size) - log probabilities.
        """
        logits = torch.matmul(x, self.c_perm.t())
        return torch.nn.functional.log_softmax(logits, dim=-1)

    def nll_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (batch_size, n_max_permutation_size, n_embed) - reconstructed embeddings.
            y (torch.Tensor): (batch_size, n_max_permutation_size) - target permutation.
        
        Returns:
            torch.Tensor: (batch_size,) - negative log likelihood loss.
        """
        log_probs = self.get_logprobs(x)
        target_log_probs = torch.gather(log_probs, dim=2, index=y.unsqueeze(2)).squeeze(2)
        loss = -target_log_probs.sum(dim=1)
        return loss
