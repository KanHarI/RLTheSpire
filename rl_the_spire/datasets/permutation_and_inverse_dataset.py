import dataclasses
from typing import Iterator, Tuple

import torch
from torch.utils.data import IterableDataset


@dataclasses.dataclass
class PermutationInverseDatasetConfig:
    """
    Configuration for the PermutationAndInverseDataset.

    Attributes:
        n_max_permutation_size (int): The fixed output length for each permutation tensor.
        gamma (float): The exponent used to bias the valid permutation length.
                       A value > 1 biases towards smaller lengths.
    """

    n_max_permutation_size: int
    gamma: float = 1.0


class PermutationAndInverseDataset(IterableDataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    An infinite iterable dataset that yields pairs of constant-length permutation tensors.

    Each sample is a tuple (p, p_inv), where:
      - p is a random permutation of a random length L (1 <= L <= n_max_permutation_size),
        with valid entries being 1-indexed and padded with zeros to length n_max_permutation_size.
      - p_inv is the inverse of p computed on the valid region such that for i in [0, L-1],
        p_inv[p[i]-1] = i+1, and padded with zeros for positions i >= L.
    """

    def __init__(self, config: PermutationInverseDatasetConfig) -> None:
        """
        Args:
            config (PermutationInverseDatasetConfig): Configuration specifying the maximum permutation size and gamma.
        """
        self.config = config

    def set_gamma(self, gamma: float) -> None:
        """
        Set the gamma parameter for the dataset.

        Args:
            gamma (float): The new gamma value.
        """
        self.config.gamma = gamma

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Yields:
            Tuple[torch.Tensor, torch.Tensor]: A pair (p, p_inv), where each tensor is of shape (n_max_permutation_size,).
            For a randomly chosen valid length L (1 <= L <= n_max_permutation_size), the first L elements of p
            are a random permutation of [1, L] (1-indexed) and p_inv is its inversion computed on the valid region.
        """
        while True:
            # Sample a uniform number u in [0,1)
            u = torch.rand(1).item()
            # Compute a valid permutation length L biased by gamma.
            L = int(u**self.config.gamma * (self.config.n_max_permutation_size - 1)) + 1

            # Generate a 1-indexed random permutation of [1, L]
            p_valid = torch.randperm(L, dtype=torch.long) + 1

            # Compute the inverse permutation.
            # p_inv_valid[p_valid[i]-1] should be set to i+1 (to preserve 1-indexing)
            p_inv_valid = torch.empty(L, dtype=torch.long)
            p_inv_valid[p_valid - 1] = torch.arange(
                1, L + 1, dtype=torch.long, device=p_valid.device
            )

            # Create padded tensors for p and p_inv.
            p = torch.zeros(self.config.n_max_permutation_size, dtype=torch.long)
            p_inv = torch.zeros(self.config.n_max_permutation_size, dtype=torch.long)
            p[:L] = p_valid
            p_inv[:L] = p_inv_valid

            yield p, p_inv
