import dataclasses
from typing import Iterator, Tuple

import torch
from torch.utils.data import IterableDataset


@dataclasses.dataclass
class ComposedPermutationDatasetConfig:
    """
    Configuration for the ComposedPermutationDataset.

    Attributes:
        n_max_permutation_size (int): The fixed output length for each permutation tensor.
        gamma (float): The exponent used to bias the valid permutation length. A value > 1 biases towards smaller lengths.
    """

    n_max_permutation_size: int
    gamma: float = 1.0


class ComposedPermutationDataset(
    IterableDataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """
    An infinite iterable dataset that yields triplets of constant-length permutation tensors.

    Each sample is a triplet (p, q, r) where:
      - p and q are random permutations of a random length L (1 <= L <= n_max_permutation_size),
        with valid entries being 1-indexed and padded with zeros to length n_max_permutation_size.
      - r is the composition of p and q, computed as:
            r[i] = p[q[i]-1] for all i in [0, L-1],
        and padded with zeros for positions i >= L.
    """

    def __init__(self, config: ComposedPermutationDatasetConfig) -> None:
        """
        Args:
            config (ComposedPermutationDatasetConfig): Configuration object specifying the maximum permutation size.
        """
        self.config = config

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Yields:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            A triplet (p, q, r), where each tensor is of shape (n_max_permutation_size,). For a randomly chosen
            valid length L (1 <= L <= n_max_permutation_size), the first L elements of p and q are a random permutation
            of [1, L] and r is computed as the composition of p and q on the valid region, with the remaining positions
            padded with zeros.
        """
        while True:
            u = torch.rand(1).item()
            # Choose a random valid permutation length L between 1 and n_max_permutation_size (inclusive)
            L = int(u**self.config.gamma * (self.config.n_max_permutation_size - 1)) + 1

            # Generate p and q as random permutations of [1, L] (1-indexed)
            p_valid = torch.randperm(L, dtype=torch.long) + 1
            q_valid = torch.randperm(L, dtype=torch.long) + 1

            # Create padded tensors for p and q
            p = torch.zeros(self.config.n_max_permutation_size, dtype=torch.long)
            q = torch.zeros(self.config.n_max_permutation_size, dtype=torch.long)
            p[:L] = p_valid
            q[:L] = q_valid

            # Compute the composition for the valid region:
            # For i in range(L), r[i] = p[q[i]-1]
            r_valid = p[:L][q_valid - 1]

            # Pad the composed permutation r to have constant size
            r = torch.zeros(self.config.n_max_permutation_size, dtype=torch.long)
            r[:L] = r_valid

            # Yield the triplet (p, q, r)
            yield p, q, r
