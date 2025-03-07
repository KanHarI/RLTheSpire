import dataclasses
from typing import Iterator

import torch
from torch.utils.data import IterableDataset


@dataclasses.dataclass
class PermutationDatasetConfig:
    """
    Configuration for the PermutationDataset.

    Attributes:
        n_max_permutation_size (int): The fixed output length for each permutation tensor.
        gamma (float): The exponent used to bias the valid permutation length. A value > 1 biases towards smaller lengths.
    """

    n_max_permutation_size: int
    gamma: float = 1.0


class PermutationDataset(IterableDataset[torch.Tensor]):
    """
    An iterable dataset that yields constant-length permutation tensors.

    Each sample is a permutation of a random length (between 1 and n_max_permutation_size)
    with valid entries being 1-indexed (i.e. starting from 1). The permutation is padded with
    zeros to have a fixed length of n_max_permutation_size.
    """

    def __init__(self, config: PermutationDatasetConfig) -> None:
        """
        Args:
            config (PermutationDatasetConfig): Configuration object specifying the maximum permutation size.
        """
        self.config = config

    def set_gamma(self, gamma: float) -> None:
        """
        Set the gamma parameter for the dataset.
        """
        self.config.gamma = gamma

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Yields:
            torch.Tensor: A tensor of shape (n_max_permutation_size,) where the first `perm_length`
            elements form a random permutation of numbers from 1 to perm_length (inclusive) and the
            remaining positions are padded with zeros.
        """
        while True:
            u = torch.rand(1).item()
            L = int(u**self.config.gamma * (self.config.n_max_permutation_size - 1)) + 1
            # Generate a random permutation of numbers 0 to perm_length-1 and shift to 1-indexed.
            permutation = torch.randperm(L) + 1
            # Create a tensor of zeros with the fixed output length.
            padded_permutation = torch.zeros(
                self.config.n_max_permutation_size, dtype=torch.long
            )
            # Fill the beginning of the tensor with the generated permutation.
            padded_permutation[:L] = permutation
            yield padded_permutation
