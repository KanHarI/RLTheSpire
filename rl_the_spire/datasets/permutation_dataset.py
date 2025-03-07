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
    """

    n_max_permutation_size: int


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

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Yields:
            torch.Tensor: A tensor of shape (n_max_permutation_size,) where the first `perm_length`
            elements form a random permutation of numbers from 1 to perm_length (inclusive) and the
            remaining positions are padded with zeros.
        """
        while True:
            # Choose a random permutation length between 1 and n_max_permutation_size (inclusive).
            perm_length = torch.randint(
                1, self.config.n_max_permutation_size + 1, (1,)
            ).item()
            # Generate a random permutation of numbers 0 to perm_length-1 and shift to 1-indexed.
            permutation = torch.randperm(perm_length) + 1  # type: ignore
            # Create a tensor of zeros with the fixed output length.
            padded_permutation = torch.zeros(
                self.config.n_max_permutation_size, dtype=torch.long
            )
            # Fill the beginning of the tensor with the generated permutation.
            padded_permutation[:perm_length] = permutation  # type: ignore
            yield padded_permutation
