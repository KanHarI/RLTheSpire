import dataclasses
from typing import Iterator, Tuple

import torch
from torch.utils.data import IterableDataset


@dataclasses.dataclass
class DirectProbabilityDistributionDatasetConfig:
    """
    Configuration for the DirectProbabilityDistributionDataset.
    """

    n_symbols: int


class DirectProbabilityDistributionDataset(
    IterableDataset[Tuple[torch.Tensor, torch.Tensor]]
):
    """
    A dataset that generates random probability distributions over subsets of symbols.

    This dataset creates samples where each sample consists of:
    1. A tensor of randomly selected symbol indices
    2. A corresponding tensor representing a random probability distribution over those symbols

    The dataset is infinite and yields a new random sample each time it is accessed.
    """

    def __init__(self, config: DirectProbabilityDistributionDatasetConfig) -> None:
        """
        Initialize the dataset with the provided configuration.

        Args:
            config: Configuration object containing dataset parameters
        """
        self.config = config

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create an iterator that yields random symbol sets and their probability distributions.

        Each yielded item is a tuple of:
        - used_symbols: Tensor of shape (num_symbols_used,) containing randomly selected symbol indices
        - probabilities: Tensor of shape (num_symbols_used,) containing a normalized random probability
                        distribution over the selected symbols

        Returns:
            An iterator yielding (used_symbols, probabilities) tuples
        """
        while True:
            # Random between 2 and n_symbols
            num_symbols_used = torch.randint(2, self.config.n_symbols, (1,)).item()

            # Select num_symbols_used random symbols from [1, n_symbols]. The +1 is because we want to include the zero symbol as padding.
            used_symbols = (
                torch.randperm(self.config.n_symbols, dtype=torch.long)[
                    :num_symbols_used  # type: ignore
                ]
                + 1
            )
            # Pad zeros to the end of the used symbols to make it [batch_size, n_symbols + 1]
            used_symbols = torch.cat(
                [
                    used_symbols,
                    torch.zeros(
                        self.config.n_symbols + 1 - num_symbols_used, dtype=torch.long  # type: ignore
                    ),
                ],
                dim=-1,
            )

            # Create a random probability distribution over the used symbols
            probabilities = torch.rand(num_symbols_used)  # type: ignore
            # Pad zeros to the end of the probabilities to make it [batch_size, n_symbols + 1]
            probabilities = torch.cat(
                [
                    probabilities,
                    torch.zeros(self.config.n_symbols + 1 - num_symbols_used),  # type: ignore
                ],
                dim=-1,
            )
            probabilities = probabilities / probabilities.sum()

            # Yield the probability distribution
            yield used_symbols, probabilities
