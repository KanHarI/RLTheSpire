import platform
from collections.abc import Iterator
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from rl_the_spire.conf.direct_probability_distribution_encoding.direct_probability_distribution_encoding_experiment_config import (
    DirectProbabilityDistributionEncodingExperimentConfig,
)
from rl_the_spire.datasets.direct_probability_distribution_dataset import (
    DirectProbabilityDistributionDataset,
    DirectProbabilityDistributionDatasetConfig,
)


def create_dataloader(
    config: DirectProbabilityDistributionEncodingExperimentConfig,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    dataset_config = DirectProbabilityDistributionDatasetConfig(
        n_symbols=config.dataloader.n_symbols
    )
    dataset = DirectProbabilityDistributionDataset(dataset_config)

    num_workers = 0
    prefetch_factor = None
    if platform.system() == "Linux":
        num_workers = config.dataloader.num_workers
        prefetch_factor = config.dataloader.prefetch_factor

    dataloader = DataLoader(
        dataset,
        batch_size=config.n_batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )
    return iter(dataloader)
