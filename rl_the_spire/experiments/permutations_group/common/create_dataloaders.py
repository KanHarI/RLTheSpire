import logging
import platform
from typing import Iterator, Tuple

import torch
from torch.utils.data import DataLoader

from rl_the_spire.conf.permutations_group.permutation_group_dataset_config import (
    PermutationGroupDatasetConfig,
)
from rl_the_spire.datasets.composed_permutations_dataset import (
    ComposedPermutationDataset,
    ComposedPermutationDatasetConfig,
)
from rl_the_spire.datasets.permutation_and_inverse_dataset import (
    PermutationAndInverseDataset,
    PermutationInverseDatasetConfig,
)

logger = logging.getLogger(__name__)


def create_dataloaders(
    config: PermutationGroupDatasetConfig,
) -> Tuple[
    Iterator[Tuple[torch.Tensor, torch.Tensor]],
    Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    """
    Create and return dataloaders (or iterators) for inversions and composition tasks.
    """
    logger.info("Creating dataloaders...")
    inversions_dataset_config = PermutationInverseDatasetConfig(
        n_max_permutation_size=config.n_max_permutation_size,
        gamma=config.gamma,
    )
    composition_dataset_config = ComposedPermutationDatasetConfig(
        n_max_permutation_size=config.n_max_permutation_size,
        gamma=config.gamma,
    )

    # 2. Build datasets
    inversions_dataset = PermutationAndInverseDataset(inversions_dataset_config)
    composition_dataset = ComposedPermutationDataset(composition_dataset_config)

    # 3. Dataloader settings
    num_workers = 0
    prefetch_factor = None
    if platform.system() == "Linux":
        num_workers = config.num_workers
        prefetch_factor = config.prefetch_factor

    # 4. Create actual dataloaders
    inversions_dataloader = DataLoader(
        inversions_dataset,
        batch_size=config.batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )
    composition_dataloader = DataLoader(
        composition_dataset,
        batch_size=config.batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )

    logger.info("Created dataloaders for inversions and composition datasets.")
    return iter(inversions_dataloader), iter(composition_dataloader)
