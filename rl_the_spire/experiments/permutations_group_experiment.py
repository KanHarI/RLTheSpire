import logging
from typing import Any

import dacite
import hydra
from torch.utils.data import DataLoader

from rl_the_spire.conf.permutations_group.PermutationsGroupExperimentConfig import (
    PermutationsGroupExperimentConfig,
)
from rl_the_spire.datasets.composed_permutations_dataset import (
    ComposedPermutationDataset,
    ComposedPermutationDatasetConfig,
)
from rl_the_spire.datasets.permutation_and_inverse_dataset import (
    PermutationAndInverseDataset,
    PermutationInverseDatasetConfig,
)

# Configure logger with timestamp and module name
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)  # <-- Logger instance


@hydra.main(
    config_path="../conf/permutations_group", config_name="default", version_base=None
)
def main(hydra_cfg: dict[Any, Any]) -> int:
    config: PermutationsGroupExperimentConfig = dacite.from_dict(
        data_class=PermutationsGroupExperimentConfig,
        data=hydra_cfg,
    )

    # Log and create inversions dataset configuration
    logger.info("Creating inversion dataset config...")
    inversions_dataset_config = PermutationInverseDatasetConfig(
        n_max_permutation_size=config.n_max_permutation_size,
        gamma=config.gamma,
    )

    # Log and create inversions dataset
    logger.info("Creating inversions dataset...")
    inversions_dataset = PermutationAndInverseDataset(inversions_dataset_config)

    # Log and create composition dataset configuration
    logger.info("Creating composition dataset config...")
    composition_dataset_config = ComposedPermutationDatasetConfig(
        n_max_permutation_size=config.n_max_permutation_size,
        gamma=config.gamma,
    )

    # Log and create composition dataset
    logger.info("Creating composition dataset...")
    composition_dataset = ComposedPermutationDataset(composition_dataset_config)

    # Log and create inversions dataloader
    logger.info("Creating inversions dataloader...")
    inversions_dataloader = DataLoader(
        inversions_dataset,
        batch_size=config.batch_size,
        num_workers=config.dataloader_num_workers,
        prefetch_factor=config.dataloader_prefetch_factor,
        pin_memory=True,
    )

    # Log and create composition dataloader
    logger.info("Creating composition dataloader...")
    composition_dataloader = DataLoader(
        composition_dataset,
        batch_size=config.batch_size,
        num_workers=config.dataloader_num_workers,
        prefetch_factor=config.dataloader_prefetch_factor,
    )

    return 0


if __name__ == "__main__":
    main()
