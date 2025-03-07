import logging
import platform
from typing import Any

import dacite
import hydra
import wandb
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
from rl_the_spire.models.permutations.permutation_encoder import (
    PermutationEncoder,
    PermutationEncoderConfig,
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
        gamma=config.dataset_gamma,
    )

    # Log and create inversions dataset
    logger.info("Creating inversions dataset...")
    inversions_dataset = PermutationAndInverseDataset(inversions_dataset_config)

    # Log and create composition dataset configuration
    logger.info("Creating composition dataset config...")
    composition_dataset_config = ComposedPermutationDatasetConfig(
        n_max_permutation_size=config.n_max_permutation_size,
        gamma=config.dataset_gamma,
    )

    # Log and create composition dataset
    logger.info("Creating composition dataset...")
    composition_dataset = ComposedPermutationDataset(composition_dataset_config)

    # Log and create inversions dataloader
    logger.info("Creating inversions dataloader...")
    num_workers = 0

    if platform.system() == "Linux":
        num_workers = config.dataloader_num_workers

    inversions_dataloader = DataLoader(
        inversions_dataset,
        batch_size=config.batch_size,
        num_workers=num_workers,
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

    # Create permutation encoder
    logger.info("Creating permutation encoder...")
    permutation_encoder_config = PermutationEncoderConfig()
    permutation_encoder = PermutationEncoder(permutation_encoder_config)

    # Initialize wandb
    logger.info("Initializing WanDB...")
    wandb.init(project="rl_the_spire.permutations_group", name=config.experiment_name)
    wandb.config.update(config)  # type: ignore

    # Run experiment
    logger.info("Running experiment...")
    TOTAL_ENCODED_PERMUTATIONS = 5
    for i in range(config.iterations):
        perm, inv = inversions_dataloader.next()
        p, q, r = composition_dataloader.next()

        # Run autoencoder
        encoded_perm = permutation_encoder(perm)
        encoded_inv = permutation_encoder(inv)
        encoded_p = permutation_encoder(p)
        encoded_q = permutation_encoder(q)
        encoded_r = permutation_encoder(r)

    return 0


if __name__ == "__main__":
    main()
