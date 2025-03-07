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


@hydra.main(
    config_path="../conf/permutations_group", config_name="default", version_base=None
)
def main(hydra_cfg: dict[Any, Any]) -> int:
    config: PermutationsGroupExperimentConfig = dacite.from_dict(
        data_class=PermutationsGroupExperimentConfig,
        data=hydra_cfg,
    )

    inversions_dataset_config = PermutationInverseDatasetConfig(
        n_max_permutation_size=config.n_max_permutation_size,
        gamma=config.gamma,
    )

    inversions_dataset = PermutationAndInverseDataset(inversions_dataset_config)

    composition_dataset_config = ComposedPermutationDatasetConfig(
        n_max_permutation_size=config.n_max_permutation_size,
        gamma=config.gamma,
    )

    composition_dataset = ComposedPermutationDataset(composition_dataset_config)

    inversions_dataloader = DataLoader(
        inversions_dataset,
        batch_size=config.batch_size,
        num_workers=config.dataloader_num_workers,
        prefetch_factor=config.dataloader_prefetch_factor,
        pin_memory=True,
    )

    composition_dataloader = DataLoader(
        composition_dataset,
        batch_size=config.batch_size,
        num_workers=config.dataloader_num_workers,
        prefetch_factor=config.dataloader_prefetch_factor,
    )
    
    return 0


if __name__ == "__main__":
    main()
