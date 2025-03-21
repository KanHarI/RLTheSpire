import dataclasses

from rl_the_spire.conf.permutations_group.permutation_group_dataset_config import (
    PermutationGroupDatasetConfig,
)


@dataclasses.dataclass
class PermutationGroupSequenceVAEConfig:
    dataset: PermutationGroupDatasetConfig
