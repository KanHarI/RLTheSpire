import dataclasses

from rl_the_spire.conf.permutations_group.permutation_group_dataset_config import (
    PermutationGroupDatasetConfig,
)
from rl_the_spire.conf.permutations_group.permutation_group_encoder_config import (
    PermutationGroupEncoderConfig,
)


@dataclasses.dataclass
class PermutationGroupExperimentConfig:
    dataset: PermutationGroupDatasetConfig
    encoder: PermutationGroupEncoderConfig
    iterations: int
    eval_interval: int
    experiment_name: str
