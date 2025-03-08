import dataclasses

from rl_the_spire.conf.permutations_group.permutation_group_conv_transformer_config import (
    PermutationGroupConvTransformerConfig,
)
from rl_the_spire.conf.permutations_group.permutation_group_dataset_config import (
    PermutationGroupDatasetConfig,
)
from rl_the_spire.conf.permutations_group.permutation_group_encoder_config import (
    PermutationGroupEncoderConfig,
)
from rl_the_spire.conf.permutations_group.permutation_group_vae_config import (
    PermutationGroupVAEConfig,
)


@dataclasses.dataclass
class PermutationGroupExperimentConfig:
    dataset: PermutationGroupDatasetConfig
    encoder: PermutationGroupEncoderConfig
    vae: PermutationGroupVAEConfig
    conv_transformer: PermutationGroupConvTransformerConfig
    iterations: int
    eval_interval: int
    experiment_name: str
    wandb_enabled: bool
    log_interval: int
    reconstruction_loss_weight: float
