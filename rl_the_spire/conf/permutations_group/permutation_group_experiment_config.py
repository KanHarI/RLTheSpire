import dataclasses

from rl_the_spire.conf.permutations_group.permutation_group_composer_network_config import (
    PermutationGroupComposerNetworkConfig,
)
from rl_the_spire.conf.permutations_group.permutation_group_conv_transformer_config import (
    PermutationGroupConvTransformerConfig,
)
from rl_the_spire.conf.permutations_group.permutation_group_dataset_config import (
    PermutationGroupDatasetConfig,
)
from rl_the_spire.conf.permutations_group.permutation_group_encoder_config import (
    PermutationGroupEncoderConfig,
)
from rl_the_spire.conf.permutations_group.permutation_group_inverter_network_config import (
    PermutationGroupInverterNetworkConfig,
)
from rl_the_spire.conf.permutations_group.permutation_group_optimizer_config import (
    PermutationGroupOptimizerConfig,
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
    inverter_network: PermutationGroupInverterNetworkConfig
    composer_network: PermutationGroupComposerNetworkConfig
    optimizer: PermutationGroupOptimizerConfig
    iterations: int
    eval_interval: int
    experiment_name: str
    wandb_enabled: bool
    log_interval: int
    reconstruction_loss_weight: float
    neural_inv_perm_loss_weight: float
    neural_comp_perm_loss_weight: float
    latent_inv_perm_loss_weight: float
    latent_comp_perm_loss_weight: float
    latent_sampled_perm_loss_weight: float
    # Latent loss warmup parameters
    latent_warmup_steps: int
    latent_warmup_start_weight: float
    latent_warmup_delay_steps: int
    use_ema_target: bool
    ema_tau: float
    # EMA tau scheduling parameters
    ema_tau_start: float
    ema_tau_final: float
    ema_tau_warmup_steps: int
    init_ema_target_as_zeros: bool = False
    num_live_to_target_adapter_layers: int = 1
