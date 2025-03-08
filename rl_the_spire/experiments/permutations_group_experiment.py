import logging
import platform
from typing import Any

import dacite
import hydra
import torch
import wandb
from torch.utils.data import DataLoader

from rl_the_spire.conf.permutations_group.permutation_group_experiment_config import (
    PermutationGroupExperimentConfig,
)
from rl_the_spire.conf.utils.activations import get_activation
from rl_the_spire.conf.utils.devices import get_device
from rl_the_spire.conf.utils.dtypes import get_dtype
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
from rl_the_spire.models.vaes.kl_loss import kl_loss

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
    config: PermutationGroupExperimentConfig = dacite.from_dict(
        data_class=PermutationGroupExperimentConfig,
        data=hydra_cfg,
    )

    # Log and create inversions dataset configuration
    logger.info("Creating inversion dataset config...")
    inversions_dataset_config = PermutationInverseDatasetConfig(
        n_max_permutation_size=config.dataset.n_max_permutation_size,
        gamma=config.dataset.gamma,
    )

    # Log and create inversions dataset
    logger.info("Creating inversions dataset...")
    inversions_dataset = PermutationAndInverseDataset(inversions_dataset_config)

    # Log and create composition dataset configuration
    logger.info("Creating composition dataset config...")
    composition_dataset_config = ComposedPermutationDatasetConfig(
        n_max_permutation_size=config.dataset.n_max_permutation_size,
        gamma=config.dataset.gamma,
    )

    # Log and create composition dataset
    logger.info("Creating composition dataset...")
    composition_dataset = ComposedPermutationDataset(composition_dataset_config)

    # Log and create inversions dataloader
    logger.info("Creating inversions dataloader...")

    num_workers = 0
    prefetch_factor = None

    if platform.system() == "Linux":
        num_workers = config.dataset.num_workers
        prefetch_factor = config.dataset.prefetch_factor

    inversions_dataloader = iter(
        DataLoader(
            inversions_dataset,
            batch_size=config.dataset.batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
        )
    )

    # Log and create composition dataloader
    logger.info("Creating composition dataloader...")
    composition_dataloader = iter(
        DataLoader(
            composition_dataset,
            batch_size=config.dataset.batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
        )
    )

    # Create permutation encoder
    logger.info("Creating permutation encoder...")
    permutation_encoder_config = PermutationEncoderConfig(
        n_max_permutation_size=config.dataset.n_max_permutation_size,
        n_embed=config.encoder.n_embed,
        n_heads=config.encoder.n_heads,
        n_layers=config.encoder.n_layers,
        attn_dropout=config.encoder.attn_dropout,
        resid_dropout=config.encoder.resid_dropout,
        dtype=get_dtype(config.encoder.dtype),
        device=get_device(config.encoder.device),
        init_std=config.encoder.init_std,
        mlp_dropout=config.encoder.mlp_dropout,
        ln_eps=config.encoder.ln_eps,
        n_output_heads=config.encoder.n_output_heads,
        n_output_embed=config.encoder.n_output_embed,
        n_output_rows=config.encoder.n_output_rows,
        n_output_columns=config.encoder.n_output_columns,
        activation=get_activation(config.encoder.activation),
        linear_size_multiplier=config.encoder.linear_size_multiplier,
        conv_transformer_n_heads=config.conv_transformer.n_heads,
    )
    permutation_encoder = PermutationEncoder(permutation_encoder_config)

    # Initialize wandb
    logger.info("Initializing WanDB...")
    # wandb.init(project="rl_the_spire.permutations_group", name=config.experiment_name)
    # wandb.config.update(config)  # type: ignore

    # Run experiment
    logger.info("Running experiment...")
    TOTAL_ENCODED_PERMUTATIONS = 5
    for i in range(config.iterations):
        perm, inv = next(inversions_dataloader)
        p, q, r = next(composition_dataloader)

        # Run autoencoder
        encoded_perm_mus, encoded_perm_logvars = permutation_encoder(perm)
        encoded_inv_mus, encoded_inv_logvars = permutation_encoder(inv)
        encoded_p_mus, encoded_p_logvars = permutation_encoder(p)
        encoded_q_mus, encoded_q_logvars = permutation_encoder(q)
        encoded_r_mus, encoded_r_logvars = permutation_encoder(r)

        # Calculate all KL losses
        kl_losses = torch.tensor(0.0)
        for (mus, logvars) in zip([encoded_perm_mus, encoded_inv_mus, encoded_p_mus, encoded_q_mus, encoded_r_mus], [encoded_perm_logvars, encoded_inv_logvars, encoded_p_logvars, encoded_q_logvars, encoded_r_logvars]):
            kl_losses += kl_loss(mus, logvars).sum() / TOTAL_ENCODED_PERMUTATIONS

        print(f"KL Losses: {kl_losses}")
         
        raise NotImplementedError

    return 0


if __name__ == "__main__":
    main()
