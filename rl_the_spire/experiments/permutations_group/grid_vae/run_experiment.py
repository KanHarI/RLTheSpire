# isort: skip_file
# Skipping isort as for some unknown reason mac and linux disagree on the order of imports

import logging
from typing import Any

import dacite
import hydra
import torch

import wandb
from rl_the_spire.conf.permutations_group.grid.permutation_group_grid_experiment_config import (
    PermutationGroupGridExperimentConfig,
)
from rl_the_spire.conf.utils.devices import get_device
from rl_the_spire.conf.utils.dtypes import get_dtype
from rl_the_spire.experiments.permutations_group.common.create_dataloaders import (
    create_dataloaders,
)
from rl_the_spire.experiments.permutations_group.grid_vae.create_models import (
    create_models,
    create_target_models,
)
from rl_the_spire.experiments.permutations_group.grid_vae.training_loop_iteration import (
    TrainingLoopInput,
    training_loop_iteration,
)
from rl_the_spire.utils.loss_utils import (
    get_kl_weight,
    get_latent_weight,
    get_vae_gamma,
)
from rl_the_spire.utils.training_utils import ema_update, get_ema_tau, lr_lambda

# Configure logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../../../conf/permutations_group/grid",  # Adjust path if needed
    config_name="default",  # The name of your .yaml file, e.g. "default.yaml"
    version_base=None,
)
def main(hydra_cfg: dict[Any, Any]) -> int:
    # 1. Parse the Hydra config into our dataclasses
    config: PermutationGroupGridExperimentConfig = dacite.from_dict(
        data_class=PermutationGroupGridExperimentConfig,
        data=hydra_cfg,
    )
    
    logger.info("Initializing experiment...")

    inversions_dataloader, composition_dataloader = create_dataloaders(config.dataset)

    # Create all models
    logger.info("Creating models...")

    # Explicitly set device and dtype for all models
    device = get_device(config.encoder.device)
    dtype = get_dtype(config.encoder.dtype)

    learned_networks_tuple = create_models(config, device, dtype)

    (
        permutation_encoder,
        positional_seq_encoder,
        denoiser_dim_expander,
        denoiser_network,
        permutations_decoder,
        inverter_network,
        composer_network,
        live_to_target_dimensionality_reducer,
        live_to_target_adapter,
        positional_grid_encoder,
    ) = learned_networks_tuple

    # Create EMA target models if enabled
    target_networks_tuple = create_target_models(
        config,
        device,
        permutation_encoder,
        positional_seq_encoder,
    )

    params_for_optimizer = [
        param for model in learned_networks_tuple for param in model.parameters()
    ]

    # 5. Create an optimizer
    logger.info("Creating AdamW optimizer...")
    optimizer = torch.optim.AdamW(
        params_for_optimizer,
        lr=config.optimizer.lr,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.optimizer.weight_decay,
    )

    # Create scheduler with linear warmup
    logger.info(
        f"Setting up learning rate scheduler with {config.optimizer.warmup_steps} warmup steps..."
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, config.optimizer.warmup_steps)
    )

    # Log about warmups here, implement them in the training loop
    # Function to calculate KL weight with warmup
    logger.info(
        f"Setting up KL weight warmup over {config.vae.kl_warmup_steps} steps from {config.vae.kl_warmup_start_weight} to {config.vae.kl_loss_weight}..."
    )

    # Add log for latent warmup configuration
    logger.info(
        f"Setting up latent loss warmup with {config.latent_warmup_delay_steps} delay steps followed by {config.latent_warmup_steps} warmup steps..."
    )

    # Function to calculate EMA tau with warmup
    logger.info(
        f"Setting up EMA tau warmup over {config.ema_tau_warmup_steps} steps from {config.ema_tau_start} to {config.ema_tau_final}..."
    )

    # Log for gamma warmup (new)
    logger.info(
        f"Setting up VAE gamma warmup over {config.vae.gamma_warmup_steps} steps from {config.vae.gamma_start} to {config.vae.gamma_final}..."
    )

    if config.wandb_enabled:
        # 6. Initialize Weights & Biases
        logger.info("Initializing WanDB...")
        wandb.init(
            project="rl_the_spire.permutations_group.grid_vae",
            name=config.experiment_name,
        )
        wandb.config.update(config)  # type: ignore

    # 7. Training/Eval Loop
    logger.info("Starting training loop...")

    for step in range(config.iterations):
        # -------------------
        #       EVAL
        # -------------------
        if step % config.eval_interval == 0:
            # We do not need eval mode, let the dropout be as it is
            # for model in learned_networks_tuple:
            #     model.eval()

            with torch.no_grad():
                # Calculate the current gamma value based on warmup schedule
                current_gamma = get_vae_gamma(
                    step,
                    config.vae.gamma_warmup_steps,
                    config.vae.gamma_start,
                    config.vae.gamma_final,
                )

                # Calculate ema_tau once for evaluation
                current_ema_tau = get_ema_tau(
                    step,
                    config.ema_tau_warmup_steps,
                    config.ema_tau_start,
                    config.ema_tau_final,
                )

                losses = training_loop_iteration(
                    TrainingLoopInput(
                        learned_networks_tuple=learned_networks_tuple,
                        target_networks_tuple=target_networks_tuple,
                        dataloaders=(inversions_dataloader, composition_dataloader),
                        vae_gamma=current_gamma,
                        device=device,
                        dtype=dtype,
                    )
                )
                kl_weight = torch.tensor(
                    get_kl_weight(
                        step,
                        config.vae.kl_warmup_steps,
                        config.vae.kl_warmup_start_weight,
                        config.vae.kl_loss_weight,
                    ),
                    device=device,
                    dtype=dtype,
                )
                kl_weighted_loss = losses.kl_losses * kl_weight
                vae_reconstruction_nll_weighted = (
                    losses.vae_reconstruction_nll * config.reconstruction_loss_weight
                )
                group_operations_weight_mult = (
                    1
                    if step > config.group_operations_warmup_steps
                    else step / config.group_operations_warmup_steps
                )
                inv_reconstruction_nll_weighted = (
                    losses.inv_reconstruction_nll
                    * config.neural_inv_perm_loss_weight
                    * group_operations_weight_mult
                )
                comp_reconstruction_nll_weighted = (
                    losses.comp_reconstruction_nll
                    * config.neural_comp_perm_loss_weight
                    * group_operations_weight_mult
                )
                latent_l2_losses_weight = torch.tensor(
                    get_latent_weight(
                        step,
                        config.latent_warmup_delay_steps,
                        config.latent_warmup_steps,
                        config.latent_warmup_start_weight,
                    ),
                    device=device,
                    dtype=dtype,
                )
                live_to_target_l2_weightes = (
                    losses.live_to_target_l2
                    * config.latent_sampled_perm_loss_weight
                    * latent_l2_losses_weight
                )
                inv_latent_l2_weight = (
                    losses.target_inv_l2
                    * config.latent_inv_perm_loss_weight
                    * latent_l2_losses_weight
                    * group_operations_weight_mult
                )
                comp_latent_l2_weight = (
                    losses.target_comp_l2
                    * config.latent_comp_perm_loss_weight
                    * latent_l2_losses_weight
                    * group_operations_weight_mult
                )
                total_loss = (
                    kl_weighted_loss
                    + vae_reconstruction_nll_weighted
                    + inv_reconstruction_nll_weighted
                    + comp_reconstruction_nll_weighted
                    + live_to_target_l2_weightes
                    + inv_latent_l2_weight
                    + comp_latent_l2_weight
                )

                if config.wandb_enabled:
                    wandb.log(
                        {
                            "eval/total_loss": total_loss.item(),
                            "eval/kl_loss": losses.kl_losses.item(),
                            "eval/kl_loss_weighted": kl_weighted_loss.item(),
                            "eval/kl_weight": get_kl_weight(
                                step,
                                config.vae.kl_warmup_steps,
                                config.vae.kl_warmup_start_weight,
                                config.vae.kl_loss_weight,
                            ),
                            "eval/vae_reconstruction_nll": losses.vae_reconstruction_nll.item(),
                            "eval/vae_reconstruction_nll_weighted": vae_reconstruction_nll_weighted.item(),
                            "eval/inv_reconstruction_nll": losses.inv_reconstruction_nll.item(),
                            "eval/inv_reconstruction_nll_weighted": inv_reconstruction_nll_weighted.item(),
                            "eval/comp_reconstruction_nll": losses.comp_reconstruction_nll.item(),
                            "eval/comp_reconstruction_nll_weighted": comp_reconstruction_nll_weighted.item(),
                            "eval/live_to_target_l2": losses.live_to_target_l2.item(),
                            "eval/live_to_target_l2_weighted": live_to_target_l2_weightes.item(),
                            "eval/inv_latent_l2": losses.target_inv_l2.item(),
                            "eval/inv_latent_l2_weighted": inv_latent_l2_weight.item(),
                            "eval/comp_latent_l2": losses.target_comp_l2.item(),
                            "eval/comp_latent_l2_weighted": comp_latent_l2_weight.item(),
                            "eval/latent_weight": get_latent_weight(
                                step,
                                config.latent_warmup_delay_steps,
                                config.latent_warmup_steps,
                                config.latent_warmup_start_weight,
                            ),
                            "eval/ema_tau": current_ema_tau,
                            "eval/vae_gamma": current_gamma,
                            "eval/group_operations_weight_mult": group_operations_weight_mult,
                        },
                        step=step,
                    )

            logger.info(
                f"[Eval step {step}] total={total_loss:.4f}, "
                f"kl={losses.kl_losses.item():.4f}, "
                f"vae_recon={losses.vae_reconstruction_nll.item():.4f}, "
                f"inv_recon={losses.inv_reconstruction_nll.item():.4f}, "
                f"comp_recon={losses.comp_reconstruction_nll.item():.4f}, "
                f"live_to_target_l2={losses.live_to_target_l2.item():.4f}, "
                f"inv_latent_l2={losses.target_inv_l2.item():.4f}, "
                f"comp_latent_l2={losses.target_comp_l2.item():.4f}, "
                f"gamma={current_gamma:.4f}, "
            )

        # -------------------
        #      TRAIN
        # -------------------
        # Commented out on purpose for now
        # for model in learned_networks_tuple:
        #     model.train()

        # Calculate the current gamma value based on warmup schedule
        current_gamma = get_vae_gamma(
            step,
            config.vae.gamma_warmup_steps,
            config.vae.gamma_start,
            config.vae.gamma_final,
        )

        losses = training_loop_iteration(
            TrainingLoopInput(
                learned_networks_tuple=learned_networks_tuple,
                target_networks_tuple=target_networks_tuple,
                dataloaders=(inversions_dataloader, composition_dataloader),
                vae_gamma=current_gamma,
                device=device,
                dtype=dtype,
            )
        )
        kl_weight = torch.tensor(
            get_kl_weight(
                step,
                config.vae.kl_warmup_steps,
                config.vae.kl_warmup_start_weight,
                config.vae.kl_loss_weight,
            ),
            device=device,
            dtype=dtype,
        )
        kl_weighted_loss = losses.kl_losses * kl_weight
        vae_reconstruction_nll_weighted = (
            losses.vae_reconstruction_nll * config.reconstruction_loss_weight
        )
        group_operations_weight_mult = (
            1
            if step > config.group_operations_warmup_steps
            else step / config.group_operations_warmup_steps
        )
        inv_reconstruction_nll_weighted = (
            losses.inv_reconstruction_nll
            * config.neural_inv_perm_loss_weight
            * group_operations_weight_mult
        )
        comp_reconstruction_nll_weighted = (
            losses.comp_reconstruction_nll
            * config.neural_comp_perm_loss_weight
            * group_operations_weight_mult
        )
        latent_l2_losses_weight = torch.tensor(
            get_latent_weight(
                step,
                config.latent_warmup_delay_steps,
                config.latent_warmup_steps,
                config.latent_warmup_start_weight,
            ),
            device=device,
            dtype=dtype,
        )
        live_to_target_l2_weightes = (
            losses.live_to_target_l2
            * config.latent_sampled_perm_loss_weight
            * latent_l2_losses_weight
        )
        inv_latent_l2_weight = (
            losses.target_inv_l2
            * config.latent_inv_perm_loss_weight
            * latent_l2_losses_weight
            * group_operations_weight_mult
        )
        comp_latent_l2_weight = (
            losses.target_comp_l2
            * config.latent_comp_perm_loss_weight
            * latent_l2_losses_weight
            * group_operations_weight_mult
        )
        total_loss = (
            kl_weighted_loss
            + vae_reconstruction_nll_weighted
            + inv_reconstruction_nll_weighted
            + comp_reconstruction_nll_weighted
            + live_to_target_l2_weightes
            + inv_latent_l2_weight
            + comp_latent_l2_weight
        )

        total_loss.backward()  # type: ignore
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Calculate ema_tau once instead of repeatedly
        current_ema_tau = get_ema_tau(
            step,
            config.ema_tau_warmup_steps,
            config.ema_tau_start,
            config.ema_tau_final,
        )

        # Update target network with EMA if enabled
        if config.use_ema_target:
            # Define source and target networks pairs
            source_target_pairs = [
                (permutation_encoder, target_networks_tuple[0]),
                (positional_seq_encoder, target_networks_tuple[1]),
            ]

            # Update all target networks with the same tau value
            for source_net, target_net in source_target_pairs:
                ema_update(source_net, target_net, current_ema_tau)

        if step % config.log_interval == 0:
            if config.wandb_enabled:
                wandb.log(
                    {
                        "train/total_loss": total_loss.item(),
                        "train/kl_loss": losses.kl_losses.item(),
                        "train/kl_loss_weighted": kl_weighted_loss.item(),
                        "train/vae_reconstruction_nll": losses.vae_reconstruction_nll.item(),
                        "train/vae_reconstruction_nll_weighted": vae_reconstruction_nll_weighted.item(),
                        "train/inv_reconstruction_nll": losses.inv_reconstruction_nll.item(),
                        "train/inv_reconstruction_nll_weighted": inv_reconstruction_nll_weighted.item(),
                        "train/comp_reconstruction_nll": losses.comp_reconstruction_nll.item(),
                        "train/comp_reconstruction_nll_weighted": comp_reconstruction_nll_weighted.item(),
                        "train/live_to_target_l2": losses.live_to_target_l2.item(),
                        "train/live_to_target_l2_weighted": live_to_target_l2_weightes.item(),
                        "train/inv_latent_l2": losses.target_inv_l2.item(),
                        "train/inv_latent_l2_weighted": inv_latent_l2_weight.item(),
                        "train/comp_latent_l2": losses.target_comp_l2.item(),
                        "train/comp_latent_l2_weighted": comp_latent_l2_weight.item(),
                        "train/latent_weight": latent_l2_losses_weight.item(),
                        "train/kl_weight": kl_weight.item(),
                        "train/ema_tau": current_ema_tau,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/vae_gamma": current_gamma,
                        "train/group_operations_weight_mult": group_operations_weight_mult,
                    },
                    step=step,
                )

            # Log metrics to console - these variables are not needed
            logger.info(
                f"[Train step {step}] total={total_loss:.4f}, "
                f"kl={losses.kl_losses.item():.4f}, "
                f"vae_recon={losses.vae_reconstruction_nll.item():.4f}, "
                f"inv_recon={losses.inv_reconstruction_nll.item():.4f}, "
                f"comp_recon={losses.comp_reconstruction_nll.item():.4f}, "
                f"live_to_target_l2={losses.live_to_target_l2.item():.4f}, "
                f"inv_latent_l2={losses.target_inv_l2.item():.4f}, "
                f"comp_latent_l2={losses.target_comp_l2.item():.4f}, "
                f"lr={scheduler.get_last_lr()[0]:.6f}, "
                f"gamma={current_gamma:.4f}, "
            )

    return 0


if __name__ == "__main__":
    main()
