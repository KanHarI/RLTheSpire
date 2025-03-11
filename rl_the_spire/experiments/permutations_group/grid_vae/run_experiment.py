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
    training_loop_iteration,
)
from rl_the_spire.models.vaes.gamma_vae_sample import gamma_vae_sample
from rl_the_spire.models.vaes.kl_loss import kl_loss
from rl_the_spire.utils.loss_utils import get_kl_weight, get_latent_weight
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

    TOTAL_ENCODED_PERMUTATIONS = 5  # constant used to average out loss components

    # Create all models
    logger.info("Creating models...")

    # Explicitly set device and dtype for all models
    device = get_device(config.encoder.device)
    dtype = get_dtype(config.encoder.dtype)

    learned_networks_tuple = create_models(config, device, dtype)

    (
        permutation_encoder,
        positional_seq_encoder,
        denoiser_network,
        permutations_decoder,
        inverter_network,
        composer_network,
        live_to_target_adapter,
        positional_grid_encoder,
    ) = learned_networks_tuple

    # Create EMA target models if enabled
    target_encoder, target_positional_encoder = create_target_models(
        config, device, permutation_encoder, positional_seq_encoder
    )

    # params_for_optimizer = (
    #     list(permutation_encoder.parameters())
    #     + list(positional_seq_encoder.parameters())
    #     + list(denoiser_network.parameters())
    #     + list(permutations_decoder.parameters())
    #     + list(inverter_network.parameters())
    #     + list(composer_network.parameters())
    #     + list(live_to_target_adapter.parameters())
    #     + list(positional_grid_encoder.parameters())
    # )

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

    if config.wandb_enabled:
        # 6. Initialize Weights & Biases
        logger.info("Initializing WanDB...")
        wandb.init(
            project="rl_the_spire.permutations_group", name=config.experiment_name
        )
        wandb.config.update(config)  # type: ignore

    # 7. Training/Eval Loop
    logger.info("Starting training loop...")

    for step in range(config.iterations):
        # -------------------
        #       EVAL
        # -------------------
        if step % config.eval_interval == 0:
            for model in learned_networks_tuple:
                model.eval()

            with torch.no_grad():
                training_loop_iteration(
                    learned_networks_tuple,
                    (inversions_dataloader, composition_dataloader),
                    device,
                )
                # Sample new data for evaluation
                eval_perm, eval_inv = next(inversions_dataloader)
                eval_p, eval_q, eval_r = next(composition_dataloader)

                # Convert input tensors to the right device - keep as long for inputs
                eval_perm = eval_perm.to(device)
                eval_inv = eval_inv.to(device)
                eval_p = eval_p.to(device)
                eval_q = eval_q.to(device)
                eval_r = eval_r.to(device)

                # Forward pass - always use the main encoder
                ep_perm_mus, ep_perm_logvars = permutation_encoder(
                    positional_seq_encoder, eval_perm
                )
                ep_inv_mus, ep_inv_logvars = permutation_encoder(
                    positional_seq_encoder, eval_inv
                )
                ep_p_mus, ep_p_logvars = permutation_encoder(
                    positional_seq_encoder, eval_p
                )
                ep_q_mus, ep_q_logvars = permutation_encoder(
                    positional_seq_encoder, eval_q
                )
                ep_r_mus, ep_r_logvars = permutation_encoder(
                    positional_seq_encoder, eval_r
                )

                kl_losses_eval = torch.tensor(0.0, device=device, dtype=dtype)
                for mus, logvars in zip(
                    [ep_perm_mus, ep_inv_mus, ep_p_mus, ep_q_mus, ep_r_mus],
                    [
                        ep_perm_logvars,
                        ep_inv_logvars,
                        ep_p_logvars,
                        ep_q_logvars,
                        ep_r_logvars,
                    ],
                ):
                    kl_losses_eval += (
                        kl_loss(mus, logvars).mean(dim=0).sum()
                        / TOTAL_ENCODED_PERMUTATIONS
                    )

                kl_losses_eval_weighted = kl_losses_eval * get_kl_weight(
                    step,
                    config.vae.kl_warmup_steps,
                    config.vae.kl_warmup_start_weight,
                    config.vae.kl_loss_weight,
                )

                # Sample & decode
                sampled_perm = denoiser_network(
                    positional_grid_encoder(
                        gamma_vae_sample(ep_perm_mus, ep_perm_logvars, config.vae.gamma)
                    )
                )
                sampled_inv = denoiser_network(
                    positional_grid_encoder(
                        gamma_vae_sample(ep_inv_mus, ep_inv_logvars, config.vae.gamma)
                    )
                )
                sampled_p = denoiser_network(
                    positional_grid_encoder(
                        gamma_vae_sample(ep_p_mus, ep_p_logvars, config.vae.gamma)
                    )
                )
                sampled_q = denoiser_network(
                    positional_grid_encoder(
                        gamma_vae_sample(ep_q_mus, ep_q_logvars, config.vae.gamma)
                    )
                )
                sampled_r = denoiser_network(
                    positional_grid_encoder(
                        gamma_vae_sample(ep_r_mus, ep_r_logvars, config.vae.gamma)
                    )
                )

                # Inverter
                neural_inv_perm = inverter_network(sampled_perm)
                neural_comp_perm = composer_network(sampled_p, sampled_q)

                dec_perm = permutations_decoder(positional_seq_encoder, sampled_perm)
                dec_inv = permutations_decoder(positional_seq_encoder, sampled_inv)
                dec_p = permutations_decoder(positional_seq_encoder, sampled_p)
                dec_q = permutations_decoder(positional_seq_encoder, sampled_q)
                dec_r = permutations_decoder(positional_seq_encoder, sampled_r)

                dec_neural_inv_perm = permutations_decoder(
                    positional_seq_encoder, neural_inv_perm
                )
                dec_neural_comp_perm = permutations_decoder(
                    positional_seq_encoder, neural_comp_perm
                )

                reconstruction_losses_eval = torch.tensor(
                    0.0, device=device, dtype=dtype
                )
                for dec, orig in zip(
                    [dec_perm, dec_inv, dec_p, dec_q, dec_r],
                    [eval_perm, eval_inv, eval_p, eval_q, eval_r],
                ):
                    reconstruction_losses_eval += (
                        permutation_encoder.embedder.nll_loss(dec, orig)
                        .mean(dim=0)
                        .sum()
                        / TOTAL_ENCODED_PERMUTATIONS
                    )
                reconstruction_losses_eval_weighted = (
                    reconstruction_losses_eval * config.reconstruction_loss_weight
                )

                neural_inv_perm_loss = (
                    permutation_encoder.embedder.nll_loss(dec_neural_inv_perm, eval_inv)
                    .mean(dim=0)
                    .sum()
                )
                neural_inv_perm_loss_weighted = (
                    neural_inv_perm_loss * config.neural_inv_perm_loss_weight
                )

                neural_comp_perm_loss = (
                    permutation_encoder.embedder.nll_loss(dec_neural_comp_perm, eval_r)
                    .mean(dim=0)
                    .sum()
                )
                neural_comp_perm_loss_weighted = (
                    neural_comp_perm_loss * config.neural_comp_perm_loss_weight
                )

                # Add L2 losses in the latent space
                # Use the target encoder for target values if available
                if (
                    config.use_ema_target
                    and target_encoder is not None
                    and target_positional_encoder is not None
                ):
                    with torch.no_grad():
                        target_inv_mus, _ = target_encoder(
                            target_positional_encoder, eval_inv
                        )
                        target_r_mus, _ = target_encoder(
                            target_positional_encoder, eval_r
                        )
                        # Get target encodings for all 5 permutations for the new loss
                        target_perm_mus, _ = target_encoder(
                            target_positional_encoder, eval_perm
                        )
                        target_p_mus, _ = target_encoder(
                            target_positional_encoder, eval_p
                        )
                        target_q_mus, _ = target_encoder(
                            target_positional_encoder, eval_q
                        )
                else:
                    with torch.no_grad():
                        target_inv_mus, _ = permutation_encoder(
                            positional_seq_encoder, eval_inv
                        )
                        target_r_mus, _ = permutation_encoder(
                            positional_seq_encoder, eval_r
                        )
                        # Get target encodings for all 5 permutations for the new loss
                        target_perm_mus, _ = permutation_encoder(
                            positional_seq_encoder, eval_perm
                        )
                        target_p_mus, _ = permutation_encoder(
                            positional_seq_encoder, eval_p
                        )
                        target_q_mus, _ = permutation_encoder(
                            positional_seq_encoder, eval_q
                        )

                latent_inv_perm_loss = torch.norm(
                    live_to_target_adapter(positional_grid_encoder(neural_inv_perm))
                    - target_inv_mus,
                    p=2,
                    dim=1,
                ).mean()
                latent_inv_perm_loss_weighted = (
                    latent_inv_perm_loss
                    * config.latent_inv_perm_loss_weight
                    * get_latent_weight(
                        step,
                        config.latent_warmup_delay_steps,
                        config.latent_warmup_steps,
                        config.latent_warmup_start_weight,
                    )
                )

                latent_comp_perm_loss = torch.norm(
                    live_to_target_adapter(positional_grid_encoder(neural_comp_perm))
                    - target_r_mus,
                    p=2,
                    dim=1,
                ).mean()
                latent_comp_perm_loss_weighted = (
                    latent_comp_perm_loss
                    * config.latent_comp_perm_loss_weight
                    * get_latent_weight(
                        step,
                        config.latent_warmup_delay_steps,
                        config.latent_warmup_steps,
                        config.latent_warmup_start_weight,
                    )
                )

                # Add the new loss for the 5 sampled permutations
                latent_sampled_perm_losses = torch.tensor(
                    0.0, device=device, dtype=dtype
                )
                for sampled, target_mus in zip(
                    [sampled_perm, sampled_inv, sampled_p, sampled_q, sampled_r],
                    [
                        target_perm_mus,
                        target_inv_mus,
                        target_p_mus,
                        target_q_mus,
                        target_r_mus,
                    ],
                ):
                    latent_sampled_perm_losses += (
                        torch.norm(
                            live_to_target_adapter(positional_grid_encoder(sampled))
                            - target_mus,
                            p=2,
                            dim=1,
                        ).mean()
                        / TOTAL_ENCODED_PERMUTATIONS
                    )  # Average over the permutations using the constant

                latent_sampled_perm_losses_weighted = (
                    latent_sampled_perm_losses
                    * config.latent_sampled_perm_loss_weight
                    * get_latent_weight(
                        step,
                        config.latent_warmup_delay_steps,
                        config.latent_warmup_steps,
                        config.latent_warmup_start_weight,
                    )
                )

                total_loss_eval = (
                    kl_losses_eval_weighted
                    + reconstruction_losses_eval_weighted
                    + neural_inv_perm_loss_weighted
                    + neural_comp_perm_loss_weighted
                    + latent_inv_perm_loss_weighted
                    + latent_comp_perm_loss_weighted
                    + latent_sampled_perm_losses_weighted
                )

            # Log eval losses
            if config.wandb_enabled:
                wandb.log(
                    {
                        "eval/total_loss": total_loss_eval.item(),
                        "eval/kl_loss": kl_losses_eval.item(),  # raw KL
                        "eval/kl_loss_weighted": kl_losses_eval_weighted.item(),
                        "eval/kl_weight": get_kl_weight(
                            step,
                            config.vae.kl_warmup_steps,
                            config.vae.kl_warmup_start_weight,
                            config.vae.kl_loss_weight,
                        ),
                        "eval/latent_weight": get_latent_weight(
                            step,
                            config.latent_warmup_delay_steps,
                            config.latent_warmup_steps,
                            config.latent_warmup_start_weight,
                        ),
                        "eval/reconstruction_loss": reconstruction_losses_eval.item(),
                        "eval/reconstruction_loss_weighted": reconstruction_losses_eval_weighted.item(),
                        "eval/neural_inv_perm_loss": neural_inv_perm_loss.item(),
                        "eval/neural_inv_perm_loss_weighted": neural_inv_perm_loss_weighted.item(),
                        "eval/neural_comp_perm_loss": neural_comp_perm_loss.item(),
                        "eval/neural_comp_perm_loss_weighted": neural_comp_perm_loss_weighted.item(),
                        "eval/latent_inv_perm_loss": latent_inv_perm_loss.item(),
                        "eval/latent_inv_perm_loss_weighted": latent_inv_perm_loss_weighted.item(),
                        "eval/latent_comp_perm_loss": latent_comp_perm_loss.item(),
                        "eval/latent_comp_perm_loss_weighted": latent_comp_perm_loss_weighted.item(),
                        "eval/latent_sampled_perm_losses": latent_sampled_perm_losses.item(),
                        "eval/latent_sampled_perm_losses_weighted": latent_sampled_perm_losses_weighted.item(),
                    },
                    step=step,
                )

            logger.info(
                f"[Eval step {step}] total={total_loss_eval:.4f}, "
                f"kl={kl_losses_eval:.4f}, "
                f"recon={reconstruction_losses_eval:.4f}, "
                f"neural_inv_perm_loss={neural_inv_perm_loss:.4f}, "
                f"neural_comp_perm_loss={neural_comp_perm_loss:.4f}, "
                f"latent_inv_perm_loss={latent_inv_perm_loss:.4f}, "
                f"latent_comp_perm_loss={latent_comp_perm_loss:.4f}, "
                f"latent_sampled_perm_losses={latent_sampled_perm_losses:.4f}, "
            )

        # -------------------
        #      TRAIN
        # -------------------
        for model in learned_networks_tuple:
            model.train()

        perm, inv = next(inversions_dataloader)
        p, q, r = next(composition_dataloader)

        # Convert input tensors to the right device - keep as long for inputs
        perm = perm.to(device)
        inv = inv.to(device)
        p = p.to(device)
        q = q.to(device)
        r = r.to(device)

        # Forward pass
        encoded_perm_mus, encoded_perm_logvars = permutation_encoder(
            positional_seq_encoder, perm
        )
        encoded_inv_mus, encoded_inv_logvars = permutation_encoder(
            positional_seq_encoder, inv
        )
        encoded_p_mus, encoded_p_logvars = permutation_encoder(
            positional_seq_encoder, p
        )
        encoded_q_mus, encoded_q_logvars = permutation_encoder(
            positional_seq_encoder, q
        )
        encoded_r_mus, encoded_r_logvars = permutation_encoder(
            positional_seq_encoder, r
        )

        # KL loss (averaged over the 5 permutations)
        kl_losses = torch.tensor(0.0, device=device, dtype=dtype)
        for mus, logvars in zip(
            [
                encoded_perm_mus,
                encoded_inv_mus,
                encoded_p_mus,
                encoded_q_mus,
                encoded_r_mus,
            ],
            [
                encoded_perm_logvars,
                encoded_inv_logvars,
                encoded_p_logvars,
                encoded_q_logvars,
                encoded_r_logvars,
            ],
        ):
            kl_losses += (
                kl_loss(mus, logvars).mean(dim=0).sum() / TOTAL_ENCODED_PERMUTATIONS
            )

        # Weight it by config.vae.kl_loss_weight
        kl_losses_weighted = kl_losses * get_kl_weight(
            step,
            config.vae.kl_warmup_steps,
            config.vae.kl_warmup_start_weight,
            config.vae.kl_loss_weight,
        )

        # Sample from the latent distribution
        sampled_perm = denoiser_network(
            positional_grid_encoder(
                gamma_vae_sample(
                    encoded_perm_mus, encoded_perm_logvars, config.vae.gamma
                )
            )
        )
        sampled_inv = denoiser_network(
            positional_grid_encoder(
                gamma_vae_sample(encoded_inv_mus, encoded_inv_logvars, config.vae.gamma)
            )
        )
        sampled_p = denoiser_network(
            positional_grid_encoder(
                gamma_vae_sample(encoded_p_mus, encoded_p_logvars, config.vae.gamma)
            )
        )
        sampled_q = denoiser_network(
            positional_grid_encoder(
                gamma_vae_sample(encoded_q_mus, encoded_q_logvars, config.vae.gamma)
            )
        )
        sampled_r = denoiser_network(
            positional_grid_encoder(
                gamma_vae_sample(encoded_r_mus, encoded_r_logvars, config.vae.gamma)
            )
        )

        # Inverter
        neural_inv_perm = inverter_network(sampled_perm)
        neural_comp_perm = composer_network(sampled_p, sampled_q)
        # Decode
        decoded_perm = permutations_decoder(positional_seq_encoder, sampled_perm)
        decoded_inv = permutations_decoder(positional_seq_encoder, sampled_inv)
        decoded_p = permutations_decoder(positional_seq_encoder, sampled_p)
        decoded_q = permutations_decoder(positional_seq_encoder, sampled_q)
        decoded_r = permutations_decoder(positional_seq_encoder, sampled_r)
        dec_neural_inv_perm = permutations_decoder(
            positional_seq_encoder, neural_inv_perm
        )
        dec_neural_comp_perm = permutations_decoder(
            positional_seq_encoder, neural_comp_perm
        )

        # Reconstruction losses (averaged over 5 permutations)
        reconstruction_losses = torch.tensor(0.0, device=device, dtype=dtype)
        for decoded, original in zip(
            [decoded_perm, decoded_inv, decoded_p, decoded_q, decoded_r],
            [perm, inv, p, q, r],
        ):
            reconstruction_losses += (
                permutation_encoder.embedder.nll_loss(decoded, original)
                .mean(dim=0)
                .sum()
                / TOTAL_ENCODED_PERMUTATIONS
            )
        reconstruction_losses_weighted = (
            reconstruction_losses * config.reconstruction_loss_weight
        )

        neural_inv_perm_loss = (
            permutation_encoder.embedder.nll_loss(dec_neural_inv_perm, inv)
            .mean(dim=0)
            .sum()
        )

        neural_inv_perm_loss_weighted = (
            neural_inv_perm_loss * config.neural_inv_perm_loss_weight
        )

        neural_comp_perm_loss = (
            permutation_encoder.embedder.nll_loss(dec_neural_comp_perm, r)
            .mean(dim=0)
            .sum()
        )
        neural_comp_perm_loss_weighted = (
            neural_comp_perm_loss * config.neural_comp_perm_loss_weight
        )

        # Add L2 losses in the latent space
        # Use the target encoder for target values if available
        if (
            config.use_ema_target
            and target_encoder is not None
            and target_positional_encoder is not None
        ):
            with torch.no_grad():
                target_inv_mus, _ = target_encoder(target_positional_encoder, inv)
                target_r_mus, _ = target_encoder(target_positional_encoder, r)
                # Get target encodings for all 5 permutations for the new loss
                target_perm_mus, _ = target_encoder(target_positional_encoder, perm)
                target_p_mus, _ = target_encoder(target_positional_encoder, p)
                target_q_mus, _ = target_encoder(target_positional_encoder, q)
        else:
            with torch.no_grad():
                target_inv_mus, _ = permutation_encoder(positional_seq_encoder, inv)
                target_r_mus, _ = permutation_encoder(positional_seq_encoder, r)
                # Get target encodings for all 5 permutations for the new loss
                target_perm_mus, _ = permutation_encoder(positional_seq_encoder, perm)
                target_p_mus, _ = permutation_encoder(positional_seq_encoder, p)
                target_q_mus, _ = permutation_encoder(positional_seq_encoder, q)

        if (
            config.latent_inv_perm_loss_weight > 0
            and get_latent_weight(
                step,
                config.latent_warmup_delay_steps,
                config.latent_warmup_steps,
                config.latent_warmup_start_weight,
            )
            > 0
        ):
            latent_inv_perm_loss = torch.norm(
                live_to_target_adapter(positional_grid_encoder(neural_inv_perm))
                - target_inv_mus,
                p=2,
                dim=1,
            ).mean()
            latent_inv_perm_loss_weighted = (
                latent_inv_perm_loss
                * config.latent_inv_perm_loss_weight
                * get_latent_weight(
                    step,
                    config.latent_warmup_delay_steps,
                    config.latent_warmup_steps,
                    config.latent_warmup_start_weight,
                )
            )
        else:
            latent_inv_perm_loss_weighted = torch.tensor(
                0.0, device=device, dtype=dtype
            )

        if (
            config.latent_comp_perm_loss_weight > 0
            and get_latent_weight(
                step,
                config.latent_warmup_delay_steps,
                config.latent_warmup_steps,
                config.latent_warmup_start_weight,
            )
            > 0
        ):
            latent_comp_perm_loss = torch.norm(
                live_to_target_adapter(positional_grid_encoder(neural_comp_perm))
                - target_r_mus,
                p=2,
                dim=1,
            ).mean()
            latent_comp_perm_loss_weighted = (
                latent_comp_perm_loss
                * config.latent_comp_perm_loss_weight
                * get_latent_weight(
                    step,
                    config.latent_warmup_delay_steps,
                    config.latent_warmup_steps,
                    config.latent_warmup_start_weight,
                )
            )
        else:
            latent_comp_perm_loss_weighted = torch.tensor(
                0.0, device=device, dtype=dtype
            )

        # Add the new loss for the 5 sampled permutations
        if (
            config.latent_sampled_perm_loss_weight > 0
            and get_latent_weight(
                step,
                config.latent_warmup_delay_steps,
                config.latent_warmup_steps,
                config.latent_warmup_start_weight,
            )
            > 0
        ):
            latent_sampled_perm_losses = torch.tensor(0.0, device=device, dtype=dtype)
            for sampled, target_mus in zip(
                [sampled_perm, sampled_inv, sampled_p, sampled_q, sampled_r],
                [
                    target_perm_mus,
                    target_inv_mus,
                    target_p_mus,
                    target_q_mus,
                    target_r_mus,
                ],
            ):
                latent_sampled_perm_losses += (
                    torch.norm(
                        live_to_target_adapter(positional_grid_encoder(sampled))
                        - target_mus,
                        p=2,
                        dim=1,
                    ).mean()
                    / TOTAL_ENCODED_PERMUTATIONS
                )  # Average over the permutations using the constant

            latent_sampled_perm_losses_weighted = (
                latent_sampled_perm_losses
                * config.latent_sampled_perm_loss_weight
                * get_latent_weight(
                    step,
                    config.latent_warmup_delay_steps,
                    config.latent_warmup_steps,
                    config.latent_warmup_start_weight,
                )
            )
        else:
            latent_sampled_perm_losses_weighted = torch.tensor(
                0.0, device=device, dtype=dtype
            )

        # Combine total loss
        total_loss = (
            kl_losses_weighted
            + reconstruction_losses_weighted
            + neural_inv_perm_loss_weighted
            + neural_comp_perm_loss_weighted
            + latent_inv_perm_loss_weighted
            + latent_comp_perm_loss_weighted
            + latent_sampled_perm_losses_weighted
        )

        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Update target network with EMA if enabled
        if (
            config.use_ema_target
            and target_encoder is not None
            and target_positional_encoder is not None
        ):
            ema_update(
                permutation_encoder,
                target_encoder,
                get_ema_tau(
                    step,
                    config.ema_tau_warmup_steps,
                    config.ema_tau_start,
                    config.ema_tau_final,
                ),
            )

            ema_update(
                positional_seq_encoder,
                target_positional_encoder,
                get_ema_tau(
                    step,
                    config.ema_tau_warmup_steps,
                    config.ema_tau_start,
                    config.ema_tau_final,
                ),
            )

        # Log training losses
        if step % config.log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            current_kl_weight = get_kl_weight(
                step,
                config.vae.kl_warmup_steps,
                config.vae.kl_warmup_start_weight,
                config.vae.kl_loss_weight,
            )
            current_latent_weight = get_latent_weight(
                step,
                config.latent_warmup_delay_steps,
                config.latent_warmup_steps,
                config.latent_warmup_start_weight,
            )
            current_ema_tau = get_ema_tau(
                step,
                config.ema_tau_warmup_steps,
                config.ema_tau_start,
                config.ema_tau_final,
            )
            if config.wandb_enabled:
                wandb.log(
                    {
                        "train/total_loss": total_loss.item(),
                        "train/kl_loss": kl_losses.item(),  # raw KL
                        "train/kl_loss_weighted": kl_losses_weighted.item(),
                        "train/reconstruction_loss": reconstruction_losses.item(),
                        "train/reconstruction_loss_weighted": reconstruction_losses_weighted.item(),
                        "train/neural_inv_perm_loss": neural_inv_perm_loss.item(),
                        "train/neural_inv_perm_loss_weighted": neural_inv_perm_loss_weighted.item(),
                        "train/neural_comp_perm_loss": neural_comp_perm_loss.item(),
                        "train/neural_comp_perm_loss_weighted": neural_comp_perm_loss_weighted.item(),
                        "train/latent_inv_perm_loss": latent_inv_perm_loss.item(),
                        "train/latent_comp_perm_loss": latent_comp_perm_loss.item(),
                        "train/latent_sampled_perm_losses": latent_sampled_perm_losses.item(),
                        "train/latent_sampled_perm_losses_weighted": latent_sampled_perm_losses_weighted.item(),
                        "train/learning_rate": current_lr,
                        "train/kl_weight": current_kl_weight,
                        "train/latent_weight": current_latent_weight,
                        "train/ema_tau": current_ema_tau,
                    },
                    step=step,
                )
            logger.info(
                f"[Train step {step}] total={total_loss:.4f}, "
                f"kl={kl_losses:.4f}, "
                f"kl_weight={current_kl_weight:.6f}, "
                f"recon={reconstruction_losses:.4f}, "
                f"neural_inv_perm_loss={neural_inv_perm_loss:.4f}, "
                f"neural_comp_perm_loss={neural_comp_perm_loss:.4f}, "
                f"latent_inv_perm_loss={latent_inv_perm_loss:.4f}, "
                f"latent_comp_perm_loss={latent_comp_perm_loss:.4f}, "
                f"latent_sampled_perm_losses={latent_sampled_perm_losses:.4f}, "
                f"lr={current_lr:.6f}"
            )

    return 0


if __name__ == "__main__":
    main()
