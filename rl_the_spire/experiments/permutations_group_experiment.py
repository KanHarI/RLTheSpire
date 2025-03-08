import copy
import logging
import platform
from typing import Any

import dacite
import hydra
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
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
from rl_the_spire.models.permutations.permutation_composer import (
    PermutationComposer,
    PermutationComposerConfig,
)
from rl_the_spire.models.permutations.permutation_decoder import (
    PermutationDecoder,
    PermutationDecoderConfig,
)
from rl_the_spire.models.permutations.permutation_encoder import (
    PermutationEncoder,
    PermutationEncoderConfig,
)
from rl_the_spire.models.transformers.conv_transformer_body import (
    ConvTransformerBody,
    ConvTransformerBodyConfig,
)
from rl_the_spire.models.vaes.gamma_vae_sample import gamma_vae_sample
from rl_the_spire.models.vaes.kl_loss import kl_loss

# Configure logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../conf/permutations_group",  # Adjust path if needed
    config_name="default",  # The name of your .yaml file, e.g. "default.yaml"
    version_base=None,
)
def main(hydra_cfg: dict[Any, Any]) -> int:
    # 1. Parse the Hydra config into our dataclasses
    config: PermutationGroupExperimentConfig = dacite.from_dict(
        data_class=PermutationGroupExperimentConfig,
        data=hydra_cfg,
    )

    logger.info("Initializing experiment...")

    TOTAL_ENCODED_PERMUTATIONS = 5  # constant used to average out loss components

    # 2. Create Datasets
    logger.info("Creating inversion dataset config...")
    inversions_dataset_config = PermutationInverseDatasetConfig(
        n_max_permutation_size=config.dataset.n_max_permutation_size,
        gamma=config.dataset.gamma,
    )

    logger.info("Creating inversions dataset...")
    inversions_dataset = PermutationAndInverseDataset(inversions_dataset_config)

    logger.info("Creating composition dataset config...")
    composition_dataset_config = ComposedPermutationDatasetConfig(
        n_max_permutation_size=config.dataset.n_max_permutation_size,
        gamma=config.dataset.gamma,
    )

    logger.info("Creating composition dataset...")
    composition_dataset = ComposedPermutationDataset(composition_dataset_config)

    # 3. Create Dataloaders
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
    composition_dataloader = iter(
        DataLoader(
            composition_dataset,
            batch_size=config.dataset.batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
        )
    )

    # 4. Create Models (Encoder/Decoder)
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
    permutation_encoder.init_weights()

    # Create target encoder if using EMA
    target_encoder = None
    if config.use_ema_target:
        logger.info("Initializing EMA target encoder")
        target_encoder = copy.deepcopy(permutation_encoder)
        target_encoder.to(get_device(config.encoder.device))
        # Set to eval mode, we never train this directly
        target_encoder.eval()

    logger.info("Creating permutations decoder...")
    permutations_decoder_config = PermutationDecoderConfig(
        n_embed_grid=config.encoder.n_output_embed,
        n_embed_sequence=config.encoder.n_embed,
        n_grid_rows=config.encoder.n_output_rows,
        n_grid_columns=config.encoder.n_output_columns,
        n_max_permutation_size=config.dataset.n_max_permutation_size,
        n_sequence_layers=config.encoder.n_layers,
        conv_transformer_n_heads=config.conv_transformer.n_heads,
        sequence_n_heads=config.encoder.n_heads,
        linear_size_multiplier=config.encoder.linear_size_multiplier,
        activation=get_activation(config.encoder.activation),
        dtype=get_dtype(config.encoder.dtype),
        device=get_device(config.encoder.device),
        init_std=config.encoder.init_std,
        ln_eps=config.encoder.ln_eps,
        attn_dropout=config.encoder.attn_dropout,
        resid_dropout=config.encoder.resid_dropout,
        mlp_dropout=config.encoder.mlp_dropout,
    )
    permutations_decoder = PermutationDecoder(permutations_decoder_config)
    permutations_decoder.init_weights()

    logger.info("Creating inverter network...")
    inverter_network_config = ConvTransformerBodyConfig(
        n_blocks=config.inverter_network.n_layers,
        n_embed=config.encoder.n_output_embed,
        n_heads=config.conv_transformer.n_heads,
        attn_dropout=config.encoder.attn_dropout,
        resid_dropout=config.encoder.resid_dropout,
        mlp_dropout=config.encoder.mlp_dropout,
        linear_size_multiplier=config.encoder.linear_size_multiplier,
        activation=get_activation(config.encoder.activation),
        dtype=get_dtype(config.encoder.dtype),
        device=get_device(config.encoder.device),
        init_std=config.encoder.init_std,
        ln_eps=config.encoder.ln_eps,
    )
    inverter_network = ConvTransformerBody(inverter_network_config)
    inverter_network.init_weights()

    logger.info("Creating composer network...")
    composer_network_config = PermutationComposerConfig(
        n_embed=config.encoder.n_output_embed,
        n_heads=config.conv_transformer.n_heads,
        attn_dropout=config.encoder.attn_dropout,
        resid_dropout=config.encoder.resid_dropout,
        mlp_dropout=config.encoder.mlp_dropout,
        linear_size_multiplier=config.encoder.linear_size_multiplier,
        activation=get_activation(config.encoder.activation),
        dtype=get_dtype(config.encoder.dtype),
        device=get_device(config.encoder.device),
        init_std=config.encoder.init_std,
        n_layers=config.composer_network.n_layers,
        ln_eps=config.encoder.ln_eps,
    )
    composer_network = PermutationComposer(composer_network_config)
    composer_network.init_weights()

    # Explicitly ensure all model parameters are using the specified dtype and device
    device = get_device(config.encoder.device)
    dtype = get_dtype(config.encoder.dtype)

    # 5. Create an optimizer
    logger.info("Creating AdamW optimizer...")
    params = (
        list(permutation_encoder.parameters())
        + list(permutations_decoder.parameters())
        + list(inverter_network.parameters())
        + list(composer_network.parameters())
    )
    optimizer = torch.optim.AdamW(
        params,
        lr=config.optimizer.lr,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.optimizer.weight_decay,
    )

    # Create a learning rate scheduler with warmup
    logger.info(
        f"Setting up LR scheduler with {config.optimizer.warmup_steps} warmup steps..."
    )

    def lr_lambda(current_step: int) -> float:
        # Linear warmup followed by constant learning rate
        if current_step < config.optimizer.warmup_steps:
            return float(current_step) / float(max(1, config.optimizer.warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Function to calculate KL weight with warmup
    logger.info(
        f"Setting up KL weight warmup over {config.vae.kl_warmup_steps} steps from {config.vae.kl_warmup_start_weight} to {config.vae.kl_loss_weight}..."
    )

    def get_kl_weight(current_step: int) -> float:
        # Linear warmup of KL weight from start_weight to target weight
        if current_step < config.vae.kl_warmup_steps:
            alpha = float(current_step) / float(max(1, config.vae.kl_warmup_steps))
            return config.vae.kl_warmup_start_weight + alpha * (
                config.vae.kl_loss_weight - config.vae.kl_warmup_start_weight
            )
        return config.vae.kl_loss_weight

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
            permutation_encoder.eval()
            permutations_decoder.eval()
            inverter_network.eval()
            composer_network.eval()

            with torch.no_grad():
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
                ep_perm_mus, ep_perm_logvars = permutation_encoder(eval_perm)
                ep_inv_mus, ep_inv_logvars = permutation_encoder(eval_inv)
                ep_p_mus, ep_p_logvars = permutation_encoder(eval_p)
                ep_q_mus, ep_q_logvars = permutation_encoder(eval_q)
                ep_r_mus, ep_r_logvars = permutation_encoder(eval_r)

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

                kl_losses_eval_weighted = kl_losses_eval * get_kl_weight(step)

                # Sample & decode
                samp_perm = gamma_vae_sample(
                    ep_perm_mus, ep_perm_logvars, config.vae.gamma
                )
                samp_inv = gamma_vae_sample(
                    ep_inv_mus, ep_inv_logvars, config.vae.gamma
                )
                samp_p = gamma_vae_sample(ep_p_mus, ep_p_logvars, config.vae.gamma)
                samp_q = gamma_vae_sample(ep_q_mus, ep_q_logvars, config.vae.gamma)
                samp_r = gamma_vae_sample(ep_r_mus, ep_r_logvars, config.vae.gamma)

                # Inverter
                neural_inv_perm = inverter_network(samp_perm)
                neural_comp_perm = composer_network(samp_p, samp_q)

                dec_perm = permutations_decoder(
                    samp_perm, permutation_encoder.embedder.pos_embedding
                )
                dec_inv = permutations_decoder(
                    samp_inv, permutation_encoder.embedder.pos_embedding
                )
                dec_p = permutations_decoder(
                    samp_p, permutation_encoder.embedder.pos_embedding
                )
                dec_q = permutations_decoder(
                    samp_q, permutation_encoder.embedder.pos_embedding
                )
                dec_r = permutations_decoder(
                    samp_r, permutation_encoder.embedder.pos_embedding
                )

                dec_neural_inv_perm = permutations_decoder(
                    neural_inv_perm, permutation_encoder.embedder.pos_embedding
                )
                dec_neural_comp_perm = permutations_decoder(
                    neural_comp_perm, permutation_encoder.embedder.pos_embedding
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
                if config.use_ema_target and target_encoder is not None:
                    with torch.no_grad():
                        target_inv_mus, _ = target_encoder(eval_inv)
                        target_r_mus, _ = target_encoder(eval_r)
                else:
                    target_inv_mus = ep_inv_mus
                    target_r_mus = ep_r_mus

                # Full L2 norm (not mean)
                latent_inv_perm_loss = torch.norm(neural_inv_perm - target_inv_mus, p=2)
                latent_inv_perm_loss_weighted = (
                    latent_inv_perm_loss * config.latent_inv_perm_loss_weight
                )

                # Full L2 norm (not mean)
                latent_comp_perm_loss = torch.norm(neural_comp_perm - target_r_mus, p=2)
                latent_comp_perm_loss_weighted = (
                    latent_comp_perm_loss * config.latent_comp_perm_loss_weight
                )

                total_loss_eval = (
                    kl_losses_eval_weighted
                    + reconstruction_losses_eval_weighted
                    + neural_inv_perm_loss_weighted
                    + neural_comp_perm_loss_weighted
                    + latent_inv_perm_loss_weighted
                    + latent_comp_perm_loss_weighted
                )

            # Log eval losses
            if config.wandb_enabled:
                wandb.log(
                    {
                        "eval/total_loss": total_loss_eval.item(),
                        "eval/kl_loss": kl_losses_eval.item(),  # raw KL
                        "eval/kl_loss_weighted": kl_losses_eval_weighted.item(),
                        "eval/kl_weight": get_kl_weight(step),
                        "eval/reconstruction_loss": reconstruction_losses_eval.item(),
                        "eval/reconstruction_loss_weighted": reconstruction_losses_eval_weighted.item(),
                        "eval/neural_inv_perm_loss": neural_inv_perm_loss.item(),
                        "eval/neural_inv_perm_loss_weighted": neural_inv_perm_loss_weighted.item(),
                        "eval/neural_comp_perm_loss": neural_comp_perm_loss.item(),
                        "eval/neural_comp_perm_loss_weighted": neural_comp_perm_loss_weighted.item(),
                        "eval/latent_inv_perm_loss": latent_inv_perm_loss.item(),
                        "eval/latent_comp_perm_loss": latent_comp_perm_loss.item(),
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
            )

        # -------------------
        #      TRAIN
        # -------------------
        permutation_encoder.train()
        permutations_decoder.train()
        inverter_network.train()
        composer_network.train()

        perm, inv = next(inversions_dataloader)
        p, q, r = next(composition_dataloader)

        # Convert input tensors to the right device - keep as long for inputs
        perm = perm.to(device)
        inv = inv.to(device)
        p = p.to(device)
        q = q.to(device)
        r = r.to(device)

        # Forward pass
        encoded_perm_mus, encoded_perm_logvars = permutation_encoder(perm)
        encoded_inv_mus, encoded_inv_logvars = permutation_encoder(inv)
        encoded_p_mus, encoded_p_logvars = permutation_encoder(p)
        encoded_q_mus, encoded_q_logvars = permutation_encoder(q)
        encoded_r_mus, encoded_r_logvars = permutation_encoder(r)

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
        kl_losses_weighted = kl_losses * get_kl_weight(step)

        # Sample from the latent distribution
        sampled_perm = gamma_vae_sample(
            encoded_perm_mus, encoded_perm_logvars, config.vae.gamma
        )
        sampled_inv = gamma_vae_sample(
            encoded_inv_mus, encoded_inv_logvars, config.vae.gamma
        )
        sampled_p = gamma_vae_sample(encoded_p_mus, encoded_p_logvars, config.vae.gamma)
        sampled_q = gamma_vae_sample(encoded_q_mus, encoded_q_logvars, config.vae.gamma)
        sampled_r = gamma_vae_sample(encoded_r_mus, encoded_r_logvars, config.vae.gamma)

        # Inverter
        neural_inv_perm = inverter_network(sampled_perm)
        neural_comp_perm = composer_network(sampled_p, sampled_q)
        # Decode
        decoded_perm = permutations_decoder(
            sampled_perm, permutation_encoder.embedder.pos_embedding
        )
        decoded_inv = permutations_decoder(
            sampled_inv, permutation_encoder.embedder.pos_embedding
        )
        decoded_p = permutations_decoder(
            sampled_p, permutation_encoder.embedder.pos_embedding
        )
        decoded_q = permutations_decoder(
            sampled_q, permutation_encoder.embedder.pos_embedding
        )
        decoded_r = permutations_decoder(
            sampled_r, permutation_encoder.embedder.pos_embedding
        )
        dec_neural_inv_perm = permutations_decoder(
            neural_inv_perm, permutation_encoder.embedder.pos_embedding
        )
        dec_neural_comp_perm = permutations_decoder(
            neural_comp_perm, permutation_encoder.embedder.pos_embedding
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
        if config.use_ema_target and target_encoder is not None:
            with torch.no_grad():
                target_inv_mus, _ = target_encoder(inv)
                target_r_mus, _ = target_encoder(r)
        else:
            target_inv_mus = encoded_inv_mus
            target_r_mus = encoded_r_mus

        # Full L2 norm (not mean)
        latent_inv_perm_loss = torch.norm(neural_inv_perm - target_inv_mus, p=2)
        latent_inv_perm_loss_weighted = (
            latent_inv_perm_loss * config.latent_inv_perm_loss_weight
        )

        # Full L2 norm (not mean)
        latent_comp_perm_loss = torch.norm(neural_comp_perm - target_r_mus, p=2)
        latent_comp_perm_loss_weighted = (
            latent_comp_perm_loss * config.latent_comp_perm_loss_weight
        )

        # Combine total loss
        total_loss = (
            kl_losses_weighted
            + reconstruction_losses_weighted
            + neural_inv_perm_loss_weighted
            + neural_comp_perm_loss_weighted
            + latent_inv_perm_loss_weighted
            + latent_comp_perm_loss_weighted
        )

        # Optimize
        optimizer.zero_grad()
        total_loss.backward()  # type: ignore
        optimizer.step()
        scheduler.step()

        # Update target network with EMA if enabled
        if config.use_ema_target and target_encoder is not None:
            ema_update(target_encoder, permutation_encoder, config.ema_tau)

        # Log training losses
        if step % config.log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            current_kl_weight = get_kl_weight(step)
            if config.wandb_enabled:
                wandb.log(
                    {
                        "train/total_loss": total_loss.item(),
                        "train/kl_loss": kl_losses.item(),  # raw KL
                        "train/kl_loss_weighted": kl_losses_weighted.item(),
                        "train/kl_weight": current_kl_weight,
                        "train/reconstruction_loss": reconstruction_losses.item(),
                        "train/reconstruction_loss_weighted": reconstruction_losses_weighted.item(),
                        "train/neural_inv_perm_loss": neural_inv_perm_loss.item(),
                        "train/neural_inv_perm_loss_weighted": neural_inv_perm_loss_weighted.item(),
                        "train/neural_comp_perm_loss": neural_comp_perm_loss.item(),
                        "train/neural_comp_perm_loss_weighted": neural_comp_perm_loss_weighted.item(),
                        "train/latent_inv_perm_loss": latent_inv_perm_loss.item(),
                        "train/latent_comp_perm_loss": latent_comp_perm_loss.item(),
                        "train/learning_rate": current_lr,
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
                f"lr={current_lr:.6f}"
            )

    return 0


def ema_update(
    target_model: torch.nn.Module, source_model: torch.nn.Module, tau: float
) -> None:
    """
    Update the target model parameters with the source model parameters using EMA.

    Args:
        target_model: The target model to be updated
        source_model: The source model to update from
        tau: The update factor (small value for slow updates)
    """
    with torch.no_grad():
        for target_param, source_param in zip(
            target_model.parameters(), source_model.parameters()
        ):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )


if __name__ == "__main__":
    main()
