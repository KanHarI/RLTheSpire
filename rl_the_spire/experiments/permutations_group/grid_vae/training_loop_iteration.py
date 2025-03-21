import dataclasses
from typing import Iterator, Tuple

import torch

from rl_the_spire.models.permutations.grid.permutation_grid_decoder import (
    PermutationGridDecoder,
)
from rl_the_spire.models.permutations.grid.permutation_grid_encoder import (
    PermutationGridEncoder,
)
from rl_the_spire.models.permutations.permutation_composer import PermutationComposer
from rl_the_spire.models.position_encodings.positional_grid_encoder import (
    PositionalGridEncoder,
)
from rl_the_spire.models.position_encodings.positional_sequence_encoder import (
    PositionalSequenceEncoder,
)
from rl_the_spire.models.transformers.conv_transformer_body import ConvTransformerBody
from rl_the_spire.models.vaes.gamma_vae_sample import gamma_vae_sample
from rl_the_spire.models.vaes.kl_loss import vectorized_kl_loss


@dataclasses.dataclass
class TrainingLoopInput:
    learned_networks_tuple: Tuple[
        PermutationGridEncoder,
        PositionalSequenceEncoder,
        torch.nn.Linear,
        ConvTransformerBody,
        PermutationGridDecoder,
        ConvTransformerBody,
        PermutationComposer,
        torch.nn.Linear,
        ConvTransformerBody,
        PositionalGridEncoder,
    ]
    target_networks_tuple: Tuple[
        PermutationGridEncoder,
        PositionalSequenceEncoder,
    ]
    dataloaders: Tuple[
        Iterator[Tuple[torch.Tensor, torch.Tensor]],
        Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]
    vae_gamma: float
    device: torch.device
    dtype: torch.dtype


@dataclasses.dataclass
class TrainingLoopIterationOutput:
    kl_losses: torch.Tensor
    vae_reconstruction_nll: torch.Tensor
    inv_reconstruction_nll: torch.Tensor
    comp_reconstruction_nll: torch.Tensor
    target_inv_l2: torch.Tensor
    target_comp_l2: torch.Tensor
    live_to_target_l2: torch.Tensor


TOTAL_ENCODED_PERMUTATIONS = 5


def training_loop_iteration(
    tl_input: TrainingLoopInput,
) -> TrainingLoopIterationOutput:
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
    ) = tl_input.learned_networks_tuple

    (
        target_permutation_encoder,
        target_positional_seq_encoder,
    ) = tl_input.target_networks_tuple

    inversions_dataloader, composition_dataloader = tl_input.dataloaders

    perm, inv = next(inversions_dataloader)
    p, q, r = next(composition_dataloader)

    # Convert input tensors to the right device - keep as long for inputs
    perm = perm.to(tl_input.device)
    inv = inv.to(tl_input.device)
    p = p.to(tl_input.device)
    q = q.to(tl_input.device)
    r = r.to(tl_input.device)

    # Encode all permutations
    ep_perm_mus, ep_perm_logvars = permutation_encoder(positional_seq_encoder, perm)
    ep_inv_mus, ep_inv_logvars = permutation_encoder(positional_seq_encoder, inv)
    ep_p_mus, ep_p_logvars = permutation_encoder(positional_seq_encoder, p)
    ep_q_mus, ep_q_logvars = permutation_encoder(positional_seq_encoder, q)
    ep_r_mus, ep_r_logvars = permutation_encoder(positional_seq_encoder, r)

    # Calculate encoder KL losses
    kl_losses = torch.tensor(0.0, device=tl_input.device, dtype=tl_input.dtype)

    all_mus = torch.stack([ep_perm_mus, ep_inv_mus, ep_p_mus, ep_q_mus, ep_r_mus])
    all_logvars = torch.stack(
        [ep_perm_logvars, ep_inv_logvars, ep_p_logvars, ep_q_logvars, ep_r_logvars]
    )

    # Replace the for-loop with vectorized operation
    kl_losses = vectorized_kl_loss(all_mus, all_logvars).mean()

    # Encode all permutations using the target encoder
    with torch.no_grad():
        te_perm_mus, _ = target_permutation_encoder(target_positional_seq_encoder, perm)
        te_inv_mus, _ = target_permutation_encoder(target_positional_seq_encoder, inv)
        te_p_mus, _ = target_permutation_encoder(target_positional_seq_encoder, p)
        te_q_mus, _ = target_permutation_encoder(target_positional_seq_encoder, q)
        te_r_mus, _ = target_permutation_encoder(target_positional_seq_encoder, r)

    # Sample and denoise all permutations
    sampled_perm = denoiser_network(
        positional_grid_encoder(
            denoiser_dim_expander(
                gamma_vae_sample(ep_perm_mus, ep_perm_logvars, tl_input.vae_gamma, 1)
            ),
        )
    )
    sampled_inv = denoiser_network(
        positional_grid_encoder(
            denoiser_dim_expander(
                gamma_vae_sample(ep_inv_mus, ep_inv_logvars, tl_input.vae_gamma, 1)
            ),
        )
    )
    sampled_p = denoiser_network(
        positional_grid_encoder(
            denoiser_dim_expander(
                gamma_vae_sample(ep_p_mus, ep_p_logvars, tl_input.vae_gamma, 1),
            )
        )
    )
    sampled_q = denoiser_network(
        positional_grid_encoder(
            denoiser_dim_expander(
                gamma_vae_sample(ep_q_mus, ep_q_logvars, tl_input.vae_gamma, 1),
            )
        )
    )
    sampled_r = denoiser_network(
        positional_grid_encoder(
            denoiser_dim_expander(
                gamma_vae_sample(ep_r_mus, ep_r_logvars, tl_input.vae_gamma, 1),
            )
        )
    )

    all_mus = torch.stack([ep_perm_mus, ep_inv_mus, ep_p_mus, ep_q_mus, ep_r_mus])
    all_samples = torch.stack(
        [sampled_perm, sampled_inv, sampled_p, sampled_q, sampled_r]
    )

    # Calculate live to target L2 losses
    live_to_target_l2 = torch.tensor(0.0, device=tl_input.device, dtype=tl_input.dtype)

    all_targets = torch.stack([te_perm_mus, te_inv_mus, te_p_mus, te_q_mus, te_r_mus])
    live_to_target_l2 = torch.norm(
        live_to_target_dimensionality_reducer(live_to_target_adapter(all_samples))
        - all_targets,
        p=2,
        dim=(2, 3, 4),
    ).mean()

    # Decode from latent space (VAE)
    dec_perm = permutations_decoder(positional_seq_encoder, sampled_perm)
    dec_inv = permutations_decoder(positional_seq_encoder, sampled_inv)
    dec_p = permutations_decoder(positional_seq_encoder, sampled_p)
    dec_q = permutations_decoder(positional_seq_encoder, sampled_q)
    dec_r = permutations_decoder(positional_seq_encoder, sampled_r)

    # Reconstruction loss
    vae_reconstruction_nll = torch.tensor(
        0.0, device=tl_input.device, dtype=tl_input.dtype
    )
    for dec, orig in zip(
        [dec_perm, dec_inv, dec_p, dec_q, dec_r],
        [perm, inv, p, q, r],
    ):
        vae_reconstruction_nll += (
            permutation_encoder.embedder.nll_loss(dec, orig).mean()
            / TOTAL_ENCODED_PERMUTATIONS
        )

    # Neural group operations
    neural_inv_perm = inverter_network(sampled_perm)
    neural_comp_perm = composer_network(sampled_p, sampled_q)

    # Decode after neural group operations
    dec_neural_inv_perm = permutations_decoder(positional_seq_encoder, neural_inv_perm)
    dec_neural_comp_perm = permutations_decoder(
        positional_seq_encoder, neural_comp_perm
    )

    # Construct l2 loss to target network inv, r
    target_inv_l2 = torch.norm(
        live_to_target_dimensionality_reducer(live_to_target_adapter(neural_inv_perm))
        - te_inv_mus,
        p=2,
        dim=(1, 2, 3),
    ).mean()
    target_comp_l2 = torch.norm(
        live_to_target_dimensionality_reducer(live_to_target_adapter(neural_comp_perm))
        - te_r_mus,
        p=2,
        dim=(1, 2, 3),
    ).mean()

    # Reconstruction loss after neural group operations
    inv_reconstruction_nll = permutation_encoder.embedder.nll_loss(
        dec_neural_inv_perm, inv
    ).mean()
    comp_reconstruction_nll = permutation_encoder.embedder.nll_loss(
        dec_neural_comp_perm, r
    ).mean()

    return TrainingLoopIterationOutput(
        kl_losses=kl_losses,
        vae_reconstruction_nll=vae_reconstruction_nll,
        inv_reconstruction_nll=inv_reconstruction_nll,
        comp_reconstruction_nll=comp_reconstruction_nll,
        live_to_target_l2=live_to_target_l2,
        target_inv_l2=target_inv_l2,
        target_comp_l2=target_comp_l2,
    )
