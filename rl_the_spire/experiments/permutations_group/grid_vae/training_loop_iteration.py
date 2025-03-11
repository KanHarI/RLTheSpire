import dataclasses
from typing import Iterator, Tuple

import torch

from rl_the_spire.models.permutations.permutation_composer import PermutationComposer
from rl_the_spire.models.permutations.permutation_grid_decoder import (
    PermutationGridDecoder,
)
from rl_the_spire.models.permutations.permutation_grid_encoder import (
    PermutationGridEncoder,
)
from rl_the_spire.models.position_encodings.positional_grid_encoder import (
    PositionalGridEncoder,
)
from rl_the_spire.models.position_encodings.positional_sequence_encoder import (
    PositionalSequenceEncoder,
)
from rl_the_spire.models.transformers.conv_transformer_body import ConvTransformerBody
from rl_the_spire.models.vaes.gamma_vae_sample import gamma_vae_sample
from rl_the_spire.models.vaes.kl_loss import kl_loss


@dataclasses.dataclass
class TrainingLoopInput:
    learned_networks_tuple: Tuple[
        PermutationGridEncoder,
        PositionalSequenceEncoder,
        ConvTransformerBody,
        PermutationGridDecoder,
        ConvTransformerBody,
        PermutationComposer,
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
    live_to_target_l2: torch.Tensor


TOTAL_ENCODED_PERMUTATIONS = 5


def training_loop_iteration(
    tl_input: TrainingLoopInput,
) -> TrainingLoopIterationOutput:
    (
        permutation_encoder,
        positional_seq_encoder,
        denoiser_network,
        permutations_decoder,
        inverter_network,
        composer_network,
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
    for mus, logvars in zip(
        [ep_perm_mus, ep_inv_mus, ep_p_mus, ep_q_mus, ep_r_mus],
        [ep_perm_logvars, ep_inv_logvars, ep_p_logvars, ep_q_logvars, ep_r_logvars],
    ):
        kl_losses += kl_loss(mus, logvars).mean(dim=0).sum()

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
            gamma_vae_sample(ep_perm_mus, ep_perm_logvars, tl_input.vae_gamma)
        )
    )
    sampled_inv = denoiser_network(
        positional_grid_encoder(
            gamma_vae_sample(ep_inv_mus, ep_inv_logvars, tl_input.vae_gamma)
        )
    )
    sampled_p = denoiser_network(
        positional_grid_encoder(
            gamma_vae_sample(ep_p_mus, ep_p_logvars, tl_input.vae_gamma)
        )
    )
    sampled_q = denoiser_network(
        positional_grid_encoder(
            gamma_vae_sample(ep_q_mus, ep_q_logvars, tl_input.vae_gamma)
        )
    )
    sampled_r = denoiser_network(
        positional_grid_encoder(
            gamma_vae_sample(ep_r_mus, ep_r_logvars, tl_input.vae_gamma)
        )
    )

    # Calculate live to target L2 losses
    live_to_target_l2 = torch.tensor(0.0, device=tl_input.device, dtype=tl_input.dtype)
    for sampled, target_mus in zip(
        [sampled_perm, sampled_inv, sampled_p, sampled_q, sampled_r],
        [te_perm_mus, te_inv_mus, ep_p_logvars, ep_q_logvars, ep_r_logvars],
    ):
        live_to_target_l2 += (
            torch.norm(
                live_to_target_adapter(positional_grid_encoder(sampled)) - target_mus,
                p=2,
                dim=1,
            ).mean()
            / TOTAL_ENCODED_PERMUTATIONS
        )

    return TrainingLoopIterationOutput(
        kl_losses=kl_losses,
        live_to_target_l2=live_to_target_l2,
    )
