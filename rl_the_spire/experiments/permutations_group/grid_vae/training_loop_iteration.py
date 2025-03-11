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


def training_loop_iteration(
    learned_networks_tuple: Tuple[
        PermutationGridEncoder,
        PositionalSequenceEncoder,
        ConvTransformerBody,
        PermutationGridDecoder,
        ConvTransformerBody,
        PermutationComposer,
        ConvTransformerBody,
        PositionalGridEncoder,
    ],
    dataloaders: Tuple[
        Iterator[Tuple[torch.Tensor, torch.Tensor]],
        Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ],
    device: torch.device,
) -> None:
    inversions_dataloader, composition_dataloader = dataloaders

    perm, inv = next(inversions_dataloader)
    p, q, r = next(composition_dataloader)

    # Convert input tensors to the right device - keep as long for inputs
    perm = perm.to(device)
    inv = inv.to(device)
    p = p.to(device)
    q = q.to(device)
    r = r.to(device)
    pass
