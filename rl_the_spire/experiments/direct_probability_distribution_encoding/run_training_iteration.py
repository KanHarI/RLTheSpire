import dataclasses
from typing import Iterator, Tuple

import torch

from rl_the_spire.models.direct_probability_distribution.direct_probability_distribution_embedder import (
    DirectProbabilityDistributionEmbedder,
)
from rl_the_spire.models.direct_probability_distribution.distribution_sampler import (
    DistributionSampler,
)
from rl_the_spire.models.position_encodings.positional_sequence_encoder import (
    PositionalSequenceEncoder,
)


@dataclasses.dataclass
class TrainingIterationInput:
    learned_models: Tuple[
        PositionalSequenceEncoder,
        DirectProbabilityDistributionEmbedder,
        DistributionSampler,
    ]
    dataloader: Iterator[Tuple[torch.Tensor, torch.Tensor]]
    device: torch.device
    dtype: torch.dtype
    distribution_n_tokens: int
    n_symbols: int
    n_embed: int
    n_samples_per_distribution: int


@dataclasses.dataclass
class TrainingIterationOutput:
    kl_losses: torch.Tensor  # Loss between actual and reconstructed distribution
    entropy_loss: torch.Tensor  # Average entropy post-reconstruction


def run_training_iteration(input: TrainingIterationInput) -> TrainingIterationOutput:
    (
        positional_sequence_encoder,
        direct_probability_distribution_embedder,
        distribution_sampler,
    ) = input.learned_models
    dataloader = input.dataloader

    used_symbols, distribution_targets = next(dataloader)
    used_symbols = used_symbols.to(input.device)
    distribution_targets = distribution_targets.to(input.device)

    encoded_probability_distributions = direct_probability_distribution_embedder(
        used_symbols,
        distribution_targets,
        positional_sequence_encoder,
    )[:, :, -input.distribution_n_tokens :]

    sampled_distributions = distribution_sampler(
        encoded_probability_distributions,
        positional_sequence_encoder,
    )

    print(sampled_distributions.shape)
    return TrainingIterationOutput(
        kl_losses=torch.tensor(0.0),
        entropy_loss=torch.tensor(0.0),
    )
