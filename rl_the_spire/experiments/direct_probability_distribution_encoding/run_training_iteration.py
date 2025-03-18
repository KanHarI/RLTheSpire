import dataclasses
from typing import Iterator, Tuple

import torch

from rl_the_spire.models.direct_probability_distribution.direct_probability_distribution_embedder import (
    DirectProbabilityDistributionEmbedder,
)
from rl_the_spire.models.position_encodings.positional_sequence_encoder import (
    PositionalSequenceEncoder,
)


@dataclasses.dataclass
class TrainingIterationInput:
    learned_models: Tuple[
        PositionalSequenceEncoder, DirectProbabilityDistributionEmbedder
    ]
    dataloader: Iterator[Tuple[torch.Tensor, torch.Tensor]]


@dataclasses.dataclass
class TrainingIterationOutput:
    kl_losses: torch.Tensor  # Loss between actual and reconstructed distribution
    entropy_loss: torch.Tensor  # Average entropy post-reconstruction


def run_training_iteration(input: TrainingIterationInput) -> TrainingIterationOutput:
    pass
