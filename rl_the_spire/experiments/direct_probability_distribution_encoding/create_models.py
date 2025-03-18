from typing import Tuple

from rl_the_spire.conf.direct_probability_distribution_encoding.direct_probability_distribution_encoding_experiment_config import (
    DirectProbabilityDistributionEncodingExperimentConfig,
)
from rl_the_spire.conf.utils.devices import get_device
from rl_the_spire.conf.utils.dtypes import get_dtype
from rl_the_spire.models.direct_probability_distribution.direct_probability_distribution_embedder import (
    DirectProbabilityDistributionEmbedder,
    DirectProbabilityDistributionEmbedderConfig,
)
from rl_the_spire.models.position_encodings.positional_sequence_encoder import (
    PositionalSequenceEncoder,
    PositionalSequenceEncoderConfig,
)


def create_models(
    config: DirectProbabilityDistributionEncodingExperimentConfig,
) -> Tuple[PositionalSequenceEncoder, DirectProbabilityDistributionEmbedder]:
    device = get_device(config.device)
    dtype = get_dtype(config.dtype)

    positional_sequence_encoder_config = PositionalSequenceEncoderConfig(
        n_embed=config.n_embed,
        max_seq_len=config.dataloader.n_symbols
        + 1,  # +1 - for the encoded distribution
        device=device,
        dtype=dtype,
        init_std=config.init_std,
    )
    positional_sequence_encoder = PositionalSequenceEncoder(
        positional_sequence_encoder_config
    )
    positional_sequence_encoder.init_weights()

    direct_probability_distribution_embedder_config = (
        DirectProbabilityDistributionEmbedderConfig(
            n_symbols=config.dataloader.n_symbols,
            n_embed=config.n_embed,
            device=device,
            dtype=dtype,
            init_std=config.init_std,
        )
    )
    direct_probability_distribution_embedder = DirectProbabilityDistributionEmbedder(
        direct_probability_distribution_embedder_config
    )
    direct_probability_distribution_embedder.init_weights()

    return positional_sequence_encoder, direct_probability_distribution_embedder
