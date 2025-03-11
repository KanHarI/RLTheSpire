import copy
import logging
from typing import Optional, Tuple

import torch

from rl_the_spire.conf.permutations_group.grid.permutation_group_grid_experiment_config import (
    PermutationGroupGridExperimentConfig,
)
from rl_the_spire.conf.utils.activations import get_activation
from rl_the_spire.models.permutations.permutation_composer import (
    PermutationComposer,
    PermutationComposerConfig,
)
from rl_the_spire.models.permutations.permutation_grid_decoder import (
    PermutationGridDecoder,
    PermutationGridDecoderConfig,
)
from rl_the_spire.models.permutations.permutation_grid_encoder import (
    PermutationGridEncoder,
    PermutationGridEncoderConfig,
)
from rl_the_spire.models.position_encodings.positional_grid_encoder import (
    PositionalGridEncoder,
    PositionalGridEncoderConfig,
)
from rl_the_spire.models.position_encodings.positional_sequence_encoder import (
    PositionalSequenceEncoder,
    PositionalSequenceEncoderConfig,
)
from rl_the_spire.models.transformers.conv_transformer_body import (
    ConvTransformerBody,
    ConvTransformerBodyConfig,
)

logger = logging.getLogger(__name__)


def create_target_models(
    config: PermutationGroupGridExperimentConfig,
    device: torch.device,
    permutation_encoder: PermutationGridEncoder,
    positional_seq_encoder: PositionalSequenceEncoder,
) -> Tuple[Optional[PermutationGridEncoder], Optional[PositionalSequenceEncoder]]:
    """
    Create target models for EMA updates.

    Args:
        config: The experiment configuration
        device: The device to create the models on
        permutation_encoder: The source permutation encoder to copy
        positional_seq_encoder: The source positional sequence encoder to copy

    Returns:
        Tuple containing:
        - target_encoder: EMA target encoder (None if not using EMA targets)
        - target_positional_encoder: EMA target positional encoder (None if not using EMA targets)
    """
    if not config.use_ema_target:
        return None, None

    logger.info("Initializing EMA target encoder")
    target_encoder = copy.deepcopy(permutation_encoder)
    target_encoder.to(device)

    logger.info("Initializing EMA target positional encoder")
    target_positional_encoder = copy.deepcopy(positional_seq_encoder)
    target_positional_encoder.to(device)

    # Initialize parameters to zeros if specified
    if config.init_ema_target_as_zeros:
        logger.info("Setting EMA target encoder parameters to zeros")
        for param in target_encoder.parameters():
            param.data.zero_()

        logger.info("Setting EMA target positional encoder parameters to zeros")
        for param in target_positional_encoder.parameters():
            param.data.zero_()

    # Set to eval mode, we never train this directly
    target_encoder.eval()
    target_positional_encoder.eval()

    return target_encoder, target_positional_encoder


def create_models(
    config: PermutationGroupGridExperimentConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[
    PermutationGridEncoder,
    PositionalSequenceEncoder,
    ConvTransformerBody,
    PermutationGridDecoder,
    ConvTransformerBody,
    PermutationComposer,
    ConvTransformerBody,
    PositionalGridEncoder,
]:
    """
    Create and return models for the grid VAE experiment.

    Args:
        config: The experiment configuration
        device: The device to create the models on
        dtype: The data type to use for the models

    Returns:
        Tuple containing:
        - permutation_encoder: Encoder for permutations
        - positional_seq_encoder: Positional encoder for sequences
        - denoiser_network: Network for denoising
        - permutations_decoder: Decoder for permutations
        - inverter_network: Network for inverting permutations
        - composer_network: Network for composing permutations
        - live_to_target_adapter: Adapter for connecting live models to target models
        - positional_grid_encoder: Positional encoder for grids
    """

    activation = get_activation(config.encoder.activation)

    # Create permutation encoder
    logger.info("Creating permutation encoder...")
    permutation_encoder_config = PermutationGridEncoderConfig(
        n_max_permutation_size=config.dataset.n_max_permutation_size,
        n_embed=config.encoder.n_embed,
        n_heads=config.encoder.n_heads,
        n_layers=config.encoder.n_layers,
        attn_dropout=config.encoder.attn_dropout,
        resid_dropout=config.encoder.resid_dropout,
        dtype=dtype,
        device=device,
        init_std=config.encoder.init_std,
        mlp_dropout=config.encoder.mlp_dropout,
        ln_eps=config.encoder.ln_eps,
        n_output_heads=config.encoder.n_output_heads,
        n_output_embed=config.encoder.n_output_embed,
        n_output_rows=config.encoder.n_output_rows,
        n_output_columns=config.encoder.n_output_columns,
        activation=activation,
        linear_size_multiplier=config.encoder.linear_size_multiplier,
        conv_transformer_n_heads=config.conv_transformer.n_heads,
        conv_transformer_blocks=config.encoder.conv_blocks,
        sigma_output=config.encoder.sigma_output,
    )
    permutation_encoder = PermutationGridEncoder(permutation_encoder_config)
    permutation_encoder.init_weights()

    # Create the positional sequence encoder
    logger.info("Creating positional sequence encoder...")
    positional_seq_encoder_config = PositionalSequenceEncoderConfig(
        n_embed=config.encoder.n_embed,
        max_seq_len=config.dataset.n_max_permutation_size,
        device=device,
        dtype=dtype,
        init_std=config.encoder.init_std,
    )
    positional_seq_encoder = PositionalSequenceEncoder(positional_seq_encoder_config)
    positional_seq_encoder.init_weights()

    # Create denoiser network
    logger.info("Creating denoiser network...")
    denoiser_network_config = ConvTransformerBodyConfig(
        n_blocks=config.conv_transformer.denoiser_blocks,
        n_embed=config.encoder.n_output_embed,
        n_heads=config.conv_transformer.n_heads,
        attn_dropout=config.encoder.attn_dropout,
        resid_dropout=config.encoder.resid_dropout,
        init_std=config.encoder.init_std,
        mlp_dropout=config.encoder.mlp_dropout,
        linear_size_multiplier=config.encoder.linear_size_multiplier,
        activation=activation,
        dtype=dtype,
        device=device,
    )
    denoiser_network = ConvTransformerBody(denoiser_network_config)
    denoiser_network.init_weights()

    # Create permutations decoder
    logger.info("Creating permutations decoder...")
    permutations_decoder_config = PermutationGridDecoderConfig(
        n_embed_grid=config.encoder.n_output_embed,
        n_embed_sequence=config.encoder.n_embed,
        n_grid_rows=config.encoder.n_output_rows,
        n_grid_columns=config.encoder.n_output_columns,
        n_max_permutation_size=config.dataset.n_max_permutation_size,
        n_sequence_layers=config.encoder.n_layers,
        conv_transformer_n_heads=config.conv_transformer.n_heads,
        sequence_n_heads=config.encoder.n_heads,
        linear_size_multiplier=config.encoder.linear_size_multiplier,
        activation=activation,
        dtype=dtype,
        device=device,
        init_std=config.encoder.init_std,
        ln_eps=config.encoder.ln_eps,
        attn_dropout=config.encoder.attn_dropout,
        resid_dropout=config.encoder.resid_dropout,
        mlp_dropout=config.encoder.mlp_dropout,
    )
    permutations_decoder = PermutationGridDecoder(permutations_decoder_config)
    permutations_decoder.init_weights()

    # Create inverter network
    logger.info("Creating inverter network...")
    inverter_network_config = ConvTransformerBodyConfig(
        n_blocks=config.inverter_network.n_layers,
        n_embed=config.encoder.n_output_embed,
        n_heads=config.conv_transformer.n_heads,
        attn_dropout=config.encoder.attn_dropout,
        resid_dropout=config.encoder.resid_dropout,
        mlp_dropout=config.encoder.mlp_dropout,
        linear_size_multiplier=config.encoder.linear_size_multiplier,
        activation=activation,
        dtype=dtype,
        device=device,
        init_std=config.encoder.init_std,
        ln_eps=config.encoder.ln_eps,
    )
    inverter_network = ConvTransformerBody(inverter_network_config)
    inverter_network.init_weights()

    # Create composer network
    logger.info("Creating composer network...")
    composer_network_config = PermutationComposerConfig(
        n_embed=config.encoder.n_output_embed,
        n_heads=config.conv_transformer.n_heads,
        attn_dropout=config.encoder.attn_dropout,
        resid_dropout=config.encoder.resid_dropout,
        mlp_dropout=config.encoder.mlp_dropout,
        linear_size_multiplier=config.encoder.linear_size_multiplier,
        activation=activation,
        dtype=dtype,
        device=device,
        init_std=config.encoder.init_std,
        n_layers=config.composer_network.n_layers,
        ln_eps=config.encoder.ln_eps,
    )
    composer_network = PermutationComposer(composer_network_config)
    composer_network.init_weights()

    # Create live to target adapter
    logger.info("Creating live to target adapter...")
    live_to_target_adapter_config = ConvTransformerBodyConfig(
        n_blocks=config.num_live_to_target_adapter_layers,
        n_embed=config.encoder.n_output_embed,
        n_heads=config.conv_transformer.n_heads,
        attn_dropout=config.encoder.attn_dropout,
        resid_dropout=config.encoder.resid_dropout,
        mlp_dropout=config.encoder.mlp_dropout,
        linear_size_multiplier=config.encoder.linear_size_multiplier,
        activation=activation,
        dtype=dtype,
        device=device,
        init_std=config.encoder.init_std,
        ln_eps=config.encoder.ln_eps,
    )
    live_to_target_adapter = ConvTransformerBody(live_to_target_adapter_config)
    live_to_target_adapter.init_weights()

    # Create positional grid encoder
    logger.info("Creating positional grid encoder...")
    positional_grid_encoder_config = PositionalGridEncoderConfig(
        n_embed=config.encoder.n_output_embed,
        n_rows=config.encoder.n_output_rows,
        n_cols=config.encoder.n_output_columns,
        device=device,
        dtype=dtype,
    )
    positional_grid_encoder = PositionalGridEncoder(positional_grid_encoder_config)
    positional_grid_encoder.init_weights()

    # Return all created models as a tuple
    return (
        permutation_encoder,
        positional_seq_encoder,
        denoiser_network,
        permutations_decoder,
        inverter_network,
        composer_network,
        live_to_target_adapter,
        positional_grid_encoder,
    )
