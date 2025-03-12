import math

import torch


def lr_lambda(current_step: int, warmup_steps: int) -> float:
    """
    Calculate learning rate multiplier for a learning rate scheduler.
    Implements cosine annealing warmup followed by constant learning rate.

    Args:
        current_step: Current training step
        warmup_steps: Number of warmup steps

    Returns:
        Learning rate multiplier
    """
    # Cosine annealing warmup followed by constant learning rate
    if current_step < warmup_steps:
        # return 0.5 * (1 + math.cos(math.pi * (1 - current_step / warmup_steps)))
        return current_step / warmup_steps
    return 1.0


def get_ema_tau(
    current_step: int, warmup_steps: int, tau_start: float, tau_final: float
) -> float:
    """
    Calculate the Exponential Moving Average (EMA) tau parameter with warmup.
    Implements cosine annealing of EMA tau in harmonic space.

    Args:
        current_step: Current training step
        warmup_steps: Number of warmup steps
        tau_start: Initial tau value
        tau_final: Final tau value after warmup

    Returns:
        Current tau value
    """
    # Cosine annealing of EMA tau in harmonic space
    if current_step < warmup_steps:
        # Convert to reciprocal space
        start_inv = 1.0 / tau_start
        final_inv = 1.0 / tau_final

        # Cosine annealing factor (0->1)
        cosine_factor = 0.5 * (
            1 - math.cos(math.pi * current_step / max(1, warmup_steps))
        )

        # Interpolate in the reciprocal space with cosine schedule
        tau_inv = start_inv + cosine_factor * (final_inv - start_inv)

        # Convert back to the original space
        return 1.0 / tau_inv
    return tau_final


def ema_update(
    target_model: torch.nn.Module, source_model: torch.nn.Module, tau: float
) -> None:
    """
    Update the target model parameters with the source model parameters using EMA.
    The tau parameter is expected to be calculated using cosine annealing in harmonic space.

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
