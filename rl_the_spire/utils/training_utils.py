import torch


def lr_lambda(current_step: int, warmup_steps: int) -> float:
    """
    Calculate learning rate multiplier for a learning rate scheduler.
    Implements linear warmup followed by constant learning rate.

    Args:
        current_step: Current training step
        warmup_steps: Number of warmup steps

    Returns:
        Learning rate multiplier
    """
    # Linear warmup followed by constant learning rate
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0


def get_ema_tau(
    current_step: int, warmup_steps: int, tau_start: float, tau_final: float
) -> float:
    """
    Calculate the Exponential Moving Average (EMA) tau parameter with warmup.
    Implements harmonic decrease of EMA tau from start_value to final_value.

    Args:
        current_step: Current training step
        warmup_steps: Number of warmup steps
        tau_start: Initial tau value
        tau_final: Final tau value after warmup

    Returns:
        Current tau value
    """
    # Harmonic decrease of EMA tau from start_value to final_value
    if current_step < warmup_steps:
        # Using harmonic interpolation: 1/(alpha/a + (1-alpha)/b)
        alpha = float(current_step) / float(max(1, warmup_steps))
        start_inv = 1.0 / tau_start
        final_inv = 1.0 / tau_final
        # Interpolate in the reciprocal space
        tau_inv = alpha * final_inv + (1.0 - alpha) * start_inv
        # Convert back to the original space
        return 1.0 / tau_inv
    return tau_final


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
