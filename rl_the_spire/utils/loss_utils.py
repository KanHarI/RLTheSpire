import torch


def get_kl_weight(
    current_step: int,
    warmup_steps: int,
    warmup_start_weight: float,
    target_weight: float,
) -> float:
    """
    Calculate the KL divergence weight with cosine annealing warmup.

    Args:
        current_step: Current training step
        warmup_steps: Number of warmup steps
        warmup_start_weight: Initial KL weight at step 0
        target_weight: Target KL weight after warmup

    Returns:
        Current KL weight
    """
    # Cosine annealing of KL weight from start_weight to target weight
    if current_step < warmup_steps:
        progress = float(current_step) / float(max(1, warmup_steps))
        cosine_factor = 0.5 * (
            1.0 - torch.cos(torch.tensor(torch.pi * progress)).item()
        )
        return (
            warmup_start_weight + (target_weight - warmup_start_weight) * cosine_factor
        )
    return target_weight


def get_vae_gamma(
    current_step: int,
    warmup_steps: int,
    gamma_start: float,
    gamma_final: float,
) -> float:
    """
    Calculate the VAE gamma parameter with cosine annealing warmup.

    Args:
        current_step: Current training step
        warmup_steps: Number of warmup steps
        gamma_start: Initial gamma value at step 0
        gamma_final: Final gamma value after warmup

    Returns:
        Current gamma value
    """
    # Cosine annealing of gamma from start to final value
    if current_step < warmup_steps:
        progress = float(current_step) / float(max(1, warmup_steps))
        cosine_factor = 0.5 * (
            1.0 - torch.cos(torch.tensor(torch.pi * progress)).item()
        )
        return gamma_start + (gamma_final - gamma_start) * cosine_factor
    return gamma_final


def get_latent_weight(
    current_step: int,
    warmup_delay_steps: int,
    warmup_steps: int,
    warmup_start_weight: float = 0.0,
) -> float:
    """
    Calculate the latent loss weight with a delay period followed by cosine annealing warmup.

    Args:
        current_step: Current training step
        warmup_delay_steps: Number of steps to delay before starting warmup
        warmup_steps: Number of warmup steps after delay
        warmup_start_weight: Initial weight at the start of warmup

    Returns:
        Current latent weight
    """
    # Keep weight at 0 during delay period
    if current_step < warmup_delay_steps:
        return 0.0

    # Cosine annealing of latent loss weights from start_weight to target weight (1.0)
    # after the delay period
    warmup_step = current_step - warmup_delay_steps
    if warmup_step < warmup_steps:
        progress = float(warmup_step) / float(max(1, warmup_steps))
        # We want a function that starts at warmup_start_weight and ends at 1.0
        # Use simple cosine schedule from 0->1
        cosine_factor = 0.5 * (
            1.0 - torch.cos(torch.tensor(torch.pi * progress)).item()
        )
        return warmup_start_weight + (1.0 - warmup_start_weight) * cosine_factor
    return 1.0
