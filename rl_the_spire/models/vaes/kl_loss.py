import torch


def kl_loss(mus: torch.Tensor, logvars: torch.Tensor) -> torch.Tensor:
    """
    Compute the KL divergence loss between the approximate posterior N(mu, sigma^2)
    and the standard normal prior N(0, 1) for each sample.

    Uses the formula:
        KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    The summation is performed over all dimensions except the batch dimension.

    Args:
        mus (torch.Tensor): Mean tensor of shape [B, ...].
        logvars (torch.Tensor): Log-variance tensor of shape [B, ...].

    Returns:
        torch.Tensor: A tensor of shape [B] containing the KL divergence for each sample.
    """
    # Sum over all dimensions except the first (batch dimension)
    dims = tuple(range(1, mus.dim()))
    kl = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp(), dim=dims)
    return kl


def vectorized_kl_loss(
    mus: torch.Tensor,
    logvars: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the KL divergence loss between the approximate posterior N(mu, sigma^2)
    and the standard normal prior N(0, 1) for stacked tensors.

    This function handles tensors with an additional stacking dimension before the batch dimension,
    allowing for vectorized computation of KL losses for multiple distributions.

    Args:
        mus (torch.Tensor): Mean tensor of shape [S, B, ...] where S is the stacking dimension.
        logvars (torch.Tensor): Log-variance tensor of shape [S, B, ...].

    Returns:
        torch.Tensor: KL divergence losses.
            Shape depends on reduction parameter.
    """
    # Sum over all dimensions except the stacking and batch dimensions
    dims = tuple(range(2, mus.dim()))
    kl = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp(), dim=dims)

    return kl
