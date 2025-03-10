import torch


def gamma_vae_sample(
    mus: torch.Tensor, logvars: torch.Tensor, gamma: float
) -> torch.Tensor:
    """
    Sample from a Gaussian distribution using the reparameterization trick with an extra uniform scaling factor.

    For each item in the batch:
      1. Sample a scalar u ~ Uniform(0, 1) and raise it to the power gamma.
      2. Compute the standard deviation sigma = exp(0.5 * logvars).
      3. Sample a standard normal epsilon.
      4. Compute z = mu + (u^gamma) * sigma * epsilon, where the scaling factor (u^gamma)
         is applied uniformly to all dimensions of that sample.

    Args:
        mus (torch.Tensor): Mean tensor of shape [B, ...].
        logvars (torch.Tensor): Log-variance tensor of shape [B, ...].
        gamma (float): Exponent controlling the scaling of the standard deviation.
                       gamma=0 yields standard sampling, higher gammas reduce effective variance.

    Returns:
        torch.Tensor: A sample tensor of shape [B, ...].
    """
    # Compute standard deviation.
    sigma = torch.exp(0.5 * logvars)

    # Sample one uniform value per batch item.
    # Create a tensor of shape [B, 1, 1, ..., 1] that can broadcast to the shape of mus.
    batch_shape = (mus.shape[0],) + (1,) * (mus.dim() - 1)
    u = torch.rand(batch_shape, dtype=mus.dtype, device=mus.device)

    # Compute the scaling factor.
    scaling = u**gamma

    # Sample epsilon with the same shape as mus.
    epsilon = torch.randn_like(mus)

    # Return the reparameterized sample.
    return mus + scaling * sigma * epsilon  # type: ignore
