import torch


def gamma_vae_sample(
    mus: torch.Tensor, logvars: torch.Tensor, gamma: float, batch_dims: int
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
        batch_dims (int): Number of batch dimensions to keep.

    Returns:
        torch.Tensor: A sample tensor of shape [B, ...].
    """
    # Compute standard deviation.
    sigma = torch.exp(0.5 * logvars)

    # Sample one uniform value per batch item.
    # Create a tensor of shape [B, 1, 1, ..., 1] that can broadcast to the shape of mus.
    batch_dims_shape = list(mus.shape[:batch_dims])
    padding_dims = [1] * (mus.dim() - batch_dims)
    batch_shape = batch_dims_shape + padding_dims
    u = torch.rand(batch_shape, dtype=mus.dtype, device=mus.device)

    # Compute the scaling factor.
    scaling = u**gamma

    # Sample epsilon with the same shape as mus.
    epsilon = torch.randn_like(mus)

    # Return the reparameterized sample.
    result: torch.Tensor = mus + scaling * sigma * epsilon
    return result
