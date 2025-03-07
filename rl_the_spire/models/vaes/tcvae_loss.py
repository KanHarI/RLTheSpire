import math

import torch


def tcvae_loss(
    mus: torch.Tensor, logvars: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the KL divergence breakdown for a factorized Gaussian posterior q(z|x)
    against a standard normal prior, returning the total KL along with its breakdown
    into mutual information (MI), total correlation (TC) and dimension-wise KL (DWKL).

    For a minibatch of B samples with latent dimension D:
      - q(z|x) = N(mu, sigma^2) with sigma = exp(0.5 * logvars)
      - The aggregated posterior is approximated using the minibatch.

    The decomposition is as follows:
        KL(q(z|x)||p(z)) = MI + TC + DWKL,
    where:
        MI    = E_{x} [ log q(z|x) - log q(z) ],
        TC    = E_{z ~ q(z)} [ log q(z) - sum_{d} log q(z_d) ],
        DWKL  = sum_{d} KL(q(z_d|x)||p(z_d)).

    Args:
        mus (torch.Tensor): Means of shape [B, D].
        logvars (torch.Tensor): Log-variances of shape [B, D].
        z (torch.Tensor, optional): Latent samples of shape [B, D]. If None, samples are generated
                                    via the reparameterization trick.

    Returns:
        tuple: A 4-tuple (total_kl, mi, tc, dwkl) where each is a scalar tensor.
    """
    B, D = mus.shape

    # --- 1. Compute log probability under q(z|x) for each sample ---
    # log_q_zx = sum_d log N(z_d; mu_d, sigma_d^2)
    # Using the Gaussian log-density formula:
    #   -0.5*(log(2*pi) + log(sigma^2)) - (z-mu)^2/(2*sigma^2)
    log2pi = math.log(2 * math.pi)
    log_q_zx = -0.5 * (((z - mus) ** 2) / torch.exp(logvars) + logvars + log2pi)
    # Sum over latent dimensions => shape [B]
    log_q_zx = log_q_zx.sum(dim=1)

    # --- 2. Estimate log q(z) for each sample ---
    # We compute pairwise log densities:
    # For each sample i and for each sample j in the minibatch, compute:
    #   log N(z_i; mu_j, sigma_j^2)
    # Let mus and logvars have shape [B, D]. We want an output of shape [B, B, D].
    z_expand = z.unsqueeze(1)  # [B, 1, D]
    mus_expand = mus.unsqueeze(0)  # [1, B, D]
    logvars_expand = logvars.unsqueeze(0)  # [1, B, D]
    # Compute per-dimension log-density for each pair (i, j).
    log_density = -0.5 * (
        ((z_expand - mus_expand) ** 2) / torch.exp(logvars_expand)
        + logvars_expand
        + log2pi
    )  # shape: [B, B, D]
    # Sum over latent dimensions => [B, B]
    log_density = log_density.sum(dim=2)  # log q(z_i|x_j)
    # For each sample i, estimate log q(z_i) using log-sum-exp over the batch and subtract log(B)
    # This yields a vector of shape [B].
    log_q_z = torch.logsumexp(log_density, dim=1) - math.log(B)

    # --- 3. Estimate log q(z_d) for each latent dimension ---
    # We compute for each sample i and each dimension d:
    #   log q(z_i[d]) = log (1/B sum_j exp(log N(z_i[d]; mu_j[d], sigma_j[d]^2)))
    z_expand_d = z.unsqueeze(1)  # [B, 1, D]
    mus_expand_d = mus.unsqueeze(0)  # [1, B, D]
    logvars_expand_d = logvars.unsqueeze(0)  # [1, B, D]
    # Compute per-dimension log density (no summation over D)
    log_density_d = -0.5 * (
        ((z_expand_d - mus_expand_d) ** 2) / torch.exp(logvars_expand_d)
        + logvars_expand_d
        + log2pi
    )  # shape: [B, B, D]
    # For each sample i and each dimension d, compute:
    # log_q_z_d[i, d] = logsumexp_{j} (log_density_d[i, j, d]) - log(B)
    log_q_z_d = torch.logsumexp(log_density_d, dim=1) - math.log(B)  # shape: [B, D]
    # Sum over dimensions to get the marginal density log q(z)
    log_q_z_d_sum = log_q_z_d.sum(dim=1)  # shape: [B]

    # --- 4. Compute the three terms ---
    # Mutual Information (MI) = mean[log_q_zx - log_q_z]
    mi = torch.mean(log_q_zx - log_q_z)
    # Total Correlation (TC) = mean[log_q_z - sum_d log_q_z_d]
    tc = torch.mean(log_q_z - log_q_z_d_sum)
    # Dimension-wise KL (DWKL) can be computed in closed form per sample:
    # For each sample i: dwkl_i = sum_d -0.5*(1 + logvar_i - mu_i^2 - exp(logvar_i))
    dwkl = -0.5 * (1 + logvars - mus.pow(2) - logvars.exp())
    dwkl = dwkl.sum(dim=1).mean()

    total_kl = mi + tc + dwkl

    return total_kl, mi, tc, dwkl
