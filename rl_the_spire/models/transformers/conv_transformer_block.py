import dataclasses
from typing import Callable

import torch
import torch.nn as nn
import numpy as np

from rl_the_spire.models.common.mlp import MLP, MLPConfig
from rl_the_spire.models.transformers.multi_headed_attention import (
    MultiHeadedAttention,
    MultiHeadedAttentionConfig,
)


@dataclasses.dataclass
class ConvTransformerBlockConfig:
    """
    Configuration for the ResidualConvTransformerBlock.

    Attributes:
        n_embed (int): Dimensionality of the token embeddings.
        n_heads (int): Number of attention heads.
        attn_dropout (float): Dropout probability for the attention modules.
        resid_dropout (float): Dropout probability for the residual connections.
        mlp_dropout (float): Dropout probability for the MLP.
        linear_size_multiplier (int): Multiplier for the hidden layer size in the MLP.
        activation (Callable[[torch.Tensor], torch.Tensor]): Activation function used in the MLP.
        dtype (torch.dtype): Data type for parameters.
        device (str): Device to store the parameters.
        init_std (float): Standard deviation for weight initialization.
        ln_eps (float): Epsilon value for layer normalization.
    """

    n_embed: int
    n_heads: int
    attn_dropout: float
    resid_dropout: float
    mlp_dropout: float
    linear_size_multiplier: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    dtype: torch.dtype
    device: torch.device
    init_std: float
    ln_eps: float = 1e-5


class ConvTransformerBlock(nn.Module):
    """
    A Residual Conv Transformer Block operating on input tensors of shape (B, N, M, EMBED).

    The block applies sequentially:
      1. Attention along the N dimension (attending over rows).
      2. Attention along the M dimension (attending over columns).
      3. A 3x3 convolution on the N x M plane.
      4. An MLP applied to each embedding vector.

    Each sub-block is preceded by layer normalization and combined with a residual connection.
    """

    def __init__(self, config: ConvTransformerBlockConfig):
        super().__init__()
        self.config = config
        n_embed = config.n_embed

        # Layer norms for each sub-block.
        self.ln_attn_n = nn.LayerNorm(
            n_embed, eps=config.ln_eps, device=config.device, dtype=config.dtype
        )
        self.ln_attn_m = nn.LayerNorm(
            n_embed, eps=config.ln_eps, device=config.device, dtype=config.dtype
        )
        self.ln_conv = nn.LayerNorm(
            n_embed, eps=config.ln_eps, device=config.device, dtype=config.dtype
        )
        self.ln_mlp_1 = nn.LayerNorm(
            n_embed, eps=config.ln_eps, device=config.device, dtype=config.dtype
        )
        self.ln_mlp_2 = nn.LayerNorm(
            n_embed, eps=config.ln_eps, device=config.device, dtype=config.dtype
        )
        self.ln_mlp_3 = nn.LayerNorm(
            n_embed, eps=config.ln_eps, device=config.device, dtype=config.dtype
        )

        # Multi-head attention along N dimension.
        # For attention along N, we flatten over the M dimension so that each column forms a sequence.
        attn_n_config = MultiHeadedAttentionConfig(
            n_embed_in=n_embed,
            n_embed_kqv=n_embed,
            n_embed_out=n_embed,
            n_heads=config.n_heads,
            attn_dropout=config.attn_dropout,
            resid_dropout=config.resid_dropout,
            dtype=config.dtype,
            device=config.device,
            init_std=config.init_std,
        )
        self.attn_n = MultiHeadedAttention(attn_n_config)

        # Multi-head attention along M dimension.
        # For attention along M, we flatten over the N dimension so that each row forms a sequence.
        attn_m_config = MultiHeadedAttentionConfig(
            n_embed_in=n_embed,
            n_embed_kqv=n_embed,
            n_embed_out=n_embed,
            n_heads=config.n_heads,
            attn_dropout=config.attn_dropout,
            resid_dropout=config.resid_dropout,
            dtype=config.dtype,
            device=config.device,
            init_std=config.init_std,
        )
        self.attn_m = MultiHeadedAttention(attn_m_config)

        # 3x3 convolution on the N x M spatial plane.
        # The convolution is applied in channel-first format: (B, EMBED, N, M).
        self.conv = nn.Conv2d(
            in_channels=n_embed,
            out_channels=n_embed,
            kernel_size=3,
            padding=1,  # maintain spatial dimensions
            bias=True,
            device=config.device,
            dtype=config.dtype,
        )

        # MLP for further processing.
        mlp_config = MLPConfig(
            n_in=n_embed,
            n_out=n_embed,
            linear_size_multiplier=config.linear_size_multiplier,
            activation=config.activation,
            dropout=config.mlp_dropout,
            dtype=config.dtype,
            device=config.device,
            init_std=config.init_std,
        )
        self.mlp_1 = MLP(mlp_config)
        self.mlp_2 = MLP(mlp_config)
        self.mlp_3 = MLP(mlp_config)

    def init_weights(self) -> None:
        """
        Initialize weights for the attention modules, convolution, and MLP.
        """
        self.attn_n.init_weights()
        self.attn_m.init_weights()
        torch.nn.init.normal_(self.conv.weight, mean=0.0, std=self.config.init_std)
        if self.conv.bias is not None:
            torch.nn.init.constant_(self.conv.bias, 0.0)
        self.mlp_1.init_weights()
        self.mlp_2.init_weights()
        self.mlp_3.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualConvTransformerBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (*B, N, M, EMBED), where *B represents
                              arbitrary leading batch dimensions (e.g., [S, B, N, M, E]).
                              N dimension represents rows, M dimension represents columns.

        Returns:
            torch.Tensor: Output tensor of shape (*B, N, M, EMBED) with residual connections.
        """
        # Get the dimensions
        dims = len(x.shape)
        *B, N, M, E = x.shape
        
        # Handle the case with no batch dimensions
        B = B if B else []
        batch_dims = dims - 3  # Number of batch dimensions (everything except N, M, E)
        
        # Calculate the product of leading batch dimensions
        B_prod = np.prod(B) if B else 1
        
        # Define dimension indices for clarity
        n_idx, m_idx, e_idx = batch_dims, batch_dims + 1, batch_dims + 2
        batch_indices = list(range(batch_dims))
        
        # --- Attention along N dimension ---
        x_ln = self.ln_attn_n(x)
        
        # For attention along N, we need to reshape to make N the sequence dimension
        # First permute to shape (*B, M, N, E) so that N becomes the "sequence" dimension
        forward_perm = batch_indices + [m_idx, n_idx, e_idx]  # Put M before N
        x_n = x_ln.permute(*forward_perm)  # shape: (*B, M, N, E)
        
        # Reshape to combine batch dims and M for attention
        x_n = x_n.reshape(B_prod * M, N, E)
        attn_n_out = self.attn_n(x_n)  # shape: (B_prod*M, N, E)
        
        # Reshape back and permute to original order
        attn_n_out = attn_n_out.reshape(*B, M, N, E)
        # Permute from (*B, M, N, E) back to (*B, N, M, E)
        # This is the inverse of the forward_perm which was batch_indices + [m_idx, n_idx, e_idx]
        # To go from (*B, M, N, E) to (*B, N, M, E), we need to swap the M and N positions
        inverse_perm = batch_indices.copy()  # Start with batch dimensions
        # Then add dimensions in the order N, M, E
        inverse_perm.extend([batch_dims + 1, batch_dims, batch_dims + 2])
        attn_n_out = attn_n_out.permute(*inverse_perm)
        
        # Add residual connection
        x = x + attn_n_out
        
        # MLP after first attention
        x = x + self.mlp_1(self.ln_mlp_1(x))
        
        # --- Attention along M dimension ---
        x_ln = self.ln_attn_m(x)
        
        # For attention along M, we can directly reshape
        # Reshape to combine batch dims and N
        x_m = x_ln.reshape(B_prod * N, M, E)
        attn_m_out = self.attn_m(x_m)  # shape: (B_prod*N, M, E)
        
        # Reshape back to original shape
        attn_m_out = attn_m_out.reshape(*B, N, M, E)
        
        # Add residual connection
        x = x + attn_m_out
        
        # MLP after second attention
        x = x + self.mlp_2(self.ln_mlp_2(x))
        
        # --- 3x3 Convolution on the (N, M) plane ---
        x_ln = self.ln_conv(x)
        
        # Reshape for convolution: (*B, N, M, E) -> (B_flat, E, N, M)
        # First flatten batch dimensions
        x_reshaped = x_ln.reshape(-1, N, M, E)
        # Then permute to channel-first format for convolution
        x_perm = x_reshaped.permute(0, 3, 1, 2)  # (B_flat, E, N, M)
        
        # Apply convolution
        conv_out = self.conv(x_perm)  # Still (B_flat, E, N, M)
        
        # Permute back and reshape to original shape
        conv_out = conv_out.permute(0, 2, 3, 1)  # (B_flat, N, M, E)
        conv_out = conv_out.reshape(*B, N, M, E)
        
        # Add residual connection
        x = x + conv_out
        
        # Final MLP
        x = x + self.mlp_3(self.ln_mlp_3(x))
        
        return x
