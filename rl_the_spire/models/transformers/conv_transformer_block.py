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
        # If B is empty (no batch dimensions), set it to an empty list for reshape operations
        B = B if B else []
        batch_dims = dims - 3  # Number of batch dimensions (everything except N, M, E)
        
        # Calculate the product of leading batch dimensions
        B_prod = np.prod(B) if B else 1
        
        # --- Attention along N dimension ---
        # Normalize 
        x_ln = self.ln_attn_n(x)
        
        # Create dynamic permutation indices to swap N and M
        # For shape (*B, N, M, E) to (*B, M, N, E)
        batch_indices = list(range(batch_dims))
        n_idx, m_idx, e_idx = batch_dims, batch_dims + 1, batch_dims + 2
        forward_perm = batch_indices + [m_idx, n_idx, e_idx]  # Put M before N
        
        # Reshape for attention on N dimension
        x_n = x_ln.permute(*forward_perm)  # shape: (*B, M, N, E)
        # Reshape to combine batch dims and M
        x_n = x_n.reshape(B_prod * M, N, E)
        attn_n_out = self.attn_n(x_n)  # shape: (B_prod*M, N, E)
        
        # Reshape back to (*B, M, N, E)
        attn_n_out = attn_n_out.reshape(*B, M, N, E)
        
        # Permute back to original dimension order (*B, N, M, E)
        inverse_perm = batch_indices + [n_idx, m_idx, e_idx]  # Put N before M
        attn_n_out = attn_n_out.permute(*inverse_perm)
        
        x = x + attn_n_out
        
        x = x + self.mlp_1(self.ln_mlp_1(x))
        
        # --- Attention along M dimension ---
        # Normalize and reshape to (B_prod*N, M, E): each row (fixed N) is a sequence of length M.
        x_ln = self.ln_attn_m(x)
        # Reshape to combine batch dims and N
        x_m = x_ln.reshape(B_prod * N, M, E)
        attn_m_out = self.attn_m(x_m)  # shape: (B_prod*N, M, E)
        attn_m_out = attn_m_out.reshape(*B, N, M, E)
        x = x + attn_m_out
        
        x = x + self.mlp_2(self.ln_mlp_2(x))
        
        # --- 3x3 Convolution on the (N, M) plane ---
        # Normalize and permute for convolution
        x_ln = self.ln_conv(x)
        
        # First reshape to (-1, N, M, E) by flattening all leading batch dimensions
        x_reshaped = x_ln.reshape(-1, N, M, E)
        
        # Standard permutation for convolutional format (B_flat, E, N, M)
        x_perm = x_reshaped.permute(0, 3, 1, 2)
        conv_out = self.conv(x_perm)
        
        # Reshape back to original shape
        conv_out = conv_out.permute(0, 2, 3, 1)  # back to (B_flat, N, M, E)
        conv_out = conv_out.reshape(*B, N, M, E)  # restore leading batch dimensions
        x = x + conv_out
        
        x = x + self.mlp_3(self.ln_mlp_3(x))
        
        return x
