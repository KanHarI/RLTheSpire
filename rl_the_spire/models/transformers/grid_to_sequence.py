import dataclasses
import logging
import math
from typing import Callable

import torch

# Assuming MLP and MLPConfig are defined elsewhere following similar conventions.
from rl_the_spire.models.common.mlp import MLP, MLPConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GridToSequenceConfig:
    """
    Configuration for the GridToSequenceTransformerBlock.

    Attributes:
        grid_n_embed (int): Dimensionality of grid token embeddings.
        seq_n_embed (int): Dimensionality of the output sequence embeddings.
        n_pos_embed (int): Dimensionality of the positional embeddings provided as query.
        n_heads (int): Number of attention heads.
        attn_dropout (float): Dropout probability for the attention weights.
        resid_dropout (float): Dropout probability for the residual connections.
        mlp_dropout (float): Dropout probability in the MLP.
        linear_size_multiplier (int): Multiplier for the hidden layer size in the MLP.
        activation (Callable[[torch.Tensor], torch.Tensor]): Activation function used in the MLP.
        dtype (torch.dtype): Data type for parameters.
        device (torch.device): Device to store the parameters.
        init_std (float): Standard deviation for weight initialization.
        ln_eps (float): Epsilon value for layer normalization.
    """

    grid_n_embed: int
    seq_n_embed: int
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


class GridToSequence(torch.nn.Module):
    """
    Transforms grid features of shape (B, N, M, grid_n_embed) together with query positional embeddings
    of shape (B, L, n_pos_embed) into a sequence representation of shape (B, L, seq_n_embed)
    using multi-head cross-attention followed by an MLP, each with residual connections.

    Linear projections are implemented using learned parameters (via torch.zeros and nn.Parameter)
    and all linear operations are performed with einsum.
    """

    def __init__(self, config: GridToSequenceConfig):
        super().__init__()
        self.config = config

        sequence_to_q_mlp_config = MLPConfig(
            n_in=config.seq_n_embed,
            n_out=config.seq_n_embed,
            linear_size_multiplier=config.linear_size_multiplier,
            activation=config.activation,
            dtype=config.dtype,
            device=config.device,
            init_std=config.init_std,
            dropout=config.mlp_dropout,
        )
        self.sequence_to_q_mlp = MLP(sequence_to_q_mlp_config)

        self.grid_to_kv_proj = torch.nn.Parameter(
            torch.zeros(
                (config.grid_n_embed, config.seq_n_embed * 2),
                dtype=config.dtype,
                device=config.device,
                requires_grad=True,
            )
        )
        self.grid_to_kv_proj_bias = torch.nn.Parameter(
            torch.zeros(
                (config.seq_n_embed * 2,),
                dtype=config.dtype,
                device=config.device,
                requires_grad=True,
            )
        )
        self.c_proj = torch.nn.Parameter(
            torch.zeros(
                (config.seq_n_embed, config.seq_n_embed),
                dtype=config.dtype,
                device=config.device,
                requires_grad=True,
            )
        )
        self.attn_dropout = torch.nn.Dropout(config.attn_dropout)
        self.resid_dropout = torch.nn.Dropout(config.resid_dropout)
        self.head_size = config.seq_n_embed // config.n_heads
        self.inv_sqrt_head_size = 1.0 / math.sqrt(self.head_size)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            logger.warning(
                "WARNING: flash attention not available, using regular attention"
            )

    def init_weights(self) -> None:
        torch.nn.init.normal_(self.grid_to_kv_proj, std=self.config.init_std)
        torch.nn.init.normal_(self.grid_to_kv_proj_bias, std=self.config.init_std)
        torch.nn.init.normal_(self.c_proj, std=self.config.init_std)

    def forward(self, x: torch.Tensor, pos_encodings: torch.Tensor) -> torch.Tensor:
        *B, R, C, Eg = x.shape
        *_, L, Es = pos_encodings.shape

        # Flatten grid dimensions
        x = x.reshape(*B, R * C, Eg)

        x = (
            torch.einsum("...gi,ij->...gj", x, self.grid_to_kv_proj)
            + self.grid_to_kv_proj_bias
        )
        k, v = x.split(self.config.seq_n_embed, dim=-1)  # type: ignore
        q = pos_encodings + self.sequence_to_q_mlp(pos_encodings)

        # q: (L, Embed)
        q = q.reshape(L, self.config.n_heads, self.head_size).transpose(0, 1)
        k = k.reshape(*B, R * C, self.config.n_heads, self.head_size).transpose(1, 2)
        v = v.reshape(*B, R * C, self.config.n_heads, self.head_size).transpose(1, 2)

        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.config.attn_dropout if self.training else 0.0,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * self.inv_sqrt_head_size
            att = torch.nn.functional.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            out = att @ v

        out = out.transpose(1, 2).contiguous().view(*B, L, self.config.seq_n_embed)
        out = pos_encodings + self.resid_dropout(
            torch.einsum("...ti,ij->...tj", (out, self.c_proj))
        )
        return out
