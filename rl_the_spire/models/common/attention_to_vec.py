import dataclasses
from typing import Callable, Optional

import torch

from rl_the_spire.models.common.mlp import MLP, MLPConfig


@dataclasses.dataclass
class AttentionToVecConfig:
    n_embed: int
    n_output_dim: int
    n_heads: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    linear_size_multiplier: int
    dtype: torch.dtype
    device: str
    init_std: float
    mlp_dropout: float


class AttentionToVec(torch.nn.Module):
    """Extract a vector from an unordered sequence of vectors using attention.
    A permutation invariant operation.
    """

    def __init__(
        self,
        config: AttentionToVecConfig,
    ):
        super().__init__()
        assert config.n_embed % config.n_heads == 0
        assert config.n_output_dim % config.n_heads == 0
        self.config = config
        self.transformer_to_kv = torch.nn.Parameter(
            torch.zeros(
                (self.config.n_embed, self.config.n_output_dim * 2),
                dtype=self.config.dtype,
                device=self.config.device,
                requires_grad=True,
            )
        )
        self.transformer_to_kv_bias = torch.nn.Parameter(
            torch.zeros(
                (self.config.n_output_dim * 2,),
                dtype=self.config.dtype,
                device=self.config.device,
                requires_grad=True,
            )
        )
        self.query = torch.nn.Parameter(
            torch.zeros(
                (
                    self.config.n_heads,
                    self.config.n_output_dim // self.config.n_heads,
                ),
                dtype=self.config.dtype,
                device=self.config.device,
                requires_grad=True,
            )
        )
        self.mlp_config = MLPConfig(
            n_in=self.config.n_output_dim,
            n_out=self.config.n_output_dim,
            linear_size_multiplier=self.config.linear_size_multiplier,
            activation=self.config.activation,
            dtype=self.config.dtype,
            device=self.config.device,
            init_std=self.config.init_std,
            dropout=self.config.mlp_dropout,
        )
        self.mlp = MLP(self.mlp_config)

    def init_weights(self) -> None:
        torch.nn.init.normal_(
            self.transformer_to_kv, mean=0.0, std=self.config.init_std
        )
        torch.nn.init.normal_(
            self.transformer_to_kv_bias, mean=0.0, std=self.config.init_std
        )
        torch.nn.init.normal_(self.query, mean=0.0, std=self.config.init_std)
        self.mlp.init_weights()

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: shape: (batch_size, sequence_len, transformer_n_embed)
        # discrete: shape: (batch_size, sequence_len, num_channels)
        kv = (
            torch.einsum(
                "ij,bni->bnj",
                self.transformer_to_kv,
                x,
            )
            + self.transformer_to_kv_bias
        )
        k, v = torch.split(kv, self.config.n_output_dim, dim=2)
        k = k.view(k.shape[0], k.shape[1], self.config.n_heads, -1)
        v = v.view(v.shape[0], v.shape[1], self.config.n_heads, -1)
        att = torch.einsum(
            "bnhi,hi->bnh",
            k,
            self.query,
        )
        if mask is not None:
            att = torch.where(~mask, att, torch.full_like(att, float("-inf")))
        att = torch.softmax(att, dim=1)
        sampled = torch.einsum(
            "bnhi,bnh->bhi",
            v,
            att,
        )
        sampled = sampled.view(sampled.shape[0], -1)
        sampled = sampled + self.mlp(sampled)
        return sampled
