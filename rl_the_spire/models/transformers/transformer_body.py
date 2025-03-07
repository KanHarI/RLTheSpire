import dataclasses
from typing import Callable

import torch

from rl_the_spire.models.transformers.transformer_block import (
    TransformerBlock,
    TransformerBlockConfig,
)


@dataclasses.dataclass
class TransformerBodyConfig:
    n_layers: int
    n_embed: int
    n_extra_embed: int
    n_heads: int
    attn_dropout: float
    resid_dropout: float
    dtype: torch.dtype
    device: torch.device
    linear_size_multiplier: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    mlp_dropout: float
    init_std: float
    ln_eps: float


class TransformerBody(torch.nn.Module):
    def __init__(self, config: TransformerBodyConfig):
        super().__init__()
        self.config = config
        self.first_block_config = TransformerBlockConfig(
            n_embed=self.config.n_embed,
            n_extra_embed=self.config.n_extra_embed,
            n_heads=self.config.n_heads,
            attn_dropout=self.config.attn_dropout,
            resid_dropout=self.config.resid_dropout,
            dtype=self.config.dtype,
            device=self.config.device,
            linear_size_multiplier=self.config.linear_size_multiplier,
            activation=self.config.activation,
            mlp_dropout=self.config.mlp_dropout,
            init_std=self.config.init_std,
            ln_eps=self.config.ln_eps,
        )
        self.first_block = TransformerBlock(self.first_block_config)
        self.other_blocks_config = TransformerBlockConfig(
            n_embed=self.config.n_embed,
            n_extra_embed=0,
            n_heads=self.config.n_heads,
            attn_dropout=self.config.attn_dropout,
            resid_dropout=self.config.resid_dropout,
            dtype=self.config.dtype,
            device=self.config.device,
            linear_size_multiplier=self.config.linear_size_multiplier,
            activation=self.config.activation,
            mlp_dropout=self.config.mlp_dropout,
            init_std=self.config.init_std,
            ln_eps=self.config.ln_eps,
        )
        self.other_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(self.other_blocks_config)
                for _ in range(self.config.n_layers - 1)
            ]
        )

    def init_weights(self) -> None:
        self.first_block.init_weights()
        for block in self.other_blocks:
            block.init_weights()

    def forward(self, x: torch.Tensor, extra_embed: torch.Tensor) -> torch.Tensor:
        x = self.first_block(x, extra_embed)
        for block in self.other_blocks:
            x = block(x)
        return x
