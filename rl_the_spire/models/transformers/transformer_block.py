import dataclasses
from typing import Callable, Optional

import torch

from rl_the_spire.models.common.mlp import MLP, MLPConfig
from rl_the_spire.models.transformers.multi_headed_attention import (
    MultiHeadedAttention,
    MultiHeadedAttentionConfig,
)


@dataclasses.dataclass
class TransformerBlockConfig:
    n_embed: int
    n_extra_embed: int
    n_heads: int
    attn_dropout: float
    resid_dropout: float
    dtype: torch.dtype
    device: str
    linear_size_multiplier: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    mlp_dropout: float
    init_std: float
    ln_eps: float


class TransformerBlock(torch.nn.Module):
    def __init__(self, config: TransformerBlockConfig):
        super().__init__()
        self.config = config
        self.ln_1 = torch.nn.LayerNorm(
            self.config.n_embed + self.config.n_extra_embed,
            device=self.config.device,
            dtype=self.config.dtype,
            eps=self.config.ln_eps,
        )
        self.attn_config = MultiHeadedAttentionConfig(
            n_embed_in=self.config.n_embed + self.config.n_extra_embed,
            n_embed_kqv=self.config.n_embed,
            n_embed_out=self.config.n_embed,
            n_heads=self.config.n_heads,
            attn_dropout=self.config.attn_dropout,
            resid_dropout=self.config.resid_dropout,
            dtype=self.config.dtype,
            device=self.config.device,
            init_std=self.config.init_std,
        )
        self.attn = MultiHeadedAttention(self.attn_config)
        self.ln_2 = torch.nn.LayerNorm(
            self.config.n_embed + self.config.n_extra_embed,
            device=self.config.device,
            dtype=self.config.dtype,
            eps=self.config.ln_eps,
        )
        self.mlp_config = MLPConfig(
            n_in=self.config.n_embed + self.config.n_extra_embed,
            n_out=self.config.n_embed,
            linear_size_multiplier=self.config.linear_size_multiplier,
            activation=self.config.activation,
            dropout=self.config.mlp_dropout,
            dtype=self.config.dtype,
            device=self.config.device,
            init_std=self.config.init_std,
        )
        self.mlp = MLP(self.mlp_config)

    def init_weights(self) -> None:
        self.attn.init_weights()
        self.mlp.init_weights()

    def forward(
        self, x: torch.Tensor, extra_embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.config.n_extra_embed == 0:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            assert extra_embed is not None
            x = x + self.attn(self.ln_1(torch.cat([x, extra_embed], dim=-1)))
            x = x + self.mlp(self.ln_2(torch.cat([x, extra_embed], dim=-1)))
        return x
