import dataclasses
import logging
import math

import torch.nn

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MultiHeadedAttentionConfig:
    n_embed_in: int
    n_embed_kqv: int
    n_embed_out: int
    n_heads: int
    attn_dropout: float
    resid_dropout: float
    dtype: torch.dtype
    device: torch.device
    init_std: float


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, config: MultiHeadedAttentionConfig) -> None:
        super().__init__()
        self.config = config
        assert config.n_embed_kqv % config.n_heads == 0
        self.c_attn = torch.nn.Parameter(
            torch.zeros(
                (config.n_embed_in, 3 * config.n_embed_kqv),
                dtype=config.dtype,
                device=config.device,
                requires_grad=True,
            )
        )
        self.c_attn_bias = torch.nn.Parameter(
            torch.zeros(
                (3 * config.n_embed_kqv,),
                dtype=config.dtype,
                device=config.device,
                requires_grad=True,
            )
        )
        self.c_proj = torch.nn.Parameter(
            torch.zeros(
                (config.n_embed_kqv, config.n_embed_out),
                dtype=config.dtype,
                device=config.device,
                requires_grad=True,
            )
        )
        self.attn_dropout = torch.nn.Dropout(config.attn_dropout)
        self.resid_dropout = torch.nn.Dropout(config.resid_dropout)
        self.head_size = config.n_embed_kqv // config.n_heads
        self.inv_sqrt_head_size = 1.0 / math.sqrt(self.head_size)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            logger.warning(
                "WARNING: flash attention not available, using regular attention"
            )

    def init_weights(self) -> None:
        torch.nn.init.normal_(self.c_attn, std=self.config.init_std)
        torch.nn.init.normal_(self.c_attn_bias, std=self.config.init_std)
        torch.nn.init.normal_(self.c_proj, std=self.config.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *B, T, C = x.shape
        x = torch.einsum("...ti,ij->...tj", x, self.c_attn) + self.c_attn_bias
        q, k, v = x.split(self.config.n_embed_kqv, dim=-1)  # type: ignore
        q = q.reshape(*B, T, self.config.n_heads, -1).transpose(1, 2)  # (B, nh, T, hs)
        k = k.reshape(*B, T, self.config.n_heads, -1).transpose(1, 2)
        v = v.reshape(*B, T, self.config.n_heads, -1).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
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
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(*B, T, self.config.n_embed_out)
        )  # re-assemble all head outputs side by side
        y = self.resid_dropout(torch.einsum("...ti,ij->...tj", (y, self.c_proj)))
        return y
