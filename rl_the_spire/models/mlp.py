import dataclasses
from typing import Callable

import torch


@dataclasses.dataclass
class MLPConfig:
    n_in: int
    n_out: int
    linear_size_multiplier: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    dtype: torch.dtype
    device: str
    init_std: float
    dropout: float


class MLP(torch.nn.Module):
    """Standard MLP"""

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.c_fc = torch.nn.Parameter(
            torch.zeros(
                (
                    self.config.n_in,
                    self.config.n_out * self.config.linear_size_multiplier,
                ),
                dtype=self.config.dtype,
                device=self.config.device,
                requires_grad=True,
            )
        )
        self.c_fc_bias = torch.nn.Parameter(
            torch.zeros(
                (self.config.n_out * self.config.linear_size_multiplier,),
                dtype=self.config.dtype,
                device=self.config.device,
                requires_grad=True,
            )
        )
        self.c_proj = torch.nn.Parameter(
            torch.zeros(
                (
                    self.config.n_out * self.config.linear_size_multiplier,
                    self.config.n_out,
                ),
                dtype=self.config.dtype,
                device=self.config.device,
                requires_grad=True,
            )
        )
        self.dropout = torch.nn.Dropout(self.config.dropout)
        self.activation = self.config.activation

    def init_weights(self) -> None:
        torch.nn.init.normal_(self.c_fc, std=self.config.init_std)
        torch.nn.init.normal_(self.c_fc_bias, std=self.config.init_std)
        torch.nn.init.normal_(self.c_proj, std=self.config.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, n_in)
        # output: (batch_size, n_out)
        output = torch.einsum("ij,...i->...j", self.c_fc, x) + self.c_fc_bias
        output = self.activation(output)
        output = torch.einsum("ij,...i->...j", self.c_proj, output)
        output = self.dropout(output)
        return output
