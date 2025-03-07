from typing import Callable

import torch


def get_activation(activation_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation_name == "gelu":
        return torch.nn.GELU()
    elif activation_name == "relu":
        return torch.nn.ReLU()
    elif activation_name == "tanh":
        return torch.nn.Tanh()
    elif activation_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation_name == "softplus":
        return torch.nn.Softplus()
    elif activation_name == "softsign":
        return torch.nn.Softsign()
    elif activation_name == "swish":
        return torch.nn.SiLU()
    elif activation_name == "mish":
        return torch.nn.Mish()
    else:
        raise ValueError(f"Activation {activation_name} not found")
