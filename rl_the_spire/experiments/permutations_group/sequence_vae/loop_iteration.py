import dataclasses

import torch


@dataclasses.dataclass
class LoopIterationInput:
    device: torch.device
    dtype: torch.dtype


@dataclasses.dataclass
class LoopIterationOutput:
    kl_loss: torch.Tensor


def loop_iteration(input: LoopIterationInput) -> LoopIterationOutput:
    kl_loss = torch.tensor(0.0, device=input.device, dtype=input.dtype)

    return LoopIterationOutput(kl_loss=kl_loss)
