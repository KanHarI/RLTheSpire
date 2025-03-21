import dataclasses
from collections.abc import Iterator
from typing import Tuple

import torch


@dataclasses.dataclass
class LoopIterationInput:
    dataloaders: Tuple[
        Iterator[Tuple[torch.Tensor, torch.Tensor]],
        Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]
    device: torch.device
    dtype: torch.dtype


@dataclasses.dataclass
class LoopIterationOutput:
    kl_loss: torch.Tensor


def loop_iteration(input: LoopIterationInput) -> LoopIterationOutput:
    kl_loss = torch.tensor(0.0, device=input.device, dtype=input.dtype)

    inv_dataloader, comp_dataloader = input.dataloaders

    perm, inv = next(inv_dataloader)
    p, q, r = next(comp_dataloader)

    # Load to device
    perm = perm.to(input.device)
    inv = inv.to(input.device)
    p = p.to(input.device)
    q = q.to(input.device)
    r = r.to(input.device)

    return LoopIterationOutput(kl_loss=kl_loss)
