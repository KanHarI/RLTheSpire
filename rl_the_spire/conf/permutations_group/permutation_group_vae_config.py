import dataclasses


@dataclasses.dataclass
class PermutationGroupVAEConfig:
    gamma: float
    kl_loss_weight: float
    kl_warmup_steps: int
    kl_warmup_start_weight: float
