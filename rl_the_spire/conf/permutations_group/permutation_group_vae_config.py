import dataclasses


@dataclasses.dataclass
class PermutationGroupVAEConfig:
    gamma: float
    kl_loss_weight: float
