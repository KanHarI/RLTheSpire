import dataclasses


@dataclasses.dataclass
class PermutationGroupOptimizerConfig:
    lr: float
    warmup_steps: int
    beta1: float
    beta2: float
    weight_decay: float
