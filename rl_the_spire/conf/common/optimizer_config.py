import dataclasses


@dataclasses.dataclass
class OptimizerConfig:
    lr: float
    warmup_steps: int
    beta1: float
    beta2: float
    weight_decay: float
