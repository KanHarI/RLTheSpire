import dataclasses


@dataclasses.dataclass
class PermutationGroupLiveToTargetAdapterConfig:
    n_layers: int
    linear_size_multiplier: int
    activation: str
    dropout: float
    use_adapter: bool
