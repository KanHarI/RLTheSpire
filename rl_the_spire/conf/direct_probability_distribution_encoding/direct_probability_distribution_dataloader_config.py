import dataclasses


@dataclasses.dataclass
class DirectProbabilityDistributionDataloaderConfig:
    n_symbols: int
    num_workers: int
    prefetch_factor: int
