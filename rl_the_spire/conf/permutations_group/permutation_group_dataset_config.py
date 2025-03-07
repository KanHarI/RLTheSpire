import dataclasses


@dataclasses.dataclass
class PermutationGroupDatasetConfig:
    n_max_permutation_size: int
    gamma: float
    batch_size: int
    num_workers: int
    prefetch_factor: int
