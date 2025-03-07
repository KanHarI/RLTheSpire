import dataclasses


@dataclasses.dataclass
class PermutationsGroupExperimentConfig:
    n_max_permutation_size: int
    gamma: float
    batch_size: int
    dataloader_num_workers: int
    dataloader_prefetch_factor: int
