import dataclasses


@dataclasses.dataclass
class PermutationsGroupExperimentConfig:
    n_max_permutation_size: int
    dataset_gamma: float
    batch_size: int
    dataloader_num_workers: int
    dataloader_prefetch_factor: int
    iterations: int
    eval_interval: int
    experiment_name: str
