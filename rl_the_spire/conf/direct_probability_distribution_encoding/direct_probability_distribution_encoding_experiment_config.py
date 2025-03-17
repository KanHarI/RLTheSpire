import dataclasses

from rl_the_spire.conf.common.optimizer_config import OptimizerConfig
from rl_the_spire.conf.direct_probability_distribution_encoding.direct_probability_distribution_dataloader_config import (
    DirectProbabilityDistributionDataloaderConfig,
)


@dataclasses.dataclass
class DirectProbabilityDistributionEncodingExperimentConfig:
    dataloader: DirectProbabilityDistributionDataloaderConfig
    optimizer: OptimizerConfig
    n_batch_size: int
    device: str
    dtype: str
    n_embed: int
    init_std: float
