import dataclasses

from rl_the_spire.conf.direct_probability_distribution_encoding.direct_probability_distribution_dataloader_config import (
    DirectProbabilityDistributionDataloaderConfig,
)


@dataclasses.dataclass
class DirectProbabilityDistributionEncodingExperimentConfig:
    dataloader: DirectProbabilityDistributionDataloaderConfig
    n_batch_size: int
