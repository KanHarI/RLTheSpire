import dataclasses

from rl_the_spire.datasets.direct_probability_distribution_dataset import (
    DirectProbabilityDistributionDatasetConfig,
)


@dataclasses.dataclass
class DirectProbabilityDistributionEncodingExperimentConfig:
    dataset: DirectProbabilityDistributionDatasetConfig
