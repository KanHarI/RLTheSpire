import logging
from typing import Any

import dacite
import hydra

from rl_the_spire.conf.direct_probability_distribution_encoding.direct_probability_distribution_encoding_experiment_config import (
    DirectProbabilityDistributionEncodingExperimentConfig,
)
from rl_the_spire.datasets.direct_probability_distribution_dataset import (
    DirectProbabilityDistributionDataset,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../../conf/direct_probability_distribution_encoding",
    config_name="default",
    version_base=None,
)
def main(hydra_cfg: dict[Any, Any]) -> int:
    config: DirectProbabilityDistributionEncodingExperimentConfig = dacite.from_dict(
        data_class=DirectProbabilityDistributionEncodingExperimentConfig,
        data=hydra_cfg,
    )

    print("AAA")

    dataset = iter(DirectProbabilityDistributionDataset(config.dataset))

    logger.info(next(dataset))

    return 0


if __name__ == "__main__":
    main()
