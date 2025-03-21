import logging
from typing import Any

import dacite
import hydra

from rl_the_spire.conf.permutations_group.sequence.permutation_group_sequence_vae_config import (
    PermutationGroupSequenceVAEConfig,
)

# Configure logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../../../conf/permutations_group/sequence",
    config_name="default",
    version_base=None,
)
def main(hydra_cfg: dict[Any, Any]) -> int:
    # 1. Parse the Hydra config into our dataclasses
    config: PermutationGroupSequenceVAEConfig = dacite.from_dict(
        data_class=PermutationGroupSequenceVAEConfig,
        data=hydra_cfg,
    )

    logger.info("Starting experiment permutation group sequence VAE")
    logger.info(f"Config: {config}")

    return 0


if __name__ == "__main__":
    main()
