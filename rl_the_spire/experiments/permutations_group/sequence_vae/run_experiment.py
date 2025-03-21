import logging
from typing import Any

import dacite
import hydra

from rl_the_spire.conf.permutations_group.sequence.permutation_group_sequence_vae_config import (
    PermutationGroupSequenceVAEConfig,
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

    logger.info(f"Config: {config}")

    return 0
