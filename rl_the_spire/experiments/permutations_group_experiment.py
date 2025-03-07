
from typing import Any
import hydra


@hydra.main(config_path="../conf/permutations_group", config_name="default", version_base=None)
def main(hydra_cfg: dict[Any, Any]) -> int:
    return 0


if __name__ == "__main__":
    main()
