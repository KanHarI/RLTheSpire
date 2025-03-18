import logging
from typing import Any

import dacite
import hydra
import torch

import wandb
from rl_the_spire.conf.direct_probability_distribution_encoding.direct_probability_distribution_encoding_experiment_config import (
    DirectProbabilityDistributionEncodingExperimentConfig,
)
from rl_the_spire.experiments.direct_probability_distribution_encoding.create_dataloader import (
    create_dataloader,
)
from rl_the_spire.experiments.direct_probability_distribution_encoding.create_models import (
    create_models,
)
from rl_the_spire.experiments.direct_probability_distribution_encoding.run_training_iteration import (
    TrainingIterationInput,
    run_training_iteration,
)
from rl_the_spire.utils.training_utils import lr_lambda

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
    logger.info("Starting experiment with config:")
    logger.info(config)

    logger.info("Creating dataloader...")
    dataloader = create_dataloader(config)

    logger.info("Creating models...")
    learned_models = create_models(config)

    logger.info("Creating optimizer...")
    optimizer_params = [
        param for model in learned_models for param in model.parameters()
    ]

    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=config.optimizer.lr,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.optimizer.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, config.optimizer.warmup_steps)
    )

    if config.wandb_enabled:
        wandb.init(
            project="direct-probability-distribution-encoding",
            name=config.experiment_name,
            config=config,
        )

    logger.info("Starting training loop...")
    step = 0
    while step < config.iterations:
        if step % config.eval_interval == 0:
            # Eval
            training_iteration_output = run_training_iteration(
                TrainingIterationInput(
                    learned_models=learned_models,
                    dataloader=dataloader,
                )
            )
            logger.info(f"Eval step {step} of {config.iterations}")
        # Training
        training_iteration_output = run_training_iteration(
            TrainingIterationInput(
                learned_models=learned_models,
                dataloader=dataloader,
            )
        )

        # Update model
        optimizer.zero_grad()
        # total_loss.backward()
        scheduler.step()
        optimizer.step()

        if step % config.log_interval == 0:
            # Log
            logger.info(f"Train step {step} of {config.iterations}")
        step += 1

    return 0


if __name__ == "__main__":
    main()
