# RLTheSpire

A neural network project for reinforcement learning experiments towards learning to play Slay the Spire.

## Installation

1. Clone the repository with submodules:
```bash
git clone https://github.com/yourusername/RLTheSpire.git
cd RLTheSpire
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package and its dependencies:
```bash
./build_lint_and_test.sh
```

# Permutations Group Neural Network

## Configuration

The project uses Hydra for configuration management. Configuration files are located in `rl_the_spire/conf/permutations_group/`.

### Available Configurations

- **default.yaml**: Default configuration that runs on CPU
- **default-h100.yaml**: Optimized configuration for H100 GPUs

Key differences in the H100 configuration:
- Uses CUDA instead of CPU
- Uses mixed precision (float16) for better performance
- Increased batch size and worker threads
- Optimized for H100 GPU architecture

## Running Experiments

To run an experiment with the default CPU configuration:

```bash
python -m rl_the_spire.experiments.permutations_group_experiment
```

To run with the H100 GPU configuration:

```bash
python -m rl_the_spire.experiments.permutations_group_experiment --config-name default-h100
```

You can also override specific configuration parameters:

```bash
python -m rl_the_spire.experiments.permutations_group_experiment encoder.device=cuda dataset.batch_size=512
```

## Monitoring

The experiment logs metrics to WandB (Weights & Biases). To disable WandB logging:

```bash
python -m rl_the_spire.experiments.permutations_group_experiment wandb_enabled=false
```

## Project Structure

- `rl_the_spire/`: Main package directory
  - `conf/`: Configuration files
  - `experiments/`: Experiment scripts
  - `models/`: Neural network models
- `external/`: External dependencies

## Requirements

This project requires:
- Python 3.8+
- PyTorch
- Hydra
- WandB (for logging)
- Other dependencies as specified in setup.py

## Development

For development, you can install the development requirements:

```bash
pip install -r dev-requirements.txt
```

Use the build and test script:

```bash
./build_lint_and_test.sh
``` 