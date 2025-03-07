from setuptools import find_packages, setup
import os

__VERSION__ = "0.0.1"

setup(
    name="RLTheSpire",
    version=__VERSION__,
    packages=["rl_the_spire"],
    install_requires=[
        "dacite",
        "einops",
        "hydra-core",
        "torch",
        "torchaudio",
        "torchvision",
        "tqdm",
        "types-tqdm",
        "wandb",
    ],
    include_package_data=True,
)