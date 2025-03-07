from setuptools import find_packages, setup
import os

__VERSION__ = "0.0.1"

setup(
    name="RLTheSpire",
    version=__VERSION__,
        packages=find_packages(where='.', include=['rl_the_spire', 'external.slaythetext']),
    package_dir={
        'rl_the_spire': 'rl_the_spire',
        'external.slaythetext': 'external/slaythetext',
    },
    install_requires=[
        "ansimarkup",
        "colorama",
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