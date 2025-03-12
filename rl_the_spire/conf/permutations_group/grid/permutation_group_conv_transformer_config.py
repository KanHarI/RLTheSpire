import dataclasses


@dataclasses.dataclass
class PermutationGroupConvTransformerConfig:
    n_heads: int
    denoiser_blocks: int
