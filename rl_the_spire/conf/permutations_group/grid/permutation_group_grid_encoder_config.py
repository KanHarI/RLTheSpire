import dataclasses


@dataclasses.dataclass
class PermutationGroupEncoderConfig:
    n_embed: int
    n_heads: int
    n_layers: int
    encoder_attn_dropout: float
    encoder_resid_dropout: float
    dtype: str
    device: str
    init_std: float
    encoder_mlp_dropout: float
    ln_eps: float
    n_output_heads: int
    n_output_embed: int
    n_output_rows: int
    n_output_columns: int
    activation: str
    linear_size_multiplier: int
    sigma_output: bool = True
    conv_blocks: int = 1
