import dataclasses


@dataclasses.dataclass
class PermutationGroupEncoderConfig:
    n_embed: int
    n_heads: int
    n_layers: int
    attn_dropout: float
    resid_dropout: float
    dtype: str
    device: str
    init_std: float
    mlp_dropout: float
    ln_eps: float
    n_output_heads: int
    n_output_embed: int
    n_output_rows: int
    n_output_columns: int
    activation: str
    linear_size_multiplier: int
