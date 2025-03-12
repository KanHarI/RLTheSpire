import dataclasses


@dataclasses.dataclass
class VAEConfig:
    kl_loss_weight: float
    kl_warmup_steps: int
    kl_warmup_start_weight: float
    # Gamma parameters
    gamma_warmup_steps: int
    gamma_final: float = 0.5  # Final gamma value after warmup
    gamma_start: float = 5.0  # Default to high value (less noise initially)
