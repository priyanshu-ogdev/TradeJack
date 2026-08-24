# scripts/deploy_config.py - Capital-Agnostic Frozen Deployment Configuration
# Segment 4.2: Hard-coded deployment parameters. No runtime modification permitted.
# The Crucible evolves. The Deployment freezes.

from dataclasses import dataclass


@dataclass(frozen=True)
class DeploymentConfig:
    # Capital settings
    starting_capital: float = 10.0
    max_position_fraction: float = 0.5

    # Anti-HFT guard (100 ticks prevents HFT fee erosion at 10 USD)
    min_hold_ticks: int = 100

    # Model settings
    frozen_model_path: str = "state/deployed/weights_promoted.pt"
    model_name: str = "Dilated-CNN-Seq2seq"

    # Symbol and venue (spot only at < 100 USD capital)
    symbol: str = "BTC-USDT"
    venue: str = "binance_spot"

    # Risk limits
    max_daily_loss_pct: float = 0.05
    max_drawdown_halt: float = 0.15
    monitoring_out_of_band_days: int = 3

    # Safety: MUST always be False in deployment
    self_mod_enabled: bool = False
    auto_promotion: bool = False

    def __post_init__(self):
        assert not self.self_mod_enabled, "DeploymentConfig: self_mod_enabled must be False"
        assert not self.auto_promotion, "DeploymentConfig: auto_promotion must be False"
        assert 0 < self.max_position_fraction <= 1.0
        assert self.min_hold_ticks >= 10
        assert self.starting_capital > 0


DEPLOY_CONFIG = DeploymentConfig()

if __name__ == "__main__":
    import json, dataclasses
    print(json.dumps(dataclasses.asdict(DEPLOY_CONFIG), indent=2))
