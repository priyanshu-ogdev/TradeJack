# scripts/deploy_config.py - Capital-Agnostic Frozen Deployment Configuration
# Segment 4.2: Hard-coded deployment parameters. No runtime modification permitted.
# The Crucible evolves. The Deployment freezes.

from dataclasses import dataclass


@dataclass
class DeploymentConfig:
    # ── Capital Settings ──
    # $100 minimum for realistic fee economics (0.2% round-trip < 1% of activity)
    starting_capital: float = 100.0
    max_position_fraction: float = 0.5

    # Anti-HFT guard (100 ticks prevents HFT fee erosion)
    min_hold_ticks: int = 100

    # ── Model Settings ──
    frozen_model_path: str = "state/deployed/weights_promoted.zip"
    model_name: str = "PPO-DilatedCNN"

    # ── Exchange Settings ──
    symbol: str = "BTC-USDT"
    venue: str = "binance_spot"

    # Exchange mode: paper | testnet | live
    # paper: PaperExchangeAdapter (no real orders)
    # testnet: BinanceSpotAdapter(testnet=True)
    # live: BinanceSpotAdapter(testnet=False) — requires testnet evidence
    exchange_mode: str = "paper"

    # API keys loaded from env vars (never stored in code)
    api_key_env_var: str = "BINANCE_API_KEY"
    api_secret_env_var: str = "BINANCE_API_SECRET"

    # ── Risk Limits ──
    max_daily_loss_pct: float = 0.05
    max_drawdown_halt: float = 0.15
    max_orders_per_minute: int = 5
    max_position_duration_hours: float = 48.0
    connection_loss_flatten_sec: int = 60
    order_reconciliation_interval_sec: int = 60
    monitoring_out_of_band_days: int = 3

    # ── Position Throttle (replaces live burn) ──
    # Scales max_position_fraction based on rolling Sortino ratio
    position_throttle_enabled: bool = True
    throttle_sortino_full: float = 1.0    # Full position at this Sortino
    throttle_sortino_zero: float = 0.0    # Near-zero position at this Sortino
    throttle_min_fraction: float = 0.05   # Minimum position fraction (survival mode)

    # ── Testnet Evidence Gate ──
    # MUST have this many weeks of testnet evidence before --live is allowed
    min_weeks_testnet_before_live: int = 2

    # ── Safety: MUST always be False in deployment ──
    self_mod_enabled: bool = False
    auto_promotion: bool = False

    def __post_init__(self):
        assert not self.self_mod_enabled, "DeploymentConfig: self_mod_enabled must be False"
        assert not self.auto_promotion, "DeploymentConfig: auto_promotion must be False"
        assert 0 < self.max_position_fraction <= 1.0
        assert self.min_hold_ticks >= 10
        assert self.starting_capital > 0
        assert self.exchange_mode in ("paper", "testnet", "live"), \
            f"Invalid exchange_mode: {self.exchange_mode}"
        assert self.min_weeks_testnet_before_live >= 1


DEPLOY_CONFIG = DeploymentConfig()

if __name__ == "__main__":
    import json, dataclasses
    print(json.dumps(dataclasses.asdict(DEPLOY_CONFIG), indent=2))
