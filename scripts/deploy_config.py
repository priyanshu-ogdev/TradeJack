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
    # testnet: BinanceSpotAdapter(testnet=True) / OandaAdapter(environment="practice")
    # live: BinanceSpotAdapter(testnet=False) / OandaAdapter(environment="live") — requires testnet evidence
    exchange_mode: str = "paper"

    # Which real broker to route non-paper orders through: "binance" (crypto
    # spot, via execution/exchange_adapter.py) or "oanda" (FX, via
    # execution/oanda_adapter.py). Only consulted when exchange_mode is
    # "testnet" or "live" -- paper mode never touches a real broker at all.
    # See oanda_adapter.py's module docstring for why OANDA rather than
    # MetaTrader: the official MT5 Python package is Windows-only, requiring
    # a Wine-hosted terminal on Linux; OANDA's v20 API is native REST/
    # streaming, matching this project's headless-Linux deployment model.
    broker: str = "binance"

    # Market-data feed used to drive decisions. Independent of exchange_mode
    # on purpose: exchange_mode controls where ORDERS go (paper wallet vs a
    # real exchange), this controls where PRICES come from. Defaults to real
    # market data (BinanceLiveDepthFeed) even in paper mode, so "paper
    # trading" evidence reflects actual market conditions rather than a
    # random walk. Set True only for pure logic/plumbing tests that
    # shouldn't depend on network access to Binance's public feed.
    use_synthetic_feed: bool = False

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

    # PHASE 2: a champion should not get promoted if the training/evaluation data
    # it was measured against is stale relative to current market conditions -- see
    # escrow/validation_airgap.py's data-freshness check. Reuses this same 3-day
    # default rather than inventing an unrelated number, since it's the same
    # underlying concept ("how long since we've genuinely seen this system's data
    # pipeline produce something current") as monitoring_out_of_band_days above.
    max_training_data_staleness_days: int = 3

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
        assert self.broker in ("binance", "oanda"), f"Invalid broker: {self.broker}"
        assert self.min_weeks_testnet_before_live >= 1


DEPLOY_CONFIG = DeploymentConfig()

if __name__ == "__main__":
    import json, dataclasses
    print(json.dumps(dataclasses.asdict(DEPLOY_CONFIG), indent=2))
