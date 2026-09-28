"""
MultiSleeveOrchestrator: runs an HFT/scalping sleeve and a position/swing
sleeve (or any list of SleeveConfigs) concurrently against ONE shared live
feed connection, with a portfolio-level risk cap on top of each sleeve's own
RiskGuardian.

HONEST FRAMING — read this before calling anything here "HFT":
Real HFT (the kind that competes with market makers on queue position and
rebate capture) requires colocation and sub-millisecond infrastructure that a
retail Binance public WebSocket + Python asyncio loop fundamentally cannot
provide — Binance's fastest public depth stream updates every 100ms, and a
realistic decision-to-order round trip over the public API is on the order of
tens to low hundreds of milliseconds, not microseconds. What this module
builds is a genuinely fast-reacting *scalping* sleeve for retail constraints —
seconds-to-minutes holding periods, reacting quickly to visible order-flow
signals — not a system that competes with colocated infrastructure. Calibrate
expectations (and position sizing) accordingly: the edge available to a
system like this, if any exists at all, is materially smaller than what a
colocated HFT firm can extract, and that's a structural fact about the
infrastructure, not something more engineering effort here fixes.

Why capital and ledgers are kept separate per sleeve rather than netted at a
shared position level: if both sleeves want to trade the same symbol and one
buys while the other sells, a naive single-position system would have them
fighting each other, and untangling "whose intent wins" adds real complexity
for limited benefit at this stage. Giving each sleeve its own capital slice
and its own PaperExchange/ledger (via SleeveConfig.account_id_offset) means
they simply can't collide — each manages its own qty against its own equity.
For a REAL multi-sleeve deployment on one exchange account, this same
argument applies to API keys: prefer separate sub-accounts / API keys per
sleeve over a shared-account netting layer, at least until there's a specific
reason netting is worth the added complexity and failure surface.
"""

import os
import time
import logging
import asyncio
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (MultiSleeve) %(message)s")
logger = logging.getLogger("MultiSleeve")

from execution.sleeve_config import SleeveConfig
from execution.shared_feed_hub import SharedLiveFeedHub
from execution.live_inference_server import LivePaperInferenceServer


class PortfolioRiskAggregator:
    """
    A second risk layer ON TOP OF each sleeve's own RiskGuardian. Each sleeve
    already can't exceed its own capital slice or its own max_position_fraction
    — this exists for the case that matters across sleeves specifically:
    total exposure across ALL sleeves combined, which no single sleeve's
    guardian can see on its own.
    """

    def __init__(self, sleeves: List["SleeveRuntime"], max_total_exposure_fraction: float = 0.8):
        self.sleeves = sleeves
        self.max_total_exposure_fraction = max_total_exposure_fraction
        self.halted = False

    def total_exposure_fraction(self) -> float:
        total_equity = sum(s.server.exchange.accounting.equity for s in self.sleeves)
        total_notional = sum(
            abs(s.server.exchange.position_qty) * (s.server.exchange.last_mid_price or 0.0)
            for s in self.sleeves
        )
        if total_equity <= 0:
            return 0.0
        return total_notional / total_equity

    def check_and_maybe_halt(self) -> bool:
        """Returns True if trading should proceed, False if halted. Writes each
        sleeve's own kill-switch file on halt, so every sleeve's own
        RiskGuardian (which already polls for that file) picks it up on its
        very next check — no separate halt-propagation mechanism needed."""
        if self.halted:
            return False
        exposure = self.total_exposure_fraction()
        if exposure > self.max_total_exposure_fraction:
            self.halted = True
            logger.critical(
                f"PORTFOLIO-LEVEL EXPOSURE BREACH: {exposure*100:.1f}% "
                f"(limit {self.max_total_exposure_fraction*100:.0f}%). Halting ALL sleeves."
            )
            for s in self.sleeves:
                try:
                    with open(s.server.risk.limits.kill_switch_path, "w") as f:
                        f.write(f"portfolio_exposure_breach at {time.time()}\n")
                except Exception as e:
                    logger.error(f"Could not write kill switch for sleeve '{s.config.name}': {e}")
            return False
        return True


class SleeveRuntime:
    """Binds one SleeveConfig to its running LivePaperInferenceServer."""

    def __init__(self, config: SleeveConfig, server: LivePaperInferenceServer):
        self.config = config
        self.server = server


class MultiSleeveOrchestrator:
    def __init__(
        self,
        symbol: str,
        sleeve_configs: List[SleeveConfig],
        total_capital: float,
        state_dir: str = "state",
        base_account_id: int = 900,
        use_synthetic_feed: bool = False,
        weights_paths: Optional[Dict[str, str]] = None,
        max_total_exposure_fraction: float = 0.8,
        portfolio_check_every_n_dispatches: int = 50,
    ):
        total_frac = sum(c.capital_fraction for c in sleeve_configs)
        if total_frac > 1.0 + 1e-9:
            raise ValueError(
                f"Sleeve capital_fractions sum to {total_frac:.3f} > 1.0 — "
                f"sleeves would be allocated more capital than exists."
            )

        self.symbol = symbol
        self.total_capital = total_capital
        self.hub = SharedLiveFeedHub(symbol=symbol, use_synthetic_feed=use_synthetic_feed)
        self.portfolio_check_every_n_dispatches = portfolio_check_every_n_dispatches

        weights_paths = weights_paths or {}
        self.sleeves: List[SleeveRuntime] = []
        for cfg in sleeve_configs:
            server = LivePaperInferenceServer(
                symbol=symbol,
                weights_path=weights_paths.get(cfg.name),
                model_name=cfg.model_name,
                initial_cash=total_capital * cfg.capital_fraction,
                seq_len=cfg.seq_len,
                use_synthetic_feed=use_synthetic_feed,
                state_dir=state_dir,
                account_id=base_account_id + cfg.account_id_offset,
                owns_feed=False,  # this orchestrator's hub owns the one real connection
                decision_interval_ticks=cfg.decision_interval_ticks,
                risk_limits_override=dict(
                    min_hold_ticks=cfg.min_hold_ticks,
                    max_position_fraction=cfg.max_position_fraction,
                    max_orders_per_minute=cfg.max_orders_per_minute,
                ),
            )
            self.hub.subscribe(server._on_update, name=cfg.name)
            self.sleeves.append(SleeveRuntime(cfg, server))
            logger.info(f"Sleeve configured — {cfg.describe()}")

        self.portfolio_risk = PortfolioRiskAggregator(self.sleeves, max_total_exposure_fraction)
        self._orig_dispatch = self.hub._dispatch
        self.hub._dispatch = self._dispatch_with_portfolio_check

    async def _dispatch_with_portfolio_check(self, kind: str, payload: Dict[str, Any]):
        await self._orig_dispatch(kind, payload)
        if self.hub.total_updates_dispatched % self.portfolio_check_every_n_dispatches == 0:
            self.portfolio_risk.check_and_maybe_halt()

    def get_combined_status(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "total_capital": self.total_capital,
            "total_exposure_fraction": self.portfolio_risk.total_exposure_fraction(),
            "portfolio_halted": self.portfolio_risk.halted,
            "sleeves": {s.config.name: s.server.get_status() for s in self.sleeves},
        }

    async def run(self, duration_sec: Optional[float] = None):
        logger.info(
            f"Starting multi-sleeve session: symbol={self.symbol} total_capital=${self.total_capital:.2f} "
            f"sleeves={[s.config.name for s in self.sleeves]}"
        )
        try:
            await self.hub.run(duration_sec=duration_sec)
        finally:
            for s in self.sleeves:
                s.server.exchange.close()
            status = self.get_combined_status()
            logger.info(f"Multi-sleeve session ended. Final status: {status}")


if __name__ == "__main__":
    print("Smoke-testing MultiSleeveOrchestrator end to end (synthetic feed, no network)...")
    from execution.sleeve_config import DEFAULT_HFT_SLEEVE, DEFAULT_POSITION_SLEEVE
    import dataclasses

    # Shrink the position sleeve's decision interval for a short smoke test —
    # the defaults assume real 100ms cadence and would never fire in a 6s test.
    fast_position_sleeve = dataclasses.replace(DEFAULT_POSITION_SLEEVE, decision_interval_ticks=10, min_hold_ticks=2)
    fast_hft_sleeve = dataclasses.replace(DEFAULT_HFT_SLEEVE, min_hold_ticks=2)

    async def _main():
        orch = MultiSleeveOrchestrator(
            symbol="BTC-USDT",
            sleeve_configs=[fast_hft_sleeve, fast_position_sleeve],
            total_capital=1000.0,
            state_dir="/tmp/tj_multisleeve_test",
            use_synthetic_feed=True,
        )
        await orch.run(duration_sec=6.0)
        print(orch.get_combined_status())

    asyncio.run(_main())
