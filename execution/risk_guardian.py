"""
Risk guardian: the safety layer between a model's decision and the paper (later,
real) exchange. Every check here is deliberately hard-coded and dumb — it should
never require the model's cooperation to work, since the entire point is to
catch cases where the model, the feed, or the risk logic itself has a bug.

This runs in paper mode from day one (not bolted on only once real money
arrives) specifically so any bug in the guardian itself surfaces against fake
money first.
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (RiskGuardian) %(message)s")
logger = logging.getLogger("RiskGuardian")


@dataclass
class RiskLimits:
    max_position_fraction: float = 0.5      # mirrors scripts/deploy_config.py DeploymentConfig
    min_hold_ticks: int = 100
    max_daily_loss_pct: float = 0.05
    max_drawdown_halt: float = 0.15
    max_orders_per_minute: int = 6
    max_book_staleness_sec: float = 2.0
    kill_switch_path: str = "state/KILL_SWITCH"   # touch this file to halt trading immediately


@dataclass
class RiskState:
    is_halted: bool = False
    halt_reason: str = ""
    current_equity: float = 100.0
    daily_start_equity: float = 100.0
    peak_equity: float = 100.0
    orders_last_minute: int = 0


class RiskGuardian:
    """
    Risk guardian safety layer. Supports both:
    1. Direct per-tick check(target_frac, ...) used by LivePaperInferenceServer
    2. Order-by-order check_order_allowed(side, qty, price) used by ExchangeAdapter/tests
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        exchange: Optional[Any] = None,
        symbol: str = "BTC/USDT",
        max_position_fraction: Optional[float] = None,
        max_daily_loss_pct: Optional[float] = None,
        max_drawdown_halt: Optional[float] = None,
        min_hold_ticks: Optional[int] = None,
        max_orders_per_minute: Optional[int] = None,
        starting_equity: float = 100.0,
        **kwargs,
    ):
        if limits is None:
            limits = RiskLimits(
                max_position_fraction=max_position_fraction if max_position_fraction is not None else 0.5,
                max_daily_loss_pct=max_daily_loss_pct if max_daily_loss_pct is not None else 0.05,
                max_drawdown_halt=max_drawdown_halt if max_drawdown_halt is not None else 0.15,
                min_hold_ticks=min_hold_ticks if min_hold_ticks is not None else 100,
                max_orders_per_minute=max_orders_per_minute if max_orders_per_minute is not None else 6,
            )
        self.limits = limits
        self.exchange = exchange
        self.symbol = symbol
        self.starting_equity = starting_equity
        self.state = RiskState(
            current_equity=starting_equity,
            daily_start_equity=starting_equity,
            peak_equity=starting_equity,
        )
        self._order_timestamps: list = []
        self._last_flip_tick: int = -10**9
        self._current_tick: int = 0
        self._daily_start_equity: Optional[float] = starting_equity
        self._daily_start_time = time.time()
        self._halted = False
        self._halt_reason: Optional[str] = None

    def _reset_daily_if_needed(self, equity: float):
        if self._daily_start_equity is None:
            self._daily_start_equity = equity
            self._daily_start_time = time.time()
            return
        if (time.time() - self._daily_start_time) >= 86400:
            self._daily_start_equity = equity
            self._daily_start_time = time.time()

    def kill_switch_engaged(self) -> bool:
        return os.path.exists(self.limits.kill_switch_path)

    def update_equity(self, current_equity: float):
        """Update tracked equity and verify daily loss / drawdown thresholds."""
        self.state.current_equity = current_equity
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity
        self._reset_daily_if_needed(current_equity)

        if self._daily_start_equity and self._daily_start_equity > 0:
            loss_pct = (self._daily_start_equity - current_equity) / self._daily_start_equity
            if loss_pct >= self.limits.max_daily_loss_pct:
                self._halted = True
                self.state.is_halted = True
                self.state.halt_reason = f"Daily loss {loss_pct*100:.1f}% exceeded limit"
                self._halt_reason = "max_daily_loss_breached"
                logger.critical(f"RiskGuardian: Daily loss {loss_pct*100:.2f}% breached! Trading halted.")

        if self.state.peak_equity > 0:
            dd = (self.state.peak_equity - current_equity) / self.state.peak_equity
            if dd >= self.limits.max_drawdown_halt:
                self._halted = True
                self.state.is_halted = True
                self.state.halt_reason = f"Drawdown {dd*100:.1f}% exceeded limit"
                self._halt_reason = "max_drawdown_breached"
                logger.critical(f"RiskGuardian: Drawdown {dd*100:.2f}% breached! Trading halted.")

    def check_order_allowed(self, side: str, qty: float, price: float) -> tuple:
        """Compatibility check: verifies order against position size, rate limits, and halts."""
        if self._halted or self.state.is_halted:
            return False, f"Trading halted: {self.state.halt_reason or self._halt_reason}"

        if self.kill_switch_engaged():
            self._halted = True
            self.state.is_halted = True
            self.state.halt_reason = "kill_switch_engaged"
            return False, "KILL SWITCH ENGAGED"

        order_cost = abs(qty * price)
        max_cost = self.state.current_equity * self.limits.max_position_fraction
        if order_cost > max_cost + 1e-6:
            return False, f"Position size ${order_cost:.2f} exceeds limit ${max_cost:.2f}"

        now = time.time()
        self._order_timestamps = [t for t in self._order_timestamps if now - t < 60.0]
        if len(self._order_timestamps) >= self.limits.max_orders_per_minute:
            return False, f"Rate limit: {len(self._order_timestamps)} orders placed in the last 60s"

        return True, "PASSED"

    def record_order_executed(self, side: str, qty: float):
        """Record order execution timestamp for rate limiting."""
        self._order_timestamps.append(time.time())

    async def execute_safe_order(self, side: str, qty: float, price: float):
        """Pre-trade risk check followed by order execution on exchange if approved."""
        allowed, reason = self.check_order_allowed(side, qty, price)
        if not allowed:
            from execution.exchange_adapter import OrderResult
            return OrderResult(
                order_id="REJECTED-RISK",
                symbol=self.symbol,
                side=side,
                qty=0.0,
                avg_price=0.0,
                cost=0.0,
                fee=0.0,
                timestamp=time.time(),
                status="rejected",
                raw={"reason": reason},
            )
        if self.exchange is not None:
            from execution.exchange_adapter import ExchangeBannedError, OrderResult
            try:
                res = await self.exchange.place_market_order(self.symbol, side, qty)
                self.record_order_executed(side, qty)
                return res
            except ExchangeBannedError as e:
                # place_market_order raises this rather than returning a status
                # when already inside a known ban window (no request sent at
                # all) — surfacing it as a normal OrderResult here rather than
                # an uncaught exception, since a risk layer crashing is worse
                # than a risk layer reporting "banned" and letting the caller
                # decide how to halt.
                logger.critical(f"Exchange is banned, order not attempted: {e}")
                return OrderResult(
                    order_id="BANNED", symbol=self.symbol, side=side, qty=0.0, avg_price=0.0,
                    cost=0.0, fee=0.0, timestamp=time.time(), status="banned", raw={"error": str(e)},
                )
        return None

    def get_risk_summary(self) -> dict:
        """Return risk guardian status summary."""
        now = time.time()
        self._order_timestamps = [t for t in self._order_timestamps if now - t < 60.0]
        is_conn = True
        if self.exchange is not None and hasattr(self.exchange, "is_connected"):
            is_conn = self.exchange.is_connected()
        return {
            "equity": self.state.current_equity,
            "is_halted": self.state.is_halted or self._halted,
            "halt_reason": self.state.halt_reason or self._halt_reason,
            "exchange_connected": is_conn,
            "orders_last_minute": len(self._order_timestamps),
        }

    def check(
        self,
        target_frac: float,
        current_tick: int,
        equity: float,
        peak_equity: float,
        max_drawdown: float,
        book_is_stale: bool,
    ) -> tuple:
        """Returns (approved: bool, reason: Optional[str])."""
        if self._halted:
            return False, f"halted_previously:{self._halt_reason}"

        if self.kill_switch_engaged():
            self._halted = True
            self._halt_reason = "kill_switch_file_present"
            self.state.is_halted = True
            logger.critical(f"KILL SWITCH ENGAGED ({self.limits.kill_switch_path}). Halting all trading.")
            return False, self._halt_reason

        if book_is_stale:
            return False, "book_stale"

        if abs(target_frac) > self.limits.max_position_fraction:
            return False, "exceeds_max_position_fraction"

        if (current_tick - self._last_flip_tick) < self.limits.min_hold_ticks:
            return False, "min_hold_ticks_not_elapsed"

        now = time.time()
        self._order_timestamps = [t for t in self._order_timestamps if now - t < 60.0]
        if len(self._order_timestamps) >= self.limits.max_orders_per_minute:
            return False, "rate_limited"

        self._reset_daily_if_needed(equity)
        if self._daily_start_equity and self._daily_start_equity > 0:
            daily_pnl_pct = (equity - self._daily_start_equity) / self._daily_start_equity
            if daily_pnl_pct <= -self.limits.max_daily_loss_pct:
                self._halted = True
                self.state.is_halted = True
                self._halt_reason = "max_daily_loss_breached"
                logger.critical(
                    f"MAX DAILY LOSS BREACHED: {daily_pnl_pct*100:.2f}% "
                    f"(limit {-self.limits.max_daily_loss_pct*100:.1f}%). Halting."
                )
                return False, self._halt_reason

        if max_drawdown >= self.limits.max_drawdown_halt:
            self._halted = True
            self.state.is_halted = True
            self._halt_reason = "max_drawdown_breached"
            logger.critical(
                f"MAX DRAWDOWN BREACHED: {max_drawdown*100:.2f}% "
                f"(limit {self.limits.max_drawdown_halt*100:.1f}%). Halting."
            )
            return False, self._halt_reason

        return True, None

    def record_order_submitted(self, tick: int):
        self._order_timestamps.append(time.time())
        self._last_flip_tick = tick

    def reset_halt(self):
        """Manual override to resume after a halt — not called automatically."""
        logger.warning(f"Risk guardian halt manually reset (was: {self._halt_reason}).")
        self._halted = False
        self.state.is_halted = False
        self._halt_reason = None
        self.state.halt_reason = ""


if __name__ == "__main__":
    print("Smoke-testing RiskGuardian...")
    guard = RiskGuardian(RiskLimits(max_daily_loss_pct=0.05, max_drawdown_halt=0.15, min_hold_ticks=5, kill_switch_path="/tmp/nonexistent_kill_switch"))

    ok, reason = guard.check(0.9, current_tick=1, equity=10.0, peak_equity=10.0, max_drawdown=0.0, book_is_stale=False)
    print("over-limit position ->", ok, reason)
    assert not ok and reason == "exceeds_max_position_fraction"

    ok, reason = guard.check(0.4, current_tick=1, equity=10.0, peak_equity=10.0, max_drawdown=0.0, book_is_stale=False)
    print("valid order ->", ok, reason)
    assert ok
    guard.record_order_submitted(tick=1)

    ok, reason = guard.check(0.4, current_tick=2, equity=10.0, peak_equity=10.0, max_drawdown=0.0, book_is_stale=False)
    print("too soon (min_hold_ticks) ->", ok, reason)
    assert not ok and reason == "min_hold_ticks_not_elapsed"

    ok, reason = guard.check(0.4, current_tick=10, equity=9.4, peak_equity=10.0, max_drawdown=0.06, book_is_stale=False)
    print("5%% daily loss breach ->", ok, reason)
    assert not ok and reason == "max_daily_loss_breached"

    guard2 = RiskGuardian(RiskLimits(min_hold_ticks=0, kill_switch_path="/tmp/nonexistent_kill_switch"))
    ok, reason = guard2.check(0.1, current_tick=1, equity=10.0, peak_equity=10.0, max_drawdown=0.0, book_is_stale=True)
    print("stale book ->", ok, reason)
    assert not ok and reason == "book_stale"

    print("All RiskGuardian assertions passed.")
