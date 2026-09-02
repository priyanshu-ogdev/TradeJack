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
from typing import Optional

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


class RiskGuardian:
    """
    Call `check(...)` before every order submission. If it returns
    (False, reason), do not submit the order — log the rejection and continue
    the loop; do not raise, since a halted trading loop should keep observing
    the market and reporting status, not crash.
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self._order_timestamps: list = []
        self._last_flip_tick: int = -10**9
        self._daily_start_equity: Optional[float] = None
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
                self._halt_reason = "max_daily_loss_breached"
                logger.critical(
                    f"MAX DAILY LOSS BREACHED: {daily_pnl_pct*100:.2f}% "
                    f"(limit {-self.limits.max_daily_loss_pct*100:.1f}%). Halting."
                )
                return False, self._halt_reason

        if max_drawdown >= self.limits.max_drawdown_halt:
            self._halted = True
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
        self._halt_reason = None


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
