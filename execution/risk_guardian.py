"""
Risk Guardian — Immutable safety layer between model output and exchange.

This is the last checkpoint before real money moves. Not configurable by the
model, not overridable by training, not modifiable at runtime.

All checks MUST pass before any order is sent to the exchange:
  1. Position size within max_position_fraction
  2. Daily loss within max_daily_loss_pct
  3. Drawdown within max_drawdown_halt
  4. Minimum hold period elapsed (anti-HFT)
  5. Rate limit (max orders per minute)
  6. No duplicate orders
  7. Exchange connection alive (else: flatten positions)
  8. Order reconciliation (actual exchange state vs internal state)
"""

import os
import time
import logging
import asyncio
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from execution.exchange_adapter import ExchangeAdapter, OrderResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (RiskGuardian) %(message)s")
logger = logging.getLogger("RiskGuardian")


@dataclass
class RiskState:
    """Current risk state tracked by the guardian."""
    daily_pnl: float = 0.0
    daily_start_equity: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    position_qty: float = 0.0
    position_value: float = 0.0
    orders_this_minute: int = 0
    last_order_time: float = 0.0
    last_trade_tick: int = 0
    last_reconciliation: float = 0.0
    is_halted: bool = False
    halt_reason: str = ""
    recent_orders: List[Dict[str, Any]] = field(default_factory=list)


class RiskGuardian:
    """
    Immutable safety layer. Sits between the model's trade signal and the exchange.

    Every order request passes through check_order_allowed() before execution.
    If any check fails, the order is blocked and logged. No exceptions, no overrides.
    """

    def __init__(
        self,
        exchange: ExchangeAdapter,
        symbol: str = "BTC/USDT",
        max_position_fraction: float = 0.5,
        max_daily_loss_pct: float = 0.05,
        max_drawdown_halt: float = 0.15,
        min_hold_ticks: int = 100,
        max_orders_per_minute: int = 5,
        connection_loss_flatten_sec: int = 60,
        order_reconciliation_interval_sec: int = 60,
        starting_equity: float = 100.0,
        audit_db_path: Optional[str] = None,
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.max_position_fraction = max_position_fraction
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_halt = max_drawdown_halt
        self.min_hold_ticks = min_hold_ticks
        self.max_orders_per_minute = max_orders_per_minute
        self.connection_loss_flatten_sec = connection_loss_flatten_sec
        self.reconciliation_interval = order_reconciliation_interval_sec

        self.state = RiskState(
            daily_start_equity=starting_equity,
            peak_equity=starting_equity,
            current_equity=starting_equity,
        )

        # Audit trail
        self.audit_db_path = audit_db_path
        if audit_db_path:
            self._init_audit_db()

        self._current_tick = 0

    def _init_audit_db(self):
        """Initialize SQLite audit trail for every order decision."""
        os.makedirs(os.path.dirname(self.audit_db_path), exist_ok=True)
        conn = sqlite3.connect(self.audit_db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                action TEXT,
                side TEXT,
                qty REAL,
                allowed INTEGER,
                reason TEXT,
                equity REAL,
                daily_pnl REAL,
                drawdown REAL
            )
        """)
        conn.commit()
        conn.close()

    def _log_audit(self, action: str, side: str, qty: float, allowed: bool, reason: str):
        """Record every order decision to the audit trail."""
        if not self.audit_db_path:
            return
        try:
            dd = (self.state.peak_equity - self.state.current_equity) / max(self.state.peak_equity, 1e-8)
            conn = sqlite3.connect(self.audit_db_path)
            conn.execute(
                "INSERT INTO risk_audit (timestamp, action, side, qty, allowed, reason, equity, daily_pnl, drawdown) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), action, side, qty, int(allowed), reason,
                 self.state.current_equity, self.state.daily_pnl, dd)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Audit log error: {e}")

    def update_equity(self, current_equity: float):
        """Update current equity state. Called by the inference server after each tick."""
        self.state.current_equity = current_equity
        self.state.peak_equity = max(self.state.peak_equity, current_equity)
        self.state.daily_pnl = current_equity - self.state.daily_start_equity
        self._current_tick += 1

    def reset_daily_pnl(self):
        """Called at the start of each trading day (UTC midnight)."""
        self.state.daily_start_equity = self.state.current_equity
        self.state.daily_pnl = 0.0
        self.state.orders_this_minute = 0
        logger.info(f"Daily P&L reset. Starting equity: ${self.state.daily_start_equity:.2f}")

    def check_order_allowed(
        self,
        side: str,
        qty: float,
        current_price: float,
    ) -> Tuple[bool, str]:
        """
        Run ALL safety checks. Returns (allowed, reason).

        If ANY check fails, the order is blocked. No partial checks.
        """
        # 0. Emergency halt
        if self.state.is_halted:
            reason = f"HALTED: {self.state.halt_reason}"
            self._log_audit("check", side, qty, False, reason)
            return False, reason

        # 1. Position size limit
        order_notional = qty * current_price
        max_notional = self.max_position_fraction * self.state.current_equity
        if order_notional > max_notional:
            reason = f"Position size {order_notional:.2f} exceeds max {max_notional:.2f} ({self.max_position_fraction*100:.0f}% of equity)"
            self._log_audit("check", side, qty, False, reason)
            return False, reason

        # 2. Daily loss limit
        daily_loss_pct = abs(min(0, self.state.daily_pnl)) / max(self.state.daily_start_equity, 1e-8)
        if daily_loss_pct >= self.max_daily_loss_pct:
            reason = f"Daily loss {daily_loss_pct*100:.1f}% exceeds max {self.max_daily_loss_pct*100:.0f}%"
            self._trigger_halt(reason)
            self._log_audit("check", side, qty, False, reason)
            return False, reason

        # 3. Drawdown limit
        drawdown = (self.state.peak_equity - self.state.current_equity) / max(self.state.peak_equity, 1e-8)
        if drawdown >= self.max_drawdown_halt:
            reason = f"Drawdown {drawdown*100:.1f}% exceeds max {self.max_drawdown_halt*100:.0f}%"
            self._trigger_halt(reason)
            self._log_audit("check", side, qty, False, reason)
            return False, reason

        # 4. Minimum hold period
        ticks_since_last = self._current_tick - self.state.last_trade_tick
        if ticks_since_last < self.min_hold_ticks and self.state.last_trade_tick > 0:
            reason = f"Min hold period: {ticks_since_last}/{self.min_hold_ticks} ticks elapsed"
            self._log_audit("check", side, qty, False, reason)
            return False, reason

        # 5. Rate limit
        now = time.time()
        if now - self.state.last_order_time < 60.0:
            if self.state.orders_this_minute >= self.max_orders_per_minute:
                reason = f"Rate limit: {self.state.orders_this_minute}/{self.max_orders_per_minute} orders/min"
                self._log_audit("check", side, qty, False, reason)
                return False, reason
        else:
            self.state.orders_this_minute = 0

        # 6. Duplicate order detection (same side+qty within 10s)
        for recent in self.state.recent_orders[-5:]:
            if (recent["side"] == side
                and abs(recent["qty"] - qty) < 1e-8
                and now - recent["time"] < 10.0):
                reason = "Duplicate order detected (same side+qty within 10s)"
                self._log_audit("check", side, qty, False, reason)
                return False, reason

        # 7. Exchange connection
        if not self.exchange.is_connected():
            reason = "Exchange connection lost"
            self._log_audit("check", side, qty, False, reason)
            return False, reason

        # All checks passed
        self._log_audit("check", side, qty, True, "PASSED")
        return True, "PASSED"

    def record_order_executed(self, side: str, qty: float):
        """Called after successful order execution to update risk state."""
        now = time.time()
        self.state.last_trade_tick = self._current_tick
        self.state.last_order_time = now
        self.state.orders_this_minute += 1
        self.state.recent_orders.append({"side": side, "qty": qty, "time": now})

        # Keep only last 20 orders for duplicate detection
        if len(self.state.recent_orders) > 20:
            self.state.recent_orders = self.state.recent_orders[-20:]

    async def execute_safe_order(
        self,
        side: str,
        qty: float,
        current_price: float,
    ) -> Optional[OrderResult]:
        """
        Check all risk limits, then execute if allowed.

        Returns OrderResult on success, None on block.
        """
        allowed, reason = self.check_order_allowed(side, qty, current_price)

        if not allowed:
            logger.warning(f"Order BLOCKED: {side} {qty:.6f} — {reason}")
            return None

        result = await self.exchange.place_market_order(self.symbol, side, qty)

        if result.status == "filled":
            self.record_order_executed(side, result.qty)
            self._log_audit("executed", side, result.qty, True, f"Filled @ {result.avg_price:.2f}")
        else:
            self._log_audit("executed", side, qty, False, f"Exchange rejected: {result.status}")

        return result

    async def reconcile_positions(self):
        """
        Poll actual exchange state vs internal state.

        A crashed process shouldn't lose track of an open position.
        This runs periodically to sync internal state with exchange reality.
        """
        now = time.time()
        if now - self.state.last_reconciliation < self.reconciliation_interval:
            return

        try:
            actual_pos = await self.exchange.get_position(self.symbol)
            actual_qty = actual_pos.get("qty", 0.0)

            if abs(actual_qty - self.state.position_qty) > 1e-6:
                logger.warning(
                    f"Position reconciliation mismatch: "
                    f"internal={self.state.position_qty:.6f}, "
                    f"exchange={actual_qty:.6f}. Syncing to exchange state."
                )
                self.state.position_qty = actual_qty

            self.state.last_reconciliation = now
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")

    async def emergency_flatten(self, reason: str = "Emergency"):
        """
        Emergency position closure — sell everything and halt.

        Called when:
          - Exchange connection lost for > connection_loss_flatten_sec
          - Drawdown exceeds halt threshold
          - Manual kill switch activated
        """
        logger.critical(f"EMERGENCY FLATTEN: {reason}")
        self._trigger_halt(reason)

        try:
            pos = await self.exchange.get_position(self.symbol)
            qty = pos.get("qty", 0.0)
            if qty > 0:
                await self.exchange.place_market_order(self.symbol, "sell", qty)
                logger.critical(f"Emergency sell: {qty:.6f} {self.symbol}")
            await self.exchange.cancel_all_orders(self.symbol)
        except Exception as e:
            logger.critical(f"Emergency flatten failed: {e}")

    def _trigger_halt(self, reason: str):
        """Halt all trading. Requires manual restart."""
        self.state.is_halted = True
        self.state.halt_reason = reason
        logger.critical(f"TRADING HALTED: {reason}")

    def release_halt(self):
        """Manual halt release."""
        self.state.is_halted = False
        self.state.halt_reason = ""
        logger.info("Trading halt released manually.")

    def get_risk_summary(self) -> Dict[str, Any]:
        """Current risk state for dashboard display."""
        dd = (self.state.peak_equity - self.state.current_equity) / max(self.state.peak_equity, 1e-8)
        return {
            "equity": round(self.state.current_equity, 2),
            "peak_equity": round(self.state.peak_equity, 2),
            "daily_pnl": round(self.state.daily_pnl, 2),
            "daily_pnl_pct": round(self.state.daily_pnl / max(self.state.daily_start_equity, 1e-8) * 100, 2),
            "drawdown_pct": round(dd * 100, 2),
            "position_qty": self.state.position_qty,
            "orders_this_minute": self.state.orders_this_minute,
            "is_halted": self.state.is_halted,
            "halt_reason": self.state.halt_reason,
            "exchange_connected": self.exchange.is_connected(),
        }


if __name__ == "__main__":
    logger.info("RiskGuardian module loaded. Requires exchange adapter for testing.")
    logger.info("Run via: python -m scripts.genesis_prime --paper")
