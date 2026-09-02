"""
Paper exchange: a synthetic wallet that fills orders against the REAL live order
book, not a formula and not a third-party demo account.

Why not a third-party simulated account (RoboForex and similar brokers offer
free "demo" accounts): those introduce a second matching engine and a second
network hop that behave nothing like the venue you'd actually deploy capital
to — their fill assumptions, spread, and latency are their product decisions,
not Binance's. Every gap between "how the demo filled it" and "how the real
venue would have filled it" is invisible until real money is on the line.

What this module does instead: it holds a synthetic cash/position ledger
(reusing physics.portfolio_tracker.PortfolioAccountingEngine rather than
reinventing equity/Sharpe/Sortino bookkeeping) and fills orders by walking the
actual live depth snapshot from binance_live_feed — consuming real visible
liquidity level by level, computing a real VWAP fill price, and refusing to
fabricate a fill beyond what the visible book could actually support. Three
real-world frictions are modeled explicitly:

  1. Latency: a decision made now is not filled now. We sleep out a sampled
     network+matching latency, THEN read whatever the live book has become by
     then. Because this runs against a genuinely live, continuously-updating
     book, that price movement during the latency window is real elapsed
     market activity, not a statistical slippage add-on.
  2. Fees: Binance's real taker/maker fee schedule (spot default tier — pass
     your actual VIP tier if different; do not assume favorable fees).
  3. Exchange filters: MIN_NOTIONAL / LOT_SIZE-style constraints. The defaults
     here are illustrative placeholders — fetch the live values from
     `GET /api/v3/exchangeInfo` for the target symbol before trusting them,
     they change per-symbol and Binance updates them periodically.

Nothing in this file places a real order. There is no exchange API key here.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import random
import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (PaperExchange) %(message)s")
logger = logging.getLogger("PaperExchange")

from physics.portfolio_tracker import PortfolioAccountingEngine


@dataclass
class ExchangeFilters:
    """Illustrative defaults — verify against GET /api/v3/exchangeInfo for the
    real symbol before relying on these for anything but logic testing."""
    min_notional: float = 10.0
    step_size: float = 0.00001
    tick_size: float = 0.01


@dataclass
class FillResult:
    requested_qty: float
    filled_qty: float
    avg_price: float
    fee_paid: float
    latency_sec: float
    levels_consumed: int
    fully_filled: bool
    rejected_reason: Optional[str] = None


def _walk_book(levels: List[Tuple[float, float]], target_qty: float) -> Tuple[float, float, int]:
    """
    Consumes visible depth level-by-level up to target_qty. Returns
    (avg_price, filled_qty, levels_consumed). Never fills beyond what the
    visible levels actually contain — if the book doesn't have enough depth,
    filled_qty < target_qty and the caller must treat this as a partial fill,
    exactly as a real exchange would report it.
    """
    remaining = target_qty
    notional = 0.0
    filled = 0.0
    levels_used = 0
    for price, size in levels:
        if remaining <= 1e-12 or price <= 0 or size <= 0:
            break
        take = min(remaining, size)
        notional += take * price
        filled += take
        remaining -= take
        levels_used += 1
    avg_price = (notional / filled) if filled > 1e-12 else 0.0
    return avg_price, filled, levels_used


class PaperExchange:
    """
    Synthetic wallet + realistic execution simulator driven by a live (or
    synthetic-replay, for testing) order book feed.
    """

    def __init__(
        self,
        symbol: str = "BTC-USDT",
        initial_cash: float = 10.0,
        taker_fee_bps: float = 10.0,       # 0.10% — Binance spot default retail taker fee
        maker_fee_bps: float = 10.0,       # 0.10% — default retail maker fee (no BNB/VIP discount assumed)
        filters: Optional[ExchangeFilters] = None,
        latency_mean_sec: float = 0.08,
        latency_std_sec: float = 0.03,
        state_dir: str = "state",
        account_id: int = 900,             # reserved id range for live-paper accounts, distinct from Crucible child_ids
        depth_levels_visible: int = 20,
    ):
        self.symbol = symbol
        self.taker_fee_bps = taker_fee_bps
        self.maker_fee_bps = maker_fee_bps
        self.filters = filters or ExchangeFilters()
        self.latency_mean_sec = latency_mean_sec
        self.latency_std_sec = latency_std_sec
        self.depth_levels_visible = depth_levels_visible
        self.account_id = account_id

        self.accounting = PortfolioAccountingEngine(
            child_id=account_id, state_dir=state_dir, initial_cash=initial_cash
        )
        self.initial_cash = initial_cash
        self.position_qty = 0.0
        self.last_mid_price: Optional[float] = None
        self._latest_snapshot: Optional[Dict[str, Any]] = None
        self._latest_snapshot_wall_time = 0.0

        self.trade_count = 0
        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0

        self._init_fill_ledger(state_dir, account_id)

    def _init_fill_ledger(self, state_dir: str, account_id: int):
        db_dir = os.path.join(os.path.abspath(state_dir), f"child_{account_id}")
        os.makedirs(db_dir, exist_ok=True)
        self.fill_db_path = os.path.join(db_dir, "fills.sqlite")
        conn = sqlite3.connect(self.fill_db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL, symbol TEXT, side TEXT,
                requested_qty REAL, filled_qty REAL, avg_price REAL, mid_at_decision REAL,
                fee_paid REAL, latency_sec REAL, levels_consumed INTEGER,
                fully_filled INTEGER, rejected_reason TEXT, resulting_cash REAL, resulting_equity REAL
            )
        """)
        conn.commit()
        conn.close()

    def _log_fill(self, side: str, result: FillResult, mid_at_decision: float):
        try:
            conn = sqlite3.connect(self.fill_db_path, timeout=5)
            conn.execute(
                """INSERT INTO fills
                (timestamp, symbol, side, requested_qty, filled_qty, avg_price, mid_at_decision,
                 fee_paid, latency_sec, levels_consumed, fully_filled, rejected_reason, resulting_cash, resulting_equity)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    time.time(), self.symbol, side, result.requested_qty, result.filled_qty,
                    result.avg_price, mid_at_decision, result.fee_paid, result.latency_sec,
                    result.levels_consumed, int(result.fully_filled), result.rejected_reason,
                    self.accounting.cash, self.accounting.equity,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Fill ledger write failed: {e}")

    def on_depth_update(self, snap: Dict[str, Any]):
        """Call this from your feed's on_update callback for every depth snapshot.
        Keeps the exchange's view of the book current for mark-to-market and fills."""
        self._latest_snapshot = snap
        self._latest_snapshot_wall_time = time.time()
        bid = snap.get("bid_px_0", 0.0)
        ask = snap.get("ask_px_0", 0.0)
        if bid > 0 and ask > 0:
            self.last_mid_price = (bid + ask) / 2.0
            self._mark_to_market(snap.get("timestamp", time.time()))

    def _mark_to_market(self, market_ts: float):
        if self.last_mid_price is None:
            return
        equity = self.accounting.cash + self.position_qty * self.last_mid_price
        self.accounting.record_step(self.accounting.cash, equity, self.accounting.ticks_active + 1, market_ts)

    def book_is_stale(self, max_age_sec: float = 2.0) -> bool:
        return (time.time() - self._latest_snapshot_wall_time) > max_age_sec

    async def submit_target_position(self, target_frac: float) -> FillResult:
        """
        Core entry point: given a target position as a fraction of equity in
        [-1, 1], computes the required order, sleeps out a sampled realistic
        latency against the REAL live book, then fills against whatever the
        book has become by the time the (simulated) order would have arrived.
        """
        if self._latest_snapshot is None or self.last_mid_price is None:
            return FillResult(0, 0, 0, 0, 0, 0, False, rejected_reason="no_book_data_yet")
        if self.book_is_stale():
            return FillResult(0, 0, 0, 0, 0, 0, False, rejected_reason="stale_book")

        mid_at_decision = self.last_mid_price
        equity = self.accounting.equity
        target_qty = (target_frac * equity) / mid_at_decision
        qty_delta = target_qty - self.position_qty
        side = "buy" if qty_delta > 0 else "sell"

        if abs(qty_delta * mid_at_decision) < self.filters.min_notional:
            result = FillResult(qty_delta, 0.0, 0.0, 0.0, 0.0, 0, False, rejected_reason="below_min_notional")
            self._log_fill(side, result, mid_at_decision)
            return result

        latency = max(0.0, random.gauss(self.latency_mean_sec, self.latency_std_sec))
        await asyncio.sleep(latency)

        if self._latest_snapshot is None or self.book_is_stale():
            result = FillResult(qty_delta, 0.0, 0.0, 0.0, latency, 0, False, rejected_reason="feed_disconnected_during_latency")
            self._log_fill(side, result, mid_at_decision)
            return result

        snap = self._latest_snapshot
        levels_key_prefix = "ask" if side == "buy" else "bid"
        levels = [
            (snap.get(f"{levels_key_prefix}_px_{i}", 0.0), snap.get(f"{levels_key_prefix}_sz_{i}", 0.0))
            for i in range(self.depth_levels_visible)
        ]

        step_qty = round(abs(qty_delta) / self.filters.step_size) * self.filters.step_size
        avg_price, filled_qty, levels_used = _walk_book(levels, step_qty)

        if filled_qty <= 1e-12:
            result = FillResult(qty_delta, 0.0, 0.0, 0.0, latency, 0, False, rejected_reason="no_visible_liquidity")
            self._log_fill(side, result, mid_at_decision)
            return result

        signed_filled = filled_qty if side == "buy" else -filled_qty
        notional = filled_qty * avg_price
        fee = notional * (self.taker_fee_bps / 10000.0)

        new_cash = self.accounting.cash - (signed_filled * avg_price) - fee
        self.position_qty += signed_filled
        new_equity = new_cash + self.position_qty * avg_price

        self.accounting.record_step(new_cash, new_equity, self.accounting.ticks_active + 1, snap.get("timestamp", time.time()))
        self.trade_count += 1
        self.total_fees_paid += fee
        self.total_slippage_cost += abs(avg_price - mid_at_decision) * filled_qty

        fully_filled = abs(filled_qty - step_qty) < (self.filters.step_size * 2)
        result = FillResult(
            requested_qty=qty_delta, filled_qty=signed_filled, avg_price=avg_price,
            fee_paid=fee, latency_sec=latency, levels_consumed=levels_used,
            fully_filled=fully_filled, rejected_reason=None if fully_filled else "insufficient_visible_depth",
        )
        self._log_fill(side, result, mid_at_decision)
        return result

    def close(self):
        self.accounting.close()


if __name__ == "__main__":
    import asyncio
    from execution.binance_live_feed import SyntheticReplayFeed  # synthetic only, see that module's warning

    print("Smoke-testing PaperExchange fill logic against SyntheticReplayFeed (not real data, logic check only)...")

    async def _main():
        exch = PaperExchange(symbol="BTC-USDT", initial_cash=10.0, state_dir="/tmp/tradejack_smoketest_state", account_id=999901)
        feed = SyntheticReplayFeed(seed=7)

        async def _on_update(kind, payload):
            if kind == "depth":
                exch.on_depth_update(payload)

        feed_task = asyncio.create_task(feed.run(_on_update, duration_sec=6.0, tick_hz=20.0))
        await asyncio.sleep(0.5)
        r1 = await exch.submit_target_position(0.5)
        print("buy attempt:", r1)
        await asyncio.sleep(1.0)
        r2 = await exch.submit_target_position(-0.5)
        print("sell attempt:", r2)
        await feed_task
        print(f"Final equity: {exch.accounting.equity:.4f}  Fees paid: {exch.total_fees_paid:.6f}  Trades: {exch.trade_count}")
        exch.close()

    asyncio.run(_main())
