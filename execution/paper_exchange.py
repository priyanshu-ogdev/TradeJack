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
from execution.exchange_adapter import ExchangeAdapter, OrderResult, OrderBook


@dataclass
class ExchangeFilters:
    """Symbol exchange trading rules and filters (min_notional, step_size, tick_size)."""
    min_notional: float = 10.0
    step_size: float = 0.00001
    tick_size: float = 0.01

    @classmethod
    def from_market(cls, market_dict: Dict[str, Any]) -> "ExchangeFilters":
        """Builds ExchangeFilters from CCXT or Binance market dictionary."""
        limits = market_dict.get("limits", {})
        precision = market_dict.get("precision", {})
        min_notional = float(limits.get("cost", {}).get("min", 10.0) or 10.0)
        step_size = float(limits.get("amount", {}).get("min", 0.00001) or 0.00001)
        tick_size = float(limits.get("price", {}).get("min", 0.01) or 0.01)
        return cls(min_notional=min_notional, step_size=step_size, tick_size=tick_size)


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
        price_sample_min_interval_sec: float = 1.0,
    ):
        self.symbol = symbol
        self.taker_fee_bps = taker_fee_bps
        self.maker_fee_bps = maker_fee_bps
        self.filters = filters or ExchangeFilters()
        self.latency_mean_sec = latency_mean_sec
        self.latency_std_sec = latency_std_sec
        self.depth_levels_visible = depth_levels_visible
        self.account_id = account_id
        self.price_sample_min_interval_sec = price_sample_min_interval_sec
        self._last_price_sample_wall_time = 0.0

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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL, mid_price REAL
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
            self._maybe_record_price_sample(snap.get("timestamp", time.time()))

    def _maybe_record_price_sample(self, market_ts: float):
        """
        Separate, lightweight price-series table (not a change to
        PortfolioAccountingEngine's own ledger schema, which other consumers
        like session_report.py already depend on the exact shape of) --
        exists specifically so a dashboard can plot a real price line with
        entry/exit fill markers overlaid, instead of only isolated trade
        points with nothing connecting them. Throttled to at most one sample
        per `price_sample_min_interval_sec` (default 1s) so a long-running
        deployment's table doesn't grow unbounded at full tick resolution --
        equity/mark-to-market still updates every tick via _mark_to_market,
        only this separate chart-oriented sampling is throttled.
        """
        now = time.time()
        if (now - self._last_price_sample_wall_time) < self.price_sample_min_interval_sec:
            return
        self._last_price_sample_wall_time = now
        try:
            conn = sqlite3.connect(self.fill_db_path, timeout=5)
            conn.execute(
                "INSERT INTO price_samples (timestamp, mid_price) VALUES (?, ?)",
                (market_ts, self.last_mid_price),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Price sample write failed: {e}")

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

        BUG FOUND WHILE BUILDING execution/live_exchange_bridge.py, not by
        reading this file: target_frac's [-1, 1] range implicitly assumes
        short-selling is possible (target_frac=-1 meaning "100% short"). This
        is a SPOT exchange -- there is no shorting, `position_qty` can never
        go negative in reality. The real exchange adapter naturally rejects
        an order that would do this (insufficient balance to sell), which is
        exactly how this got caught -- but THIS class, being a local
        simulation with no such natural constraint, was silently allowing the
        model to "sell" more than it held, going net negative, simulating a
        short position that could never actually exist on a real spot
        account. That means every past paper-trading result that involved
        strongly negative target_frac values was optimistic in a way that
        would not survive contact with a real exchange. Clamped below to
        long-only: any target implying a short is capped at fully exiting
        the position (qty_delta down to -position_qty, never further).
        """
        if self._latest_snapshot is None or self.last_mid_price is None:
            return FillResult(0, 0, 0, 0, 0, 0, False, rejected_reason="no_book_data_yet")
        if self.book_is_stale():
            return FillResult(0, 0, 0, 0, 0, 0, False, rejected_reason="stale_book")

        mid_at_decision = self.last_mid_price
        equity = self.accounting.equity
        target_qty = (target_frac * equity) / mid_at_decision
        qty_delta = target_qty - self.position_qty

        clamped = False
        if self.position_qty + qty_delta < 0:
            qty_delta = -self.position_qty  # long-only: sell at most everything we hold, never short
            clamped = True

        side = "buy" if qty_delta > 0 else "sell"

        if abs(qty_delta * mid_at_decision) < self.filters.min_notional:
            reason = "below_min_notional" if not clamped else "below_min_notional_after_long_only_clamp"
            result = FillResult(qty_delta, 0.0, 0.0, 0.0, 0.0, 0, False, rejected_reason=reason)
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


class PaperExchangeAdapter(ExchangeAdapter):
    """
    Simulated exchange for paper trading conforming to ExchangeAdapter interface.
    Used by scripts/run_all_checks.py and tests/test_v3_execution.py.
    """

    def __init__(
        self,
        initial_balance_usdt: float = 100.0,
        taker_fee_bps: float = 10.0,
        simulated_kyles_lambda: float = 0.001,
    ):
        self.balances: Dict[str, float] = {"USDT": initial_balance_usdt}
        self.taker_fee_rate = taker_fee_bps / 10_000.0
        self.kyles_lambda = simulated_kyles_lambda
        self._connected = True
        self.order_history: List[OrderResult] = []
        self._order_counter = 0

        self._current_prices: Dict[str, float] = {
            "BTC/USDT": 60000.0,
            "BTC-USDT": 60000.0,
            "ETH/USDT": 3000.0,
            "ETH-USDT": 3000.0,
            "SOL/USDT": 150.0,
            "SOL-USDT": 150.0,
        }

    async def connect(self) -> None:
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def set_price(self, symbol: str, price: float):
        self._current_prices[symbol] = price
        norm_sym = symbol.replace("-", "/")
        self._current_prices[norm_sym] = price

    async def get_ticker(self, symbol: str) -> float:
        norm_sym = symbol.replace("-", "/")
        return self._current_prices.get(norm_sym, self._current_prices.get(symbol, 0.0))

    async def get_orderbook(self, symbol: str, depth: int = 5) -> OrderBook:
        mid = await self.get_ticker(symbol)
        if mid <= 0:
            return OrderBook(symbol=symbol, bids=[], asks=[], timestamp=time.time(), mid_price=0.0)

        spread = mid * 0.0001
        bids = [[mid - spread * (i + 1), 1.0 + i * 0.5] for i in range(depth)]
        asks = [[mid + spread * (i + 1), 1.0 + i * 0.5] for i in range(depth)]
        return OrderBook(symbol=symbol, bids=bids, asks=asks, timestamp=time.time(), mid_price=mid)

    async def get_balance(self, asset: str) -> float:
        return self.balances.get(asset, 0.0)

    async def get_all_balances(self) -> Dict[str, float]:
        return {k: v for k, v in self.balances.items() if v > 0}

    def _compute_fill_price(self, symbol: str, side: str, qty: float) -> float:
        norm_sym = symbol.replace("-", "/")
        mid = self._current_prices.get(norm_sym, self._current_prices.get(symbol, 0.0))
        if mid <= 0:
            return 0.0
        slippage = self.kyles_lambda * abs(qty)
        if side == "buy":
            return mid + slippage * mid
        else:
            return mid - slippage * mid

    async def place_market_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter:06d}"

        fill_price = self._compute_fill_price(symbol, side, qty)
        if fill_price <= 0:
            return OrderResult(
                order_id=order_id, symbol=symbol, side=side, qty=0.0,
                avg_price=0.0, cost=0.0, fee=0.0, timestamp=time.time(),
                status="rejected", raw={"error": f"No price for {symbol}"},
            )

        notional = qty * fill_price
        fee = notional * self.taker_fee_rate

        parts = symbol.replace("-", "/").split("/")
        base = parts[0] if len(parts) >= 2 else symbol
        quote = parts[1] if len(parts) >= 2 else "USDT"

        if side == "buy":
            total_cost = notional + fee
            if self.balances.get(quote, 0.0) < total_cost:
                return OrderResult(
                    order_id=order_id, symbol=symbol, side=side, qty=0.0,
                    avg_price=fill_price, cost=0.0, fee=0.0, timestamp=time.time(),
                    status="rejected", raw={"error": "Insufficient balance"},
                )
            self.balances[quote] = self.balances.get(quote, 0.0) - total_cost
            self.balances[base] = self.balances.get(base, 0.0) + qty

        elif side == "sell":
            if self.balances.get(base, 0.0) < qty:
                return OrderResult(
                    order_id=order_id, symbol=symbol, side=side, qty=0.0,
                    avg_price=fill_price, cost=0.0, fee=0.0, timestamp=time.time(),
                    status="rejected", raw={"error": "Insufficient balance"},
                )
            self.balances[base] = self.balances.get(base, 0.0) - qty
            self.balances[quote] = self.balances.get(quote, 0.0) + notional - fee

        result = OrderResult(
            order_id=order_id, symbol=symbol, side=side, qty=qty,
            avg_price=fill_price, cost=notional, fee=fee,
            timestamp=time.time(), status="filled",
        )
        self.order_history.append(result)
        return result

    async def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        return []

    async def cancel_all_orders(self, symbol: str) -> int:
        return 0

    async def get_position(self, symbol: str) -> Dict[str, float]:
        parts = symbol.replace("-", "/").split("/")
        base = parts[0] if len(parts) >= 2 else symbol
        qty = self.balances.get(base, 0.0)
        return {"qty": qty, "entry_price": 0.0}

    def get_equity(self, base_symbol: str = "BTC/USDT") -> float:
        total = self.balances.get("USDT", 0.0)
        for asset, qty in self.balances.items():
            if asset == "USDT" or qty <= 0:
                continue
            sym = f"{asset}/USDT"
            price = self._current_prices.get(sym, 0.0)
            total += qty * price
        return total

    def get_trade_summary(self) -> Dict[str, Any]:
        buys = [o for o in self.order_history if o.side == "buy" and o.status == "filled"]
        sells = [o for o in self.order_history if o.side == "sell" and o.status == "filled"]
        total_fees = sum(o.fee for o in self.order_history if o.status == "filled")
        return {
            "total_trades": len(self.order_history),
            "filled_buys": len(buys),
            "filled_sells": len(sells),
            "total_fees": round(total_fees, 6),
            "final_equity": round(self.get_equity(), 4),
            "balances": {k: round(v, 8) for k, v in self.balances.items() if v > 0},
        }


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
