"""
LiveExchangeBridge: the missing connection between BinanceSpotAdapter (real
order placement, fully built and unit-tested in isolation) and the actual
trading decision loop (LivePaperInferenceServer._maybe_act()).

THE GAP THIS CLOSES, found by tracing every construction site of
BinanceSpotAdapter across the repo: `LivePaperInferenceServer.__init__`
unconditionally constructed `PaperExchange`, regardless of any configured
exchange_mode. `BinanceSpotAdapter` was only ever constructed in the
dashboard's read-only balance display and its own self-tests -- zero
trading-loop call sites existed. Despite extensive, careful work on
BinanceSpotAdapter (Ed25519 keys, rate-limit tracking, filter validation),
the RL model's buy/sell/hold decisions never had a path to a real exchange
at all, under any configuration. This file is that path.

WHY A BRIDGE CLASS RATHER THAN REWIRING live_inference_server.py: PaperExchange
and BinanceSpotAdapter evolved with genuinely different interfaces --
PaperExchange exposes `submit_target_position(frac)`, `.accounting.equity`,
`.position_qty`, `on_depth_update()`, `book_is_stale()`; BinanceSpotAdapter
implements the ExchangeAdapter ABC (`place_market_order`, `get_balance`,
`get_position`). Rather than rewriting the trading loop to branch on which
interface it's talking to (real risk of subtly different behavior on the
paper path that's already been extensively tested), this class presents
PaperExchange's exact interface while delegating actual order placement to a
real BinanceSpotAdapter underneath -- LivePaperInferenceServer's code doesn't
change at all beyond which object gets constructed at __init__ time.

THE SYNC/ASYNC MISMATCH THIS HANDLES: PaperExchange can answer "what's my
equity right now" synchronously and instantly, because it's all local state.
A real exchange cannot -- knowing your actual balance requires a network
round trip. This bridge resolves that by keeping a locally-cached
equity/position estimate, updated immediately and synchronously from every
real fill's own response (no network wait needed for that part), with a
periodic background reconciliation against the exchange's real balance to
catch drift (external deposits, fills placed outside this process, rounding).
This is the standard pattern for exactly this mismatch, not a shortcut.

STATUS, STATED PRECISELY: this has been tested against a MOCK BinanceSpotAdapter
in this sandbox (no outbound network access to Binance's domains exists here).
It has NOT been run against a real exchange connection. Test on Binance
testnet, extensively, before real capital -- this is exactly the kind of
real-money-adjacent code this project's own testnet-gating philosophy
(scripts/deploy_config.py's min_weeks_testnet_before_live) exists for.
"""

import os
import time
import random
import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (LiveExchangeBridge) %(message)s")
logger = logging.getLogger("LiveExchangeBridge")

from execution.paper_exchange import FillResult  # exact same return shape _maybe_act() already expects


@dataclass
class _AccountingShim:
    """Mimics PortfolioAccountingEngine's fields that live_inference_server.py
    and risk_guardian.py read directly (equity, cash, peak_equity,
    max_drawdown) -- kept as locally-cached values updated on every fill and
    by periodic reconciliation, not fetched synchronously from the network
    on every read (which the calling code isn't written to await anyway)."""
    cash: float = 0.0
    equity: float = 0.0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0

    def update_equity(self, new_equity: float):
        self.equity = new_equity
        self.peak_equity = max(self.peak_equity, new_equity)
        if self.peak_equity > 0:
            self.max_drawdown = max(self.max_drawdown, (self.peak_equity - new_equity) / self.peak_equity)


class LiveExchangeBridge:
    """
    Presents PaperExchange's exact interface (on_depth_update, book_is_stale,
    submit_target_position, .accounting, .position_qty, .initial_cash,
    .trade_count, .total_fees_paid, .fill_db_path, .close) while placing REAL
    orders through a real BinanceSpotAdapter underneath.
    """

    def __init__(
        self,
        symbol: str,
        adapter: Any,  # BinanceSpotAdapter, typed loosely to avoid a hard import cycle
        state_dir: str = "state",
        account_id: int = 901,  # distinct default from PaperExchange's 900, so paper and live never collide on one state_dir
        reconcile_interval_sec: float = 30.0,
        latency_mean_sec: float = 0.08,
        latency_std_sec: float = 0.03,
    ):
        self.symbol = symbol
        self.adapter = adapter
        self.account_id = account_id
        self.reconcile_interval_sec = reconcile_interval_sec
        self.latency_mean_sec = latency_mean_sec
        self.latency_std_sec = latency_std_sec

        self.accounting = _AccountingShim()
        self.position_qty = 0.0
        self.initial_cash = 0.0  # filled in by the first reconciliation -- unknown until we actually ask the exchange
        self.last_mid_price: Optional[float] = None
        self._latest_snapshot_wall_time = 0.0
        self._initialized = False
        self._reconcile_task: Optional[asyncio.Task] = None

        self.trade_count = 0
        self.total_fees_paid = 0.0

        self._init_fill_ledger(state_dir, account_id)

    def _init_fill_ledger(self, state_dir: str, account_id: int):
        """Identical schema to PaperExchange's fills.sqlite (and the same
        price_samples table) specifically so dashboard/telemetry_server.py's
        /api/trades and /api/price-series work unmodified for live sessions
        too -- the dashboard shouldn't need to know or care whether a given
        account traded on paper or for real."""
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
        self._latest_snapshot_wall_time = time.time()
        bid = snap.get("bid_px_0", 0.0)
        ask = snap.get("ask_px_0", 0.0)
        if bid > 0 and ask > 0:
            self.last_mid_price = (bid + ask) / 2.0
        try:
            conn = sqlite3.connect(self.fill_db_path, timeout=5)
            conn.execute(
                "INSERT INTO price_samples (timestamp, mid_price) VALUES (?, ?)",
                (snap.get("timestamp", time.time()), self.last_mid_price),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

        if not self._initialized:
            self._initialized = True
            self._reconcile_task = asyncio.create_task(self._reconcile_loop())

    def book_is_stale(self, max_age_sec: float = 2.0) -> bool:
        return (time.time() - self._latest_snapshot_wall_time) > max_age_sec

    async def _reconcile_once(self):
        """Fetches REAL balance/position from the exchange and corrects the
        locally-cached accounting state. Runs once immediately on startup
        (so we don't trade blind against a stale initial_cash=0) and then
        periodically -- see the module docstring for why this exists instead
        of an equity property that awaits a network call, which the
        synchronous call sites in risk_guardian.py/live_inference_server.py
        aren't written to support."""
        try:
            if not self.adapter.is_connected():
                await self.adapter.connect()
            base_asset = self.symbol.split("/")[0] if "/" in self.symbol else self.symbol.replace("USDT", "").replace("-", "")
            quote_asset = self.symbol.split("/")[1] if "/" in self.symbol else "USDT"
            base_balance = await self.adapter.get_balance(base_asset)
            quote_balance = await self.adapter.get_balance(quote_asset)

            self.position_qty = base_balance
            self.accounting.cash = quote_balance
            equity = quote_balance + base_balance * (self.last_mid_price or 0.0)
            self.accounting.update_equity(equity)
            if self.initial_cash == 0.0:
                self.initial_cash = equity  # first reconciliation sets the baseline for reporting, same role PaperExchange's initial_cash param plays
            logger.info(f"Reconciled real balance: {base_asset}={base_balance:.8f} {quote_asset}={quote_balance:.4f} equity=${equity:.4f}")
        except Exception as e:
            logger.error(f"Balance reconciliation failed (will retry): {e}")

    async def _reconcile_loop(self):
        await self._reconcile_once()
        while True:
            await asyncio.sleep(self.reconcile_interval_sec)
            await self._reconcile_once()

    async def submit_target_position(self, target_frac: float) -> FillResult:
        """
        Same contract as PaperExchange.submit_target_position: given a
        target position as a fraction of equity in [-1, 1], places a REAL
        market order for the qty delta needed to reach it.
        """
        if self.last_mid_price is None:
            return FillResult(0, 0, 0, 0, 0, 0, False, rejected_reason="no_book_data_yet")
        if self.book_is_stale():
            return FillResult(0, 0, 0, 0, 0, 0, False, rejected_reason="stale_book")
        if self.accounting.equity <= 0:
            return FillResult(0, 0, 0, 0, 0, 0, False, rejected_reason="equity_not_yet_reconciled")

        mid_at_decision = self.last_mid_price
        target_qty = (target_frac * self.accounting.equity) / mid_at_decision
        qty_delta = target_qty - self.position_qty

        # Long-only clamp: same fix applied to paper_exchange.py's
        # submit_target_position and for the same reason -- this is a spot
        # exchange, there is no shorting. Real Binance would reject an order
        # to sell more than we hold (insufficient balance); clamping here
        # means the model's raw target gets sensibly reinterpreted as "exit
        # fully" rather than sent to the exchange and rejected outright.
        if self.position_qty + qty_delta < 0:
            qty_delta = -self.position_qty

        side = "buy" if qty_delta > 0 else "sell"

        latency = max(0.0, random.gauss(self.latency_mean_sec, self.latency_std_sec))
        await asyncio.sleep(latency)  # models the real decision-to-order network latency, same as PaperExchange does deliberately

        try:
            order_result = await self.adapter.place_market_order(self.symbol, side, abs(qty_delta))
        except Exception as e:
            logger.error(f"Real order placement raised unexpectedly: {e}")
            return FillResult(qty_delta, 0.0, 0.0, 0.0, latency, 0, False, rejected_reason=f"exception:{e}")

        if order_result.status != "filled":
            result = FillResult(
                requested_qty=qty_delta, filled_qty=0.0, avg_price=0.0, fee_paid=0.0,
                latency_sec=latency, levels_consumed=0, fully_filled=False,
                rejected_reason=order_result.status,  # "rejected" | "invalid" | "banned" | "rate_limited" -- see exchange_adapter.py
            )
            self._log_fill(side, result, mid_at_decision)
            return result

        signed_filled = order_result.qty if side == "buy" else -order_result.qty
        self.position_qty += signed_filled
        self.accounting.cash -= (signed_filled * order_result.avg_price) + order_result.fee
        new_equity = self.accounting.cash + self.position_qty * order_result.avg_price
        self.accounting.update_equity(new_equity)

        self.trade_count += 1
        self.total_fees_paid += order_result.fee

        result = FillResult(
            requested_qty=qty_delta, filled_qty=signed_filled, avg_price=order_result.avg_price,
            fee_paid=order_result.fee, latency_sec=latency, levels_consumed=1, fully_filled=True,
            rejected_reason=None,
        )
        self._log_fill(side, result, mid_at_decision)
        logger.warning(  # warning level, not info -- this is a REAL fill with real money, worth standing out in logs
            f"REAL FILL: {side} {order_result.qty:.8f} {self.symbol} @ {order_result.avg_price:.2f} "
            f"fee={order_result.fee:.6f} equity=${new_equity:.4f}"
        )
        return result

    def close(self):
        if self._reconcile_task is not None:
            self._reconcile_task.cancel()
        # Adapter's own connection close is async; scheduled best-effort since
        # this method itself is sync (matching PaperExchange.close()'s signature
        # that live_inference_server.py already calls without awaiting it).
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.adapter.close())
        except Exception:
            pass


if __name__ == "__main__":
    print("Smoke-testing LiveExchangeBridge against a MOCK adapter (no real network) ...")

    class MockOrderResult:
        def __init__(self, status, qty=0.0, avg_price=0.0, fee=0.0):
            self.status = status
            self.qty = qty
            self.avg_price = avg_price
            self.fee = fee

    class MockAdapter:
        def __init__(self):
            self._connected = False
            self.balances = {"BTC": 0.0, "USDT": 1000.0}

        def is_connected(self):
            return self._connected

        async def connect(self):
            self._connected = True

        async def get_balance(self, asset):
            return self.balances.get(asset, 0.0)

        async def place_market_order(self, symbol, side, qty):
            price = 60000.0
            fee = qty * price * 0.001
            if side == "buy":
                cost = qty * price + fee
                if cost > self.balances["USDT"]:
                    return MockOrderResult("rejected")
                self.balances["USDT"] -= cost
                self.balances["BTC"] += qty
            else:
                if qty > self.balances["BTC"]:
                    return MockOrderResult("rejected")
                self.balances["BTC"] -= qty
                self.balances["USDT"] += qty * price - fee
            return MockOrderResult("filled", qty=qty, avg_price=price, fee=fee)

        async def close(self):
            pass

    async def main():
        adapter = MockAdapter()
        bridge = LiveExchangeBridge(symbol="BTC/USDT", adapter=adapter, state_dir="/tmp/tj_bridge_test", account_id=999)

        bridge.on_depth_update({"bid_px_0": 59999.0, "ask_px_0": 60001.0, "timestamp": time.time()})
        await asyncio.sleep(0.3)  # let the background reconcile task run once
        print(f"After initial reconcile: equity=${bridge.accounting.equity:.2f} initial_cash=${bridge.initial_cash:.2f}")
        assert bridge.accounting.equity == 1000.0

        result = await bridge.submit_target_position(0.5)
        print(f"Buy 50% target result: {result}")
        assert result.fully_filled
        assert bridge.position_qty > 0
        print(f"Position after buy: {bridge.position_qty:.6f} BTC, equity=${bridge.accounting.equity:.2f}, cash=${bridge.accounting.cash:.2f}")

        result2 = await bridge.submit_target_position(-0.5)
        print(f"Target -50% (long-only clamp should cap this at 'sell everything'): {result2}")
        assert result2.fully_filled
        assert abs(bridge.position_qty) < 1e-9, "long-only clamp should have brought position to ~0, not negative"
        print(f"Position after clamp: {bridge.position_qty:.10f} BTC (correctly clamped to flat, not short)")

        result3 = await bridge.submit_target_position(50.0)  # absurd target -> should get rejected by the mock adapter (insufficient balance)
        print(f"Absurd oversized target result: {result3.status if hasattr(result3,'status') else result3.rejected_reason}")
        assert result3.rejected_reason is not None

        print(f"Trade count: {bridge.trade_count}, total fees: {bridge.total_fees_paid:.6f}")
        bridge.close()
        print("\nLiveExchangeBridge smoke test passed (mock adapter only -- see module docstring on real-exchange testing status).")

    asyncio.run(main())
