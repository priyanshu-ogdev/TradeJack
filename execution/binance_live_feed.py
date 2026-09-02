"""
Live Binance market data feed for the paper-trading validation loop.

Why Binance's real public WebSocket instead of a third-party simulated-account
broker (e.g. RoboForex-style demo accounts): a demo broker account introduces its
own matching engine, its own (usually optimistic) fill assumptions, and an extra
network hop with its own latency profile that has nothing to do with the venue
you'd actually deploy to. Every discrepancy between "how the demo fills orders"
and "how Binance actually fills orders" is invisible until real money is on the
table. Reading Binance's real order book directly and simulating the wallet
ourselves (see paper_exchange.py) means the only thing being approximated is our
own execution model, not a third party's — and that's the one place we can
reason about and calibrate directly (real fee schedule, real depth, real latency
sampled from real elapsed wall-clock time on a live socket).

This module does NOT modify or duplicate data_forge/lob_collector.py — it reuses
`LocalOrderBook`, the exact Binance local-order-book reconstruction (REST snapshot
+ diff application + sequence-gap detection) that already lives there, so there is
one implementation of that protocol in the repo, not two. What's new here is a
*live callback* interface: lob_collector.py's LOBCollector only persists snapshots
to Parquet for later batch replay — it has no hook for an active loop to react to
each update as it arrives, which is what a trading loop needs.

NETWORK NOTE: this file was authored and syntax/logic-checked in a sandboxed
environment that cannot reach api.binance.com / stream.binance.com. It has not
been live-connection-tested from here. Run it in an environment with outbound
network access before trusting it; a `SyntheticReplayFeed` is included below so
the rest of the pipeline (features, paper exchange, risk guardian) can be
exercised end-to-end without real network access, for logic verification only —
never treat its output as a performance estimate.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import json
import random
import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (LiveFeed) %(message)s")
logger = logging.getLogger("LiveFeed")

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Reuse, don't duplicate: the exact same local-order-book class data_forge uses.
from data_forge.lob_collector import LocalOrderBook
from data_forge.config import config as forge_config

UpdateCallback = Callable[[str, Dict[str, Any]], Any]  # (kind: "depth"|"trade", payload) -> None | awaitable


class BinanceLiveDepthFeed:
    """
    Real-time Binance depth + trade stream with a per-update callback, for a live
    trading loop (as opposed to LOBCollector, which is a persist-to-disk collector).
    """

    def __init__(
        self,
        symbol: str = "BTC-USDT",
        depth_levels: int = 20,
        on_update: Optional[UpdateCallback] = None,
        max_resyncs_before_giveup: int = 20,
    ):
        self.symbol = symbol
        self.binance_symbol = symbol.replace("-", "").lower()
        self.depth_levels = depth_levels
        self.on_update = on_update
        self.order_book = LocalOrderBook(depth_levels=depth_levels)
        self.resync_count = 0
        self.max_resyncs_before_giveup = max_resyncs_before_giveup
        self.last_update_wall_time: float = 0.0
        self.total_depth_updates = 0
        self.total_trades = 0
        self._stop = False

    async def _fetch_rest_snapshot(self) -> Optional[dict]:
        if not AIOHTTP_AVAILABLE:
            return None
        sym = self.symbol.replace("-", "").upper()
        url = f"{forge_config.binance_rest_url}/api/v3/depth?symbol={sym}&limit=1000"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    logger.error(f"REST snapshot failed: HTTP {resp.status}")
        except Exception as e:
            logger.error(f"REST snapshot error: {e}")
        return None

    async def _resync(self) -> bool:
        self.resync_count += 1
        if self.resync_count > self.max_resyncs_before_giveup:
            logger.error(f"Exceeded {self.max_resyncs_before_giveup} resyncs. Giving up — check connectivity.")
            return False
        logger.warning(f"RESYNC #{self.resync_count}: re-fetching REST snapshot...")
        self.order_book.initialized = False
        snapshot = await self._fetch_rest_snapshot()
        if snapshot:
            self.order_book.apply_snapshot(snapshot)
            return True
        await asyncio.sleep(5)
        return True

    async def _dispatch(self, kind: str, payload: Dict[str, Any]):
        if self.on_update is None:
            return
        result = self.on_update(kind, payload)
        if inspect.isawaitable(result):
            await result

    async def run(self, duration_sec: Optional[float] = None):
        """
        Connects to Binance's combined depth@100ms + aggTrade stream and dispatches
        every update to `on_update` as it arrives. Runs until `duration_sec` elapses
        or `stop()` is called.
        """
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is required for the live feed (pip install aiohttp).")

        snapshot = await self._fetch_rest_snapshot()
        if not snapshot:
            raise RuntimeError("Could not fetch initial REST order book snapshot — check network/connectivity.")
        self.order_book.apply_snapshot(snapshot)
        self.last_update_wall_time = time.time()

        streams = f"{self.binance_symbol}@depth@100ms/{self.binance_symbol}@aggTrade"
        ws_base = forge_config.binance_ws_url.replace("/ws", "")  # e.g. wss://stream.binance.com:9443
        ws_url = f"{ws_base}/stream?streams={streams}"
        logger.info(f"Connecting to combined stream: {ws_url}")

        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            while not self._stop:
                try:
                    async with session.ws_connect(ws_url, heartbeat=20) as ws:
                        logger.info(f"WebSocket connected. Streaming live {self.symbol} depth + trades...")
                        async for msg in ws:
                            if self._stop:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                envelope = json.loads(msg.data)
                                payload = envelope.get("data", envelope)

                                if "b" in payload and "a" in payload and "u" in payload:
                                    ok = self.order_book.apply_diff(payload)
                                    if not ok:
                                        if not await self._resync():
                                            return
                                        continue
                                    self.last_update_wall_time = time.time()
                                    if self.order_book.initialized:
                                        self.total_depth_updates += 1
                                        snap = self.order_book.get_depth_snapshot(levels=self.depth_levels)
                                        await self._dispatch("depth", snap)

                                elif payload.get("e") == "aggTrade":
                                    self.total_trades += 1
                                    trade = {
                                        "price": float(payload["p"]),
                                        "qty": float(payload["q"]),
                                        "is_buyer_maker": bool(payload["m"]),
                                        "timestamp": float(payload.get("T", time.time() * 1000)) / 1000.0,
                                    }
                                    await self._dispatch("trade", trade)

                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                logger.warning(f"WebSocket closed/error: {msg.data}. Reconnecting in 2s...")
                                break

                            if duration_sec and (time.time() - start_time) >= duration_sec:
                                logger.info(f"Duration {duration_sec}s reached. Stopping.")
                                self._stop = True
                                break
                except Exception as e:
                    logger.error(f"WebSocket error: {e}. Reconnecting in 2s...")

                if not self._stop and (not duration_sec or (time.time() - start_time) < duration_sec):
                    await asyncio.sleep(2)

        logger.info(
            f"Live feed stopped. Depth updates: {self.total_depth_updates}, "
            f"Trades: {self.total_trades}, Resyncs: {self.resync_count}"
        )

    def stop(self):
        self._stop = True

    def is_stale(self, max_age_sec: float = 2.0) -> bool:
        """True if we haven't received a depth update recently — the risk guardian
        should refuse to trade on a stale book (this is what a real deployment would
        need to avoid trading blind after a silent disconnect)."""
        return (time.time() - self.last_update_wall_time) > max_age_sec


class SyntheticReplayFeed:
    """
    NOT real market data. A local random-walk order book generator used only to
    exercise paper_exchange.py and risk_guardian.py end-to-end when there is no
    outbound network access (as in the sandbox this was authored in). Never use
    this to estimate performance — it has no relationship to real market dynamics.
    """

    def __init__(self, symbol: str = "BTC-USDT", start_price: float = 60000.0, depth_levels: int = 20, seed: int = 0):
        self.symbol = symbol
        self.price = start_price
        self.depth_levels = depth_levels
        self.rng = random.Random(seed)
        self.last_update_wall_time = time.time()
        self._stop = False

    def _make_snapshot(self) -> Dict[str, Any]:
        self.price *= (1.0 + self.rng.gauss(0, 0.0003))
        spread = max(0.5, self.price * 0.00005)
        snap: Dict[str, Any] = {"timestamp": time.time()}
        for i in range(self.depth_levels):
            bid_px = self.price - spread / 2 - i * spread * 0.5
            ask_px = self.price + spread / 2 + i * spread * 0.5
            snap[f"bid_px_{i}"] = round(bid_px, 2)
            snap[f"bid_sz_{i}"] = round(abs(self.rng.gauss(0.5, 0.3)) + 0.01, 5)
            snap[f"ask_px_{i}"] = round(ask_px, 2)
            snap[f"ask_sz_{i}"] = round(abs(self.rng.gauss(0.5, 0.3)) + 0.01, 5)
        return snap

    def is_stale(self, max_age_sec: float = 2.0) -> bool:
        return (time.time() - self.last_update_wall_time) > max_age_sec

    async def run(self, on_update: UpdateCallback, duration_sec: Optional[float] = None, tick_hz: float = 10.0):
        start = time.time()
        while not self._stop:
            snap = self._make_snapshot()
            self.last_update_wall_time = time.time()
            result = on_update("depth", snap)
            if inspect.isawaitable(result):
                await result
            if self.rng.random() < 0.3:
                is_sell = self.rng.random() < 0.5
                trade = {
                    "price": snap["bid_px_0"] if is_sell else snap["ask_px_0"],
                    "qty": abs(self.rng.gauss(0.05, 0.05)) + 0.001,
                    "is_buyer_maker": is_sell,
                    "timestamp": time.time(),
                }
                result = on_update("trade", trade)
                if inspect.isawaitable(result):
                    await result
            if duration_sec and (time.time() - start) >= duration_sec:
                break
            await asyncio.sleep(1.0 / tick_hz)

    def stop(self):
        self._stop = True


if __name__ == "__main__":
    logger.info("Smoke-testing SyntheticReplayFeed (no network required)...")

    def _print_update(kind, payload):
        if kind == "depth":
            logger.info(f"depth: bid={payload['bid_px_0']} ask={payload['ask_px_0']}")
        else:
            logger.info(f"trade: {payload['qty']:.4f} @ {payload['price']:.2f} (buyer_maker={payload['is_buyer_maker']})")

    feed = SyntheticReplayFeed(seed=42)
    asyncio.run(feed.run(_print_update, duration_sec=1.0, tick_hz=5.0))
