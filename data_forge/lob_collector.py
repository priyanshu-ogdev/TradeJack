"""
Live L2 Order Book Collector — Binance WebSocket Depth Stream.
Captures real-time L2 depth snapshots (top 20 levels) for live forward-testing.

This is currently the ONLY source of real L2 order book depth in data_forge.
There is no free bulk *historical* L2 depth source available (Bybit's public
dump only has trade ticks, not depth — see bybit_ingest.py's docstring), so
historical LOB-depth training data must come from either: (a) accumulating
this collector's live output over time, or (b) synthetic_diffusion.py's
generated depth, which is clearly synthetic and should be weighted/labeled
as such during training, not treated as real market data.

Architecture:
  1. Initializes with a REST API snapshot to seed the local order book state
  2. Subscribes to the depth@100ms WebSocket stream for incremental diffs
  3. Maintains a local order book with lastUpdateId verification
  4. Writes micro-batches to daily Parquet partitions every flush_interval_sec
  5. Resync Guardian: auto-detects dropped packets and re-initializes from REST snapshot
  6. Integrates with the Warden heartbeat for live pipeline health monitoring

Output Format:
  data_store/live/{symbol}/YYYY/MM/DD/depth_HH.parquet
  data_store/live/{symbol}/YYYY/MM/DD/trades_HH.parquet  (added: see "Live trade
    capture" below — same partition scheme, same flush cadence, sibling file)

Live trade capture (Phase 2 of the upgrade plan — closing the live-learning loop):
Real trade prints WITH a genuine Binance aggressor tag (`is_buyer_maker`) already
flow through execution/binance_live_feed.py's combined `@depth@100ms/@aggTrade`
stream during live trading (see execution/streaming_features.py's on_trade(), which
consumes exactly this) -- they were simply never PERSISTED anywhere. This collector
now subscribes to the same combined stream and writes full-fidelity aggTrade records
(all fields: agg_trade_id, price, quantity, first_trade_id, last_trade_id,
transact_time, is_buyer_maker, is_best_match) to trades_HH.parquet, in exactly the
column schema data_forge/feature_engineering.py's TradeFlowPhysics._load_agg_trades()
already expects from a Binance-Vision bulk download. See
data_forge/live_trade_bridge.py for the module that turns a day of these files into
something TradeFlowPhysics.process_daily_file() can run over unmodified -- reusing
the existing, hardened batch bucketing/remainder-carryover logic rather than
reimplementing an approximation of it, which is what closes the actual gap named in
execution/streaming_features.py's own module docstring ("port TradeFlowPhysics's
bucket logic to a streaming form... as its own reviewed change to data_forge").
"""

import os
import json
import time
import asyncio
import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from collections import defaultdict

from data_forge.config import config

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (LOBCollector) %(message)s")
logger = logging.getLogger("LOBCollector")


class LocalOrderBook:
    """
    Maintains a local reconstruction of the Binance L2 order book.
    Implements the official Binance order book management protocol:
    https://binance-docs.github.io/apidocs/spot/en/#how-to-manage-a-local-order-book-correctly
    """

    def __init__(self, depth_levels: int = 20):
        self.depth_levels = depth_levels
        self.bids: Dict[float, float] = {}  # price → quantity
        self.asks: Dict[float, float] = {}  # price → quantity
        self.last_update_id: int = 0
        self.initialized: bool = False

    def apply_snapshot(self, snapshot: dict):
        """Initializes the order book from a REST API snapshot."""
        self.bids.clear()
        self.asks.clear()
        for price, qty in snapshot.get("bids", []):
            p, q = float(price), float(qty)
            if q > 0:
                self.bids[p] = q
        for price, qty in snapshot.get("asks", []):
            p, q = float(price), float(qty)
            if q > 0:
                self.asks[p] = q
        self.last_update_id = int(snapshot.get("lastUpdateId", 0))
        self.initialized = True
        logger.debug(f"Snapshot applied: {len(self.bids)} bids, {len(self.asks)} asks, lastUpdateId={self.last_update_id}")

    def apply_diff(self, diff: dict) -> bool:
        """
        Applies an incremental depth diff update.
        Returns False if the diff is stale/out of sequence (triggers resync).
        """
        first_update_id = diff.get("U", diff.get("u", 0))
        final_update_id = diff.get("u", 0)

        # Discard stale updates
        if final_update_id <= self.last_update_id:
            return True

        # Detect gap: if first_update_id > last_update_id + 1, we missed packets
        if self.initialized and first_update_id > self.last_update_id + 1:
            logger.warning(
                f"SEQUENCE GAP DETECTED: expected <={self.last_update_id + 1}, "
                f"got {first_update_id}. Triggering Resync Guardian."
            )
            return False

        # Apply bid updates
        for price, qty in diff.get("b", []):
            p, q = float(price), float(qty)
            if q == 0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = q

        # Apply ask updates
        for price, qty in diff.get("a", []):
            p, q = float(price), float(qty)
            if q == 0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = q

        self.last_update_id = final_update_id
        return True

    def get_depth_snapshot(self, levels: int = None) -> Dict[str, Any]:
        """Returns the current top-N depth snapshot in TradeJack LOB format."""
        levels = levels or self.depth_levels
        ts = time.time()

        sorted_bids = sorted(self.bids.items(), key=lambda x: -x[0])[:levels]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:levels]

        snapshot = {"timestamp": ts}
        for i in range(levels):
            if i < len(sorted_bids):
                snapshot[f"bid_px_{i}"] = sorted_bids[i][0]
                snapshot[f"bid_sz_{i}"] = sorted_bids[i][1]
            else:
                snapshot[f"bid_px_{i}"] = 0.0
                snapshot[f"bid_sz_{i}"] = 0.0

            if i < len(sorted_asks):
                snapshot[f"ask_px_{i}"] = sorted_asks[i][0]
                snapshot[f"ask_sz_{i}"] = sorted_asks[i][1]
            else:
                snapshot[f"ask_px_{i}"] = 0.0
                snapshot[f"ask_sz_{i}"] = 0.0

        return snapshot


class LOBCollector:
    """
    Real-time L2 Order Book Collector using Binance public WebSocket.
    Writes micro-batch ZSTD Parquet partitions for the KvikIO streamer.
    """

    def __init__(
        self,
        symbol: str = "BTC-USDT",
        flush_interval_sec: int = 60,
        depth_levels: int = 20,
    ):
        self.symbol = symbol
        self.binance_symbol = symbol.replace("-", "").upper().lower()
        self.flush_interval = flush_interval_sec
        self.depth_levels = depth_levels
        self.order_book = LocalOrderBook(depth_levels=depth_levels)
        self.buffer: List[Dict[str, Any]] = []
        self.trades_buffer: List[Dict[str, Any]] = []
        self.resync_count = 0
        self.total_snapshots = 0
        self.total_trades = 0

        self.live_dir = os.path.join(config.data_store_dir, "live", symbol)
        os.makedirs(self.live_dir, exist_ok=True)

        # Warden heartbeat integration
        self.state_dir = os.path.join(config.data_store_dir, "..", "state")
        os.makedirs(os.path.join(self.state_dir, "logs"), exist_ok=True)

    def _get_partition_path(self) -> str:
        """Returns the current hourly partition path: live/{symbol}/YYYY/MM/DD/depth_HH.parquet"""
        now = datetime.now(timezone.utc)
        part_dir = os.path.join(self.live_dir, f"{now.year:04d}", f"{now.month:02d}", f"{now.day:02d}")
        os.makedirs(part_dir, exist_ok=True)
        return os.path.join(part_dir, f"depth_{now.hour:02d}.parquet")

    def _get_trades_partition_path(self) -> str:
        """Same partition scheme as depth, sibling file: live/{symbol}/YYYY/MM/DD/trades_HH.parquet"""
        now = datetime.now(timezone.utc)
        part_dir = os.path.join(self.live_dir, f"{now.year:04d}", f"{now.month:02d}", f"{now.day:02d}")
        os.makedirs(part_dir, exist_ok=True)
        return os.path.join(part_dir, f"trades_{now.hour:02d}.parquet")

    def _flush_buffer(self):
        """Writes the buffered snapshots to a ZSTD Parquet partition using atomic rename."""
        if not self.buffer or not POLARS_AVAILABLE:
            return

        try:
            df = pl.DataFrame(self.buffer)
            out_path = self._get_partition_path()
            tmp_path = out_path + ".tmp"

            # If partition already exists, append by reading + concatenating
            if os.path.exists(out_path):
                existing = pl.read_parquet(out_path)
                df = pl.concat([existing, df])

            df.write_parquet(
                tmp_path,
                compression=config.compression_codec,
                compression_level=config.compression_level,
                row_group_size=config.row_group_size,
            )
            os.replace(tmp_path, out_path)
            logger.debug(f"Flushed {len(self.buffer)} snapshots to {out_path} (total: {len(df)} rows)")
            self.buffer.clear()
        except Exception as e:
            logger.error(f"Buffer flush failed: {e}")

    def _flush_trades_buffer(self):
        """
        Writes buffered aggTrade records to a ZSTD Parquet partition, using the exact
        column schema data_forge/feature_engineering.py's TradeFlowPhysics expects from
        a Binance-Vision bulk download (agg_trade_id, price, quantity, first_trade_id,
        last_trade_id, transact_time, is_buyer_maker, is_best_match). Mirrors
        _flush_buffer()'s atomic-rename/append-if-exists pattern exactly, kept as a
        separate method (rather than parameterizing _flush_buffer) because the two
        buffers flush on the same cadence but are logically distinct streams a reader
        should be able to reason about independently.
        """
        if not self.trades_buffer or not POLARS_AVAILABLE:
            return

        try:
            df = pl.DataFrame(self.trades_buffer)
            out_path = self._get_trades_partition_path()
            tmp_path = out_path + ".tmp"

            if os.path.exists(out_path):
                existing = pl.read_parquet(out_path)
                df = pl.concat([existing, df])

            df.write_parquet(
                tmp_path,
                compression=config.compression_codec,
                compression_level=config.compression_level,
                row_group_size=config.row_group_size,
            )
            os.replace(tmp_path, out_path)
            logger.debug(f"Flushed {len(self.trades_buffer)} trades to {out_path} (total: {len(df)} rows)")
            self.trades_buffer.clear()
        except Exception as e:
            logger.error(f"Trades buffer flush failed: {e}")

    def _log_resync_event(self, reason: str):
        """Logs resync events to the Warden quarantine log for auditing."""
        log_path = os.path.join(self.state_dir, "logs", "quarantine_events.jsonl")
        event = {
            "timestamp": time.time(),
            "file_path": f"lob_collector/{self.symbol}",
            "error": f"RESYNC_GUARDIAN: {reason} (resync #{self.resync_count})"
        }
        try:
            fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
            os.write(fd, (json.dumps(event) + "\n").encode("utf-8"))
            os.close(fd)
        except Exception:
            pass

    async def _fetch_rest_snapshot(self) -> Optional[dict]:
        """Fetches a full order book snapshot from the Binance REST API."""
        if not AIOHTTP_AVAILABLE:
            return None
        sym = self.symbol.replace("-", "").upper()
        url = f"{config.binance_rest_url}/api/v3/depth?symbol={sym}&limit=1000"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(f"REST snapshot fetched: {len(data.get('bids', []))} bids, {len(data.get('asks', []))} asks")
                        return data
                    else:
                        logger.error(f"REST snapshot failed: HTTP {resp.status}")
        except Exception as e:
            logger.error(f"REST snapshot error: {e}")
        return None

    async def _resync(self):
        """Resync Guardian: discards corrupted state and re-initializes from REST snapshot."""
        self.resync_count += 1
        logger.warning(f"RESYNC GUARDIAN ACTIVATED (#{self.resync_count}): Re-fetching REST snapshot...")
        self._log_resync_event(f"Sequence gap detected, resync #{self.resync_count}")

        self.order_book.initialized = False
        snapshot = await self._fetch_rest_snapshot()
        if snapshot:
            self.order_book.apply_snapshot(snapshot)
            logger.info(f"Resync complete. lastUpdateId={self.order_book.last_update_id}")
        else:
            logger.error("Resync failed: could not fetch REST snapshot. Retrying in 5s...")
            await asyncio.sleep(5)

    async def run(self, duration_sec: Optional[int] = None):
        """
        Main collection loop. Connects to Binance WebSocket and collects depth data.
        Runs indefinitely unless duration_sec is specified.
        """
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp required for WebSocket collection. Cannot start.")
            return

        # Official Binance local-order-book protocol:
        # https://binance-docs.github.io/apidocs/spot/en/#how-to-manage-a-local-order-book-correctly
        # Open the WebSocket connection FIRST, THEN fetch the REST snapshot. Incoming diff
        # frames queue in the OS/transport socket buffer while we await the REST call (we
        # haven't started `async for msg in ws` yet, so nothing is dropped); apply_diff's
        # existing gap check (first_update_id > last_update_id + 1) still catches any genuine
        # gap and triggers _resync(). Doing the REST fetch BEFORE opening the socket (the
        # original bug) guaranteed a gap on every single startup, because the socket handshake
        # alone takes 300-500ms during which the snapshot's lastUpdateId was already stale.
        #
        # Combined stream (depth + aggTrade), matching execution/binance_live_feed.py's exact
        # URL construction, for the trade-capture extension described in this module's
        # docstring. Combined-stream frames arrive wrapped in {"stream": ..., "data": {...}}
        # envelopes -- unlike the old single-stream URL, which delivered bare diff dicts.
        ws_base = config.binance_ws_url.replace("/ws", "")
        streams = f"{self.binance_symbol}@depth@100ms/{self.binance_symbol}@aggTrade"
        ws_url = f"{ws_base}/stream?streams={streams}"
        logger.info(f"Connecting to combined stream: {ws_url}")

        start_time = time.time()
        last_flush = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    logger.info(f"WebSocket connected. Buffering diffs and fetching REST snapshot for {self.symbol}...")

                    # Fetch the snapshot only after the socket is open and receiving.
                    snapshot = await self._fetch_rest_snapshot()
                    if not snapshot:
                        logger.error("Cannot initialize: REST snapshot unavailable.")
                        return
                    self.order_book.apply_snapshot(snapshot)
                    logger.info(f"WebSocket connected. Collecting L2 depth + trades for {self.symbol}...")

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            envelope = json.loads(msg.data)
                            # Combined-stream envelope unwrap -- see comment above. Falls back
                            # to treating the message as unwrapped if "data" is absent, so this
                            # doesn't silently break if Binance ever changes framing.
                            payload = envelope.get("data", envelope)
                            event_type = payload.get("e")

                            if event_type == "aggTrade":
                                self.trades_buffer.append({
                                    "agg_trade_id": int(payload["a"]),
                                    "price": float(payload["p"]),
                                    "quantity": float(payload["q"]),
                                    "first_trade_id": int(payload["f"]),
                                    "last_trade_id": int(payload["l"]),
                                    "transact_time": int(payload["T"]),
                                    "is_buyer_maker": bool(payload["m"]),
                                    # "M" (best-price-match) is a deprecated/legacy Binance
                                    # field not always present; default True rather than
                                    # crash the whole capture loop over an absent optional
                                    # field feature_engineering.py doesn't even use in its
                                    # bucketing math.
                                    "is_best_match": bool(payload.get("M", True)),
                                })
                                self.total_trades += 1

                            else:
                                # Depth diff (no "e" field on Binance's depth-diff payload).
                                if not self.order_book.apply_diff(payload):
                                    await self._resync()
                                    continue

                                if self.order_book.initialized:
                                    snap = self.order_book.get_depth_snapshot(levels=self.depth_levels)
                                    self.buffer.append(snap)
                                    self.total_snapshots += 1

                            # Periodic flush (both streams, same cadence).
                            now = time.time()
                            if now - last_flush >= self.flush_interval:
                                self._flush_buffer()
                                self._flush_trades_buffer()
                                last_flush = now

                            # Duration check
                            if duration_sec and (now - start_time) >= duration_sec:
                                logger.info(f"Collection duration ({duration_sec}s) reached. Stopping.")
                                break

                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            logger.warning(f"WebSocket closed/error: {msg.data}. Reconnecting...")
                            break

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Final flush (both streams)
            self._flush_buffer()
            self._flush_trades_buffer()
            logger.info(
                f"LOB Collector stopped. Total snapshots: {self.total_snapshots}, "
                f"Total trades: {self.total_trades}, Resyncs: {self.resync_count}"
            )


if __name__ == "__main__":
    logger.info("Starting LOB Collector (10-second test mode)...")
    collector = LOBCollector(symbol="BTC-USDT", flush_interval_sec=5, depth_levels=8)
    asyncio.run(collector.run(duration_sec=10))
