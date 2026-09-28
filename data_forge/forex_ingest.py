"""
Dukascopy Free Historical FX Tick Ingest Engine.

Adds a forex (and CFD/metals) data leg to data_forge, alongside the crypto
sources (Binance Vision, Bybit, live Binance WS). Crypto and FX have very
different microstructure (24/7 continuous order books vs. FX's OTC dealer
network with real weekend gaps and no single consolidated tape), so an RL
agent that only ever sees crypto patterns will not transfer to FX, and vice
versa. This module exists to give it real FX examples to learn from.

Data Source: Dukascopy Bank's public historical tick data feed.
  https://datafeed.dukascopy.com/datafeed/{PAIR}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5
No API key or registration required. This is the same free source used by most
third-party "free forex tick data" tools (e.g. Dukascopy Historical Data Feed,
tick-data-suite, dukascopy-node). Coverage is deep (many pairs back to the
early 2000s) and the data is *quoted* bid/ask ticks from Dukascopy's own ECN,
not a consolidated cross-venue tape — treat it as "one broker's real prices",
which is the standard caveat for all free retail-accessible FX tick data.

Format: hourly `.bi5` files. Each is raw LZMA-compressed binary; empty hours
(e.g. most of the weekend) return a 0-byte body, which is a completely normal,
expected response — not a download failure.

Each decompressed record is a fixed 20-byte struct (big-endian):
    uint32  time_ms   — milliseconds since the start of that hour
    uint32  ask_raw   — ask price * point_value
    uint32  bid_raw   — bid price * point_value
    float32 ask_vol   — ask-side volume (in millions of the base currency)
    float32 bid_vol   — bid-side volume

`point_value` (the fixed-point scale factor) is pair-specific: 100000 for most
5-decimal pairs (EURUSD, GBPUSD, ...), 1000 for JPY-quoted pairs (USDJPY,
EURJPY, ...). Getting this wrong silently produces prices off by 100x, so it
is looked up per-pair rather than hardcoded to one value.
"""

import os
import struct
import lzma
import aiohttp
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

from data_forge.config import config
from data_forge.schema import ForexTickSchema

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (ForexIngest) %(message)s")
logger = logging.getLogger("ForexIngest")

# Point value (fixed-point scale) per pair. JPY-quoted pairs use 1000; almost
# everything else uses 100000. Extend this map if a pair prices out wrong.
_JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CHFJPY", "CADJPY", "NZDJPY"}


def _point_value(pair: str) -> int:
    return 1000 if pair.upper() in _JPY_PAIRS else 100000


def _decode_bi5(raw: bytes, pair: str, hour_start: datetime) -> List[dict]:
    """Decompresses and decodes a Dukascopy .bi5 hourly tick file into tick dicts."""
    if not raw:
        return []
    try:
        decompressed = lzma.decompress(raw)
    except lzma.LZMAError as e:
        logger.warning(f"LZMA decode failed for {pair} {hour_start.isoformat()}: {e}")
        return []

    record_size = 20  # 3x uint32 + 2x float32, big-endian
    n_records = len(decompressed) // record_size
    pv = _point_value(pair)
    ticks = []
    for i in range(n_records):
        chunk = decompressed[i * record_size:(i + 1) * record_size]
        time_ms, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(">IIIff", chunk)
        ticks.append({
            "timestamp": hour_start.timestamp() * 1000 + time_ms,  # epoch ms
            "bid": bid_raw / pv,
            "ask": ask_raw / pv,
            "bid_volume": float(bid_vol),
            "ask_volume": float(ask_vol),
        })
    return ticks


class DukascopyForexIngest:
    """
    Downloads Dukascopy's free public historical FX tick data (hourly .bi5 files)
    and converts a full day into a single ZSTD-L3 Parquet file, validated against
    ForexTickSchema.
    """

    def __init__(self):
        self.base_url = config.dukascopy_base_url
        self.store_dir = config.data_store_dir
        self._semaphore = asyncio.Semaphore(config.max_concurrent_downloads)

    def _get_target_path(self, pair: str, date: str) -> str:
        target_dir = os.path.join(self.store_dir, "raw", pair, "forex_ticks", date.replace("-", "/"))
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, f"{pair.upper()}-ticks-{date}.parquet")

    async def _download_hour(self, session: aiohttp.ClientSession, pair: str, hour_dt: datetime) -> bytes:
        # Dukascopy month component in the URL is zero-indexed (January = 00).
        url = (
            f"{self.base_url}{pair.upper()}/{hour_dt.year:04d}/{hour_dt.month - 1:02d}/"
            f"{hour_dt.day:02d}/{hour_dt.hour:02d}h_ticks.bi5"
        )
        for attempt in range(config.download_retry_max):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
                    elif response.status == 404:
                        # Normal for hours with zero ticks (weekends, illiquid pairs).
                        return b""
                    else:
                        logger.warning(f"HTTP {response.status} for {url} (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Download error for {url} (attempt {attempt + 1}): {e}")
            if attempt < config.download_retry_max - 1:
                await asyncio.sleep(2 ** attempt)
        return b""

    async def download_daily_ticks(self, pair: str, date: str) -> str:
        """Downloads all 24 hourly .bi5 files for one day, decodes, and writes one
        ZSTD Parquet file per day. Weekend days will legitimately produce an empty
        (or near-empty) file — that is expected FX market behavior, not a bug."""
        parquet_path = self._get_target_path(pair, date)
        if os.path.exists(parquet_path):
            logger.debug(f"Forex ticks Parquet already exists for {pair} {date}. Skipping.")
            return parquet_path

        day_start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        all_ticks: List[dict] = []

        async with self._semaphore:
            async with aiohttp.ClientSession() as session:
                for hour in range(24):
                    hour_dt = day_start.replace(hour=hour)
                    raw = await self._download_hour(session, pair, hour_dt)
                    all_ticks.extend(_decode_bi5(raw, pair, hour_dt))

        if not all_ticks:
            logger.info(f"No ticks for {pair} on {date} (likely a weekend/holiday). Skipping write.")
            return ""

        if not POLARS_AVAILABLE:
            logger.warning("Polars required to write forex ticks. Skipping write.")
            return ""

        df = pl.DataFrame(all_ticks).with_columns(
            pl.from_epoch("timestamp", time_unit="ms").alias("timestamp")
        ).sort("timestamp")

        try:
            ForexTickSchema.validate(df)
        except Exception as e:
            logger.error(f"Forex ticks failed schema validation for {pair} {date}: {e}")
            return ""

        df.write_parquet(
            parquet_path,
            compression=config.compression_codec,
            compression_level=config.compression_level,
            row_group_size=config.row_group_size,
        )
        logger.info(f"Ingested forex ticks {pair} {date} ({len(df)} ticks) -> ZSTD Parquet")
        return parquet_path

    async def ingest_range(self, pair: str, start_date: str, end_date: str) -> List[str]:
        """Ingests a date range of daily FX ticks. Note this is much heavier than the
        crypto ingest paths: 24 HTTP requests per day per pair, so keep ranges modest
        or expect this to take a while under the download semaphore."""
        import pandas as pd
        dates = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()
        results = []
        for date in dates:
            results.append(await self.download_daily_ticks(pair, date))
        successful = [r for r in results if r]
        logger.info(f"Forex ingest complete for {pair}. {len(successful)}/{len(dates)} days produced data.")
        return successful

    async def ingest_all_pairs(self, start_date: str, end_date: str) -> dict:
        """Ingests ticks for all configured default forex pairs."""
        results = {}
        for pair in config.default_forex_pairs:
            logger.info(f"Starting forex ingest for {pair}...")
            results[pair] = await self.ingest_range(pair, start_date, end_date)
        return results


if __name__ == "__main__":
    async def main():
        ingestor = DukascopyForexIngest()
        await ingestor.ingest_range("EURUSD", "2024-01-02", "2024-01-03")

    asyncio.run(main())
