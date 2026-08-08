"""
Bybit Free Historical L2 Order Book Ingest Engine.
Downloads free daily L2 depth snapshots from Bybit's public repository.
Provides years of multi-level order book data for training RL agents on
liquidity exhaustion, spoofing detection, and real LOB friction physics.

Data Source: https://public.bybit.com/orderbook/{SYMBOL}/
Format: GZIP CSV → ZSTD-L3 Parquet (direct conversion, no intermediate waste)

This module is the "Historical Training" half of the hybrid L2 strategy:
  - Historical: Bybit L2 dumps (this module)
  - Live: Binance WebSocket collector (lob_collector.py)
"""

import os
import io
import gzip
import aiohttp
import asyncio
import logging
import re
from datetime import datetime
from typing import List, Optional

from data_forge.config import config

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (BybitIngest) %(message)s")
logger = logging.getLogger("BybitIngest")

# Bybit orderbook CSV columns
BYBIT_OB_COLUMNS = ["timestamp", "side", "price", "size"]


class BybitL2Ingest:
    """
    Downloads Bybit's free public historical L2 order book data.
    Converts GZIP CSV dumps to ZSTD-L3 Parquet partitioned by date.
    Produces schema-compatible depth snapshots for the TradeJack LOB physics engine.
    """

    def __init__(self):
        self.base_url = config.bybit_history_base_url
        self.store_dir = config.data_store_dir
        self._semaphore = asyncio.Semaphore(config.max_concurrent_downloads)

    def _get_target_dir(self, symbol: str, date: str) -> str:
        target_dir = os.path.join(self.store_dir, "raw", symbol, "depth", date.replace("-", "/"))
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def _bybit_symbol(self, symbol: str) -> str:
        """Converts TradeJack symbol format to Bybit format (BTC-USDT → BTCUSDT)."""
        return symbol.replace("-", "").upper()

    async def _download_with_retry(self, session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
        for attempt in range(config.download_retry_max):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
                    elif response.status == 404:
                        logger.debug(f"Not found (404): {url}")
                        return None
                    else:
                        logger.warning(f"HTTP {response.status} for {url} (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Download error: {e} (attempt {attempt + 1})")
            if attempt < config.download_retry_max - 1:
                await asyncio.sleep(2 ** attempt)
        return None

    def _convert_orderbook_to_depth_parquet(self, raw_data: bytes, parquet_path: str, is_gzip: bool = True) -> bool:
        """
        Converts raw Bybit orderbook CSV/GZIP data into a depth-snapshot Parquet file.
        Pivots the side/price/size rows into bid_px_0..N / ask_px_0..N format
        compatible with the TradeJack LOB physics engine.
        """
        if not POLARS_AVAILABLE:
            logger.warning("Polars required for Bybit L2 conversion. Skipping.")
            return False

        try:
            if is_gzip:
                csv_bytes = gzip.decompress(raw_data)
            else:
                csv_bytes = raw_data

            df = pl.read_csv(
                io.BytesIO(csv_bytes),
                has_header=True,
                columns=BYBIT_OB_COLUMNS,
                dtypes={
                    "timestamp": pl.Float64,
                    "side": pl.Utf8,
                    "price": pl.Float64,
                    "size": pl.Float64,
                }
            )

            if df.is_empty():
                logger.warning(f"Empty orderbook data, skipping: {parquet_path}")
                return False

            # Separate bids and asks
            bids = df.filter(pl.col("side") == "Buy").sort(["timestamp", "price"], descending=[False, True])
            asks = df.filter(pl.col("side") == "Sell").sort(["timestamp", "price"], descending=[False, False])

            # Group by timestamp and extract top N levels
            depth_levels = 8  # Match TradeJack LOB env's 8-tier depth

            def _extract_levels(group_df: pl.DataFrame, prefix: str, num_levels: int) -> pl.DataFrame:
                """Extracts top N price/size levels per timestamp."""
                group_df = group_df.with_columns(
                    pl.col("price").rank("ordinal").over("timestamp").alias("level")
                )
                records = []
                for ts, ts_group in group_df.group_by("timestamp"):
                    row = {"timestamp": ts[0]}
                    ts_sorted = ts_group.sort("level")
                    for i in range(min(num_levels, len(ts_sorted))):
                        row[f"{prefix}_px_{i}"] = float(ts_sorted["price"][i])
                        row[f"{prefix}_sz_{i}"] = float(ts_sorted["size"][i])
                    # Pad missing levels with 0
                    for i in range(len(ts_sorted), num_levels):
                        row[f"{prefix}_px_{i}"] = 0.0
                        row[f"{prefix}_sz_{i}"] = 0.0
                    records.append(row)
                return pl.DataFrame(records) if records else pl.DataFrame()

            bid_levels = _extract_levels(bids, "bid", depth_levels)
            ask_levels = _extract_levels(asks, "ask", depth_levels)

            if bid_levels.is_empty() or ask_levels.is_empty():
                logger.warning(f"Insufficient bid/ask data for depth conversion: {parquet_path}")
                return False

            # Join bids and asks on timestamp
            depth_df = bid_levels.join(ask_levels, on="timestamp", how="inner")
            depth_df = depth_df.sort("timestamp")

            depth_df.write_parquet(
                parquet_path,
                compression=config.compression_codec,
                compression_level=config.compression_level,
                row_group_size=config.row_group_size,
            )
            logger.info(f"Bybit L2 → Parquet ({len(depth_df)} snapshots): {parquet_path}")
            return True

        except Exception as e:
            logger.error(f"Bybit L2 conversion failed: {e}")
            return False

    async def download_daily_depth(self, symbol: str, date: str) -> str:
        """
        Downloads a single day of L2 orderbook data from Bybit public repository.
        Bybit stores files as: https://public.bybit.com/orderbook/{SYMBOL}/{SYMBOL}{DATE}.csv.gz
        """
        target_dir = self._get_target_dir(symbol, date)
        bybit_sym = self._bybit_symbol(symbol)
        parquet_path = os.path.join(target_dir, f"{bybit_sym}-depth-{date}.parquet")

        if os.path.exists(parquet_path):
            logger.debug(f"Bybit depth Parquet exists for {symbol} {date}. Skipping.")
            return parquet_path

        # Bybit naming convention: BTCUSDT2024-01-01.csv.gz
        gz_filename = f"{bybit_sym}{date}.csv.gz"
        url = f"{self.base_url}orderbook/{bybit_sym}/{gz_filename}"

        async with self._semaphore:
            async with aiohttp.ClientSession() as session:
                content = await self._download_with_retry(session, url)
                if content is None:
                    return ""

                if self._convert_orderbook_to_depth_parquet(content, parquet_path, is_gzip=True):
                    return parquet_path
                return ""

    async def ingest_range(self, symbol: str, start_date: str, end_date: str) -> List[str]:
        """Ingests a date range of L2 depth data with rate-limited concurrency."""
        import pandas as pd
        dates = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()
        tasks = [self.download_daily_depth(symbol, date) for date in dates]
        results = await asyncio.gather(*tasks)
        successful = [r for r in results if r]
        logger.info(f"Bybit L2 ingest complete for {symbol}. {len(successful)}/{len(dates)} days.")
        return successful

    async def ingest_all_symbols(self, start_date: str, end_date: str) -> dict:
        """Ingests L2 depth for all configured default symbols."""
        results = {}
        for symbol in config.default_symbols:
            logger.info(f"Starting Bybit L2 depth ingest for {symbol}...")
            results[symbol] = await self.ingest_range(symbol, start_date, end_date)
        return results


if __name__ == "__main__":
    async def main():
        ingestor = BybitL2Ingest()
        await ingestor.ingest_range("BTC-USDT", "2024-01-01", "2024-01-03")

    asyncio.run(main())
