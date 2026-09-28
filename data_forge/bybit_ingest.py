"""
Bybit Free Historical Trades Ingest Engine.

IMPORTANT -- read before changing the URL:
Bybit's public dump (public.bybit.com) does NOT publish free historical L2 order book
depth. The only free daily bulk dump is executed TRADE ticks, served under
`spot/{SYMBOL}/` with columns:
    timestamp, symbol, side, size, price, tickDirection, trdMatchID,
    grossValue, homeNotional, foreignNotional
An earlier version of this module pointed at a non-existent `orderbook/{SYMBOL}/`
path (404 on every request) and, worse, tried to synthesize fake L2 depth by
grouping trade ticks by timestamp and pivoting them into bid_px_N/ask_px_N
columns. Trade prints are not resting orders -- that synthesis does not produce
real order book depth and has been removed entirely.

What this module now does: downloads the REAL Bybit trade-tick dump and converts
it to ZSTD Parquet under `raw/{symbol}/bybit_trades/`, validated against
BybitTradeSchema. This is a second, independent trade-tick source (alongside
Binance Vision aggTrades) -- useful for cross-exchange trade-flow features and
liquidity/venue comparison, NOT a source of L2 depth.

For real, free L2 order book depth, the only source in this project is
lob_collector.py (live Binance WebSocket depth@100ms). There is currently no
free bulk *historical* L2 depth source wired into data_forge -- see
docs/DATA_FORGE.md, section "Known Data Gaps", for options if that's needed.

Data Source: https://public.bybit.com/spot/{SYMBOL}/{SYMBOL}_{DATE}.csv.gz
Format: GZIP CSV -> ZSTD-L3 Parquet (direct conversion, no intermediate waste)
"""

import os
import io
import gzip
import aiohttp
import asyncio
import logging
from typing import List, Optional

from data_forge.config import config
from data_forge.schema import BybitTradeSchema

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (BybitIngest) %(message)s")
logger = logging.getLogger("BybitIngest")

# Real Bybit public trade-tick CSV columns (public.bybit.com/spot/{SYMBOL}/)
# NOTE: column list carried over from the derivatives "trading/" dump's known
# schema. The spot dump's exact columns have not been empirically verified
# against a real download in this sandbox (no network access) -- confirm
# against one real file before trusting this list.
BYBIT_TRADE_COLUMNS = [
    "timestamp", "symbol", "side", "size", "price",
    "tickDirection", "trdMatchID", "grossValue", "homeNotional", "foreignNotional",
]


class BybitTradesIngest:
    """
    Downloads Bybit's free public historical trade-tick dump.
    Converts GZIP CSV dumps to ZSTD-L3 Parquet partitioned by date.
    This is a trade-flow source (comparable to Binance aggTrades), not L2 depth.
    """

    def __init__(self):
        self.base_url = config.bybit_history_base_url
        self.store_dir = config.data_store_dir
        self._semaphore = asyncio.Semaphore(config.max_concurrent_downloads)

    def _get_target_dir(self, symbol: str, date: str) -> str:
        target_dir = os.path.join(self.store_dir, "raw", symbol, "bybit_trades", date.replace("-", "/"))
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def _bybit_symbol(self, symbol: str) -> str:
        """Converts TradeJack symbol format to Bybit format (BTC-USDT -> BTCUSDT)."""
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

    def _convert_trades_to_parquet(self, raw_data: bytes, parquet_path: str) -> bool:
        """Converts raw Bybit trade-tick GZIP CSV into ZSTD Parquet. Validates the result
        against BybitTradeSchema before writing."""
        if not POLARS_AVAILABLE:
            logger.warning("Polars required for Bybit trades conversion. Skipping.")
            return False

        try:
            csv_bytes = gzip.decompress(raw_data)
            df = pl.read_csv(
                io.BytesIO(csv_bytes),
                has_header=True,
                columns=BYBIT_TRADE_COLUMNS,
                schema_overrides={
                    "timestamp": pl.Float64,
                    "symbol": pl.Utf8,
                    "side": pl.Utf8,
                    "size": pl.Float64,
                    "price": pl.Float64,
                    "tickDirection": pl.Utf8,
                    "trdMatchID": pl.Utf8,
                    "grossValue": pl.Float64,
                    "homeNotional": pl.Float64,
                    "foreignNotional": pl.Float64,
                },
            )

            if df.is_empty():
                logger.warning(f"Empty trades data, skipping: {parquet_path}")
                return False

            try:
                BybitTradeSchema.validate(df.select(["timestamp", "symbol", "side", "size", "price"]))
            except Exception as schema_err:
                logger.error(f"Bybit trades failed schema validation, skipping: {schema_err}")
                return False

            df = df.sort("timestamp")
            df.write_parquet(
                parquet_path,
                compression=config.compression_codec,
                compression_level=config.compression_level,
                row_group_size=config.row_group_size,
            )
            logger.info(f"Bybit trades -> Parquet ({len(df)} ticks): {parquet_path}")
            return True

        except Exception as e:
            logger.error(f"Bybit trades conversion failed: {e}")
            return False

    async def download_daily_trades(self, symbol: str, date: str) -> str:
        """
        Downloads a single day of trade-tick data from Bybit's real public dump.
        URL layout: https://public.bybit.com/spot/{SYMBOL}/{SYMBOL}_{DATE}.csv.gz
        """
        target_dir = self._get_target_dir(symbol, date)
        bybit_sym = self._bybit_symbol(symbol)
        parquet_path = os.path.join(target_dir, f"{bybit_sym}-trades-{date}.parquet")

        if os.path.exists(parquet_path):
            logger.debug(f"Bybit trades Parquet exists for {symbol} {date}. Skipping.")
            return parquet_path

        # BYBIT PATH/FILENAME FIX (verified against an independent Bybit
        # downloader's documented behavior, since this sandbox has no network
        # access to check public.bybit.com directly): Bybit's public dump splits
        # trade data into separate top-level paths by market type -- 'trading/'
        # for derivatives (no separator in the filename, e.g.
        # BTCUSD2024-01-01.csv.gz) and 'spot/' for spot pairs (underscore
        # separator, e.g. BTCUSDT_2024-01-01.csv.gz). This project is spot-only
        # (see docs/ARCHITECTURE.md), so it must use the 'spot/' path with the
        # underscore -- a prior version of this fix corrected the fake-depth-
        # synthesis bug but still pointed at 'trading/' with no separator,
        # which would 404 or silently pull the wrong market's data for a spot
        # symbol like BTCUSDT.
        gz_filename = f"{bybit_sym}_{date}.csv.gz"
        url = f"{self.base_url}spot/{bybit_sym}/{gz_filename}"

        async with self._semaphore:
            async with aiohttp.ClientSession() as session:
                content = await self._download_with_retry(session, url)
                if content is None:
                    return ""

                if self._convert_trades_to_parquet(content, parquet_path):
                    return parquet_path
                return ""

    async def ingest_range(self, symbol: str, start_date: str, end_date: str) -> List[str]:
        """Ingests a date range of Bybit trade ticks with rate-limited concurrency."""
        import pandas as pd
        dates = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()
        tasks = [self.download_daily_trades(symbol, date) for date in dates]
        results = await asyncio.gather(*tasks)
        successful = [r for r in results if r]
        logger.info(f"Bybit trades ingest complete for {symbol}. {len(successful)}/{len(dates)} days.")
        return successful

    async def ingest_all_symbols(self, start_date: str, end_date: str) -> dict:
        """Ingests trade ticks for all configured default crypto symbols."""
        results = {}
        for symbol in config.default_symbols:
            logger.info(f"Starting Bybit trades ingest for {symbol}...")
            results[symbol] = await self.ingest_range(symbol, start_date, end_date)
        return results


# Backwards-compatible alias -- old code/tests may still import BybitL2Ingest by name.
# It is no longer an L2 depth ingestor; it downloads trade ticks. Prefer BybitTradesIngest.
BybitL2Ingest = BybitTradesIngest


if __name__ == "__main__":
    async def main():
        ingestor = BybitTradesIngest()
        await ingestor.ingest_range("BTC-USDT", "2024-01-01", "2024-01-03")

    asyncio.run(main())
