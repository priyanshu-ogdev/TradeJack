"""
Binance Vision Kline (OHLCV) Ingest Engine.
Downloads free 1-minute klines from Binance Vision for macro-regime detection
and cross-asset correlation analysis. Supplements aggTrades-based physics pipeline.

Data Source: https://data.binance.vision/data/spot/daily/klines/{SYMBOL}/1m/
Format: ZIP → CSV → ZSTD-L3 Parquet (no intermediate CSV waste)
"""

import os
import io
import hashlib
import aiohttp
import asyncio
import logging
import zipfile
from typing import List, Optional

from data_forge.config import config

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (KlineIngest) %(message)s")
logger = logging.getLogger("KlineIngest")

# Binance kline CSV columns (no header)
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
]


class BinanceKlineIngest:
    """
    Downloads and converts Binance Vision 1-minute kline data to ZSTD-L3 Parquet.
    Provides clean OHLCV for macro-regime detection (volatility clustering, cross-asset correlations).
    """

    def __init__(self):
        self.base_url = config.binance_vision_base_url
        self.store_dir = config.data_store_dir
        self._semaphore = asyncio.Semaphore(config.max_concurrent_downloads)

    def _get_target_dir(self, symbol: str, date: str) -> str:
        target_dir = os.path.join(self.store_dir, "raw", symbol, "klines", date.replace("-", "/"))
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    async def _download_with_retry(self, session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
        for attempt in range(config.download_retry_max):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
                    elif response.status == 404:
                        logger.warning(f"Not found (404): {url}")
                        return None
                    else:
                        logger.warning(f"HTTP {response.status} for {url} (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Download error: {e} (attempt {attempt + 1})")
            if attempt < config.download_retry_max - 1:
                await asyncio.sleep(2 ** attempt)
        return None

    async def _verify_checksum(self, session: aiohttp.ClientSession, zip_url: str, zip_content: bytes) -> bool:
        checksum_url = zip_url + ".CHECKSUM"
        checksum_data = await self._download_with_retry(session, checksum_url)
        if not checksum_data:
            return True
        try:
            expected = checksum_data.decode("utf-8").strip().split()[0].lower()
            actual = hashlib.sha256(zip_content).hexdigest().lower()
            if expected != actual:
                logger.error(f"CHECKSUM MISMATCH: {zip_url}")
                return False
            return True
        except Exception:
            return True

    async def download_daily_klines(self, symbol: str, date: str, interval: str = "1m") -> str:
        """Downloads a single day of kline data and converts to ZSTD Parquet."""
        target_dir = self._get_target_dir(symbol, date)
        sym = symbol.replace("-", "").upper()
        parquet_path = os.path.join(target_dir, f"{sym}-{interval}-{date}.parquet")

        if os.path.exists(parquet_path):
            logger.debug(f"Kline Parquet already exists for {symbol} {date}. Skipping.")
            return parquet_path

        zip_filename = f"{sym}-{interval}-{date}.zip"
        url = f"{self.base_url}data/spot/daily/klines/{sym}/{interval}/{zip_filename}"

        async with self._semaphore:
            async with aiohttp.ClientSession() as session:
                content = await self._download_with_retry(session, url)
                if content is None:
                    return ""

                if not await self._verify_checksum(session, url, content):
                    return ""

                try:
                    with zipfile.ZipFile(io.BytesIO(content)) as z:
                        csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                        if not csv_names:
                            logger.error(f"No CSV found in ZIP for {symbol} {date}")
                            return ""

                        csv_data = z.read(csv_names[0])

                    if POLARS_AVAILABLE:
                        df = pl.read_csv(
                            io.BytesIO(csv_data),
                            has_header=False,
                            new_columns=KLINE_COLUMNS,
                        )
                        # Drop the 'ignore' column
                        df = df.drop("ignore")
                        df.write_parquet(
                            parquet_path,
                            compression=config.compression_codec,
                            compression_level=config.compression_level,
                            row_group_size=config.row_group_size,
                        )
                        logger.info(f"Ingested klines {symbol} {date} ({len(df)} rows) → ZSTD Parquet")
                        return parquet_path
                    else:
                        # Fallback: save raw CSV
                        csv_path = parquet_path.replace(".parquet", ".csv")
                        with open(csv_path, "wb") as f:
                            f.write(csv_data)
                        logger.info(f"Ingested klines {symbol} {date} (raw CSV fallback)")
                        return csv_path

                except Exception as e:
                    logger.error(f"Kline extraction failed for {symbol} {date}: {e}")
                    return ""

    async def ingest_range(self, symbol: str, start_date: str, end_date: str, interval: str = "1m") -> List[str]:
        """Ingests a date range of klines with rate-limited concurrency."""
        import pandas as pd
        dates = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()
        tasks = [self.download_daily_klines(symbol, date, interval) for date in dates]
        results = await asyncio.gather(*tasks)
        successful = [r for r in results if r]
        logger.info(f"Kline ingest complete for {symbol}. {len(successful)}/{len(dates)} days.")
        return successful

    async def ingest_all_symbols(self, start_date: str, end_date: str, interval: str = "1m") -> dict:
        """Ingests klines for all configured default symbols."""
        results = {}
        for symbol in config.default_symbols:
            logger.info(f"Starting kline ingest for {symbol}...")
            results[symbol] = await self.ingest_range(symbol, start_date, end_date, interval)
        return results


if __name__ == "__main__":
    async def main():
        ingestor = BinanceKlineIngest()
        await ingestor.ingest_range("BTC-USDT", "2024-01-01", "2024-01-03")

    asyncio.run(main())
