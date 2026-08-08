"""
Binance Vision Asynchronous Ingest Engine (Trade-Flow Physics).
Downloads historical `aggTrades` (executed trades) to power VPIN, OFI, and Kyle's Lambda calculations.
Completely bypasses spoofable L2 depth by tracking real money changing hands.

SOTA Upgrades:
  - CSV→Parquet on ingest: immediately converts raw CSVs to ZSTD-L3 Parquet, then deletes the CSV
  - ZIP cleanup: deletes ZIP payload after extraction to prevent storage bloat
  - SHA256 checksum verification: validates data integrity using Binance's .CHECKSUM files
  - Semaphore-based rate limiting: prevents Binance Vision rate-limit bans
  - Exponential backoff retries: resilient download with configurable retry count
  - Multi-symbol support: ingest across configurable symbol list
"""

import os
import io
import hashlib
import aiohttp
import asyncio
import logging
import zipfile
from datetime import datetime
from typing import List, Optional

from data_forge.config import config

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (MicroIngest) %(message)s")
logger = logging.getLogger("MicroIngest")

# Column names for Binance aggTrades CSVs (no header)
AGG_TRADES_COLUMNS = [
    "agg_trade_id", "price", "quantity",
    "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker", "is_best_match"
]


class BinanceVisionIngest:
    """
    Asynchronously downloads, verifies, and converts Binance Vision monthly/daily aggTrades.
    Raw CSVs are immediately converted to ZSTD-L3 Parquet to eliminate storage waste.
    """

    def __init__(self):
        self.base_url = config.binance_vision_base_url
        self.store_dir = config.data_store_dir
        self._semaphore = asyncio.Semaphore(config.max_concurrent_downloads)

    def _get_target_path(self, symbol: str, date: str, ext: str = ".parquet") -> str:
        """Returns the canonical storage path for a given symbol/date."""
        target_dir = os.path.join(self.store_dir, "raw", symbol, "aggTrades", date.replace("-", "/"))
        os.makedirs(target_dir, exist_ok=True)
        sym = symbol.replace("-", "").upper()
        return os.path.join(target_dir, f"{sym}-aggTrades-{date}{ext}")

    async def _download_with_retry(self, session: aiohttp.ClientSession, url: str, max_retries: int = None) -> Optional[bytes]:
        """Downloads a URL with exponential backoff retry."""
        max_retries = max_retries or config.download_retry_max
        for attempt in range(max_retries):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
                    elif response.status == 404:
                        logger.warning(f"Not found (404): {url}")
                        return None
                    else:
                        logger.warning(f"HTTP {response.status} for {url} (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logger.warning(f"Download error for {url} (attempt {attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                backoff = 2 ** attempt
                logger.debug(f"Retrying in {backoff}s...")
                await asyncio.sleep(backoff)

        logger.error(f"Failed to download {url} after {max_retries} attempts.")
        return None

    async def _verify_checksum(self, session: aiohttp.ClientSession, zip_url: str, zip_content: bytes) -> bool:
        """Verifies SHA256 checksum using Binance's .CHECKSUM file."""
        checksum_url = zip_url + ".CHECKSUM"
        checksum_data = await self._download_with_retry(session, checksum_url, max_retries=2)
        if not checksum_data:
            logger.debug(f"No checksum file available for {zip_url}, skipping verification.")
            return True  # Checksum file is optional; proceed without verification

        try:
            expected_hash = checksum_data.decode("utf-8").strip().split()[0].lower()
            actual_hash = hashlib.sha256(zip_content).hexdigest().lower()
            if expected_hash == actual_hash:
                logger.debug(f"Checksum verified for {zip_url}")
                return True
            else:
                logger.error(f"CHECKSUM MISMATCH for {zip_url}: expected={expected_hash}, got={actual_hash}")
                return False
        except Exception as e:
            logger.warning(f"Checksum parse error for {zip_url}: {e}")
            return True  # Don't block on malformed checksum files

    def _convert_csv_to_parquet(self, csv_path: str, parquet_path: str) -> bool:
        """Converts raw aggTrades CSV to ZSTD-L3 Parquet using Polars streaming."""
        if not POLARS_AVAILABLE:
            logger.debug("Polars not available; keeping raw CSV.")
            return False
        try:
            df = pl.read_csv(
                csv_path,
                has_header=False,
                new_columns=AGG_TRADES_COLUMNS,
            )
            df.write_parquet(
                parquet_path,
                compression=config.compression_codec,
                compression_level=config.compression_level,
                row_group_size=config.row_group_size,
            )
            logger.debug(f"Converted CSV→Parquet ({len(df)} rows): {parquet_path}")
            return True
        except Exception as e:
            logger.error(f"CSV→Parquet conversion failed for {csv_path}: {e}")
            return False

    async def download_daily_agg_trades(self, symbol: str, date: str) -> str:
        """
        Downloads a single daily aggTrades ZIP file for a given symbol and date (YYYY-MM-DD),
        extracts, converts to ZSTD Parquet, and cleans up intermediate files.
        """
        parquet_path = self._get_target_path(symbol, date, ext=".parquet")
        csv_path = self._get_target_path(symbol, date, ext=".csv")

        # Skip if already converted to Parquet
        if os.path.exists(parquet_path):
            logger.debug(f"Parquet already exists for {symbol} on {date}. Skipping.")
            return parquet_path

        # Skip if CSV already exists (convert it)
        if os.path.exists(csv_path):
            logger.debug(f"CSV exists for {symbol} on {date}. Converting to Parquet...")
            if self._convert_csv_to_parquet(csv_path, parquet_path):
                os.remove(csv_path)
                return parquet_path
            return csv_path

        sym = symbol.replace("-", "").upper()
        zip_filename = f"{sym}-aggTrades-{date}.zip"
        url = f"{self.base_url}data/spot/daily/aggTrades/{sym}/{zip_filename}"

        logger.info(f"Downloading aggTrades for {symbol} on {date}...")

        async with self._semaphore:
            async with aiohttp.ClientSession() as session:
                content = await self._download_with_retry(session, url)
                if content is None:
                    return ""

                # SHA256 checksum verification
                if not await self._verify_checksum(session, url, content):
                    logger.error(f"Skipping {symbol} {date} due to checksum failure.")
                    return ""

                try:
                    target_dir = os.path.dirname(csv_path)
                    with zipfile.ZipFile(io.BytesIO(content)) as z:
                        z.extractall(target_dir)

                    # Find extracted CSV files
                    extracted_csvs = [
                        os.path.join(target_dir, f) for f in os.listdir(target_dir)
                        if f.endswith(".csv")
                    ]

                    if extracted_csvs and POLARS_AVAILABLE:
                        # Convert to ZSTD-L3 Parquet immediately
                        source_csv = extracted_csvs[0]
                        if self._convert_csv_to_parquet(source_csv, parquet_path):
                            # Clean up: delete all extracted CSVs
                            for csv_f in extracted_csvs:
                                try:
                                    os.remove(csv_f)
                                except OSError:
                                    pass
                            logger.info(f"Successfully ingested {symbol} {date} as ZSTD Parquet.")
                            return parquet_path

                    logger.info(f"Successfully downloaded and extracted {symbol} aggTrades for {date}.")
                    return csv_path if os.path.exists(csv_path) else ""

                except Exception as e:
                    logger.error(f"Extraction failed for {url}: {e}")
                    return ""

    async def ingest_range(self, symbol: str, start_date: str, end_date: str) -> List[str]:
        """
        Ingests a date range of daily aggTrades with rate-limited concurrency.
        """
        import pandas as pd
        dates = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()

        tasks = [self.download_daily_agg_trades(symbol, date) for date in dates]
        results = await asyncio.gather(*tasks)

        successful = [r for r in results if r]
        logger.info(f"Ingest complete for {symbol}. Successfully downloaded {len(successful)}/{len(dates)} days.")
        return successful

    async def ingest_all_symbols(self, start_date: str, end_date: str) -> dict:
        """Ingests aggTrades for all configured default symbols."""
        results = {}
        for symbol in config.default_symbols:
            logger.info(f"Starting multi-symbol ingest for {symbol}...")
            results[symbol] = await self.ingest_range(symbol, start_date, end_date)
        return results


if __name__ == "__main__":
    async def main():
        ingestor = BinanceVisionIngest()
        # Download 2 days of BTC-USDT as a test
        await ingestor.ingest_range("BTC-USDT", "2024-01-01", "2024-01-02")

    asyncio.run(main())
