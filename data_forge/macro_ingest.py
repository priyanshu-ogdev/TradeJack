"""
Macro-Regime Foundation Ingest Engine.
Downloads historical 1-minute OHLCV datasets from multiple free sources:
  1. Primary: Binance Vision klines via kline_ingest.py (free, verified)
  2. Secondary: HuggingFace datasets (e.g. mito0o852/OHLCV-1m) for enrichment/gap-filling
Teaches the swarm long-term volatility clustering, macroeconomic cycles, and cross-asset correlations.

SOTA Upgrades:
  - Integration with BinanceKlineIngest as primary free source
  - Progress logging for large HuggingFace downloads
  - Multi-symbol support via config.default_symbols
"""

import os
import logging
import asyncio
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from data_forge.config import config
from data_forge.kline_ingest import BinanceKlineIngest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (MacroIngest) %(message)s")
logger = logging.getLogger("MacroIngest")

class MacroOHLCVIngest:
    def __init__(self):
        self.store_dir = os.path.join(config.data_store_dir, "macro")
        os.makedirs(self.store_dir, exist_ok=True)
        self.hf_token = config.hf_token if config.hf_token else None
        self.kline_ingestor = BinanceKlineIngest()

    async def ingest_binance_klines(self, start_date: str, end_date: str, interval: str = "1m") -> dict:
        """
        Primary macro data source: downloads Binance Vision 1-minute klines for all default symbols.
        This is the preferred free data source for macro-regime detection.
        """
        logger.info(f"Ingesting Binance Vision klines ({interval}) for {config.default_symbols} from {start_date} to {end_date}...")
        results = await self.kline_ingestor.ingest_all_symbols(start_date, end_date, interval)
        total = sum(len(v) for v in results.values())
        logger.info(f"Binance kline macro ingest complete: {total} total partition files across {len(results)} symbols.")
        return results

    async def download_hf_dataset(self, repo_id: str, dataset_type: str = "dataset"):
        """
        Secondary enrichment source: downloads a full dataset snapshot from HuggingFace.
        Specifically targets datasets like 'mito0o852/OHLCV-1m'.
        Includes progress logging for large downloads (87GB+).
        """
        logger.info(f"Initiating HuggingFace snapshot download for {repo_id}...")
        logger.info(f"NOTE: Large datasets (e.g., 87GB) may take hours. Progress is logged by huggingface_hub.")

        target_dir = os.path.join(self.store_dir, repo_id.replace("/", "_"))

        # Run synchronous HF download in a thread to avoid blocking asyncio loop
        loop = asyncio.get_running_loop()
        try:
            local_dir = await loop.run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=repo_id,
                    repo_type=dataset_type,
                    local_dir=target_dir,
                    token=self.hf_token,
                    resume_download=True,
                    max_workers=4
                )
            )
            logger.info(f"Successfully downloaded {repo_id} to {local_dir}")
            return local_dir
        except Exception as e:
            logger.error(f"Failed to download HF dataset {repo_id}: {e}")
            return None

    async def full_macro_ingest(self, start_date: str, end_date: str):
        """
        Executes a complete macro-regime data ingest:
        1. Primary: Binance Vision klines (free, verified)
        2. Secondary: HuggingFace enrichment (optional, large)
        """
        logger.info("Starting full macro-regime data ingest...")

        # Step 1: Binance Vision klines (fast, reliable, free)
        kline_results = await self.ingest_binance_klines(start_date, end_date)

        # Step 2: HuggingFace enrichment (optional — uncomment for full 87GB dataset)
        # hf_result = await self.download_hf_dataset("mito0o852/OHLCV-1m")

        logger.info("Macro-regime foundation data ingest complete.")
        return kline_results

if __name__ == "__main__":
    async def main():
        ingestor = MacroOHLCVIngest()
        # Test: ingest 3 days of klines for all default symbols
        await ingestor.ingest_binance_klines("2024-01-01", "2024-01-03")
        logger.info("Macro Ingest Engine Ready.")

    asyncio.run(main())
