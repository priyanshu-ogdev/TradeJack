"""
Configuration Loader for TradeJack Data Forge.
Uses Pydantic Settings to load and type-cast environment variables safely.
Supports ZSTD-L3 compression, 2.5 TB storage budgets, multi-symbol ingest,
Binance WebSocket live depth collection, and Bybit historical L2 data.

Platform-agnostic: auto-detects project root from this file's location.
Works on both Windows (development) and Linux DGX Spark (production).
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List

# Auto-detect project root: config.py is at data_forge/config.py → project root is one level up
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
_DEFAULT_DATA_STORE = os.path.join(_PROJECT_ROOT, "data_store")


class ForgeConfig(BaseSettings):
    # ── Data Sources ──────────────────────────────────────────────────
    # Binance Vision endpoints (Free public bulk data: aggTrades, klines)
    binance_vision_base_url: str = Field(
        default="https://data.binance.vision/",
        description="Base URL for Binance Vision bulk data"
    )
    # Binance WebSocket (Free real-time L2 depth stream)
    binance_ws_url: str = Field(
        default="wss://stream.binance.com:9443/ws",
        description="Binance public WebSocket endpoint for live L2 depth"
    )
    # Binance REST API (Free snapshot for LOB initialization)
    binance_rest_url: str = Field(
        default="https://api.binance.com",
        description="Binance REST API for order book snapshots"
    )
    # Bybit Historical Trades (Free public bulk data: executed trade ticks, NOT L2 depth)
    # NOTE: Bybit's public dump only publishes executed trades, no L2 depth. For a
    # SPOT symbol this lives under `spot/{SYMBOL}/{SYMBOL}_{DATE}.csv.gz` (underscore
    # separator) -- NOT `trading/{SYMBOL}/{SYMBOL}{DATE}.csv.gz` (no separator), which
    # is the path/filename convention for derivatives, not spot. This project is
    # spot-only (see docs/ARCHITECTURE.md). Do not point this at "orderbook/" either --
    # that directory doesn't exist. See bybit_ingest.py for the full explanation.
    bybit_history_base_url: str = Field(
        default="https://public.bybit.com/",
        description="Base URL for Bybit's free historical spot trades dump (public.bybit.com/spot/)"
    )

    # Dukascopy (Free public bulk data: FX/CFD/metals tick data, no API key required)
    dukascopy_base_url: str = Field(
        default="https://datafeed.dukascopy.com/datafeed/",
        description="Base URL for Dukascopy's free historical FX tick data (.bi5 hourly files)"
    )

    # Qdrant Vector DB
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="URL for the Qdrant instance"
    )

    # HuggingFace (Optional)
    hf_token: str = Field(
        default="",
        description="Optional HuggingFace token for private repos"
    )

    # ── Multi-Symbol Support ──────────────────────────────────────────
    default_symbols: List[str] = Field(
        default=["BTC-USDT", "ETH-USDT", "SOL-USDT"],
        description="Default crypto trading symbols for multi-symbol ingest pipelines"
    )
    default_forex_pairs: List[str] = Field(
        default=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF"],
        description="Default FX pairs for the Dukascopy forex ingest pipeline (Dukascopy naming, no slash)"
    )

    # ── Data Storage ──────────────────────────────────────────────────
    data_store_dir: str = Field(
        default=_DEFAULT_DATA_STORE,
        description="Absolute path to the data storage directory (auto-detected from project root)"
    )
    storage_budget_tb: float = Field(
        default=2.5,
        description="Maximum storage budget in terabytes for the entire data_store"
    )

    # ── Compression Settings (SOTA ZSTD-L3) ──────────────────────────
    compression_codec: str = Field(
        default="zstd",
        description="Parquet compression codec (zstd, snappy, lz4, gzip)"
    )
    compression_level: int = Field(
        default=3,
        description="ZSTD compression level (1-22). Level 3 is the sweet spot for time-series"
    )
    row_group_size: int = Field(
        default=250_000,
        description="Parquet row group size for optimal partition pruning and I/O"
    )

    # ── Network & Rate Limiting ───────────────────────────────────────
    max_concurrent_downloads: int = Field(
        default=8,
        description="Max concurrent async downloads to prevent exchange rate-limit bans"
    )
    download_retry_max: int = Field(
        default=3,
        description="Max retry attempts for failed downloads with exponential backoff"
    )

    # ── Timestamp Handling ────────────────────────────────────────────
    # Binance SPOT data changed from milliseconds to microseconds on 2025-01-01
    binance_us_cutover_epoch_ms: int = Field(
        default=1_735_689_600_000,
        description="Epoch ms threshold: Binance timestamps above this are microseconds (2025-01-01T00:00:00Z)"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# Singleton config instance
config = ForgeConfig()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ForgeConfig")
    logger.info("Loaded Configuration:")
    logger.info(f"Binance Vision URL: {config.binance_vision_base_url}")
    logger.info(f"Binance WS URL: {config.binance_ws_url}")
    logger.info(f"Bybit History URL: {config.bybit_history_base_url}")
    logger.info(f"Qdrant URL: {config.qdrant_url}")
    logger.info(f"Data Store: {config.data_store_dir}")
    logger.info(f"Storage Budget: {config.storage_budget_tb} TB")
    logger.info(f"Compression: {config.compression_codec} (Level {config.compression_level})")
    logger.info(f"Row Group Size: {config.row_group_size}")
    logger.info(f"Default Symbols: {config.default_symbols}")
    logger.info(f"Max Concurrent Downloads: {config.max_concurrent_downloads}")
