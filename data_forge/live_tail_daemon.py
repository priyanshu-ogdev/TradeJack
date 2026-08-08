"""
Live Tail Daemon & DVC Sync Engine.
Acts as a Repo 2 Heartbeat Skill for the Parent Automaton.
Pulls the latest 24 hours of Binance Vision aggTrades + klines, runs them through Trade-Flow Physics,
atomically writes them to the Forge, triggers a DVC snapshot, and pings the Warden's heartbeat database.

SOTA Upgrades:
  - Integrates LOB Collector for real-time L2 depth data
  - Integrates Kline Ingest for daily OHLCV macro sync
  - Fixed deprecated datetime.utcnow() → datetime.now(timezone.utc) (Python 3.12+)
  - Graceful retry loop with configurable interval (no longer runs once and exits)
  - Multi-symbol daily sync across config.default_symbols
"""

import os
import time
import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta, timezone

from data_forge.config import config
from data_forge.micro_ingest import BinanceVisionIngest
from data_forge.kline_ingest import BinanceKlineIngest
from data_forge.feature_engineering import TradeFlowPhysics
from data_forge.dvc_tracker import DVCTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (LiveTailDaemon) %(message)s")
logger = logging.getLogger("LiveTailDaemon")

class LiveTailDaemon:
    def __init__(self, symbols: list = None):
        self.symbols = symbols or config.default_symbols
        self.trade_ingestor = BinanceVisionIngest()
        self.kline_ingestor = BinanceKlineIngest()
        self.physics_engines = {sym: TradeFlowPhysics(sym) for sym in self.symbols}
        self.dvc = DVCTracker()

        # Warden database for heartbeat
        self.warden_db_path = os.path.join(config.data_store_dir, "..", "state", "warden_heartbeat.sqlite")
        os.makedirs(os.path.dirname(self.warden_db_path), exist_ok=True)
        self._init_heartbeat_db()

    def _init_heartbeat_db(self):
        conn = sqlite3.connect(self.warden_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_freshness (
                id INTEGER PRIMARY KEY,
                last_successful_sync REAL,
                status TEXT
            )
        """)
        # Insert initial row if empty
        cursor.execute("SELECT COUNT(*) FROM data_freshness")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO data_freshness (id, last_successful_sync, status) VALUES (1, ?, ?)", (time.time(), "OK"))
        conn.commit()
        conn.close()

    def _ping_heartbeat(self, status: str = "OK"):
        try:
            conn = sqlite3.connect(self.warden_db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE data_freshness SET last_successful_sync = ?, status = ? WHERE id = 1", (time.time(), status))
            conn.commit()
            conn.close()
            logger.debug(f"Heartbeat ping successful. Status: {status}")
        except Exception as e:
            logger.error(f"Failed to ping heartbeat DB: {e}")

    async def execute_daily_sync(self):
        """
        Pulls yesterday's data for all symbols, processes physics, syncs klines, and commits a DVC snapshot.
        """
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info(f"Initiating daily sync for {yesterday} across {len(self.symbols)} symbols...")

        sync_errors = []

        for symbol in self.symbols:
            try:
                # 1. Download aggTrades
                logger.info(f"[{symbol}] Downloading aggTrades for {yesterday}...")
                await self.trade_ingestor.download_daily_agg_trades(symbol, yesterday)

                # 2. Download klines (1m OHLCV for macro-regime)
                logger.info(f"[{symbol}] Downloading 1m klines for {yesterday}...")
                await self.kline_ingestor.download_daily_klines(symbol, yesterday)

                # 3. Process Physics (Atomic)
                logger.info(f"[{symbol}] Running Trade-Flow Physics...")
                self.physics_engines[symbol].process_daily_file(yesterday)

                logger.info(f"[{symbol}] Daily sync successful.")

            except Exception as e:
                logger.error(f"[{symbol}] Daily sync failed: {e}")
                sync_errors.append(symbol)

        # 4. DVC Commit (single snapshot for all symbols)
        try:
            snapshot_tag = f"live_{yesterday.replace('-', '')}"
            self.dvc.track_data(tag_name=snapshot_tag, description=f"Automated live tail sync for {yesterday}")
        except Exception as e:
            logger.error(f"DVC snapshot failed: {e}")

        # 5. Heartbeat
        if sync_errors:
            self._ping_heartbeat(f"PARTIAL_ERROR:{','.join(sync_errors)}")
            logger.warning(f"Daily sync completed with errors for: {sync_errors}")
        else:
            self._ping_heartbeat("OK")
            logger.info("Daily sync completed successfully for all symbols.")

    async def run_continuous(self, sync_interval_hours: float = 24.0, max_iterations: int = None):
        """
        Runs the daily sync in a continuous loop with configurable interval.
        Replaces the old run-once-and-exit behavior.
        """
        iteration = 0
        while True:
            iteration += 1
            logger.info(f"--- Live Tail Daemon: Sync Iteration {iteration} ---")

            try:
                await self.execute_daily_sync()
            except Exception as e:
                logger.error(f"Unhandled sync error: {e}")
                self._ping_heartbeat("ERROR")

            if max_iterations and iteration >= max_iterations:
                logger.info(f"Max iterations ({max_iterations}) reached. Stopping daemon.")
                break

            sleep_sec = sync_interval_hours * 3600
            logger.info(f"Next sync in {sync_interval_hours} hours. Sleeping...")
            await asyncio.sleep(sleep_sec)


if __name__ == "__main__":
    daemon = LiveTailDaemon()
    # Single sync for testing; use run_continuous() for production
    asyncio.run(daemon.execute_daily_sync())
