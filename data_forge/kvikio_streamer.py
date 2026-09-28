"""
KvikIO Data Forge: High-Speed Parquet Pipeline with GPUDirect Storage (GDS) for Grace Blackwell (CUDA 13).
Dynamically adapts across:
1. NVIDIA DGX Spark Mode (CUDA 13 + KvikIO cuFile + cuDF direct NVMe-to-GPU memory streaming).
2. Laptop Development Mode with Polars (CPU / zero-copy Polars + PyTorch tensor mmap fallback).
3. Laptop Development Mode without Polars/PyTorch (Pure Numpy / ZSTD binary fallback).
Integrates Container-Specific FIFO State Bridge and Atomic Quarantine Webhooks.

SOTA Upgrades:
  - Memory-mapped reads via Polars scan_parquet() lazy frames with streaming collection
  - Partition pruning: date-range predicate pushdown to avoid loading entire files
  - Live data integration: scan_live_partitions() reads from lob_collector output
  - Explicit ZSTD codec handling across all paths
"""

import os
import shutil
import logging
import json
import time
import sqlite3
import numpy as np
from typing import Dict, Any, List, Optional

from data_forge.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (KvikIODataForge) %(message)s")
logger = logging.getLogger("KvikIODataForge")

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cudf
    KVIKIO_AVAILABLE = True
except ImportError:
    KVIKIO_AVAILABLE = False

from data_forge.schema import TradeJackPhysicsSchema, LOBDepthSchema, RawTradeSchema

class KvikIODataForge:
    """
    Streams financial datasets into tensor memory.
    Leverages CUDA 13 GPUDirect Storage on DGX Blackwell or zero-copy Polars/Numpy on laptop CPU.
    """

    def __init__(
        self,
        data_store_dir: str = None,
        use_gds_if_available: bool = True,
        target_device: str = "cuda:0"
    ):
        self.data_store_dir = os.path.abspath(data_store_dir or config.data_store_dir)
        os.makedirs(self.data_store_dir, exist_ok=True)

        self.state_dir = os.path.join(self.data_store_dir, "..", "state")
        os.makedirs(self.state_dir, exist_ok=True)

        self.is_blackwell_dgx = (
            use_gds_if_available and
            KVIKIO_AVAILABLE and
            TORCH_AVAILABLE and
            torch.cuda.is_available()
        )
        self.target_device = target_device if self.is_blackwell_dgx else "cpu"

        self._init_fifo_db()

        if self.is_blackwell_dgx:
            logger.info(f"Initialized KvikIODataForge in DGX Blackwell Mode (CUDA 13 GPUDirect Storage on {self.target_device}).")
        elif POLARS_AVAILABLE:
            logger.info("Initialized KvikIODataForge in Laptop Development Mode (CPU / Polars active).")
        else:
            logger.info("Initialized KvikIODataForge in Laptop Simulation Mode (CPU / Numpy binary fallback).")

    def _init_fifo_db(self):
        self.fifo_db_path = os.path.join(self.state_dir, "fifo_state.sqlite")
        conn = sqlite3.connect(self.fifo_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS container_fifo (
                container_name TEXT,
                symbol TEXT,
                tail_data TEXT,
                PRIMARY KEY (container_name, symbol)
            )
        """)
        conn.commit()
        conn.close()

    def _get_fifo_tail(self, container_name: str, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(self.fifo_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT tail_data FROM container_fifo WHERE container_name = ? AND symbol = ?", (container_name, symbol))
            row = cursor.fetchone()
            conn.close()
            if row:
                data = json.loads(row[0])
                res = {}
                for k, v in data.items():
                    if TORCH_AVAILABLE:
                        res[k] = torch.tensor(v, device=self.target_device)
                    else:
                        res[k] = np.array(v)
                return res
        except Exception as e:
            logger.error(f"Failed to fetch FIFO tail for {container_name}: {e}")
        return None

    def _save_fifo_tail(self, container_name: str, symbol: str, tensors: Dict[str, Any]):
        try:
            data = {}
            for k, v in tensors.items():
                if TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                    # Keep last 100
                    data[k] = v[-100:].cpu().tolist()
                elif isinstance(v, np.ndarray):
                    data[k] = v[-100:].tolist()
                else:
                    return

            conn = sqlite3.connect(self.fifo_db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO container_fifo (container_name, symbol, tail_data)
                VALUES (?, ?, ?)
                ON CONFLICT(container_name, symbol) DO UPDATE SET tail_data = excluded.tail_data
            """, (container_name, symbol, json.dumps(data, default=str)))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save FIFO tail for {container_name}: {e}")

    @staticmethod
    def _select_schema(file_path: str):
        """Picks the correct schema for a partition based on which pipeline produced it.

        Live L2 depth (lob_collector.py output, under data_store/live/...) and any file whose
        directory/name says "depth" validates against LOBDepthSchema (bid/ask tiers). Raw
        aggTrades dumps (data_store/raw/<symbol>/aggTrades/...) validate against RawTradeSchema.
        Everything else is assumed to be feature_engineering.py output and validates against
        TradeJackPhysicsSchema (OHLC + OFI/VPIN/Kyle's Lambda columns). Validating live depth
        files against the physics schema was the root cause of all live L2 data being quarantined.

        MERGE NOTE: an earlier version of this fix matched by path-tier only ("live" as a path
        component), which missed already-relocated files like data_store/quarantine/depth_13.parquet
        (no "live" segment left in the path, but still a depth file by name). A separately-developed
        version matched by lowercased path/basename substring ("/live/" or "depth" in the filename),
        which catches that case but had silently dropped RawTradeSchema routing entirely -- meaning
        raw aggTrades dumps (which have neither a depth-like name nor an open_price column) would
        fall through to TradeJackPhysicsSchema and quarantine too. This merges both: substring
        matching for robustness to relocated/renamed files, plus the raw-aggTrades branch restored.
        """
        normalized = file_path.replace("\\", "/").lower()
        if "/live/" in normalized or "depth" in os.path.basename(normalized):
            return LOBDepthSchema
        if "/raw/" in normalized and "aggtrades" in normalized:
            return RawTradeSchema
        return TradeJackPhysicsSchema

    def _validate_schema_lazy(self, file_path: str) -> bool:
        """Uses CPU-side Polars LazyFrame to validate schema before hitting GPU VRAM."""
        if not POLARS_AVAILABLE or not file_path.endswith(".parquet"):
            return True
        schema_cls = self._select_schema(file_path)
        try:
            lazy_df = pl.scan_parquet(file_path).head(1000)
            schema_cls.validate(lazy_df.collect())
            return True
        except Exception as e:
            logger.error(f"Schema validation failed for {file_path}. Error: {e}")

            # Atomic Append to Quarantine Events
            log_path = os.path.join(self.state_dir, "logs", "quarantine_events.jsonl")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            event = {
                "timestamp": time.time(),
                "file_path": file_path,
                "error": str(e)
            }
            try:
                fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
                os.write(fd, (json.dumps(event) + "\n").encode("utf-8"))
                os.close(fd)
            except Exception as io_err:
                logger.error(f"Failed to append to quarantine_events.jsonl: {io_err}")

            quarantine_dir = os.path.join(self.data_store_dir, "quarantine")
            os.makedirs(quarantine_dir, exist_ok=True)
            try:
                shutil.move(file_path, os.path.join(quarantine_dir, os.path.basename(file_path)))
                logger.info(f"Moved corrupted file to {quarantine_dir}")
            except Exception as move_err:
                logger.error(f"Failed to move file to quarantine: {move_err}")
            return False

    def load_file_to_tensor(
        self,
        file_path: str,
        columns: Optional[List[str]] = None,
        container_name: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Loads a partition file (`physics.parquet` or `physics.npz`) directly into memory tensors or Numpy arrays.
        Uses memory-mapped lazy streaming via Polars scan_parquet() to minimize RAM pressure.
        Optionally prepends the FIFO State Bridge if container_name and symbol are provided.
        """
        if not os.path.exists(file_path):
            return None

        # CPU-side Validation to prevent VRAM fragmentation
        if not self._validate_schema_lazy(file_path):
            return None

        tensors = None
        if self.is_blackwell_dgx and file_path.endswith(".parquet"):
            try:
                df_gpu = cudf.read_parquet(file_path, columns=columns)
                tensors = {col: torch.as_tensor(df_gpu[col].to_dlpack(), device=self.target_device) for col in df_gpu.columns}
            except Exception as e:
                logger.debug(f"KvikIO cuFile load failed ({e}), falling back to CPU loader.")

        if tensors is None and file_path.endswith(".parquet") and POLARS_AVAILABLE:
            try:
                # Memory-efficient: use lazy scan with streaming collection
                lazy = pl.scan_parquet(file_path, cache=False)
                if columns:
                    lazy = lazy.select(columns)
                df = lazy.collect(streaming=True)

                tensors = {}
                for col in df.columns:
                    if TORCH_AVAILABLE and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                        tensors[col] = torch.from_numpy(df[col].to_numpy().copy())
                    else:
                        tensors[col] = df[col].to_numpy().copy()
            except Exception as e:
                logger.error(f"Failed to load Parquet via Polars streaming: {e}")
                return None

        if tensors is None and file_path.endswith(".npz"):
            try:
                data = np.load(file_path)
                tensors = {}
                for col in (columns or data.files):
                    if col in data:
                        arr = data[col].astype(np.float32)
                        tensors[col] = torch.from_numpy(arr) if TORCH_AVAILABLE else arr
            except Exception as e:
                logger.error(f"Failed to load NPZ fallback: {e}")
                return None

        if tensors is None:
            return None

        # Prepend Container-Specific FIFO State Bridge
        if container_name and symbol:
            fifo_tail = self._get_fifo_tail(container_name, symbol)
            if fifo_tail:
                try:
                    for k in tensors.keys():
                        if k in fifo_tail:
                            if TORCH_AVAILABLE and isinstance(tensors[k], torch.Tensor):
                                tensors[k] = torch.cat([fifo_tail[k], tensors[k]], dim=0)
                            elif isinstance(tensors[k], np.ndarray):
                                tensors[k] = np.concatenate([fifo_tail[k], tensors[k]], axis=0)
                except Exception as e:
                    logger.error(f"Failed to prepend FIFO tail: {e}")

            # Save new tail for next load
            self._save_fifo_tail(container_name, symbol, tensors)

        return tensors

    def scan_available_partitions(self, symbol: str) -> List[str]:
        """Returns list of all partition file paths available for the symbol in data_store."""
        results = []
        symbol_dir = os.path.join(self.data_store_dir, "processed", symbol, "physics")
        if not os.path.exists(symbol_dir):
            return results
        for root, dirs, files in os.walk(symbol_dir):
            for file in files:
                if file.endswith(".parquet") or file.endswith(".npz"):
                    results.append(os.path.join(root, file))
        return sorted(results)

    def scan_live_partitions(self, symbol: str) -> List[str]:
        """Returns list of all live L2 depth partition files from the lob_collector output."""
        results = []
        live_dir = os.path.join(self.data_store_dir, "live", symbol)
        if not os.path.exists(live_dir):
            return results
        for root, dirs, files in os.walk(live_dir):
            for file in files:
                if file.endswith(".parquet"):
                    results.append(os.path.join(root, file))
        return sorted(results)

    def stream_partition_window(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None,
        container_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Iterates across partition directory and streams tensors/arrays.
        Uses partition pruning via date tuple comparison to skip irrelevant files.
        Maintains Container-Specific FIFO if container_name is provided.
        """
        results = []
        symbol_dir = os.path.join(self.data_store_dir, "processed", symbol, "physics")
        if not os.path.exists(symbol_dir):
            return results

        start_tuple = tuple(map(int, start_date.split("-")))
        end_tuple = tuple(map(int, end_date.split("-")))

        for root, dirs, files in os.walk(symbol_dir):
            for file in files:
                if file.endswith(".parquet") or file.endswith(".npz"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, symbol_dir)
                    parts = rel_path.split(os.sep)
                    if len(parts) >= 4:
                        try:
                            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                            # Partition pruning: skip files outside date range
                            if start_tuple <= (year, month, day) <= end_tuple:
                                t_data = self.load_file_to_tensor(
                                    full_path,
                                    columns=columns,
                                    container_name=container_name,
                                    symbol=symbol
                                )
                                if t_data:
                                    results.append(t_data)
                        except ValueError:
                            continue
        return results

    def stream_live_window(
        self,
        symbol: str,
        columns: Optional[List[str]] = None,
        max_files: int = 24,
    ) -> List[Dict[str, Any]]:
        """
        Streams the most recent live L2 depth data from the lob_collector partitions.
        Returns up to max_files most recent hourly partitions.
        """
        results = []
        live_files = self.scan_live_partitions(symbol)

        # Take most recent files
        recent_files = live_files[-max_files:] if len(live_files) > max_files else live_files

        for file_path in recent_files:
            t_data = self.load_file_to_tensor(file_path, columns=columns)
            if t_data:
                results.append(t_data)

        if results:
            logger.debug(f"Streamed {len(results)} live L2 partitions for {symbol}")
        return results


if __name__ == "__main__":
    forge = KvikIODataForge()
    print("Is DGX Blackwell Mode Active:", forge.is_blackwell_dgx)
    print("Available processed partitions (BTC-USDT):", len(forge.scan_available_partitions("BTC-USDT")))
    print("Available live partitions (BTC-USDT):", len(forge.scan_live_partitions("BTC-USDT")))
