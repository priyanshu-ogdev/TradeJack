"""
KvikIO Data Forge: High-Speed Parquet Pipeline with GPUDirect Storage (GDS) for Grace Blackwell (CUDA 13).
Dynamically adapts across:
1. NVIDIA DGX Spark Mode (CUDA 13 + KvikIO cuFile + cuDF direct NVMe-to-GPU memory streaming).
2. Laptop Development Mode with Polars (CPU / zero-copy Polars + PyTorch tensor mmap fallback).
3. Laptop Development Mode without Polars/PyTorch (Pure Numpy / ZSTD binary fallback).
"""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

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
    import kvikio
    import kvikio.cufile
    import cudf
    KVIKIO_AVAILABLE = True
except ImportError:
    KVIKIO_AVAILABLE = False


class KvikIODataForge:
    """
    Streams financial datasets into tensor memory.
    Leverages CUDA 13 GPUDirect Storage on DGX Blackwell or zero-copy Polars/Numpy on laptop CPU.
    """

    def __init__(
        self,
        data_store_dir: str = "d:/TradeJack/data_store",
        use_gds_if_available: bool = True,
        target_device: str = "cuda:0"
    ):
        self.data_store_dir = os.path.abspath(data_store_dir)
        os.makedirs(self.data_store_dir, exist_ok=True)
        
        self.is_blackwell_dgx = (
            use_gds_if_available and 
            KVIKIO_AVAILABLE and 
            TORCH_AVAILABLE and 
            torch.cuda.is_available()
        )
        self.target_device = target_device if self.is_blackwell_dgx else "cpu"
        
        if self.is_blackwell_dgx:
            logger.info(f"Initialized KvikIODataForge in DGX Blackwell Mode (CUDA 13 GPUDirect Storage on {self.target_device}).")
        elif POLARS_AVAILABLE:
            logger.info("Initialized KvikIODataForge in Laptop Development Mode (CPU / Polars active).")
        else:
            logger.info("Initialized KvikIODataForge in Laptop Simulation Mode (CPU / Numpy binary fallback).")

    def load_file_to_tensor(
        self,
        file_path: str,
        columns: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Loads a partition file (`depth.parquet` or `depth.npz`) directly into memory tensors or Numpy arrays.
        """
        if not os.path.exists(file_path):
            return None
            
        if self.is_blackwell_dgx and file_path.endswith(".parquet"):
            try:
                df_gpu = cudf.read_parquet(file_path, columns=columns)
                tensors = {col: torch.as_tensor(df_gpu[col].to_dlpack(), device=self.target_device) for col in df_gpu.columns}
                return tensors
            except Exception as e:
                logger.debug(f"KvikIO cuFile load failed ({e}), falling back to CPU loader.")
                
        if file_path.endswith(".parquet") and POLARS_AVAILABLE:
            try:
                df = pl.read_parquet(file_path, columns=columns)
                result = {}
                for col in df.columns:
                    if TORCH_AVAILABLE and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                        result[col] = torch.from_numpy(df[col].to_numpy())
                    else:
                        result[col] = df[col].to_numpy()
                return result
            except Exception as e:
                logger.error(f"Failed to load Parquet via Polars: {e}")
                return None
                
        # Numpy NPZ binary fallback for laptop without Polars
        if file_path.endswith(".npz"):
            try:
                data = np.load(file_path)
                result = {}
                for col in (columns or data.files):
                    if col in data:
                        arr = data[col].astype(np.float32)
                        result[col] = torch.from_numpy(arr) if TORCH_AVAILABLE else arr
                return result
            except Exception as e:
                logger.error(f"Failed to load NPZ fallback: {e}")
                return None
                
        return None

    def scan_available_partitions(self, symbol: str) -> List[str]:
        """Returns list of all partition file paths available for the symbol in data_store."""
        results = []
        symbol_dir = os.path.join(self.data_store_dir, symbol)
        if not os.path.exists(symbol_dir):
            return results
        for root, dirs, files in os.walk(symbol_dir):
            for file in files:
                if file.endswith(".parquet") or file.endswith(".npz"):
                    results.append(os.path.join(root, file))
        return sorted(results)

    def stream_partition_window(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Iterates across partition directory (`data_store/symbol/year/month/day/depth.*`)
        within [start_date, end_date] and streams tensors/arrays.
        """
        results = []
        symbol_dir = os.path.join(self.data_store_dir, symbol)
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
                            if start_tuple <= (year, month, day) <= end_tuple:
                                t_data = self.load_file_to_tensor(full_path, columns=columns)
                                if t_data:
                                    results.append(t_data)
                        except ValueError:
                            continue
        return results


if __name__ == "__main__":
    forge = KvikIODataForge(data_store_dir="d:/TradeJack/data_store")
    print("Is DGX Blackwell Mode Active:", forge.is_blackwell_dgx)
