"""
Parquet Ingest Pipeline: Asyncio & Polars/Numpy Processing Engine for Project TradeJack.
Cleans, aligns, computes quantitative order flow imbalances, and compresses historical depth snapshots
into partition-aligned files (`symbol/YYYY/MM/DD/depth.parquet` or `depth.npz`).
Works cleanly in local laptop simulation mode or high-throughput DGX server mode.
"""

import os
import sys
import time
import math
import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (ParquetIngest) %(message)s")
logger = logging.getLogger("ParquetIngest")

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    logger.info("Polars not installed; ParquetIngest operating in high-performance Numpy NPZ fallback mode.")


class ParquetIngestPipeline:
    """
    High-speed historical data transformation and partitioning engine.
    """

    def __init__(self, data_store_dir: str = "d:/TradeJack/data_store"):
        self.data_store_dir = os.path.abspath(data_store_dir)
        os.makedirs(self.data_store_dir, exist_ok=True)

    def save_partition_numpy(self, symbol: str, year: int, month: int, day: int, data_dict: Dict[str, np.ndarray]) -> str:
        """Saves a day's partition as compressed Numpy (.npz) file when Polars is not available."""
        part_dir = os.path.join(self.data_store_dir, symbol, f"{year:04d}", f"{month:02d}", f"{day:02d}")
        os.makedirs(part_dir, exist_ok=True)
        out_path = os.path.join(part_dir, "depth.npz")
        np.savez_compressed(out_path, **data_dict)
        logger.debug(f"Wrote partition NPZ ({len(data_dict.get('timestamp', []))} rows): {out_path}")
        return out_path

    async def generate_synthetic_crucible_data(
        self,
        symbol: str = "BTC-USDT",
        num_days: int = 3,
        ticks_per_day: int = 150,
        start_date: str = "2024-01-01",
        base_price: float = 65000.0,
        volatility: float = 0.02
    ) -> List[str]:
        """
        Async generator creating high-precision synthetic Limit Order Book data for local laptop testing
        and initial Genesis Crucible validation.
        """
        logger.info(f"Generating {num_days} days of realistic LOB depth data for {symbol} ({ticks_per_day} ticks/day)...")
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        all_written = []
        
        current_price = base_price
        import random
        random.seed(42)
        
        for d in range(num_days):
            day_dt = start_dt + timedelta(days=d)
            timestamps = []
            bid_px_0, bid_sz_0 = [], []
            ask_px_0, ask_sz_0 = [], []
            bid_px_1, bid_sz_1 = [], []
            ask_px_1, ask_sz_1 = [], []
            volumes = []
            
            is_flash_crash_day = (d == 1)
            
            for t in range(ticks_per_day):
                ts = (day_dt + timedelta(seconds=t * (86400 / ticks_per_day))).timestamp()
                timestamps.append(ts)
                
                shock = random.gauss(0, volatility * current_price / math.sqrt(ticks_per_day))
                if is_flash_crash_day and 50 <= t <= 70:
                    shock -= current_price * 0.003
                    spread_ratio = 0.005
                    depth_mult = 0.1
                else:
                    spread_ratio = 0.0002
                    depth_mult = 1.0
                    
                current_price = max(100.0, current_price + shock)
                spread = current_price * spread_ratio
                
                bp0 = current_price - spread / 2.0
                ap0 = current_price + spread / 2.0
                bs0 = max(0.1, random.expovariate(1.0 / (10.0 * depth_mult)))
                as0 = max(0.1, random.expovariate(1.0 / (10.0 * depth_mult)))
                
                bp1 = bp0 - current_price * 0.0005
                ap1 = ap0 + current_price * 0.0005
                bs1 = max(0.5, random.expovariate(1.0 / (25.0 * depth_mult)))
                as1 = max(0.5, random.expovariate(1.0 / (25.0 * depth_mult)))
                
                bid_px_0.append(bp0)
                bid_sz_0.append(bs0)
                ask_px_0.append(ap0)
                ask_sz_0.append(as0)
                bid_px_1.append(bp1)
                bid_sz_1.append(bs1)
                ask_px_1.append(ap1)
                ask_sz_1.append(as1)
                volumes.append(bs0 + as0 + random.uniform(1.0, 50.0))
                
            arr_ts = np.array(timestamps, dtype=np.float64)
            arr_bp0 = np.array(bid_px_0, dtype=np.float32)
            arr_bs0 = np.array(bid_sz_0, dtype=np.float32)
            arr_ap0 = np.array(ask_px_0, dtype=np.float32)
            arr_as0 = np.array(ask_sz_0, dtype=np.float32)
            arr_bp1 = np.array(bid_px_1, dtype=np.float32)
            arr_bs1 = np.array(bid_sz_1, dtype=np.float32)
            arr_ap1 = np.array(ask_px_1, dtype=np.float32)
            arr_as1 = np.array(ask_sz_1, dtype=np.float32)
            arr_vol = np.array(volumes, dtype=np.float32)
            
            # Compute quantitative features
            arr_mid = (arr_bp0 + arr_ap0) / 2.0
            arr_spread = arr_ap0 - arr_bp0
            arr_imbalance = (arr_bs0 - arr_as0) / (arr_bs0 + arr_as0 + 1e-8)
            arr_log_ret = np.zeros_like(arr_mid)
            arr_log_ret[1:] = np.log(arr_mid[1:] / (arr_mid[:-1] + 1e-8))
            
            data_dict = {
                "timestamp": arr_ts,
                "bid_px_0": arr_bp0, "bid_sz_0": arr_bs0,
                "ask_px_0": arr_ap0, "ask_sz_0": arr_as0,
                "bid_px_1": arr_bp1, "bid_sz_1": arr_bs1,
                "ask_px_1": arr_ap1, "ask_sz_1": arr_as1,
                "volume": arr_vol,
                "mid_price": arr_mid,
                "spread": arr_spread,
                "order_flow_imbalance": arr_imbalance,
                "log_return": arr_log_ret
            }
            
            if POLARS_AVAILABLE:
                df = pl.DataFrame(data_dict)
                part_dir = os.path.join(self.data_store_dir, symbol, f"{day_dt.year:04d}", f"{day_dt.month:02d}", f"{day_dt.day:02d}")
                os.makedirs(part_dir, exist_ok=True)
                out_path = os.path.join(part_dir, "depth.parquet")
                df.write_parquet(out_path)
                all_written.append(out_path)
            else:
                out_path = self.save_partition_numpy(symbol, day_dt.year, day_dt.month, day_dt.day, data_dict)
                all_written.append(out_path)
                
            await asyncio.sleep(0.01)
            
        logger.info(f"Synthetic Data Generation Complete across {len(all_written)} partition files.")
        return all_written


if __name__ == "__main__":
    logger.info("Testing ParquetIngestPipeline standalone execution...")
    pipeline = ParquetIngestPipeline(data_store_dir="d:/TradeJack/data_store")
    asyncio.run(pipeline.generate_synthetic_crucible_data(symbol="BTC-USDT", num_days=3, ticks_per_day=100))
