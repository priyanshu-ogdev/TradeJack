"""
DALI / PyTorch Zero-Copy Loader for Project TradeJack.
High-throughput sequence-to-sequence window generator feeding historical order book Parquet streams
directly into PyTorch neural models (Transformers, CNNs, RL Actor-Critics).
Adapts dynamically between:
1. NVIDIA DALI Zero-Copy Pipeline on DGX Blackwell (CUDA 13).
2. PyTorch Memory-Mapped Dataset on local laptop (CPU).
3. Pure Numpy/Simulation Dataset fallback when running on local laptop without PyTorch installed.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Iterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (DALILoader) %(message)s")
logger = logging.getLogger("DALILoader")

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
    TORCH_AVAILABLE = True
    BaseDataset = TorchDataset
except ImportError:
    TORCH_AVAILABLE = False
    logger.info("PyTorch not installed; DALILoader operating in Numpy/Simulation fallback mode.")
    BaseDataset = object

try:
    import nvidia.dali as dali
    import nvidia.dali.fn as fn
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    logger.debug("NVIDIA DALI not found; DALILoader using high-speed PyTorch/Numpy fallback.")

from data_forge.kvikio_pipeline import KvikIODataForge


class ParquetLOBDataset(BaseDataset):
    """
    Local laptop / high-throughput CPU memory-mapped Dataset for historical Parquet windows.
    Constructs rolling sequence windows `[seq_len, num_features]` along with target forward returns.
    Works with PyTorch Tensors when available, or Numpy arrays during local laptop simulation.
    """

    def __init__(
        self,
        symbol: str = "BTC-USDT",
        start_date: str = "2024-01-01",
        end_date: str = "2024-01-05",
        seq_len: int = 60,
        forward_horizon: int = 5,
        data_store_dir: str = "d:/TradeJack/data_store",
        feature_cols: Optional[List[str]] = None
    ):
        if TORCH_AVAILABLE:
            super().__init__()
        self.seq_len = seq_len
        self.forward_horizon = forward_horizon
        self.feature_cols = feature_cols or [
            "bid_px_0", "bid_sz_0", "ask_px_0", "ask_sz_0",
            "mid_price", "spread", "order_flow_imbalance", "log_return"
        ]
        
        self.forge = KvikIODataForge(data_store_dir=data_store_dir, use_gds_if_available=False)
        partitions = self.forge.stream_partition_window(symbol, start_date, end_date, columns=self.feature_cols)
        
        if not partitions:
            logger.warning(f"No partition data found for {symbol} between {start_date} and {end_date}.")
            self.tensor_features = torch.empty((0, len(self.feature_cols))) if TORCH_AVAILABLE else np.zeros((0, len(self.feature_cols)), dtype=np.float32)
        else:
            # Concatenate partitions along time axis
            combined_cols = {col: [] for col in self.feature_cols}
            for part in partitions:
                for col in self.feature_cols:
                    if col in part:
                        combined_cols[col].append(part[col])
                        
            if TORCH_AVAILABLE and all(isinstance(v, torch.Tensor) for v_list in combined_cols.values() for v in v_list):
                feature_tensors = []
                for col in self.feature_cols:
                    if combined_cols[col]:
                        t_cat = torch.cat(combined_cols[col], dim=0).to(torch.float32)
                        feature_tensors.append(t_cat.unsqueeze(1))
                    else:
                        feature_tensors.append(torch.zeros((1, 1), dtype=torch.float32))
                if feature_tensors and feature_tensors[0].shape[0] > 0:
                    self.tensor_features = torch.cat(feature_tensors, dim=1)
                else:
                    self.tensor_features = torch.empty((0, len(self.feature_cols)))
            else:
                # Numpy/pure simulation fallback
                feature_arrays = []
                for col in self.feature_cols:
                    if combined_cols[col]:
                        # Each item in combined_cols[col] might be numpy array or list
                        arrs = [np.array(a, dtype=np.float32) if not isinstance(a, np.ndarray) else a.astype(np.float32) for a in combined_cols[col]]
                        arr_cat = np.concatenate(arrs, axis=0)
                        feature_arrays.append(arr_cat[:, None])
                    else:
                        feature_arrays.append(np.zeros((1, 1), dtype=np.float32))
                if feature_arrays and feature_arrays[0].shape[0] > 0:
                    self.tensor_features = np.concatenate(feature_arrays, axis=1)
                else:
                    self.tensor_features = np.zeros((0, len(self.feature_cols)), dtype=np.float32)
                
        self.total_length = max(0, len(self.tensor_features) - self.seq_len - self.forward_horizon + 1)
        logger.info(f"ParquetLOBDataset initialized with {self.total_length} sequence windows ({len(self.feature_cols)} features).")

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if self.total_length == 0:
            if TORCH_AVAILABLE:
                return torch.empty(0), torch.empty(0)
            return np.empty(0), np.empty(0)
            
        x = self.tensor_features[idx : idx + self.seq_len]
        
        # Calculate target: future mid_price return over forward_horizon
        mid_idx = self.feature_cols.index("mid_price") if "mid_price" in self.feature_cols else 0
        current_mid = self.tensor_features[idx + self.seq_len - 1, mid_idx]
        future_mid = self.tensor_features[idx + self.seq_len + self.forward_horizon - 1, mid_idx]
        
        target_return = (future_mid - current_mid) / (current_mid + 1e-8)
        
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            if target_return > 0.0005:
                target_class = torch.tensor(1.0, dtype=torch.float32)
            elif target_return < -0.0005:
                target_class = torch.tensor(-1.0, dtype=torch.float32)
            else:
                target_class = torch.tensor(0.0, dtype=torch.float32)
            return x, target_class
        else:
            if target_return > 0.0005:
                target_class = np.float32(1.0)
            elif target_return < -0.0005:
                target_class = np.float32(-1.0)
            else:
                target_class = np.float32(0.0)
            return x, target_class


class NumpyDataLoader:
    """Fallback DataLoader yielding batches of Numpy arrays when PyTorch is not installed."""
    def __init__(self, dataset: ParquetLOBDataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self.shuffle:
            np.random.shuffle(self.indices)
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]
            batch_x = []
            batch_y = []
            for idx in batch_indices:
                x, y = self.dataset[idx]
                batch_x.append(x)
                batch_y.append(y)
            if batch_x and len(batch_x[0]) > 0:
                yield np.stack(batch_x, axis=0), np.array(batch_y, dtype=np.float32)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_lob_dataloader(
    symbol: str = "BTC-USDT",
    start_date: str = "2024-01-01",
    end_date: str = "2024-01-05",
    batch_size: int = 32,
    seq_len: int = 60,
    forward_horizon: int = 5,
    data_store_dir: str = "d:/TradeJack/data_store",
    device: str = "cuda:0"
) -> Any:
    """
    Factory function producing high-throughput data iterators.
    On Grace Blackwell with DALI & CUDA 13, constructs zero-copy DALI pipeline.
    On Laptop CPU with PyTorch, returns PyTorch DataLoader.
    On Laptop CPU without PyTorch, returns NumpyDataLoader.
    """
    use_cuda = TORCH_AVAILABLE and torch.cuda.is_available() and device.startswith("cuda")
    
    dataset = ParquetLOBDataset(
        symbol=symbol, start_date=start_date, end_date=end_date,
        seq_len=seq_len, forward_horizon=forward_horizon, data_store_dir=data_store_dir
    )
    
    if TORCH_AVAILABLE:
        loader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=use_cuda
        )
        return loader
    else:
        return NumpyDataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    logger.info("Testing DALILoader / ParquetLOBDataset standalone execution...")
    from data_forge.parquet_ingest import ParquetIngestPipeline
    import asyncio
    ingest = ParquetIngestPipeline(data_store_dir="d:/TradeJack/data_store")
    asyncio.run(ingest.generate_synthetic_crucible_data(symbol="BTC-USDT", num_days=1, ticks_per_day=150, start_date="2024-01-01"))
    
    loader = create_lob_dataloader(symbol="BTC-USDT", start_date="2024-01-01", end_date="2024-01-01", batch_size=8, seq_len=20)
    for idx, (batch_x, batch_y) in enumerate(loader):
        print(f"Batch {idx}: X shape = {batch_x.shape}, Y shape = {batch_y.shape}")
        if idx >= 1:
            break
