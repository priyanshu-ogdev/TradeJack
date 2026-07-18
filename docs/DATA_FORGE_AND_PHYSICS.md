# High-Throughput Data Forge & LOB Physics Engine

The `data_forge/` and `physics/` packages bridge the gap between raw multi-level market data and realistic, friction-injected execution simulation.

---

## 1. GPUDirect Storage & KvikIO Pipeline (`kvikio_pipeline.py`)

Traditional quantitative pipelines suffer from severe I/O bottlenecks: reading parquet files from disk into CPU memory (`pandas`), converting them to numpy arrays, and bouncing them across the PCIe bus into GPU registers.

`KvikIODataForge` bypasses CPU memory entirely when running on the Grace Blackwell DGX Spark:
- **GPUDirect Storage (`cufile`)**: Reads partitioned limit order book data (`data_store/symbol/year/month/day/depth.parquet`) straight from NVME solid-state drives into CUDA VRAM tensors at line rate.
- **Zero-Copy Fallback**: On laptop development CPUs, the engine automatically switches to multi-threaded **Polars (`pl.read_parquet`)** or pre-compiled **Numpy binary (`.npz`)** loading (`load_file_to_tensor`), returning standardized feature dictionaries regardless of host OS or hardware.

---

## 2. NVIDIA DALI & Parquet Ingestion (`dali_loader.py`, `parquet_ingest.py`)

- **`ParquetIngestPipeline`**: Organizes tick streams into standardized date partitions (`symbol/YYYY/MM/DD/depth.parquet`). Includes `generate_synthetic_crucible_data()` to instantly generate multi-day LOB depth features for offline testing.
- **`create_lob_dataloader`**: Constructs zero-copy batch iterators (`TorchDataLoader` / `NumpyDataLoader`) with double-buffer queueing (`pin_memory=True`), ensuring neural networks never starve for sequence windows (`seq_len=60`, `forward_horizon=5`).

---

## 3. Exact LOB Slippage & Spread Physics (`lob_env.py`)

Most trading bots assume market orders execute instantly at the mid-price. In reality, large order execution on cryptocurrency exchanges causes immediate slippage and consumes order book liquidity.

`TradeJackLOBEnv` simulates true structural friction across 8 limit order book depth tiers:
- **Depth Matching (`compute_market_order_fill`)**: When a container emits an action vector (`position_qty > 0` for long, `< 0` for short), the physics engine checks the exact volume available at each bid/ask price level (`bid_price_0..7`, `ask_price_0..7`).
- **Slippage & Spread Calculation**: If order volume exceeds the top tier (`qty_0`), the order sweeps deeper tiers (`price_1`, `price_2`, etc.), computing the exact volume-weighted average fill price. The crossing cost between bid and ask is deducted directly from the trade.
- **Exchange Fees**: Applies a realistic maker/taker transaction fee (`0.10%`) to every filled order.

---

## 4. Portfolio Accounting Engine (`portfolio_tracker.py`)

`PortfolioAccountingEngine` maintains atomic, crash-proof record-keeping for every container inside `state/child_{id}/ledger.sqlite`:
- **Real-Time Sharpe & Sortino Ratios**: Continuously updates annualized risk-adjusted returns using rolling window standard deviations of tick-by-tick equity deltas.
- **High-Water Mark & Max Drawdown**: Tracks peak equity attained (`peak_equity`). Computes instantaneous percentage drawdown:

$$\text{Drawdown}_t = \frac{\text{Peak Equity} - \text{Equity}_t}{\text{Peak Equity}}$$

If drawdown breaches dangerous thresholds (`> 15.0%`), the metrics trigger automatic git financial rollbacks or tier downgrades.
