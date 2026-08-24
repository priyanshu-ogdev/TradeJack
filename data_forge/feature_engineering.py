"""
GPU-Accelerated Feature Engineering (Trade-Flow Physics) with Volume-Gated MAD.
Calculates Institutional Quant metrics (VPIN, OFI, Kyle's Lambda) directly from Binance Vision `aggTrades`.
Leverages RAPIDS cuDF for massive bare-metal GPU acceleration, completely eliminating CPU bottlenecks.
Includes a Polars fallback for local development environments where cuDF is unavailable.
Ensures atomic writing to prevent race conditions with the DVC/KvikIO streamer.
Implements Stateful Prepending (Dynamic Lookback) to fix the Midnight Reset Bug.

SOTA Upgrades:
  - Unified ZSTD-L3 compression across both cuDF and Polars paths
  - Automatic Binance 2025 μs timestamp detection (ms→μs cutover)
  - Fixed None typing for Polars strict mode compatibility
  - Configurable row_group_size for optimal partition pruning
"""

import os
import glob
import logging
import uuid
import numpy as np
import json

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    try:
        import polars as pl
        POLARS_AVAILABLE = True
    except ImportError:
        POLARS_AVAILABLE = False

from data_forge.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (FeatureEngineering) %(message)s")
logger = logging.getLogger("FeatureEngineering")


def _detect_timestamp_unit(sample_value: int) -> str:
    """
    Auto-detects whether a Binance timestamp is in milliseconds or microseconds.
    Binance SPOT data switched from ms to μs on 2025-01-01.
    Values > 1e15 are microseconds; values in the 1e12-1e14 range are milliseconds.
    """
    if sample_value > 1e15:
        return "us"
    return "ms"


class TradeFlowPhysics:
    def __init__(self, symbol: str):
        self.symbol = symbol.replace("-", "").upper()
        self.raw_dir = os.path.join(config.data_store_dir, "raw", symbol, "aggTrades")
        self.processed_dir = os.path.join(config.data_store_dir, "processed", symbol, "physics")
        self.tmp_dir = os.path.join(config.data_store_dir, ".tmp")
        self.state_file = os.path.join(self.processed_dir, "state.json")
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

        if CUDF_AVAILABLE:
            logger.info("RAPIDS cuDF is available. Operating in Bare-Metal GPU Physics Mode.")
        else:
            logger.warning("RAPIDS cuDF not found. Falling back to Polars (CPU) for local simulation.")

    def _get_previous_remainder(self, date: str = None) -> float:
        """
        BUG-17 FIX: Returns volume remainder keyed to the date PRIOR to the given date.
        Old code used a single global key — reprocessing any day out of order would pick up
        today's remainder instead of that day's, silently shifting bucket boundaries.
        """
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                if date:
                    # Compute previous date key: YYYY-MM-DD minus 1 day
                    from datetime import datetime, timedelta
                    try:
                        prev_date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
                        key = f"{self.symbol}/{prev_date}"
                        return float(state.get(key, {}).get("volume_remainder", 0.0))
                    except ValueError:
                        pass
                # Fallback: legacy global key for backward compat
                return float(state.get("volume_remainder", 0.0))
            except Exception as e:
                logger.error(f"Failed to read state.json: {e}")
        return 0.0

    def _save_remainder(self, date: str, remainder: float):
        """
        BUG-17 FIX: Saves volume remainder keyed per (symbol, date).
        Format: {"BTC-USDT/2024-01-01": {"volume_remainder": 3.7}, ...}
        """
        state = {}
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
            except Exception:
                pass
        if date:
            key = f"{self.symbol}/{date}"
            state[key] = {"volume_remainder": float(remainder)}
        else:
            state["volume_remainder"] = float(remainder)  # backward compat
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Failed to write state.json: {e}")

    def _get_dynamic_lookback_path(self, current_output_path: str) -> str:
        all_parquets = glob.glob(os.path.join(self.processed_dir, "**", "*.parquet"), recursive=True)
        valid_parquets = [p for p in all_parquets if os.path.abspath(p) != os.path.abspath(current_output_path)]
        if not valid_parquets:
            return None
        return sorted(valid_parquets)[-1]

    def _process_with_cudf(self, file_path: str, output_path: str, volume_bucket_size: float = 10.0, date: str = None):
        logger.debug(f"Loading {file_path} into cuDF...")
        df = cudf.read_csv(file_path, names=["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker", "is_best_match"])

        prev_remainder = self._get_previous_remainder(date)

        # Auto-detect timestamp unit (ms vs μs for Binance 2025+ data)
        sample_ts = int(df['transact_time'].iloc[0])
        ts_unit = _detect_timestamp_unit(sample_ts)
        logger.debug(f"Detected timestamp unit: {ts_unit} (sample: {sample_ts})")

        # Strict Epoch Timestamp Normalization
        df['transact_time'] = cudf.to_datetime(df['transact_time'], unit=ts_unit)

        df['trade_direction'] = cudf.where(df['is_buyer_maker'], -1, 1)
        df['signed_volume'] = df['quantity'] * df['trade_direction']
        df['cum_volume'] = df['quantity'].cumsum() + prev_remainder
        df['bucket_id'] = (df['cum_volume'] // volume_bucket_size).astype('int32')

        new_remainder = df['cum_volume'].iloc[-1] % volume_bucket_size
        self._save_remainder(date, new_remainder)

        bucket_df = df.groupby('bucket_id').agg({
            'quantity': 'sum',
            'signed_volume': 'sum',
            'price': ['first', 'last'],
            'transact_time': 'last'
        })
        bucket_df.columns = ['volume', 'order_flow_imbalance', 'open_price', 'close_price', 'timestamp']
        bucket_df = bucket_df.reset_index().drop(columns=['bucket_id'])

        # Dynamic Lookback
        lookback_path = self._get_dynamic_lookback_path(output_path)
        seed_max_ts = None
        if lookback_path:
            logger.debug(f"Pre-pending Dynamic Lookback state from {lookback_path}")
            seed_df = cudf.read_parquet(lookback_path).tail(100)
            seed_base = seed_df[['volume', 'ofi', 'open_price', 'close_price', 'timestamp']]
            seed_base.columns = ['volume', 'order_flow_imbalance', 'open_price', 'close_price', 'timestamp']
            seed_max_ts = seed_base['timestamp'].max()
            bucket_df = cudf.concat([seed_base, bucket_df], ignore_index=True)

        # Strict Monotonic Enforcement
        bucket_df = bucket_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

        # Volume-Gated MAD Filter
        bucket_df['rolling_price_med'] = bucket_df['close_price'].rolling(window=100, min_periods=1).median()
        bucket_df['rolling_vol_med'] = bucket_df['volume'].rolling(window=100, min_periods=1).median()
        bucket_df['price_mad'] = (bucket_df['close_price'] - bucket_df['rolling_price_med']).abs().rolling(window=100, min_periods=1).median()

        bucket_df['z_score'] = (bucket_df['close_price'] - bucket_df['rolling_price_med']) / (bucket_df['price_mad'] * 1.4826 + 1e-8)
        is_glitch = (bucket_df['z_score'].abs() > 5.0) & (bucket_df['volume'] < bucket_df['rolling_vol_med'])

        bucket_df['close_price'] = cudf.where(is_glitch, None, bucket_df['close_price']).ffill()
        bucket_df['open_price'] = cudf.where(is_glitch, None, bucket_df['open_price']).ffill()

        # VPIN, OFI, Kyle's Lambda — BUG-16 FIX: windowed OLS regression
        # Old: price_change / (ofi + 1e-8) — OFI near-zero/negative produces huge sign-flipping lambda
        # New: rolling covariance(price_change, ofi) / variance(ofi) over 50-bucket window
        # with a minimum variance floor scaled to typical bucket volume (not fixed epsilon)
        bucket_df['abs_imbalance'] = bucket_df['order_flow_imbalance'].abs()
        bucket_df['vpin_50'] = bucket_df['abs_imbalance'].rolling(window=50, min_periods=1).mean() / bucket_df['volume'].rolling(window=50, min_periods=1).mean()
        bucket_df['ofi'] = bucket_df['order_flow_imbalance']
        bucket_df['price_change'] = bucket_df['close_price'] - bucket_df['open_price']
        # Rolling OLS numerator: cov(price_change, ofi) approximated as rolling mean of product minus product of means
        w = 50
        pc = bucket_df['price_change']
        ofi_col = bucket_df['ofi']
        roll_mean_pc = pc.rolling(window=w, min_periods=1).mean()
        roll_mean_ofi = ofi_col.rolling(window=w, min_periods=1).mean()
        roll_mean_pc_ofi = (pc * ofi_col).rolling(window=w, min_periods=1).mean()
        roll_cov = roll_mean_pc_ofi - roll_mean_pc * roll_mean_ofi
        roll_var_ofi = (ofi_col ** 2).rolling(window=w, min_periods=1).mean() - roll_mean_ofi ** 2
        # Volume-scaled variance floor: prevents explosion in balanced-flow buckets
        vol_floor = bucket_df['volume'].rolling(window=w, min_periods=1).median() * 0.01
        vol_floor = vol_floor.clip(lower=1e-6)
        safe_var = cudf.where(roll_var_ofi.abs() > vol_floor, roll_var_ofi.abs(), vol_floor)
        bucket_df['kyles_lambda'] = roll_cov / safe_var

        # Slice off Dynamic Lookback Seed
        if seed_max_ts is not None:
            bucket_df = bucket_df[bucket_df['timestamp'] > seed_max_ts]

        final_df = bucket_df[['timestamp', 'open_price', 'close_price', 'volume', 'ofi', 'vpin_50', 'kyles_lambda']]

        # Atomic Partitioning — ZSTD compression unified across GPU path
        tmp_path = os.path.join(self.tmp_dir, f"{uuid.uuid4().hex}.parquet")
        final_df.to_parquet(tmp_path, engine='cudf', compression='zstd')
        os.replace(tmp_path, output_path)
        logger.info(f"Saved cuDF processed features to {output_path}")

    def _process_with_polars(self, file_path: str, output_path: str, volume_bucket_size: float = 10.0, date: str = None):
        logger.debug(f"Loading {file_path} into Polars...")
        df = pl.read_csv(file_path, has_header=False, new_columns=["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker", "is_best_match"])

        prev_remainder = self._get_previous_remainder(date)

        # Auto-detect timestamp unit (ms vs μs for Binance 2025+ data)
        sample_ts = int(df["transact_time"][0])
        ts_unit = _detect_timestamp_unit(sample_ts)
        logger.debug(f"Detected timestamp unit: {ts_unit} (sample: {sample_ts})")

        # Strict Epoch Timestamp Normalization with auto-detected unit
        df = df.with_columns(pl.from_epoch(pl.col("transact_time"), time_unit=ts_unit))

        df = df.with_columns(pl.when(pl.col("is_buyer_maker")).then(-1).otherwise(1).alias("trade_direction"))
        df = df.with_columns([
            (pl.col("quantity") * pl.col("trade_direction")).alias("signed_volume"),
            (pl.col("quantity").cum_sum() + prev_remainder).alias("cum_volume")
        ])

        df = df.with_columns((pl.col("cum_volume") // volume_bucket_size).cast(pl.Int32).alias("bucket_id"))

        new_remainder = df.select(pl.col("cum_volume")).tail(1).item() % volume_bucket_size
        self._save_remainder(date, new_remainder)

        bucket_df = df.group_by("bucket_id").agg([
            pl.col("quantity").sum().alias("volume"),
            pl.col("signed_volume").sum().alias("ofi"),
            pl.col("price").first().alias("open_price"),
            pl.col("price").last().alias("close_price"),
            pl.col("transact_time").last().alias("timestamp")
        ]).sort("bucket_id")

        # Dynamic Lookback
        lookback_path = self._get_dynamic_lookback_path(output_path)
        seed_max_ts = None
        if lookback_path:
            logger.debug(f"Pre-pending Dynamic Lookback state from {lookback_path}")
            seed_df = pl.read_parquet(lookback_path).tail(100)
            seed_base = seed_df.select(['volume', 'ofi', 'open_price', 'close_price', 'timestamp'])
            seed_base = seed_base.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))
            seed_max_ts = seed_base.select(pl.col('timestamp')).tail(1).item()
            bucket_df = bucket_df.drop("bucket_id")
            bucket_df = bucket_df.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))
            bucket_df = pl.concat([seed_base, bucket_df])
        else:
            bucket_df = bucket_df.drop("bucket_id")
            bucket_df = bucket_df.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))

        # Strict Monotonic Enforcement
        bucket_df = bucket_df.sort("timestamp").unique(subset=["timestamp"], keep="last")

        # Volume-Gated MAD Filter
        bucket_df = bucket_df.with_columns([
            pl.col("close_price").rolling_median(window_size=100, min_samples=1).alias("rolling_price_med"),
            pl.col("volume").rolling_median(window_size=100, min_samples=1).alias("rolling_vol_med")
        ])

        bucket_df = bucket_df.with_columns(
            (pl.col("close_price") - pl.col("rolling_price_med")).abs().rolling_median(window_size=100, min_samples=1).alias("price_mad")
        )

        bucket_df = bucket_df.with_columns(
            ((pl.col("close_price") - pl.col("rolling_price_med")) / (pl.col("price_mad") * 1.4826 + 1e-8)).alias("z_score")
        )

        is_glitch = (pl.col("z_score").abs() > 5.0) & (pl.col("volume") < pl.col("rolling_vol_med"))
        # Fixed None typing: explicit dtype prevents Polars strict mode errors
        bucket_df = bucket_df.with_columns([
            pl.when(is_glitch).then(pl.lit(None, dtype=pl.Float64)).otherwise(pl.col("close_price")).forward_fill().alias("close_price"),
            pl.when(is_glitch).then(pl.lit(None, dtype=pl.Float64)).otherwise(pl.col("open_price")).forward_fill().alias("open_price")
        ])

        # BUG-16 FIX: Replace naive division with windowed OLS regression for Kyle's Lambda
        # Old: price_change / (ofi + 1e-8) explodes when ofi is near-zero or negative
        # New: rolling cov(price_change, ofi) / var(ofi) over 50-bucket window
        # Variance floor scaled to median bucket volume to prevent explosion in balanced-flow regimes
        w = 50
        bucket_df = bucket_df.with_columns([
            pl.col("ofi").abs().alias("abs_imbalance"),
            (pl.col("close_price") - pl.col("open_price")).alias("price_change")
        ])

        bucket_df = bucket_df.with_columns([
            (pl.col("abs_imbalance").rolling_mean(window_size=w, min_samples=1) /
             pl.col("volume").rolling_mean(window_size=w, min_samples=1)).alias("vpin_50"),
            # Rolling OLS numerator: E[pc*ofi] - E[pc]*E[ofi]
            (pl.col("price_change") * pl.col("ofi")).rolling_mean(window_size=w, min_samples=1).alias("_mean_pc_ofi"),
            pl.col("price_change").rolling_mean(window_size=w, min_samples=1).alias("_mean_pc"),
            pl.col("ofi").rolling_mean(window_size=w, min_samples=1).alias("_mean_ofi"),
            # Rolling OLS denominator: E[ofi^2] - E[ofi]^2
            (pl.col("ofi") ** 2).rolling_mean(window_size=w, min_samples=1).alias("_mean_ofi2"),
            # Volume-scaled floor
            pl.col("volume").rolling_median(window_size=w, min_samples=1).alias("_vol_med")
        ])

        bucket_df = bucket_df.with_columns([
            (pl.col("_mean_pc_ofi") - pl.col("_mean_pc") * pl.col("_mean_ofi")).alias("_roll_cov"),
            ((pl.col("_mean_ofi2") - pl.col("_mean_ofi") ** 2).abs()).alias("_roll_var_ofi"),
            (pl.col("_vol_med") * 0.01).clip(lower_bound=1e-6).alias("_var_floor")
        ])

        bucket_df = bucket_df.with_columns(
            (pl.col("_roll_cov") /
             pl.when(pl.col("_roll_var_ofi") > pl.col("_var_floor"))
             .then(pl.col("_roll_var_ofi"))
             .otherwise(pl.col("_var_floor"))
            ).alias("kyles_lambda")
        )

        # Drop intermediate computation columns
        bucket_df = bucket_df.drop(["_mean_pc_ofi", "_mean_pc", "_mean_ofi", "_mean_ofi2", "_vol_med", "_roll_cov", "_roll_var_ofi", "_var_floor"])

        # Slice off Dynamic Lookback Seed
        if seed_max_ts is not None:
            bucket_df = bucket_df.filter(pl.col("timestamp") > seed_max_ts)

        final_df = bucket_df.select(['timestamp', 'open_price', 'close_price', 'volume', 'ofi', 'vpin_50', 'kyles_lambda'])

        # Atomic Partitioning — ZSTD-L3 with optimal row group size
        tmp_path = os.path.join(self.tmp_dir, f"{uuid.uuid4().hex}.parquet")
        final_df.write_parquet(
            tmp_path,
            compression=config.compression_codec,
            compression_level=config.compression_level,
            row_group_size=config.row_group_size,
        )
        os.replace(tmp_path, output_path)
        logger.info(f"Saved Polars processed features to {output_path}")

    def process_daily_file(self, date: str):
        date_path = date.replace("-", "/")
        # Search for both raw CSV and pre-converted Parquet files
        search_csv = os.path.join(self.raw_dir, date_path, "*.csv")
        search_pq = os.path.join(self.raw_dir, date_path, "*.parquet")
        files = glob.glob(search_csv) + glob.glob(search_pq)

        if not files:
            logger.warning(f"No aggTrades found for {self.symbol} on {date} in {self.raw_dir}")
            return

        file_path = files[0]
        out_dir = os.path.join(self.processed_dir, date_path)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "physics.parquet")

        if CUDF_AVAILABLE:
            self._process_with_cudf(file_path, out_path, date=date)
        else:
            self._process_with_polars(file_path, out_path, date=date)

if __name__ == "__main__":
    physics_engine = TradeFlowPhysics("BTC-USDT")
    physics_engine.process_daily_file("2024-01-01")
    physics_engine.process_daily_file("2024-01-02")
    logger.info("Feature Engineering Complete.")
