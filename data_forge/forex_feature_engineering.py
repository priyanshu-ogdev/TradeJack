"""
FX Microstructure Feature Engineering — quote-stream analog of feature_engineering.py.

feature_engineering.py's OFI/VPIN/Kyle's Lambda were built for a single trade-print
stream (Binance aggTrades: one price + one side per tick). Dukascopy's forex ticks are
a two-sided quote stream (bid, ask, bid_volume, ask_volume) instead — there is no
"aggressor side" to infer OFI from the same way, and spread is directly observable
instead of needing to be inferred from trade clustering. So this module computes the
FX-appropriate analogs rather than forcing the crypto formulas onto data they don't fit:

  - mid_price, spread, relative_spread   — direct liquidity-cost signal (FX gives this
                                            for free; crypto's OFI/VPIN exist partly
                                            because trade-only feeds don't).
  - quote_imbalance                      — (bid_vol - ask_vol) / (bid_vol + ask_vol),
                                            the direct FX analog of OFI.
  - quote_intensity                      — ticks per minute bucket, a liquidity/session proxy.
  - log_return                           — mid-price log return per bucket.
  - is_gap                               — flags any bucket following a real market closure
                                            (weekend/holiday) so gap-crossing "returns" are
                                            never silently computed across a closed market.

Output: one ZSTD-L3 Parquet file per (pair, day) under processed/{PAIR}/fx_physics/,
validated against ForexPhysicsSchema.
"""

import os
import glob
import logging
from typing import Optional

import numpy as np

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from data_forge.config import config
from data_forge.schema import ForexPhysicsSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (ForexFeatures) %(message)s")
logger = logging.getLogger("ForexFeatures")

# Any gap between consecutive raw ticks longer than this is treated as a real market
# closure (weekend/holiday), not a bucket to forward-fill returns across. FX weekend
# closures run ~48h; anything over 4h during the week is already unusual enough to flag.
DEFAULT_MAX_GAP_SECONDS = 4 * 3600


class ForexFeatureEngine:
    """
    Computes microstructure features from raw Dukascopy tick Parquet files
    (produced by forex_ingest.py) at a configurable resample bucket size.
    """

    def __init__(self, pair: str, bucket_seconds: int = 60, max_gap_seconds: int = DEFAULT_MAX_GAP_SECONDS):
        self.pair = pair.upper()
        self.bucket_seconds = bucket_seconds
        self.max_gap_seconds = max_gap_seconds
        self.raw_dir = os.path.join(config.data_store_dir, "raw", self.pair, "forex_ticks")
        self.processed_dir = os.path.join(config.data_store_dir, "processed", self.pair, "fx_physics")

    def _load_raw_ticks(self, date: str) -> Optional["pl.DataFrame"]:
        date_path = date.replace("-", "/")
        search = os.path.join(self.raw_dir, date_path, "*.parquet")
        files = glob.glob(search)
        if not files:
            logger.warning(f"No raw forex ticks found for {self.pair} {date} at {search}")
            return None
        return pl.read_parquet(files[0])

    def _compute_features(self, df: "pl.DataFrame") -> "pl.DataFrame":
        if df.is_empty():
            return df

        df = df.sort("timestamp").with_columns(
            ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid_price"),
            (pl.col("ask") - pl.col("bid")).alias("spread"),
        )
        df = df.with_columns(
            (pl.col("spread") / np.clip(df["mid_price"].to_numpy(), 1e-10, None)).alias("relative_spread"),
            (
                (pl.col("bid_volume") - pl.col("ask_volume"))
                / (pl.col("bid_volume") + pl.col("ask_volume") + 1e-10)
            ).alias("quote_imbalance"),
        )

        # Flag real market-closure gaps in the raw tick stream BEFORE resampling, so a
        # bucket that starts right after a weekend closure is marked rather than having
        # its log_return silently computed against Friday's last tick.
        gap_seconds = df["timestamp"].diff().dt.total_seconds().fill_null(0.0)
        df = df.with_columns(pl.Series("_pre_gap", gap_seconds > self.max_gap_seconds))

        bucketed = (
            df.group_by_dynamic("timestamp", every=f"{self.bucket_seconds}s")
            .agg(
                pl.col("mid_price").last().alias("mid_price"),
                pl.col("spread").mean().alias("spread"),
                pl.col("relative_spread").mean().alias("relative_spread"),
                pl.col("quote_imbalance").mean().alias("quote_imbalance"),
                pl.count().alias("quote_intensity"),
                pl.col("_pre_gap").any().alias("is_gap"),
            )
            .sort("timestamp")
        )

        mid = np.clip(bucketed["mid_price"].to_numpy(), 1e-10, None)
        log_return = np.concatenate([[np.nan], np.diff(np.log(mid))])
        # A bucket flagged is_gap has its log_return nulled — it would otherwise be a
        # return computed across a real market closure (e.g. Friday close -> Sunday open),
        # which is not tradeable price action and would poison training if left in.
        is_gap = bucketed["is_gap"].to_numpy()
        log_return = np.where(is_gap, np.nan, log_return)

        bucketed = bucketed.with_columns(pl.Series("log_return", log_return))
        return bucketed.select(
            ["timestamp", "mid_price", "spread", "relative_spread", "quote_imbalance",
             "quote_intensity", "log_return", "is_gap"]
        )

    def process_daily_file(self, date: str) -> str:
        if not POLARS_AVAILABLE:
            logger.error("Polars required for forex feature engineering.")
            return ""

        raw = self._load_raw_ticks(date)
        if raw is None or raw.is_empty():
            logger.info(f"No FX ticks to process for {self.pair} {date} (likely a weekend/holiday).")
            return ""

        features = self._compute_features(raw)
        if features.is_empty():
            return ""

        try:
            ForexPhysicsSchema.validate(features)
        except Exception as e:
            logger.error(f"FX features failed schema validation for {self.pair} {date}: {e}")
            return ""

        out_dir = os.path.join(self.processed_dir, date.replace("-", "/"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{self.pair}-fx_physics-{date}.parquet")

        features.write_parquet(
            out_path,
            compression=config.compression_codec,
            compression_level=config.compression_level,
            row_group_size=config.row_group_size,
        )
        logger.info(f"FX features -> {out_path} ({len(features)} buckets, {int(features['is_gap'].sum())} gap-flagged)")
        return out_path


if __name__ == "__main__":
    engine = ForexFeatureEngine("EURUSD")
    engine.process_daily_file("2024-01-02")
