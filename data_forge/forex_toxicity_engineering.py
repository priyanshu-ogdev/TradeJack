"""
FX Toxicity / Price-Impact Feature Engineering — BVC-based VPIN & Kyle's Lambda.

See docs/DATA_FORGE_FX_TOXICITY_PLAN.md for the full design rationale. Short version:

Dukascopy ticks are quotes (bid, ask, bid_volume, ask_volume) with NO trade-direction
tag, unlike Binance aggTrades (which has `is_buyer_maker`). feature_engineering.py's
VPIN/Kyle's Lambda both start from signed trade volume, which doesn't exist here — so
this module does NOT reuse quote_imbalance (a *quoted-depth* signal) as a stand-in for
signed *traded* flow. Instead it applies Bulk Volume Classification (BVC; Easley,
Lopez de Prado & O'Hara 2012), the standard technique for inferring buy/sell volume
from price and volume alone when no trade-direction tag is available:

    z_t        = price_change_t / rolling_std(price_change, window=W)
    buy_frac_t = Phi(z_t)                      # standard normal CDF
    buy_volume_t  = volume_t * buy_frac_t
    sell_volume_t = volume_t - buy_volume_t

`volume_t` = bid_volume + ask_volume per tick — quoted size, not executed size
(Dukascopy's free tick feed doesn't publish executed size either; this is a real
data-ceiling limitation, not a modeling shortcut).

Bucketing is by VOLUME, not wall-clock time (a different axis from
forex_feature_engineering.py's time-bucketed spread/quote_imbalance features) —
VPIN's defining property is that bucket size adapts to how busy the market is, which
is lost if you bucket by clock time instead.

The rolling-OLS Kyle's Lambda formula (cov(price_change, flow) / var(flow) with a
volume-scaled variance floor) is the same numerically-stabilized shape already
reviewed and hardened in feature_engineering.py (its BUG-16 fix) — reused here for
consistency and because that formula was already proven against near-zero-variance
blowups, just fed BVC-classified flow instead of signed trade volume.

Output: processed/{PAIR}/fx_toxicity/YYYY/MM/DD/, validated against
ForexToxicitySchema. Meant to be joined onto the time-bucketed fx_physics table via
time_alignment.py (TimeAligner), not merged inside this module — see the design doc
for why keeping the two bucketing axes separate matters.
"""

import os
import glob
import math
import logging
from typing import Optional

import numpy as np

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from data_forge.config import config
from data_forge.schema import ForexToxicitySchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (ForexToxicity) %(message)s")
logger = logging.getLogger("ForexToxicity")

_SQRT2 = math.sqrt(2.0)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF via math.erf (stdlib — no scipy dependency needed for this)."""
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / _SQRT2))


class ForexToxicityEngine:
    """
    Computes BVC-based VPIN and Kyle's Lambda analogs from raw Dukascopy tick Parquet
    files (produced by forex_ingest.py), volume-bucketed rather than time-bucketed.
    """

    def __init__(self, pair: str, volume_bucket_size: float = 1_000_000.0, std_window: int = 50, rolling_window: int = 50):
        """
        volume_bucket_size: cumulative quoted volume per bucket, in the same units as
            Dukascopy's bid_volume/ask_volume (millions of base currency). 1e6 is a
            reasonable starting point for a major pair; illiquid pairs/sessions should
            use a smaller bucket size or they'll rarely complete a bucket.
        std_window: tick-level rolling window for the BVC price-change std. Needs to be
            long enough that z_t isn't dominated by one or two noisy ticks.
        rolling_window: bucket-level rolling window for bvc_vpin / kyles_lambda / amihud,
            mirroring feature_engineering.py's default (w=50).
        """
        self.pair = pair.upper()
        self.volume_bucket_size = volume_bucket_size
        self.std_window = std_window
        self.rolling_window = rolling_window
        self.raw_dir = os.path.join(config.data_store_dir, "raw", self.pair, "forex_ticks")
        self.processed_dir = os.path.join(config.data_store_dir, "processed", self.pair, "fx_toxicity")

    def _load_raw_ticks(self, date: str) -> Optional["pl.DataFrame"]:
        date_path = date.replace("-", "/")
        files = glob.glob(os.path.join(self.raw_dir, date_path, "*.parquet"))
        if not files:
            logger.warning(f"No raw forex ticks found for {self.pair} {date}")
            return None
        return pl.read_parquet(files[0])

    def _classify_bvc(self, df: "pl.DataFrame") -> "pl.DataFrame":
        """Adds tick-level volume, price_change, and BVC-classified buy_volume/sell_volume."""
        df = df.sort("timestamp").with_columns(
            ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid_price"),
            (pl.col("bid_volume") + pl.col("ask_volume")).alias("tick_volume"),
        )
        mid = df["mid_price"].to_numpy()
        price_change = np.concatenate([[0.0], np.diff(mid)])

        rolling_std = (
            pl.Series("price_change", price_change)
            .rolling_std(window_size=self.std_window, min_samples=5)
            .fill_null(0.0)
            .to_numpy()
        )
        # Ticks with no meaningful recent volatility (std ~ 0, e.g. right at the start of
        # a session) get an even 50/50 split rather than a division-by-near-zero blowup.
        safe_std = np.where(rolling_std > 1e-12, rolling_std, np.inf)
        z = price_change / safe_std
        buy_frac = _normal_cdf(z)

        tick_volume = df["tick_volume"].to_numpy()
        buy_volume = tick_volume * buy_frac
        sell_volume = tick_volume - buy_volume

        return df.with_columns(
            pl.Series("price_change", price_change),
            pl.Series("buy_volume", buy_volume),
            pl.Series("sell_volume", sell_volume),
        )

    def _bucket_and_compute(self, df: "pl.DataFrame") -> "pl.DataFrame":
        cum_volume = df["tick_volume"].cum_sum()
        bucket_id = (cum_volume // self.volume_bucket_size).cast(pl.Int64)
        df = df.with_columns(bucket_id.alias("bucket_id"))

        bucket_df = df.group_by("bucket_id").agg(
            pl.col("timestamp").last().alias("timestamp"),
            pl.col("tick_volume").sum().alias("volume"),
            pl.col("buy_volume").sum().alias("buy_volume"),
            pl.col("sell_volume").sum().alias("sell_volume"),
            pl.col("mid_price").first().alias("open_mid"),
            pl.col("mid_price").last().alias("close_mid"),
        ).sort("bucket_id")

        bucket_df = bucket_df.with_columns(
            (pl.col("close_mid") - pl.col("open_mid")).alias("price_change"),
            (pl.col("buy_volume") - pl.col("sell_volume")).alias("flow"),
        )

        w = self.rolling_window
        bucket_df = bucket_df.with_columns(
            pl.col("flow").abs().alias("abs_flow"),
        )
        bucket_df = bucket_df.with_columns(
            (
                pl.col("abs_flow").rolling_mean(window_size=w, min_samples=1)
                / (pl.col("volume").rolling_mean(window_size=w, min_samples=1) + 1e-10)
            ).alias("bvc_vpin"),
            (pl.col("price_change").abs() / (pl.col("volume") + 1e-10))
            .rolling_mean(window_size=w, min_samples=1)
            .alias("amihud_illiquidity"),
            (pl.col("price_change") * pl.col("flow")).rolling_mean(window_size=w, min_samples=1).alias("_mean_pc_flow"),
            pl.col("price_change").rolling_mean(window_size=w, min_samples=1).alias("_mean_pc"),
            pl.col("flow").rolling_mean(window_size=w, min_samples=1).alias("_mean_flow"),
            (pl.col("flow") ** 2).rolling_mean(window_size=w, min_samples=1).alias("_mean_flow2"),
            pl.col("volume").rolling_median(window_size=w, min_samples=1).alias("_vol_med"),
        )
        bucket_df = bucket_df.with_columns(
            (pl.col("_mean_pc_flow") - pl.col("_mean_pc") * pl.col("_mean_flow")).alias("_roll_cov"),
            ((pl.col("_mean_flow2") - pl.col("_mean_flow") ** 2).abs()).alias("_roll_var_flow"),
            (pl.col("_vol_med") * 0.01).clip(lower_bound=1e-6).alias("_var_floor"),
        )
        bucket_df = bucket_df.with_columns(
            (
                pl.col("_roll_cov")
                / pl.when(pl.col("_roll_var_flow") > pl.col("_var_floor"))
                .then(pl.col("_roll_var_flow"))
                .otherwise(pl.col("_var_floor"))
            ).alias("kyles_lambda")
        )

        bucket_df = bucket_df.drop(
            ["_mean_pc_flow", "_mean_pc", "_mean_flow", "_mean_flow2", "_vol_med", "_roll_cov", "_roll_var_flow", "_var_floor", "abs_flow", "open_mid", "close_mid", "flow", "bucket_id"]
        )
        return bucket_df.select(
            ["timestamp", "volume", "buy_volume", "sell_volume", "price_change", "bvc_vpin", "kyles_lambda", "amihud_illiquidity"]
        )

    def process_daily_file(self, date: str) -> str:
        if not POLARS_AVAILABLE:
            logger.error("Polars required for forex toxicity feature engineering.")
            return ""

        raw = self._load_raw_ticks(date)
        if raw is None or raw.is_empty():
            logger.info(f"No FX ticks to process for {self.pair} {date} (likely a weekend/holiday).")
            return ""

        classified = self._classify_bvc(raw)
        buckets = self._bucket_and_compute(classified)
        if buckets.is_empty():
            logger.info(f"No complete volume bucket for {self.pair} {date} — raw volume below bucket size.")
            return ""

        try:
            ForexToxicitySchema.validate(buckets)
        except Exception as e:
            logger.error(f"FX toxicity features failed schema validation for {self.pair} {date}: {e}")
            return ""

        out_dir = os.path.join(self.processed_dir, date.replace("-", "/"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{self.pair}-fx_toxicity-{date}.parquet")

        buckets.write_parquet(
            out_path,
            compression=config.compression_codec,
            compression_level=config.compression_level,
            row_group_size=config.row_group_size,
        )
        logger.info(f"FX toxicity features -> {out_path} ({len(buckets)} volume buckets)")
        return out_path


if __name__ == "__main__":
    engine = ForexToxicityEngine("EURUSD")
    engine.process_daily_file("2024-01-02")
