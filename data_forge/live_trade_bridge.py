"""
Live Trade Bridge — Phase 2 of the upgrade plan ("close the live-learning loop").

Turns a day of data_forge/lob_collector.py's live-captured aggTrade records
(data_store/live/{symbol}/{date_path}/trades_HH.parquet) into a file
TradeFlowPhysics.process_daily_file() (data_forge/feature_engineering.py) can run
over completely unmodified — by writing it to the exact location and column schema
that method already searches for and expects from a Binance-Vision bulk download.

Why bridge into the batch pipeline rather than build a separate "live" bucketing
path: execution/streaming_features.py's own module docstring already names this
exact gap ("port TradeFlowPhysics's bucket logic to a streaming form... as its own
reviewed change") and explicitly flags itself as an incremental APPROXIMATION of
that logic, not the same thing. Reusing the existing, hardened batch pipeline
(remainder-carryover across days, the variance-floor fix in Kyle's Lambda, the
timestamp-unit auto-detection) via this bridge means "real live experience reaches
retraining" without re-deriving or re-risking any of that — the bucketing math
itself is untouched.

This module deliberately does NOT call process_daily_file() itself — that's
training/continuous_trainer.py's job (see ContinuousTrainer._bridge_live_data(),
which calls bridge_live_trades_to_raw() then TradeFlowPhysics.process_daily_file()
back to back, once per cycle). Keeping this module to "get the bytes into the right
place" and nothing else makes it independently testable and reusable (e.g. from a
manual CLI backfill) without needing a live ContinuousTrainer instance.

VERIFICATION STATUS: no polars or pydantic_settings in this sandbox (same
limitation as every other data_forge module in this project), so the glob/
concat/de-dup logic here is verified by direct source tracing against
lob_collector.py's actual output schema and TradeFlowPhysics.process_daily_file()'s
actual search pattern (both read directly, not assumed), plus py_compile — not by
execution. Run this against real captured files before trusting it in production.
"""

import os
import glob
import logging
from typing import Optional

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from data_forge.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (LiveTradeBridge) %(message)s")
logger = logging.getLogger("LiveTradeBridge")


def bridge_live_trades_to_raw(symbol: str, date: str, data_store_dir: Optional[str] = None) -> bool:
    """
    Reads data_store/live/{symbol}/{date_path}/trades_*.parquet for `date`, concatenates
    them, de-duplicates on agg_trade_id (Binance's own unique trade ID -- the correct
    key; several trades can share a transact_time, so timestamp alone is not unique),
    sorts by transact_time, and writes the result to
    data_store/raw/{symbol}/aggTrades/{date_path}/{symbol}-live-aggTrades-{date}.parquet
    -- the exact directory TradeFlowPhysics.process_daily_file() globs, so it can pick
    this up with zero changes to that method.

    CAVEAT, stated plainly rather than glossed over: as of this writing,
    process_daily_file() sorts its file candidates deterministically and prefers a
    genuine bulk-download file over a live-bridged one when both exist for the same
    date (see feature_engineering.py's process_daily_file() for the fix and why the
    original unsorted glob.glob() ordering was a real, previously-undocumented
    correctness bug, not just a caveat). Even so, the two sources are NOT merged --
    only one is ever processed into physics.parquet for a given date. In practice
    this bridge is intended for "today," which a T+1-lagged bulk archive typically
    doesn't have yet, so the collision case is rare -- but it is a real possibility
    worth knowing about, not an assumption to skip checking.

    Returns True if anything was bridged, False if there were no live trade capture
    files for this date (a normal, non-error outcome -- e.g. the collector wasn't
    running that day, or is only just starting to accumulate history).
    """
    if not POLARS_AVAILABLE:
        logger.error("Polars required to bridge live trades.")
        return False

    store_dir = data_store_dir or config.data_store_dir
    date_path = date.replace("-", "/")
    live_dir = os.path.join(store_dir, "live", symbol, date_path)
    pattern = os.path.join(live_dir, "trades_*.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        logger.info(f"No live trade captures found for {symbol} on {date} in {live_dir} -- nothing to bridge.")
        return False

    frames = []
    for f in files:
        try:
            frames.append(pl.read_parquet(f))
        except Exception as e:
            logger.error(f"Failed to read {f}: {e}")

    if not frames:
        logger.error(f"All {len(files)} matched live trade file(s) failed to read -- nothing bridged.")
        return False

    # how="vertical_relaxed" tolerates minor schema drift across hourly files (e.g. one
    # hour's file written by a slightly different lob_collector.py version) rather than
    # hard-failing the whole bridge over one inconsistent partition.
    merged = pl.concat(frames, how="vertical_relaxed")
    merged = merged.unique(subset=["agg_trade_id"], keep="last").sort("transact_time")

    out_dir = os.path.join(store_dir, "raw", symbol, "aggTrades", date_path)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}-live-aggTrades-{date}.parquet")
    tmp_path = out_path + ".tmp"
    merged.write_parquet(
        tmp_path,
        compression=config.compression_codec,
        compression_level=config.compression_level,
        row_group_size=config.row_group_size,
    )
    os.replace(tmp_path, out_path)
    logger.info(f"Bridged {len(files)} live trade file(s) ({len(merged)} rows after de-dup) -> {out_path}")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bridge a day of live-captured trades into TradeFlowPhysics' input format.")
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    ok = bridge_live_trades_to_raw(args.symbol, args.date)
    raise SystemExit(0 if ok else 1)
