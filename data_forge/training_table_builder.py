"""
Training Table Builder.

Everything upstream of this module (feature_engineering.py, forex_feature_engineering.py,
forex_toxicity_engineering.py, sentiment_source.py) produces per-asset, per-day Parquet
files on disk. Nothing before this module actually assembles them into one training-ready
table: TradeJackLOBEnv only ever reads data_store/processed/<symbol>/physics/ directly and
has no notion of forex or sentiment at all. This module is that missing assembly step:

  1. For a date range, locate each requested leg's daily Parquet files on disk (crypto
     physics, one or more FX pairs' physics + toxicity, sentiment).
  2. Concatenate each leg's files into one per-leg DataFrame. A leg with zero files
     anywhere in the range is represented as an explicit schemaless placeholder
     (a DataFrame with only a `timestamp` column) rather than being silently omitted --
     this is deliberate: TimeAligner._align_one() already knows how to handle a
     schemaless leg correctly (drop it, record it in the manifest as dropped) since the
     bug fix in time_alignment.py. Passing every *requested* leg through, present or not,
     means the manifest ends up as a complete record of "what was asked for vs. what
     actually went in" -- not just "what happened to exist".
  3. Hand all legs to TimeAligner.align() to produce one merged, gap-aware table on a
     shared UTC grid.
  4. Write the merged table to Parquet and the AlignmentManifest to a JSON sidecar file
     next to it, so a training run has a durable, machine-readable record of exactly
     what data went into it.

Deliberately NOT done here: wiring this into dali_loader.py / TradeJackLOBEnv for actual
model consumption. That's a model input-shape decision (how a multi-asset observation
should be structured, whether FX/sentiment features are optional inputs or always
present, etc.) -- a modeling choice, not a data-plumbing one -- and shouldn't be bundled
into a data-assembly module's own scope.

VERIFICATION STATUS: no polars in this sandbox (same as every other data_forge module
touched so far), so this is verified by direct source-level tracing of each producer's
real output path convention (confirmed by reading feature_engineering.py,
forex_feature_engineering.py, forex_toxicity_engineering.py, and sentiment_source.py's
actual path-construction code, not assumed) plus py_compile, not by execution. Run
`python -m unittest tests/test_data_forge.py -v` yourself with a real environment before
trusting this against real data.
"""

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from data_forge.config import config
from data_forge.time_alignment import TimeAligner, AlignmentResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (TrainingTableBuilder) %(message)s")
logger = logging.getLogger("TrainingTableBuilder")


def _date_range(start_date: str, end_date: str) -> List[str]:
    """Inclusive list of 'YYYY-MM-DD' strings from start_date to end_date."""
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end < start:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")
    out = []
    d = start
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def _empty_timestamp_frame() -> "pl.DataFrame":
    """The schemaless-leg placeholder TimeAligner._align_one() is designed to detect
    and drop (see time_alignment.py's bug-fix comment). Used whenever a requested leg
    has zero files anywhere in the date range -- so the leg still shows up in
    AlignmentManifest.dropped_legs instead of being silently absent from the request
    entirely."""
    return pl.DataFrame({"timestamp": []}, schema={"timestamp": pl.Datetime})


class TrainingTableBuilder:
    """
    Assembles a multi-asset, time-aligned training table from the daily per-asset
    Parquet files that feature_engineering.py / forex_feature_engineering.py /
    forex_toxicity_engineering.py / sentiment_source.py already produce on disk.
    """

    def __init__(self, data_store_dir: Optional[str] = None):
        self.data_store_dir = data_store_dir or config.data_store_dir

    # ── Path construction, one per producer. Each mirrors that producer's own
    # write-path logic exactly (verified by reading each module's source, not
    # inferred) so a mismatch here would be a copy-paste bug, not a design guess. ──

    def _crypto_physics_path(self, symbol: str, date: str) -> str:
        # Matches feature_engineering.py: processed_dir/<date.replace('-','/')>/physics.parquet
        return os.path.join(self.data_store_dir, "processed", symbol, "physics", date.replace("-", "/"), "physics.parquet")

    def _fx_physics_path(self, pair: str, date: str) -> str:
        # Matches forex_feature_engineering.py's out_path construction.
        return os.path.join(
            self.data_store_dir, "processed", pair, "fx_physics", date.replace("-", "/"),
            f"{pair}-fx_physics-{date}.parquet",
        )

    def _fx_toxicity_path(self, pair: str, date: str) -> str:
        # Matches forex_toxicity_engineering.py's out_path construction.
        return os.path.join(
            self.data_store_dir, "processed", pair, "fx_toxicity", date.replace("-", "/"),
            f"{pair}-fx_toxicity-{date}.parquet",
        )

    def _sentiment_path(self, symbol: str, date: str) -> str:
        # Matches sentiment_source.py's out_path construction. NOTE: flat, no Y/M/D
        # subdirectory -- unlike every other leg here -- because sentiment_source.py
        # writes one file per UTC day directly under processed/<symbol>/sentiment/.
        return os.path.join(self.data_store_dir, "processed", symbol, "sentiment", f"{symbol}-sentiment-{date}.parquet")

    def _load_leg(self, leg_name: str, paths: List[str]) -> "pl.DataFrame":
        """Loads and concatenates whichever of `paths` actually exist on disk. Missing
        individual days are expected and normal (FX weekends, a day sentiment scraping
        failed, etc.) and only logged at debug level. A leg with NO files anywhere in
        the range gets the explicit schemaless placeholder instead of being dropped
        from the request silently -- see module docstring and _empty_timestamp_frame()."""
        found = [p for p in paths if os.path.exists(p)]
        missing = len(paths) - len(found)
        if missing:
            logger.debug(f"'{leg_name}': {missing}/{len(paths)} daily file(s) not found (normal for weekends/gaps).")

        if not found:
            logger.warning(f"'{leg_name}': 0/{len(paths)} daily files found in range -- leg will be dropped by TimeAligner.")
            return _empty_timestamp_frame()

        frames = []
        for p in found:
            try:
                frames.append(pl.read_parquet(p))
            except Exception as e:
                logger.error(f"'{leg_name}': failed to read {p}: {e}")

        if not frames:
            logger.warning(f"'{leg_name}': all {len(found)} matched file(s) failed to read -- leg will be dropped.")
            return _empty_timestamp_frame()

        # how="diagonal_relaxed" tolerates minor schema drift across days (e.g. an
        # extra column added by a later feature_engineering.py version) rather than
        # hard-failing the whole build over one inconsistent day.
        merged = pl.concat(frames, how="diagonal_relaxed")
        # Daily files can overlap by a bucket at day boundaries depending on how each
        # producer buckets its last/first row; de-dup on exact timestamp, keep last
        # write (later files reflect any reprocessing), then sort.
        merged = merged.unique(subset=["timestamp"], keep="last").sort("timestamp")
        logger.info(f"'{leg_name}': loaded {len(found)} file(s), {len(merged)} rows after de-dup.")
        return merged

    def load_legs(
        self,
        start_date: str,
        end_date: str,
        crypto_symbol: Optional[str] = None,
        forex_pairs: Optional[List[str]] = None,
        sentiment_symbol: Optional[str] = None,
    ) -> Dict[str, "pl.DataFrame"]:
        """
        Loads every requested leg over [start_date, end_date] (inclusive) into a
        {leg_name: DataFrame} dict ready for TimeAligner.align(). Every requested leg
        is present in the returned dict, even if empty (see _load_leg) -- callers
        should not filter this dict before passing it to align().
        """
        dates = _date_range(start_date, end_date)
        legs: Dict[str, "pl.DataFrame"] = {}

        if crypto_symbol:
            paths = [self._crypto_physics_path(crypto_symbol, d) for d in dates]
            legs[crypto_symbol.lower().replace("-", "")] = self._load_leg(f"crypto:{crypto_symbol}", paths)

        for pair in (forex_pairs or []):
            pair_key = pair.lower()
            physics_paths = [self._fx_physics_path(pair, d) for d in dates]
            legs[f"{pair_key}_physics"] = self._load_leg(f"fx_physics:{pair}", physics_paths)

            toxicity_paths = [self._fx_toxicity_path(pair, d) for d in dates]
            legs[f"{pair_key}_toxicity"] = self._load_leg(f"fx_toxicity:{pair}", toxicity_paths)

        if sentiment_symbol:
            paths = [self._sentiment_path(sentiment_symbol, d) for d in dates]
            legs[f"{sentiment_symbol.lower().replace('-', '')}_sentiment"] = self._load_leg(
                f"sentiment:{sentiment_symbol}", paths
            )

        return legs

    def build(
        self,
        start_date: str,
        end_date: str,
        crypto_symbol: Optional[str] = "BTC-USDT",
        forex_pairs: Optional[List[str]] = None,
        sentiment_symbol: Optional[str] = None,
        frequency: str = "1m",
        max_staleness_buckets: int = 5,
        output_path: Optional[str] = None,
    ) -> Optional[AlignmentResult]:
        """
        End-to-end: load every requested leg, align them, and (if output_path is given)
        write the merged table plus a JSON provenance manifest sidecar.

        Returns the AlignmentResult (result.data / result.manifest), or None if Polars
        is unavailable or every leg came back empty.
        """
        if not POLARS_AVAILABLE:
            logger.error("Polars required to build a training table.")
            return None

        legs = self.load_legs(
            start_date, end_date,
            crypto_symbol=crypto_symbol,
            forex_pairs=forex_pairs,
            sentiment_symbol=sentiment_symbol,
        )
        if not legs:
            logger.error("No legs requested -- nothing to build.")
            return None

        aligner = TimeAligner(frequency=frequency, max_staleness_buckets=max_staleness_buckets)
        result = aligner.align(legs)
        if result is None:
            logger.error("Alignment produced no result (all requested legs were empty).")
            return None

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            result.data.write_parquet(
                output_path,
                compression=config.compression_codec,
                row_group_size=config.row_group_size,
            )
            manifest_path = output_path + ".manifest.json"
            manifest_payload = dict(result.manifest.to_dict())
            manifest_payload.update({
                "start_date": start_date,
                "end_date": end_date,
                "crypto_symbol": crypto_symbol,
                "forex_pairs": forex_pairs or [],
                "sentiment_symbol": sentiment_symbol,
                "built_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            with open(manifest_path, "w") as f:
                json.dump(manifest_payload, f, indent=2)
            logger.info(f"Wrote training table -> {output_path} ({len(result.data)} rows), manifest -> {manifest_path}")

        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a time-aligned multi-asset training table.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--crypto-symbol", default="BTC-USDT")
    parser.add_argument("--forex-pairs", nargs="*", default=None)
    parser.add_argument("--sentiment-symbol", default=None)
    parser.add_argument("--frequency", default="1m")
    parser.add_argument("--max-staleness-buckets", type=int, default=5)
    parser.add_argument("--output", required=True, help="Output .parquet path")
    args = parser.parse_args()

    builder = TrainingTableBuilder()
    res = builder.build(
        start_date=args.start,
        end_date=args.end,
        crypto_symbol=args.crypto_symbol,
        forex_pairs=args.forex_pairs,
        sentiment_symbol=args.sentiment_symbol,
        frequency=args.frequency,
        max_staleness_buckets=args.max_staleness_buckets,
        output_path=args.output,
    )
    if res is None:
        raise SystemExit(1)
