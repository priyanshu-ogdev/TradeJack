"""
Cross-Asset Time Alignment.

Crypto (continuous UTC), forex (session-based, real weekend/holiday gaps), and
sentiment (event-driven, sparse) all have different native time structure. Before
Phase 2, nothing in data_forge aligned them onto a shared clock for a multi-asset
training batch — that was an unstated design gap, not a solved problem.

This module resamples a set of per-asset feature frames onto one shared UTC grid,
with an explicit, stated gap policy instead of an implicit one:

  - Forward-fill is allowed up to `max_staleness` buckets.
  - Beyond that, the bucket's `is_stale` flag is set to 1 and the filled values are
    still forward-filled (so downstream code always gets a numeric value, never a
    NaN it has to special-case) — but the flag makes the staleness visible to the
    model/training loop instead of hiding it as if it were fresh data.

This matters most for forex, which has ~48h weekend closures every single week:
those should show up to the RL agent as "market closed" (via is_stale), not as
48 hours of suspiciously flat price action that looks like a real quiet period.
Crypto and sentiment will rarely hit the stale path at all under normal ingestion.

`align()` returns an `AlignmentResult` (data + a structured `AlignmentManifest`)
rather than a bare DataFrame. This is deliberate: a dropped schemaless leg or a
heavily-stale one used to be visible only as a log line a human might miss.
training_table_builder.py (built in parallel, not part of this module) needs a
machine-readable record of what actually went into a training table -- which
legs were silently excluded and how, not just the merged data with no memory
of what's missing from it. `AlignmentManifest.to_dict()` is JSON-serializable
so it can be written alongside the output Parquet as provenance.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (TimeAlignment) %(message)s")
logger = logging.getLogger("TimeAlignment")


@dataclass
class AlignmentManifest:
    """Structured record of what align() actually did -- meant to be persisted
    alongside the merged output (e.g. as a sidecar JSON file) so a training run
    has a durable answer to "what data actually went into this table", not just
    the table itself."""

    frequency: str
    max_staleness_buckets: int
    total_buckets: int
    included_legs: List[str] = field(default_factory=list)
    dropped_legs: List[str] = field(default_factory=list)
    # {leg_name: number of buckets flagged is_stale for that leg}
    stale_bucket_counts: Dict[str, int] = field(default_factory=dict)

    def stale_fraction(self, leg: str) -> Optional[float]:
        """Fraction of buckets flagged stale for `leg`, or None if the leg isn't
        in this manifest (e.g. it was dropped, or never passed in)."""
        if leg not in self.stale_bucket_counts or self.total_buckets == 0:
            return None
        return self.stale_bucket_counts[leg] / self.total_buckets

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AlignmentResult:
    """Return type of TimeAligner.align(): the merged frame plus a manifest of
    what went into producing it."""

    data: "pl.DataFrame"
    manifest: AlignmentManifest


class TimeAligner:
    """
    Aligns multiple per-asset feature DataFrames (each with a `timestamp` column)
    onto one shared UTC grid at a fixed frequency.
    """

    def __init__(self, frequency: str = "1m", max_staleness_buckets: int = 5):
        """
        frequency: Polars interval string for the shared grid (e.g. "1m", "5m", "1h").
        max_staleness_buckets: how many consecutive buckets a value may be forward-filled
            for before being flagged `is_stale`. Chosen in bucket units (not wall-clock
            time) so it scales automatically with `frequency`.
        """
        self.frequency = frequency
        self.max_staleness_buckets = max_staleness_buckets

    def _build_grid(self, frames: Dict[str, "pl.DataFrame"]) -> "pl.DataFrame":
        start = min(df["timestamp"].min() for df in frames.values() if not df.is_empty())
        end = max(df["timestamp"].max() for df in frames.values() if not df.is_empty())
        return pl.DataFrame({"timestamp": pl.datetime_range(start, end, interval=self.frequency, eager=True)})

    def _align_one(self, grid: "pl.DataFrame", df: "pl.DataFrame", prefix: str) -> Optional["pl.DataFrame"]:
        if df.is_empty():
            logger.warning(f"'{prefix}' frame is empty — its columns will be entirely stale/null on the shared grid.")

        value_cols = [c for c in df.columns if c != "timestamp"]

        # BUG FIX: a leg with zero feature columns (e.g. an asset for which nothing
        # could be fetched at all, represented as pl.DataFrame({"timestamp": []}) or
        # similar) used to fall through the `if value_cols else ...` branch below to
        # `had_value = all True` -- meaning a completely absent asset was reported as
        # ALWAYS FRESH, the exact opposite of the truth, and silently contributed an
        # `{prefix}__is_stale` column that's always False with no value columns behind
        # it for anything downstream to notice the asset is missing. Reproduced by
        # tracing this branch directly: value_cols=[] takes the falsy path of the
        # ternary unconditionally, regardless of whether the frame has any rows.
        # Correct handling: a schemaless leg carries no information, so drop it here
        # (log loudly) rather than inventing a misleading always-fresh flag for it.
        if not value_cols:
            logger.error(
                f"'{prefix}' frame has no feature columns (schemaless) -- dropping it "
                f"from alignment entirely rather than emitting a fabricated "
                f"'{prefix}__is_stale=False' flag with no data behind it. If this asset "
                f"is genuinely unavailable, callers should see it missing from the "
                f"output, not see it reported as fresh."
            )
            return None

        joined = grid.join(df, on="timestamp", how="left").sort("timestamp")

        # Track which rows had a real observation at this exact bucket, before filling,
        # so staleness distance is computed from genuine gaps, not from the fill itself.
        had_value = joined[value_cols[0]].is_not_null()

        # Distance (in buckets) since the last real observation, computed via a running
        # "group id that only increments on a real value" trick, then a cumulative count
        # within each group.
        group_id = had_value.cum_sum()
        stale_distance = (
            pl.DataFrame({"group_id": group_id, "had_value": had_value})
            .with_columns(pl.int_range(pl.len()).over("group_id").alias("distance"))
        )["distance"]
        # Rows that had a real value are distance 0 by definition (already fresh).
        stale_distance = pl.Series(
            [0 if hv else d for hv, d in zip(had_value.to_list(), stale_distance.to_list())]
        )

        filled = joined.with_columns([pl.col(c).forward_fill().alias(f"{prefix}__{c}") for c in value_cols])
        is_stale = (stale_distance > self.max_staleness_buckets).rename(f"{prefix}__is_stale")

        out_cols = ["timestamp"] + [f"{prefix}__{c}" for c in value_cols]
        result = filled.select(out_cols).with_columns(is_stale)
        return result

    def align(self, frames: Dict[str, "pl.DataFrame"]) -> Optional[AlignmentResult]:
        """
        frames: dict of {asset_name: DataFrame}, each with a `timestamp` column plus
            one or more feature columns. Column names are prefixed with `{asset_name}__`
            in the output to avoid collisions (e.g. `btc__mid_price`, `eurusd__spread`).
        Returns an AlignmentResult (merged DataFrame + AlignmentManifest), or None if
        Polars is unavailable or every input frame is empty.
        """
        if not POLARS_AVAILABLE:
            logger.error("Polars required for time alignment.")
            return None

        non_empty = {k: v for k, v in frames.items() if not v.is_empty()}
        if not non_empty:
            logger.error("All input frames are empty — nothing to align.")
            return None

        grid = self._build_grid(non_empty)
        merged = grid
        included_legs: List[str] = []
        dropped_legs: List[str] = []
        stale_bucket_counts: Dict[str, int] = {}
        for name, df in frames.items():
            aligned = self._align_one(grid, df, name)
            if aligned is None:
                dropped_legs.append(name)  # schemaless leg -- logged inside _align_one
                continue
            merged = merged.join(aligned, on="timestamp", how="left")
            included_legs.append(name)
            stale_bucket_counts[name] = int(aligned[f"{name}__is_stale"].sum())

        manifest = AlignmentManifest(
            frequency=self.frequency,
            max_staleness_buckets=self.max_staleness_buckets,
            total_buckets=len(merged),
            included_legs=included_legs,
            dropped_legs=dropped_legs,
            stale_bucket_counts=stale_bucket_counts,
        )

        logger.info(
            f"Aligned {len(included_legs)}/{len(frames)} asset streams onto {len(merged)} buckets "
            f"at '{self.frequency}' frequency (max_staleness={self.max_staleness_buckets} buckets). "
            f"Dropped: {dropped_legs or 'none'}."
        )
        return AlignmentResult(data=merged, manifest=manifest)


if __name__ == "__main__":
    # Minimal smoke-test with synthetic frames — no network/disk dependency.
    if POLARS_AVAILABLE:
        import datetime as _dt

        base = _dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc)
        crypto_df = pl.DataFrame({
            "timestamp": [base + _dt.timedelta(minutes=i) for i in range(10)],
            "mid_price": [100.0 + i for i in range(10)],
        })
        # Simulate an FX weekend gap: only 2 of 10 buckets have real ticks.
        fx_df = pl.DataFrame({
            "timestamp": [base, base + _dt.timedelta(minutes=8)],
            "mid_price": [1.10, 1.11],
        })

        aligner = TimeAligner(frequency="1m", max_staleness_buckets=2)
        result = aligner.align({"crypto": crypto_df, "fx": fx_df})
        print(result.data)
        print(result.manifest.to_dict())
