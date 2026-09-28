"""
Verification Test 2: Data Forge Subsystems (`test_data_forge.py`).
Tests KvikIO GPUDirect streaming, Polars/NPZ cleaning pipeline, and double-buffer pre-fetching.
"""

import os
import sys
import json
import asyncio
import unittest
import shutil

# BUG-9 FIX: module is 'kvikio_streamer', not 'kvikio_pipeline'
from data_forge.kvikio_streamer import KvikIODataForge
from data_forge.parquet_ingest import ParquetIngestPipeline
from data_forge.dali_loader import create_lob_dataloader

try:
    import polars as pl
    import datetime as _dt
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from data_forge.forex_feature_engineering import ForexFeatureEngine
from data_forge.forex_toxicity_engineering import ForexToxicityEngine
from data_forge.sentiment_source import SentimentSource, _matches_symbol
from data_forge.time_alignment import TimeAligner
from data_forge.training_table_builder import TrainingTableBuilder


class TestDataForgeSubsystems(unittest.TestCase):

    def setUp(self):
        self.test_data_store = os.path.abspath("d:/TradeJack/data_store_test_forge")
        os.makedirs(self.test_data_store, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_data_store):
            shutil.rmtree(self.test_data_store, ignore_errors=True)

    def test_synthetic_generation_and_ingestion(self):
        ingest = ParquetIngestPipeline(data_store_dir=self.test_data_store)
        asyncio.run(ingest.generate_synthetic_crucible_data(symbol="ETH-USDT", num_days=1, ticks_per_day=50))
        
        eth_dir = os.path.join(self.test_data_store, "processed", "ETH-USDT", "physics")
        self.assertTrue(os.path.exists(eth_dir))
        files = os.listdir(eth_dir)
        self.assertGreater(len(files), 0)

    def test_kvikio_data_forge_streaming(self):
        ingest = ParquetIngestPipeline(data_store_dir=self.test_data_store)
        asyncio.run(ingest.generate_synthetic_crucible_data(symbol="SOL-USDT", num_days=1, ticks_per_day=30))
        
        forge = KvikIODataForge(data_store_dir=self.test_data_store)
        files = forge.scan_available_partitions(symbol="SOL-USDT")
        self.assertGreater(len(files), 0)
        
        tensor_data = forge.load_file_to_tensor(files[0])
        self.assertIsNotNone(tensor_data)
        for col in tensor_data:
            self.assertGreaterEqual(len(tensor_data[col]), 1)
            break

    def test_dali_loader_batch_iteration(self):
        ingest = ParquetIngestPipeline(data_store_dir=self.test_data_store)
        asyncio.run(ingest.generate_synthetic_crucible_data(symbol="ADA-USDT", num_days=1, ticks_per_day=40, start_date="2024-01-01"))
        
        loader = create_lob_dataloader(
            symbol="ADA-USDT", start_date="2024-01-01", end_date="2024-01-01",
            batch_size=8, seq_len=10, data_store_dir=self.test_data_store
        )
        for batch_x, batch_y in loader:
            self.assertIsNotNone(batch_x)
            self.assertEqual(batch_x.shape[0], 8)
            self.assertEqual(batch_x.shape[1], 10)
            self.assertEqual(batch_x.shape[2], 6)
            break


@unittest.skipUnless(POLARS_AVAILABLE, "Polars not installed")
class TestForexFeatureEngineering(unittest.TestCase):
    """Offline tests — synthetic tick fixtures only, no network dependency."""

    def setUp(self):
        self.test_data_store = os.path.abspath("d:/TradeJack/data_store_test_fx")
        os.makedirs(self.test_data_store, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_data_store):
            shutil.rmtree(self.test_data_store, ignore_errors=True)

    def _write_raw_ticks(self, pair: str, date: str, ticks: "pl.DataFrame"):
        out_dir = os.path.join(self.test_data_store, "raw", pair, "forex_ticks", date.replace("-", "/"))
        os.makedirs(out_dir, exist_ok=True)
        ticks.write_parquet(os.path.join(out_dir, f"{pair}-ticks-{date}.parquet"))

    def test_features_computed_without_gap(self):
        base = _dt.datetime(2024, 1, 2, 8, 0, tzinfo=_dt.timezone.utc)
        ticks = pl.DataFrame({
            "timestamp": [base + _dt.timedelta(seconds=i * 10) for i in range(20)],
            "bid": [1.1000 + i * 0.0001 for i in range(20)],
            "ask": [1.1002 + i * 0.0001 for i in range(20)],
            "bid_volume": [1.0] * 20,
            "ask_volume": [1.2] * 20,
        })
        self._write_raw_ticks("EURUSD", "2024-01-02", ticks)

        engine = ForexFeatureEngine("EURUSD", bucket_seconds=60)
        engine.raw_dir = os.path.join(self.test_data_store, "raw", "EURUSD", "forex_ticks")
        engine.processed_dir = os.path.join(self.test_data_store, "processed", "EURUSD", "fx_physics")
        out_path = engine.process_daily_file("2024-01-02")

        self.assertTrue(out_path)
        result = pl.read_parquet(out_path)
        self.assertIn("quote_imbalance", result.columns)
        self.assertIn("is_gap", result.columns)
        # Continuous ticks with no weekend gap: no bucket should be flagged stale.
        self.assertEqual(int(result["is_gap"].sum()), 0)
        # bid_volume < ask_volume in this fixture -> imbalance should be negative.
        self.assertTrue((result["quote_imbalance"].drop_nulls() < 0).all())

    def test_gap_is_flagged_and_return_nulled(self):
        base = _dt.datetime(2024, 1, 5, 20, 0, tzinfo=_dt.timezone.utc)  # Friday evening
        pre_gap = [base + _dt.timedelta(seconds=i * 10) for i in range(5)]
        post_gap = [base + _dt.timedelta(hours=50, seconds=i * 10) for i in range(5)]  # after weekend
        timestamps = pre_gap + post_gap
        n = len(timestamps)
        ticks = pl.DataFrame({
            "timestamp": timestamps,
            "bid": [1.10] * n,
            "ask": [1.1002] * n,
            "bid_volume": [1.0] * n,
            "ask_volume": [1.0] * n,
        })
        self._write_raw_ticks("EURUSD", "2024-01-05", ticks)

        engine = ForexFeatureEngine("EURUSD", bucket_seconds=60, max_gap_seconds=3600)
        engine.raw_dir = os.path.join(self.test_data_store, "raw", "EURUSD", "forex_ticks")
        engine.processed_dir = os.path.join(self.test_data_store, "processed", "EURUSD", "fx_physics")
        out_path = engine.process_daily_file("2024-01-05")

        result = pl.read_parquet(out_path)
        self.assertGreaterEqual(int(result["is_gap"].sum()), 1)
        gapped_rows = result.filter(pl.col("is_gap"))
        # The bucket right after the weekend gap must not carry a return across it.
        self.assertTrue(gapped_rows["log_return"].is_null().any())


@unittest.skipUnless(POLARS_AVAILABLE, "Polars not installed")
class TestForexToxicityEngineering(unittest.TestCase):
    """Offline tests for BVC-based VPIN/Kyle's Lambda — synthetic tick fixtures only."""

    def setUp(self):
        self.test_data_store = os.path.abspath("d:/TradeJack/data_store_test_fx_tox")
        os.makedirs(self.test_data_store, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_data_store):
            shutil.rmtree(self.test_data_store, ignore_errors=True)

    def _write_raw_ticks(self, pair: str, date: str, ticks: "pl.DataFrame"):
        out_dir = os.path.join(self.test_data_store, "raw", pair, "forex_ticks", date.replace("-", "/"))
        os.makedirs(out_dir, exist_ok=True)
        ticks.write_parquet(os.path.join(out_dir, f"{pair}-ticks-{date}.parquet"))

    def _make_engine(self, pair: str, **kwargs) -> ForexToxicityEngine:
        engine = ForexToxicityEngine(pair, **kwargs)
        engine.raw_dir = os.path.join(self.test_data_store, "raw", pair, "forex_ticks")
        engine.processed_dir = os.path.join(self.test_data_store, "processed", pair, "fx_toxicity")
        return engine

    def test_strong_uptrend_produces_high_vpin_and_positive_lambda(self):
        # Construct a strongly one-directional price move: mid price climbs steadily on
        # steady volume. BVC should classify most of this volume as "buy" (z >> 0), so
        # bvc_vpin should be high (flow dominated by one side) and kyles_lambda positive
        # (price rises alongside the classified buy-heavy flow).
        base = _dt.datetime(2024, 1, 2, 8, 0, tzinfo=_dt.timezone.utc)
        # A perfectly smooth linear trend has zero rolling variance, which makes BVC's
        # z-score collapse to 0 (neutral) — there's no "surprise" to classify against.
        # Use a varying-but-always-positive increment pattern instead: still a clear net
        # uptrend, but with real tick-to-tick variance for BVC to compare against.
        increments = [0.0003, 0.0007, 0.0005, 0.0009, 0.0002]
        n = 300
        bids = [1.1000]
        for i in range(1, n):
            bids.append(bids[-1] + increments[i % len(increments)])
        ticks = pl.DataFrame({
            "timestamp": [base + _dt.timedelta(seconds=i) for i in range(n)],
            "bid": bids,
            "ask": [b + 0.0002 for b in bids],
            "bid_volume": [1.0] * n,
            "ask_volume": [1.0] * n,
        })
        self._write_raw_ticks("EURUSD", "2024-01-02", ticks)

        engine = self._make_engine("EURUSD", volume_bucket_size=20.0, std_window=10, rolling_window=5)
        out_path = engine.process_daily_file("2024-01-02")

        self.assertTrue(out_path)
        result = pl.read_parquet(out_path)
        self.assertGreater(len(result), 0)
        # Skip the first bucket or two (rolling stats still warming up).
        stable = result.tail(max(1, len(result) - 2))
        self.assertGreater(stable["bvc_vpin"].mean(), 0.3)
        self.assertGreater(stable["kyles_lambda"].mean(), 0.0)

    def test_flat_market_produces_low_vpin(self):
        # No net price movement -> BVC should classify flow close to 50/50 buy/sell,
        # so bvc_vpin should stay low (order flow is not one-sided).
        base = _dt.datetime(2024, 1, 2, 8, 0, tzinfo=_dt.timezone.utc)
        n = 300
        ticks = pl.DataFrame({
            "timestamp": [base + _dt.timedelta(seconds=i) for i in range(n)],
            "bid": [1.1000] * n,
            "ask": [1.1002] * n,
            "bid_volume": [1.0] * n,
            "ask_volume": [1.0] * n,
        })
        self._write_raw_ticks("EURUSD", "2024-01-02", ticks)

        engine = self._make_engine("EURUSD", volume_bucket_size=20.0, std_window=10, rolling_window=5)
        out_path = engine.process_daily_file("2024-01-02")

        self.assertTrue(out_path)
        result = pl.read_parquet(out_path)
        self.assertLess(result["bvc_vpin"].mean(), 0.3)


class TestSentimentSource(unittest.TestCase):
    """Tests the parts of sentiment_source.py that don't require a live network call."""

    def test_symbol_keyword_matching(self):
        self.assertTrue(_matches_symbol("Bitcoin surges past $50k", "BTC-USDT"))
        self.assertFalse(_matches_symbol("Ethereum gas fees drop", "BTC-USDT"))
        # Symbols with no keyword map fall back to matching everything.
        self.assertTrue(_matches_symbol("Completely unrelated headline", "XYZ-UNKNOWN"))

    def test_score_headline_neutral_without_vader(self):
        source = SentimentSource()
        source._analyzer = None  # simulate vaderSentiment not installed
        self.assertEqual(source.score_headline("Markets rally on strong earnings"), 0.0)


@unittest.skipUnless(POLARS_AVAILABLE, "Polars not installed")
class TestTimeAlignment(unittest.TestCase):

    def test_alignment_flags_staleness(self):
        base = _dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc)
        crypto_df = pl.DataFrame({
            "timestamp": [base + _dt.timedelta(minutes=i) for i in range(10)],
            "mid_price": [100.0 + i for i in range(10)],
        })
        # Sparse FX frame: real ticks only at bucket 0 and bucket 8.
        fx_df = pl.DataFrame({
            "timestamp": [base, base + _dt.timedelta(minutes=8)],
            "mid_price": [1.10, 1.11],
        })

        aligner = TimeAligner(frequency="1m", max_staleness_buckets=2)
        result = aligner.align({"crypto": crypto_df, "fx": fx_df})

        self.assertIsNotNone(result)
        data = result.data
        self.assertEqual(len(data), 10)
        self.assertIn("fx__is_stale", data.columns)
        self.assertIn("crypto__is_stale", data.columns)
        # crypto has a real value every bucket -> never stale.
        self.assertEqual(int(data["crypto__is_stale"].sum()), 0)
        # fx has real values only at buckets 0 and 8, with max_staleness=2 buckets ->
        # buckets 3,4,5,6,7 (distance 3-7 from bucket 0) must be flagged stale.
        self.assertGreaterEqual(int(data["fx__is_stale"].sum()), 4)

        # Manifest must reflect the same facts the data itself shows -- this is the
        # whole point of returning a manifest instead of a bare DataFrame: a caller
        # (e.g. training_table_builder.py) should be able to answer "how much of
        # each leg was real vs. filled" without re-deriving it from the columns.
        manifest = result.manifest
        self.assertEqual(manifest.total_buckets, 10)
        self.assertEqual(set(manifest.included_legs), {"crypto", "fx"})
        self.assertEqual(manifest.dropped_legs, [])
        self.assertEqual(manifest.stale_bucket_counts["crypto"], 0)
        self.assertGreaterEqual(manifest.stale_bucket_counts["fx"], 4)
        self.assertAlmostEqual(manifest.stale_fraction("fx"), manifest.stale_bucket_counts["fx"] / 10)
        # to_dict() must actually be JSON-serializable (the whole point of the
        # manifest existing as a sidecar provenance record).
        import json
        json.dumps(manifest.to_dict())

    def test_schemaless_leg_is_dropped_not_marked_fresh(self):
        """
        BUG FIX regression test: a leg with zero feature columns (asset entirely
        unavailable) used to fall through to `had_value = all True`, meaning it was
        reported as ALWAYS FRESH -- the opposite of the truth -- and produced an
        `{prefix}__is_stale` column that's always False with no data behind it.
        Correct behavior: the schemaless leg is dropped from the merged output
        entirely (logged loudly), not silently marked fresh -- and now, that fact
        is also recorded in the manifest, not just a log line a caller might miss.
        """
        base = _dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc)
        crypto_df = pl.DataFrame({
            "timestamp": [base + _dt.timedelta(minutes=i) for i in range(5)],
            "mid_price": [100.0 + i for i in range(5)],
        })
        # A leg for an asset with genuinely zero data available: only a timestamp
        # column, no feature columns at all.
        missing_asset_df = pl.DataFrame({"timestamp": []}, schema={"timestamp": pl.Datetime})

        aligner = TimeAligner(frequency="1m", max_staleness_buckets=2)
        result = aligner.align({"crypto": crypto_df, "missing_asset": missing_asset_df})

        self.assertIsNotNone(result)
        data = result.data
        # The schemaless leg must not appear at all -- not as data, and NOT as a
        # fabricated always-False is_stale flag either.
        self.assertNotIn("missing_asset__is_stale", data.columns)
        self.assertFalse(any(c.startswith("missing_asset__") for c in data.columns))
        # The real leg is unaffected.
        self.assertIn("crypto__mid_price", data.columns)

        # The manifest is the machine-readable record of the drop -- this is the
        # actual regression guard: a human reading logs could miss this, a
        # caller checking manifest.dropped_legs cannot.
        manifest = result.manifest
        self.assertIn("missing_asset", manifest.dropped_legs)
        self.assertNotIn("missing_asset", manifest.included_legs)
        self.assertNotIn("missing_asset", manifest.stale_bucket_counts)
        self.assertIsNone(manifest.stale_fraction("missing_asset"))


@unittest.skipUnless(POLARS_AVAILABLE, "Polars not installed")
class TestTrainingTableBuilder(unittest.TestCase):
    """
    Exercises TrainingTableBuilder against real on-disk fixture files, written at each
    producer's real path convention (verified against feature_engineering.py,
    forex_feature_engineering.py, forex_toxicity_engineering.py, sentiment_source.py's
    own source -- not guessed), rather than mocking the loader.
    """

    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp(prefix="ttb_test_")
        self.builder = TrainingTableBuilder(data_store_dir=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_crypto_day(self, symbol: str, date: str, n_rows: int = 5):
        base = _dt.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc)
        df = pl.DataFrame({
            "timestamp": [base + _dt.timedelta(minutes=i) for i in range(n_rows)],
            "open_price": [100.0 + i for i in range(n_rows)],
            "close_price": [100.5 + i for i in range(n_rows)],
            "volume": [10.0] * n_rows,
            "ofi": [0.1] * n_rows,
            "vpin_50": [0.3] * n_rows,
            "kyles_lambda": [0.02] * n_rows,
        })
        # Path built independently here (literal string, mirroring
        # feature_engineering.py's own processed_dir/date_path/physics.parquet
        # construction verbatim) rather than by calling the method under test --
        # so this actually pins TrainingTableBuilder._crypto_physics_path() against
        # the real producer's convention instead of only checking self-consistency.
        path = os.path.join(self.tmp_dir, "processed", symbol, "physics", date.replace("-", "/"), "physics.parquet")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.write_parquet(path)
        return path

    def _write_fx_physics_day(self, pair: str, date: str, n_rows: int = 4):
        base = _dt.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc)
        df = pl.DataFrame({
            "timestamp": [base + _dt.timedelta(minutes=i) for i in range(n_rows)],
            "mid_price": [1.10 + 0.001 * i for i in range(n_rows)],
            "spread": [0.0002] * n_rows,
            "relative_spread": [0.0002] * n_rows,
            "quote_imbalance": [0.0] * n_rows,
            "quote_intensity": [5.0] * n_rows,
            "log_return": [0.0001] * n_rows,
            "is_gap": [False] * n_rows,
        })
        # Same independence note as _write_crypto_day: literal path mirroring
        # forex_feature_engineering.py's own construction, not the method under test.
        path = os.path.join(
            self.tmp_dir, "processed", pair, "fx_physics", date.replace("-", "/"),
            f"{pair}-fx_physics-{date}.parquet",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.write_parquet(path)
        return path

    def test_load_legs_uses_real_producer_paths(self):
        """Fixture files written at each producer's real path convention must actually
        be found -- this is the load-bearing assertion: if any _*_path() method drifts
        from what the real producer writes, this test fails immediately instead of
        silently returning empty legs at training time."""
        self._write_crypto_day("BTC-USDT", "2024-01-01")
        self._write_fx_physics_day("EURUSD", "2024-01-01")

        legs = self.builder.load_legs(
            "2024-01-01", "2024-01-01",
            crypto_symbol="BTC-USDT",
            forex_pairs=["EURUSD"],
            sentiment_symbol="BTC-USDT",
        )

        self.assertIn("btcusdt", legs)
        self.assertGreater(len(legs["btcusdt"]), 0)
        self.assertIn("open_price", legs["btcusdt"].columns)

        self.assertIn("eurusd_physics", legs)
        self.assertGreater(len(legs["eurusd_physics"]), 0)

        # No FX toxicity file was written for this date -> schemaless placeholder,
        # not a KeyError and not silently absent from the dict.
        self.assertIn("eurusd_toxicity", legs)
        self.assertEqual(legs["eurusd_toxicity"].columns, ["timestamp"])
        self.assertEqual(len(legs["eurusd_toxicity"]), 0)

        # No sentiment file was written either -> same schemaless contract.
        self.assertIn("btcusdt_sentiment", legs)
        self.assertEqual(legs["btcusdt_sentiment"].columns, ["timestamp"])

    def test_multi_day_concatenation_and_dedup(self):
        """Two daily files where day 2 re-includes day 1's last timestamp (simulating
        a reprocessing overlap at the day boundary) must merge into one sorted,
        de-duplicated frame -- not silently double-count that bucket."""
        base = _dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc)
        day1 = pl.DataFrame({
            "timestamp": [base + _dt.timedelta(minutes=i) for i in range(3)],  # :00, :01, :02
            "open_price": [100.0, 101.0, 102.0],
            "close_price": [100.5, 101.5, 102.5],
            "volume": [10.0, 10.0, 10.0],
            "ofi": [0.1, 0.1, 0.1],
            "vpin_50": [0.3, 0.3, 0.3],
            "kyles_lambda": [0.02, 0.02, 0.02],
        })
        # Day 2 re-includes the :02 timestamp with a DIFFERENT value (simulating a
        # reprocessed/corrected row) plus one genuinely new bucket at :03.
        day2 = pl.DataFrame({
            "timestamp": [base + _dt.timedelta(minutes=i) for i in (2, 3)],
            "open_price": [999.0, 103.0],  # :02 corrected value should win (keep="last")
            "close_price": [999.5, 103.5],
            "volume": [10.0, 10.0],
            "ofi": [0.1, 0.1],
            "vpin_50": [0.3, 0.3],
            "kyles_lambda": [0.02, 0.02],
        })
        path1 = os.path.join(self.tmp_dir, "processed", "BTC-USDT", "physics", "2024/01/01", "physics.parquet")
        path2 = os.path.join(self.tmp_dir, "processed", "BTC-USDT", "physics", "2024/01/02", "physics.parquet")
        os.makedirs(os.path.dirname(path1), exist_ok=True)
        os.makedirs(os.path.dirname(path2), exist_ok=True)
        day1.write_parquet(path1)
        day2.write_parquet(path2)

        legs = self.builder.load_legs("2024-01-01", "2024-01-02", crypto_symbol="BTC-USDT")
        leg = legs["btcusdt"]
        # 3 + 2 with one genuine overlap at :02 -> 4 unique timestamps, not 5.
        self.assertEqual(len(leg), 4)
        self.assertTrue(leg["timestamp"].is_sorted())
        # The corrected (day2) value for :02 must win, per keep="last" -- proves
        # de-dup actually resolved the conflict rather than keeping the stale row.
        row_at_02 = leg.filter(pl.col("timestamp") == base + _dt.timedelta(minutes=2))
        self.assertEqual(row_at_02["open_price"][0], 999.0)

    def test_build_writes_table_and_manifest_with_dropped_legs_recorded(self):
        """End-to-end: only a crypto leg exists on disk; a requested FX pair and
        sentiment leg do not. The build must still succeed, and the manifest sidecar
        must record the missing legs as dropped -- not just quietly exclude them."""
        self._write_crypto_day("BTC-USDT", "2024-01-01", n_rows=10)

        out_path = os.path.join(self.tmp_dir, "training_table.parquet")
        result = self.builder.build(
            start_date="2024-01-01",
            end_date="2024-01-01",
            crypto_symbol="BTC-USDT",
            forex_pairs=["EURUSD"],
            sentiment_symbol="BTC-USDT",
            frequency="1m",
            max_staleness_buckets=2,
            output_path=out_path,
        )

        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(out_path))
        manifest_path = out_path + ".manifest.json"
        self.assertTrue(os.path.exists(manifest_path))

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.assertIn("btcusdt", manifest["included_legs"])
        self.assertIn("eurusd_physics", manifest["dropped_legs"])
        self.assertIn("eurusd_toxicity", manifest["dropped_legs"])
        self.assertIn("btcusdt_sentiment", manifest["dropped_legs"])
        self.assertEqual(manifest["crypto_symbol"], "BTC-USDT")
        self.assertEqual(manifest["forex_pairs"], ["EURUSD"])
        self.assertIn("built_at_utc", manifest)

    def test_build_returns_none_when_no_legs_have_data(self):
        """No fixtures written at all -> every leg is schemaless -> align() itself
        returns None (all-empty case already handled by time_alignment.py) -> build()
        must propagate that as None, not raise or fabricate an empty table."""
        result = self.builder.build(
            start_date="2024-01-01",
            end_date="2024-01-01",
            crypto_symbol="BTC-USDT",
            output_path=os.path.join(self.tmp_dir, "should_not_exist.parquet"),
        )
        self.assertIsNone(result)
        self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, "should_not_exist.parquet")))


if __name__ == "__main__":
    unittest.main()
