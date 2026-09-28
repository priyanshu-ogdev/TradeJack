"""
Real, executable tests for scripts/train_supervised_predictor.py's pure-logic pieces
(_log_returns, and train_and_write_leg's grouping-by-date logic against a REAL trained
model) -- everything except the actual write_predictions_leg() Parquet write, which
needs polars (unavailable in this sandbox, mocked out here so the rest of the pipeline
-- training, gating, predicting, grouping -- is still genuinely exercised).
"""
import datetime as dt
import unittest
from unittest.mock import patch

import numpy as np

from scripts.train_supervised_predictor import _log_returns, train_and_write_leg


class TestLogReturns(unittest.TestCase):
    def test_first_value_is_nan_not_zero(self):
        result = _log_returns(np.array([100.0, 101.0, 99.0]))
        self.assertTrue(np.isnan(result[0]))

    def test_matches_manual_log_diff(self):
        prices = np.array([100.0, 105.0, 98.0, 110.0])
        result = _log_returns(prices)
        manual = np.diff(np.log(prices))
        np.testing.assert_allclose(result[1:], manual, rtol=1e-12)


class TestTrainAndWriteLeg(unittest.TestCase):
    """Uses a REAL trained-and-promoted SupervisedPredictor (same fixture pattern as
    tests/test_supervised_predictor.py), mocking only write_predictions_leg (the one
    call requiring polars) -- everything upstream of that call is genuinely exercised."""

    def _make_promotable_inputs(self, seed=123, n=3000):
        rng = np.random.default_rng(seed)
        signal = rng.standard_normal(n)
        lagged_signal = np.roll(signal, 1)
        lagged_signal[0] = 0.0
        features = np.column_stack([signal, rng.standard_normal((n, 2))])
        returns = 0.05 * lagged_signal + rng.standard_normal(n) * 0.02
        base = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        # Spread timestamps across multiple calendar days to genuinely exercise the
        # by-date grouping logic (roughly 500 rows/day at this spacing).
        timestamps = [base + dt.timedelta(minutes=i) for i in range(n)]
        return timestamps, features, returns

    def test_rejected_model_writes_nothing(self):
        timestamps, features, returns = self._make_promotable_inputs()
        # Pure noise target -- should fail promotion.
        rng = np.random.default_rng(0)
        noise_returns = rng.standard_normal(len(returns)) * 0.02

        with patch("training.supervised_predictor.write_predictions_leg") as mock_write:
            result = train_and_write_leg(
                "EURUSD", timestamps, features, noise_returns,
                lookback=5, horizon=3, min_mean_edge=0.02, n_splits=5,
                model_out_dir=None, data_store_dir=None,
            )
        self.assertFalse(result)
        mock_write.assert_not_called()

    def test_promoted_model_calls_write_predictions_leg_once_per_calendar_date(self):
        timestamps, features, returns = self._make_promotable_inputs()

        with patch("training.supervised_predictor.write_predictions_leg") as mock_write:
            mock_write.return_value = "/fake/path.parquet"
            result = train_and_write_leg(
                "EURUSD", timestamps, features, returns,
                lookback=5, horizon=3, min_mean_edge=0.02, n_splits=5,
                model_out_dir=None, data_store_dir=None,
            )

        self.assertTrue(result, "fixture should have promoted -- same construction as test_supervised_predictor.py's own promotable fixture")
        self.assertGreater(mock_write.call_count, 0)

        # Verify every call's timestamps genuinely fall on ONE calendar date (the
        # grouping logic's actual job) and that direction_up_proba/expected_return
        # arrays are the same length as that date's timestamps.
        seen_dates = set()
        for call in mock_write.call_args_list:
            kwargs = call.kwargs
            dates_in_call = {ts.strftime("%Y-%m-%d") for ts in kwargs["timestamps"]}
            self.assertEqual(len(dates_in_call), 1, "each write_predictions_leg call must cover exactly one calendar date")
            date_str = next(iter(dates_in_call))
            self.assertNotIn(date_str, seen_dates, "no calendar date should be written more than once")
            seen_dates.add(date_str)
            self.assertEqual(kwargs["date"], date_str)
            self.assertEqual(len(kwargs["direction_up_proba"]), len(kwargs["timestamps"]))
            self.assertEqual(len(kwargs["expected_return"]), len(kwargs["timestamps"]))
            # direction_up_proba must be real probabilities.
            self.assertTrue(np.all((kwargs["direction_up_proba"] >= 0.0) & (kwargs["direction_up_proba"] <= 1.0)))


if __name__ == "__main__":
    unittest.main()
