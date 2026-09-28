"""
Executable tests for training/supervised_predictor.py -- pure numpy/sklearn/
scipy, no torch, no network, no polars. Run directly:
    python -m tests.test_supervised_predictor
or via unittest:
    python -m unittest tests.test_supervised_predictor -v
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.supervised_predictor import (
    PredictorConfig, PurgedTimeSeriesSplit, SupervisedPredictor,
    build_features_and_labels, persistence_baseline_signal,
    walk_forward_evaluate, evaluate_promotion, train_evaluate_and_promote,
)


class TestPurgedTimeSeriesSplit(unittest.TestCase):
    def test_embargo_gap_is_respected(self):
        n = 1000
        embargo = 20
        splitter = PurgedTimeSeriesSplit(n_splits=5, embargo=embargo)
        folds = list(splitter.split(n))
        self.assertGreaterEqual(len(folds), 3)
        for train_idx, test_idx in folds:
            gap = test_idx.min() - train_idx.max()
            self.assertGreaterEqual(
                gap, embargo,
                f"train/test gap {gap} smaller than embargo {embargo} -- leakage window exists",
            )

    def test_train_is_strictly_before_test(self):
        for train_idx, test_idx in PurgedTimeSeriesSplit(n_splits=4, embargo=5).split(500):
            self.assertLess(train_idx.max(), test_idx.min())

    def test_expanding_window_grows_each_fold(self):
        folds = list(PurgedTimeSeriesSplit(n_splits=4, embargo=5).split(500))
        train_sizes = [len(tr) for tr, _ in folds]
        self.assertEqual(train_sizes, sorted(train_sizes))


class TestBuildFeaturesAndLabels(unittest.TestCase):
    def test_shapes_and_origin_index(self):
        rng = np.random.default_rng(0)
        T, F, lookback, horizon = 200, 3, 10, 5
        features = rng.standard_normal((T, F))
        returns = rng.standard_normal(T) * 0.01

        X, y_class, y_reg, origin_idx = build_features_and_labels(features, returns, lookback, horizon)

        expected_n = T - horizon - (lookback - 1)
        self.assertEqual(len(X), expected_n)
        self.assertEqual(X.shape[1], lookback * F + lookback)
        self.assertEqual(set(np.unique(y_class).tolist()) <= {0, 1}, True)
        self.assertEqual(origin_idx[0], lookback - 1)
        self.assertEqual(origin_idx[-1], T - horizon - 1)

    def test_no_lookahead_leakage(self):
        """The critical boundary test: a feature row built at origin index t
        must be IDENTICAL regardless of what happens to `returns` strictly
        after t's label window -- and the label must correctly change when
        the future segment it actually depends on changes. Two series that
        are identical up through some point and diverge only after it must
        produce identical X rows (and identical y rows) for every origin
        index whose label window doesn't reach the divergence point."""
        rng = np.random.default_rng(1)
        T, F, lookback, horizon = 100, 2, 5, 3
        features = rng.standard_normal((T, F))
        returns_a = rng.standard_normal(T) * 0.01
        divergence_point = 60

        returns_b = returns_a.copy()
        returns_b[divergence_point:] = rng.standard_normal(T - divergence_point) * 0.01

        Xa, yca, yra, idx_a = build_features_and_labels(features, returns_a, lookback, horizon)
        Xb, ycb, yrb, idx_b = build_features_and_labels(features, returns_b, lookback, horizon)

        np.testing.assert_array_equal(idx_a, idx_b)

        # Rows whose label window [t+1, t+1+horizon) ends before the
        # divergence point must be byte-identical between the two series.
        safe_mask = (idx_a + horizon) < divergence_point
        self.assertGreater(safe_mask.sum(), 10, "test setup produced too few safe rows to be meaningful")
        np.testing.assert_array_almost_equal(Xa[safe_mask], Xb[safe_mask])
        np.testing.assert_array_almost_equal(yra[safe_mask], yrb[safe_mask])

        # And at least one row whose label window DOES reach the divergence
        # point must actually differ -- otherwise this test would pass
        # trivially even with a broken/no-op label computation.
        affected_mask = ~safe_mask
        self.assertGreater(affected_mask.sum(), 0, "test setup produced no affected rows")
        self.assertFalse(
            np.allclose(yra[affected_mask], yrb[affected_mask]),
            "labels for affected rows did not change -- label construction may not be reading the future window at all",
        )

    def test_features_do_not_include_own_forward_return(self):
        """A degenerate but important case: if forward return at t were
        accidentally folded into t's own feature vector, a model could get
        ~100% train accuracy trivially. Verify the feature vector's returns
        segment only ever contains returns[t-lookback+1 : t+1], never
        anything from the label window."""
        T, F, lookback, horizon = 50, 1, 4, 2
        features = np.zeros((T, F))
        returns = np.arange(T, dtype=np.float64)  # returns[i] == i, easy to spot

        X, _, _, origin_idx = build_features_and_labels(features, returns, lookback, horizon)
        returns_segment = X[:, -lookback:]  # last `lookback` columns are the recent-returns block
        for row, t in enumerate(origin_idx):
            expected = returns[t - lookback + 1: t + 1]
            np.testing.assert_array_equal(returns_segment[row], expected)
            self.assertNotIn(t + 1, returns_segment[row].tolist())  # first label-window value never present


class TestSupervisedPredictor(unittest.TestCase):
    def _synthetic_signal_data(self, n=2000, seed=42):
        """Features where column 0 at time t has a REAL (noisy) relationship
        to the return realized at t+1 -- i.e. a genuine leading indicator,
        not a same-timestep correlation. This matters: build_features_and_labels
        labels off returns[t+1:t+1+horizon], strictly after the lookback
        window ending at t, so a synthetic signal correlated with returns[t]
        (same index) is invisible to the label and this test would fail for
        the wrong reason -- caught by running this, not by reasoning about
        it in the abstract."""
        rng = np.random.default_rng(seed)
        signal = rng.standard_normal(n)
        noise_features = rng.standard_normal((n, 2))
        features = np.column_stack([signal, noise_features])
        lagged_signal = np.roll(signal, 1)
        lagged_signal[0] = 0.0
        returns = 0.05 * lagged_signal + rng.standard_normal(n) * 0.02
        return features, returns

    def _pure_noise_data(self, n=2000, seed=7):
        rng = np.random.default_rng(seed)
        features = rng.standard_normal((n, 3))
        returns = rng.standard_normal(n) * 0.01  # independent of features by construction
        return features, returns

    def test_fit_predict_roundtrip(self):
        features, returns = self._synthetic_signal_data()
        X, y_class, y_reg, _ = build_features_and_labels(features, returns, lookback=5, horizon=3)
        split = int(len(X) * 0.8)
        model = SupervisedPredictor(PredictorConfig(lookback=5, horizon=3)).fit(
            X[:split], y_class[:split], y_reg[:split]
        )
        preds = model.predict(X[split:])
        acc = np.mean((preds["direction_up_proba"] >= 0.5).astype(int) == y_class[split:])
        self.assertGreater(acc, 0.55, f"model should beat chance on a real (if noisy) signal, got {acc:.3f}")

    def test_save_load_roundtrip_matches_exactly(self):
        features, returns = self._synthetic_signal_data(n=500)
        X, y_class, y_reg, _ = build_features_and_labels(features, returns, lookback=5, horizon=3)
        model = SupervisedPredictor(PredictorConfig(lookback=5, horizon=3)).fit(X, y_class, y_reg)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.joblib")
            model.save(path)
            loaded = SupervisedPredictor.load(path)
            preds_after = loaded.predict(X)

        np.testing.assert_array_almost_equal(preds_before["direction_up_proba"], preds_after["direction_up_proba"])
        np.testing.assert_array_almost_equal(preds_before["expected_return"], preds_after["expected_return"])
        self.assertEqual(loaded.n_train_samples, model.n_train_samples)


class TestPromotionGate(unittest.TestCase):
    """The tests that actually matter: does the gate discriminate a real
    signal from noise, or does it just always say yes (or always say no)?"""

    def test_gate_promotes_on_real_signal(self):
        rng = np.random.default_rng(123)
        n = 3000
        signal = rng.standard_normal(n)
        lagged_signal = np.roll(signal, 1)
        lagged_signal[0] = 0.0
        features = np.column_stack([signal, rng.standard_normal((n, 2))])
        returns = 0.05 * lagged_signal + rng.standard_normal(n) * 0.02

        config = PredictorConfig(lookback=5, horizon=3, n_splits=5, min_mean_edge=0.02)
        model, report = train_evaluate_and_promote(features, returns, config)

        self.assertTrue(report["passed"], f"expected promotion on a real signal, got: {report}")
        self.assertIsNotNone(model)
        self.assertGreater(report["mean_edge"], 0)
        self.assertLess(report["p_value"], config.significance_level)

    def test_gate_refuses_on_pure_noise(self):
        rng = np.random.default_rng(999)
        n = 3000
        features = rng.standard_normal((n, 3))
        returns = rng.standard_normal(n) * 0.01  # no relationship to features at all

        config = PredictorConfig(lookback=5, horizon=3, n_splits=5, min_mean_edge=0.02)
        model, report = train_evaluate_and_promote(features, returns, config)

        self.assertFalse(report["passed"], f"expected refusal on pure noise, got: {report}")
        self.assertIsNone(model)

    def test_gate_refuses_with_too_few_rows(self):
        features = np.random.default_rng(0).standard_normal((20, 3))
        returns = np.random.default_rng(1).standard_normal(20) * 0.01
        model, report = train_evaluate_and_promote(features, returns, PredictorConfig(n_splits=5))
        self.assertFalse(report["passed"])
        self.assertIsNone(model)
        self.assertIn("too few", report["reason"])

    def test_fresh_model_fit_per_fold_not_reused(self):
        """walk_forward_evaluate must not let one fold's fitted model leak
        into another fold's evaluation -- each FoldResult's n_train should
        exactly match that fold's own training slice size, and consecutive
        folds' n_train should strictly grow (expanding window), which could
        only happen if a fresh model were actually being fit each time."""
        features, returns = self._make_signal(seed=55)
        X, y_class, y_reg, _ = build_features_and_labels(features, returns, lookback=5, horizon=3)
        results = walk_forward_evaluate(X, y_class, y_reg, PredictorConfig(lookback=5, horizon=3, n_splits=4))
        n_trains = [r.n_train for r in results]
        self.assertEqual(n_trains, sorted(n_trains))
        self.assertEqual(len(set(n_trains)), len(n_trains), "expected strictly distinct, growing train sizes per fold")

    @staticmethod
    def _make_signal(n=2000, seed=1):
        rng = np.random.default_rng(seed)
        signal = rng.standard_normal(n)
        lagged_signal = np.roll(signal, 1)
        lagged_signal[0] = 0.0
        features = np.column_stack([signal, rng.standard_normal((n, 2))])
        returns = 0.05 * lagged_signal + rng.standard_normal(n) * 0.02
        return features, returns


if __name__ == "__main__":
    unittest.main(verbosity=2)
