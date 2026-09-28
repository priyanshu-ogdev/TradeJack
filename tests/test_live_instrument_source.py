import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.live_instrument_source import LiveInstrumentSource
from execution.portfolio_allocator import PortfolioAllocator, PortfolioAllocatorConfig
from execution.portfolio_orchestrator import PortfolioOrchestrator, VolatilityCorrelationTracker


class _FakeModel:
    """Stands in for a fitted sklearn HistGradientBoostingClassifier -- real
    predict()/predict_proba()/classes_ shape, scripted output."""
    def __init__(self, predicted_class=1, proba_for_class=0.8, classes=(-1, 0, 1)):
        self.predicted_class = predicted_class
        self.proba_for_class = proba_for_class
        self.classes_ = list(classes)
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        return np.array([self.predicted_class])

    def predict_proba(self, X):
        proba = np.full(len(self.classes_), (1.0 - self.proba_for_class) / (len(self.classes_) - 1))
        proba[self.classes_.index(self.predicted_class)] = self.proba_for_class
        return np.array([proba])


class _FakeDirectionPredictor:
    def __init__(self, model, lookback=5):
        self.model = model
        self.lookback = lookback


class _FakeReturnPredictor:
    def __init__(self, expected_return=0.02):
        self.expected_return = expected_return
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        return {"expected_return": np.array([self.expected_return])}


class _FakeRiskGuardian:
    def __init__(self, used_fraction=0.1, raise_error=False):
        self.used_fraction = used_fraction
        self.raise_error = raise_error

    def risk_budget_used_fraction(self):
        if self.raise_error:
            raise RuntimeError("simulated risk guardian failure")
        return self.used_fraction


def _window(lookback=5, features=2):
    return np.random.default_rng(0).standard_normal((lookback, features))


class TestLiveInstrumentSourceSnapshot(unittest.TestCase):
    def test_current_price_returns_the_price(self):
        src = LiveInstrumentSource(
            "EURUSD", _FakeDirectionPredictor(None), lambda: None, _FakeRiskGuardian(), price_fn=lambda: 1.105,
        )
        self.assertEqual(src.current_price, 1.105)

    def test_gate_passed_false_when_model_is_none(self):
        src = LiveInstrumentSource(
            "EURUSD", _FakeDirectionPredictor(None), lambda: _window(), _FakeRiskGuardian(), price_fn=lambda: 1.1,
        )
        src.current_price  # trigger snapshot
        self.assertFalse(src.gate_passed)

    def test_gate_passed_false_when_predicted_class_is_flat(self):
        model = _FakeModel(predicted_class=0)
        src = LiveInstrumentSource(
            "EURUSD", _FakeDirectionPredictor(model), lambda: _window(), _FakeRiskGuardian(), price_fn=lambda: 1.1,
        )
        src.current_price
        self.assertFalse(src.gate_passed, "a 'flat' (class 0) prediction must not be treated as a directional opinion")

    def test_gate_passed_true_and_confidence_correct_when_directional(self):
        model = _FakeModel(predicted_class=1, proba_for_class=0.73)
        src = LiveInstrumentSource(
            "EURUSD", _FakeDirectionPredictor(model), lambda: _window(), _FakeRiskGuardian(), price_fn=lambda: 1.1,
        )
        src.current_price
        self.assertTrue(src.gate_passed)
        self.assertAlmostEqual(src.direction_confidence, 0.73, places=9)

    def test_insufficient_feature_window_treated_as_ungated(self):
        model = _FakeModel(predicted_class=1)
        src = LiveInstrumentSource(
            "EURUSD", _FakeDirectionPredictor(model, lookback=10),
            lambda: _window(lookback=3),  # shorter than predictor.lookback=10
            _FakeRiskGuardian(), price_fn=lambda: 1.1,
        )
        src.current_price
        self.assertFalse(src.gate_passed)
        self.assertEqual(model.calls, 0, "must not even call predict() on an insufficient window")

    def test_properties_read_before_current_price_degrade_safely(self):
        model = _FakeModel(predicted_class=1, proba_for_class=0.9)
        src = LiveInstrumentSource(
            "EURUSD", _FakeDirectionPredictor(model), lambda: _window(), _FakeRiskGuardian(), price_fn=lambda: 1.1,
        )
        # No current_price access yet.
        self.assertFalse(src.gate_passed)
        self.assertEqual(src.direction_confidence, 0.0)
        self.assertEqual(src.expected_return, 0.0)


class TestExpectedReturn(unittest.TestCase):
    def test_zero_without_a_return_predictor(self):
        model = _FakeModel(predicted_class=1)
        src = LiveInstrumentSource(
            "EURUSD", _FakeDirectionPredictor(model), lambda: _window(), _FakeRiskGuardian(), price_fn=lambda: 1.1,
        )
        src.current_price
        self.assertTrue(src.gate_passed)
        self.assertEqual(src.expected_return, 0.0, "absence of a return predictor must mean 0.0, not a fabricated magnitude")

    def test_uses_return_predictor_when_gated(self):
        model = _FakeModel(predicted_class=1)
        return_predictor = _FakeReturnPredictor(expected_return=0.045)
        src = LiveInstrumentSource(
            "EURUSD", _FakeDirectionPredictor(model), lambda: _window(), _FakeRiskGuardian(), price_fn=lambda: 1.1,
            return_predictor=return_predictor, return_feature_window_fn=lambda: _window(features=3).flatten(),
        )
        src.current_price
        self.assertAlmostEqual(src.expected_return, 0.045, places=9)
        self.assertEqual(return_predictor.calls, 1)

    def test_return_predictor_not_called_when_not_gated(self):
        """Correctness AND efficiency: no reason to run a regression predictor
        for an instrument with no directional opinion at all."""
        model = _FakeModel(predicted_class=0)  # flat -- not gated
        return_predictor = _FakeReturnPredictor(expected_return=0.09)
        src = LiveInstrumentSource(
            "EURUSD", _FakeDirectionPredictor(model), lambda: _window(), _FakeRiskGuardian(), price_fn=lambda: 1.1,
            return_predictor=return_predictor, return_feature_window_fn=lambda: _window(features=3).flatten(),
        )
        src.current_price
        self.assertEqual(src.expected_return, 0.0)
        self.assertEqual(return_predictor.calls, 0)


class TestFailureModesDegradeSafely(unittest.TestCase):
    def test_direction_prediction_exception_degrades_to_ungated(self):
        class _BrokenModel(_FakeModel):
            def predict(self, X):
                raise RuntimeError("simulated model failure")

        src = LiveInstrumentSource(
            "EURUSD", _FakeDirectionPredictor(_BrokenModel()), lambda: _window(), _FakeRiskGuardian(), price_fn=lambda: 1.1,
        )
        price = src.current_price  # must not raise
        self.assertEqual(price, 1.1)
        self.assertFalse(src.gate_passed)

    def test_return_prediction_exception_leaves_expected_return_zero(self):
        model = _FakeModel(predicted_class=1)

        class _BrokenReturnPredictor(_FakeReturnPredictor):
            def predict(self, X):
                raise RuntimeError("simulated regressor failure")

        src = LiveInstrumentSource(
            "EURUSD", _FakeDirectionPredictor(model), lambda: _window(), _FakeRiskGuardian(), price_fn=lambda: 1.1,
            return_predictor=_BrokenReturnPredictor(), return_feature_window_fn=lambda: _window().flatten(),
        )
        src.current_price  # must not raise
        self.assertTrue(src.gate_passed)  # direction side still worked
        self.assertEqual(src.expected_return, 0.0)

    def test_risk_budget_exception_fails_safe_to_fully_used(self):
        model = _FakeModel(predicted_class=1)
        src = LiveInstrumentSource(
            "EURUSD", _FakeDirectionPredictor(model), lambda: _window(), _FakeRiskGuardian(raise_error=True), price_fn=lambda: 1.1,
        )
        src.current_price
        self.assertEqual(src.risk_budget_used_fraction, 1.0, "an unreadable risk state must look fully-used, never free headroom")


class TestFullOrchestratorIntegration(unittest.TestCase):
    """The actual point of this whole class: prove PortfolioOrchestrator ->
    LiveInstrumentSource -> (predictor, risk guardian) works end to end, not
    just that each piece works in isolation."""

    def test_end_to_end_cycle_produces_sensible_allocation(self):
        prices_a = list(100.0 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.005, 20))))
        prices_b = list(100.0 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.005, 20))))
        idx = {"a": 0, "b": 0}

        def make_price_fn(key, prices):
            def _fn():
                i = min(idx[key], len(prices) - 1)
                idx[key] += 1
                return prices[i]
            return _fn

        strong_model = _FakeModel(predicted_class=1, proba_for_class=0.85)
        weak_model = _FakeModel(predicted_class=1, proba_for_class=0.55)

        sources = {
            "EURUSD": LiveInstrumentSource(
                "EURUSD", _FakeDirectionPredictor(strong_model, lookback=3), lambda: _window(lookback=3),
                _FakeRiskGuardian(used_fraction=0.1), price_fn=make_price_fn("a", prices_a),
                return_predictor=_FakeReturnPredictor(0.03), return_feature_window_fn=lambda: _window(lookback=3).flatten(),
            ),
            "GBPUSD": LiveInstrumentSource(
                "GBPUSD", _FakeDirectionPredictor(weak_model, lookback=3), lambda: _window(lookback=3),
                _FakeRiskGuardian(used_fraction=0.1), price_fn=make_price_fn("b", prices_b),
                return_predictor=_FakeReturnPredictor(0.01), return_feature_window_fn=lambda: _window(lookback=3).flatten(),
            ),
        }

        orchestrator = PortfolioOrchestrator(
            PortfolioAllocator(PortfolioAllocatorConfig(min_dwell_cycles=0)), sources,
            vol_tracker=VolatilityCorrelationTracker(window=20, min_observations=10),
        )

        for _ in range(20):
            result = orchestrator.run_cycle()

        roles = {d.instrument: d.role for d in result.decisions}
        self.assertIn("primary", roles.values(), "a clearly stronger, gated signal should win a role")
        # EURUSD has both higher confidence and higher expected_return -- should be primary.
        self.assertEqual(roles["EURUSD"], "primary")


if __name__ == "__main__":
    unittest.main(verbosity=2)
