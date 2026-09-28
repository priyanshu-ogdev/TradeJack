"""
Real, executable, full-stack test for execution/live_composer.py -- a genuine
RiskGuardian, a genuine trained SignalPredictor, and a genuine PerformanceStreakTracker,
composed together with no mocks except the exchange side risk_guardian itself doesn't
need one for (check_order_allowed only touches self.state/self.limits).
"""
import time
import unittest
import numpy as np

from data_forge.predictor import SignalPredictor
from execution.risk_guardian import RiskGuardian
from execution.performance_tracker import PerformanceStreakTracker
from execution.composition_layer import SignalComposer
from execution.live_composer import LiveOrderComposer


def _make_trained_predictor(lookback=20, horizon=5, seed=7):
    rng = np.random.RandomState(seed)
    T, F = 1500, 3
    features = rng.randn(T, F) * 0.01
    price = np.zeros(T)
    price[0] = 100.0
    for t in range(1, T):
        lag_idx = max(0, t - lookback)
        drift = 0.004 * np.sign(features[lag_idx, 0])
        price[t] = price[t - 1] * (1 + drift + rng.randn() * 0.0005)

    predictor = SignalPredictor(lookback=lookback, horizon=horizon,
                                 up_threshold=0.0005, down_threshold=0.0005, n_splits=4)
    result = predictor.fit_and_gate(features, price)
    assert result.passed
    return predictor, features


class TestLiveComposerFullStack(unittest.TestCase):
    def setUp(self):
        self.predictor, self.features = _make_trained_predictor()
        window = self.features[-self.predictor.lookback:].flatten().reshape(1, -1)
        self.pred_dir = int(self.predictor.model.predict(window)[0])
        self.rg = RiskGuardian(starting_equity=10_000.0, max_daily_loss_pct=0.05)
        self.live = LiveOrderComposer(self.rg)

    def test_fresh_account_composes_a_real_order(self):
        intent = self.live.compose_order(
            rl_action=float(self.pred_dir), base_qty=10.0, price=1.10,
            predictor=self.predictor, feature_window=self.features,
        )
        self.assertEqual(intent.direction, self.pred_dir)
        self.assertGreater(intent.size_fraction, 0.0)
        # Fresh account: no risk-budget throttle, no performance throttle.
        self.assertEqual(intent.risk_budget_scalar, 1.0)
        self.assertEqual(intent.performance_scalar, 1.0)

    def test_real_daily_loss_actually_throttles_the_next_order(self):
        intent_before = self.live.compose_order(
            rl_action=float(self.pred_dir), base_qty=10.0, price=1.10,
            predictor=self.predictor, feature_window=self.features,
        )
        # A real 4% loss against a real 5% daily limit -- risk_guardian.update_equity()
        # is the same call a live P&L feed would make.
        self.rg.update_equity(9_600.0)
        intent_after = self.live.compose_order(
            rl_action=float(self.pred_dir), base_qty=10.0, price=1.10,
            predictor=self.predictor, feature_window=self.features,
        )
        self.assertLess(intent_after.risk_budget_scalar, intent_before.risk_budget_scalar)
        self.assertLess(intent_after.size_fraction, intent_before.size_fraction)

    def test_real_daily_loss_breach_produces_a_real_veto(self):
        self.rg.update_equity(9_400.0)  # 6% loss, breaches the 5% limit -> real halt
        self.assertTrue(self.rg.state.is_halted)
        intent = self.live.compose_order(
            rl_action=float(self.pred_dir), base_qty=10.0, price=1.10,
            predictor=self.predictor, feature_window=self.features,
        )
        self.assertEqual(intent.direction, 0)
        self.assertEqual(intent.bucket, "risk_vetoed")
        self.assertIn("halted", intent.risk_reason.lower())

    def test_record_fill_feeds_performance_tracker_into_next_order(self):
        for _ in range(6):
            self.live.record_fill(-1.0)  # 6 real losing fills
        intent = self.live.compose_order(
            rl_action=float(self.pred_dir), base_qty=10.0, price=1.10,
            predictor=self.predictor, feature_window=self.features,
        )
        self.assertLess(intent.performance_scalar, 1.0)
        self.assertEqual(self.live.performance_tracker.current_streak, -6)

    def test_reset_for_new_session_clears_streak_but_not_risk_guardian(self):
        for _ in range(6):
            self.live.record_fill(-1.0)
        self.rg.update_equity(9_600.0)
        self.live.reset_for_new_session()
        self.assertEqual(self.live.performance_tracker.current_streak, 0)
        # risk_guardian's own equity state is untouched by this call -- it's a separate
        # concern, deliberately not reset here.
        self.assertEqual(self.rg.state.current_equity, 9_600.0)


class _FakeToxicityBuilder:
    """Stands in for TrainingTableBuilder in tests -- returns a scripted sequence of
    readings and counts how many times it was actually called, so the cache-vs-refresh
    behavior can be asserted directly rather than inferred from timing."""

    def __init__(self, readings):
        self._readings = list(readings)
        self.calls = 0

    def latest_toxicity_reading(self, symbol):
        self.calls += 1
        idx = min(self.calls - 1, len(self._readings) - 1)
        return self._readings[idx]


class TestToxicityAutoLoad(unittest.TestCase):
    """Real execution against a fake builder (dependency-injected via
    toxicity_table_builder) -- proves the caching, None-handling, and
    explicit-override-always-wins behavior without needing polars/pydantic_settings
    at all, and without touching disk."""

    def setUp(self):
        self.rg = RiskGuardian(starting_equity=10_000.0, max_daily_loss_pct=0.05)
        self.predictor, self.features = _make_trained_predictor()

    def _live(self, fake_builder, ttl=300.0):
        return LiveOrderComposer(
            risk_guardian=self.rg, toxicity_symbol="BTC-USDT",
            toxicity_table_builder=fake_builder, toxicity_cache_ttl_seconds=ttl,
        )

    def test_auto_loads_when_bvc_vpin_not_supplied(self):
        fake = _FakeToxicityBuilder([0.6])  # above vpin_caution_threshold -> should throttle
        live = self._live(fake)
        intent_no_throttle = SignalComposer().compose(
            rl_action=1.0, base_qty=10.0, price=1.10, predictor=self.predictor,
            feature_window=self.features, risk_check_fn=lambda s, q, p: (True, None),
        )
        intent = live.compose_order(rl_action=1.0, base_qty=10.0, price=1.10,
                                     predictor=self.predictor, feature_window=self.features)
        self.assertEqual(fake.calls, 1)
        self.assertLess(intent.toxicity_scalar, intent_no_throttle.toxicity_scalar)

    def test_explicit_bvc_vpin_always_wins_over_auto_load(self):
        fake = _FakeToxicityBuilder([0.9])  # would throttle hard if used
        live = self._live(fake)
        intent = live.compose_order(rl_action=1.0, base_qty=10.0, price=1.10,
                                     predictor=self.predictor, feature_window=self.features,
                                     bvc_vpin=0.0)  # explicit override: no toxicity
        self.assertEqual(fake.calls, 0, "explicit bvc_vpin must short-circuit the auto-load entirely")
        self.assertEqual(intent.toxicity_scalar, 1.0)

    def test_repeated_calls_within_ttl_use_cache_not_a_fresh_read(self):
        fake = _FakeToxicityBuilder([0.6, 0.9, 0.9, 0.9])
        live = self._live(fake, ttl=300.0)
        for _ in range(4):
            live.compose_order(rl_action=1.0, base_qty=10.0, price=1.10,
                                predictor=self.predictor, feature_window=self.features)
        self.assertEqual(fake.calls, 1, "cached reading should serve all 4 calls within the TTL")

    def test_cache_refreshes_once_ttl_elapses(self):
        fake = _FakeToxicityBuilder([0.1, 0.9])
        live = self._live(fake, ttl=0.01)
        live.compose_order(rl_action=1.0, base_qty=10.0, price=1.10,
                            predictor=self.predictor, feature_window=self.features)
        time.sleep(0.02)
        live.compose_order(rl_action=1.0, base_qty=10.0, price=1.10,
                            predictor=self.predictor, feature_window=self.features)
        self.assertEqual(fake.calls, 2, "cache should refresh once the TTL has elapsed")

    def test_no_toxicity_symbol_configured_never_touches_the_builder(self):
        fake = _FakeToxicityBuilder([0.9])
        live = LiveOrderComposer(risk_guardian=self.rg, toxicity_table_builder=fake)  # toxicity_symbol left None
        live.compose_order(rl_action=1.0, base_qty=10.0, price=1.10,
                            predictor=self.predictor, feature_window=self.features)
        self.assertEqual(fake.calls, 0, "auto-load must stay fully opt-in via toxicity_symbol")


if __name__ == "__main__":
    unittest.main()
