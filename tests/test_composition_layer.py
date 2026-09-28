"""
Real, executable tests for execution/composition_layer.py. Trains an actual
SignalPredictor (sklearn) on a genuine lagged relationship so the "agree"/"disagree"
paths are tested against a real trained model, not a mock -- same as test_predictor.py.
"""
import unittest
import numpy as np

from data_forge.predictor import SignalPredictor
from execution.composition_layer import SignalComposer, CompositionConfig


def _make_trained_predictor(lookback=20, horizon=5, seed=7):
    """Builds and gates a real predictor on a genuine lagged relationship, mirroring
    test_predictor.py's fixture exactly (deliberately reused, not reinvented, since
    that construction was already checked for the leakage mistake)."""
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
    assert result.passed, f"Fixture predictor failed to gate: {result.reason}"
    return predictor, features


def _agreeing_action_and_confidence(predictor, features):
    window = features[-predictor.lookback:].flatten().reshape(1, -1)
    pred_dir = int(predictor.model.predict(window)[0])
    proba = predictor.model.predict_proba(window)[0]
    classes = list(predictor.model.classes_)
    confidence = float(proba[classes.index(pred_dir)])
    return pred_dir, confidence


ALWAYS_ALLOW = lambda side, qty, price: (True, None)
ALWAYS_DENY = lambda side, qty, price: (False, "synthetic veto for test")


class TestFlatRLNeverBecomesATrade(unittest.TestCase):
    def test_flat_action_produces_no_trade_even_with_confident_predictor(self):
        predictor, features = _make_trained_predictor()
        composer = SignalComposer()
        intent = composer.compose(
            rl_action=0.01,  # inside deadzone
            base_qty=100.0, price=1.10,
            predictor=predictor, feature_window=features,
            risk_check_fn=ALWAYS_ALLOW,
        )
        self.assertEqual(intent.direction, 0)
        self.assertEqual(intent.bucket, "flat_rl")
        self.assertEqual(intent.size_fraction, 0.0)


class TestGreedyAxis(unittest.TestCase):
    """The "greedy" half: real, above-threshold confidence should earn MORE than the
    flat agree_multiplier used to give -- this is the actual upgrade over the old
    fixed-bucket version."""

    def setUp(self):
        self.predictor, self.features = _make_trained_predictor()
        self.pred_dir, self.confidence = _agreeing_action_and_confidence(self.predictor, self.features)

    def test_high_confidence_agreement_exceeds_flat_agree_multiplier(self):
        config = CompositionConfig()
        composer = SignalComposer(config)
        intent = composer.compose(
            rl_action=float(self.pred_dir), base_qty=100.0, price=1.10,
            predictor=self.predictor, feature_window=self.features, risk_check_fn=ALWAYS_ALLOW,
        )
        self.assertEqual(intent.bucket, "agree")
        self.assertEqual(intent.direction, self.pred_dir)
        # This is the actual "greedy" assertion: confidence above the trust threshold
        # earns size ABOVE the flat baseline, up to (but never past) greedy_ceiling.
        self.assertGreaterEqual(intent.size_fraction, config.agree_multiplier)
        self.assertLessEqual(intent.size_fraction, config.greedy_ceiling)
        if self.confidence > config.min_predictor_confidence:
            self.assertGreater(intent.size_fraction, config.agree_multiplier)

    def test_disabling_greedy_axis_falls_back_to_flat_multiplier(self):
        # greedy_ceiling == agree_multiplier disables the boost entirely -- old behavior.
        config = CompositionConfig(greedy_ceiling=1.0, agree_multiplier=1.0)
        composer = SignalComposer(config)
        intent = composer.compose(
            rl_action=float(self.pred_dir), base_qty=100.0, price=1.10,
            predictor=self.predictor, feature_window=self.features, risk_check_fn=ALWAYS_ALLOW,
        )
        self.assertAlmostEqual(intent.size_fraction, 1.0, places=6)

    def test_size_never_exceeds_hard_cap_even_with_maximal_config(self):
        # Deliberately misconfigured (greedy_ceiling > max_size_fraction) to prove the
        # final clip is a correctness guarantee independent of the tuning knobs.
        config = CompositionConfig(greedy_ceiling=10.0, max_size_fraction=1.2)
        composer = SignalComposer(config)
        intent = composer.compose(
            rl_action=float(self.pred_dir), base_qty=100.0, price=1.10,
            predictor=self.predictor, feature_window=self.features, risk_check_fn=ALWAYS_ALLOW,
        )
        self.assertLessEqual(intent.size_fraction, 1.2)


class TestCautiousAxes(unittest.TestCase):
    """The "cautious" half: even a high-confidence agreement must be throttled down as
    the risk budget depletes or toxicity rises -- greed alone never overrides caution."""

    def setUp(self):
        self.predictor, self.features = _make_trained_predictor()
        self.pred_dir, self.confidence = _agreeing_action_and_confidence(self.predictor, self.features)
        self.composer = SignalComposer()

    def test_risk_budget_throttle_reduces_size_on_agreement(self):
        intent_full_budget = self.composer.compose(
            rl_action=float(self.pred_dir), base_qty=100.0, price=1.10,
            predictor=self.predictor, feature_window=self.features, risk_check_fn=ALWAYS_ALLOW,
            risk_budget_used_fraction=0.0,
        )
        intent_near_limit = self.composer.compose(
            rl_action=float(self.pred_dir), base_qty=100.0, price=1.10,
            predictor=self.predictor, feature_window=self.features, risk_check_fn=ALWAYS_ALLOW,
            risk_budget_used_fraction=0.95,
        )
        self.assertLess(intent_near_limit.risk_budget_scalar, intent_full_budget.risk_budget_scalar)
        self.assertLess(intent_near_limit.size_fraction, intent_full_budget.size_fraction)
        # Near-exhausted budget should pull a greedy-eligible agreement back down to a
        # small fraction, not just slightly below its uncapped value.
        self.assertLess(intent_near_limit.size_fraction, 1.0)

    def test_toxicity_throttle_reduces_size_on_agreement(self):
        intent_calm = self.composer.compose(
            rl_action=float(self.pred_dir), base_qty=100.0, price=1.10,
            predictor=self.predictor, feature_window=self.features, risk_check_fn=ALWAYS_ALLOW,
            bvc_vpin=0.05,
        )
        intent_toxic = self.composer.compose(
            rl_action=float(self.pred_dir), base_qty=100.0, price=1.10,
            predictor=self.predictor, feature_window=self.features, risk_check_fn=ALWAYS_ALLOW,
            bvc_vpin=0.9,
        )
        self.assertLess(intent_toxic.toxicity_scalar, intent_calm.toxicity_scalar)
        self.assertLess(intent_toxic.size_fraction, intent_calm.size_fraction)

    def test_both_throttles_combine_multiplicatively(self):
        intent = self.composer.compose(
            rl_action=float(self.pred_dir), base_qty=100.0, price=1.10,
            predictor=self.predictor, feature_window=self.features, risk_check_fn=ALWAYS_ALLOW,
            risk_budget_used_fraction=0.9, bvc_vpin=0.9,
        )
        expected_upper_bound = intent.risk_budget_scalar * intent.toxicity_scalar * CompositionConfig().greedy_ceiling
        self.assertLessEqual(intent.size_fraction, expected_upper_bound + 1e-6)
        self.assertGreater(intent.size_fraction, 0.0)  # still trades, just much smaller

    def test_absent_risk_and_toxicity_readings_do_not_penalize(self):
        intent = self.composer.compose(
            rl_action=float(self.pred_dir), base_qty=100.0, price=1.10,
            predictor=self.predictor, feature_window=self.features, risk_check_fn=ALWAYS_ALLOW,
        )
        self.assertEqual(intent.risk_budget_scalar, 1.0)
        self.assertEqual(intent.toxicity_scalar, 1.0)


class TestDisagreementIsInverted(unittest.TestCase):
    """A configured disagree_multiplier floor should shrink further as disagreement
    confidence rises -- never grow. Default disagree_multiplier=0.0 stays 0.0 regardless."""

    def setUp(self):
        self.predictor, self.features = _make_trained_predictor()
        self.pred_dir, self.confidence = _agreeing_action_and_confidence(self.predictor, self.features)
        self.opposite_rl_action = float(-self.pred_dir)

    def test_default_disagree_multiplier_stays_zero(self):
        composer = SignalComposer()
        intent = composer.compose(
            rl_action=self.opposite_rl_action, base_qty=100.0, price=1.10,
            predictor=self.predictor, feature_window=self.features, risk_check_fn=ALWAYS_ALLOW,
        )
        self.assertEqual(intent.bucket, "disagree")
        self.assertEqual(intent.direction, 0)
        self.assertEqual(intent.size_fraction, 0.0)

    def test_configured_disagree_floor_shrinks_with_confidence_not_grows(self):
        # With a nonzero floor, higher-confidence disagreement must leave LESS of the
        # floor intact than a hypothetical lower-confidence one would -- verified here
        # by checking the composed size sits strictly below the configured floor itself
        # whenever real confidence exceeds the minimum threshold (norm_conf > 0).
        config = CompositionConfig(disagree_multiplier=0.3)
        composer = SignalComposer(config)
        intent = composer.compose(
            rl_action=self.opposite_rl_action, base_qty=100.0, price=1.10,
            predictor=self.predictor, feature_window=self.features, risk_check_fn=ALWAYS_ALLOW,
        )
        self.assertEqual(intent.bucket, "disagree")
        if self.confidence > config.min_predictor_confidence:
            self.assertLess(intent.size_fraction, config.disagree_multiplier)


class TestRiskVetoOverridesEverything(unittest.TestCase):
    def test_risk_veto_zeroes_even_a_greedily_scaled_agreeing_trade(self):
        predictor, features = _make_trained_predictor()
        pred_dir, _ = _agreeing_action_and_confidence(predictor, features)
        composer = SignalComposer()

        intent = composer.compose(
            rl_action=float(pred_dir), base_qty=100.0, price=1.10,
            predictor=predictor, feature_window=features,
            risk_check_fn=ALWAYS_DENY,
        )
        self.assertEqual(intent.direction, 0)
        self.assertEqual(intent.bucket, "risk_vetoed")
        self.assertFalse(intent.risk_allowed)
        self.assertIn("synthetic veto", intent.risk_reason)


class TestPredictorLockedInterlock(unittest.TestCase):
    def test_locked_predictor_blocks_the_order_regardless_of_rl_action(self):
        composer = SignalComposer()
        intent = composer.compose(
            rl_action=1.0, base_qty=100.0, price=1.10,
            predictor=None, feature_window=None,
            risk_check_fn=ALWAYS_ALLOW, predictor_locked=True,
        )
        self.assertEqual(intent.direction, 0)
        self.assertEqual(intent.bucket, "predictor_locked")


class TestPerformanceScalarAxis(unittest.TestCase):
    """The fifth axis: a caller-supplied performance_scalar (typically from
    PerformanceStreakTracker.scalar()) combines multiplicatively with everything else,
    and absence of it is neutral -- same convention as risk_budget/toxicity."""

    def setUp(self):
        self.predictor, self.features = _make_trained_predictor()
        self.pred_dir, self.confidence = _agreeing_action_and_confidence(self.predictor, self.features)
        self.composer = SignalComposer()

    def test_absent_performance_scalar_is_neutral(self):
        intent = self.composer.compose(
            rl_action=float(self.pred_dir), base_qty=100.0, price=1.10,
            predictor=self.predictor, feature_window=self.features, risk_check_fn=ALWAYS_ALLOW,
        )
        self.assertEqual(intent.performance_scalar, 1.0)

    def test_losing_streak_scalar_throttles_a_greedy_agreement(self):
        from execution.performance_tracker import PerformanceStreakTracker
        tracker = PerformanceStreakTracker()
        for _ in range(6):
            tracker.record_trade_outcome(-1.0)

        intent_no_streak = self.composer.compose(
            rl_action=float(self.pred_dir), base_qty=100.0, price=1.10,
            predictor=self.predictor, feature_window=self.features, risk_check_fn=ALWAYS_ALLOW,
        )
        intent_losing_streak = self.composer.compose(
            rl_action=float(self.pred_dir), base_qty=100.0, price=1.10,
            predictor=self.predictor, feature_window=self.features, risk_check_fn=ALWAYS_ALLOW,
            performance_scalar=tracker.scalar(),
        )
        self.assertEqual(intent_losing_streak.performance_scalar, tracker.scalar())
        self.assertLess(intent_losing_streak.size_fraction, intent_no_streak.size_fraction)

    def test_performance_scalar_clipped_to_max_size_fraction(self):
        # A pathological caller-supplied value above max_size_fraction must not blow
        # past the hard cap once combined with everything else.
        intent = self.composer.compose(
            rl_action=float(self.pred_dir), base_qty=100.0, price=1.10,
            predictor=self.predictor, feature_window=self.features, risk_check_fn=ALWAYS_ALLOW,
            performance_scalar=100.0,
        )
        self.assertLessEqual(intent.size_fraction, CompositionConfig().max_size_fraction)


class TestUnconfirmedBucket(unittest.TestCase):
    def test_no_predictor_uses_flat_unconfirmed_multiplier_not_greedy_scaled(self):
        composer = SignalComposer()
        config = CompositionConfig()
        intent = composer.compose(
            rl_action=1.0, base_qty=100.0, price=1.10,
            predictor=None, feature_window=None, risk_check_fn=ALWAYS_ALLOW,
        )
        self.assertEqual(intent.bucket, "unconfirmed")
        self.assertEqual(intent.direction, 1)
        self.assertAlmostEqual(intent.size_fraction, config.unconfirmed_multiplier)
        self.assertFalse(intent.predictor_gate_passed)

    def test_unconfirmed_still_subject_to_risk_budget_throttle(self):
        composer = SignalComposer()
        intent = composer.compose(
            rl_action=1.0, base_qty=100.0, price=1.10,
            predictor=None, feature_window=None, risk_check_fn=ALWAYS_ALLOW,
            risk_budget_used_fraction=0.95,
        )
        self.assertEqual(intent.bucket, "unconfirmed")
        self.assertLess(intent.size_fraction, CompositionConfig().unconfirmed_multiplier)


if __name__ == "__main__":
    unittest.main()
