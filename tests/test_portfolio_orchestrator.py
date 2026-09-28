import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.portfolio_allocator import PortfolioAllocator, PortfolioAllocatorConfig
from execution.portfolio_orchestrator import VolatilityCorrelationTracker, PortfolioOrchestrator


class TestVolatilityCorrelationTracker(unittest.TestCase):
    def test_no_reading_until_min_observations(self):
        tracker = VolatilityCorrelationTracker(window=50, min_observations=10)
        for i in range(9):
            tracker.update("EURUSD", 1.10 + i * 0.0001)
        self.assertIsNone(tracker.realized_volatility("EURUSD"), "9 returns is one short of min_observations=10")

    def test_volatility_matches_known_synthetic_series(self):
        """Feed a price series with a KNOWN log-return standard deviation and
        confirm the tracker recovers it, not just 'returns something'."""
        rng = np.random.default_rng(0)
        true_vol = 0.01
        log_returns = rng.normal(0, true_vol, 500)
        prices = 100.0 * np.exp(np.cumsum(log_returns))

        tracker = VolatilityCorrelationTracker(window=500, min_observations=10)
        tracker.update("EURUSD", 100.0)
        for p in prices:
            tracker.update("EURUSD", p)

        measured = tracker.realized_volatility("EURUSD")
        self.assertAlmostEqual(measured, true_vol, delta=true_vol * 0.15)  # within 15% of the true generating vol

    def test_window_trims_old_observations(self):
        tracker = VolatilityCorrelationTracker(window=20, min_observations=5)
        price = 100.0
        # Feed a long, calm run first...
        for _ in range(100):
            tracker.update("EURUSD", price)
            price *= 1.00001
        calm_vol = tracker.realized_volatility("EURUSD")
        # ...then a short, violent run -- only the last 20 observations should
        # matter once the window is full.
        for _ in range(20):
            price *= 1.05
            tracker.update("EURUSD", price)
        volatile_vol = tracker.realized_volatility("EURUSD")
        self.assertGreater(volatile_vol, calm_vol * 10, "old calm history should have been trimmed out of the window")

    def test_non_positive_price_ignored_not_crashing(self):
        tracker = VolatilityCorrelationTracker()
        tracker.update("EURUSD", 1.10)
        tracker.update("EURUSD", -5.0)  # must not raise, must not corrupt state
        tracker.update("EURUSD", 1.11)
        self.assertIsNotNone(tracker._last_price.get("EURUSD"))

    def test_correlation_recovers_known_relationship(self):
        rng = np.random.default_rng(1)
        n = 300
        shared = rng.normal(0, 0.01, n)
        noise_a = rng.normal(0, 0.001, n)
        noise_b = rng.normal(0, 0.001, n)
        returns_a = shared + noise_a          # highly correlated with `shared`
        returns_b = -shared + noise_b         # highly NEGATIVELY correlated with `shared`, so with A too
        returns_c = rng.normal(0, 0.01, n)    # independent

        tracker = VolatilityCorrelationTracker(window=n + 1, min_observations=10)
        price_a = price_b = price_c = 100.0
        tracker.update("A", price_a); tracker.update("B", price_b); tracker.update("C", price_c)
        for ra, rb, rc in zip(returns_a, returns_b, returns_c):
            price_a *= np.exp(ra); price_b *= np.exp(rb); price_c *= np.exp(rc)
            tracker.update("A", price_a); tracker.update("B", price_b); tracker.update("C", price_c)

        corr = tracker.correlation_matrix()
        self.assertLess(corr[("A", "B")], -0.7, "A and B were constructed to be strongly negatively correlated")
        self.assertAlmostEqual(corr[("A", "B")], corr[("B", "A")], places=9, msg="correlation must be symmetric")
        self.assertLess(abs(corr[("A", "C")]), 0.3, "C was constructed to be independent of A")

    def test_constant_price_series_excluded_not_crashing(self):
        tracker = VolatilityCorrelationTracker(min_observations=5)
        for _ in range(20):
            tracker.update("FLAT", 1.0)  # zero variance -- log returns are all exactly 0
            tracker.update("EURUSD", 1.10 + np.random.default_rng(2).normal(0, 0.001))
        corr = tracker.correlation_matrix()
        self.assertNotIn(("FLAT", "EURUSD"), corr, "a zero-variance series has undefined correlation and must be skipped, not divide-by-zero")


class _FakeSource:
    def __init__(self, gate_passed, expected_return, direction_confidence, risk_budget_used_fraction, prices):
        self.gate_passed = gate_passed
        self.expected_return = expected_return
        self.direction_confidence = direction_confidence
        self.risk_budget_used_fraction = risk_budget_used_fraction
        self._prices = list(prices)
        self._idx = 0

    @property
    def current_price(self):
        p = self._prices[min(self._idx, len(self._prices) - 1)]
        self._idx += 1
        return p


class TestPortfolioOrchestrator(unittest.TestCase):
    def _warm_prices(self, n=15, start=100.0, seed=0):
        rng = np.random.default_rng(seed)
        return list(start * np.exp(np.cumsum(rng.normal(0, 0.005, n))))

    def test_insufficient_history_treated_as_ungated_even_if_source_says_passed(self):
        sources = {"EURUSD": _FakeSource(True, 0.02, 0.8, 0.0, [1.10, 1.101])}  # only 2 prices -- not enough for volatility
        orchestrator = PortfolioOrchestrator(PortfolioAllocator(), sources)
        result = orchestrator.run_cycle()
        self.assertEqual(result.decisions[0].role, "inactive")
        self.assertFalse(result.signals[0].gate_passed, "an instrument with no volatility reading yet must be treated as ungated regardless of its own source")

    def test_full_cycle_selects_primary_and_secondary_from_real_sources(self):
        sources = {
            "EURUSD": _FakeSource(True, 0.03, 0.8, 0.0, self._warm_prices(seed=1)),
            "GBPUSD": _FakeSource(True, 0.02, 0.7, 0.0, self._warm_prices(seed=2)),
            "USDJPY": _FakeSource(False, 0.0, 0.0, 0.0, self._warm_prices(seed=3)),
        }
        orchestrator = PortfolioOrchestrator(
            PortfolioAllocator(PortfolioAllocatorConfig(min_dwell_cycles=0)), sources,
            vol_tracker=VolatilityCorrelationTracker(window=20, min_observations=10),
        )
        # run_cycle() pulls exactly one price per source per call (one real
        # market tick) -- warming up the rolling window means calling it
        # repeatedly, the same way it would actually be driven in production,
        # not expecting one call to consume an entire warm-up price history.
        for _ in range(len(self._warm_prices())):
            result = orchestrator.run_cycle()
        roles = {d.instrument: d.role for d in result.decisions}
        self.assertEqual(roles["USDJPY"], "inactive", "ungated source must never be selected")
        self.assertIn("primary", roles.values())
        self.assertTrue(len(result.correlation_matrix) > 0)

    def test_source_exception_does_not_crash_the_cycle(self):
        class _BrokenSource(_FakeSource):
            @property
            def gate_passed(self):
                raise RuntimeError("simulated live-feed failure")

            @gate_passed.setter
            def gate_passed(self, value):
                pass

        sources = {
            "EURUSD": _BrokenSource(True, 0.03, 0.8, 0.0, self._warm_prices(seed=4)),
            "GBPUSD": _FakeSource(True, 0.02, 0.7, 0.0, self._warm_prices(seed=5)),
        }
        orchestrator = PortfolioOrchestrator(
            PortfolioAllocator(PortfolioAllocatorConfig(min_dwell_cycles=0)), sources,
            vol_tracker=VolatilityCorrelationTracker(window=20, min_observations=10),
        )
        result = orchestrator.run_cycle()  # must not raise
        eurusd_decision = [d for d in result.decisions if d.instrument == "EURUSD"][0]
        self.assertEqual(eurusd_decision.role, "inactive", "a source that raises must degrade to ungated, not crash the cycle")

    def test_persistence_round_trip(self):
        import tempfile, shutil
        tmp_dir = tempfile.mkdtemp(prefix="portfolio_state_test_")
        try:
            state_path = os.path.join(tmp_dir, "PORTFOLIO_ALLOCATION.json")
            sources = {
                "EURUSD": _FakeSource(True, 0.03, 0.8, 0.0, self._warm_prices(seed=6)),
                "GBPUSD": _FakeSource(True, 0.02, 0.7, 0.0, self._warm_prices(seed=7)),
            }
            orchestrator = PortfolioOrchestrator(
                PortfolioAllocator(PortfolioAllocatorConfig(min_dwell_cycles=0)), sources,
                vol_tracker=VolatilityCorrelationTracker(window=20, min_observations=10),
                state_path=state_path,
            )
            self.assertIsNone(PortfolioOrchestrator.read_latest_allocation(state_path), "nothing persisted yet")
            for _ in range(len(self._warm_prices())):
                orchestrator.run_cycle()
            self.assertTrue(os.path.exists(state_path))

            loaded = PortfolioOrchestrator.read_latest_allocation(state_path)
            self.assertIsNotNone(loaded)
            self.assertIn("decisions", loaded)
            self.assertIn("generated_at", loaded)
            roles = {d["instrument"]: d["role"] for d in loaded["decisions"]}
            self.assertIn("primary", roles.values())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_persistence_disabled_by_default(self):
        """state_path=None (the default) must never attempt file I/O at all."""
        sources = {"EURUSD": _FakeSource(True, 0.03, 0.8, 0.0, self._warm_prices(seed=8))}
        orchestrator = PortfolioOrchestrator(
            PortfolioAllocator(PortfolioAllocatorConfig(min_dwell_cycles=0)), sources,
            vol_tracker=VolatilityCorrelationTracker(window=20, min_observations=10),
        )
        for _ in range(len(self._warm_prices())):
            orchestrator.run_cycle()  # must not raise even though no state_path was given


if __name__ == "__main__":
    unittest.main(verbosity=2)
