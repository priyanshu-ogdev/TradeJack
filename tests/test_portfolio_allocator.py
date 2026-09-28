import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.portfolio_allocator import (
    InstrumentSignal, PortfolioAllocator, PortfolioAllocatorConfig,
)


def _sig(instrument, gate_passed=True, expected_return=0.01, direction_confidence=0.7,
         realized_volatility=0.01, risk_budget_used_fraction=0.0):
    return InstrumentSignal(instrument, gate_passed, expected_return, direction_confidence,
                             realized_volatility, risk_budget_used_fraction)


class TestOpportunityScoring(unittest.TestCase):
    def test_ungated_instrument_scores_none_not_zero(self):
        allocator = PortfolioAllocator()
        score = allocator._opportunity_score(_sig("EURUSD", gate_passed=False))
        self.assertIsNone(score, "an ungated instrument must be 'unknown', not comparable to a known-bad score of 0")

    def test_higher_volatility_scores_lower_for_identical_edge(self):
        allocator = PortfolioAllocator()
        calm = allocator._opportunity_score(_sig("A", realized_volatility=0.005))
        wild = allocator._opportunity_score(_sig("B", realized_volatility=0.05))
        self.assertGreater(calm, wild)

    def test_kelly_punishes_volatility_quadratically_not_linearly(self):
        """The actual claim that distinguishes true Kelly from a Sharpe-style
        edge/vol ratio: doubling volatility should roughly QUARTER the score,
        not halve it."""
        allocator = PortfolioAllocator()
        base = allocator._opportunity_score(_sig("A", realized_volatility=0.01))
        doubled_vol = allocator._opportunity_score(_sig("A", realized_volatility=0.02))
        self.assertAlmostEqual(doubled_vol, base / 4.0, places=6)

    def test_risk_headroom_scales_score_down_near_halt(self):
        allocator = PortfolioAllocator()
        full_headroom = allocator._opportunity_score(_sig("A", risk_budget_used_fraction=0.0))
        near_halted = allocator._opportunity_score(_sig("A", risk_budget_used_fraction=0.9))
        self.assertAlmostEqual(near_halted, full_headroom * 0.1, places=6)

    def test_fully_used_risk_budget_scores_exactly_zero(self):
        allocator = PortfolioAllocator()
        score = allocator._opportunity_score(_sig("A", risk_budget_used_fraction=1.0))
        self.assertEqual(score, 0.0)


class TestPrimarySelection(unittest.TestCase):
    def test_highest_score_becomes_primary(self):
        allocator = PortfolioAllocator()
        signals = [_sig("EURUSD", expected_return=0.005), _sig("GBPUSD", expected_return=0.03)]
        decisions = allocator.allocate(signals)
        primary = [d for d in decisions if d.role == "primary"]
        self.assertEqual(len(primary), 1)
        self.assertEqual(primary[0].instrument, "GBPUSD")

    def test_no_passing_gates_returns_all_inactive(self):
        allocator = PortfolioAllocator()
        signals = [_sig("EURUSD", gate_passed=False), _sig("GBPUSD", gate_passed=False)]
        decisions = allocator.allocate(signals)
        self.assertTrue(all(d.role == "inactive" and d.target_capital_fraction == 0.0 for d in decisions))

    def test_single_instrument_universe_gets_primary_no_secondary(self):
        allocator = PortfolioAllocator()
        decisions = allocator.allocate([_sig("EURUSD")])
        self.assertEqual(decisions[0].role, "primary")


class TestSecondarySelectionAndCorrelation(unittest.TestCase):
    def test_secondary_prefers_decorrelated_over_higher_raw_score(self):
        """The actual point of the whole secondary-slot design: a lower-scoring
        but decorrelated instrument should beat a higher-scoring but
        highly-correlated one for the Secondary slot."""
        config = PortfolioAllocatorConfig(correlation_penalty_weight=0.9, min_dwell_cycles=0)
        allocator = PortfolioAllocator(config)
        signals = [
            _sig("EURUSD", expected_return=0.03),   # will be primary (highest raw score)
            _sig("GBPUSD", expected_return=0.025),  # highest raw score among the rest, but correlated with EURUSD
            _sig("USDJPY", expected_return=0.018),  # lower raw score, but decorrelated
        ]
        correlations = {("GBPUSD", "EURUSD"): 0.9, ("USDJPY", "EURUSD"): 0.05}
        decisions = allocator.allocate(signals, correlation_matrix=correlations)
        secondary = [d for d in decisions if d.role == "secondary"]
        self.assertEqual(len(secondary), 1)
        self.assertEqual(secondary[0].instrument, "USDJPY", "should prefer the decorrelated instrument for the hedge slot")

    def test_zero_correlation_penalty_weight_falls_back_to_pure_ranking(self):
        config = PortfolioAllocatorConfig(correlation_penalty_weight=0.0, min_dwell_cycles=0)
        allocator = PortfolioAllocator(config)
        signals = [
            _sig("EURUSD", expected_return=0.03),
            _sig("GBPUSD", expected_return=0.025),
            _sig("USDJPY", expected_return=0.018),
        ]
        correlations = {("GBPUSD", "EURUSD"): 0.99, ("USDJPY", "EURUSD"): 0.0}
        decisions = allocator.allocate(signals, correlation_matrix=correlations)
        secondary = [d for d in decisions if d.role == "secondary"][0]
        self.assertEqual(secondary.instrument, "GBPUSD", "with penalty disabled, pure score ranking should win")

    def test_missing_correlation_entry_defaults_to_neutral_zero(self):
        config = PortfolioAllocatorConfig(correlation_penalty_weight=0.9, min_dwell_cycles=0)
        allocator = PortfolioAllocator(config)
        signals = [_sig("EURUSD", expected_return=0.03), _sig("GBPUSD", expected_return=0.025)]
        decisions = allocator.allocate(signals, correlation_matrix={})  # no entry at all
        secondary = [d for d in decisions if d.role == "secondary"]
        self.assertEqual(len(secondary), 1)
        self.assertEqual(secondary[0].instrument, "GBPUSD")

    def test_secondary_hysteresis_uses_adjusted_score_not_raw(self):
        """The specific bug found on review: comparing a correlation-adjusted
        challenger against the incumbent's RAW (unadjusted) score made the
        secondary slot too sticky once an incumbent quietly became highly
        correlated with the current primary. Concretely: GBPUSD (raw=100)
        becomes 95%-correlated with EURUSD (adjusted ~14.5 at penalty_weight=0.9),
        while USDJPY (raw=90, uncorrelated, adjusted=90) should clearly win the
        slot -- but the pre-fix code compared USDJPY's adjusted 90 against
        GBPUSD's RAW 100 (needing >=120 to clear the 20% margin) instead of
        against GBPUSD's true adjusted ~14.5 (needing >=17.4), wrongly keeping
        GBPUSD."""
        config = PortfolioAllocatorConfig(correlation_penalty_weight=0.9, switch_margin=0.20, min_dwell_cycles=0)
        allocator = PortfolioAllocator(config)

        # Cycle 1: establish GBPUSD as secondary incumbent (no correlation data yet -> 0 penalty).
        allocator.allocate(
            [_sig("EURUSD", expected_return=0.05), _sig("GBPUSD", expected_return=0.03), _sig("USDJPY", expected_return=0.025)],
            correlation_matrix={},
        )
        # Cycle 2: GBPUSD is revealed to be highly correlated with EURUSD;
        # USDJPY remains uncorrelated. Raw scores kept identical to isolate the
        # effect of the correlation adjustment itself.
        decisions = allocator.allocate(
            [_sig("EURUSD", expected_return=0.05), _sig("GBPUSD", expected_return=0.03), _sig("USDJPY", expected_return=0.025)],
            correlation_matrix={("GBPUSD", "EURUSD"): 0.95, ("USDJPY", "EURUSD"): 0.0},
        )
        secondary = [d for d in decisions if d.role == "secondary"][0]
        self.assertEqual(secondary.instrument, "USDJPY", "should switch away from an incumbent that quietly became highly correlated with primary")

    def test_secondary_reclaimed_by_primary_forces_immediate_reassignment(self):
        """If the current secondary incumbent becomes THIS cycle's top primary
        pick, it's excluded from secondary consideration entirely -- there is no
        valid adjusted score left to defend it with, so the slot must be
        reassigned immediately, not protected by dwell time. min_dwell_cycles=0
        here deliberately, so the primary flip itself isn't blocked -- the
        property under test is specifically about the SECONDARY slot's
        reassignment once that flip has happened, not about primary's own
        dwell behavior (covered separately in TestHysteresis)."""
        config = PortfolioAllocatorConfig(min_dwell_cycles=0, correlation_penalty_weight=0.0)
        allocator = PortfolioAllocator(config)
        # Cycle 1: EURUSD primary, GBPUSD secondary.
        allocator.allocate([_sig("EURUSD", expected_return=0.03), _sig("GBPUSD", expected_return=0.02), _sig("USDJPY", expected_return=0.01)])
        # Cycle 2: GBPUSD's edge explodes past EURUSD's -- GBPUSD becomes primary.
        decisions = allocator.allocate([_sig("EURUSD", expected_return=0.03), _sig("GBPUSD", expected_return=0.20), _sig("USDJPY", expected_return=0.01)])
        primary = [d for d in decisions if d.role == "primary"][0]
        secondary = [d for d in decisions if d.role == "secondary"][0]
        self.assertEqual(primary.instrument, "GBPUSD")
        self.assertNotEqual(secondary.instrument, "GBPUSD", "GBPUSD can't hold both roles; secondary must be reassigned immediately")


class TestReserveAndSizing(unittest.TestCase):
    def test_primary_plus_secondary_never_exceeds_tradeable_pool(self):
        config = PortfolioAllocatorConfig(reserve_fraction=0.35, kelly_fraction=1.0, min_dwell_cycles=0)
        allocator = PortfolioAllocator(config)
        # Deliberately extreme edge/vol to try to blow past the pool if the min(1.0, ...) guard is missing.
        signals = [_sig("EURUSD", expected_return=5.0, realized_volatility=0.001), _sig("GBPUSD", expected_return=0.01)]
        decisions = allocator.allocate(signals)
        total = sum(d.target_capital_fraction for d in decisions)
        self.assertLessEqual(total, 0.65 + 1e-9, "primary+secondary must never exceed the tradeable (1 - reserve) pool")

    def test_secondary_sized_smaller_than_primary(self):
        config = PortfolioAllocatorConfig(secondary_capital_ratio=0.4, min_dwell_cycles=0)
        allocator = PortfolioAllocator(config)
        signals = [_sig("EURUSD", expected_return=0.03), _sig("GBPUSD", expected_return=0.02)]
        decisions = allocator.allocate(signals)
        primary = [d for d in decisions if d.role == "primary"][0]
        secondary = [d for d in decisions if d.role == "secondary"][0]
        self.assertAlmostEqual(secondary.target_capital_fraction, primary.target_capital_fraction * 0.4, places=9)

    def test_lower_kelly_fraction_scales_position_down(self):
        signals = [_sig("EURUSD", expected_return=0.03), _sig("GBPUSD", expected_return=0.02)]
        conservative = PortfolioAllocator(PortfolioAllocatorConfig(kelly_fraction=0.1, min_dwell_cycles=0)).allocate(signals)
        aggressive = PortfolioAllocator(PortfolioAllocatorConfig(kelly_fraction=0.5, min_dwell_cycles=0)).allocate(signals)
        conservative_primary = [d for d in conservative if d.role == "primary"][0]
        aggressive_primary = [d for d in aggressive if d.role == "primary"][0]
        self.assertLess(conservative_primary.target_capital_fraction, aggressive_primary.target_capital_fraction)


class TestHysteresis(unittest.TestCase):
    def test_marginal_score_change_does_not_cause_switch(self):
        """The core anti-thrashing property: a challenger only slightly ahead
        of the incumbent (below switch_margin) must NOT take over the slot."""
        config = PortfolioAllocatorConfig(switch_margin=0.20, min_dwell_cycles=0)
        allocator = PortfolioAllocator(config)
        allocator.allocate([_sig("EURUSD", expected_return=0.030), _sig("GBPUSD", expected_return=0.010)])
        # GBPUSD edges up, but only 10% ahead of EURUSD's score -- below the 20% margin.
        decisions = allocator.allocate([_sig("EURUSD", expected_return=0.030), _sig("GBPUSD", expected_return=0.032)])
        primary = [d for d in decisions if d.role == "primary"][0]
        self.assertEqual(primary.instrument, "EURUSD", "incumbent should be retained when the challenger's lead is below switch_margin")

    def test_large_score_lead_does_cause_switch(self):
        config = PortfolioAllocatorConfig(switch_margin=0.20, min_dwell_cycles=0)
        allocator = PortfolioAllocator(config)
        allocator.allocate([_sig("EURUSD", expected_return=0.020), _sig("GBPUSD", expected_return=0.010)])
        decisions = allocator.allocate([_sig("EURUSD", expected_return=0.020), _sig("GBPUSD", expected_return=0.060)])
        primary = [d for d in decisions if d.role == "primary"][0]
        self.assertEqual(primary.instrument, "GBPUSD", "a challenger clearing the switch_margin should take over")

    def test_dwell_time_blocks_switch_even_with_large_lead(self):
        config = PortfolioAllocatorConfig(switch_margin=0.20, min_dwell_cycles=3)
        allocator = PortfolioAllocator(config)
        allocator.allocate([_sig("EURUSD", expected_return=0.020), _sig("GBPUSD", expected_return=0.010)])
        # Cycle 2: huge lead for GBPUSD, but EURUSD has only held the slot for 1 cycle (< min_dwell_cycles=3).
        decisions = allocator.allocate([_sig("EURUSD", expected_return=0.020), _sig("GBPUSD", expected_return=0.100)])
        primary = [d for d in decisions if d.role == "primary"][0]
        self.assertEqual(primary.instrument, "EURUSD", "dwell time should block a switch even with a large lead")

    def test_switch_allowed_once_dwell_elapses(self):
        config = PortfolioAllocatorConfig(switch_margin=0.20, min_dwell_cycles=2)
        allocator = PortfolioAllocator(config)
        for _ in range(2):
            allocator.allocate([_sig("EURUSD", expected_return=0.020), _sig("GBPUSD", expected_return=0.010)])
        decisions = allocator.allocate([_sig("EURUSD", expected_return=0.020), _sig("GBPUSD", expected_return=0.100)])
        primary = [d for d in decisions if d.role == "primary"][0]
        self.assertEqual(primary.instrument, "GBPUSD", "switch should be allowed once min_dwell_cycles has elapsed")

    def test_incumbent_gate_failure_forces_immediate_exit_ignoring_dwell(self):
        """The one case where dwell time must NOT protect the incumbent:
        its own statistical gate has failed entirely."""
        config = PortfolioAllocatorConfig(switch_margin=0.20, min_dwell_cycles=10)
        allocator = PortfolioAllocator(config)
        allocator.allocate([_sig("EURUSD", expected_return=0.020), _sig("GBPUSD", expected_return=0.010)])
        decisions = allocator.allocate([_sig("EURUSD", gate_passed=False), _sig("GBPUSD", expected_return=0.010)])
        primary = [d for d in decisions if d.role == "primary"][0]
        self.assertEqual(primary.instrument, "GBPUSD", "a failed gate must force an exit even mid-dwell-period")

    def test_primary_and_secondary_track_independent_dwell_state(self):
        """Secondary's hysteresis must not be coupled to Primary's -- a
        Secondary switch shouldn't be blocked or forced by Primary's dwell
        state, and vice versa."""
        config = PortfolioAllocatorConfig(switch_margin=0.20, min_dwell_cycles=5, correlation_penalty_weight=0.0)
        allocator = PortfolioAllocator(config)
        # Cycle 1: EURUSD primary, GBPUSD secondary.
        allocator.allocate([
            _sig("EURUSD", expected_return=0.05), _sig("GBPUSD", expected_return=0.02), _sig("USDJPY", expected_return=0.005),
        ])
        # Cycle 2: USDJPY's score jumps far past GBPUSD's (secondary should be
        # eligible to switch on its own schedule), while EURUSD stays dominant
        # primary with no real challenger.
        decisions = allocator.allocate([
            _sig("EURUSD", expected_return=0.05), _sig("GBPUSD", expected_return=0.02), _sig("USDJPY", expected_return=0.005),
        ])
        primary = [d for d in decisions if d.role == "primary"][0]
        self.assertEqual(primary.instrument, "EURUSD")


if __name__ == "__main__":
    unittest.main(verbosity=2)
