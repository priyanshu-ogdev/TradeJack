"""
PortfolioAllocator -- decides which ONE instrument gets Primary capital, which ONE
gets Secondary (hedge) capital, and how much of total equity each gets, out of a
universe of instruments each already scored by their own independent
SignalPredictor/SupervisedPredictor stack (data_forge/predictor.py,
training/supervised_predictor.py -- both already built and gated this session).

This module deliberately does NOT touch physics/lob_env.py, TradeJackLOBEnv, or any
per-instrument RL policy. Each instrument's agent keeps reacting to its own order
flow exactly as before; this sits one layer above, deciding capital allocation
only. See the design discussion this module was built from: RL policy = "react to
my own instrument's order flow," portfolio allocator = "decide where capital sits
right now given relative risk-adjusted opportunity" -- two different timescales,
two different information sets, deliberately kept as two different components
rather than asking one policy to solve both (the same reasoning
training/scenario_injection.py used to avoid touching TradeJackLOBEnv for scenario
diversity: reuse what's proven, add a layer, don't rearchitect the core).

The math: true Kelly (edge / variance), not a Sharpe-style edge / volatility ratio --
deliberately, because the stated goal is capital growth that survives volatility,
not risk-adjusted ranking for its own sake. Kelly is the unique sizing rule that
maximizes long-run compounded growth rate while structurally avoiding ruin, which is
the formal name for "grow hungrily but sustainably." Run at a fraction of full Kelly
(default 0.25) because full Kelly is only correct if the edge estimate is exact,
which it never is -- this is the standard practitioner discount for estimation
error, not an arbitrary safety margin.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (PortfolioAllocator) %(message)s")
logger = logging.getLogger("PortfolioAllocator")


@dataclass
class InstrumentSignal:
    """One instrument's current state, as already produced by the existing
    pipeline -- nothing here is computed by this module."""
    instrument: str
    gate_passed: bool
    expected_return: float          # SupervisedPredictor's regression head; signed
    direction_confidence: float     # SignalPredictor's predict_proba for its called direction, 0..1
    realized_volatility: float      # rolling stdev of LOG returns, same lookback as the predictor -- NOT
                                     # raw price volatility. Log-return space is what makes this number
                                     # comparable across instruments with different price scales (EURUSD's
                                     # 4th-decimal moves vs BTC-USDT's hundred-dollar moves are not
                                     # comparable in raw terms; they are in log-return terms). Getting this
                                     # wrong silently breaks every comparison this module makes.
    risk_budget_used_fraction: float  # from that instrument's own RiskGuardian.risk_budget_used_fraction()


@dataclass
class AllocationDecision:
    instrument: str
    role: str                       # "primary" | "secondary" | "inactive"
    target_capital_fraction: float  # fraction of TOTAL equity, not of the tradeable pool
    opportunity_score: Optional[float]
    reason: str


@dataclass
class PortfolioAllocatorConfig:
    reserve_fraction: float = 0.35
    # Never allocated, full stop, before Kelly sizing is even computed. Sits ABOVE
    # each instrument's own RiskGuardian -- two independent layers of protection.
    kelly_fraction: float = 0.25
    # Fraction of full Kelly actually bet. Full Kelly assumes an exact edge
    # estimate; this is the standard practitioner discount for estimation error.
    secondary_capital_ratio: float = 0.4
    # Secondary's allocation, as a fraction of what Primary gets (not of the
    # tradeable pool directly) -- Secondary is sized as a smaller hedge position,
    # not a second equal bet.
    correlation_penalty_weight: float = 0.6
    # How hard a Secondary candidate is penalized for correlation with Primary.
    # 0.0 = ignore correlation entirely (picks purely by raw score --
    # reduces to "second-best bet", NOT a hedge). 1.0 = a perfectly-correlated
    # (corr=1.0) candidate is fully zeroed out as a Secondary pick.
    switch_margin: float = 0.20
    # A challenger must beat the incumbent's score by this fraction before a
    # swap happens -- prevents thrashing/switching on noise between rescoring
    # cycles that are each individually noisy.
    min_dwell_cycles: int = 5
    # Minimum cycles an incumbent holds its role before it can be challenged at
    # all (ignored if the incumbent's own gate fails -- see allocate()).
    min_score_floor: float = 1e-9
    # Scores at or below this are treated as "no real opportunity" -- guards
    # against a near-zero-but-technically-positive score being selected as
    # Primary just because it's the least-bad of a bad set.


class PortfolioAllocator:
    def __init__(self, config: Optional[PortfolioAllocatorConfig] = None):
        self.config = config or PortfolioAllocatorConfig()
        self._role_incumbent: Dict[str, str] = {}  # "primary"/"secondary" -> instrument (role -> instrument)
        self._cycles_in_role: Dict[str, int] = {}  # instrument -> consecutive cycles held in its CURRENT role
        self._cycle_count = 0

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def _opportunity_score(self, sig: InstrumentSignal) -> Optional[float]:
        """None means "no opinion available" -- an ungated instrument is UNKNOWN,
        not "known to be bad" (a bad-but-gated instrument would score near zero
        and rank low; an ungated one must never be comparable to that at all).
        This mirrors the same "absence is neutral, not evidence" discipline used
        throughout composition_layer.py and data_forge/training_table_builder.py."""
        if not sig.gate_passed:
            return None
        vol = max(sig.realized_volatility, 1e-9)
        edge = abs(sig.expected_return) * max(0.0, min(1.0, sig.direction_confidence))
        raw_kelly = edge / (vol ** 2)
        headroom = max(0.0, 1.0 - sig.risk_budget_used_fraction)
        score = raw_kelly * headroom
        return score if score > self.config.min_score_floor else 0.0

    def _correlation_penalty_multiplier(self, correlation: float) -> float:
        """1.0 at correlation 0 (no penalty), down to (1 - correlation_penalty_weight)
        at correlation +/-1. Uses abs(correlation) -- a Secondary that moves exactly
        OPPOSITE Primary is just as unhelpful a "hedge" as one that moves in lockstep
        for THIS purpose (this allocator wants a decorrelated second bet to survive
        an unrelated adverse event in Primary's instrument, not an intentional
        offsetting hedge -- that would be a different, deliberate strategy, not
        something to fall into by only penalizing positive correlation)."""
        c = max(-1.0, min(1.0, correlation))
        return max(0.0, 1.0 - self.config.correlation_penalty_weight * abs(c))

    # ------------------------------------------------------------------ #
    # Hysteresis
    # ------------------------------------------------------------------ #

    def _select_with_hysteresis(
        self, role: str, ranked: List[Tuple[str, float]], signals_by_instrument: Dict[str, InstrumentSignal],
        excluded: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[float]]:
        """ranked: [(instrument, score), ...] sorted descending, already excluding
        `excluded` (used so Primary's selection never considers the instrument
        already reserved for the other slot when called for Secondary, avoiding a
        one-cycle ordering dependency between the two selections)."""
        candidates = [(i, s) for i, s in ranked if i != excluded]
        if not candidates:
            return None, None

        top_instrument, top_score = candidates[0]
        incumbent = self._role_incumbent.get(role)

        if incumbent is None or incumbent not in signals_by_instrument:
            return top_instrument, top_score

        # The incumbent's score for comparison purposes MUST come from THIS
        # cycle's `candidates` list, not a fresh raw _opportunity_score() call --
        # for the secondary role, `candidates` is already correlation-adjusted
        # against the current primary, so comparing an adjusted challenger
        # against a freshly-recomputed RAW incumbent score would be comparing
        # apples to oranges. Concretely: an incumbent secondary that has quietly
        # become highly correlated with the current primary would still show its
        # old, unadjusted (inflated) score under the raw computation, making it
        # look artificially strong against challengers and making the slot too
        # sticky -- caught by test_secondary_hysteresis_uses_adjusted_score_not_raw.
        #
        # If the incumbent isn't in `candidates` at all this cycle (gate failed,
        # OR -- for the secondary role specifically -- it's now the top primary
        # pick and was excluded from secondary consideration entirely), there is
        # no valid adjusted score to defend it with: treat that exactly like a
        # failed gate. No dwell-time protection, reassign immediately.
        candidate_scores = dict(candidates)
        incumbent_score = candidate_scores.get(incumbent)
        if incumbent_score is None:
            return top_instrument, top_score

        dwell = self._cycles_in_role.get(incumbent, 0)
        if dwell < self.config.min_dwell_cycles:
            # Too soon to challenge -- keep the incumbent regardless of the
            # challenger's score (we already know the incumbent has a valid
            # score this cycle, since incumbent_score would have short-circuited
            # above otherwise).
            return incumbent, incumbent_score

        if top_instrument == incumbent:
            return incumbent, incumbent_score

        if top_score >= incumbent_score * (1.0 + self.config.switch_margin):
            return top_instrument, top_score

        return incumbent, incumbent_score

    # ------------------------------------------------------------------ #
    # Allocation
    # ------------------------------------------------------------------ #

    def allocate(
        self, signals: List[InstrumentSignal], correlation_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> List[AllocationDecision]:
        """correlation_matrix: {(a, b): correlation} for any pair, either order --
        looked up both ways, missing pairs treated as 0.0 (unknown correlation is
        NOT treated as "safely decorrelated" by inflating it to a bad number, nor
        as "definitely correlated" -- 0.0 is the neutral, no-penalty assumption,
        consistent with "absence is neutral" elsewhere in this design; supply real
        correlations when available rather than relying on this default)."""
        self._cycle_count += 1
        correlation_matrix = correlation_matrix or {}
        signals_by_instrument = {s.instrument: s for s in signals}

        scored = [(s.instrument, self._opportunity_score(s)) for s in signals]
        rankable = sorted([(i, sc) for i, sc in scored if sc is not None], key=lambda x: -x[1])

        if not rankable or rankable[0][1] <= 0.0:
            self._role_incumbent = {}
            self._cycles_in_role = {}
            return [
                AllocationDecision(s.instrument, "inactive", 0.0, dict(scored).get(s.instrument), "no instrument currently clears the opportunity floor")
                for s in signals
            ]

        primary_instrument, primary_score = self._select_with_hysteresis("primary", rankable, signals_by_instrument)

        secondary_ranked_raw = [(i, sc) for i, sc in rankable if i != primary_instrument]
        secondary_ranked_adjusted = []
        for instrument, score in secondary_ranked_raw:
            corr = correlation_matrix.get((instrument, primary_instrument), correlation_matrix.get((primary_instrument, instrument), 0.0))
            adjusted = score * self._correlation_penalty_multiplier(corr)
            secondary_ranked_adjusted.append((instrument, adjusted))
        secondary_ranked_adjusted.sort(key=lambda x: -x[1])

        secondary_instrument, secondary_score = self._select_with_hysteresis(
            "secondary", secondary_ranked_adjusted, signals_by_instrument, excluded=primary_instrument,
        )

        # Update role/dwell bookkeeping for next cycle's hysteresis check. Two
        # separate mappings, deliberately: _role_incumbent (role -> instrument) is
        # what _select_with_hysteresis reads back; _cycles_in_role (instrument ->
        # count) tracks how long THAT instrument has held ITS CURRENT role,
        # resetting to 1 if its role changed (e.g. secondary promoted to primary)
        # rather than carrying over a dwell count earned in a different role.
        new_role_incumbent = {}
        if primary_instrument is not None:
            new_role_incumbent["primary"] = primary_instrument
        if secondary_instrument is not None:
            new_role_incumbent["secondary"] = secondary_instrument

        prev_role_by_instrument = {i: r for r, i in self._role_incumbent.items()}
        new_dwell = {}
        for role, instrument in new_role_incumbent.items():
            prev_role = prev_role_by_instrument.get(instrument)
            new_dwell[instrument] = (self._cycles_in_role.get(instrument, 0) + 1) if prev_role == role else 1
        self._role_incumbent = new_role_incumbent
        self._cycles_in_role = new_dwell

        tradeable_fraction = 1.0 - self.config.reserve_fraction
        # Primary is capped so that Primary + Secondary (Secondary is sized as
        # secondary_capital_ratio OF Primary, added on top) can never together
        # exceed the tradeable pool -- capping Primary alone against the full
        # pool and adding Secondary on top of that would let the two jointly
        # blow past the reserve ceiling (caught by
        # test_primary_plus_secondary_never_exceeds_tradeable_pool with an
        # extreme edge/volatility ratio).
        max_primary_fraction = (
            tradeable_fraction / (1.0 + self.config.secondary_capital_ratio)
            if secondary_instrument is not None else tradeable_fraction
        )
        primary_fraction = (
            min(max_primary_fraction, tradeable_fraction * self.config.kelly_fraction * min(1.0, primary_score))
            if primary_instrument else 0.0
        )
        secondary_fraction = primary_fraction * self.config.secondary_capital_ratio if secondary_instrument else 0.0

        decisions = []
        for s in signals:
            score = dict(scored).get(s.instrument)
            if s.instrument == primary_instrument:
                decisions.append(AllocationDecision(s.instrument, "primary", primary_fraction, score, "highest Kelly-adjusted opportunity score"))
            elif s.instrument == secondary_instrument:
                decisions.append(AllocationDecision(s.instrument, "secondary", secondary_fraction, score, "best correlation-adjusted opportunity after primary"))
            else:
                reason = "gate not passing" if score is None else "not selected for primary or secondary this cycle"
                decisions.append(AllocationDecision(s.instrument, "inactive", 0.0, score, reason))
        return decisions
