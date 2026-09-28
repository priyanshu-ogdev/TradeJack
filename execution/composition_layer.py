"""
Signal Composition Layer.

Ties together the three decision-relevant pieces that don't talk to each other yet:
the RL policy's action, the supervised predictor's (gated) opinion, and risk_guardian's
veto. The design principles, stated explicitly because getting the order of operations
wrong here is exactly the kind of bug that looks fine in backtest and blows up live:

1. THE PREDICTOR IS A FILTER, NEVER A SIGNAL GENERATOR. It can only scale an RL-proposed
   trade — up, down, or to zero — never manufacture a trade the RL policy didn't propose.
   If the RL action is flat (direction 0), composition returns flat immediately, full
   stop, regardless of what the predictor thinks. This is the same shape as
   risk_guardian.check_order_allowed(): a downstream check narrows or reweights what
   upstream proposed, never invents a new proposal.

2. THE PREDICTOR'S GATE STATUS MUST BE CHECKED HERE, NOT ASSUMED. `SignalPredictor.model`
   is None until `fit_and_gate()` has actually passed (see data_forge/predictor.py). A
   predictor that hasn't passed its gate — or whose current confidence on this specific
   window is too low to trust — is treated as "no opinion available" (UNCONFIRMED
   bucket), not as "predicts flat".

3. RISK_GUARDIAN IS A HARD VETO ON THE FINAL, ALREADY-COMPOSED ORDER — NOT ONE VOICE IN
   AN ENSEMBLE. It runs last, against the actual (already-scaled) qty that would be sent,
   using its own existing check_order_allowed(side, qty, price) signature unchanged.

4. NO ORDER IS PROPOSED IF THE PREDICTOR IS MID-RESET (`predictor_locked=True`), the
   PlasticityManager interlock.

5. SIZING IS CONTINUOUS, NOT A FIXED LOOKUP TABLE — this is the upgrade this revision
   makes. The old version picked one of three flat multipliers (1.0 / 0.5 / 0.0) purely
   from which bucket a trade fell into. That's cautious by construction but never greedy:
   a 99%-confidence agreement with a wide-open risk budget and a calm market got exactly
   the same 1.0x as a 56%-confidence agreement one tick above the trust threshold with a
   nearly-exhausted risk budget. Four independent, multiplicatively-combined scalars fix
   that — each one bounded and independently testable, rather than one opaque formula:

     a) GREEDY axis -- confidence-scaled agreement bonus. On the "agree" bucket only, size
        scales continuously from `agree_multiplier` (at the trust threshold) up to
        `greedy_ceiling` (at full predictor confidence). This is the actual "greedy" half:
        real, above-baseline conviction earns a larger position, bounded by
        `greedy_ceiling` so greed is never unbounded.
     b) CAUTIOUS axis 1 -- risk-budget throttle. As the fraction of the daily loss budget
        already used climbs past `risk_budget_caution_start`, size ramps down toward
        `risk_budget_min_scalar`. This is a PRE-EMPTIVE, graduated throttle that
        complements risk_guardian's binary halt -- it makes the desk quieter as the day's
        cushion shrinks, rather than trading at full size right up until the hard veto
        fires.
     c) CAUTIOUS axis 2 -- toxicity throttle. If a `bvc_vpin`-style toxicity reading is
        supplied (see data_forge/forex_toxicity_engineering.py), size ramps down as
        informed-trading risk rises past `vpin_caution_threshold` -- adverse selection is
        a real, currently-elevated cost, not a hypothetical one.
     d) DISAGREEMENT is inverted relative to (a): if `disagree_multiplier` is configured
        above its 0.0 default (some strategies may want a small floor rather than an
        outright skip), a MORE confident disagreement shrinks that floor further, never
        grows it -- confidently disagreeing is never a reason to size up.

   All four scalars are clipped to sane ranges and combined multiplicatively, then the
   whole thing is clipped again to `[0, max_size_fraction]` as a final hard cap -- the
   individual scalars are a design choice (see docs/COMPOSITION_LAYER.md for the
   reasoning and why these specific defaults are a starting point, not a backtested-
   optimal setting), but the hard cap is a correctness guarantee independent of them.

VERIFIED BY REAL EXECUTION (sklearn + numpy are available in this sandbox, unlike
torch/polars): tests/test_composition_layer.py trains an actual SignalPredictor and
runs it through SignalComposer, asserting the greedy scaling actually exceeds 1.0x under
favorable conditions, that risk-budget/toxicity throttling actually pulls it back down
even on a high-confidence agreement, and that the hard cap and all the original
categorical guarantees (flat-RL, disagree, risk-veto, predictor-locked) still hold.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from data_forge.predictor import SignalPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (Composer) %(message)s")
logger = logging.getLogger("Composer")

# side, qty, price -> (allowed, reason). Matches RiskGuardian.check_order_allowed's
# actual signature exactly, so a live caller can pass that bound method directly with
# no adapter needed.
RiskCheckFn = Callable[[str, float, float], Tuple[bool, Optional[str]]]


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class CompositionConfig:
    """See docs/COMPOSITION_LAYER.md for the full reasoning behind each default. All
    are a deliberately conservative starting point, not a backtested-optimal setting."""

    direction_deadzone: float = 0.05
    # Below this |rl_action|, the RL policy is treated as proposing no trade at all.

    # --- Greedy axis: confidence-scaled agreement bonus ---
    agree_multiplier: float = 1.0
    # Size at the AGREE bucket when the predictor's confidence is exactly at
    # min_predictor_confidence (the minimum to be trusted at all).
    greedy_ceiling: float = 1.5
    # Size at the AGREE bucket when the predictor's confidence is 1.0 (maximal). Set
    # equal to agree_multiplier to disable the greedy boost entirely and fall back to
    # the old fixed-1.0x behavior.

    # --- Baseline / disagreement ---
    unconfirmed_multiplier: float = 0.5
    # Predictor's gate hasn't passed, confidence is below threshold, or it confidently
    # predicts flat while RL wants to trade. Not confidence-scaled (there's no reliable
    # confidence signal to scale by in this bucket) but still subject to the two
    # cautious-axis throttles below.
    disagree_multiplier: float = 0.0
    # Floor size retained on a CONFIDENT disagreement, before that floor is itself
    # shrunk further as confidence rises (see module docstring, point 5d). Default 0.0
    # means disagreement always zeroes the trade regardless of confidence.

    min_predictor_confidence: float = 0.55
    # Below this predict_proba confidence, the predictor's opinion is discarded
    # entirely (routed to UNCONFIRMED) rather than trusted at a low weight.

    # --- Cautious axis 1: risk-budget throttle ---
    risk_budget_caution_start: float = 0.5
    # Fraction (0-1) of the daily-loss budget already used before this pre-emptive
    # throttle starts ramping size down. Below this, no throttle applied (scalar=1.0).
    risk_budget_min_scalar: float = 0.15
    # Scalar floor as risk_budget_used_fraction approaches 1.0 (budget fully used).
    # Deliberately not 0.0 -- the actual hard stop is risk_guardian's binary veto; this
    # is a graduated pre-emptive throttle that complements it, not a substitute for it.

    # --- Cautious axis 2: toxicity throttle ---
    vpin_caution_threshold: float = 0.3
    # BVC-VPIN reading (0-1, see forex_toxicity_engineering.py) above which the
    # toxicity throttle starts ramping size down.
    vpin_min_scalar: float = 0.3
    # Scalar floor as vpin approaches 1.0 (maximal informed-trading risk).

    max_size_fraction: float = 1.5
    # Hard cap on the final composed size_fraction, applied after every other scalar.
    # This is the one number in this config that's a correctness guarantee rather than
    # a tuning knob -- keep it >= greedy_ceiling or the greedy axis is silently capped
    # below where it's configured to reach.


@dataclass
class ComposedOrderIntent:
    direction: int  # -1, 0, +1 — the FINAL direction after all scaling AND risk veto
    size_fraction: float  # composed size, as a fraction of the RL-intended size (may exceed 1.0 on the greedy axis, never exceeds max_size_fraction)
    bucket: str  # "flat_rl" | "agree" | "unconfirmed" | "disagree" | "predictor_locked" | "risk_vetoed"
    reason: str
    rl_direction: int
    predictor_direction: Optional[int]
    predictor_confidence: Optional[float]
    predictor_gate_passed: bool
    risk_budget_scalar: float  # the cautious-axis-1 throttle actually applied (1.0 if not supplied)
    toxicity_scalar: float  # the cautious-axis-2 throttle actually applied (1.0 if not supplied)
    performance_scalar: float  # the streak-based axis actually applied (1.0 if not supplied)
    risk_allowed: bool
    risk_reason: Optional[str]


class SignalComposer:
    def __init__(self, config: Optional[CompositionConfig] = None):
        self.config = config or CompositionConfig()

    def _rl_direction(self, rl_action: float) -> int:
        if abs(rl_action) < self.config.direction_deadzone:
            return 0
        return 1 if rl_action > 0 else -1

    def _predictor_opinion(
        self, predictor: Optional[SignalPredictor], feature_window: Optional[np.ndarray]
    ) -> Tuple[Optional[int], Optional[float], bool]:
        """Returns (direction, confidence, gate_passed). direction/confidence are None
        when there's no usable opinion (gate not passed, or feature_window not given)."""
        if predictor is None or predictor.model is None:
            return None, None, False
        if feature_window is None or len(feature_window) < predictor.lookback:
            return None, None, True  # gate passed, but not enough data for THIS window

        window = feature_window[-predictor.lookback:].flatten().reshape(1, -1)

        # DEFENSIVE FIX: predictor.model.predict()/.predict_proba() are real sklearn
        # calls that raise a hard ValueError on any shape/dtype mismatch (e.g. a
        # caller's feature_window not matching the exact width the model was fit
        # on) -- previously unguarded, which meant a live caller feeding a
        # differently-shaped window would crash straight through compose() and
        # take down the whole decision loop on what should be a recoverable,
        # "no usable opinion this cycle" situation. Treated the same as "not
        # enough data": gate_passed=True (the predictor itself is fine), no
        # opinion this cycle -- SignalComposer already has a safe fallback for
        # exactly this (the "unconfirmed" bucket), so this degrades gracefully
        # instead of crashing.
        try:
            pred_class = int(predictor.model.predict(window)[0])
            confidence = None
            if hasattr(predictor.model, "predict_proba"):
                proba = predictor.model.predict_proba(window)[0]
                classes = list(predictor.model.classes_)
                confidence = float(proba[classes.index(pred_class)])
            return pred_class, confidence, True
        except Exception as e:
            logger.warning(f"Predictor call failed ({e}) -- treating as no usable opinion this cycle.")
            return None, None, True

    def _normalized_confidence(self, confidence: Optional[float]) -> float:
        """Maps raw predict_proba confidence to [0, 1], where 0 = exactly at the trust
        threshold and 1 = maximal confidence. None (no confidence available) -> 0."""
        if confidence is None:
            return 0.0
        span = 1.0 - self.config.min_predictor_confidence
        if span <= 0:
            return 1.0 if confidence >= self.config.min_predictor_confidence else 0.0
        return _clip((confidence - self.config.min_predictor_confidence) / span, 0.0, 1.0)

    def _risk_budget_scalar(self, risk_budget_used_fraction: Optional[float]) -> float:
        """1.0 while under `risk_budget_caution_start`; ramps linearly down to
        `risk_budget_min_scalar` as usage approaches 1.0 (budget fully used). None
        (no risk-budget reading supplied) -> 1.0, i.e. this throttle is opt-in."""
        if risk_budget_used_fraction is None:
            return 1.0
        u = _clip(risk_budget_used_fraction, 0.0, 1.0)
        start = self.config.risk_budget_caution_start
        if u <= start:
            return 1.0
        span = 1.0 - start
        t = (u - start) / span if span > 0 else 1.0
        floor = self.config.risk_budget_min_scalar
        return 1.0 - t * (1.0 - floor)

    def _toxicity_scalar(self, bvc_vpin: Optional[float]) -> float:
        """1.0 while below `vpin_caution_threshold`; ramps linearly down to
        `vpin_min_scalar` as vpin approaches 1.0. None (no toxicity reading supplied)
        -> 1.0, i.e. this throttle is opt-in -- absence of data is not itself evidence
        of elevated toxicity."""
        if bvc_vpin is None:
            return 1.0
        v = _clip(bvc_vpin, 0.0, 1.0)
        threshold = self.config.vpin_caution_threshold
        if v <= threshold:
            return 1.0
        span = 1.0 - threshold
        t = (v - threshold) / span if span > 0 else 1.0
        floor = self.config.vpin_min_scalar
        return 1.0 - t * (1.0 - floor)

    def compose(
        self,
        rl_action: float,
        base_qty: float,
        price: float,
        predictor: Optional[SignalPredictor] = None,
        feature_window: Optional[np.ndarray] = None,
        risk_check_fn: Optional[RiskCheckFn] = None,
        predictor_locked: bool = False,
        risk_budget_used_fraction: Optional[float] = None,
        bvc_vpin: Optional[float] = None,
        performance_scalar: Optional[float] = None,
    ) -> ComposedOrderIntent:
        """
        ...
        performance_scalar: optional, caller-supplied -- typically
            PerformanceStreakTracker.scalar() (see execution/performance_tracker.py).
            None (not supplied) -> 1.0, no penalty or boost, same "absence is neutral,
            not evidence" convention as risk_budget_used_fraction/bvc_vpin.
        """
        rl_dir = self._rl_direction(rl_action)
        risk_scalar = self._risk_budget_scalar(risk_budget_used_fraction)
        tox_scalar = self._toxicity_scalar(bvc_vpin)
        perf_scalar = 1.0 if performance_scalar is None else _clip(performance_scalar, 0.0, self.config.max_size_fraction)

        # Principle 1: predictor/risk-budget/toxicity never override a flat RL action
        # into a trade.
        if rl_dir == 0:
            return ComposedOrderIntent(
                direction=0, size_fraction=0.0, bucket="flat_rl",
                reason="RL action within deadzone -- no trade proposed.",
                rl_direction=0, predictor_direction=None, predictor_confidence=None,
                predictor_gate_passed=(predictor is not None and predictor.model is not None),
                risk_budget_scalar=risk_scalar, toxicity_scalar=tox_scalar, performance_scalar=perf_scalar,
                risk_allowed=True, risk_reason=None,
            )

        # Principle 4: never propose while the predictor is mid-reset.
        if predictor_locked:
            return ComposedOrderIntent(
                direction=0, size_fraction=0.0, bucket="predictor_locked",
                reason="Predictor is mid-plasticity-reset -- no order proposed this cycle.",
                rl_direction=rl_dir, predictor_direction=None, predictor_confidence=None,
                predictor_gate_passed=False, risk_budget_scalar=risk_scalar, toxicity_scalar=tox_scalar,
                performance_scalar=perf_scalar, risk_allowed=True, risk_reason=None,
            )

        pred_dir, pred_conf, gate_passed = self._predictor_opinion(predictor, feature_window)
        norm_conf = self._normalized_confidence(pred_conf)

        if pred_dir is None or pred_conf is None or pred_conf < self.config.min_predictor_confidence:
            bucket = "unconfirmed"
            base = self.config.unconfirmed_multiplier
            reason = (
                "Predictor gate not passed or no window available." if not gate_passed
                else f"Predictor confidence {pred_conf:.2f} below threshold "
                     f"{self.config.min_predictor_confidence} -- treated as no opinion."
                if pred_conf is not None else "Predictor produced no confidence estimate."
            )
        elif pred_dir == rl_dir:
            bucket = "agree"
            # Greedy axis: confidence-scaled bonus above agree_multiplier, up to
            # greedy_ceiling at full confidence.
            base = self.config.agree_multiplier + (self.config.greedy_ceiling - self.config.agree_multiplier) * norm_conf
            reason = f"Predictor agrees (confidence {pred_conf:.2f}, norm={norm_conf:.2f}) -- greedy-scaled to {base:.2f}x before risk/toxicity throttle."
        elif pred_dir == -rl_dir:
            bucket = "disagree"
            # A more confident disagreement shrinks the floor further, never grows it.
            base = self.config.disagree_multiplier * (1.0 - norm_conf)
            reason = f"Predictor confidently predicts the OPPOSITE direction (confidence {pred_conf:.2f}) -- floor shrunk to {base:.2f}x."
        else:
            bucket = "unconfirmed"
            base = self.config.unconfirmed_multiplier
            reason = f"Predictor confidently predicts flat (confidence {pred_conf:.2f}) -- no confirmation."

        multiplier = _clip(base * risk_scalar * tox_scalar * perf_scalar, 0.0, self.config.max_size_fraction)
        if risk_scalar < 1.0 or tox_scalar < 1.0 or perf_scalar != 1.0:
            reason += (f" [risk_budget_scalar={risk_scalar:.2f}, toxicity_scalar={tox_scalar:.2f}, "
                       f"performance_scalar={perf_scalar:.2f}] -> final {multiplier:.2f}x")

        proposed_qty = base_qty * multiplier
        proposed_direction = rl_dir if multiplier > 1e-9 else 0

        if proposed_direction == 0 or risk_check_fn is None:
            risk_allowed, risk_reason = True, None
        else:
            side = "buy" if proposed_direction > 0 else "sell"
            risk_allowed, risk_reason = risk_check_fn(side, proposed_qty, price)

        if not risk_allowed:
            return ComposedOrderIntent(
                direction=0, size_fraction=0.0, bucket="risk_vetoed",
                reason=f"risk_guardian vetoed the composed order: {risk_reason}",
                rl_direction=rl_dir, predictor_direction=pred_dir, predictor_confidence=pred_conf,
                predictor_gate_passed=gate_passed, risk_budget_scalar=risk_scalar, toxicity_scalar=tox_scalar,
                performance_scalar=perf_scalar, risk_allowed=False, risk_reason=risk_reason,
            )

        return ComposedOrderIntent(
            direction=proposed_direction, size_fraction=multiplier, bucket=bucket, reason=reason,
            rl_direction=rl_dir, predictor_direction=pred_dir, predictor_confidence=pred_conf,
            predictor_gate_passed=gate_passed, risk_budget_scalar=risk_scalar, toxicity_scalar=tox_scalar,
            performance_scalar=perf_scalar, risk_allowed=True, risk_reason=None,
        )
