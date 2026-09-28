"""
Live Order Composer — closes the "hand-fed numbers" gap.

composition_layer.py's SignalComposer.compose() deliberately takes
risk_budget_used_fraction / performance_scalar as plain floats rather than importing
RiskGuardian or PerformanceStreakTracker itself (keeps the composer unit-testable in
isolation, per its own docstring). Someone still has to actually pull those numbers from
the real objects on every order cycle — this module is that someone, so a live caller
doesn't have to hand-write the same three-line glue at every call site.

This is intentionally a thin wrapper: it does not add any new decision logic of its own
-- every actual composition rule lives in SignalComposer, every actual risk rule lives in
RiskGuardian, every actual streak rule lives in PerformanceStreakTracker. Duplicating
their logic here would be exactly the kind of drift bug (two copies of the same rule
slowly diverging) already flagged and fixed once in risk_guardian.py's own
`risk_budget_used_fraction()` (originally computed inline in two places).
"""

import time
import logging
from typing import Any, Optional

import numpy as np

from data_forge.predictor import SignalPredictor
from execution.composition_layer import SignalComposer, ComposedOrderIntent
from execution.performance_tracker import PerformanceStreakTracker
from execution.risk_guardian import RiskGuardian

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (LiveComposer) %(message)s")
logger = logging.getLogger("LiveComposer")


class LiveOrderComposer:
    """
    Binds one RiskGuardian + one SignalComposer + one PerformanceStreakTracker together
    for one symbol's live order-composition cycle. A caller still supplies the per-cycle
    inputs that only IT knows (the RL action, the latest feature window, the current
    price, the current toxicity reading, whether the predictor is mid-reset) -- this
    class only removes the need to manually re-derive risk_budget_used_fraction and
    performance_scalar from RiskGuardian/PerformanceStreakTracker on every call.
    """

    def __init__(
        self,
        risk_guardian: RiskGuardian,
        composer: Optional[SignalComposer] = None,
        performance_tracker: Optional[PerformanceStreakTracker] = None,
        toxicity_symbol: Optional[str] = None,
        toxicity_table_builder: Optional[Any] = None,
        toxicity_cache_ttl_seconds: float = 300.0,
    ):
        self.risk_guardian = risk_guardian
        self.composer = composer or SignalComposer()
        self.performance_tracker = performance_tracker or PerformanceStreakTracker()

        # Toxicity-leg auto-loading. toxicity_symbol=None (default) keeps the old
        # behavior exactly: bvc_vpin must be caller-supplied or stays None (no
        # throttle). Setting it opts into auto-loading the latest vpin_50/bvc_vpin
        # reading from disk via TrainingTableBuilder.latest_toxicity_reading() --
        # see that method for which leg it reads and why. toxicity_table_builder is
        # accepted purely for dependency injection (tests use a fake with a
        # `.calls` counter and a scripted return sequence; a real caller can just
        # leave it as None and get a real TrainingTableBuilder).
        self.toxicity_symbol = toxicity_symbol
        self._toxicity_builder = toxicity_table_builder
        self.toxicity_cache_ttl_seconds = toxicity_cache_ttl_seconds
        self._cached_toxicity: Optional[float] = None
        self._toxicity_cached_at: float = 0.0

    def _current_toxicity_reading(self) -> Optional[float]:
        """Returns the cached auto-loaded toxicity reading, refreshing from disk only
        once every toxicity_cache_ttl_seconds -- toxicity moves on the order of
        minutes (it's computed from bucketed/time-windowed features), not every tick,
        so re-reading a parquet file (and importing polars) on every single
        compose_order() call would be pure overhead for a value that mostly hasn't
        changed. Returns None if toxicity_symbol wasn't configured (auto-loading is
        opt-in) or nothing was found/the read failed -- same "absence is neutral, not
        an error" handling as latest_toxicity_reading() itself."""
        if self.toxicity_symbol is None:
            return None
        now = time.time()
        if self._cached_toxicity is not None and (now - self._toxicity_cached_at) < self.toxicity_cache_ttl_seconds:
            return self._cached_toxicity
        try:
            if self._toxicity_builder is not None:
                builder = self._toxicity_builder
            else:
                # Lazy import: TrainingTableBuilder pulls in data_forge.config (needs
                # pydantic_settings) and polars, neither of which should be required
                # just to construct a LiveOrderComposer that never uses auto-loaded
                # toxicity (toxicity_symbol=None, the default). Importing at module
                # level broke exactly that -- caught by tests/test_live_composer.py
                # failing to even import in an environment without pydantic_settings.
                from data_forge.training_table_builder import TrainingTableBuilder
                builder = TrainingTableBuilder()
            reading = builder.latest_toxicity_reading(self.toxicity_symbol)
        except Exception as e:
            logger.error(f"Toxicity auto-load failed for {self.toxicity_symbol}: {e}")
            reading = None
        self._cached_toxicity = reading
        self._toxicity_cached_at = now
        return reading

    def compose_order(
        self,
        rl_action: float,
        base_qty: float,
        price: float,
        predictor: Optional[SignalPredictor] = None,
        feature_window: Optional[np.ndarray] = None,
        predictor_locked: bool = False,
        bvc_vpin: Optional[float] = None,
    ) -> ComposedOrderIntent:
        """
        Pulls risk_budget_used_fraction from the bound RiskGuardian and
        performance_scalar from the bound PerformanceStreakTracker automatically, then
        delegates everything else to SignalComposer.compose() unchanged.

        bvc_vpin: an explicit value here always wins -- a caller with its own fresher
        or streaming toxicity reading is never second-guessed by the disk-backed
        auto-load. Only when bvc_vpin is None AND toxicity_symbol was configured at
        construction does this fall back to _current_toxicity_reading()'s cached
        auto-load; otherwise it stays None (no throttle), exactly like before this
        wiring existed.
        """
        if bvc_vpin is None:
            bvc_vpin = self._current_toxicity_reading()

        return self.composer.compose(
            rl_action=rl_action,
            base_qty=base_qty,
            price=price,
            predictor=predictor,
            feature_window=feature_window,
            risk_check_fn=self.risk_guardian.check_order_allowed,
            predictor_locked=predictor_locked,
            risk_budget_used_fraction=self.risk_guardian.risk_budget_used_fraction(),
            bvc_vpin=bvc_vpin,
            performance_scalar=self.performance_tracker.scalar(),
        )

    def record_fill(self, pnl: float):
        """Call after a trade closes with its realized P&L -- feeds the performance
        streak tracker so the NEXT compose_order() call reflects it. Does not touch
        RiskGuardian (that's updated separately via risk_guardian.update_equity(),
        which the caller must still call with the account's actual current equity --
        record_fill() is not a substitute for that)."""
        self.performance_tracker.record_trade_outcome(pnl)

    def reset_for_new_session(self):
        """Call when a human reviews and resets a halt (risk_guardian.reset_halt()) or
        at the start of a new trading day -- clears the performance streak so it
        doesn't keep throttling (or boosting) size based on a streak from before
        conditions genuinely changed. Does NOT call risk_guardian.reset_halt() itself
        -- that remains a deliberate, separate human action per
        docs/PROCESS_SUPERVISION.md's "risk halt != process crash" principle; this
        method only resets the streak tracker's own state."""
        self.performance_tracker.reset()
