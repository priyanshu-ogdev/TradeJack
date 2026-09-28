"""
PortfolioOrchestrator -- the piece that actually calls PortfolioAllocator.allocate()
on a schedule, with real InstrumentSignals built from live per-instrument state.

Two responsibilities, deliberately kept separate and both independently testable:

1. VolatilityCorrelationTracker -- maintains a rolling window of LOG returns per
   instrument (not raw price, not simple returns -- see InstrumentSignal's
   docstring in portfolio_allocator.py for why log-return space is the one that
   makes volatility comparable across instruments with different price scales)
   and computes realized volatility + a pairwise correlation matrix from it. Pure
   numpy, zero dependency on the rest of the pipeline -- it only ever sees prices
   fed to it.

2. PortfolioOrchestrator -- ties the tracker together with one "instrument
   source" per instrument (anything exposing .gate_passed / .expected_return /
   .direction_confidence / .risk_budget_used_fraction / .current_price -- in
   practice a thin adapter over that instrument's own SignalPredictor +
   RiskGuardian + live price feed) and PortfolioAllocator, and runs one
   allocation cycle: pull each source's current state, update volatility/
   correlation, build InstrumentSignals, call allocate().

What this deliberately does NOT do: define what an "instrument source" actually
is beyond a duck-typed protocol. Wiring a real SignalPredictor's feature window
and a real live price feed into a concrete InstrumentSource implementation is a
live_inference_server-level integration concern (per-instrument feature
plumbing), not an orchestration-layer one -- keeping that out of this module is
what makes the orchestration logic itself testable with plain fakes, the same
reason execution/live_composer.py's toxicity auto-load took a
toxicity_table_builder dependency instead of hardcoding TrainingTableBuilder.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np

from execution.portfolio_allocator import InstrumentSignal, PortfolioAllocator, AllocationDecision

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (PortfolioOrchestrator) %(message)s")
logger = logging.getLogger("PortfolioOrchestrator")


class VolatilityCorrelationTracker:
    """Rolling log-return window per instrument -> realized volatility (stdev of
    log returns over the window) and pairwise Pearson correlation. `window`
    should match the lookback the per-instrument SignalPredictor itself reasons
    over (same principle as InstrumentSignal.realized_volatility's docstring:
    volatility and the predictor's own edge estimate should be measured over
    comparable horizons, or the allocator would be sizing a short-horizon edge
    against a long-horizon (or vice versa) risk estimate)."""

    def __init__(self, window: int = 50, min_observations: int = 10):
        self.window = window
        self.min_observations = min_observations
        self._last_price: Dict[str, float] = {}
        self._log_returns: Dict[str, List[float]] = {}

    def update(self, instrument: str, price: float) -> None:
        if price <= 0:
            logger.warning(f"Ignoring non-positive price for {instrument}: {price}")
            return
        prev = self._last_price.get(instrument)
        self._last_price[instrument] = price
        if prev is None or prev <= 0:
            return  # first observation for this instrument -- no return to compute yet
        log_return = float(np.log(price / prev))
        buf = self._log_returns.setdefault(instrument, [])
        buf.append(log_return)
        if len(buf) > self.window:
            del buf[: len(buf) - self.window]

    def realized_volatility(self, instrument: str) -> Optional[float]:
        """None (not 0.0) if there isn't yet enough history -- a volatility of
        0.0 would make that instrument look risk-free to the Kelly formula
        (score = edge / vol^2 -> infinite as vol -> 0), which is the opposite of
        the truth for an instrument nobody has observed long enough to measure."""
        buf = self._log_returns.get(instrument)
        if buf is None or len(buf) < self.min_observations:
            return None
        return float(np.std(buf, ddof=1))

    def correlation_matrix(self) -> Dict[Tuple[str, str], float]:
        """Pairwise Pearson correlation of log returns, computed only over each
        pair's OVERLAPPING recent history (the shorter of the two buffers,
        aligned from the most recent observation backward) -- two instruments
        observed for different lengths of time should be compared on what they
        actually share, not padded with zeros or silently misaligned."""
        result: Dict[Tuple[str, str], float] = {}
        instruments = [i for i, buf in self._log_returns.items() if len(buf) >= self.min_observations]
        for idx_a in range(len(instruments)):
            for idx_b in range(idx_a + 1, len(instruments)):
                a, b = instruments[idx_a], instruments[idx_b]
                buf_a, buf_b = self._log_returns[a], self._log_returns[b]
                n = min(len(buf_a), len(buf_b))
                if n < self.min_observations:
                    continue
                arr_a = np.array(buf_a[-n:])
                arr_b = np.array(buf_b[-n:])
                if np.std(arr_a) == 0 or np.std(arr_b) == 0:
                    continue  # a constant series has undefined correlation -- skip rather than divide by zero
                corr = float(np.corrcoef(arr_a, arr_b)[0, 1])
                result[(a, b)] = corr
                result[(b, a)] = corr
        return result


class InstrumentSource(Protocol):
    """Duck-typed -- anything exposing these four properties works. In
    production this wraps that instrument's own SignalPredictor (gate_passed =
    model is not None, expected_return/direction_confidence from its most recent
    prediction) and RiskGuardian (risk_budget_used_fraction), plus a live price
    feed for current_price. Not implemented here -- see module docstring."""
    @property
    def gate_passed(self) -> bool: ...
    @property
    def expected_return(self) -> float: ...
    @property
    def direction_confidence(self) -> float: ...
    @property
    def risk_budget_used_fraction(self) -> float: ...
    @property
    def current_price(self) -> float: ...


@dataclass
class OrchestratorCycleResult:
    decisions: List[AllocationDecision]
    signals: List[InstrumentSignal]
    correlation_matrix: Dict[Tuple[str, str], float]


class PortfolioOrchestrator:
    def __init__(
        self,
        allocator: PortfolioAllocator,
        instrument_sources: Dict[str, InstrumentSource],
        vol_tracker: Optional[VolatilityCorrelationTracker] = None,
    ):
        self.allocator = allocator
        self.instrument_sources = instrument_sources
        self.vol_tracker = vol_tracker or VolatilityCorrelationTracker()

    def run_cycle(self) -> OrchestratorCycleResult:
        """One allocation cycle: update volatility tracking from each source's
        current price, build this cycle's InstrumentSignals, and delegate to
        PortfolioAllocator.allocate(). An instrument with no volatility reading
        yet (insufficient history) is passed through as gate_passed=False
        regardless of what its source reports -- the allocator cannot safely
        score an instrument it has no risk estimate for, even if that
        instrument's own predictor gate is passing."""
        for instrument, source in self.instrument_sources.items():
            try:
                self.vol_tracker.update(instrument, source.current_price)
            except Exception as e:
                logger.error(f"Failed to update price for {instrument}: {e}")

        signals = []
        for instrument, source in self.instrument_sources.items():
            vol = self.vol_tracker.realized_volatility(instrument)
            if vol is None:
                signals.append(InstrumentSignal(instrument, False, 0.0, 0.0, 0.0, 1.0))
                continue
            try:
                signals.append(InstrumentSignal(
                    instrument=instrument,
                    gate_passed=source.gate_passed,
                    expected_return=source.expected_return,
                    direction_confidence=source.direction_confidence,
                    realized_volatility=vol,
                    risk_budget_used_fraction=source.risk_budget_used_fraction,
                ))
            except Exception as e:
                logger.error(f"Failed to read source state for {instrument}, treating as ungated this cycle: {e}")
                signals.append(InstrumentSignal(instrument, False, 0.0, 0.0, vol, 1.0))

        correlations = self.vol_tracker.correlation_matrix()
        decisions = self.allocator.allocate(signals, correlation_matrix=correlations)
        return OrchestratorCycleResult(decisions=decisions, signals=signals, correlation_matrix=correlations)
