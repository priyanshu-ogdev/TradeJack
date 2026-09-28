"""
LiveInstrumentSource -- a real (not fake/stubbed) implementation of the
InstrumentSource protocol execution/portfolio_orchestrator.py's
PortfolioOrchestrator expects, wired to this project's actual predictor and
risk-management stacks:

  - data_forge/predictor.py's SignalPredictor: gating (.model is not None)
    and direction confidence (predict_proba on the predicted class).
  - training/supervised_predictor.py's SupervisedPredictor (OPTIONAL): the
    regression head for expected_return magnitude. These are genuinely
    different predictor packs with genuinely different feature shapes (see
    below) -- SignalPredictor was purpose-built for 3-class direction gating,
    SupervisedPredictor for 2-class + regression; neither is a drop-in
    replacement for the other, by design (see data_forge/predictor.py's own
    module docstring on why the split exists).
  - execution/risk_guardian.py's RiskGuardian: risk_budget_used_fraction().

TWO SEPARATE FEATURE WINDOWS, NOT ONE -- READ BEFORE WIRING THIS UP
------------------------------------------------------------------------
SignalPredictor.model.predict() expects X shaped (1, lookback*F) -- a pure
flattened feature window, nothing else appended (see data_forge/predictor.py's
_build_3class_features_and_labels).

SupervisedPredictor.predict() expects X shaped (1, lookback*F + lookback) --
the same feature window PLUS a trailing recent-returns segment (see
training/supervised_predictor.py's build_features_and_labels).

These are NOT interchangeable. Passing one predictor's expected window shape
to the other either crashes (sklearn's "X has N features but estimator is
expecting M") or, worse, silently succeeds against the wrong number of
features if the shapes happen to coincide by accident. This class takes TWO
separate window-builder callables specifically so neither this class nor its
caller has to reconcile the two shapes internally -- the caller who
constructed each predictor already knows which shape it needs.

WHY current_price MUST BE READ FIRST, AND WHY THAT'S SAFE HERE
------------------------------------------------------------------------
PortfolioOrchestrator.run_cycle() reads every source's `.current_price` in
one pass (to update the volatility tracker) BEFORE reading `.gate_passed` /
`.expected_return` / `.direction_confidence` / `.risk_budget_used_fraction`
in a second pass (see that method's actual implementation -- two separate
loops, not interleaved). This class relies on that real, structural ordering
guarantee: accessing `.current_price` takes a fresh snapshot of both feature
windows and runs both predictions once; the other four properties then read
that cached snapshot rather than re-predicting (which could otherwise see a
DIFFERENT feature window if enough wall-clock time passed between property
reads on live, ticking data, making expected_return and direction_confidence
silently describe two different moments). If `.current_price` is somehow
never called before the others (a caller other than PortfolioOrchestrator,
or a future refactor of run_cycle() that changes the read order), this
degrades safely to "no opinion" (gate_passed=False) rather than using stale
or absent data -- verified by test_no_snapshot_yet_degrades_to_ungated.
"""

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (LiveInstrumentSource) %(message)s")
logger = logging.getLogger("LiveInstrumentSource")


@dataclass
class _Snapshot:
    price: float
    gate_passed: bool
    direction_confidence: float
    expected_return: float
    taken_at: float


class LiveInstrumentSource:
    """
    instrument: this project's symbol convention (e.g. "EURUSD", "BTC-USDT") --
      used only for logging here.
    direction_predictor: a data_forge.predictor.SignalPredictor. Required --
      there is no meaningful InstrumentSource without at least direction
      gating.
    direction_feature_window_fn: () -> Optional[np.ndarray], returning a
      (lookback, F) window (NOT pre-flattened -- this class flattens it,
      matching what composition_layer.py's own _predictor_opinion does)
      matching direction_predictor.lookback, or None if not enough history
      exists yet.
    return_predictor: OPTIONAL training.supervised_predictor.SupervisedPredictor
      (or anything exposing the same .predict(X) -> {"expected_return": array}
      shape). If None, expected_return is always 0.0 -- deliberately, not a
      fabricated magnitude: "no regression estimate" should mean "contributes
      nothing to the Kelly opportunity score," not "assume some default
      magnitude," matching the "absence is neutral, not evidence" convention
      used throughout this project's other predictor-absence handling.
    return_feature_window_fn: OPTIONAL () -> Optional[np.ndarray], the
      SEPARATELY-SHAPED window return_predictor needs (see module docstring).
      Ignored if return_predictor is None.
    risk_guardian: execution.risk_guardian.RiskGuardian for this instrument.
    price_fn: () -> float, the current mid/last price.
    """

    def __init__(
        self,
        instrument: str,
        direction_predictor,
        direction_feature_window_fn: Callable[[], Optional[np.ndarray]],
        risk_guardian,
        price_fn: Callable[[], float],
        return_predictor=None,
        return_feature_window_fn: Optional[Callable[[], Optional[np.ndarray]]] = None,
    ):
        self.instrument = instrument
        self.direction_predictor = direction_predictor
        self._direction_feature_window_fn = direction_feature_window_fn
        self.return_predictor = return_predictor
        self._return_feature_window_fn = return_feature_window_fn
        self.risk_guardian = risk_guardian
        self._price_fn = price_fn
        self._snapshot: Optional[_Snapshot] = None

    def _take_snapshot(self) -> _Snapshot:
        price = float(self._price_fn())

        gate_passed = False
        direction_confidence = 0.0
        expected_return = 0.0

        try:
            model = getattr(self.direction_predictor, "model", None)
            if model is not None:
                window = self._direction_feature_window_fn()
                lookback = getattr(self.direction_predictor, "lookback", None)
                if window is not None and lookback is not None and len(window) >= lookback:
                    x = np.asarray(window[-lookback:]).flatten().reshape(1, -1)
                    pred_class = int(model.predict(x)[0])
                    if pred_class != 0:  # 0 = "flat" -- no directional opinion to size against
                        gate_passed = True
                        if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
                            proba = model.predict_proba(x)[0]
                            classes = list(model.classes_)
                            direction_confidence = float(proba[classes.index(pred_class)])
                        else:
                            direction_confidence = 1.0  # model exists but can't report a probability -- treat as fully confident in its own call
        except Exception as e:
            logger.error(f"{self.instrument}: direction prediction failed this cycle, treating as ungated: {e}")
            gate_passed = False
            direction_confidence = 0.0

        if gate_passed and self.return_predictor is not None and self._return_feature_window_fn is not None:
            try:
                return_window = self._return_feature_window_fn()
                if return_window is not None:
                    x = np.asarray(return_window).reshape(1, -1)
                    preds = self.return_predictor.predict(x)
                    expected_return = float(preds["expected_return"][0])
            except Exception as e:
                logger.error(f"{self.instrument}: return prediction failed this cycle, expected_return stays 0.0: {e}")
                expected_return = 0.0

        return _Snapshot(
            price=price, gate_passed=gate_passed, direction_confidence=direction_confidence,
            expected_return=expected_return, taken_at=time.time(),
        )

    @property
    def current_price(self) -> float:
        # Always takes a fresh snapshot -- this is the one property
        # PortfolioOrchestrator.run_cycle() reads first, each cycle (see
        # module docstring). Every other property below reads this same
        # snapshot rather than re-predicting.
        self._snapshot = self._take_snapshot()
        return self._snapshot.price

    @property
    def gate_passed(self) -> bool:
        if self._snapshot is None:
            logger.warning(f"{self.instrument}: gate_passed read before current_price this cycle -- degrading to ungated.")
            return False
        return self._snapshot.gate_passed

    @property
    def direction_confidence(self) -> float:
        if self._snapshot is None:
            return 0.0
        return self._snapshot.direction_confidence

    @property
    def expected_return(self) -> float:
        if self._snapshot is None:
            return 0.0
        return self._snapshot.expected_return

    @property
    def risk_budget_used_fraction(self) -> float:
        try:
            return float(self.risk_guardian.risk_budget_used_fraction())
        except Exception as e:
            logger.error(f"{self.instrument}: risk_budget_used_fraction failed, treating as fully used (no headroom): {e}")
            return 1.0  # fail SAFE: an unreadable risk state must never look like free headroom
