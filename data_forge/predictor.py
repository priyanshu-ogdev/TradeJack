"""
SignalPredictor -- the module execution/composition_layer.py and
execution/live_composer.py import from but that was never actually
delivered. Confirmed missing by grepping every zip uploaded in this
project's history for `data_forge/predictor.py` / `class SignalPredictor`
before writing this: zero matches anywhere prior to this file, and
tests/test_composition_layer.py / tests/test_live_composer.py both reference
a `tests/test_predictor.py` fixture that also never arrived. The exact API
below (constructor kwargs, `fit_and_gate(features, price)` taking a PRICE
series rather than returns, a dataclass result with `.passed`/`.reason`) was
reverse-engineered by reading what the two delivered callers and their test
files actually call -- not guessed -- and was corrected once already after
a first draft's constructor signature didn't match
(`flat_threshold` -> separate `up_threshold`/`down_threshold`, dict result ->
`GateResult` dataclass) once the real test failures said so.

Relationship to training/supervised_predictor.py
---------------------------------------------------
That module is a 2-class (up/down) classifier + regressor pack, meant to be
consumed as a joinable data_forge leg via training_table_builder.py.
composition_layer.py needs something that module can't provide: an explicit
"confidently predicts FLAT" outcome, distinct from "predicts down" -- its
agree/disagree/unconfirmed bucketing depends on a real third class, not a
binary split with a probability near 0.5 standing in for "flat". Hence a
separate 3-class (down=-1 / flat=0 / up=+1) predictor here, rather than
force-fitting the 2-class one into a job it wasn't built for.

What is NOT duplicated: PurgedTimeSeriesSplit (embargoed walk-forward CV) is
imported from training/supervised_predictor.py rather than reimplemented --
it's generic over the number of classes, already tested
(tests/test_supervised_predictor.py), and a second hand-copied version of the
same ~15 lines is exactly how these two predictor packs would quietly drift
apart over time.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.stats import ttest_1samp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from training.supervised_predictor import PurgedTimeSeriesSplit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (SignalPredictor) %(message)s")
logger = logging.getLogger("SignalPredictor")


@dataclass
class GateResult:
    """fit_and_gate()'s return value. Accessed by attribute
    (result.passed / result.reason), not as a dict -- matches how
    tests/test_composition_layer.py and tests/test_live_composer.py actually
    use it."""
    passed: bool
    reason: str
    mean_edge: Optional[float] = None
    p_value: Optional[float] = None
    n_folds: int = 0
    fold_edges: List[float] = field(default_factory=list)
    n_usable_rows: int = 0


def _prices_to_returns(prices: np.ndarray) -> np.ndarray:
    """Simple returns from a price level series, same length as the input
    (returns[0] = 0.0, since there is no prior price to compute it from --
    that row's lookback window at t=0 doesn't reach past it anyway once
    lookback>=1 trims the first lookback-1 origins)."""
    prices = np.asarray(prices, dtype=np.float64)
    returns = np.zeros_like(prices)
    returns[1:] = np.diff(prices) / np.clip(prices[:-1], 1e-12, None)
    return returns


def _build_3class_features_and_labels(
    features: np.ndarray, returns: np.ndarray, lookback: int, horizon: int,
    up_threshold: float, down_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same windowing/no-lookahead boundary as
    training/supervised_predictor.py's build_features_and_labels (a row at
    origin t uses the feature window up through t, labels off returns[t+1 :
    t+1+horizon] strictly after t) -- only the label itself differs (3-way
    sign-with-asymmetric-deadzone instead of binary sign).

    Unlike training/supervised_predictor.py's builder, X here is JUST the
    flattened feature window (lookback*F), with no recent-returns history
    appended -- inference-time callers (composition_layer.py's
    _predictor_opinion, and every test fixture) build their window the same
    way: `features[-lookback:].flatten()`. Appending anything else to X here
    would silently require every inference-time caller to reproduce that
    exact extra step or hit a hard feature-count mismatch from sklearn --
    which is exactly the bug an earlier version of this had, caught only by
    actually running tests/test_composition_layer.py (60 features supplied
    at inference vs. 80 expected at training time)."""
    features = np.asarray(features, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    T, F = features.shape
    if returns.shape[0] != T:
        raise ValueError(f"features has {T} rows but returns has {returns.shape[0]}")

    rows_X, rows_y, origin_idx = [], [], []
    for t in range(lookback - 1, T - horizon):
        window = features[t - lookback + 1: t + 1]
        forward_ret = float(np.sum(returns[t + 1: t + 1 + horizon]))
        if forward_ret > up_threshold:
            y = 1
        elif forward_ret < -down_threshold:
            y = -1
        else:
            y = 0
        rows_X.append(window.flatten())
        rows_y.append(y)
        origin_idx.append(t)

    if not rows_X:
        return np.empty((0, lookback * F)), np.empty((0,), dtype=int), np.empty((0,), dtype=int)
    return np.array(rows_X), np.array(rows_y, dtype=int), np.array(origin_idx, dtype=int)


def _persistence_baseline_3class(
    returns: np.ndarray, origin_idx: np.ndarray, up_threshold: float, down_threshold: float
) -> np.ndarray:
    """Baseline: same sign as the most recently realized return, i.e.
    returns[t] for each row's origin index t -- sourced directly from the
    `returns` array (not read back out of X, since X no longer carries a
    returns-history segment at all -- see _build_3class_features_and_labels)."""
    last_ret = returns[origin_idx]
    y = np.zeros(len(last_ret), dtype=int)
    y[last_ret > up_threshold] = 1
    y[last_ret < -down_threshold] = -1
    return y


class SignalPredictor:
    """A gated 3-class (down=-1 / flat=0 / up=+1) predictor, trained directly
    from a price series (not pre-computed returns -- fit_and_gate derives
    returns internally via _prices_to_returns).

    `.model` is None until fit_and_gate() both trains AND its walk-forward
    promotion gate passes. composition_layer.py depends on exactly this:
    a None .model is treated as "no opinion available" (UNCONFIRMED bucket),
    never silently treated as "predicts flat" -- see composition_layer.py's
    module docstring, point 2.
    """

    def __init__(
        self, lookback: int = 10, horizon: int = 5,
        up_threshold: float = 0.0, down_threshold: float = 0.0,
        n_splits: int = 5, embargo: Optional[int] = None,
        min_mean_edge: float = 0.02, significance_level: float = 0.05,
        max_depth: int = 4, max_iter: int = 150, l2_regularization: float = 1.0,
        random_state: int = 42,
    ):
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for SignalPredictor.")
        self.lookback = lookback
        self.horizon = horizon
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.n_splits = n_splits
        self.embargo = embargo if embargo is not None else (lookback + horizon)
        self.min_mean_edge = min_mean_edge
        self.significance_level = significance_level
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.l2_regularization = l2_regularization
        self.random_state = random_state

        self.model = None
        self.last_gate_result: Optional[GateResult] = None
        self.trained_at: Optional[float] = None

    def _new_classifier(self) -> "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier(
            max_depth=self.max_depth, max_iter=self.max_iter,
            l2_regularization=self.l2_regularization, random_state=self.random_state,
        )

    def fit_and_gate(self, features: np.ndarray, price: np.ndarray) -> GateResult:
        """features: (T, F) engineered features. price: (T,) a raw PRICE
        LEVEL series (not returns) -- returns are derived internally so
        every caller building labels off "what the price actually did"
        shares one return-computation path rather than each re-deriving it
        (and potentially disagreeing on simple-vs-log returns) independently.

        Only sets self.model (fit on ALL data, not any single fold's model)
        if the gate passes. Always returns a GateResult and never raises on
        a refusal -- a refused gate is an ordinary, expected outcome, exactly
        like train_evaluate_and_promote() in training/supervised_predictor.py
        and RL candidate promotion in training/continuous_trainer.py. Leaves
        self.model at its previous value (None, or a previously-gated model)
        on refusal -- callers doing periodic retraining keep the last
        promoted model rather than losing their predictor to one bad
        retraining window."""
        returns = _prices_to_returns(price)
        X, y, origin_idx = _build_3class_features_and_labels(
            features, returns, self.lookback, self.horizon, self.up_threshold, self.down_threshold
        )
        min_rows = (self.n_splits + 1) * 10
        if len(X) < min_rows:
            result = GateResult(passed=False, reason=f"only {len(X)} usable rows -- need >= {min_rows} for {self.n_splits} folds")
            self.last_gate_result = result
            return result

        if not SCIPY_AVAILABLE:
            result = GateResult(passed=False, reason="scipy is required for the promotion gate's significance test")
            self.last_gate_result = result
            return result

        splitter = PurgedTimeSeriesSplit(n_splits=self.n_splits, embargo=self.embargo)
        baseline_signal = _persistence_baseline_3class(returns, origin_idx, self.up_threshold, self.down_threshold)

        fold_edges = []
        for train_idx, test_idx in splitter.split(len(X)):
            clf = self._new_classifier().fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            model_acc = float(np.mean(pred == y[test_idx]))
            baseline_acc = float(np.mean(baseline_signal[test_idx] == y[test_idx]))
            fold_edges.append(model_acc - baseline_acc)

        if len(fold_edges) < 2:
            result = GateResult(passed=False, reason=f"only {len(fold_edges)} usable fold(s); need >= 2 for a significance test")
            self.last_gate_result = result
            return result

        edges = np.array(fold_edges)
        mean_edge = float(np.mean(edges))
        if np.allclose(edges, edges[0]):
            # Zero variance across folds -- a t-test divides by zero here.
            # A perfectly consistent edge in the same direction across every
            # fold is the strongest possible evidence, not an edge case to
            # apologize for: p=0 if that edge is positive, p=1 if it isn't.
            p_value = 0.0 if edges[0] > 0 else 1.0
        else:
            # One-sample t-test (H0: mean fold edge <= 0), not Wilcoxon
            # signed-rank: Wilcoxon's minimum achievable one-sided p-value
            # with n paired samples is 1/2^n (all differences the same sign
            # is the single most extreme rank configuration under the null).
            # At n_splits=4 that floor is 0.0625 -- ABOVE this gate's default
            # 0.05 significance_level -- so a model that wins every single
            # fold by a wide margin still could not pass at 4 folds; this
            # was caught by a real execution, not spotted by inspection
            # (tests/test_composition_layer.py's fixture uses n_splits=4 and
            # a mean_edge of 0.465 -- about as strong a synthetic signal as
            # this test suite constructs anywhere -- and Wilcoxon still
            # refused it at p=0.0625). A t-test has no such floor and is a
            # reasonable choice for a handful of fold-level means regardless.
            try:
                t_result = ttest_1samp(edges, popmean=0.0, alternative="greater")
                p_value = float(t_result.pvalue)
            except (ValueError, FloatingPointError) as e:
                result = GateResult(passed=False, reason=f"t-test failed ({e}); refusing to promote on an unverifiable result")
                self.last_gate_result = result
                return result

        passed = bool(mean_edge >= self.min_mean_edge and p_value < self.significance_level)
        result = GateResult(
            passed=passed,
            reason=(
                "promoted: 3-class edge over the persistence baseline is positive and statistically significant"
                if passed else
                f"not promoted: mean_edge={mean_edge:.4f} (need >= {self.min_mean_edge}), "
                f"p={p_value:.4f} (need < {self.significance_level})"
            ),
            mean_edge=mean_edge, p_value=float(p_value), n_folds=len(fold_edges),
            fold_edges=edges.tolist(), n_usable_rows=int(len(X)),
        )
        self.last_gate_result = result
        if passed:
            self.model = self._new_classifier().fit(X, y)
            self.trained_at = time.time()
        return result
