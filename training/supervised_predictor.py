"""
Supervised Predictor -- a clean, independent prediction pack that produces an
auxiliary signal (predicted forward-return direction + magnitude) from the
same engineered features data_forge already computes, validated with a real
statistical gate before anything gets promoted.

Why this exists, and what it deliberately is not
--------------------------------------------------
The RL agent (swarm/rl_trainer.py) learns a *policy* end-to-end from raw
sequences via reward signal alone -- it never gets an explicit "the model
thinks price goes up" feature; the DilatedCNN encoder has to discover
anything like that itself, from a reward signal that's sparse and noisy
compared to direct supervision. A supervised model trained directly on
"predict the forward return" is a much denser, much cheaper-to-fit learning
problem, and its output can be handed to the RL agent as one more observation
feature -- cheap, stable, well-understood domain features an actor-critic
would otherwise have to reinvent from scratch. This is not a replacement for
the RL policy; it's an auxiliary signal.

What this deliberately does NOT do: touch physics/lob_env.py's observation
space. That's the same discipline as training_table_builder.py -- producing
a joinable artifact vs. widening the model's input shape are two different
decisions, and the second one belongs to whoever owns the RL architecture,
made on purpose, not as a side effect of adding a prediction pack.

Design choices, with reasons
------------------------------
- HistGradientBoostingClassifier/Regressor (sklearn), not a hand-rolled
  network: no GPU dependency, trains in seconds on this data volume, native
  NaN handling (the *_is_stale forward-filled gaps in every leg this project
  produces are NaN or stale-flagged, not clean), and no scaling/imputation
  pipeline to get subtly wrong. This is the appropriate first supervised
  model for tabular microstructure features -- swap in something heavier
  later if walk-forward results say it's worth it, not before.
- Purged, embargoed time-series CV (Lopez de Prado, "Advances in Financial
  Machine Learning", 2018, ch. 7), not plain k-fold or a bare
  scikit-learn TimeSeriesSplit: every row's features span `lookback` past
  steps and its label spans `horizon` future steps, so a naive split leaks
  information across the train/test boundary whenever a training window and
  a test window overlap in real time. The embargo gap removes exactly that.
- A persistence baseline (predict the same direction as the most recently
  realized return), not a coin-flip baseline: a coin-flip is trivially easy
  to beat on directional accuracy and proves nothing about whether the model
  learned anything a much dumber heuristic didn't already give you for free.
- Wilcoxon signed-rank on fold-level edge, not a raw pooled accuracy
  comparison: matches this project's existing statistical-gate style
  (escrow/validation_airgap.py's Mann-Whitney gate for RL candidates) rather
  than inventing a different methodology for this one component, and is
  non-parametric -- appropriate for a handful of fold-level numbers with no
  reason to assume they're normally distributed.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (SupervisedPredictor) %(message)s")
logger = logging.getLogger("SupervisedPredictor")


# ---------------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------------- #

@dataclass
class PredictorConfig:
    lookback: int = 10          # steps of feature history flattened into each row
    horizon: int = 5            # steps ahead the label looks forward over
    n_splits: int = 5           # walk-forward folds for the promotion gate
    embargo: Optional[int] = None  # rows purged between train/test; default = lookback + horizon
    min_mean_edge: float = 0.02    # required mean directional-accuracy edge over the persistence baseline
    significance_level: float = 0.05
    max_depth: int = 4
    max_iter: int = 150
    l2_regularization: float = 1.0
    random_state: int = 42

    def resolved_embargo(self) -> int:
        # Exactly enough of a gap that no test-fold row's lookback window or
        # forward-label window can touch a training-fold row's, in either
        # direction across the boundary.
        return self.embargo if self.embargo is not None else (self.lookback + self.horizon)


# ---------------------------------------------------------------------- #
# Purged, embargoed walk-forward split
# ---------------------------------------------------------------------- #

class PurgedTimeSeriesSplit:
    """Contiguous, expanding-window walk-forward split with an embargo gap
    between each train block and its following test block. Deliberately not
    randomly shuffled k-fold -- these are time-ordered feature rows built
    from overlapping windows, and shuffling would put a training row's
    "future" on both sides of a test row, which is a worse leak than the one
    the embargo prevents."""

    def __init__(self, n_splits: int = 5, embargo: int = 0):
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, n_samples: int):
        fold_size = n_samples // (self.n_splits + 1)
        if fold_size <= 0:
            return
        for i in range(1, self.n_splits + 1):
            train_end = i * fold_size
            test_start = train_end + self.embargo
            test_end = min(test_start + fold_size, n_samples)
            if test_start >= test_end or train_end <= 0:
                continue
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx


# ---------------------------------------------------------------------- #
# Feature / label construction
# ---------------------------------------------------------------------- #

def build_features_and_labels(
    features: np.ndarray, returns: np.ndarray, lookback: int = 10, horizon: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    features: (T, F) engineered features, one row per timestep. NaN is fine
    (HistGradientBoosting handles it natively) -- do not impute upstream of
    this and silently hide staleness from the model.
    returns: (T,) per-timestep simple or log returns.

    A row built at origin index t uses features[t-lookback+1 : t+1]
    (inclusive of t) plus returns[t-lookback+1 : t+1] as its feature vector,
    and labels off returns[t+1 : t+1+horizon] -- strictly after t. That
    boundary is exactly where a lookahead bug hides, so it's covered by
    test_build_features_no_lookahead in tests/test_supervised_predictor.py
    rather than left as an assertion nobody checks.

    Returns (X, y_class, y_reg, origin_idx):
      X          (n, lookback*F + lookback)
      y_class    (n,) in {0, 1} -- 1 if forward cumulative return > 0
      y_reg      (n,) forward cumulative return over `horizon` steps
      origin_idx (n,) the t each row was built from, for mapping back to a
                 timestamp/date at the call site.
    """
    features = np.asarray(features, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    T, F = features.shape
    if returns.shape[0] != T:
        raise ValueError(f"features has {T} rows but returns has {returns.shape[0]}")

    rows_X, rows_yc, rows_yr, origin_idx = [], [], [], []
    for t in range(lookback - 1, T - horizon):
        window = features[t - lookback + 1: t + 1]
        recent_returns = returns[t - lookback + 1: t + 1]
        x = np.concatenate([window.flatten(), recent_returns])
        forward_ret = float(np.sum(returns[t + 1: t + 1 + horizon]))
        rows_X.append(x)
        rows_yr.append(forward_ret)
        rows_yc.append(1 if forward_ret > 0 else 0)
        origin_idx.append(t)

    if not rows_X:
        n_feat = lookback * F + lookback
        return (np.empty((0, n_feat)), np.empty((0,), dtype=int),
                np.empty((0,)), np.empty((0,), dtype=int))

    return (np.array(rows_X), np.array(rows_yc, dtype=int),
            np.array(rows_yr), np.array(origin_idx, dtype=int))


def persistence_baseline_signal(X: np.ndarray) -> np.ndarray:
    """The persistence baseline's own signal for each row: the most recently
    realized return, i.e. the LAST engineered column build_features_and_labels
    puts into X (recent_returns[-1]). Read back out of X rather than
    recomputed separately, so the baseline definition can never silently
    drift from what's actually in the feature vector the model itself sees."""
    return X[:, -1]


# ---------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------- #

class SupervisedPredictor:
    """A trained (classifier, regressor) pair predicting forward direction
    and forward magnitude from the same feature vector. Two separate models
    rather than one multi-output model because a classifier calibrated on
    0/1 direction and a regressor fit on signed magnitude have different loss
    surfaces; sklearn's HistGradientBoosting doesn't support true multi-task
    output anyway, and forcing one estimator to do both jobs is not simpler,
    just more implicit."""

    def __init__(self, config: Optional[PredictorConfig] = None):
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for SupervisedPredictor.")
        self.config = config or PredictorConfig()
        self.classifier: Optional[HistGradientBoostingClassifier] = None
        self.regressor: Optional[HistGradientBoostingRegressor] = None
        self.trained_at: Optional[float] = None
        self.n_train_samples: Optional[int] = None

    def _new_classifier(self) -> "HistGradientBoostingClassifier":
        c = self.config
        return HistGradientBoostingClassifier(
            max_depth=c.max_depth, max_iter=c.max_iter,
            l2_regularization=c.l2_regularization, random_state=c.random_state,
        )

    def _new_regressor(self) -> "HistGradientBoostingRegressor":
        c = self.config
        return HistGradientBoostingRegressor(
            max_depth=c.max_depth, max_iter=c.max_iter,
            l2_regularization=c.l2_regularization, random_state=c.random_state,
        )

    def fit(self, X: np.ndarray, y_class: np.ndarray, y_reg: np.ndarray) -> "SupervisedPredictor":
        self.classifier = self._new_classifier().fit(X, y_class)
        self.regressor = self._new_regressor().fit(X, y_reg)
        self.trained_at = time.time()
        self.n_train_samples = int(len(X))
        return self

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        if self.classifier is None or self.regressor is None:
            raise RuntimeError("SupervisedPredictor.predict() called before fit()/load().")
        proba_up = self.classifier.predict_proba(X)[:, 1]
        expected_return = self.regressor.predict(X)
        return {"direction_up_proba": proba_up, "expected_return": expected_return}

    def save(self, path: str):
        if not JOBLIB_AVAILABLE:
            raise RuntimeError("joblib is required to save/load SupervisedPredictor.")
        joblib.dump({
            "config": self.config, "classifier": self.classifier, "regressor": self.regressor,
            "trained_at": self.trained_at, "n_train_samples": self.n_train_samples,
        }, path)

    @classmethod
    def load(cls, path: str) -> "SupervisedPredictor":
        if not JOBLIB_AVAILABLE:
            raise RuntimeError("joblib is required to save/load SupervisedPredictor.")
        payload = joblib.load(path)
        obj = cls(config=payload["config"])
        obj.classifier = payload["classifier"]
        obj.regressor = payload["regressor"]
        obj.trained_at = payload["trained_at"]
        obj.n_train_samples = payload["n_train_samples"]
        return obj


# ---------------------------------------------------------------------- #
# Walk-forward evaluation + promotion gate
# ---------------------------------------------------------------------- #

@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    model_accuracy: float
    baseline_accuracy: float
    edge: float
    model_mae: float
    baseline_mae: float


def walk_forward_evaluate(
    X: np.ndarray, y_class: np.ndarray, y_reg: np.ndarray, config: Optional[PredictorConfig] = None
) -> List[FoldResult]:
    """Trains a fresh model per fold (never reuses a fold's fitted model
    across folds -- that would leak the later folds' influence backward) and
    scores it against the persistence baseline on that fold's held-out data."""
    config = config or PredictorConfig()
    baseline_signal = persistence_baseline_signal(X)
    splitter = PurgedTimeSeriesSplit(n_splits=config.n_splits, embargo=config.resolved_embargo())

    results = []
    for i, (train_idx, test_idx) in enumerate(splitter.split(len(X))):
        model = SupervisedPredictor(config).fit(X[train_idx], y_class[train_idx], y_reg[train_idx])
        preds = model.predict(X[test_idx])

        pred_class = (preds["direction_up_proba"] >= 0.5).astype(int)
        model_acc = float(np.mean(pred_class == y_class[test_idx]))

        baseline_pred_class = (baseline_signal[test_idx] > 0).astype(int)
        baseline_acc = float(np.mean(baseline_pred_class == y_class[test_idx]))

        model_mae = float(np.mean(np.abs(preds["expected_return"] - y_reg[test_idx])))
        baseline_mae = float(np.mean(np.abs(y_reg[test_idx])))  # baseline: predict zero return

        results.append(FoldResult(
            fold=i, n_train=len(train_idx), n_test=len(test_idx),
            model_accuracy=model_acc, baseline_accuracy=baseline_acc, edge=model_acc - baseline_acc,
            model_mae=model_mae, baseline_mae=baseline_mae,
        ))
    return results


def evaluate_promotion(fold_results: List[FoldResult], config: Optional[PredictorConfig] = None) -> Dict[str, Any]:
    """The actual gate: promote only if the model's directional-accuracy edge
    over the persistence baseline is both practically meaningful
    (mean_edge >= min_mean_edge) and statistically reliable across folds
    (one-sided Wilcoxon signed-rank p < significance_level) -- not either one
    alone. A model that's barely better on average but consistent, or wildly
    better on one lucky fold and useless on the rest, should both fail this."""
    config = config or PredictorConfig()
    if not SCIPY_AVAILABLE:
        return {"passed": False, "reason": "scipy is required for the promotion gate's significance test"}
    if len(fold_results) < 2:
        return {"passed": False, "reason": f"only {len(fold_results)} usable fold(s); need >=2 for a significance test"}
    if len(fold_results) < 5:
        logger.warning(
            f"evaluate_promotion: only {len(fold_results)} folds -- one-sided Wilcoxon's minimum "
            f"achievable p-value at this n is {1.0 / (2 ** len(fold_results)):.4f}, which may sit above "
            f"config.significance_level={config.significance_level} regardless of effect size. "
            f"See data_forge/predictor.py's fit_and_gate for where this bit a real test at n_splits=4."
        )

    edges = np.array([f.edge for f in fold_results])
    mean_edge = float(np.mean(edges))

    if np.allclose(edges, edges[0]):
        # Wilcoxon is undefined when every paired difference is identical
        # (all-zero differences after centering) -- scipy raises ValueError
        # in that case. Handle the degenerate case explicitly instead of
        # letting that surface as an opaque crash from inside the gate.
        p_value = 0.0 if edges[0] > 0 else 1.0
    else:
        # CONSTRAINT DISCOVERED VIA REAL EXECUTION, NOT INSPECTION (see
        # data_forge/predictor.py's fit_and_gate for the incident): one-sided
        # Wilcoxon signed-rank's minimum achievable p-value with n paired
        # samples is 1/2^n (all differences the same sign is the single most
        # extreme rank configuration under the null). Below n_splits=5, that
        # floor (1/16=0.0625 at n=4) sits ABOVE this gate's default 0.05
        # significance_level, so a model winning every fold by a wide margin
        # still cannot pass -- data_forge/predictor.py hit exactly this with
        # n_splits=4 and switched to a one-sample t-test instead. This module
        # keeps Wilcoxon (its default n_splits=5 clears the 1/32=0.03125
        # floor), but do not lower n_splits below 5 without either raising
        # significance_level accordingly or switching tests here too.
        try:
            _, p_value = wilcoxon(edges, alternative="greater")
        except ValueError as e:
            return {"passed": False, "reason": f"Wilcoxon test failed ({e}); refusing to promote on an unverifiable result"}

    passed = bool(mean_edge >= config.min_mean_edge and p_value < config.significance_level)
    return {
        "passed": passed,
        "mean_edge": mean_edge,
        "p_value": float(p_value),
        "n_folds": len(fold_results),
        "min_mean_edge_required": config.min_mean_edge,
        "significance_level_required": config.significance_level,
        "fold_edges": edges.tolist(),
        "reason": (
            "promoted: edge over the persistence baseline is positive and statistically significant"
            if passed else
            f"not promoted: mean_edge={mean_edge:.4f} (need >= {config.min_mean_edge}), "
            f"p={p_value:.4f} (need < {config.significance_level})"
        ),
    }


def train_evaluate_and_promote(
    features: np.ndarray, returns: np.ndarray, config: Optional[PredictorConfig] = None
) -> Tuple[Optional[SupervisedPredictor], Dict[str, Any]]:
    """End-to-end: build features/labels, walk-forward evaluate, gate, and --
    only if the gate passes -- fit and return one final model on ALL the
    data (the walk-forward models were fold-scoped and already discarded;
    they exist to prove the *method* works, not to be the deployed model).
    Returns (model_or_None, gate_report)."""
    config = config or PredictorConfig()
    X, y_class, y_reg, origin_idx = build_features_and_labels(
        features, returns, lookback=config.lookback, horizon=config.horizon
    )
    if len(X) < (config.n_splits + 1) * 10:
        return None, {"passed": False, "reason": f"only {len(X)} usable rows -- too few for {config.n_splits} folds"}

    fold_results = walk_forward_evaluate(X, y_class, y_reg, config)
    gate_report = evaluate_promotion(fold_results, config)
    gate_report["n_usable_rows"] = int(len(X))

    if not gate_report["passed"]:
        return None, gate_report

    final_model = SupervisedPredictor(config).fit(X, y_class, y_reg)
    return final_model, gate_report


# ---------------------------------------------------------------------- #
# Leg writer -- slots into data_forge/training_table_builder.py's existing
# convention with zero changes needed to TimeAligner or the alignment core.
# ---------------------------------------------------------------------- #

def write_predictions_leg(
    symbol: str, timestamps, direction_up_proba, expected_return: "np.ndarray",
    model_version: str, date: str, data_store_dir: Optional[str] = None,
) -> str:
    """
    Writes one day's predictions to processed/<symbol>/supervised_pred/YYYY/MM/DD/
    <symbol>-supervised_pred-<date>.parquet -- exactly the path
    TrainingTableBuilder._supervised_pred_path() expects, so the two sides of this
    integration can only drift apart if one of them is edited without the other
    (there is no third, independent source of truth for the path to fall out of
    sync with).

    timestamps: sequence of datetime (UTC) -- one per prediction, the origin
    timestamp `t` from build_features_and_labels (NOT the future timestamp the
    prediction is *about* -- consumers joining this leg via TimeAligner want it
    indexed by when the prediction was available, same as every other leg).

    Not executable in this sandbox (no polars here) -- this is the one part of
    this module not covered by tests/test_supervised_predictor.py's real runs.
    Static review only; run data_forge's own offline test suite against this
    before trusting it, same caveat as the rest of data_forge's recent history.
    """
    try:
        import polars as pl
    except ImportError as e:
        raise RuntimeError("polars is required to write a supervised_pred leg.") from e

    from data_forge.config import config as _config
    from data_forge.schema import SupervisedPredictionSchema

    store_dir = data_store_dir or _config.data_store_dir
    out_dir = os.path.join(store_dir, "processed", symbol, "supervised_pred", date.replace("-", "/"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}-supervised_pred-{date}.parquet")

    df = pl.DataFrame({
        "timestamp": list(timestamps),
        "direction_up_proba": np.asarray(direction_up_proba, dtype=np.float64).tolist(),
        "expected_return": np.asarray(expected_return, dtype=np.float64).tolist(),
        "model_version": [model_version] * len(list(timestamps)),
    })
    df = SupervisedPredictionSchema.validate(df)
    df.write_parquet(out_path, compression=_config.compression_codec, row_group_size=_config.row_group_size)
    logger.info(f"Wrote {len(df)} predictions -> {out_path}")
    return out_path
