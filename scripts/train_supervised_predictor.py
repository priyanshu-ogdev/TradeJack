"""
Train Supervised Predictor
===========================
Builds a training table (data_forge/training_table_builder.py), trains and
walk-forward-gates a SupervisedPredictor (training/supervised_predictor.py) on it,
and -- only if it actually passes promotion -- writes its predictions leg
(supervised_pred/) so data_forge/training_table_builder.py's own
latest_supervised_prediction() has real data to read.

This is the piece that was missing even after last session's fix: the schema and path
methods existed, write_predictions_leg() would no longer crash, but nothing actually
called train_evaluate_and_promote() + write_predictions_leg() together against real
data. Nothing before this script has ever run this end-to-end.

WHAT'S CONFIGURABLE VS. ASSUMED (read before running against a real data store):
  --feature-columns / --price-column are NOT auto-detected. The merged training
  table's columns are prefixed per TimeAligner's convention (e.g.
  "eurusd_physics__mid_price", "eurusd_toxicity__bvc_vpin") and differ depending on
  which legs were requested -- guessing a default here risked silently training on
  the wrong (or an accidentally non-existent) column with no error, so both are
  required CLI arguments, not defaulted. Run --list-columns first against your
  target date range to see what's actually available before choosing them.

A REJECTED MODEL WRITES NOTHING: if train_evaluate_and_promote's gate doesn't pass,
this script logs why and exits without calling write_predictions_leg() at all --
consistent with the "absence is neutral, not evidence" convention this leg (and
every other optional leg in this codebase) already follows. An untrustworthy model
must not populate a leg downstream code would otherwise treat as a real signal.

NOT EXECUTABLE IN THE SANDBOX THIS WAS WRITTEN IN: everything downstream of
TrainingTableBuilder.build() needs polars + pydantic_settings, neither installed
here. Written and statically reviewed against the already-tested primitives it
calls (train_evaluate_and_promote, build_features_and_labels, write_predictions_leg
-- each independently covered by tests/test_supervised_predictor.py's real runs),
not executed end-to-end. Run this for real, on a real data store, before trusting it
in a scheduled job.

Usage:
  python scripts/train_supervised_predictor.py \\
      --symbol EURUSD --forex-pairs EURUSD \\
      --start-date 2026-01-01 --end-date 2026-01-31 \\
      --price-column eurusd_physics__mid_price \\
      --feature-columns eurusd_physics__spread eurusd_physics__quote_imbalance \\
                         eurusd_toxicity__bvc_vpin eurusd_toxicity__kyles_lambda \\
      --model-out state/supervised_predictors/EURUSD

  python scripts/train_supervised_predictor.py --list-columns \\
      --symbol EURUSD --forex-pairs EURUSD --start-date 2026-01-01 --end-date 2026-01-31
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (TrainSupervisedPredictor) %(message)s")
logger = logging.getLogger("TrainSupervisedPredictor")


def _log_returns(price: np.ndarray) -> np.ndarray:
    """Same log-return convention as execution/portfolio_signal_builder.py's
    realized_volatility() -- returns[0] is NaN (undefined, no prior price), not 0.0,
    so it's honestly missing rather than silently claiming "no change" for the first
    row. HistGradientBoosting (the model this feeds) handles NaN natively."""
    prices = np.clip(np.asarray(price, dtype=np.float64), 1e-12, None)
    out = np.full(len(prices), np.nan, dtype=np.float64)
    out[1:] = np.diff(np.log(prices))
    return out


def build_training_inputs(
    start_date: str, end_date: str, price_column: str, feature_columns: List[str],
    crypto_symbol: Optional[str] = None, forex_pairs: Optional[List[str]] = None,
    sentiment_symbol: Optional[str] = None, data_store_dir: Optional[str] = None,
    frequency: str = "1m", max_staleness_buckets: int = 5,
):
    """Builds the merged training table and extracts (timestamps, features, returns)
    from it. Returns None if the table build produced nothing (see
    TrainingTableBuilder.build()'s own "all requested legs were empty" case)."""
    from data_forge.training_table_builder import TrainingTableBuilder

    builder = TrainingTableBuilder(data_store_dir=data_store_dir)
    result = builder.build(
        start_date, end_date, crypto_symbol=crypto_symbol, forex_pairs=forex_pairs,
        sentiment_symbol=sentiment_symbol, frequency=frequency, max_staleness_buckets=max_staleness_buckets,
    )
    if result is None:
        logger.error(f"No training table could be built for {start_date}..{end_date}.")
        return None

    df = result.data
    missing = [c for c in [price_column] + feature_columns if c not in df.columns]
    if missing:
        logger.error(f"Requested columns not present in the built table: {missing}. "
                     f"Available columns: {sorted(df.columns)}")
        return None

    timestamps = df["timestamp"].to_list()
    price = df[price_column].to_numpy()
    features = df.select(feature_columns).to_numpy()
    returns = _log_returns(price)

    logger.info(f"Built training inputs: {len(timestamps)} rows, {len(feature_columns)} feature columns.")
    return timestamps, features, returns


def train_and_write_leg(
    symbol: str, timestamps, features: np.ndarray, returns: np.ndarray,
    lookback: int, horizon: int, min_mean_edge: float, n_splits: int,
    model_out_dir: Optional[str], data_store_dir: Optional[str],
) -> bool:
    """Trains, gates, and (only on pass) writes the predictions leg + saves the model.
    Returns True if promoted and written, False if rejected (nothing written)."""
    from training.supervised_predictor import (
        PredictorConfig, train_evaluate_and_promote, build_features_and_labels, write_predictions_leg,
    )

    config = PredictorConfig(lookback=lookback, horizon=horizon, min_mean_edge=min_mean_edge, n_splits=n_splits)
    model, report = train_evaluate_and_promote(features, returns, config)

    if not report.get("passed"):
        logger.info(f"{symbol}: predictor REJECTED by promotion gate -- {report}. Writing nothing.")
        return False

    logger.info(f"{symbol}: predictor PROMOTED -- {report}")

    # Predict across every historical window (not just the latest) so the written
    # leg has a full time series to join via TimeAligner, same shape as every other
    # leg in this codebase -- not just a single most-recent point.
    X, _, _, origin_idx = build_features_and_labels(features, returns, lookback, horizon)
    preds = model.predict(X)
    pred_timestamps = [timestamps[i] for i in origin_idx]

    model_version = f"{symbol}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    # NOTE: evaluate_promotion()'s report dict has no "trained_at" key (checked
    # directly against training/supervised_predictor.py's source, not assumed) --
    # model_version is derived from the actual current time instead, so it's always a
    # real, unique identifier rather than silently containing the literal string
    # "unknown" for every promoted model.

    # Group predictions by calendar date -- write_predictions_leg() writes one
    # Parquet file per day, matching every other leg's daily-partition convention.
    by_date = {}
    for ts, p_up, exp_ret in zip(pred_timestamps, preds["direction_up_proba"], preds["expected_return"]):
        date_str = ts.strftime("%Y-%m-%d")
        by_date.setdefault(date_str, {"timestamps": [], "proba": [], "expected": []})
        by_date[date_str]["timestamps"].append(ts)
        by_date[date_str]["proba"].append(p_up)
        by_date[date_str]["expected"].append(exp_ret)

    written_paths = []
    for date_str, rows in sorted(by_date.items()):
        out_path = write_predictions_leg(
            symbol=symbol, timestamps=rows["timestamps"],
            direction_up_proba=np.array(rows["proba"]), expected_return=np.array(rows["expected"]),
            model_version=model_version, date=date_str, data_store_dir=data_store_dir,
        )
        written_paths.append(out_path)

    if model_out_dir:
        os.makedirs(model_out_dir, exist_ok=True)
        model_path = os.path.join(model_out_dir, f"{model_version}.joblib")
        model.save(model_path)
        report_path = os.path.join(model_out_dir, f"{model_version}.report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"{symbol}: model saved -> {model_path}, gate report -> {report_path}")

    logger.info(f"{symbol}: wrote {len(written_paths)} daily prediction legs.")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True, help="Instrument symbol to train a predictor for.")
    parser.add_argument("--crypto-symbol", default=None, help="Crypto leg symbol (mutually exclusive in practice with --forex-pairs, though both can be passed if you know what you're doing).")
    parser.add_argument("--forex-pairs", nargs="*", default=None, help="FX pairs to pull physics+toxicity legs for.")
    parser.add_argument("--sentiment-symbol", default=None)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--price-column", required=True, help="Column to derive log returns from (e.g. eurusd_physics__mid_price).")
    parser.add_argument("--feature-columns", nargs="+", default=None, help="Feature columns to train on. Required unless --list-columns.")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--min-mean-edge", type=float, default=0.02)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--data-store", default=None)
    parser.add_argument("--model-out", default=None, help="Directory to save the promoted model + gate report. Omit to skip saving.")
    parser.add_argument("--list-columns", action="store_true", help="Build the table, print available columns, and exit without training.")
    args = parser.parse_args()

    if args.list_columns:
        from data_forge.training_table_builder import TrainingTableBuilder
        builder = TrainingTableBuilder(data_store_dir=args.data_store)
        result = builder.build(
            args.start_date, args.end_date, crypto_symbol=args.crypto_symbol,
            forex_pairs=args.forex_pairs, sentiment_symbol=args.sentiment_symbol,
        )
        if result is None:
            logger.error("No table could be built -- nothing to list.")
            sys.exit(1)
        print("Available columns:")
        for c in sorted(result.data.columns):
            print(f"  {c}")
        return

    if not args.feature_columns:
        logger.error("--feature-columns is required (unless --list-columns). Run --list-columns first to see what's available.")
        sys.exit(1)

    built = build_training_inputs(
        args.start_date, args.end_date, args.price_column, args.feature_columns,
        crypto_symbol=args.crypto_symbol, forex_pairs=args.forex_pairs,
        sentiment_symbol=args.sentiment_symbol, data_store_dir=args.data_store,
    )
    if built is None:
        sys.exit(1)
    timestamps, features, returns = built

    promoted = train_and_write_leg(
        args.symbol, timestamps, features, returns,
        lookback=args.lookback, horizon=args.horizon, min_mean_edge=args.min_mean_edge, n_splits=args.n_splits,
        model_out_dir=args.model_out, data_store_dir=args.data_store,
    )
    sys.exit(0 if promoted else 2)  # distinct exit code for "ran fine, but gate rejected" vs. a real error


if __name__ == "__main__":
    main()
