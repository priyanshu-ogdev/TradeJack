"""
10x Rolling Validation Airgap (`ValidationAirgapEngine`).
Evaluates candidate model checkpoints in a sandboxed simulation across N out-of-sample
historical market splits (including simulated liquidity vacuums and high volatility crashes).
If the candidate model fails on key metrics, the weights are classified as overfitted and rejected.

v3 Upgrade:
  - Loads SB3 checkpoints (.zip) via model.load() instead of raw state_dict
  - Falls back to baseline models for dummy/missing weight paths
  - Supports both v3 model names (PPO-DilatedCNN, SAC-DilatedCNN, etc.)
    and legacy names (Dilated-CNN-Seq2seq) for backward compatibility
"""

import os
import sys
import glob
import time
import math
import random
import json
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (ValidationAirgap) %(message)s")
logger = logging.getLogger("ValidationAirgap")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from stable_baselines3 import PPO, SAC, DQN
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from physics.lob_env import TradeJackLOBEnv

# v3 model name mapping (legacy → new)
LEGACY_MODEL_MAP = {
    "Dilated-CNN-Seq2seq": "PPO-DilatedCNN",
    "Attention-is-all-you-Need": "PPO-Transformer",
    "Deep-Q-learning": "DuelingDQN",
    "LSTM-Seq2Seq-VAE": "SAC-DilatedCNN",
    "Actor-Critic-Duel-Agent": "SAC-DilatedCNN",
}


def latest_processed_data_date(symbol: str, data_store_dir: str) -> Optional[str]:
    """
    PHASE 2: scans data_store/processed/{symbol}/physics/{Y}/{M}/{D}/*.parquet for the
    most recent date a physics file actually exists for, by partition directory name
    (not file mtime -- a file's mtime can be newer than the market data it contains,
    e.g. if it was re-processed or copied; the Y/M/D partition path is what the data
    itself claims to represent, matching how feature_engineering.py writes it and how
    training/continuous_trainer.py's _bridge_live_data() names it).

    Returns "YYYY-MM-DD", or None if no processed physics files exist at all for this
    symbol (a fresh install, or a symbol that's never been processed) -- distinct from
    "found files but they're all old," which returns a real (stale) date instead.

    Pure filesystem logic -- no torch/polars/pydantic_settings dependency, so this is
    fully testable and usable even in environments missing those.
    """
    physics_dir = os.path.join(data_store_dir, "processed", symbol, "physics")
    pattern = os.path.join(physics_dir, "*", "*", "*", "*.parquet")
    files = glob.glob(pattern)
    if not files:
        return None

    dates = []
    for f in files:
        # Path shape: .../physics/{Y}/{M}/{D}/whatever.parquet -- pull Y/M/D from the
        # three directory components immediately above the file, not from the
        # filename (which varies: "physics.parquet" for the bulk pipeline output).
        parts = os.path.normpath(f).split(os.sep)
        try:
            day, month, year = parts[-2], parts[-3], parts[-4]
            dates.append(f"{year}-{month}-{day}")
        except IndexError:
            continue  # unexpected path shape -- skip rather than crash the whole scan

    if not dates:
        return None
    return max(dates)  # ISO "YYYY-MM-DD" strings sort correctly as plain strings


def check_data_freshness(
    symbol: str, data_store_dir: str, max_staleness_days: int
) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Returns (is_fresh, latest_date_found, days_stale). is_fresh=False whenever
    either no processed data exists at all, or the newest partition found is older
    than max_staleness_days relative to now (UTC). days_stale is None when
    latest_date_found is None (nothing to measure staleness against).
    """
    latest = latest_processed_data_date(symbol, data_store_dir)
    if latest is None:
        return False, None, None

    latest_dt = datetime.strptime(latest, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days_stale = (datetime.now(timezone.utc) - latest_dt).days
    return days_stale <= max_staleness_days, latest, days_stale


class ValidationAirgapEngine:
    """
    Sandboxed evaluation chamber running candidate weights across N distinct historical stress tests.
    """

    def __init__(
        self,
        num_splits: int = 10,
        min_required_sharpe: float = 1.0,
        max_allowed_drawdown: float = 0.15,
        data_store_dir: str = "data_store"
    ):
        self.num_splits = num_splits
        self.min_sharpe = min_required_sharpe
        self.max_drawdown = max_allowed_drawdown
        self.data_store_dir = os.path.abspath(data_store_dir)

    def _resolve_model_name(self, model_type: str) -> str:
        """Map legacy model names to v3 names."""
        return LEGACY_MODEL_MAP.get(model_type, model_type)

    def _load_model_from_weights(self, weights_path: str, model_type: str = "PPO-DilatedCNN"):
        """
        Load a candidate model for evaluation.

        v3: Loads SB3 checkpoint (.zip) via PPO.load() / SAC.load() / DQN.load().
        Falls back to a random-weight baseline if the checkpoint doesn't exist
        (which is correct — dummy weights SHOULD fail the airgap).
        """
        resolved = self._resolve_model_name(model_type)

        # Determine SB3 algorithm class from model name
        algo_map = {"PPO": PPO, "SAC": SAC, "DQN": DQN} if SB3_AVAILABLE else {}
        algo_prefix = resolved.split("-")[0] if "-" in resolved else resolved
        algo_cls = algo_map.get(algo_prefix)

        # Try loading SB3 checkpoint
        if SB3_AVAILABLE and algo_cls:
            for ext in ["", ".zip"]:
                path = weights_path + ext if ext else weights_path
                if os.path.exists(path):
                    try:
                        model = algo_cls.load(path, device="cpu")
                        logger.info(f"Loaded SB3 checkpoint: {path} ({resolved})")
                        return model
                    except Exception as e:
                        logger.warning(f"Failed to load SB3 checkpoint ({path}): {e}")

        # Fallback: use a baseline model (random policy or momentum)
        # This is intentional — a dummy/missing checkpoint SHOULD fail the airgap
        logger.info(f"No valid checkpoint at '{weights_path}'. Using random baseline for evaluation.")
        return None  # Signals to use random/flat actions

    def _predict_action(self, model, obs: Dict[str, np.ndarray]) -> float:
        """Get action from model, handling both SB3 and None (random) models."""
        if model is None:
            # Random/flat baseline — should fail airgap
            return float(np.random.uniform(-0.2, 0.2))

        if SB3_AVAILABLE and hasattr(model, "predict"):
            try:
                action, _ = model.predict(obs, deterministic=True)
                if isinstance(action, np.ndarray):
                    return float(np.clip(action[0], -1.0, 1.0))
                return float(np.clip(action, -1.0, 1.0))
            except Exception:
                return 0.0

        return 0.0

    def evaluate_candidate_weights(
        self,
        weights_path: str,
        model_type: str = "PPO-DilatedCNN",
        symbol: str = "BTC-USDT"
    ) -> Dict[str, Any]:
        """
        Runs the Nx Rolling Validation Airgap. Returns validation status and metrics across splits.
        """
        resolved = self._resolve_model_name(model_type)
        logger.info(f"[AIRGAP COMMENCING] Sandboxing weights '{weights_path}' ({resolved}) across {self.num_splits} stress splits...")

        model = self._load_model_from_weights(weights_path, model_type=model_type)
        split_sharpes: List[float] = []
        split_drawdowns: List[float] = []
        split_equities: List[float] = []

        for split_idx in range(self.num_splits):
            # Each split tests a different deterministic random seed and simulated LOB shock profile
            seed = 1000 + split_idx * 17
            random.seed(seed)
            np.random.seed(seed)

            env = TradeJackLOBEnv(
                symbol=symbol,
                initial_cash=10.0,
                data_store_dir=self.data_store_dir,
                child_id=999  # Airgap sandbox id
            )

            obs, info = env.reset(seed=seed)
            terminated = False
            truncated = False

            # BUG FOUND BY RUNNING THIS, NOT BY READING IT: env.reset(seed=seed) does
            # NOT actually vary which market data this split sees.
            # TradeJackLOBEnv._partition_streamer() always walks
            # sorted(scan_available_partitions(...)) from the start, and
            # scan_available_partitions() returns a fixed sorted() list independent
            # of the seed — so every split replayed the exact same tick-0 window.
            # Combined with a deterministic frozen policy (see _predict_action's
            # deterministic=True below), that means every "stress split" produced
            # bit-identical results — confirmed empirically: all 5 splits returned
            # avg_sharpe=0.13789063752799535 to full float precision in testing.
            # A validation gate that always evaluates the same scenario provides
            # zero information about robustness, regardless of how many "splits"
            # it claims to run.
            #
            # Fix: actually vary the window per split by skipping a random number
            # of ticks (seeded, so reproducible) before starting the measured
            # window, using a neutral action so the skip itself has no effect on
            # accounting beyond time passing. This works whether the data store
            # has 2 days (synthetic/dev) or 200 (real historical) — it doesn't
            # depend on there being enough distinct partition files to pick from.
            max_warmup = 400  # keep well under a typical partition's tick count so the eval window doesn't run dry
            warmup_ticks = int(np.random.randint(0, max_warmup)) if split_idx > 0 else 0
            for _ in range(warmup_ticks):
                obs, _, terminated, truncated, info = env.step([0.0])  # neutral: don't distort the measured window's start
                if terminated or truncated:
                    break

            # Run up to 100 steps on this split
            for _ in range(100):
                action = self._predict_action(model, obs)
                obs, reward, terminated, truncated, info = env.step([action])
                if terminated or truncated:
                    break

            # Calculate split metrics
            eq = info["equity"]
            dd = info["max_drawdown"]
            # Approximate split Sharpe from equity change
            ret = (eq - 10.0) / 10.0
            split_sharpe = ret * 10.0 if dd < 0.05 else ret / (dd + 1e-4)

            split_equities.append(eq)
            split_drawdowns.append(dd)
            split_sharpes.append(split_sharpe)

        avg_sharpe = float(np.mean(split_sharpes))
        max_dd = float(np.max(split_drawdowns))
        passed_airgap = (avg_sharpe >= self.min_sharpe) and (max_dd <= self.max_drawdown)

        if passed_airgap:
            logger.info(f"[AIRGAP PASSED] Candidate weights cleared validation. Avg Sharpe: {avg_sharpe:.2f}, Max DD: {max_dd*100:.1f}%.")
        else:
            logger.warning(
                f"[AIRGAP REJECTED] Candidate weights failed validation! Avg Sharpe: {avg_sharpe:.2f} (Req >= {self.min_sharpe}), "
                f"Max DD: {max_dd*100:.1f}% (Req <= {self.max_drawdown*100:.1f}%). Flagged as overfitted/poisoned."
            )

        return {
            "passed": passed_airgap,
            "weights_path": weights_path,
            "model_type": resolved,
            "avg_sharpe": avg_sharpe,
            "average_sharpe": avg_sharpe,      # backward compat
            "max_drawdown": max_dd,
            "split_results": {
                "equities": split_equities,
                "drawdowns": split_drawdowns,
                "sharpes": split_sharpes
            }
        }

    def validate_promotion_candidate(
        self,
        weights_path: str,
        model_type: str,
        min_sharpe_override: float = 1.0,
        max_drawdown_override: float = 0.15,
        symbol: str = "BTC-USDT",
        max_staleness_days: Optional[int] = None,
    ) -> dict:
        """
        Strict promotion gate for Crucible->Deployment transition.
        Candidate must clear: Avg Sharpe >= 1.0, Max DD <= 15%, AND the training data
        this evaluation actually ran against must not be stale (see
        check_data_freshness()) -- a promotion decision made on data that's stopped
        updating is the same failure mode as the force-promoted, statistically-
        rejected checkpoint this project's own history already found sitting in
        state/deployed/ once, just quieter: everything downstream (Sharpe, drawdown)
        can look fine while measuring a market that no longer exists.

        max_staleness_days=None (the default) reads
        DEPLOY_CONFIG.max_training_data_staleness_days at call time rather than at
        import time, so changing the config value doesn't require re-importing this
        module.
        """
        if max_staleness_days is None:
            from scripts.deploy_config import DEPLOY_CONFIG

            max_staleness_days = getattr(DEPLOY_CONFIG, "max_training_data_staleness_days", 3)

        is_fresh, latest_date, days_stale = check_data_freshness(symbol, self.data_store_dir, max_staleness_days)
        if not is_fresh:
            if latest_date is None:
                reason = f"no processed physics data found for {symbol} in {self.data_store_dir}"
            else:
                reason = f"newest available data is from {latest_date} ({days_stale} day(s) old, limit {max_staleness_days})"
            logger.warning(
                f"[PROMOTION GATE BLOCKED — DATA STALE] {weights_path}: {reason}. "
                f"Refusing to run the (expensive) airgap simulation against data this stale — "
                f"fix the data pipeline (see training/continuous_trainer.py's _bridge_live_data()) "
                f"before re-attempting promotion."
            )
            return {
                "passed": False,
                "weights_path": weights_path,
                "model_type": self._resolve_model_name(model_type),
                "avg_sharpe": None,
                "average_sharpe": None,
                "max_drawdown": None,
                "reason": "data_stale",
                "data_staleness_detail": reason,
                "split_results": None,
            }

        resolved = self._resolve_model_name(model_type)
        original_sharpe = self.min_sharpe
        original_dd = self.max_drawdown
        self.min_sharpe = min_sharpe_override
        self.max_drawdown = max_drawdown_override

        logger.info(
            f"[PROMOTION GATE] Evaluating {weights_path} ({resolved}). "
            f"Requirements: Avg Sharpe >= {min_sharpe_override}, Max DD <= {max_drawdown_override*100:.0f}%."
        )

        try:
            result = self.evaluate_candidate_weights(weights_path, model_type=model_type, symbol=symbol)
        finally:
            self.min_sharpe = original_sharpe
            self.max_drawdown = original_dd

        if result["passed"]:
            logger.info(
                f"[PROMOTION GATE PASSED] Candidate {weights_path} is cleared for deployment. "
                f"Avg Sharpe: {result['avg_sharpe']:.3f}, Max DD: {result['max_drawdown']*100:.1f}%."
            )
        else:
            logger.warning(
                f"[PROMOTION GATE BLOCKED] Candidate rejected. "
                f"Avg Sharpe: {result['avg_sharpe']:.3f} (required >= {min_sharpe_override}), "
                f"Max DD: {result['max_drawdown']*100:.1f}% (required <= {max_drawdown_override*100:.0f}%)."
            )
        return result


if __name__ == "__main__":
    logger.info("Testing ValidationAirgapEngine standalone...")
    from data_forge.parquet_ingest import ParquetIngestPipeline
    import asyncio
    ingest = ParquetIngestPipeline(data_store_dir="data_store")
    asyncio.run(ingest.generate_synthetic_crucible_data(symbol="BTC-USDT", num_days=1, ticks_per_day=150))

    airgap = ValidationAirgapEngine(num_splits=3, min_required_sharpe=0.0, max_allowed_drawdown=0.5)
    res = airgap.evaluate_candidate_weights("dummy_weights.pt", model_type="PPO-DilatedCNN")
    print("Airgap Evaluation Summary:", json.dumps(res, indent=2))
