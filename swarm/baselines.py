"""
Non-RL baseline models for promotion hurdle testing.

Every RL model must beat BOTH of these baselines after fees before
it's even considered for promotion to live trading.

1. BuyAndHoldBaseline — buy $equity of BTC at t=0, hold forever
2. MomentumBaseline — simple dual-MA crossover signal

These are not "models" in the RL sense — they produce deterministic
actions from fixed rules. They exist to answer: "is the RL model
actually finding alpha, or is it just tracking the underlying asset?"
"""

import logging
import numpy as np
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (Baselines) %(message)s")
logger = logging.getLogger("Baselines")


class BuyAndHoldBaseline:
    """
    Buy max position at t=0, hold forever. Fee-adjusted.

    This is the simplest possible "strategy" — if an RL model can't
    beat this after fees, it has found no edge.
    """

    model_name = "BuyAndHold-Baseline"

    def __init__(self, target_fraction: float = 1.0):
        self.target_fraction = target_fraction
        self.has_entered = False

    def predict(self, obs: Dict[str, np.ndarray]) -> float:
        """Returns target position fraction. Always full long after first step."""
        if not self.has_entered:
            self.has_entered = True
            return self.target_fraction
        return self.target_fraction  # Hold

    def reset(self):
        self.has_entered = False


class MomentumBaseline:
    """
    Dual moving-average crossover momentum strategy.

    Goes long when short MA > long MA, flat otherwise.
    Uses the close price column (index 0) from the LOB sequence.

    This is the "smart naive" baseline — a well-known signal that
    works in trending markets. If the RL model can't beat this,
    there's no justification for the complexity.
    """

    model_name = "Momentum-Baseline"

    def __init__(self, short_window: int = 5, long_window: int = 20, scale: float = 50.0):
        self.short_window = short_window
        self.long_window = long_window
        self.scale = scale

    def predict(self, obs: Dict[str, np.ndarray]) -> float:
        """
        Returns target position fraction in [-1, 1].

        Uses the raw close price column from the LOB observation.
        Positive = long, negative = short, near-zero = flat.
        """
        lob_seq = obs.get("lob_sequence")
        if lob_seq is None or len(lob_seq) < self.long_window:
            return 0.0

        # Close price is column 0 in the (seq_len, 5) observation
        prices = lob_seq[:, 0]

        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])

        if abs(long_ma) < 1e-8:
            return 0.0

        diff = (short_ma - long_ma) / abs(long_ma)
        return float(np.clip(diff * self.scale, -1.0, 1.0))

    def reset(self):
        pass


class MeanReversionBaseline:
    """
    Simple mean-reversion baseline — contrarian to momentum.

    Goes short when price is above rolling mean, long when below.
    Useful as a second hurdle to ensure the RL model isn't just
    accidentally aligned with one regime type.
    """

    model_name = "MeanReversion-Baseline"

    def __init__(self, window: int = 20, z_threshold: float = 1.5, scale: float = 30.0):
        self.window = window
        self.z_threshold = z_threshold
        self.scale = scale

    def predict(self, obs: Dict[str, np.ndarray]) -> float:
        lob_seq = obs.get("lob_sequence")
        if lob_seq is None or len(lob_seq) < self.window:
            return 0.0

        prices = lob_seq[:, 0]
        rolling_mean = np.mean(prices[-self.window:])
        rolling_std = np.std(prices[-self.window:])

        if rolling_std < 1e-8:
            return 0.0

        z = (prices[-1] - rolling_mean) / rolling_std

        # Contrarian: go short when overbought, long when oversold
        signal = -z / self.z_threshold
        return float(np.clip(signal, -1.0, 1.0))

    def reset(self):
        pass


def evaluate_baseline_on_env(baseline, env, max_steps: int = 1000) -> Dict[str, float]:
    """
    Run a baseline model through a TradeJackLOBEnv and return performance metrics.

    Returns:
        dict with keys: final_equity, max_drawdown, total_return, trade_count
    """
    obs, info = env.reset()
    baseline.reset()

    total_trades = 0
    prev_action = 0.0

    for step in range(max_steps):
        action = baseline.predict(obs)

        # Count trades (position changes)
        if abs(action - prev_action) > 0.05:
            total_trades += 1
        prev_action = action

        obs, reward, terminated, truncated, info = env.step([action])
        if terminated or truncated:
            break

    return {
        "model_name": baseline.model_name,
        "final_equity": info.get("equity", 0.0),
        "peak_equity": info.get("peak_equity", 0.0),
        "max_drawdown": info.get("max_drawdown", 0.0),
        "trade_count": total_trades,
        "steps": step + 1,
    }


if __name__ == "__main__":
    logger.info("Testing baselines with synthetic observations...")

    # Simulate observations
    fake_obs = {
        "lob_sequence": np.random.randn(64, 5).astype(np.float32),
        "portfolio_state": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }

    bh = BuyAndHoldBaseline()
    logger.info(f"BuyAndHold action: {bh.predict(fake_obs)}")

    mom = MomentumBaseline()
    logger.info(f"Momentum action: {mom.predict(fake_obs)}")

    mr = MeanReversionBaseline()
    logger.info(f"MeanReversion action: {mr.predict(fake_obs)}")

    logger.info("Baseline tests passed.")
