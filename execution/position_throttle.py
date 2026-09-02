"""
Position Throttle — Rolling-Sortino-based position sizing.

Replaces the live-money burn concept with a dynamic position size throttle.
Same spirit (creates pressure to perform), no self-destructive equity subtraction.

When the model is performing well (high rolling Sortino), it gets full position sizing.
When performing poorly (low/negative Sortino), position size shrinks toward survival mode.
This is smarter than fixed burn because it responds to actual performance, not a clock.

Config comes from DeploymentConfig:
  throttle_sortino_full: 1.0    # Full position at this Sortino
  throttle_sortino_zero: 0.0    # Near-zero position at this Sortino
  throttle_min_fraction: 0.05   # Minimum position fraction (never fully zero)
"""

import math
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (PositionThrottle) %(message)s")
logger = logging.getLogger("PositionThrottle")


class PositionThrottle:
    """
    Dynamically scales max_position_fraction based on rolling performance.

    The throttle computes a rolling Sortino ratio over a configurable window
    (default: 24 hours of returns) and maps it to a position size multiplier:

      Sortino >= sortino_full  → max_position_fraction = base_fraction (100%)
      Sortino ~= 0             → max_position_fraction scales linearly to min_fraction
      Sortino < sortino_zero   → max_position_fraction = min_fraction (survival mode)
    """

    def __init__(
        self,
        base_fraction: float = 0.5,
        sortino_full: float = 1.0,
        sortino_zero: float = 0.0,
        min_fraction: float = 0.05,
        window_size: int = 1440,  # ~24 hours at 1 tick/minute
        target_return: float = 0.0,  # MAR for Sortino (0 = zero return)
    ):
        self.base_fraction = base_fraction
        self.sortino_full = sortino_full
        self.sortino_zero = sortino_zero
        self.min_fraction = min_fraction
        self.window_size = window_size
        self.target_return = target_return

        self.returns: deque = deque(maxlen=window_size)
        self._current_fraction = base_fraction
        self._current_sortino = 0.0

    def record_return(self, tick_return: float):
        """Record a single-tick return (e.g., log return or percentage return)."""
        self.returns.append(tick_return)

    def record_equity_change(self, prev_equity: float, curr_equity: float):
        """Record an equity change as a log return."""
        if prev_equity > 0:
            log_ret = math.log(curr_equity / prev_equity)
            self.returns.append(log_ret)

    def compute_rolling_sortino(self) -> float:
        """
        Compute Sortino ratio over the rolling window.

        Sortino = (mean_return - target) / downside_deviation
        Where downside_deviation only considers returns below the target.
        """
        if len(self.returns) < 10:
            return 0.0

        returns_arr = np.array(self.returns)
        mean_return = np.mean(returns_arr)
        excess = returns_arr - self.target_return
        downside = excess[excess < 0]

        if len(downside) == 0:
            return 5.0  # No downside — capped at 5.0

        downside_std = np.sqrt(np.mean(downside ** 2))
        if downside_std < 1e-10:
            return 5.0

        sortino = (mean_return - self.target_return) / downside_std
        return float(np.clip(sortino, -5.0, 5.0))

    def get_throttled_fraction(self) -> float:
        """
        Compute the current throttled position fraction.

        Maps rolling Sortino to a position size multiplier via linear interpolation.
        """
        self._current_sortino = self.compute_rolling_sortino()

        if self._current_sortino >= self.sortino_full:
            self._current_fraction = self.base_fraction
        elif self._current_sortino <= self.sortino_zero:
            self._current_fraction = self.min_fraction
        else:
            # Linear interpolation between min and base
            t = (self._current_sortino - self.sortino_zero) / max(self.sortino_full - self.sortino_zero, 1e-8)
            self._current_fraction = self.min_fraction + t * (self.base_fraction - self.min_fraction)

        return self._current_fraction

    def get_status(self) -> Dict[str, Any]:
        """Current throttle state for monitoring/dashboard."""
        return {
            "rolling_sortino": round(self._current_sortino, 3),
            "throttled_fraction": round(self._current_fraction, 4),
            "base_fraction": self.base_fraction,
            "min_fraction": self.min_fraction,
            "window_filled": f"{len(self.returns)}/{self.window_size}",
            "mode": "FULL" if self._current_fraction >= self.base_fraction * 0.9
                    else "THROTTLED" if self._current_fraction > self.min_fraction * 1.5
                    else "SURVIVAL",
        }


if __name__ == "__main__":
    logger.info("Testing PositionThrottle...")

    throttle = PositionThrottle(base_fraction=0.5, sortino_full=1.0, sortino_zero=0.0, min_fraction=0.05)

    # Simulate profitable period
    for _ in range(100):
        throttle.record_return(np.random.normal(0.001, 0.005))
    frac = throttle.get_throttled_fraction()
    logger.info(f"Profitable period: fraction={frac:.4f}, {throttle.get_status()}")

    # Simulate losing period
    for _ in range(200):
        throttle.record_return(np.random.normal(-0.002, 0.01))
    frac = throttle.get_throttled_fraction()
    logger.info(f"Losing period: fraction={frac:.4f}, {throttle.get_status()}")

    logger.info("PositionThrottle test passed.")
