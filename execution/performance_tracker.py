"""
Performance Streak Tracker — the fifth composition axis.

Flagged explicitly as missing in the prior round rather than silently treated as
"composition_layer.py is now complete": a scalar based on REALIZED trade outcomes,
independent of the equity-based risk-budget throttle (composition_layer.py's axis 1).
The distinction matters: risk-budget throttle asks "how much of today's loss allowance
is gone" (an equity fact); this tracker asks "have my last few decisions actually been
right" (a hit-rate fact). A single large loss and five small losses can consume the same
risk budget but say very different things about whether the current signal is working.

DELIBERATELY ASYMMETRIC — read before changing the defaults:

  - LOSS-STREAK CAUTION is well-grounded and ON by default: after consecutive losing
    trades, reduce size. This is standard practom risk management (protect capital when
    something may be wrong with the current regime/signal) and is not contested.

  - WIN-STREAK "GREED" (sizing UP after consecutive wins) is NOT enabled by default
    (`win_boost_ceiling` defaults to 1.0, i.e. no boost). This is a genuinely contested
    idea in trading: it can be read as "momentum of skill" (the signal IS currently
    working, lean into it) OR as the gambler's-fallacy-in-reverse / hot-hand fallacy
    (a short win streak is weak evidence in a noisy market, and sizing up on it invites
    a much larger loss on the inevitable reversion). This module does not take a
    position on which is correct — it exposes `win_boost_ceiling` as an explicit,
    off-by-default opt-in, so enabling it is a deliberate choice made with this tension
    in view, not an accidental default.

The scalar this produces is meant to be passed as composition_layer.py's
`performance_scalar` — combined multiplicatively with the risk-budget and toxicity
throttles, same as those two.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (PerformanceTracker) %(message)s")
logger = logging.getLogger("PerformanceTracker")


@dataclass
class PerformanceConfig:
    loss_streak_caution_start: int = 2
    # Consecutive losing trades before the caution throttle starts ramping down.
    loss_streak_min_scalar: float = 0.3
    # Scalar floor reached at loss_streak_floor_at consecutive losses.
    loss_streak_floor_at: int = 6
    # Consecutive losses at which the scalar bottoms out at loss_streak_min_scalar
    # (ramps linearly between caution_start and this point, then holds at the floor
    # rather than continuing to shrink toward 0 -- the risk-budget throttle and
    # risk_guardian's hard halt are the backstops for "things have gone badly enough
    # to stop entirely"; this tracker's job is to lean cautious, not to be the veto).

    win_boost_ceiling: float = 1.0
    # See module docstring: OFF by default (1.0 = no boost). Set above 1.0 only as a
    # deliberate, informed choice.
    win_streak_boost_start: int = 3
    # Consecutive wins before the (opt-in) boost starts ramping up, if enabled.
    win_streak_ceiling_at: int = 8
    # Consecutive wins at which the boost reaches win_boost_ceiling (ramps linearly
    # between boost_start and this point).

    history_len: int = 50
    # How many recent trade outcomes to retain for reporting/inspection. The streak
    # counters themselves don't need the full history (they're running counts reset on
    # a sign flip), but keeping recent history makes debugging/inspection possible.


class PerformanceStreakTracker:
    """
    Tracks consecutive win/loss streaks from realized trade P&L and exposes a bounded
    multiplicative scalar for composition_layer.py's `performance_scalar` argument.
    Pure Python, no dependencies -- deliberately, so it's usable anywhere in the stack
    (backtest, paper, live) without a data_forge or sklearn dependency.
    """

    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self._current_streak: int = 0  # positive = consecutive wins, negative = consecutive losses
        self._history: Deque[float] = deque(maxlen=self.config.history_len)

    def record_trade_outcome(self, pnl: float):
        """Records one realized trade's P&L. pnl > 0 extends/starts a win streak;
        pnl < 0 extends/starts a loss streak; pnl == 0 (breakeven) resets the streak
        to 0 without counting as either -- a scratch trade is not evidence for or
        against the current signal."""
        self._history.append(pnl)
        if pnl > 0:
            self._current_streak = self._current_streak + 1 if self._current_streak >= 0 else 1
        elif pnl < 0:
            self._current_streak = self._current_streak - 1 if self._current_streak <= 0 else -1
        else:
            self._current_streak = 0

    def reset(self):
        """Clears the streak -- e.g. at the start of a new trading day, or after a
        manual risk_guardian.reset_halt(), so a streak from before a human-reviewed
        halt doesn't silently keep throttling (or boosting) size after conditions have
        genuinely changed."""
        self._current_streak = 0
        self._history.clear()

    @property
    def current_streak(self) -> int:
        return self._current_streak

    def scalar(self) -> float:
        """Returns the bounded multiplicative scalar for the current streak state."""
        streak = self._current_streak
        cfg = self.config

        if streak <= -cfg.loss_streak_caution_start:
            losses = -streak
            span = cfg.loss_streak_floor_at - cfg.loss_streak_caution_start
            if span <= 0:
                return cfg.loss_streak_min_scalar
            t = min(1.0, (losses - cfg.loss_streak_caution_start) / span)
            return 1.0 - t * (1.0 - cfg.loss_streak_min_scalar)

        if streak >= cfg.win_streak_boost_start and cfg.win_boost_ceiling > 1.0:
            wins = streak
            span = cfg.win_streak_ceiling_at - cfg.win_streak_boost_start
            if span <= 0:
                return cfg.win_boost_ceiling
            t = min(1.0, (wins - cfg.win_streak_boost_start) / span)
            return 1.0 + t * (cfg.win_boost_ceiling - 1.0)

        return 1.0
