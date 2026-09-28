"""Real, executable tests for execution/performance_tracker.py -- pure stdlib, no deps."""
import unittest

from execution.performance_tracker import PerformanceStreakTracker, PerformanceConfig


class TestLossStreakCaution(unittest.TestCase):
    def test_fresh_tracker_is_neutral(self):
        t = PerformanceStreakTracker()
        self.assertEqual(t.scalar(), 1.0)

    def test_below_caution_start_stays_neutral(self):
        t = PerformanceStreakTracker()
        t.record_trade_outcome(-5.0)  # 1 loss, caution_start default is 2
        self.assertEqual(t.current_streak, -1)
        self.assertEqual(t.scalar(), 1.0)

    def test_loss_streak_ramps_down_and_holds_at_floor(self):
        cfg = PerformanceConfig(loss_streak_caution_start=2, loss_streak_floor_at=6, loss_streak_min_scalar=0.3)
        t = PerformanceStreakTracker(cfg)
        for _ in range(6):
            t.record_trade_outcome(-1.0)
        self.assertEqual(t.current_streak, -6)
        self.assertAlmostEqual(t.scalar(), 0.3, places=6)

        for _ in range(4):  # well past the floor point
            t.record_trade_outcome(-1.0)
        self.assertAlmostEqual(t.scalar(), 0.3, places=6, msg="scalar must hold at the floor, not keep shrinking")

    def test_scalar_monotonically_decreases_through_the_ramp(self):
        t = PerformanceStreakTracker(PerformanceConfig(loss_streak_caution_start=2, loss_streak_floor_at=6))
        scalars = []
        for _ in range(6):
            t.record_trade_outcome(-1.0)
            scalars.append(t.scalar())
        for earlier, later in zip(scalars, scalars[1:]):
            self.assertLessEqual(later, earlier)


class TestWinStreakOffByDefault(unittest.TestCase):
    def test_win_streak_does_not_boost_by_default(self):
        t = PerformanceStreakTracker()
        for _ in range(10):
            t.record_trade_outcome(1.0)
        self.assertEqual(t.current_streak, 10)
        self.assertEqual(t.scalar(), 1.0, "win-streak boost must be off unless explicitly enabled")

    def test_win_streak_boost_when_explicitly_enabled(self):
        cfg = PerformanceConfig(win_boost_ceiling=1.3, win_streak_boost_start=3, win_streak_ceiling_at=8)
        t = PerformanceStreakTracker(cfg)
        for _ in range(8):
            t.record_trade_outcome(1.0)
        self.assertAlmostEqual(t.scalar(), 1.3, places=6)


class TestStreakResets(unittest.TestCase):
    def test_scratch_trade_resets_streak_to_zero(self):
        t = PerformanceStreakTracker()
        t.record_trade_outcome(1.0)
        t.record_trade_outcome(1.0)
        self.assertEqual(t.current_streak, 2)
        t.record_trade_outcome(0.0)
        self.assertEqual(t.current_streak, 0)

    def test_sign_flip_restarts_streak_at_one_not_accumulates(self):
        t = PerformanceStreakTracker()
        for _ in range(4):
            t.record_trade_outcome(-1.0)
        self.assertEqual(t.current_streak, -4)
        t.record_trade_outcome(1.0)
        self.assertEqual(t.current_streak, 1, "a single win after a loss streak must restart at +1, not partially cancel")

    def test_manual_reset_clears_streak_and_history(self):
        t = PerformanceStreakTracker()
        for _ in range(5):
            t.record_trade_outcome(-1.0)
        t.reset()
        self.assertEqual(t.current_streak, 0)
        self.assertEqual(t.scalar(), 1.0)
        self.assertEqual(len(t._history), 0)


if __name__ == "__main__":
    unittest.main()
