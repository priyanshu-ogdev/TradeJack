"""
Verification Test 7 (v3): Execution Pipeline.
Tests paper exchange, risk guardian, position throttle, and exchange adapter integration.
"""

import os
import sys
import unittest
import asyncio
import shutil
import numpy as np


class TestPaperExchange(unittest.TestCase):
    """Test the simulated paper exchange."""

    def test_buy_and_sell_round_trip(self):
        """Paper exchange processes orders with slippage and fees."""
        from execution.paper_exchange import PaperExchangeAdapter

        async def _run():
            paper = PaperExchangeAdapter(initial_balance_usdt=100.0)
            await paper.connect()

            # Initial state
            self.assertTrue(paper.is_connected())
            self.assertEqual(await paper.get_balance("USDT"), 100.0)
            self.assertEqual(await paper.get_balance("BTC"), 0.0)

            # Buy
            r = await paper.place_market_order("BTC/USDT", "buy", 0.001)
            self.assertEqual(r.status, "filled")
            self.assertGreater(r.avg_price, 0)
            self.assertGreater(r.fee, 0)

            # BTC balance increased
            btc_bal = await paper.get_balance("BTC")
            self.assertAlmostEqual(btc_bal, 0.001, places=6)

            # Sell
            r = await paper.place_market_order("BTC/USDT", "sell", 0.001)
            self.assertEqual(r.status, "filled")

            # Equity decreased by fees + spread
            equity = paper.get_equity()
            self.assertLess(equity, 100.0)  # Lost to fees
            self.assertGreater(equity, 99.0)  # But not catastrophically

            summary = paper.get_trade_summary()
            self.assertEqual(summary["total_trades"], 2)
            self.assertEqual(summary["filled_buys"], 1)
            self.assertEqual(summary["filled_sells"], 1)

            await paper.close()

        asyncio.run(_run())

    def test_insufficient_balance_rejection(self):
        """Orders exceeding balance are rejected."""
        from execution.paper_exchange import PaperExchangeAdapter

        async def _run():
            paper = PaperExchangeAdapter(initial_balance_usdt=10.0)
            await paper.connect()

            # Try to buy more BTC than we can afford
            r = await paper.place_market_order("BTC/USDT", "buy", 1.0)  # ~$60K
            self.assertEqual(r.status, "rejected")
            self.assertEqual(r.qty, 0.0)

            # Balance unchanged
            self.assertEqual(await paper.get_balance("USDT"), 10.0)
            await paper.close()

        asyncio.run(_run())

    def test_price_feed_injection(self):
        """External price feed injection works."""
        from execution.paper_exchange import PaperExchangeAdapter

        async def _run():
            paper = PaperExchangeAdapter(initial_balance_usdt=100.0)
            await paper.connect()
            paper.set_price("BTC/USDT", 50000.0)
            price = await paper.get_ticker("BTC/USDT")
            self.assertEqual(price, 50000.0)
            await paper.close()

        asyncio.run(_run())


class TestRiskGuardian(unittest.TestCase):
    """Test the immutable safety layer."""

    def _make_guardian(self, starting_equity=100.0):
        from execution.paper_exchange import PaperExchangeAdapter
        from execution.risk_guardian import RiskGuardian

        exchange = PaperExchangeAdapter(initial_balance_usdt=starting_equity)
        # Sync connect
        asyncio.run(exchange.connect())
        guardian = RiskGuardian(
            exchange=exchange,
            symbol="BTC/USDT",
            max_position_fraction=0.5,
            max_daily_loss_pct=0.05,
            max_drawdown_halt=0.15,
            min_hold_ticks=10,
            max_orders_per_minute=3,
            starting_equity=starting_equity,
        )
        return guardian, exchange

    def test_position_size_limit(self):
        """Orders exceeding position limit are blocked."""
        guardian, _ = self._make_guardian(starting_equity=100.0)
        guardian.update_equity(100.0)

        # Try to place an order worth $60 (60% of equity, limit is 50%)
        allowed, reason = guardian.check_order_allowed("buy", 0.001, 60000.0)
        # 0.001 * 60000 = $60, max is $50 (50% of $100)
        self.assertFalse(allowed)
        self.assertIn("Position size", reason)

    def test_order_within_limits_passes(self):
        """Orders within limits are allowed."""
        guardian, _ = self._make_guardian(starting_equity=100.0)
        guardian.update_equity(100.0)

        # $30 order (30% of equity, limit is 50%) — should pass
        allowed, reason = guardian.check_order_allowed("buy", 0.0005, 60000.0)
        self.assertTrue(allowed)
        self.assertEqual(reason, "PASSED")

    def test_daily_loss_halt(self):
        """Daily loss exceeding limit triggers halt."""
        guardian, _ = self._make_guardian(starting_equity=100.0)
        guardian.update_equity(94.0)  # 6% daily loss (limit 5%)

        allowed, reason = guardian.check_order_allowed("buy", 0.0001, 60000.0)
        self.assertFalse(allowed)
        self.assertTrue(guardian.state.is_halted)

    def test_rate_limiting(self):
        """Rate limit blocks excessive orders."""
        guardian, _ = self._make_guardian(starting_equity=100.0)
        guardian.update_equity(100.0)
        guardian._current_tick = 100  # Advance past min_hold

        for i in range(3):
            guardian.record_order_executed("buy", 0.0001)

        allowed, reason = guardian.check_order_allowed("buy", 0.0001, 60000.0)
        self.assertFalse(allowed)
        self.assertIn("Rate limit", reason)

    def test_risk_summary(self):
        """Risk summary returns correct fields."""
        guardian, _ = self._make_guardian(starting_equity=100.0)
        guardian.update_equity(100.0)
        summary = guardian.get_risk_summary()
        self.assertIn("equity", summary)
        self.assertIn("is_halted", summary)
        self.assertIn("exchange_connected", summary)
        self.assertEqual(summary["equity"], 100.0)


class TestPositionThrottle(unittest.TestCase):
    """Test rolling-Sortino position sizing."""

    def test_profitable_period_full_fraction(self):
        """Good performance → full position size."""
        from execution.position_throttle import PositionThrottle
        throttle = PositionThrottle(base_fraction=0.5, sortino_full=1.0)

        # Simulate profitable returns
        for _ in range(100):
            throttle.record_return(np.random.normal(0.002, 0.005))

        frac = throttle.get_throttled_fraction()
        # Should be close to full fraction
        self.assertGreater(frac, 0.3)

    def test_losing_period_reduced_fraction(self):
        """Poor performance → reduced position size."""
        from execution.position_throttle import PositionThrottle
        throttle = PositionThrottle(base_fraction=0.5, min_fraction=0.05)

        # Simulate losing returns
        for _ in range(200):
            throttle.record_return(np.random.normal(-0.003, 0.01))

        frac = throttle.get_throttled_fraction()
        status = throttle.get_status()
        # Should be throttled toward min
        self.assertLess(frac, 0.3)
        self.assertIn(status["mode"], ["THROTTLED", "SURVIVAL"])

    def test_status_fields(self):
        """Status returns all expected fields."""
        from execution.position_throttle import PositionThrottle
        throttle = PositionThrottle()
        status = throttle.get_status()
        self.assertIn("rolling_sortino", status)
        self.assertIn("throttled_fraction", status)
        self.assertIn("mode", status)


class TestExchangeAdapterInterface(unittest.TestCase):
    """Test the abstract adapter interface compliance."""

    def test_paper_implements_interface(self):
        """PaperExchangeAdapter implements all ExchangeAdapter methods."""
        from execution.exchange_adapter import ExchangeAdapter
        from execution.paper_exchange import PaperExchangeAdapter
        self.assertTrue(issubclass(PaperExchangeAdapter, ExchangeAdapter))

        # Verify all abstract methods are implemented
        paper = PaperExchangeAdapter()
        required_methods = [
            "connect", "close", "get_ticker", "get_orderbook",
            "get_balance", "get_all_balances", "place_market_order",
            "get_open_orders", "cancel_all_orders", "get_position", "is_connected"
        ]
        for method_name in required_methods:
            self.assertTrue(hasattr(paper, method_name),
                            f"PaperExchangeAdapter missing method: {method_name}")


if __name__ == "__main__":
    unittest.main()
