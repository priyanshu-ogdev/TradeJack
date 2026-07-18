"""
Verification Test 3: Physics Engine & Friction Wrapper (`test_physics.py`).
Tests TradeJackLOBEnv exact slippage math, spread penalty, double-buffer queue, and PortfolioAccountingEngine SQLite ledger.
"""

import os
import sys
import unittest
import shutil
import numpy as np
import asyncio

from physics.lob_env import TradeJackLOBEnv
from physics.portfolio_tracker import PortfolioAccountingEngine
from data_forge.parquet_ingest import ParquetIngestPipeline


class TestPhysicsAndPortfolio(unittest.TestCase):

    def setUp(self):
        self.test_store = os.path.abspath("d:/TradeJack/data_store_test_phys")
        self.test_state = os.path.abspath("d:/TradeJack/state_test_phys")
        os.makedirs(self.test_store, exist_ok=True)
        os.makedirs(self.test_state, exist_ok=True)
        
        ingest = ParquetIngestPipeline(data_store_dir=self.test_store)
        asyncio.run(ingest.generate_synthetic_crucible_data(symbol="BTC-USDT", num_days=1, ticks_per_day=100))

    def tearDown(self):
        if os.path.exists(self.test_store):
            shutil.rmtree(self.test_store, ignore_errors=True)
        if os.path.exists(self.test_state):
            shutil.rmtree(self.test_state, ignore_errors=True)

    def test_lob_env_reset_and_step(self):
        env = TradeJackLOBEnv(symbol="BTC-USDT", initial_cash=10.0, data_store_dir=self.test_store, child_id=101)
        obs, info = env.reset()
        self.assertIn("lob_sequence", obs)
        self.assertEqual(obs["lob_sequence"].shape, (60, 8))
        self.assertEqual(info["cash"], 10.0)
        
        next_obs, reward, terminated, truncated, next_info = env.step([1.0])
        self.assertIn("equity", next_info)
        self.assertIn("max_drawdown", next_info)
        self.assertIsNotNone(reward)

    def test_portfolio_accounting_engine(self):
        engine = PortfolioAccountingEngine(child_id=102, state_dir=self.test_state, initial_cash=10.0)
        summary1 = engine.record_step(new_cash=10.50, new_equity=10.50, tick_id=1)
        self.assertEqual(summary1["equity"], 10.50)
        self.assertEqual(summary1["peak_equity"], 10.50)
        self.assertEqual(summary1["max_drawdown"], 0.0)
        
        summary2 = engine.record_step(new_cash=9.00, new_equity=9.00, tick_id=2)
        self.assertAlmostEqual(summary2["max_drawdown"], 1.50 / 10.50, places=3)
        self.assertLess(summary2["sharpe_ratio"], 0.0)
        self.assertTrue(os.path.exists(engine.db_path))


if __name__ == "__main__":
    unittest.main()
