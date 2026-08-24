"""
Verification Test 1: Warden Hypervisor & Memory Controller (`test_warden_hardware.py`).
Tests MIG memory slicing, Logarithmic + Stagnation burn rate, OOM Watchdog penalty, and Unified Memory pinning.
"""

import os
import sys
import time
import unittest
import shutil
import sqlite3

from warden.warden_core import WardenHypervisor, ContainerLedgerSummary
from warden.unified_memory_swap import BlackwellUnifiedAllocator
from warden.oom_watchdog import RecklessnessWatchdog
from warden.lineage_vector_db import LineageVectorDB


class TestWardenHardwareAndTaxes(unittest.TestCase):

    def setUp(self):
        self.test_state = os.path.abspath("d:/TradeJack/state_test_warden")
        os.makedirs(self.test_state, exist_ok=True)
        self.warden = WardenHypervisor(swarm_size=5, state_dir=self.test_state)

    def tearDown(self):
        if os.path.exists(self.test_state):
            shutil.rmtree(self.test_state, ignore_errors=True)

    def test_mig_memory_slicing_and_tiers(self):
        self.warden.init_child_ledger(child_id=0, initial_cash=10.0)
        summary0 = self.warden.audit_child_ledger(child_id=0)
        tier0 = self.warden.assign_memory_tier(child_id=0, summary=summary0)
        self.assertEqual(tier0, 2)
        
        ledger1 = self.warden.init_child_ledger(child_id=1, initial_cash=60.0)
        conn = sqlite3.connect(ledger1)
        cursor = conn.cursor()
        # BUG-4 FIX: Use correct column 'lifetime_sharpe' (not 'sharpe_ratio' which doesn't exist in schema)
        cursor.execute("UPDATE portfolio_state SET cash=60.0, equity=60.0, lifetime_sharpe=3.0, max_drawdown=0.05 WHERE tick_id=0")
        conn.commit()
        conn.close()
        
        summary1 = self.warden.audit_child_ledger(child_id=1)
        tier1 = self.warden.assign_memory_tier(child_id=1, summary=summary1)
        self.assertEqual(tier1, 1)

    def test_logarithmic_stagnation_tax(self):
        ledger2 = self.warden.init_child_ledger(child_id=2, initial_cash=100.0)
        conn = sqlite3.connect(ledger2)
        cursor = conn.cursor()
        # BUG-4 FIX: Remove 'ticks_stagnant' (computed from timestamp gap, not stored).
        # Set stagnation via last_hwm_market_timestamp far in the past (>4hr gap).
        stagnant_ts = time.time() - 25200.0  # 7 hours ago => triggers stagnation penalty
        cursor.execute(
            "UPDATE portfolio_state SET cash=100.0, equity=100.0, ticks_active=50, "
            "last_hwm_market_timestamp=?, market_timestamp=? WHERE tick_id=0",
            (stagnant_ts, time.time())
        )
        conn.commit()
        conn.close()
        
        summary2 = self.warden.audit_child_ledger(child_id=2)
        tax = self.warden.apply_survival_tax(child_id=2, summary=summary2)
        # BUG-6 FIX verification: apply_survival_tax now returns total_tax (was None before)
        self.assertIsNotNone(tax)
        self.assertGreater(tax, 0.0)

    def test_oom_watchdog_penalty(self):
        self.warden.init_child_ledger(child_id=3, initial_cash=25.0)
        watchdog = RecklessnessWatchdog(state_dir=self.test_state)
        watchdog.apply_oom_penalty(child_id=3, container_name="tradejack_child_3", reason="CUDA_OOM")
        
        # Verify recklessness log was written
        self.assertTrue(os.path.exists(watchdog.recklessness_log_path))
        
        # BUG-2 FIX VERIFICATION: tax_assessments must have the penalty entry with correct schema.
        # Design is event-sourcing: the OOM penalty lands in tax_assessments (not directly in cash).
        # PortfolioAccountingEngine.record_step() reconciles the pending tax on next tick.
        # The old INSERT used wrong columns (timestamp, reason) -- was silently swallowed.
        # This test now verifies the INSERT actually succeeded with the correct schema.
        db_path = self.warden.get_ledger_path(child_id=3)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT child_id, tax_type, amount FROM tax_assessments WHERE tax_type=?",
            ("OOM_RECKLESSNESS_PENALTY",)
        )
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row, "OOM penalty INSERT failed -- BUG-2 not fixed")
        self.assertEqual(row[0], 3)
        self.assertEqual(row[1], "OOM_RECKLESSNESS_PENALTY")
        self.assertAlmostEqual(row[2], watchdog.penalty_dollars, places=1)

    def test_unified_memory_allocator(self):
        allocator = BlackwellUnifiedAllocator(vram_quota_gb=128.0)
        dummy_weights = {"layer.weight": [0.1, 0.2, -0.3], "layer.bias": [0.0]}
        bytes_pinned = allocator.pin_to_unified_memory("model_test", dummy_weights)
        self.assertIn("model_test", allocator.pinned_pool)

    def test_survival_mode_and_tier_transitions(self):
        self.warden.init_child_ledger(child_id=4, initial_cash=10.0)
        summary4 = self.warden.audit_child_ledger(child_id=4)
        
        # Test NORMAL and CRITICAL modes
        mode_normal = self.warden.check_survival_mode(child_id=4, summary=summary4)
        self.assertEqual(mode_normal, "NORMAL")
        
        summary4.equity = 2.0
        mode_critical = self.warden.check_survival_mode(child_id=4, summary=summary4)
        self.assertEqual(mode_critical, "CRITICAL")
        
        # Verify tier_transitions table populated when transition occurs
        self.warden.record_tier_transition(child_id=4, from_tier=2, to_tier=3, summary=summary4, reason="Low equity survival forced")
        db_path = self.warden.get_ledger_path(child_id=4)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT from_tier, to_tier, reason FROM tier_transitions ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 2)
        self.assertEqual(row[1], 3)
        self.assertIn("Low equity survival forced", row[2])


if __name__ == "__main__":
    unittest.main()
