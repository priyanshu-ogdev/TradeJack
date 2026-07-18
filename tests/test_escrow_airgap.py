"""
Verification Test 5: P2P Escrow Bridge & 10x Validation Airgap (`test_escrow_airgap.py`).
Tests trustless USDC cash locking, 10x out-of-sample stress validation across historical splits,
and exact settlement or refund logic.
"""

import os
import sys
import unittest
import shutil
import asyncio

from escrow.escrow_contract import P2PEscrowBridge
from escrow.validation_airgap import ValidationAirgapEngine
from data_forge.parquet_ingest import ParquetIngestPipeline


class TestEscrowAndAirgap(unittest.TestCase):

    def setUp(self):
        self.test_state = os.path.abspath("d:/TradeJack/state_test_escrow")
        self.test_store = os.path.abspath("d:/TradeJack/data_store_test_escrow")
        os.makedirs(self.test_state, exist_ok=True)
        os.makedirs(self.test_store, exist_ok=True)
        
        ingest = ParquetIngestPipeline(data_store_dir=self.test_store)
        asyncio.run(ingest.generate_synthetic_crucible_data(symbol="BTC-USDT", num_days=1, ticks_per_day=50))

    def tearDown(self):
        if os.path.exists(self.test_state):
            shutil.rmtree(self.test_state, ignore_errors=True)
        if os.path.exists(self.test_store):
            shutil.rmtree(self.test_store, ignore_errors=True)

    def test_escrow_initiation_and_locking(self):
        bridge = P2PEscrowBridge(state_dir=self.test_state, data_store_dir=self.test_store)
        # Verify initial balances
        self.assertEqual(bridge.get_balance(10), 10.0)
        self.assertEqual(bridge.get_balance(20), 10.0)
        
        # Initiate escrow locking $3.00 from Child 10 to Child 20
        init_res = bridge.initiate_escrow(buyer_id=10, seller_id=20, weights_path="dummy_weights.pt", price_usdc=3.00)
        self.assertEqual(init_res["status"], "LOCKED")
        self.assertIn("tx_id", init_res)
        
        # Check buyer balance reduced by $3.00 while locked
        self.assertEqual(bridge.get_balance(10), 7.00)
        # Check seller balance untouched until validation pass
        self.assertEqual(bridge.get_balance(20), 10.00)

    def test_validation_airgap_and_refund_on_rejection(self):
        bridge = P2PEscrowBridge(state_dir=self.test_state, data_store_dir=self.test_store)
        init_res = bridge.initiate_escrow(buyer_id=11, seller_id=21, weights_path="dummy.pt", price_usdc=2.00)
        tx_id = init_res["tx_id"]
        
        # Settle (with default min_sharpe=1.0, un-calibrated dummy weights will fail airgap and get rejected)
        settle_res = bridge.settle_or_refund(tx_id)
        # Should be REFUNDED because airgap rejected weak/overfitted dummy weights
        self.assertEqual(settle_res["status"], "REFUNDED")
        self.assertFalse(settle_res["airgap_report"]["passed"])
        
        # Verify buyer balance fully refunded back to $10.00
        self.assertEqual(bridge.get_balance(11), 10.00)
        self.assertEqual(bridge.get_balance(21), 10.00)

    def test_airgap_validation_engine_splits(self):
        airgap = ValidationAirgapEngine(num_splits=3, min_required_sharpe=0.0, max_allowed_drawdown=0.5, data_store_dir=self.test_store)
        report = airgap.evaluate_candidate_weights("dummy.pt", model_type="Dilated-CNN-Seq2seq", symbol="BTC-USDT")
        self.assertIn("passed", report)
        self.assertIn("split_results", report)
        self.assertEqual(len(report["split_results"]["equities"]), 3)

    def test_signed_social_gossip_and_escrow_bridge(self):
        from swarm.social_relay import SocialRelayBridge, verify_message
        relay_seller = SocialRelayBridge(child_id=30, state_dir=self.test_state)
        relay_seller.escrow_bridge = P2PEscrowBridge(state_dir=self.test_state, data_store_dir=self.test_store)
        
        # Broadcast signed insight
        lineage_id = relay_seller.broadcast_market_insight(equity=15.0, sharpe_ratio=2.5, state_dict_path="dummy.pt")
        
        # Verify packet stored and valid
        relay_buyer = SocialRelayBridge(child_id=31, state_dir=self.test_state)
        relay_buyer.escrow_bridge = relay_seller.escrow_bridge
        peers = relay_buyer.query_top_peers(min_sharpe=1.0)
        self.assertGreaterEqual(len(peers), 1)
        self.assertTrue(verify_message(peers[0]))
        
        # Test direct escrow bridge settlement/refund via social relay
        res = relay_buyer.request_peer_weights_via_escrow(lineage_id, offered_usdc=1.0)
        self.assertIn("status", res)
        self.assertIn(res["status"], ["SETTLED", "REFUNDED"])


if __name__ == "__main__":
    unittest.main()
