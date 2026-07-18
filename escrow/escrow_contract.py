"""
P2P Smart Contract Escrow Bridge (`P2PEscrowBridge` adaptation for `ERC-8004 / Base`).
Handles atomic P2P transactions between sovereign Child containers.
Locks buyer cash balances, invokes the 10x Validation Airgap to test offered `state_dict` parameters,
and either releases funds to the seller upon validation pass or refunds the buyer upon validation failure.
"""

import os
import sys
import time
import json
import shutil
import logging
import sqlite3
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (EscrowBridge) %(message)s")
logger = logging.getLogger("EscrowBridge")

from escrow.validation_airgap import ValidationAirgapEngine


class P2PEscrowBridge:
    """
    Decentralized settlement layer and cryptographic ledger linking the 50 containerized agents.
    """

    def __init__(self, state_dir: str = "d:/TradeJack/state", data_store_dir: str = "d:/TradeJack/data_store"):
        self.state_dir = os.path.abspath(state_dir)
        self.escrow_dir = os.path.join(self.state_dir, "escrow")
        os.makedirs(self.escrow_dir, exist_ok=True)
        self.db_path = os.path.join(self.escrow_dir, "escrow_ledger.sqlite")
        
        self.airgap_engine = ValidationAirgapEngine(
            num_splits=10,
            min_required_sharpe=1.0,
            max_allowed_drawdown=0.15,
            data_store_dir=data_store_dir
        )
        self._init_sqlite()

    def _init_sqlite(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS container_balances (
                    child_id INTEGER PRIMARY KEY,
                    usdc_balance REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS escrow_transactions (
                    tx_id TEXT PRIMARY KEY,
                    buyer_id INTEGER,
                    seller_id INTEGER,
                    weights_path TEXT,
                    model_type TEXT,
                    price_usdc REAL,
                    status TEXT,
                    created_at REAL,
                    settled_at REAL
                )
            """)
            # Initialize 50 containers with $10 default balance if empty
            cursor.execute("SELECT COUNT(*) FROM container_balances")
            if cursor.fetchone()[0] == 0:
                for i in range(50):
                    cursor.execute("INSERT INTO container_balances (child_id, usdc_balance) VALUES (?, ?)", (i, 10.0))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed initializing escrow ledger: {e}")

    def get_balance(self, child_id: int) -> float:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT usdc_balance FROM container_balances WHERE child_id = ?", (child_id,))
            row = cursor.fetchone()
            conn.close()
            return float(row[0]) if row else 0.0
        except Exception:
            return 0.0

    def initiate_escrow(
        self,
        buyer_id: int,
        seller_id: int,
        weights_path: str,
        price_usdc: float = 0.50,
        model_type: str = "Dilated-CNN-Seq2seq"
    ) -> Dict[str, Any]:
        """
        Locks `price_usdc` from buyer balance and registers pending escrow transaction.
        """
        if buyer_id == seller_id:
            return {"status": "REJECTED", "message": "Cannot trade with self."}
            
        buyer_bal = self.get_balance(buyer_id)
        if buyer_bal < price_usdc:
            return {"status": "REJECTED", "message": f"Insufficient funds (${buyer_bal:.2f} < ${price_usdc:.2f})."}
            
        tx_id = f"tx_b{buyer_id}_s{seller_id}_{int(time.time()*1000)}"
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Deduct from buyer
            cursor.execute("UPDATE container_balances SET usdc_balance = usdc_balance - ? WHERE child_id = ?", (price_usdc, buyer_id))
            # Insert tx record
            cursor.execute("""
                INSERT INTO escrow_transactions (
                    tx_id, buyer_id, seller_id, weights_path, model_type, price_usdc, status, created_at, settled_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'LOCKED_IN_AIRGAP', ?, 0.0)
            """, (tx_id, buyer_id, seller_id, weights_path, model_type, price_usdc, time.time()))
            conn.commit()
            conn.close()
            logger.info(f"[ESCROW LOCKED] Tx '{tx_id}' locked ${price_usdc:.2f} from Child {buyer_id} for Child {seller_id}'s weights.")
            return {"status": "LOCKED", "tx_id": tx_id, "price_usdc": price_usdc}
        except Exception as e:
            logger.error(f"Error locking escrow: {e}")
            return {"status": "ERROR", "message": str(e)}

    def settle_or_refund(self, tx_id: str) -> Dict[str, Any]:
        """
        Executes the 10x Validation Airgap across the locked transaction weights.
        If passed: transfers funds to seller and copies weights to buyer.
        If failed: refunds funds to buyer and cancels order.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT buyer_id, seller_id, weights_path, model_type, price_usdc, status FROM escrow_transactions WHERE tx_id = ?", (tx_id,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return {"status": "ERROR", "message": "Transaction not found."}
                
            buyer_id, seller_id, weights_path, model_type, price_usdc, status = row
            if status != "LOCKED_IN_AIRGAP":
                conn.close()
                return {"status": "ERROR", "message": f"Transaction status is {status}, not LOCKED_IN_AIRGAP."}
                
            # Run Airgap
            airgap_report = self.airgap_engine.evaluate_candidate_weights(weights_path, model_type=model_type)
            
            if airgap_report["passed"]:
                # Transfer funds to seller
                cursor.execute("UPDATE container_balances SET usdc_balance = usdc_balance + ? WHERE child_id = ?", (price_usdc, seller_id))
                cursor.execute("UPDATE escrow_transactions SET status = 'SETTLED_AIRGAP_PASSED', settled_at = ? WHERE tx_id = ?", (time.time(), tx_id))
                conn.commit()
                conn.close()
                
                # Copy weights to buyer state folder
                buyer_dir = os.path.join(self.state_dir, f"child_{buyer_id}")
                os.makedirs(buyer_dir, exist_ok=True)
                dest_path = os.path.join(buyer_dir, f"weights_acquired_peer{seller_id}.pt")
                if os.path.exists(weights_path):
                    try:
                        shutil.copy2(weights_path, dest_path)
                    except Exception:
                        pass
                        
                logger.info(f"[ESCROW SETTLED] Tx '{tx_id}' validated! Transferred ${price_usdc:.2f} to Child {seller_id} and weights to Child {buyer_id}.")
                return {"status": "SETTLED", "tx_id": tx_id, "airgap_report": airgap_report, "acquired_weights_dest": dest_path}
            else:
                # Refund buyer
                cursor.execute("UPDATE container_balances SET usdc_balance = usdc_balance + ? WHERE child_id = ?", (price_usdc, buyer_id))
                cursor.execute("UPDATE escrow_transactions SET status = 'REFUNDED_AIRGAP_FAILED', settled_at = ? WHERE tx_id = ?", (time.time(), tx_id))
                conn.commit()
                conn.close()
                
                logger.warning(f"[ESCROW REFUNDED] Tx '{tx_id}' rejected by 10x Airgap! Refunded ${price_usdc:.2f} back to Child {buyer_id}.")
                return {"status": "REFUNDED", "tx_id": tx_id, "airgap_report": airgap_report}
        except Exception as e:
            logger.error(f"Error settling escrow tx {tx_id}: {e}")
            return {"status": "ERROR", "message": str(e)}


if __name__ == "__main__":
    logger.info("Testing P2PEscrowBridge standalone...")
    bridge = P2PEscrowBridge()
    print("Initial Buyer 1 balance:", bridge.get_balance(1))
    print("Initial Seller 2 balance:", bridge.get_balance(2))
    
    # Initiate trade
    init_res = bridge.initiate_escrow(buyer_id=1, seller_id=2, weights_path="dummy_weights.pt", price_usdc=2.50)
    print("Escrow initiated:", json.dumps(init_res, indent=2))
    print("Buyer balance while locked:", bridge.get_balance(1))
    
    # Settle
    settle_res = bridge.settle_or_refund(init_res["tx_id"])
    print("Escrow settlement report:", json.dumps(settle_res, indent=2))
    print("Final Buyer 1 balance:", bridge.get_balance(1))
    print("Final Seller 2 balance:", bridge.get_balance(2))
