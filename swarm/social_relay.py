"""
P2P Social Communication Relay (`SocialRelayBridge` adaptation for `automaton` `social/`).
Allows sovereign Child agents to broadcast profitable model lineages, query high-Sharpe peers,
and negotiate trustless `state_dict` exchanges via the Escrow smart contract and 10x Airgap.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (SocialRelay) %(message)s")
logger = logging.getLogger("SocialRelay")

from warden.lineage_vector_db import LineageVectorDB


class SocialRelayBridge:
    """
    Decentralized communication layer linking the 50 containerized agents.
    """

    def __init__(self, child_id: int = 0, state_dir: str = "d:/TradeJack/state"):
        self.child_id = child_id
        self.state_dir = os.path.abspath(state_dir)
        self.relay_dir = os.path.join(self.state_dir, "social_relay")
        os.makedirs(self.relay_dir, exist_ok=True)
        self.vector_db = LineageVectorDB(db_path=os.path.join(self.state_dir, "chroma_db"))

    def broadcast_market_insight(
        self,
        equity: float,
        sharpe_ratio: float,
        state_dict_path: str,
        description: str = "High-Sharpe Dilated CNN on BTC-USDT flash crash"
    ) -> str:
        """
        Registers current lineage performance and state_dict pointer into ChromaDB and local relay directory.
        """
        lineage_id = f"child{self.child_id}_{int(time.time())}"
        metadata = {
            "child_id": self.child_id,
            "equity": equity,
            "sharpe_ratio": sharpe_ratio,
            "state_dict_path": state_dict_path,
            "timestamp": time.time()
        }
        
        # Save to local relay JSON directory
        insight_path = os.path.join(self.relay_dir, f"{lineage_id}.json")
        with open(insight_path, "w", encoding="utf-8") as f:
            json.dump({"lineage_id": lineage_id, "metadata": metadata, "description": description}, f, indent=2)
            
        # Index in Vector DB
        self.vector_db.register_lineage(
            lineage_id=lineage_id,
            metadata=metadata,
            description=description
        )
        logger.info(f"[SOCIAL RELAY] Child {self.child_id} broadcasted insight '{lineage_id}' (Sharpe {sharpe_ratio:.2f}).")
        return lineage_id

    def query_top_peers(self, min_sharpe: float = 1.0, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Queries top performing peer models across the relay directory and vector DB.
        """
        results = []
        if not os.path.exists(self.relay_dir):
            return results
            
        for file in os.listdir(self.relay_dir):
            if file.endswith(".json"):
                try:
                    with open(os.path.join(self.relay_dir, file), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        meta = data.get("metadata", {})
                        if meta.get("child_id") != self.child_id and meta.get("sharpe_ratio", 0.0) >= min_sharpe:
                            results.append(data)
                except Exception:
                    continue
                    
        # Sort descending by Sharpe
        results.sort(key=lambda x: x["metadata"]["sharpe_ratio"], reverse=True)
        return results[:limit]

    def request_peer_weights_via_escrow(self, target_lineage_id: str, offered_usdc: float = 0.50) -> Dict[str, Any]:
        """
        Initiates a trustless atomic exchange via `escrow/escrow_contract.py`.
        """
        logger.info(f"Child {self.child_id} initiating escrow request for peer lineage '{target_lineage_id}' offering ${offered_usdc:.2f}.")
        # Will bridge to P2PEscrowBridge when escrow module is instantiated
        return {
            "status": "ESCROW_INITIATED",
            "target_lineage_id": target_lineage_id,
            "buyer_child_id": self.child_id,
            "offered_usdc": offered_usdc
        }


if __name__ == "__main__":
    logger.info("Testing SocialRelayBridge standalone...")
    relay = SocialRelayBridge(child_id=1)
    relay.broadcast_market_insight(equity=14.50, sharpe_ratio=2.3, state_dict_path="state/child_1/weights.pt")
    peers = relay.query_top_peers(min_sharpe=1.0)
    print("Found top peers:", json.dumps(peers, indent=2))
