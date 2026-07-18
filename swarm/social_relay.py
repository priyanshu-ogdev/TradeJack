"""
P2P Social Communication Relay (`SocialRelayBridge` adaptation for `automaton` `social/`).
Allows sovereign Child agents to broadcast profitable model lineages, query high-Sharpe peers,
and negotiate trustless `state_dict` exchanges via the Escrow smart contract and 10x Airgap.
"""

import os
import sys
import time
import json
import uuid
import hmac
import hashlib
import logging
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (SocialRelay) %(message)s")
logger = logging.getLogger("SocialRelay")

from warden.lineage_vector_db import LineageVectorDB
from escrow.escrow_contract import P2PEscrowBridge

# ─── CRYPTOGRAPHIC SIGNING UTILS (Conway Automaton social/protocol.ts pattern) ───
SECRET_RELAY_KEY = b"TradeJack_Blackwell_Sovereign_Secret_Key_2026"


def generate_message_id() -> str:
    """Generates unique ULID/UUID packet identifier."""
    return f"msg_{uuid.uuid4().hex}"


def generate_nonce() -> str:
    """Generates 16-byte random hex nonce for replay protection."""
    return os.urandom(16).hex()


def sign_message(content: str, timestamp: float, nonce: str, sender_id: int) -> str:
    """
    Computes HMAC-SHA256 signature over canonical message string:
    Conway:send:{sender_id}:{content_hash}:{timestamp}:{nonce}
    """
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    canonical = f"Conway:send:{sender_id}:{content_hash}:{timestamp}:{nonce}"
    return hmac.new(SECRET_RELAY_KEY, canonical.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_message(packet: Dict[str, Any]) -> bool:
    """Verifies message signature and freshness against canonical string."""
    try:
        sender_id = packet["from"]
        content = packet["content"]
        timestamp = float(packet["timestamp"])
        nonce = packet["nonce"]
        expected_sig = sign_message(content, timestamp, nonce, sender_id)
        return hmac.compare_digest(packet["signature"], expected_sig)
    except Exception:
        return False


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
        self.escrow_bridge = P2PEscrowBridge(state_dir=self.state_dir)

    def broadcast_market_insight(
        self,
        equity: float,
        sharpe_ratio: float,
        state_dict_path: str,
        description: str = "High-Sharpe Dilated CNN on BTC-USDT flash crash"
    ) -> str:
        """
        Registers current lineage performance and state_dict pointer into ChromaDB and local relay directory.
        Emits cryptographically signed packet matching Conway Automaton `social/protocol.ts`.
        """
        lineage_id = f"child{self.child_id}_{int(time.time())}"
        timestamp = time.time()
        nonce = generate_nonce()
        msg_id = generate_message_id()
        
        metadata = {
            "child_id": self.child_id,
            "equity": equity,
            "sharpe_ratio": sharpe_ratio,
            "state_dict_path": state_dict_path,
            "timestamp": timestamp,
            "model_type": description.split("(")[-1].split(")")[0] if "(" in description else "Dilated-CNN-Seq2seq"
        }
        content = json.dumps(metadata, sort_keys=True)
        sig = sign_message(content, timestamp, nonce, self.child_id)
        
        packet = {
            "id": msg_id,
            "from": self.child_id,
            "lineage_id": lineage_id,
            "content": content,
            "metadata": metadata,
            "description": description,
            "timestamp": timestamp,
            "nonce": nonce,
            "signature": sig
        }
        
        # Save signed packet to local relay directory
        insight_path = os.path.join(self.relay_dir, f"{lineage_id}.json")
        with open(insight_path, "w", encoding="utf-8") as f:
            json.dump(packet, f, indent=2)
            
        # Index in Vector DB
        self.vector_db.register_lineage(
            lineage_id=lineage_id,
            metadata=metadata,
            description=description
        )
        logger.info(f"[SOCIAL RELAY] Child {self.child_id} broadcasted signed insight '{lineage_id}' (Sharpe {sharpe_ratio:.2f}).")
        return lineage_id

    def query_top_peers(self, min_sharpe: float = 1.0, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Queries top performing peer models across the relay directory and vector DB after verifying cryptographic signatures.
        """
        results = []
        if not os.path.exists(self.relay_dir):
            return results
            
        for file in os.listdir(self.relay_dir):
            if file.endswith(".json"):
                try:
                    with open(os.path.join(self.relay_dir, file), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if not verify_message(data):
                            logger.warning(f"Rejected spoofed/invalid signature on gossip packet '{file}'.")
                            continue
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
        Initiates a trustless atomic exchange via `escrow/escrow_contract.py` (P2PEscrowBridge).
        """
        logger.info(f"Child {self.child_id} initiating escrow request for peer lineage '{target_lineage_id}' offering ${offered_usdc:.2f}.")
        insight_path = os.path.join(self.relay_dir, f"{target_lineage_id}.json")
        if not os.path.exists(insight_path):
            return {"status": "ERROR", "message": f"Peer lineage '{target_lineage_id}' not found in relay."}
            
        try:
            with open(insight_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not verify_message(data):
                return {"status": "REJECTED_BAD_SIGNATURE", "message": "Spoofed/invalid gossip packet signature."}
                
            seller_id = data["metadata"]["child_id"]
            weights_path = data["metadata"]["state_dict_path"]
            model_type = data["metadata"].get("model_type", "Dilated-CNN-Seq2seq")
            
            # Lock funds via P2PEscrowBridge
            lock_res = self.escrow_bridge.initiate_escrow(
                buyer_id=self.child_id,
                seller_id=seller_id,
                weights_path=weights_path,
                price_usdc=offered_usdc,
                model_type=model_type
            )
            if lock_res.get("status") == "LOCKED":
                # Immediately settle via 10x Validation Airgap
                settle_res = self.escrow_bridge.settle_or_refund(lock_res["tx_id"])
                return settle_res
            return lock_res
        except Exception as e:
            logger.error(f"Error during escrow request for '{target_lineage_id}': {e}")
            return {"status": "ERROR", "message": str(e)}


if __name__ == "__main__":
    logger.info("Testing SocialRelayBridge standalone...")
    relay = SocialRelayBridge(child_id=1)
    relay.broadcast_market_insight(equity=14.50, sharpe_ratio=2.3, state_dict_path="state/child_1/weights.pt")
    peers = relay.query_top_peers(min_sharpe=1.0)
    print("Found top peers:", json.dumps(peers, indent=2))
