"""
Lineage Vector Database for Project TradeJack.
Stores PyTorch state_dict pointers and market regime vector embeddings in ChromaDB (or high-performance local cosine index).
Enables Child agents to query P2P social relay: 'Give me Gen-X weights that survived the BTC liquidity vacuum.'
"""

import os
import sys
import time
import json
import math
import logging
from typing import Dict, Any, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (LineageVectorDB) %(message)s")
logger = logging.getLogger("LineageVectorDB")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not installed; LineageVectorDB will use local ZSTD/SQLite Cosine Index fallback.")


class LineageVectorDB:
    """
    Tracks evolutionary lineage across all 50 sovereign Child containers.
    """

    def __init__(self, db_path: str = "d:/TradeJack/state/lineage_db"):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(self.db_path, exist_ok=True)
        self.fallback_file = os.path.join(self.db_path, "lineage_records.json")
        self.collection = None
        self.local_records: List[Dict[str, Any]] = []
        
        self.chroma_available = CHROMA_AVAILABLE
        if self.chroma_available:
            try:
                self.client = chromadb.PersistentClient(path=self.db_path)
                self.collection = self.client.get_or_create_collection("tradejack_lineage")
                logger.info("ChromaDB PersistentClient initialized.")
            except Exception as e:
                logger.warning(f"Could not initialize ChromaDB client ({e}). Using local fallback.")
                self.chroma_available = False
                
        if not self.chroma_available:
            if os.path.exists(self.fallback_file):
                try:
                    with open(self.fallback_file, "r", encoding="utf-8") as f:
                        self.local_records = json.load(f)
                except Exception:
                    self.local_records = []

    def insert_lineage_record(
        self,
        child_id: int,
        generation: int,
        model_type: str,
        regime_vector: List[float],  # e.g., [volatility, spread, trend_slope, imbalance, volume_zscore]
        sharpe: float,
        state_dict_path: str,
        regime_description: str = "Standard Market Regime"
    ) -> str:
        """
        Records a surviving model weights generation and its exact market regime vector.
        """
        record_id = f"gen{generation}_child{child_id}_{int(time.time())}"
        metadata = {
            "child_id": child_id,
            "generation": generation,
            "model_type": model_type,
            "sharpe": float(sharpe),
            "state_dict_path": state_dict_path,
            "regime_description": regime_description,
            "timestamp": time.time()
        }
        
        if self.chroma_available and self.collection is not None:
            try:
                self.collection.add(
                    ids=[record_id],
                    embeddings=[regime_vector],
                    metadatas=[metadata],
                    documents=[regime_description]
                )
                logger.debug(f"Inserted lineage record '{record_id}' into ChromaDB.")
                return record_id
            except Exception as e:
                logger.error(f"Error inserting into ChromaDB: {e}")
                
        # Local Cosine Fallback
        record = {
            "record_id": record_id,
            "regime_vector": regime_vector,
            "metadata": metadata,
            "document": regime_description
        }
        self.local_records.append(record)
        try:
            with open(self.fallback_file, "w", encoding="utf-8") as f:
                json.dump(self.local_records, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save local lineage records: {e}")
            
        return record_id

    def query_surviving_weights(
        self,
        current_regime_vector: List[float],
        min_sharpe: float = 1.5,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Queries the lineage store for historical models that survived similar market regimes.
        Uses cosine similarity across the regime embedding vector.
        """
        results: List[Dict[str, Any]] = []
        
        if self.chroma_available and self.collection is not None:
            try:
                query_res = self.collection.query(
                    query_embeddings=[current_regime_vector],
                    n_results=top_k,
                    where={"sharpe": {"$gte": min_sharpe}}
                )
                if query_res and query_res["metadatas"] and len(query_res["metadatas"]) > 0:
                    for idx, meta in enumerate(query_res["metadatas"][0]):
                        distance = query_res["distances"][0][idx] if "distances" in query_res and query_res["distances"] else 0.0
                        results.append({
                            "metadata": meta,
                            "similarity_score": max(0.0, 1.0 - distance),
                            "record_id": query_res["ids"][0][idx]
                        })
                return results
            except Exception as e:
                logger.debug(f"Chroma query error: {e}. Falling back to local search.")
                
        # Local Cosine Similarity Calculation
        def cosine_sim(vec_a: List[float], vec_b: List[float]) -> float:
            if len(vec_a) != len(vec_b) or not vec_a:
                return 0.0
            dot = sum(a * b for a, b in zip(vec_a, vec_b))
            norm_a = math.sqrt(sum(a * a for a in vec_a))
            norm_b = math.sqrt(sum(b * b for b in vec_b))
            return dot / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0.0

        candidates = []
        for rec in self.local_records:
            meta = rec["metadata"]
            if meta["sharpe"] >= min_sharpe:
                sim = cosine_sim(current_regime_vector, rec["regime_vector"])
                candidates.append({
                    "metadata": meta,
                    "similarity_score": sim,
                    "record_id": rec["record_id"]
                })
                
        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return candidates[:top_k]


if __name__ == "__main__":
    db = LineageVectorDB()
    db.insert_lineage_record(
        child_id=1, generation=2, model_type="Attention-is-all-you-Need",
        regime_vector=[0.8, 0.2, -0.5, -0.4, 3.1], sharpe=2.8,
        state_dict_path="/forge/weights/gen2_btc_vacuum.pt",
        regime_description="2024 BTC Liquidity Vacuum & High Volatility Flash Crash"
    )
    res = db.query_surviving_weights(current_regime_vector=[0.75, 0.25, -0.45, -0.38, 2.9])
    print("Query results:", json.dumps(res, indent=2))
