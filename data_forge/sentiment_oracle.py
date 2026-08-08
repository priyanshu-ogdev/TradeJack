"""
Sentiment & On-Chain Oracle Engine.
Interfaces with a local Qdrant Vector DB to store and retrieve dense NLP embeddings (BGE-Large-v1.5).
Optimized for 500GB+ of vector data via disk-based memory mapping (mmap).
"""

import os
import logging
from typing import List, Dict, Any

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from data_forge.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (SentimentOracle) %(message)s")
logger = logging.getLogger("SentimentOracle")

class SentimentOracle:
    def __init__(self, collection_name: str = "tradejack_sentiment"):
        self.collection_name = collection_name
        
        if not QDRANT_AVAILABLE:
            logger.error("qdrant-client not installed. Cannot connect to Vector DB.")
            self.client = None
            return
            
        try:
            self.client = QdrantClient(url=config.qdrant_url)
            self._ensure_collection()
            logger.info(f"Connected to Qdrant at {config.qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.client = None

    def _ensure_collection(self):
        """Creates the collection with mmap config if it doesn't exist."""
        if not self.client:
            return
            
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            logger.info(f"Creating Qdrant collection: {self.collection_name}")
            
            # For BGE-Large-v1.5, vector size is 1024.
            # HNSW mmap is enabled to keep RAM usage extremely low for 500GB scale.
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=1024, 
                    distance=rest.Distance.COSINE
                ),
                optimizers_config=rest.OptimizersConfigDiff(
                    memmap_threshold=10000  # Map to disk aggressively
                ),
                hnsw_config=rest.HnswConfigDiff(
                    mmap=True,
                    payload_mmap=True
                )
            )

    def ingest_embeddings(self, embeddings: List[List[float]], payloads: List[Dict[str, Any]], ids: List[str] = None):
        """
        Bulk upserts embeddings into Qdrant.
        """
        if not self.client:
            return
            
        import uuid
        if not ids:
            ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
            
        points = [
            rest.PointStruct(id=idx, vector=vector, payload=payload)
            for idx, vector, payload in zip(ids, embeddings, payloads)
        ]
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.debug(f"Ingested {len(points)} vectors into {self.collection_name}.")
        except Exception as e:
            logger.error(f"Failed to ingest vectors: {e}")

    def query_sentiment(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the Vector DB for the most relevant historical context.
        """
        if not self.client:
            return []
            
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return [{"id": hit.id, "score": hit.score, "payload": hit.payload} for hit in results]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

if __name__ == "__main__":
    logger.info("Initializing Sentiment Oracle...")
    oracle = SentimentOracle()
    if oracle.client:
        logger.info("Oracle is active and collection is ready.")
    else:
        logger.warning("Oracle is running in degraded state (no Qdrant connection).")
