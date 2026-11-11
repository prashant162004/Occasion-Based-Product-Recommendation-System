import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.app.schemas import ProductPayload

logger = logging.getLogger("VectorStore")

class VectorStore:
    """
    High-performance vector database wrapper.
    Uses embedded disk vector engine with automatic memory fallback.
    Computes Cosine Similarity across dense semantic vectors.
    """
    def __init__(self, storage_dir: Path, collection_name: str = "occasion_products", vector_dim: int = 384):
        self.storage_dir = storage_dir
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.client = None
        self._memory_vectors: Dict[str, List[float]] = {}
        self._memory_payloads: Dict[str, Dict[str, Any]] = {}
        self._initialize_store()

    def _initialize_store(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self.storage_dir.mkdir(parents=True, exist_ok=True)
            lock_file = self.storage_dir / ".lock"
            if lock_file.exists():
                try:
                    lock_file.unlink(missing_ok=True)
                except Exception:
                    pass
            self.client = QdrantClient(path=str(self.storage_dir))
            
            # Check or create collection
            collections = [col.name for col in self.client.get_collections().collections]
            if self.collection_name in collections:
                try:
                    col_info = self.client.get_collection(self.collection_name)
                    existing_dim = getattr(col_info.config.params.vectors, "size", None)
                    if existing_dim and existing_dim != self.vector_dim:
                        logger.warning(f"[WARNING] Existing collection dim ({existing_dim}) != required ({self.vector_dim}). Recreating collection.")
                        self.client.delete_collection(self.collection_name)
                        collections.remove(self.collection_name)
                except Exception as ex:
                    logger.debug(f"[DEBUG] Collection inspection note: {ex}")

            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
                )
                logger.info(f"[INFO] Initialized vector collection: {self.collection_name} (dim={self.vector_dim})")
            else:
                logger.info(f"[INFO] Re-connected to existing vector collection: {self.collection_name} (dim={self.vector_dim})")
        except Exception as e:
            logger.warning(f"[WARNING] Local vector DB storage initialization notice ({e}). Operating in resilient memory mode.")
            self.client = None

    def upsert_products(self, products: List[ProductPayload], vectors: List[List[float]]):
        """Insert or update products with corresponding vector embeddings."""
        if not products or not vectors or len(products) != len(vectors):
            raise ValueError("Products and vectors must have equal non-zero lengths.")

        # Always store in memory cache for instant lookups & fallback
        for idx, (prod, vec) in enumerate(zip(products, vectors)):
            payload = prod.model_dump()
            self._memory_vectors[prod.id] = vec
            self._memory_payloads[prod.id] = payload

        # Upsert into vector engine if available
        if self.client is not None:
            try:
                from qdrant_client.models import PointStruct
                points = [
                    PointStruct(
                        id=idx + 1,
                        vector=vec,
                        payload={"product_id": prod.id, **prod.model_dump()}
                    )
                    for idx, (prod, vec) in enumerate(zip(products, vectors))
                ]
                self.client.upsert(collection_name=self.collection_name, points=points)
                logger.info(f"[INFO] Indexed {len(points)} product vectors in storage collection.")
            except Exception as e:
                logger.warning(f"[WARNING] Storage upsert deferred to memory matrix ({e}).")

    def search(self, query_vector: List[float], top_k: int = 15, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Execute Cosine Similarity search against product vectors.
        Returns top matching product payloads with similarity scores (0.0 to 1.0).
        """
        if self.client is not None:
            try:
                hits = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k * 2
                ).points
                results = []
                for hit in hits:
                    payload = hit.payload
                    score = float(hit.score)
                    if category and payload.get("category_name", "").lower() != category.lower():
                        continue
                    results.append({"payload": payload, "score": score})
                    if len(results) >= top_k:
                        break
                if results:
                    return results
            except Exception as e:
                logger.warning(f"[WARNING] Vector query fallback to memory matrix ({e}).")

        # In-Memory Cosine Similarity Matrix Search
        return self._search_memory(query_vector, top_k=top_k, category=category)

    def _search_memory(self, query_vector: List[float], top_k: int = 15, category: Optional[str] = None) -> List[Dict[str, Any]]:
        q_arr = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        if q_norm > 0:
            q_arr = q_arr / q_norm

        scores = []
        for p_id, vec in self._memory_vectors.items():
            payload = self._memory_payloads.get(p_id, {})
            if category and payload.get("category_name", "").lower() != category.lower():
                continue

            v_arr = np.array(vec, dtype=np.float32)
            v_norm = np.linalg.norm(v_arr)
            if v_norm > 0:
                v_arr = v_arr / v_norm

            sim = float(np.dot(q_arr, v_arr))
            # Normalize to 0.0 - 1.0 range
            norm_sim = (sim + 1.0) / 2.0
            scores.append({"payload": payload, "score": norm_sim})

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    def count(self) -> int:
        if self.client is not None:
            try:
                return self.client.count(collection_name=self.collection_name).count
            except Exception:
                pass
        return len(self._memory_vectors)
