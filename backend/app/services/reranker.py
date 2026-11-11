import logging
import re
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("Reranker")

class CrossEncoderReranker:
    """
    Precision Re-ranker utilizing cross-encoder deep contextual interaction.
    Scores (query, product) pairs to refine vector retrieval ranking.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"[INFO] Initializing re-ranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("[INFO] Re-ranker model loaded successfully.")
        except Exception as e:
            logger.warning(f"[WARNING] CrossEncoder notice ({e}). Operating in lexical-semantic hybrid reranker mode.")
            self.model = None

    def rerank(self, query: str, candidate_items: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Takes candidate items from Vector Store and rescores them relative to the query.
        Returns top_k re-ranked items with updated relevance scores.
        """
        if not candidate_items:
            return []

        pairs = []
        for item in candidate_items:
            payload = item["payload"]
            title = payload.get("title", "")
            category = payload.get("category_name", "")
            desc = payload.get("description", "")
            brand = payload.get("brand", "")
            doc_text = f"{title}. Category: {category}. Brand: {brand}. Details: {desc}"
            pairs.append([query, doc_text])

        # Step 1: Compute neural cross-encoder scores if available
        if self.model is not None:
            try:
                raw_scores = self.model.predict(pairs)
                # Sigmoid normalization to 0.0 - 1.0 range
                import numpy as np
                norm_scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores, dtype=np.float32)))
                for item, score in zip(candidate_items, norm_scores):
                    # Ensemble score: 60% Cross-Encoder + 40% Vector Cosine Similarity
                    item["rerank_score"] = float(0.6 * score + 0.4 * item.get("score", 0.5))
                candidate_items.sort(key=lambda x: x["rerank_score"], reverse=True)
                return candidate_items[:top_k]
            except Exception as e:
                logger.warning(f"[WARNING] Neural reranking fallback: {e}")

        # Step 2: Contextual hybrid re-scoring fallback
        query_tokens = set(re.findall(r'\w+', query.lower()))
        for item in candidate_items:
            payload = item["payload"]
            title_tokens = set(re.findall(r'\w+', payload.get("title", "").lower()))
            cat_tokens = set(re.findall(r'\w+', payload.get("category_name", "").lower()))
            desc_tokens = set(re.findall(r'\w+', payload.get("description", "").lower()))

            overlap_title = len(query_tokens.intersection(title_tokens)) * 2.0
            overlap_cat = len(query_tokens.intersection(cat_tokens)) * 1.5
            overlap_desc = len(query_tokens.intersection(desc_tokens)) * 0.8

            lexical_boost = min(0.35, (overlap_title + overlap_cat + overlap_desc) * 0.08)
            base_score = item.get("score", 0.5)
            item["rerank_score"] = min(0.99, base_score + lexical_boost)

        candidate_items.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidate_items[:top_k]
