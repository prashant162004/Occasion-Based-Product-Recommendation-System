import time
import logging
from typing import List, Optional, Dict, Any
from backend.app.config import settings
from backend.app.schemas import ProductPayload, ProductCard, RecommendRequest, RecommendResponse
from backend.app.utils.data_loader import load_catalog
from backend.app.services.embedding_engine import EmbeddingEngine
from backend.app.services.vector_store import VectorStore
from backend.app.services.reranker import CrossEncoderReranker
from backend.app.services.llm_engine import LLMEngine

logger = logging.getLogger("RecommendationPipeline")

class RecommendationPipeline:
    """
    End-to-End Occasion-Based Product Recommendation Orchestrator.
    Executes:
    1. Query Intent Expansion (AI Reasoner)
    2. Dense Vector Retrieval (Vector Store)
    3. Deep Interaction Re-ranking (Cross-Encoder)
    4. Structured Rationale Synthesis (AI Reasoner)
    5. Clean Presentation Normalization
    """
    def __init__(self):
        self.embedding_engine = EmbeddingEngine(
            model_name=settings.EMBEDDING_MODEL_NAME,
            dimension=settings.EMBEDDING_DIMENSION,
            fallback_model_name=settings.FALLBACK_EMBEDDING_MODEL
        )
        self.vector_store = VectorStore(
            storage_dir=settings.VECTOR_STORAGE_DIR,
            collection_name=settings.COLLECTION_NAME,
            vector_dim=self.embedding_engine.dimension
        )
        self.reranker = CrossEncoderReranker(
            model_name=settings.RERANKER_MODEL_NAME
        )
        self.llm_engine = LLMEngine(
            api_key=settings.LLM_API_KEY,
            model_name=settings.LLM_MODEL
        )
        self.products: List[ProductPayload] = []
        self._is_indexed = False

    def initialize_index(self):
        """Loads product catalog and indexes into vector database."""
        logger.info(f"[INFO] Loading product catalog from {settings.DATA_PATH}")
        self.products = load_catalog(settings.DATA_PATH)
        logger.info(f"[INFO] Loaded {len(self.products)} products.")

        # Check if pre-computed vectors already exist in disk cache for instant startup
        cache_file = settings.VECTOR_STORAGE_DIR / f"vectors_{self.embedding_engine.dimension}.npy"
        if cache_file.exists():
            try:
                import numpy as np
                cached_data = np.load(cache_file)
                if len(cached_data) == len(self.products):
                    logger.info(f"[INFO] Loaded {len(cached_data)} pre-computed embeddings from disk cache ({cache_file.name}).")
                    vectors = cached_data.tolist()
                    self.vector_store.upsert_products(self.products, vectors)
                    self._is_indexed = True
                    logger.info("[INFO] Recommendation Pipeline initialized instantly from cache.")
                    return
            except Exception as e:
                logger.warning(f"[WARNING] Cache load error: {e}. Recomputing embeddings.")

        # Generate embeddings if cache is not present
        searchable_texts = [p.searchable_text for p in self.products]
        logger.info(f"[INFO] Computing dense vector embeddings for {len(searchable_texts)} products...")
        vectors = self.embedding_engine.embed_batch(searchable_texts)

        # Persist vectors to disk cache
        try:
            import numpy as np
            settings.VECTOR_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, np.array(vectors, dtype=np.float32))
            logger.info(f"[INFO] Persisted embeddings to disk cache: {cache_file.name}")
        except Exception as e:
            logger.debug(f"[DEBUG] Cache write note: {e}")

        # Upsert to Vector Store
        self.vector_store.upsert_products(self.products, vectors)
        self._is_indexed = True
        logger.info("[INFO] Recommendation Pipeline successfully indexed and ready.")

    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        start_time = time.time()

        if not self._is_indexed:
            self.initialize_index()

        # Step 1: Expand query intent
        expansion = self.llm_engine.expand_occasion_intent(req.occasion)
        detected_intent = expansion["detected_intent"]
        search_query = expansion["search_query"]
        summary = expansion["summary"]

        # Step 2: Vector Search (Dense Semantic Retrieval)
        query_vector = self.embedding_engine.embed_text(search_query)
        candidates = self.vector_store.search(
            query_vector=query_vector,
            top_k=min(30, max(25, req.top_k * 2)),
            category=req.category
        )

        if not candidates:
            latency = round((time.time() - start_time) * 1000, 2)
            return RecommendResponse(
                occasion=req.occasion,
                detected_intent=detected_intent,
                summary="No matching products found in the catalog for this occasion.",
                total_found=0,
                latency_ms=latency,
                results=[]
            )

        # Step 3: Deep Cross-Encoder Re-ranking
        reranked = self.reranker.rerank(
            query=f"{req.occasion} {detected_intent}",
            candidate_items=candidates,
            top_k=min(len(candidates), 20)
        )

        # Step 4: Intelligent AI Semantic Filter & Rationale Synthesis
        enriched_items = self.llm_engine.filter_and_rationalize(
            occasion=req.occasion,
            candidate_products=reranked,
            max_items=req.top_k
        )

        # Step 5: Format Product Cards with Dynamic Calibrated Relevance Scores
        raw_scores = [float(item.get("rerank_score", item.get("score", 0.5))) for item in enriched_items]
        max_s = max(raw_scores) if raw_scores else 1.0
        min_s = min(raw_scores) if raw_scores else 0.0
        spread = max_s - min_s
        total_items = len(enriched_items)

        cards: List[ProductCard] = []
        for rank_idx, item in enumerate(enriched_items):
            payload = item["payload"]
            raw_s = float(item.get("rerank_score", item.get("score", 0.5)))

            if spread > 0.005 and total_items > 1:
                # Dynamic Min-Max score combined with rank decay for smooth, natural distribution
                ratio = (raw_s - min_s) / spread
                rank_ratio = 1.0 - (rank_idx / max(1, total_items - 1))
                blended = (0.7 * ratio) + (0.3 * rank_ratio)
                calibrated = 66.0 + (blended * 30.0)
            else:
                # Rank-based decay if scores are identical or single item
                calibrated = max(60.0, 95.0 - (rank_idx * 2.5))

            # Ensure strictly descending, non-repeating order for logical ranking presentation
            score_pct = int(round(min(98, max(58, calibrated))))
            if cards and score_pct >= cards[-1].relevance_score:
                score_pct = max(55, cards[-1].relevance_score - 1)

            desc = payload.get("description", "")
            desc_snippet = desc[:120] + "..." if len(desc) > 120 else desc

            price_val = float(payload.get("price", 0.0))
            formatted_price = f"INR {price_val:,.2f}"

            img_url = payload.get("main_image") or (payload.get("files")[0] if payload.get("files") else None)

            stock_val = int(payload.get("stock", 0))
            stock_status = "In Stock" if stock_val > 0 else "Low Stock"

            card = ProductCard(
                product_id=payload.get("id"),
                title=payload.get("title", ""),
                category=payload.get("category_name", ""),
                description=desc,
                description_snippet=desc_snippet,
                image_url=img_url,
                price=price_val,
                formatted_price=formatted_price,
                discount_percentage=float(payload.get("discount_percentage", 0.0)),
                brand=payload.get("brand", "Curated Brand"),
                relevance_score=score_pct,
                recommendation_rationale=item.get("rationale", ""),
                stock_status=stock_status
            )
            cards.append(card)

        latency = round((time.time() - start_time) * 1000, 2)

        return RecommendResponse(
            occasion=req.occasion,
            detected_intent=detected_intent,
            summary=summary,
            total_found=len(cards),
            latency_ms=latency,
            results=cards
        )

    def get_categories(self) -> List[str]:
        if not self.products:
            self.initialize_index()
        cats = sorted(list(set(p.category_name for p in self.products if p.category_name)))
        return cats

# Singleton instance
pipeline = RecommendationPipeline()
