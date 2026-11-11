import pytest
from pathlib import Path
from backend.app.config import settings
from backend.app.utils.data_loader import load_catalog
from backend.app.services.embedding_engine import EmbeddingEngine
from backend.app.services.vector_store import VectorStore
from backend.app.services.reranker import CrossEncoderReranker
from backend.app.services.pipeline import pipeline
from backend.app.schemas import RecommendRequest

def test_catalog_loader():
    products = load_catalog(settings.DATA_PATH)
    assert len(products) >= 40, f"Expected at least 40 products, got {len(products)}"
    for p in products:
        assert p.id, "Product ID must not be empty"
        assert p.title, "Product Title must not be empty"
        assert p.category_name, "Category must not be empty"
        assert p.searchable_text, "Searchable text must be constructed"

def test_embedding_generation():
    engine = EmbeddingEngine()
    vec = engine.embed_text("Wedding celebration gift")
    assert len(vec) in (384, 1024)
    assert isinstance(vec[0], float)

def test_vector_store_search():
    store = VectorStore(storage_dir=settings.VECTOR_STORAGE_DIR / "test", collection_name="test_col")
    engine = EmbeddingEngine()

    products = load_catalog(settings.DATA_PATH)[:10]
    vectors = engine.embed_batch([p.searchable_text for p in products])
    store.upsert_products(products, vectors)

    q_vec = engine.embed_text("romantic flower bouquet")
    hits = store.search(q_vec, top_k=5)
    assert len(hits) > 0
    assert "payload" in hits[0]
    assert "score" in hits[0]

def test_reranker():
    reranker = CrossEncoderReranker()
    candidates = [
        {"payload": {"title": "Forever Rose Bouquet", "category_name": "Romantic Gifts", "description": "Romantic rose for lovers"}, "score": 0.75},
        {"payload": {"title": "Office Formal Shirts", "category_name": "Apparel", "description": "Business attire for meetings"}, "score": 0.72}
    ]
    reranked = reranker.rerank("Valentine's Day romantic gift", candidates, top_k=2)
    assert len(reranked) == 2
    # The rose should rank higher for Valentine's Day
    assert "Rose" in reranked[0]["payload"]["title"]

def test_end_to_end_pipeline():
    req = RecommendRequest(occasion="Wedding Ceremony", top_k=6)
    res = pipeline.recommend(req)
    assert res.total_found > 0
    assert len(res.results) <= 6
    assert res.latency_ms > 0
    assert res.detected_intent != ""
    assert res.summary != ""
    for card in res.results:
        assert card.title
        assert card.category
        assert card.relevance_score >= 0
        assert card.recommendation_rationale != ""
