import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.config import settings
from backend.app.schemas import RecommendRequest, RecommendResponse, OccasionPreset
from backend.app.services.pipeline import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("App")

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize vector store and pipeline index
    logger.info("[INFO] Starting Occasion Recommendation Service...")
    pipeline.initialize_index()
    logger.info("[INFO] Service is live and ready to accept requests.")
    yield
    logger.info("[INFO] Shutting down service.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Intelligent occasion-based product recommendation system powered by vector search and semantic re-ranking.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for UI
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/", include_in_schema=False)
async def serve_ui():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Occasion Recommendation Service API is running. Access /docs for API documentation."}

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "indexed_products": len(pipeline.products),
        "vector_storage": settings.COLLECTION_NAME
    }

@app.get("/api/occasions", response_model=list[OccasionPreset])
async def get_preset_occasions():
    """Returns curated preset occasions for 1-click discovery."""
    return [
        OccasionPreset(label="Birthday", sample_query="Birthday party celebration gift"),
        OccasionPreset(label="Wedding", sample_query="Grand wedding celebration and formal gifts"),
        OccasionPreset(label="Diwali", sample_query="Diwali festive prosperity gift hampers and decor"),
        OccasionPreset(label="Valentine's Day", sample_query="Valentine's Day romantic surprise for partner"),
        OccasionPreset(label="Office Party", sample_query="Office corporate celebration and professional gifts"),
        OccasionPreset(label="Housewarming", sample_query="New home housewarming decor and soothing essentials"),
        OccasionPreset(label="Baby Shower", sample_query="Baby shower and newborn celebration gifts")
    ]

@app.get("/api/categories")
async def get_categories():
    return {"categories": pipeline.get_categories()}

@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend_products(request: RecommendRequest):
    """
    Main recommendation endpoint.
    Accepts an occasion string, executes semantic vector search,
    applies precision re-ranking, and synthesizes structured rationales.
    """
    if not request.occasion or len(request.occasion.strip()) < 2:
        raise HTTPException(status_code=400, detail="Occasion query must be at least 2 characters long.")

    try:
        response = pipeline.recommend(request)
        return response
    except Exception as e:
        logger.error(f"[ERROR] Recommendation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal recommendation processing error.")
