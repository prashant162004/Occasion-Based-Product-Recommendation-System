from typing import List, Optional
from pydantic import BaseModel, Field

class ProductPayload(BaseModel):
    id: str
    shop_id: Optional[str] = None
    category_id: Optional[str] = None
    category_name: str
    title: str
    description: str
    short_description: Optional[str] = ""
    main_image: Optional[str] = None
    files: List[str] = Field(default_factory=list)
    brand: Optional[str] = "Generic"
    sku: Optional[str] = None
    price: float = 0.0
    discount_percentage: float = 0.0
    stock: int = 0
    status: str = "ACTIVE"
    searchable_text: str = ""

class ProductCard(BaseModel):
    product_id: str
    title: str
    category: str
    description: str
    description_snippet: str
    image_url: Optional[str] = None
    price: float
    formatted_price: str
    discount_percentage: float
    brand: str
    relevance_score: int = Field(..., ge=0, le=100, description="Match confidence percentage 0-100")
    recommendation_rationale: str
    stock_status: str

class RecommendRequest(BaseModel):
    occasion: str = Field(..., min_length=2, max_length=150, description="Occasion name or query description")
    top_k: int = Field(default=12, ge=1, le=50)
    category: Optional[str] = None

class RecommendResponse(BaseModel):
    occasion: str
    detected_intent: str
    summary: str
    total_found: int
    latency_ms: float
    results: List[ProductCard]

class OccasionPreset(BaseModel):
    label: str
    category_hint: Optional[str] = None
    sample_query: str
