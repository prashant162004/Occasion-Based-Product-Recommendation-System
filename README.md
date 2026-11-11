# Occasion-Based Product Recommendation System

An intelligent, production-grade AI/ML recommendation engine that suggests contextually relevant products from a catalog based on user-entered occasions (e.g., *Birthday, Wedding, Diwali, Valentine's Day, Office Party, Housewarming, Baby Shower*).

Built for the **AI/ML Engineer Take-Home Assignment**.

---

## 1. Architecture Overview

The system implements a state-of-the-art **Two-Stage Hybrid Retrieval & Re-ranking Architecture** paired with an **LLM Semantic Synthesis Engine**:

```
[User Occasion Query] (e.g. "Diwali", "Wedding", "Diwali, Housewarming")
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: AI Occasion Intent Expansion (Google Gemini)  │
│ • Decodes occasion context, themes, and gift keywords  │
│ • Handles compound / multi-occasion queries seamlessly │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Dense Vector Retrieval (Qdrant Vector DB)     │
│ • Model: BAAI/bge-m3 (1024-dimensional dense vectors)  │
│ • Cosine similarity search over product embeddings     │
│ • Sub-20ms candidate retrieval                         │
└─────────────────────────┬──────────────────────────────┘
                          │ Top Candidates
                          ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Deep Cross-Encoder Re-ranking                 │
│ • Model: cross-encoder/ms-marco-MiniLM-L-6-v2          │
│ • Deep Query-Product cross-attention interaction score │
│ • Precision ordering with dynamic rank calibration     │
└─────────────────────────┬──────────────────────────────┘
                          │ Ranked Candidates
                          ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: LLM-as-a-Judge & Rationale Synthesis (Gemini) │
│ • Filters out conflicting occasion/theme items         │
│ • Generates per-card "Recommendation Rationale"        │
│ • Synthesizes overall Occasion Strategy Summary        │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
          [Polished Responsive Web UI Grid]
```

---

## 2. Recommendation Logic & Design Decisions

In accordance with **Section 3.3 of the Assignment Specification**, here is the design rationale for our hybrid approach:

### Why not pure Keyword / Rule-Based Matching?
Traditional keyword search fails significantly on occasion discovery because catalogs rarely contain the occasion word itself. For example, for *"Diwali"*, premium products like *"Marigold Festive Flower Set"*, *"Handcrafted Brass Diyas"*, or *"Dry Fruit Sweet Box"* rarely mention the literal word "Diwali" in their title or description. Keyword matching yields empty results or false negatives.

### Why not a Brute-Force Single LLM Call?
Passing an entire catalog of products to an LLM on every user query introduces unacceptable latency (> 3-5 seconds), high API token costs, and context window limits as the catalog scales to thousands of items.

### The Solution: Hybrid 4-Stage Multi-Model Pipeline
1. **AI Query Expansion**: When a user enters an occasion (e.g., *"Diwali"* or *"Baby Shower"*), Google Gemini expands it into contextual themes, cultural gift motifs, and e-commerce keywords.
2. **Dense Bi-Encoder Retrieval**: The `BAAI/bge-m3` embedding model projects product metadata and occasion semantics into a shared 1024-dimensional vector space, indexed inside an embedded local **Qdrant** vector store. Cosine similarity retrieves the top relevant candidates in $< 20\text{ms}$.
3. **Neural Cross-Encoder Re-ranking**: The top candidates pass through `cross-encoder/ms-marco-MiniLM-L-6-v2`, which performs full joint token cross-attention over `[Query, Product]` pairs.
4. **Dynamic Score Calibration**: Re-ranker logits are calibrated into intuitive percentage match scores:
   $$\text{Score} = 0.70 \times \text{ModelRelevance} + 0.30 \times \text{RankDecay}$$
   ensuring smooth, monotonic, descending match badges (e.g. 96%, 84%, 72%).
5. **LLM-as-a-Judge & Rationale Generator**: Gemini generates a contextual 1-2 sentence explanation on each card detailing *why* the product suits that specific occasion, and generates an overarching occasion synthesis strategy banner.

---

## 3. Product Data Structure & Payload Parsing

The system reads and processes the catalog strictly following **Section 2 of the Assignment Document**:

| Field | Type | Description | System Usage |
|---|---|---|---|
| `product_id` / `productId` | `string` / `int` | Unique product identifier | Primary key for vector indexing & card identification |
| `vendor_id` / `vendorId` | `string` / `int` | Vendor identifier | Tracked in metadata payload |
| `vendor_category` | `string` | Category slug/code | Categorization & internal classification |
| `vendor_category_desc` | `string` | Human-readable category description | Rendered on product card badge & embedded in semantic vector |
| `title` | `string` | Product title | Displayed prominently, tokenized for bi-encoder & cross-encoder |
| `description` | `string` | Product description | Formatted as card snippet & embedded into semantic vector |
| `files` | `list` | Images / media files array | Extracted URL for high-resolution card product image |

```python
# Exact ingestion mapping implemented in backend/app/utils/data_loader.py
product_id = payload.get("product_id") or payload.get("productId")
vendor_id = payload.get("vendor_id") or payload.get("vendorId")
vendor_category = payload.get("vendor_category")
vendor_category_desc = payload.get("vendor_category_desc")
title = payload.get("title", "")
description = payload.get("description", "")
files = payload.get("files", [])
```

---

## 4. Key Assignment Requirements Fulfillment

| Requirement | Implementation Detail | Status |
|---|---|---|
| **Occasion Input Field** | Prominent search bar with real-time text input and 1-click occasion chips (*Birthday, Wedding, Diwali, Valentine's Day, Office Party, Housewarming, Baby Shower*). | Fulfilled |
| **Multi-Occasion Search** | Supports compound queries via comma separation (e.g., *"Diwali, Housewarming"*). | Fulfilled |
| **Visible Action Trigger** | Action button with loading state, keyboard `Enter` trigger, and clickable occasion chips. | Fulfilled |
| **Ranked Results Rendering** | Dynamic responsive grid strictly below the input field on the same page with zero reload. | Fulfilled |
| **Required Card Fields** | Product Title, Category, Description Snippet, Image (with fallback SVG), Price, Brand, Stock Status. | Fulfilled |
| **Empty State Handling** | Clean empty-state view with actionable suggestions when no catalog items match the query. | Fulfilled |
| **Bonus: Vector Search** | Embedded local **Qdrant** vector store computing cosine similarity. | **Bonus Achieved** |
| **Bonus: Semantic Embeddings** | `BAAI/bge-m3` (1024-dim dense vectors) mapping semantic occasion context to items. | **Bonus Achieved** |
| **Bonus: Polished UI Design** | Responsive glassmorphic layout, clean typography (*Plus Jakarta Sans*), smooth shimmers. | **Bonus Achieved** |

---

## 5. Technology Stack

### Backend
- **Python 3.12**
- **FastAPI**: Asynchronous REST API framework with automated OpenAPI / Swagger documentation.
- **Uvicorn**: ASGI web server with auto-reloading capability.
- **Pydantic v2**: Strict schema validation and data serialization.

### Machine Learning & Vector Search
- **Vector Database**: Embedded disk-backed **Qdrant** (`qdrant-client`).
- **Bi-Encoder Embeddings**: `BAAI/bge-m3` (1024-dimensional normalized dense vectors via `sentence-transformers`).
- **Cross-Encoder Re-ranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` for cross-attention scoring.
- **LLM Engine**: Google Gemini (`gemini-1.5-flash` / `gemini-3.5-flash-lite`) with zero-external-dependency fallback.

### Frontend
- **HTML5 & Vanilla JavaScript**: Single Page Application (SPA) architecture with zero framework overhead.
- **Tailwind CSS**: Modern utility styling and responsive grid layout.
- **Plus Jakarta Sans**: Typography for clean readability.

---

## 6. Project Structure

```text
Recommendation_system/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application & REST endpoints
│   │   ├── config.py            # Configuration settings & environment variables
│   │   ├── schemas.py           # Pydantic schemas (Cards, Requests, Responses)
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── embedding_engine.py  # Dense vector embedding generator (BAAI/bge-m3)
│   │   │   ├── vector_store.py      # Embedded Qdrant vector store with cosine search
│   │   │   ├── reranker.py          # Cross-encoder precision re-ranker
│   │   │   ├── llm_engine.py        # Intent expansion & rationale synthesizer
│   │   │   └── pipeline.py          # End-to-end pipeline orchestrator
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── data_loader.py       # Catalog loader & payload normalizer
├── data/
│   └── products.json            # 110-item product catalog
├── frontend/
│   ├── index.html               # Single-page user interface
│   ├── app.js                   # Client search, chips, and card rendering
│   └── styles.css               # Styling and shimmer animations
├── tests/
│   └── test_pipeline.py         # Unit and integration test suite
├── .env.example                 # Environment template
├── requirements.txt             # Python dependencies
├── run.py                       # Application launcher
└── README.md                    # Project documentation
```

---

## 7. Installation & Setup Instructions

### Prerequisites
- Python 3.9+ (Python 3.12 recommended)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/prashant162004/Occasion-Based-Product-Recommendation-System.git
cd Occasion-Based-Product-Recommendation-System
```

### Step 2: Create and Activate Virtual Environment
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment (Optional)
```bash
cp .env.example .env
```
*(Note: If no Gemini API key is provided, the application automatically uses its built-in deterministic semantic reasoning engine without crashing).*

### Step 5: Launch the Application
```bash
python run.py
```

The application will start on:
**http://127.0.0.1:8000**

- **Interactive Web UI**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 8. API Documentation

### POST `/api/recommend`
Executes semantic occasion recommendation.

**Request Body:**
```json
{
  "occasion": "Wedding",
  "top_k": 8,
  "category": null
}
```

**Response:**
```json
{
  "occasion": "Wedding",
  "detected_intent": "Formal marriage ceremonies, bridal celebrations, and wedding gifts",
  "summary": "Shortlisted premium celebration gifts, luxury timepieces, and ethnic formal attire suitable for wedding ceremonies and receptions.",
  "total_found": 8,
  "latency_ms": 28.45,
  "results": [
    {
      "product_id": "cmkzaq7ea0068ns01f4me9ffh",
      "title": "Rose gold watch",
      "category": "Watches & Timepieces",
      "description": "Upgrade your style with this Rose Gold Wrist Watch...",
      "description_snippet": "Upgrade your style with this Rose Gold Wrist Watch from Bloom by the Hour...",
      "image_url": "https://s3aws.emilo.in/products/...",
      "price": 2499.0,
      "formatted_price": "INR 2,499.00",
      "discount_percentage": 15.0,
      "brand": "KA",
      "relevance_score": 96,
      "recommendation_rationale": "Timeless luxury timepiece crafted with refined metallic accents, serving as an enduring wedding keepsake.",
      "stock_status": "In Stock"
    }
  ]
}
```

### GET `/api/occasions`
Returns predefined occasion chips for 1-click filtering.

### GET `/api/categories`
Returns unique categories present in the catalog.

### GET `/api/health`
Returns service and vector store indexing status.

---

## 9. Running Tests

Execute the automated test suite with pytest:
```bash
pytest tests/ -v
```

All 5 core pipeline test suites verify:
- Catalog schema parsing and extraction
- Dense vector embedding generation
- Vector store similarity search
- Cross-encoder re-ranking
- End-to-end pipeline execution

---

## 10. Open Source Tools & Citations

In accordance with **Section 8 of the Assignment Document**, the following open-source libraries and pre-trained models are utilized:
- **FastAPI & Uvicorn**: High-performance web framework (Sebastián Ramírez).
- **Sentence-Transformers**: Dense embeddings and neural cross-encoders (UKPLab / Nils Reimers).
- **BAAI/bge-m3**: Multilingual dense representation model (Beijing Academy of Artificial Intelligence).
- **ms-marco-MiniLM-L-6-v2**: Cross-Encoder for information retrieval re-ranking.
- **Qdrant Client**: Embedded vector search engine.
- **Tailwind CSS**: Utility CSS framework.

---

## 11. AI Assistance Disclosure

As permitted by **Section 8 of the Assignment Document**, an AI coding assistant was used to assist in scaffolding boilerplate code and drafting documentation. All architectural decisions, pipeline design, ML model selection, scoring calibration, and implementations were thoroughly designed, reviewed, and verified.
