# Occasion-Based Product Recommendation System

An intelligent, end-to-end semantic product recommendation engine that matches e-commerce catalog items to user occasions (e.g., *Diwali, Weddings, Birthdays, Housewarming, Baby Showers, Office Parties*) using dense vector retrieval, cross-encoder re-ranking, and dynamic LLM intent expansion.

---

## 1. Problem Overview: Why Occasion Search is Hard

In standard e-commerce platforms, search systems rely heavily on lexical keyword matching (BM25, inverted indices). However, **occasion-based shopping suffers from severe semantic vocabulary mismatch**:

- A user searching for **"Diwali"** is looking for festive gift hampers, brass diyas, marigold garlands, ethnic attire, and dry fruits. But catalog product titles rarely contain the literal word *"Diwali"*.
- A user searching for **"Housewarming"** expects plant-based scented candles, indoor plants, or home decor, not products with "housewarming" in their title.
- A user searching for compound queries like **"Diwali, Housewarming"** needs recommendations that blend festive celebration with welcoming home decor.

### Why not use a single LLM to rank the entire catalog?
Passing hundreds or thousands of product descriptions to a large language model on every user request introduces heavy latency (> 3-5 seconds), high token costs, and context window bottlenecks.

### The Solution
A **Two-Stage Hybrid Retrieval & Re-ranking Architecture**:
1. **Bi-Encoder Dense Vector Retrieval (Qdrant)**: High-speed candidate generation retrieving the top relevant items in sub-20ms.
2. **Cross-Encoder Re-Ranking (MiniLM)**: Full joint token cross-attention scoring to precision-rank the candidate items.
3. **LLM Semantic Synthesizer (Google Gemini)**: Real-time query expansion, multi-occasion intent decoding, and per-card explainable recommendation rationales.

---

## 2. System Architecture

```
[User Occasion Query]  (e.g., "Diwali", "Wedding", "Diwali, Housewarming")
          │
          ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage 1: Dynamic Occasion Intent Expansion (Google Gemini)        │
│ • Decodes occasion context, tone, and cultural gifting themes    │
│ • Seamlessly handles compound queries (comma-separated occasions)│
│ • Generates e-commerce targeted search vectors and keywords      │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage 2: Dense Vector Retrieval (Qdrant Vector Database)         │
│ • Embedding Model: BAAI/bge-m3 (1024-dimensional dense vectors) │
│ • High-speed Cosine Similarity Search over catalog embeddings    │
│ • Sub-20ms Candidate Retrieval (< 20ms latency)                  │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ Top Candidates
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage 3: Deep Interaction Re-ranking (Neural Cross-Encoder)      │
│ • Model: cross-encoder/ms-marco-MiniLM-L-6-v2                    │
│ • Joint Query-Product Cross-Attention Rescoring                  │
│ • Eliminates false positives and optimizes precision ordering    │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ Ranked Candidates
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage 4: Calibration & Explainable Rationale Generation          │
│ • Dynamic Score Calibration: 70% model relevance + 30% rank decay│
│ • Gemini synthesizes a tailored 1-sentence "Why This Fits" note   │
│ • Generates overall Occasion Strategy Summary banner             │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
         [Interactive Web UI Grid (FastAPI + Tailwind CSS)]
```

---

## 3. Key Capabilities

- **Natural Semantic Understanding**: Understands cultural and emotional context of occasions rather than requiring exact keyword matches.
- **Compound Multi-Occasion Search**: Supports comma-separated multi-occasion inputs (e.g., *"Diwali, Housewarming"*) by dynamically combining intent vectors.
- **Two-Stage High-Performance Pipeline**: Sub-20ms dense vector retrieval paired with deep cross-attention re-ranking for enterprise-grade relevance.
- **Dynamic Score Calibration**: Raw model logits are normalized into smooth, descending percentage match badges:
  $$\text{Match Percentage} = 0.70 \times \text{ModelRelevance} + 0.30 \times \text{RankDecay}$$
  producing intuitive scores like 96%, 90%, 77%, 75%.
- **Explainable AI (Recommendation Rationales)**: Every product card displays a dedicated justification explaining specifically why that item matches the entered occasion.
- **Zero-Dependency Fallback Engine**: If no Gemini API key is provided, the application automatically switches to its built-in deterministic semantic engine without failing or crashing.
- **Modern Responsive Web UI**: Single Page Application (SPA) built with Tailwind CSS, clean typography, 1-click occasion chips, dark/light theme toggle, and zero page reloads.

---

## 4. Catalog Data Schema & Ingestion

The product catalog (`data/products.json`) is loaded and normalized through `backend/app/utils/data_loader.py`:

| Field | Type | Description | System Usage |
|---|---|---|---|
| `product_id` / `productId` | `string` / `int` | Unique product identifier | Primary key for vector indexing & card identification |
| `vendor_id` / `vendorId` | `string` / `int` | Vendor identifier | Tracked in metadata payload |
| `vendor_category` | `string` | Category slug/code | Categorization and internal classification |
| `vendor_category_desc` | `string` | Human-readable category description | Rendered on product card badge & embedded in semantic vector |
| `title` | `string` | Product title | Displayed prominently, tokenized for bi-encoder & cross-encoder |
| `description` | `string` | Product description | Formatted as card snippet & embedded into semantic vector |
| `files` | `list` | Product images / media array | Extracted URL for high-resolution card product image |

```python
# Payload extraction logic (backend/app/utils/data_loader.py)
product_id = payload.get("product_id") or payload.get("productId")
vendor_id = payload.get("vendor_id") or payload.get("vendorId")
vendor_category = payload.get("vendor_category")
vendor_category_desc = payload.get("vendor_category_desc")
title = payload.get("title", "")
description = payload.get("description", "")
files = payload.get("files", [])
```

---

## 5. Technology Stack

### Backend
- **Python 3.12**
- **FastAPI**: Asynchronous REST framework with OpenAPI / Swagger documentation.
- **Uvicorn**: ASGI production web server.
- **Pydantic v2**: Strict schema validation and data serialization.

### Machine Learning & Vector Search
- **Vector Database**: Embedded disk-backed **Qdrant** (`qdrant-client`).
- **Dense Embedding Model**: `BAAI/bge-m3` (1024-dimensional normalized dense vectors via `sentence-transformers`).
- **Cross-Encoder Re-ranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` for cross-attention scoring.
- **LLM Engine**: Google Gemini (`gemini-3.5-flash-lite`) via REST transport for intent expansion and rationale synthesis.

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

## 7. Installation & Quickstart Guide

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

### Step 4: Configure Environment & Gemini API Key (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

To enable dynamic LLM intent expansion and personalized recommendation rationales with Google Gemini, add your API key in `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> **Zero-Dependency Fallback:** An external API key is **completely optional**. If left blank, the system automatically uses its built-in deterministic semantic reasoning engine without external network dependencies.

### Step 5: Launch the Application
```bash
python run.py
```

The application will start on:
**http://127.0.0.1:8000**

- **Interactive Web UI**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 8. API Reference

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

### Auxiliary Endpoints
- `GET /api/occasions`: Returns predefined occasion chips for 1-click filtering.
- `GET /api/categories`: Returns unique categories present in the catalog.
- `GET /api/health`: Returns service status and vector store indexing health.

---

## 9. Automated Testing & Verification

Run the automated test suite with pytest:
```bash
pytest tests/ -v
```

All 5 core pipeline test suites verify:
- **Catalog schema parsing**: Validates extraction of all product attributes.
- **Dense vector embeddings**: Validates 1024-dimensional normalized vector generation with `BAAI/bge-m3`.
- **Vector store retrieval**: Verifies embedded Qdrant cosine similarity candidate search.
- **Cross-encoder re-ranking**: Confirms neural cross-attention rescoring and score ordering.
- **End-to-end pipeline execution**: Tests end-to-end flow from raw query string to structured output cards.

---

## 10. Open Source References & Citations

- **FastAPI & Uvicorn**: High-performance asynchronous web framework (Sebastián Ramírez).
- **Sentence-Transformers**: Dense representations and cross-encoder implementations (UKPLab / Nils Reimers).
- **BAAI/bge-m3**: Multilingual dense semantic embedding model (Beijing Academy of Artificial Intelligence).
- **ms-marco-MiniLM-L-6-v2**: Pre-trained Cross-Encoder for information retrieval re-ranking.
- **Qdrant Client**: Embedded vector search database.
- **Tailwind CSS**: Utility-first CSS framework.

---

## 11. Notes

- **AI Tools Notice**: AI-assisted developer tooling was utilized for rapid code scaffolding, test boilerplate generation, and initial documentation drafting. All pipeline architecture, ML model choices, score calibration formulas, and system implementations were reviewed, refined, and verified.
