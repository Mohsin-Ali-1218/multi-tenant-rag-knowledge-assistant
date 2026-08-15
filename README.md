# Multi-Tenant RAG Knowledge Assistant

An independently developed, production-oriented **Multi-Tenant RAG Knowledge Assistant** backend serving isolated tenant document stores, vector indices, and conversation histories from a single FastAPI deployment.

---

## 🏗️ Architecture Diagram

```
                               ┌───────────────────────────────────────────────┐
                               │                 Client Request                │
                               └───────────────────────┬───────────────────────┘
                                                       │ (Bearer JWT Token)
                                                       ▼
                               ┌───────────────────────────────────────────────┐
                               │                 FastAPI App                   │
                               │          [Tenant Resolver Dependency]         │
                               └───────────────────────┬───────────────────────┘
                                                       │
                                   Extracts & Validates `tenant_id` Claim
                                                       │
                           ┌───────────────────────────┴───────────────────────────┐
                           ▼                                                       ▼
           ┌───────────────────────────────┐                       ┌───────────────────────────────┐
           │        MongoDB Storage        │                       │       Qdrant Vector DB        │
           │  All queries strictly filter  │                       │ Dedicated vector collections  │
           │ by {"tenant_id": tenant_id}   │                       │   tenant_{tenant_id}_docs     │
           │ Compound Index (tenant_id, _id)│                      │  Strict collection isolation  │
           └───────────────────────────────┘                       └───────────────────────────────┘
                                                       │
                                                       ▼
                               ┌───────────────────────────────────────────────┐
                               │                Query Rewriter                 │
                               │      (Context-aware standalone query via      │
                               │              Gemini 3.5 Flash)                │
                               └───────────────────────┬───────────────────────┘
                                                       │
                                                       ▼
                               ┌───────────────────────────────────────────────┐
                               │                 Agent Router                  │
                               │          Exposes tools to Gemini:             │
                               │    - search_knowledge_base(query)             │
                               │    - calculate(expression)                    │
                               └───────────────────────┬───────────────────────┘
                                                       │
                                ┌──────────────────────┴──────────────────────┐
                                ▼                                             ▼
                ┌───────────────────────────────┐             ┌───────────────────────────────┐
                │     RAG Vector Retrieval      │             │    Safe Calculator Engine     │
                │     (BAAI/bge-m3 embeddings   │             │   (asteval arithmetic, no    │
                │     from tenant collection)   │             │      unsafe eval execution)   │
                └───────────────┬───────────────┘             └───────────────┬───────────────┘
                                │                                             │
                                └──────────────────────┬──────────────────────┘
                                                       │
                                                       ▼
                               ┌───────────────────────────────────────────────┐
                               │              Gemini 3.5 Flash                 │
                               │         (Final Synthesis & Citations)         │
                               └───────────────────────┬───────────────────────┘
                                                       │
                                                       ▼
                               ┌───────────────────────────────────────────────┐
                               │           Answer + Cited Sources              │
                               └───────────────────────────────────────────────┘
```

---

## 🚀 Tech Stack

- **Package/Env Manager**: `uv` with `pyproject.toml`
- **Language**: Python 3.12+
- **API Framework**: FastAPI + Uvicorn
- **Database**: MongoDB via **PyMongo Native Async API** (`pymongo>=4.15`, `from pymongo import AsyncMongoClient`)
- **Vector DB**: Qdrant (`qdrant-client>=1.19`) with `AsyncQdrantClient` using dedicated per-tenant collections (`tenant_{tenant_id}_docs`)
- **Embeddings**: `BAAI/bge-m3` (1024-dim dense mode) running locally via `sentence-transformers`
- **LLM**: Google Gemini (`gemini-3.5-flash`) via `google-genai` SDK (`google-genai>=2.0.0`)
- **Auth**: JWT tokens (`pyjwt`) with `passlib[bcrypt]` for password hashing
- **Safe Calculator**: `asteval` safe arithmetic evaluator
- **Config**: `pydantic-settings` reading `.env`
- **Containerization**: `docker-compose.yml` for MongoDB and Qdrant

---

## 🔒 Multi-Tenancy & Tenant Isolation

1. **JWT Auth Claim**: Tenant identity is resolved **exclusively** from the verified JWT `tenant_id` claim. No request header, query parameter, or body field can override tenant scope.
2. **Qdrant Vector Isolation**: Each tenant's vector embeddings reside in a separate collection (`tenant_{tenant_id}_docs`). Search queries cannot touch another tenant's vector space.
3. **MongoDB Scope & Indexing**: Every collection (`documents`, `conversations`, `messages`, `users`) requires `tenant_id`. Every query explicitly includes `{"tenant_id": current_tenant_id}` and utilizes compound indexes `(tenant_id, _id)`.
4. **Cross-Tenant Guard (403 Forbidden)**: Direct document fetch requests (`GET /documents/{id}`) verify `document.tenant_id == current_tenant_id`. Access attempts across tenants yield `403 Forbidden` with zero data leakage.

---

## 🛠️ Setup & Running Instructions

### 1. Start Database & Vector Services
Ensure Docker is running, then start MongoDB and Qdrant:
```bash
docker compose up -d
```

### 2. Install Dependencies
Install all project dependencies into a virtual environment using `uv`:
```bash
uv sync
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and set your `GEMINI_API_KEY`:
```bash
cp .env.example .env
```

### 4. Seed Demo Tenants & Documents
Run `seed.py` to populate two sample tenants (`Acme Legal` and `Acme Healthcare`), seed sample policy docs/FAQs, chunk and embed vectors into Qdrant collections, and register demo admin users:
```bash
uv run python seed.py
```

### 5. Launch FastAPI Server
Start the development server:
```bash
uv run uvicorn backend.main:app --reload --port 8000
```

---

## 🧪 Automated Integration Tests

Run the full tenant isolation integration test suite proving the cross-tenant `403 Forbidden` security requirement:
```bash
uv run pytest tests/test_tenant_isolation.py -v
```

---

## 📖 API Usage Examples

> **Note**: Demo credentials — for local seeded data only. Never use these credentials in a real deployment.

### 1. Authenticate & Obtain JWT Token (Tenant A - Acme Legal)
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@acmelegal.com",
    "password": "password123"
  }'
```
*Output*:
```json
{
  "access_token": "eyJhbGciOiJIUzI1Ni...",
  "token_type": "bearer",
  "user_id": "67a5...",
  "tenant_id": "67a5...",
  "role": "admin"
}
```

### 2. Upload Document (Tenant A)
```bash
curl -X POST "http://localhost:8000/documents" \
  -H "Authorization: Bearer <TENANT_A_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Intellectual Property Policy",
    "raw_text": "All software code created by employees belongs exclusively to Acme Legal."
  }'
```

### 3. List Documents (Tenant A only)
```bash
curl -X GET "http://localhost:8000/documents" \
  -H "Authorization: Bearer <TENANT_A_TOKEN>"
```

### 4. Multi-Tenant RAG Chat (Query Rewrite + Search + Calculator)
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Authorization: Bearer <TENANT_A_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is our retainer refund policy and calculate the refund for a $1000 order?"
  }'
```
*Output*:
```json
{
  "conversation_id": "67a5...",
  "answer": "Under Acme Legal's Refund and Retainer Terms, unused retainers are refunded at 85% minus a $100 administrative fee. For a $1000 order: 85% of $1000 = $850, minus $100 fee yields a net refund of $750.",
  "sources": [
    {
      "document_title": "Refund and Retainer Terms",
      "chunk_text": "Retainer deposits for legal counsel are non-refundable once billable work has commenced...",
      "score": 0.82
    }
  ]
}
```

### 5. Cross-Tenant 403 Forbidden Security Verification
Attempt to fetch a document belonging to Tenant B (`Acme Healthcare`) using Tenant A's token:
```bash
curl -X GET "http://localhost:8000/documents/<TENANT_B_DOC_ID>" \
  -H "Authorization: Bearer <TENANT_A_TOKEN>"
```
*Response* (`HTTP 403 Forbidden`):
```json
{
  "detail": "Access forbidden: Document belongs to a different tenant"
}
```
