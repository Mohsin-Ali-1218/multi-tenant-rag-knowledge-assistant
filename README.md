# Multi-Tenant RAG Knowledge Assistant

A multi-tenant Retrieval-Augmented Generation (RAG) backend built with FastAPI, MongoDB, Qdrant, and Google Gemini. The system provides strict tenant-isolated document storage, vector indices, and conversation histories from a unified service architecture.

---

## 🏗️ Architecture & Request Flow

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
            │  All collections filter by:   │                       │ Dedicated vector collections: │
            │  {"tenant_id": tenant_id}     │                       │    tenant_{tenant_id}_docs    │
            │  Compound index (tenant_id,_id│                       │  Strict collection isolation  │
            └───────────────┬───────────────┘                       └───────────────┬───────────────┘
                            │                                                       │
                            │             Batch Parent ID Validation                │
                            │       (Discards orphaned & cross-tenant hits)         │
                            └───────────────────────────◄───────────────────────────┘
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
                 │      + Parent Doc Validation) │             │      unsafe eval execution)   │
                 └───────────────┬───────────────┘             └───────────────┬───────────────┘
                                 │                                             │
                                 └──────────────────────┬──────────────────────┘
                                                        │
                                                        ▼
                                ┌───────────────────────────────────────────────┐
                                │              Gemini 3.5 Flash                 │
                                │         (Grounded Synthesis & Citations)      │
                                │         [Fails safe with is_degraded]         │
                                └───────────────────────┬───────────────────────┘
                                                        │
                                                        ▼
                                ┌───────────────────────────────────────────────┐
                                │       Answer + Structured Cited Sources       │
                                └───────────────────────────────────────────────┘
```

### Core Components
1. **Tenant Isolation**: Tenant identity is resolved exclusively from the verified JWT `tenant_id` claim. Every database query enforces tenant scoping, and Qdrant isolates vector spaces into separate collections (`tenant_{tenant_id}_docs`).
2. **Retrieval Integrity & Orphaned Vector Protection**: Hits retrieved from Qdrant are batch-validated against MongoDB by authenticated tenant ID. Hits referencing deleted, malformed, or cross-tenant parent documents are pruned prior to synthesis. If validation fails, retrieval fails closed.
3. **Structured Source Metadata**: Retains filenames and 1-based PDF page numbers directly from vector payloads into response sources and persisted message history.
4. **Graceful Degradation (`is_degraded`)**: If Gemini answer generation fails (e.g., HTTP 429 quota exhaustion or API error), the system returns a stable, generic unavailable message with empty sources and `is_degraded: true`, avoiding raw document dumps or leaked database errors.

---

## 🚀 Tech Stack

- **Package/Env Manager**: `uv` with `pyproject.toml`
- **Language**: Python 3.12+ (tested on Python 3.13)
- **API Framework**: FastAPI + Uvicorn
- **Database**: MongoDB via **PyMongo Native Async API** (`AsyncMongoClient`)
- **Vector DB**: Qdrant (`qdrant-client` async) with dedicated per-tenant collections
- **Embeddings**: `BAAI/bge-m3` (1024-dim dense) running locally via `sentence-transformers`
- **CPU-Only PyTorch**: Configured via `uv` wheel index to avoid multi-gigabyte CUDA packages
- **Document Extractors**: `pypdf` (PDF), `python-docx` (DOCX), pure UTF-8 (TXT)
- **LLM**: Google Gemini (`gemini-3.5-flash`) via `google-genai` SDK
- **Safe Calculator**: `asteval` safe arithmetic evaluation (no arbitrary code execution)
- **Authentication**: JWT tokens (`pyjwt`) with `bcrypt` password hashing

---

## 🛠️ Local Setup Instructions

### 1. Prerequisites
- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- [uv](https://docs.astral.sh/uv/) (Modern Python package installer and resolver)

### 2. Start Storage Services
Start local instances of MongoDB 7.0 and Qdrant using the provided `docker-compose.yml`:
```bash
docker compose up -d
```
- MongoDB: `localhost:27017`
- Qdrant: `localhost:6333`

### 3. Install Dependencies (CPU-Only PyTorch)
The project configuration in `pyproject.toml` directs `torch` and `torchvision` to the official PyTorch CPU wheel index (`https://download.pytorch.org/whl/cpu`):
```bash
uv sync
```
This installs all application dependencies while avoiding heavy NVIDIA/CUDA packages.

### 4. Configure Environment
Copy the example configuration to `.env`:
```bash
cp .env.example .env
```

Key environment settings in `.env.example`:
| Variable | Description | Default / Example |
|---|---|---|
| `MONGODB_URI` | MongoDB connection string | `mongodb://localhost:27017` |
| `MONGODB_DB_NAME` | MongoDB database name | `multitenant_rag` |
| `QDRANT_URL` | Qdrant service URL | `http://localhost:6333` |
| `GEMINI_API_KEY` | Google Gemini API key | Optional for tests; required for live synthesis |
| `JWT_SECRET` | Secret key for signing tokens | Set a 32+ character random secret |
| `JWT_ALGORITHM` | Algorithm for JWT tokens | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token lifetime | `1440` (24 hours) |
| `EMBEDDING_MODEL_NAME` | Sentence transformer model | `BAAI/bge-m3` |
| `RAG_SIMILARITY_THRESHOLD` | Cosine similarity threshold | `0.50` |
| `MAX_UPLOAD_SIZE_MB` | Upload file size ceiling | `15` |
| `MAX_DOCUMENT_PAGES` | Page limit for PDF uploads | `100` |
| `MAX_DOCX_UNCOMPRESSED_SIZE_MB` | Uncompressed DOCX size cap | `50` |
| `MAX_EXTRACTED_TEXT_LENGTH` | Character limit on parsed text | `1000000` |
| `MAX_DOCUMENT_CHUNKS` | Maximum chunks per document | `1000` |

### 5. (Optional) Seed Sample Tenants
To seed sample legal and healthcare tenants with pre-indexed documents and test admin accounts:
```bash
uv run python seed.py
```

### 6. Launch the Server
Start the FastAPI server with Uvicorn:
```bash
uv run uvicorn backend.main:app --reload --port 8000
```
Interactive OpenAPI documentation will be available at `http://localhost:8000/docs`.

---

## 📖 Minimal API Usage Walkthrough

### 1. Register a Tenant & Admin User
Each registration creates a distinct tenant workspace:
```bash
curl -s -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "owner@tenant-alpha.demo",
    "password": "Password123!",
    "tenant_name": "Alpha Corp"
  }'
```
*Response*:
```json
{
  "access_token": "<TOKEN_ALPHA>",
  "token_type": "bearer",
  "user_id": "<USER_ID>",
  "tenant_id": "<TENANT_A_ID>",
  "role": "owner"
}
```

### 2. Ingest Documents (PDF, DOCX, TXT, Raw Text)

#### Multi-page PDF Upload:
```bash
curl -s -X POST "http://localhost:8000/documents/upload" \
  -H "Authorization: Bearer <TOKEN_ALPHA>" \
  -F "file=@handbook.pdf;type=application/pdf" \
  -F "title=Employee Handbook"
```

#### DOCX / TXT Upload:
```bash
curl -s -X POST "http://localhost:8000/documents/upload" \
  -H "Authorization: Bearer <TOKEN_ALPHA>" \
  -F "file=@guidelines.docx;type=application/vnd.openxmlformats-officedocument.wordprocessingml.document"

curl -s -X POST "http://localhost:8000/documents/upload" \
  -H "Authorization: Bearer <TOKEN_ALPHA>" \
  -F "file=@readme.txt;type=text/plain"
```

#### Raw JSON Text Ingestion:
```bash
curl -s -X POST "http://localhost:8000/documents" \
  -H "Authorization: Bearer <TOKEN_ALPHA>" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Remote Work Policy",
    "raw_text": "Remote employees are eligible for a $500 annual home office reimbursement."
  }'
```

### 3. Ask Questions via Chat
```bash
curl -s -X POST "http://localhost:8000/chat" \
  -H "Authorization: Bearer <TOKEN_ALPHA>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the equipment reimbursement for remote workers?"
  }'
```
*Successful Response*:
```json
{
  "conversation_id": "<CONVERSATION_ID>",
  "answer": "Remote employees are entitled to a $500 annual home office equipment reimbursement.",
  "sources": [
    {
      "document_title": "Employee Handbook",
      "chunk_text": "Remote employees are eligible for a $500 annual home office reimbursement...",
      "score": 0.88,
      "filename": "handbook.pdf",
      "page_number": 2
    }
  ],
  "is_degraded": false
}
```

### 4. Fetch Conversation History
Messages and source citations are persisted in MongoDB:
```bash
curl -s -X GET "http://localhost:8000/chat/conversations/<CONVERSATION_ID>/messages" \
  -H "Authorization: Bearer <TOKEN_ALPHA>"
```
*Response*:
```json
{
  "conversation_id": "<CONVERSATION_ID>",
  "messages": [
    {
      "role": "user",
      "content": "What is the equipment reimbursement for remote workers?",
      "sources": [],
      "is_degraded": false,
      "created_at": "2026-10-18T22:00:00Z"
    },
    {
      "role": "assistant",
      "content": "Remote employees are entitled to a $500 annual home office equipment reimbursement.",
      "sources": [
        {
          "document_title": "Employee Handbook",
          "chunk_text": "Remote employees are eligible for a $500 annual home office reimbursement...",
          "score": 0.88,
          "filename": "handbook.pdf",
          "page_number": 2
        }
      ],
      "is_degraded": false,
      "created_at": "2026-10-18T22:00:02Z"
    }
  ]
}
```

### 5. Multi-Tenant Cross-Tenant Enforcement
Attempting to fetch a document belonging to another tenant yields `403 Forbidden`:
```bash
curl -s -X GET "http://localhost:8000/documents/<OTHER_TENANT_DOC_ID>" \
  -H "Authorization: Bearer <TOKEN_ALPHA>"
```
```json
{
  "detail": "Access forbidden: Document belongs to a different tenant"
}
```

---

## 🔍 Metadata & Degraded Response Semantics

### Source Metadata Rules
- **PDF files**: `page_number` preserves the 1-based page index extracted from the PDF (`page_number: 1`, `page_number: 2`, etc.).
- **DOCX, TXT, and raw text**: `page_number` is always `null`.
- **Uploaded files**: `filename` reflects the uploaded filename (e.g., `handbook.pdf`).
- **Raw text documents**: `filename` is `null`.
- **Deduplication**: Chunks from the same document ID and chunk index are deduplicated. Identical text in distinct documents or on different PDF pages is retained as separate source entries.

### Graceful Degradation (`is_degraded`)
- If Gemini answer generation encounters an HTTP 429 quota exhaustion, network exception, or missing API key, the service safely fails closed:
  - `is_degraded: true`
  - `sources: []`
  - `answer`: `"Answer generation is temporarily unavailable. Please try again later."`
- The system will never return raw retrieved text dumps as pseudo-answers.
- If query rewriting fails but answer generation succeeds, `is_degraded: true` is returned to signal that conversational context may have been lost.

---

## 🧪 Automated Testing

The repository maintains an automated test suite executed with `pytest`:
```bash
# Run the complete test suite
uv run pytest -v

# Verify installed dependency compatibility
uv pip check
```

**Latest Test Run Results**:
- **Status**: 43 passed in ~66 seconds.
- **Dependency Status**: 88 packages checked; all compatible.

### Test Coverage Highlights
- **Retrieval Integrity & Orphan Discarding** (`tests/test_orphaned_vector_protection.py`): Validates that vectors referencing missing, cross-tenant, or malformed parent IDs are discarded and never leak into answers or sources.
- **Sanitized Exception Handling** (`tests/test_invitation_exception_disclosure.py`): Asserts database failures in both multi-document transaction and standalone compensation paths return generic HTTP 500 messages without disclosing internal database details, while preserving diagnostic logs.
- **Degraded Chat Mode** (`tests/test_degraded_chat_responses.py`): Tests HTTP 429 quota exhaustion, query rewrite degradation, and message history persistence.
- **Source Metadata & Deduplication** (`tests/test_chat_sources_metadata.py`, `tests/test_source_deduplication.py`): Proves 1-based page tracking, legacy payload support, and source deduplication.
- **Tenant Isolation & Security** (`tests/test_tenant_isolation.py`, `tests/test_auth_registration_security.py`): Proves cross-tenant access returns 403, tenant registration slugs remain isolated, and invitation race conditions resolve deterministically.
- **Resource Limits & Rollback** (`tests/test_document_upload.py`): Validates file size limits, page limits, uncompressed size guards, and compensating cleanup.

---

## ⚠️ Tested vs. Unverified Status

- **Ingestion, Chunking, Retrieval & Isolation**: Fully tested and verified through automated test suites (parsing, BGE-M3 local vector generation, Qdrant multi-tenant filtering, parent document validation, source mapping, and RBAC).
- **Live Gemini Answer Generation**: In live local end-to-end smoke testing, calls to the external Google Gemini API encountered free-tier HTTP 429 quota limits. Consequently, **live Gemini answer synthesis remains unverified against live external LLM endpoints**. In test suites, Gemini interactions are hermetically mocked.
- **No Grounding Guarantees**: This software does not guarantee factual correctness or hallucination-free generation from external LLM providers. Do not deploy to production without external quota provisioning and application-level evaluation.

---

## 📌 Known Limitations

1. **No OCR Processing**: Document parsing (`pypdf`, `python-docx`) extracts embedded text streams only. Scanned images or image-only PDF pages will yield empty text.
2. **Uncalibrated Similarity Threshold**: The default similarity threshold (`RAG_SIMILARITY_THRESHOLD=0.50`) is a heuristic baseline and has not been tuned or calibrated for domain-specific embedding distributions.
3. **Synchronous Ingestion**: File reading, text extraction, chunking, and BGE-M3 embedding generation occur synchronously within the FastAPI request lifecycle. Large files may cause request timeouts or memory spikes.
4. **Best-Effort Compensating Cleanup**: Indexing failures trigger independent compensating deletions on MongoDB and Qdrant. The system does not use a distributed two-phase commit (2PC); if Qdrant cleanup fails, an observable critical log is generated, but orphaned vectors must be filtered by the retrieval validation layer.
5. **Standalone MongoDB Invitation Recovery**: In standalone MongoDB instances (without replica sets), invitations use atomic status updates and compensating rollbacks. This presents a narrower recovery window during server crashes compared to replica-set multi-document transactions.
6. **No Public-Demo Abuse Controls**: The application currently lacks IP/tenant rate limiting (e.g., Redis leaky bucket), CAPTCHA guards on registration/invitations, or tenant token quota enforcement. It should not be exposed to the public Internet without an edge gateway or reverse proxy providing rate limiting and WAF capabilities.
