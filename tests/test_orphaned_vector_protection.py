import uuid
from unittest.mock import MagicMock
import pytest
from bson import ObjectId
from httpx import AsyncClient, ASGITransport
from pymongo.errors import PyMongoError
from qdrant_client.models import PointStruct

from backend.main import app
from backend.app.db.mongo import get_db
from backend.app.db.vector import (
    get_qdrant_client,
    get_tenant_collection_name,
    ensure_tenant_collection,
    search_tenant_docs,
    validate_parent_documents
)
from backend.app.services.embedding import get_embedding
from backend.app.services.llm import TEMPORARILY_UNAVAILABLE_MESSAGE, GROUNDED_REJECTION_MESSAGE

async def create_auth_tenant(ac: AsyncClient, prefix: str) -> tuple[dict, str]:
    res = await ac.post("/auth/register", json={
        "email": f"orphan_{prefix}@test.demo",
        "password": "Password123!",
        "tenant_name": f"Orphan Test {prefix}"
    })
    assert res.status_code == 200
    data = res.json()
    headers = {"Authorization": f"Bearer {data['access_token']}"}
    return headers, data["tenant_id"]

@pytest.fixture(autouse=True)
def mock_llm_calls(monkeypatch):
    """Hermetic mock for LLM generation to prevent external network/quota reliance."""
    mock_client = MagicMock()

    def fake_generate_content(*args, **kwargs):
        contents = str(kwargs.get("contents") or (args[0] if args else ""))
        if "Strict Grounding Rules:" in contents:
            if "Valid Document Alpha" in contents or "quantum cryptography" in contents:
                return MagicMock(text="Quantum cryptography keys are rotated daily.", function_calls=None)
            return MagicMock(text=GROUNDED_REJECTION_MESSAGE, function_calls=None)
        if contents.startswith("User Query:"):
            return MagicMock(text="Search", function_calls=None)
        return MagicMock(text="OK", function_calls=None)

    mock_client.models.generate_content.side_effect = fake_generate_content
    monkeypatch.setattr("backend.app.services.llm.get_genai_client", lambda: mock_client)

@pytest.mark.asyncio
async def test_valid_hits_preserved_with_score_metadata_and_order():
    """Valid hits with existing parent documents preserve score, metadata, and retrieval order."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        pfx = uuid.uuid4().hex[:6]
        headers, tenant_id = await create_auth_tenant(ac, pfx)

        up_res = await ac.post("/documents", headers=headers, json={
            "title": "Valid Document Alpha",
            "raw_text": "Quantum cryptography keys are rotated daily at midnight UTC."
        })
        assert up_res.status_code == 201
        doc_id = up_res.json()["id"]

        q_vec = get_embedding("Quantum cryptography keys rotation schedule")
        hits = await search_tenant_docs(tenant_id=tenant_id, query_vector=q_vec, limit=3, score_threshold=0.3)

        assert len(hits) >= 1
        valid_hit = hits[0]
        assert valid_hit["document_id"] == doc_id
        assert valid_hit["document_title"] == "Valid Document Alpha"
        assert "rotated daily" in valid_hit["chunk_text"]
        assert valid_hit["score"] >= 0.3
        assert valid_hit["page_number"] is None

@pytest.mark.asyncio
async def test_orphaned_vectors_after_indexing_cleanup_failure_discarded():
    """
    Simulate orphaned vectors (e.g. Qdrant cleanup failed after indexing rollback).
    Vectors must be discarded and never reach generated answers or sources.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        pfx = uuid.uuid4().hex[:6]
        headers, tenant_id = await create_auth_tenant(ac, pfx)

        # Inject an orphaned vector directly into Qdrant for this tenant
        client = get_qdrant_client()
        coll_name = get_tenant_collection_name(tenant_id)
        q_vec = get_embedding("Top secret ghost project blueprint details")
        await ensure_tenant_collection(tenant_id, vector_size=len(q_vec))

        phantom_doc_id = str(ObjectId())  # Does NOT exist in MongoDB
        secret_ghost_text = "Top secret ghost project: reactor core operational frequency is 432.8 MHz."
        await client.upsert(
            collection_name=coll_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=q_vec,
                    payload={
                        "document_id": phantom_doc_id,
                        "tenant_id": tenant_id,
                        "document_title": "Phantom Document",
                        "filename": "phantom.pdf",
                        "page_number": 1,
                        "text": secret_ghost_text,
                        "chunk_index": 0
                    }
                )
            ]
        )

        # 1. Direct retrieval must discard the orphaned vector
        hits = await search_tenant_docs(tenant_id=tenant_id, query_vector=q_vec, limit=5, score_threshold=0.3)
        assert len(hits) == 0, "Orphaned vector hit should have been discarded!"

        # 2. Chat endpoint must refuse and never leak orphaned text
        chat_res = await ac.post("/chat", headers=headers, json={
            "message": "What is the reactor core operational frequency of the ghost project?"
        })
        assert chat_res.status_code == 200
        data = chat_res.json()
        assert data["sources"] == []
        assert "432.8" not in data["answer"]
        assert "reactor core" not in data["answer"]
        assert data["answer"] == GROUNDED_REJECTION_MESSAGE

@pytest.mark.asyncio
async def test_malformed_document_ids_in_vector_payload_discarded():
    """Payloads with non-ObjectId, empty, or None document_ids must be safely discarded without crashing."""
    pfx = uuid.uuid4().hex[:6]
    tenant_id = f"malformed_test_{pfx}"

    malformed_hits = [
        {"document_id": "not-an-objectid-123", "document_title": "Bad ID 1", "chunk_text": "text1", "score": 0.9},
        {"document_id": "", "document_title": "Empty ID", "chunk_text": "text2", "score": 0.85},
        {"document_id": None, "document_title": "None ID", "chunk_text": "text3", "score": 0.8},
        {"document_id": 12345, "document_title": "Integer ID", "chunk_text": "text4", "score": 0.75},
    ]

    validated = await validate_parent_documents(tenant_id=tenant_id, hits=malformed_hits)
    assert validated == [], "All malformed hits must be safely discarded"

@pytest.mark.asyncio
async def test_cross_tenant_parent_records_discarded():
    """
    If a vector payload contains a document_id belonging to a DIFFERENT tenant,
    it must be discarded as unauthorized.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        pfx_a = uuid.uuid4().hex[:6]
        headers_a, tenant_a = await create_auth_tenant(ac, pfx_a)

        pfx_b = uuid.uuid4().hex[:6]
        headers_b, tenant_b = await create_auth_tenant(ac, pfx_b)

        # Tenant A uploads a document
        up_res = await ac.post("/documents", headers=headers_a, json={
            "title": "Tenant A Financial Audit",
            "raw_text": "Tenant A Q4 revenue reached 12.5 million dollars."
        })
        assert up_res.status_code == 201
        doc_a_id = up_res.json()["id"]

        # Inject a vector into Tenant B's Qdrant collection pointing to Tenant A's document_id
        client = get_qdrant_client()
        coll_b = get_tenant_collection_name(tenant_b)
        q_vec = get_embedding("Tenant A Q4 revenue figures")
        await ensure_tenant_collection(tenant_b, vector_size=len(q_vec))

        await client.upsert(
            collection_name=coll_b,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=q_vec,
                    payload={
                        "document_id": doc_a_id,  # Belongs to Tenant A!
                        "tenant_id": tenant_b,
                        "document_title": "Cross Tenant Audit",
                        "text": "Tenant A Q4 revenue reached 12.5 million dollars.",
                        "chunk_index": 0
                    }
                )
            ]
        )

        # Tenant B search must reject the hit because parent document belongs to Tenant A
        hits = await search_tenant_docs(tenant_id=tenant_b, query_vector=q_vec, limit=5, score_threshold=0.3)
        assert len(hits) == 0, "Cross-tenant parent document ID must be discarded!"

        # Tenant B chat must not cite or answer with Tenant A's data
        chat_res = await ac.post("/chat", headers=headers_b, json={
            "message": "What was the Q4 revenue?"
        })
        assert chat_res.status_code == 200
        data = chat_res.json()
        assert data["sources"] == []
        assert "12.5 million" not in data["answer"]
        assert data["answer"] == GROUNDED_REJECTION_MESSAGE

@pytest.mark.asyncio
async def test_mixed_valid_and_orphaned_results_preserves_valid_and_discards_orphaned():
    """
    When retrieval returns both valid and orphaned points:
    - Orphaned points are discarded.
    - Valid points preserve their relative rank, score, and metadata.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        pfx = uuid.uuid4().hex[:6]
        headers, tenant_id = await create_auth_tenant(ac, pfx)

        # Upload Doc 1 (Valid)
        up1 = await ac.post("/documents", headers=headers, json={
            "title": "Doc 1 Valid",
            "raw_text": "Network security guideline: SSH access is restricted to port 2222."
        })
        assert up1.status_code == 201
        doc1_id = up1.json()["id"]

        # Upload Doc 2 (Will become orphan)
        up2 = await ac.post("/documents", headers=headers, json={
            "title": "Doc 2 To Be Orphaned",
            "raw_text": "Legacy database backup password is SuperSecretAdmin123."
        })
        assert up2.status_code == 201
        doc2_id = up2.json()["id"]

        # Directly delete Doc 2 from MongoDB only (simulating an orphaned vector)
        db = get_db()
        del_res = await db.documents.delete_one({"_id": ObjectId(doc2_id)})
        assert del_res.deleted_count == 1

        # Query matching both
        q_vec = get_embedding("security guidelines and backup password")
        hits = await search_tenant_docs(tenant_id=tenant_id, query_vector=q_vec, limit=5, score_threshold=0.2)

        # Assert Doc 1 is present and Doc 2 is discarded
        assert len(hits) >= 1
        retrieved_doc_ids = [h["document_id"] for h in hits]
        assert doc1_id in retrieved_doc_ids
        assert doc2_id not in retrieved_doc_ids, "Orphaned Doc 2 must be discarded!"

        for h in hits:
            assert "SuperSecretAdmin123" not in h["chunk_text"]

@pytest.mark.asyncio
async def test_mongodb_lookup_failure_fails_closed_without_leaking_details(monkeypatch):
    """
    If MongoDB validation query fails (e.g. database timeout / connection error):
    - Fails closed immediately (never returns unvalidated hits)
    - Chat response returns degraded unavailable message with zero sources
    - Internal database error details are NOT disclosed in the HTTP response.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        pfx = uuid.uuid4().hex[:6]
        headers, tenant_id = await create_auth_tenant(ac, pfx)

        await ac.post("/documents", headers=headers, json={
            "title": "Sample Doc",
            "raw_text": "Server maintenance window is every Saturday at 02:00 AM."
        })

        secret_db_error = "FATAL_MONGODB_SOCKET_TIMEOUT_CLUSTER_SPLIT_0x992B"

        # Mock MongoDB find in vector module to raise a PyMongoError
        mock_db = MagicMock()
        mock_db.documents.find.side_effect = PyMongoError(secret_db_error)
        monkeypatch.setattr("backend.app.db.vector.get_db", lambda: mock_db)

        # 1. Direct call to validate_parent_documents must raise RuntimeError and fail closed
        with pytest.raises(RuntimeError, match="Database validation failed for document hits"):
            await validate_parent_documents(
                tenant_id=tenant_id,
                hits=[{"document_id": str(ObjectId()), "document_title": "T", "chunk_text": "C", "score": 0.9}]
            )

        # 2. Chat endpoint must fail closed, return degraded unavailable message, zero sources, and no error leak
        chat_res = await ac.post("/chat", headers=headers, json={
            "message": "When is the server maintenance window?"
        })
        assert chat_res.status_code == 200
        data = chat_res.json()

        assert data["is_degraded"] is True
        assert data["answer"] == TEMPORARILY_UNAVAILABLE_MESSAGE
        assert data["sources"] == []
        assert secret_db_error not in chat_res.text
