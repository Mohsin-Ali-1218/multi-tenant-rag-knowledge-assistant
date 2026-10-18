import uuid
import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app

@pytest.mark.asyncio
async def test_rag_grounded_answer_and_relevance():
    """
    Regression test suite for reliable grounded RAG behavior:
    1. Empty retrieval results:
       - When a tenant has no documents uploaded, questions are rejected with
         "I couldn't find this information in your documents." and zero citations.
    2. Relevant questions:
       - When a document contains the answer, the response is grounded and cites
         only the actual retrieved document.
    3. Irrelevant questions:
       - When the question cannot be answered from the tenant's documents,
         it is rejected with "I couldn't find this information in your documents."
         and zero citations (never inventing citations).
    4. Cross-tenant retrieval:
       - Tenant A cannot retrieve or cite Tenant B's documents.
       - Asking about Tenant B's content from Tenant A returns rejection with zero citations.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        prefix = uuid.uuid4().hex[:6]

        # =====================================================================
        # 1. Empty retrieval results (Tenant with no documents)
        # =====================================================================
        empty_email = f"empty_{prefix}@org.com"
        reg_empty = await ac.post("/auth/register", json={
            "email": empty_email,
            "password": "Password123!",
            "tenant_name": f"Empty Tenant {prefix}"
        })
        assert reg_empty.status_code == 200
        empty_headers = {"Authorization": f"Bearer {reg_empty.json()['access_token']}"}

        empty_chat_res = await ac.post("/chat", headers=empty_headers, json={
            "message": "What is our company policy on remote work stipends?"
        })
        assert empty_chat_res.status_code == 200
        empty_data = empty_chat_res.json()
        assert "I couldn't find this information in your documents." in empty_data["answer"]
        assert empty_data["sources"] == [], "Empty retrieval must return zero sources"

        # =====================================================================
        # 2. Relevant questions (Tenant Legal)
        # =====================================================================
        legal_email = f"legal_{prefix}@org.com"
        reg_legal = await ac.post("/auth/register", json={
            "email": legal_email,
            "password": "Password123!",
            "tenant_name": f"Legal Tenant {prefix}"
        })
        assert reg_legal.status_code == 200
        legal_headers = {"Authorization": f"Bearer {reg_legal.json()['access_token']}"}

        doc_legal_title = "Remote Work and Equipment Policy"
        doc_legal_text = (
            "Employees working remotely are entitled to a $500 home office equipment "
            "reimbursement after completing 90 days of continuous employment. Expense reports "
            "must be submitted with itemized receipts to HR."
        )
        up_legal = await ac.post("/documents", headers=legal_headers, json={
            "title": doc_legal_title,
            "raw_text": doc_legal_text
        })
        assert up_legal.status_code == 201

        # Relevant question
        rel_chat_res = await ac.post("/chat", headers=legal_headers, json={
            "message": "What is the equipment reimbursement amount for remote employees?"
        })
        assert rel_chat_res.status_code == 200
        rel_data = rel_chat_res.json()
        assert "$500" in rel_data["answer"] or "equipment reimbursement" in rel_data["answer"].lower()
        assert len(rel_data["sources"]) > 0, "Relevant query must cite retrieved source"
        assert rel_data["sources"][0]["document_title"] == doc_legal_title
        assert rel_data["sources"][0]["score"] >= 0.45

        # =====================================================================
        # 3. Irrelevant questions (Tenant Legal has only Remote Work policy)
        # =====================================================================
        irrel_chat_res = await ac.post("/chat", headers=legal_headers, json={
            "message": "What is the best recipe for baking chocolate chip cookies?"
        })
        assert irrel_chat_res.status_code == 200
        irrel_data = irrel_chat_res.json()
        assert "I couldn't find this information in your documents." in irrel_data["answer"]
        assert irrel_data["sources"] == [], "Irrelevant query must return zero sources"

        # Another irrelevant query
        unrelated_chat_res = await ac.post("/chat", headers=legal_headers, json={
            "message": "Who won the 1994 FIFA World Cup soccer championship?"
        })
        assert unrelated_chat_res.status_code == 200
        unrelated_data = unrelated_chat_res.json()
        assert "I couldn't find this information in your documents." in unrelated_data["answer"]
        assert unrelated_data["sources"] == [], "Unrelated query must return zero sources"

        # =====================================================================
        # 4. Cross-tenant retrieval
        # =====================================================================
        # Register Tenant Healthcare and upload medical privacy policy
        health_email = f"health_{prefix}@org.com"
        reg_health = await ac.post("/auth/register", json={
            "email": health_email,
            "password": "Password123!",
            "tenant_name": f"Health Tenant {prefix}"
        })
        assert reg_health.status_code == 200
        health_headers = {"Authorization": f"Bearer {reg_health.json()['access_token']}"}

        doc_health_title = "Patient Health Data Encryption Standard"
        doc_health_text = (
            "All patient medical records and electronic protected health information (ePHI) "
            "must be encrypted in transit and at rest using AES-256 standard encryption algorithms."
        )
        up_health = await ac.post("/documents", headers=health_headers, json={
            "title": doc_health_title,
            "raw_text": doc_health_text
        })
        assert up_health.status_code == 201

        # Tenant Legal asks about Tenant Healthcare's specific document topic
        cross_chat_res = await ac.post("/chat", headers=legal_headers, json={
            "message": "What encryption standard is required for patient medical records?"
        })
        assert cross_chat_res.status_code == 200
        cross_data = cross_chat_res.json()

        # Tenant Legal does NOT have health documents -> must reject with no leaked sources
        assert "I couldn't find this information in your documents." in cross_data["answer"]
        assert cross_data["sources"] == [], "Tenant Legal must not cite Tenant Healthcare documents"
        assert doc_health_title not in cross_chat_res.text

        # Tenant Healthcare asks the same question -> succeeds and cites only Healthcare document
        health_chat_res = await ac.post("/chat", headers=health_headers, json={
            "message": "What encryption standard is required for patient medical records?"
        })
        assert health_chat_res.status_code == 200
        health_data = health_chat_res.json()
        assert "AES-256" in health_data["answer"] or "encryption" in health_data["answer"].lower()
        assert len(health_data["sources"]) > 0
        assert health_data["sources"][0]["document_title"] == doc_health_title
        for src in health_data["sources"]:
            assert src["document_title"] != doc_legal_title
