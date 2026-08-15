import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app

@pytest.mark.asyncio
async def test_tenant_isolation_and_403_enforcement():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        # Step 1: Login or Register Tenant A (Acme Legal)
        res_a = await ac.post("/auth/login", json={
            "email": "admin@acmelegal.com",
            "password": "password123"
        })
        if res_a.status_code == 401:
            res_a = await ac.post("/auth/register", json={
                "email": "admin@acmelegal.com",
                "password": "password123",
                "tenant_name": "Acme Legal"
            })
        assert res_a.status_code == 200, f"Tenant A auth failed: {res_a.text}"
        data_a = res_a.json()
        token_a = data_a["access_token"]
        headers_a = {"Authorization": f"Bearer {token_a}"}

        # Step 2: Login or Register Tenant B (Acme Healthcare)
        res_b = await ac.post("/auth/login", json={
            "email": "admin@acmehealthcare.com",
            "password": "password123"
        })
        if res_b.status_code == 401:
            res_b = await ac.post("/auth/register", json={
                "email": "admin@acmehealthcare.com",
                "password": "password123",
                "tenant_name": "Acme Healthcare"
            })
        assert res_b.status_code == 200, f"Tenant B auth failed: {res_b.text}"
        data_b = res_b.json()
        token_b = data_b["access_token"]
        headers_b = {"Authorization": f"Bearer {token_b}"}

        # Verify tenant IDs are different
        assert data_a["tenant_id"] != data_b["tenant_id"], "Tenants must have different tenant_ids"

        # Step 3: Fetch or Create Tenant B's documents
        docs_b_res = await ac.get("/documents", headers=headers_b)
        assert docs_b_res.status_code == 200
        docs_b = docs_b_res.json()
        if not docs_b:
            up_b = await ac.post("/documents", headers=headers_b, json={
                "title": "HIPAA Compliance Overview",
                "raw_text": "Acme Healthcare requires all patient health records to be encrypted at rest."
            })
            assert up_b.status_code == 201
            doc_b_id = up_b.json()["id"]
            doc_b_title = up_b.json()["title"]
            doc_b_raw_text = up_b.json()["raw_text"]
        else:
            doc_b_id = docs_b[0]["id"]
            doc_b_title = docs_b[0]["title"]
            doc_b_raw_text = docs_b[0]["raw_text"]

        # Step 4: Tenant A asks a question answerable only from Tenant A's docs
        chat_a_res = await ac.post("/chat", headers=headers_a, json={
            "message": "What is the non-disclosure agreement duration for client engagements?"
        })
        assert chat_a_res.status_code == 200
        chat_a_data = chat_a_res.json()
        assert "answer" in chat_a_data
        
        # Verify no Tenant B sources were leaked in RAG response
        for src in chat_a_data.get("sources", []):
            assert doc_b_title not in src["document_title"], "Tenant A chat leaked Tenant B document!"

        # Step 5: Attempt cross-tenant document retrieval (Tenant A -> Tenant B document ID)
        forbidden_res = await ac.get(f"/documents/{doc_b_id}", headers=headers_a)

        # MUST return 403 Forbidden with 0 data leaked
        assert forbidden_res.status_code == 403, f"Expected 403 Forbidden, got {forbidden_res.status_code}"
        assert doc_b_title not in forbidden_res.text, "Leaked Tenant B document title in 403 response!"
        assert doc_b_raw_text not in forbidden_res.text, "Leaked Tenant B raw text in 403 response!"
        
        forbidden_json = forbidden_res.json()
        assert "detail" in forbidden_json
        assert "forbidden" in forbidden_json["detail"].lower()
