import uuid
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from httpx import AsyncClient, ASGITransport

from backend.main import app
from backend.app.services.llm import deduplicate_sources

def test_identical_content_from_two_distinct_document_ids_retains_both_sources():
    """
    Identical content from two distinct document IDs must retain both sources,
    even when title, chunk text, filename, and page number match.
    """
    doc_1_src = {
        "document_id": "doc_uuid_alpha_001",
        "chunk_index": 0,
        "document_title": "Compliance Manual",
        "chunk_text": "Mandatory security training must be completed annually by all staff.",
        "filename": "compliance_manual.pdf",
        "page_number": 1,
        "score": 0.92,
    }
    doc_2_src = {
        "document_id": "doc_uuid_beta_002",
        "chunk_index": 0,
        "document_title": "Compliance Manual",
        "chunk_text": "Mandatory security training must be completed annually by all staff.",
        "filename": "compliance_manual.pdf",
        "page_number": 1,
        "score": 0.89,
    }

    result = deduplicate_sources([doc_1_src, doc_2_src])
    assert len(result) == 2, "Both distinct document IDs must be retained"
    # Preserves retrieval order
    assert result[0]["score"] == 0.92
    assert result[1]["score"] == 0.89
    # Output schema retains expected fields without exposing internal identifiers
    assert "document_id" not in result[0]
    assert "chunk_index" not in result[0]
    assert result[0]["filename"] == "compliance_manual.pdf"
    assert result[0]["page_number"] == 1
    assert result[1]["filename"] == "compliance_manual.pdf"
    assert result[1]["page_number"] == 1

def test_repeated_retrieval_of_same_document_chunk_produces_one_source():
    """
    Repeated retrieval of the exact same document chunk produces one source.
    """
    first_hit = {
        "document_id": "doc_uuid_gamma_003",
        "chunk_index": 2,
        "document_title": "Operational Guidelines",
        "chunk_text": "Servers in the primary cluster require dual power supplies.",
        "filename": "ops_guidelines.docx",
        "page_number": None,
        "score": 0.94,
    }
    duplicate_hit = {
        "document_id": "doc_uuid_gamma_003",
        "chunk_index": 2,
        "document_title": "Operational Guidelines",
        "chunk_text": "Servers in the primary cluster require dual power supplies.",
        "filename": "ops_guidelines.docx",
        "page_number": None,
        "score": 0.91,
    }

    result = deduplicate_sources([first_hit, duplicate_hit])
    assert len(result) == 1, "Duplicate retrieval of the same chunk must collapse to one source"
    assert result[0]["score"] == 0.94
    assert result[0]["document_title"] == "Operational Guidelines"
    assert result[0]["filename"] == "ops_guidelines.docx"
    assert result[0]["page_number"] is None

def test_legacy_sources_with_different_filenames_remain_separate():
    """
    Legacy sources lacking document_id must remain separate if their filenames differ.
    """
    legacy_file_1 = {
        "document_title": "Server Handbook",
        "chunk_text": "Default SSH port is configured as 2222 across all bastion hosts.",
        "filename": "server_handbook_v1.txt",
        "page_number": None,
        "score": 0.85,
    }
    legacy_file_2 = {
        "document_title": "Server Handbook",
        "chunk_text": "Default SSH port is configured as 2222 across all bastion hosts.",
        "filename": "server_handbook_v2.txt",
        "page_number": None,
        "score": 0.82,
    }

    result = deduplicate_sources([legacy_file_1, legacy_file_2])
    assert len(result) == 2, "Legacy sources with different filenames must remain separate"
    assert result[0]["filename"] == "server_handbook_v1.txt"
    assert result[1]["filename"] == "server_handbook_v2.txt"

    # Verifies identical legacy sources with the same filename collapse
    duplicate_legacy = {
        "document_title": "Server Handbook",
        "chunk_text": "Default SSH port is configured as 2222 across all bastion hosts.",
        "filename": "server_handbook_v1.txt",
        "page_number": None,
        "score": 0.79,
    }
    collapsed = deduplicate_sources([legacy_file_1, duplicate_legacy])
    assert len(collapsed) == 1, "Identical legacy sources with same filename must collapse to one"

def test_identical_text_on_different_pdf_pages_remains_separate():
    """
    Identical text appearing on different pages of a PDF must remain separate.
    """
    # 1. With document_id and chunk_index (modern Qdrant payload)
    page_1_chunk = {
        "document_id": "doc_pdf_delta_004",
        "chunk_index": 0,
        "document_title": "Quarterly Financial Report",
        "chunk_text": "Confidential - Proprietary and Restricted Distribution",
        "filename": "q3_report.pdf",
        "page_number": 1,
        "score": 0.88,
    }
    page_2_chunk = {
        "document_id": "doc_pdf_delta_004",
        "chunk_index": 4,
        "document_title": "Quarterly Financial Report",
        "chunk_text": "Confidential - Proprietary and Restricted Distribution",
        "filename": "q3_report.pdf",
        "page_number": 2,
        "score": 0.87,
    }

    result_modern = deduplicate_sources([page_1_chunk, page_2_chunk])
    assert len(result_modern) == 2, "Identical text on distinct PDF pages must remain separate"
    assert result_modern[0]["page_number"] == 1
    assert result_modern[1]["page_number"] == 2

    # 2. Without document_id (legacy fallback)
    legacy_page_1 = {
        "document_title": "Quarterly Financial Report",
        "chunk_text": "Confidential - Proprietary and Restricted Distribution",
        "filename": "q3_report.pdf",
        "page_number": 1,
        "score": 0.88,
    }
    legacy_page_2 = {
        "document_title": "Quarterly Financial Report",
        "chunk_text": "Confidential - Proprietary and Restricted Distribution",
        "filename": "q3_report.pdf",
        "page_number": 2,
        "score": 0.87,
    }

    result_legacy = deduplicate_sources([legacy_page_1, legacy_page_2])
    assert len(result_legacy) == 2, "Legacy identical text on distinct PDF pages must remain separate"
    assert result_legacy[0]["page_number"] == 1
    assert result_legacy[1]["page_number"] == 2

@pytest.mark.asyncio
async def test_chat_e2e_two_distinct_documents_with_identical_content(monkeypatch):
    """
    End-to-end integration test through POST /chat verifying that when two distinct
    uploaded documents contain identical content, both are preserved in chat sources.
    """
    monkeypatch.setenv("RAG_SIMILARITY_THRESHOLD", "0.40")
    mock_client = MagicMock()

    def fake_generate_content(*args, **kwargs):
        contents = kwargs.get("contents") or (args[0] if args else "")
        contents_str = str(contents)
        if "User Query:" in contents_str:
            return MagicMock(text="Analyzing...", function_calls=None)
        if "Strict Grounding Rules:" in contents_str:
            return MagicMock(text="Based on company guidelines, the evacuation assembly point is Zone 4.", function_calls=None)
        return MagicMock(text="Grounded answer.", function_calls=None)

    mock_client.models.generate_content.side_effect = fake_generate_content
    monkeypatch.setattr("backend.app.services.llm.get_genai_client", lambda: mock_client)
    monkeypatch.setattr("backend.app.services.llm.rewrite_query", AsyncMock(side_effect=lambda history, msg: msg))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        pfx = uuid.uuid4().hex[:6]
        # Register test tenant
        reg = await ac.post("/auth/register", json={
            "email": f"dedup_e2e_{pfx}@tenant.com",
            "password": "password123",
            "tenant_name": f"Dedup Tenant {pfx}"
        })
        assert reg.status_code == 200
        headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

        identical_text = "Emergency evacuation procedure: all personnel must assemble at designated Zone 4."

        # Upload Document 1
        res_doc1 = await ac.post(
            "/documents/upload",
            headers=headers,
            files={"file": ("safety_v1.txt", identical_text.encode("utf-8"), "text/plain")},
            data={"title": "Facility Safety V1"}
        )
        assert res_doc1.status_code == 201
        doc1_id = res_doc1.json()["id"]

        # Upload Document 2 with identical text but different document identity and title
        res_doc2 = await ac.post(
            "/documents/upload",
            headers=headers,
            files={"file": ("safety_v2.txt", identical_text.encode("utf-8"), "text/plain")},
            data={"title": "Facility Safety V2"}
        )
        assert res_doc2.status_code == 201
        doc2_id = res_doc2.json()["id"]
        assert doc1_id != doc2_id

        # Query chat
        res_chat = await ac.post("/chat", headers=headers, json={
            "message": "Where must personnel assemble during an emergency evacuation procedure?"
        })
        assert res_chat.status_code == 200
        data_chat = res_chat.json()
        sources = data_chat["sources"]

        # Both distinct documents must be cited in sources
        assert len(sources) >= 2, f"Expected both documents in sources, got {len(sources)}"
        titles = [s["document_title"] for s in sources]
        assert "Facility Safety V1" in titles
        assert "Facility Safety V2" in titles

        # Verify saved history also preserves both sources
        conv_id = data_chat["conversation_id"]
        res_hist = await ac.get(f"/conversations/{conv_id}/messages", headers=headers)
        assert res_hist.status_code == 200
        asst_msg = [m for m in res_hist.json()["messages"] if m["role"] == "assistant"][0]
        hist_titles = [s["document_title"] for s in asst_msg["sources"]]
        assert "Facility Safety V1" in hist_titles
        assert "Facility Safety V2" in hist_titles
