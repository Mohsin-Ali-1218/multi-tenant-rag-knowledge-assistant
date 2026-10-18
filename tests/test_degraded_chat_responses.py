import uuid
from unittest.mock import MagicMock
import pytest
from httpx import AsyncClient, ASGITransport

from backend.main import app
from backend.app.db.mongo import get_db
from backend.app.services.llm import TEMPORARILY_UNAVAILABLE_MESSAGE, GROUNDED_REJECTION_MESSAGE

async def create_auth_tenant(ac: AsyncClient, prefix: str) -> tuple[dict, str]:
    res = await ac.post("/auth/register", json={
        "email": f"degrade_{prefix}@test.demo",
        "password": "Password123!",
        "tenant_name": f"Degraded Test {prefix}"
    })
    assert res.status_code == 200
    data = res.json()
    headers = {"Authorization": f"Bearer {data['access_token']}"}
    return headers, data["tenant_id"]

@pytest.mark.asyncio
async def test_generation_http_429_returns_degraded_response_and_zero_sources(monkeypatch):
    """
    When Gemini returns HTTP 429 during answer generation:
    - Return stable TEMPORARILY_UNAVAILABLE_MESSAGE
    - Return sources = []
    - Set is_degraded = True
    - Never return retrieved chunks as answers or sources, even when retrieval finds high-scoring hits.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        pfx = uuid.uuid4().hex[:6]
        headers, _ = await create_auth_tenant(ac, pfx)

        # Ingest a document with sensitive synthetic facts that would produce high-scoring hits
        secret_fact = "Project Phoenix flight plan: altitude 45000 feet, cruising speed Mach 0.85, heading 270 degrees."
        up_res = await ac.post("/documents", headers=headers, json={
            "title": "Phoenix Flight Plan",
            "raw_text": secret_fact
        })
        assert up_res.status_code == 201

        # Mock Gemini generate_content to raise HTTP 429
        mock_client = MagicMock()
        def fail_429(*args, **kwargs):
            raise Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for metric generate_content_free_tier_requests")

        mock_client.models.generate_content.side_effect = fail_429
        monkeypatch.setattr("backend.app.services.llm.get_genai_client", lambda: mock_client)

        chat_res = await ac.post("/chat", headers=headers, json={
            "message": "What is the Project Phoenix flight plan altitude and cruising speed?"
        })
        assert chat_res.status_code == 200
        data = chat_res.json()

        assert data["is_degraded"] is True
        assert data["answer"] == TEMPORARILY_UNAVAILABLE_MESSAGE
        assert data["sources"] == []

        # Verify no retrieved text was leaked into the answer
        assert "45000" not in data["answer"]
        assert "Phoenix" not in data["answer"]
        assert "Mach 0.85" not in data["answer"]
        assert "Quota exceeded" not in data["answer"]

@pytest.mark.asyncio
async def test_generation_generic_exception_returns_degraded_response(monkeypatch):
    """
    When Gemini answer generation fails with any unexpected error (e.g. network disconnect, timeout):
    - Return stable TEMPORARILY_UNAVAILABLE_MESSAGE
    - Return sources = []
    - Set is_degraded = True
    - Technical exception details must remain in server logs, never exposed to client.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        pfx = uuid.uuid4().hex[:6]
        headers, _ = await create_auth_tenant(ac, pfx)

        secret_err_token = "INTERNAL_RPC_GATEWAY_TIMEOUT_SOCKET_ERR_9981"
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError(secret_err_token)
        monkeypatch.setattr("backend.app.services.llm.get_genai_client", lambda: mock_client)

        chat_res = await ac.post("/chat", headers=headers, json={
            "message": "What is our company holiday policy?"
        })
        assert chat_res.status_code == 200
        data = chat_res.json()

        assert data["is_degraded"] is True
        assert data["answer"] == TEMPORARILY_UNAVAILABLE_MESSAGE
        assert data["sources"] == []
        assert secret_err_token not in chat_res.text

@pytest.mark.asyncio
async def test_rewrite_only_failure_marks_response_degraded_with_valid_answer(monkeypatch):
    """
    When query rewriting fails on a multi-turn conversation, but answer generation succeeds:
    - Mark is_degraded = True (because conversational retrieval context may have been lost)
    - Preserve the generated answer and cited sources.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        pfx = uuid.uuid4().hex[:6]
        headers, _ = await create_auth_tenant(ac, pfx)

        up_res = await ac.post("/documents", headers=headers, json={
            "title": "Hardware Guidelines",
            "raw_text": "Laptop hardware refresh interval is exactly 36 months for all engineering workstations."
        })
        assert up_res.status_code == 201

        # Turn 1: Normal initial interaction
        mock_client = MagicMock()
        def fake_turn1(*args, **kwargs):
            contents = str(kwargs.get("contents") or (args[0] if args else ""))
            if "Strict Grounding Rules:" in contents:
                return MagicMock(text="Workstations are replaced every 36 months.", function_calls=None)
            if contents.startswith("User Query:"):
                return MagicMock(text="Looking up knowledge base.", function_calls=None)
            return MagicMock(text="General answer.", function_calls=None)

        mock_client.models.generate_content.side_effect = fake_turn1
        monkeypatch.setattr("backend.app.services.llm.get_genai_client", lambda: mock_client)

        res_turn1 = await ac.post("/chat", headers=headers, json={
            "message": "What is the laptop refresh interval?"
        })
        assert res_turn1.status_code == 200
        data1 = res_turn1.json()
        conv_id = data1["conversation_id"]
        assert data1["is_degraded"] is False

        # Turn 2: Query rewriting raises an exception, but run_rag_agent succeeds
        def fake_turn2(*args, **kwargs):
            contents = str(kwargs.get("contents") or (args[0] if args else ""))
            if "Given the following conversation history" in contents:
                # Query rewrite failure!
                raise RuntimeError("Query rewrite network error")
            if "Strict Grounding Rules:" in contents:
                return MagicMock(text="Workstations are replaced every 36 months.", function_calls=None)
            if contents.startswith("User Query:"):
                return MagicMock(text="Looking up knowledge base.", function_calls=None)
            return MagicMock(text="Grounded answer.", function_calls=None)

        mock_client.models.generate_content.side_effect = fake_turn2

        res_turn2 = await ac.post("/chat", headers=headers, json={
            "conversation_id": conv_id,
            "message": "What is the hardware refresh interval?"
        })
        assert res_turn2.status_code == 200
        data2 = res_turn2.json()

        # Must be marked degraded because rewriting failed
        assert data2["is_degraded"] is True
        # But answer generation itself succeeded!
        assert "36 months" in data2["answer"]
        assert len(data2["sources"]) > 0

@pytest.mark.asyncio
async def test_normal_success_and_grounded_refusal_are_not_degraded(monkeypatch):
    """
    Verify normal operation is not marked degraded:
    - Normal successful grounded synthesis: is_degraded = False
    - Normal grounded refusal (query cannot be answered from documents): is_degraded = False
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        pfx = uuid.uuid4().hex[:6]
        headers, _ = await create_auth_tenant(ac, pfx)

        await ac.post("/documents", headers=headers, json={
            "title": "Office Protocol",
            "raw_text": "Badges must be worn visibly at all times within company facilities."
        })

        mock_client = MagicMock()
        def fake_normal(*args, **kwargs):
            contents = str(kwargs.get("contents") or (args[0] if args else ""))
            if "Strict Grounding Rules:" in contents:
                if "Badges must be worn" in contents:
                    return MagicMock(text="Badges must be worn visibly at all times.", function_calls=None)
                return MagicMock(text=GROUNDED_REJECTION_MESSAGE, function_calls=None)
            if contents.startswith("User Query:"):
                return MagicMock(text="Search", function_calls=None)
            return MagicMock(text="OK", function_calls=None)

        mock_client.models.generate_content.side_effect = fake_normal
        monkeypatch.setattr("backend.app.services.llm.get_genai_client", lambda: mock_client)

        # Case A: Normal success
        res_success = await ac.post("/chat", headers=headers, json={"message": "What is the badge policy?"})
        assert res_success.status_code == 200
        data_s = res_success.json()
        assert data_s["is_degraded"] is False
        assert "Badges must be worn" in data_s["answer"]
        assert len(data_s["sources"]) > 0

        # Case B: Normal grounded refusal
        res_refusal = await ac.post("/chat", headers=headers, json={"message": "What is the secret recipe for tiramisu?"})
        assert res_refusal.status_code == 200
        data_r = res_refusal.json()
        assert data_r["is_degraded"] is False
        assert data_r["answer"] == GROUNDED_REJECTION_MESSAGE
        assert data_r["sources"] == []

@pytest.mark.asyncio
async def test_conversation_history_persistence_and_backward_compatibility(monkeypatch):
    """
    Verify is_degraded is persisted and older records without is_degraded default to False:
    - Direct MongoDB insert of legacy message record (no is_degraded field)
    - Querying /conversations/{id}/messages returns is_degraded = False for legacy record
    - New messages correctly reflect their is_degraded state in conversation history
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        pfx = uuid.uuid4().hex[:6]
        headers, tenant_id = await create_auth_tenant(ac, pfx)

        # 1. Start a conversation
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(text="Hello", function_calls=None)
        monkeypatch.setattr("backend.app.services.llm.get_genai_client", lambda: mock_client)

        res1 = await ac.post("/chat", headers=headers, json={"message": "Hello"})
        assert res1.status_code == 200
        conv_id = res1.json()["conversation_id"]

        db = get_db()
        # 2. Insert a simulated legacy message record without 'is_degraded' field
        legacy_doc = {
            "conversation_id": conv_id,
            "tenant_id": tenant_id,
            "role": "assistant",
            "content": "This is a legacy assistant message from before is_degraded existed.",
            "sources": [],
            "created_at": "2025-01-01T00:00:00Z"
        }
        await db.messages.insert_one(legacy_doc)

        # 3. Simulate a degraded message in the same conversation
        mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED")
        res_degraded = await ac.post("/chat", headers=headers, json={
            "conversation_id": conv_id,
            "message": "A query that encounters 429"
        })
        assert res_degraded.status_code == 200
        assert res_degraded.json()["is_degraded"] is True

        # 4. Fetch conversation history
        res_hist = await ac.get(f"/conversations/{conv_id}/messages", headers=headers)
        assert res_hist.status_code == 200
        msgs = res_hist.json()["messages"]

        asst_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert len(asst_msgs) == 3

        # First message was normal
        assert asst_msgs[0]["is_degraded"] is False

        # Second message was legacy (no is_degraded in DB -> defaults to False)
        legacy_msg = [m for m in asst_msgs if "legacy assistant message" in m["content"]][0]
        assert legacy_msg["is_degraded"] is False

        # Third message was degraded (encountering 429 -> is_degraded: True)
        assert asst_msgs[2]["is_degraded"] is True
        assert asst_msgs[2]["answer"] if "answer" in asst_msgs[2] else asst_msgs[2]["content"] == TEMPORARILY_UNAVAILABLE_MESSAGE
