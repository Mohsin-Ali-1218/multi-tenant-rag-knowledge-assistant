import uuid
import logging
from unittest.mock import patch, MagicMock
import pytest
from bson import ObjectId
from httpx import AsyncClient, ASGITransport
from pymongo.errors import PyMongoError
from pymongo.asynchronous.collection import AsyncCollection

from backend.main import app
from backend.app.db.mongo import get_db, get_mongo_client
from backend.app.api.auth import INVITATION_ACCEPTANCE_ERROR_DETAIL

logger = logging.getLogger("multitenant_rag.auth")

async def setup_test_invitation(ac: AsyncClient, prefix: str):
    """Helper to create a tenant, owner, and pending invitation."""
    # Register owner
    res_owner = await ac.post("/auth/register", json={
        "email": f"owner_{prefix}@disclose-test.demo",
        "password": "Password123!",
        "tenant_name": f"Disclosure Org {prefix}"
    })
    assert res_owner.status_code == 200
    owner_data = res_owner.json()
    owner_headers = {"Authorization": f"Bearer {owner_data['access_token']}"}
    tenant_id = owner_data["tenant_id"]

    # Create invitation
    invite_email = f"invitee_{prefix}@disclose-test.demo"
    inv_res = await ac.post("/auth/invitations", headers=owner_headers, json={
        "email": invite_email,
        "role": "member"
    })
    assert inv_res.status_code == 201
    inv_data = inv_res.json()

    return {
        "tenant_id": tenant_id,
        "invitation_id": inv_data["id"],
        "invitation_token": inv_data["invitation_token"],
        "email": invite_email,
        "password": "Password123!"
    }

class MockTxSession:
    """Mock MongoDB async session that simulates transaction support."""
    def __init__(self):
        self.aborted = False
        self.committed = False
        self.in_transaction = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def start_transaction(self):
        self.in_transaction = True

    async def commit_transaction(self):
        self.committed = True
        self.in_transaction = False

    async def abort_transaction(self):
        self.aborted = True
        self.in_transaction = False

@pytest.mark.asyncio
async def test_invitation_acceptance_standalone_compensation_path_sanitizes_errors(caplog):
    """
    Standalone path: A database failure during user insertion must:
    1. Not leak the sensitive database error string or internals to the client.
    2. Return stable generic HTTP 500 error message.
    3. Retain diagnostic error and traceback in server logs with identifiers.
    4. Safely revert the invitation to 'pending' allowing retry.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        pfx = uuid.uuid4().hex[:6]
        inv_info = await setup_test_invitation(ac, pfx)

        sensitive_error_string = f"SENSITIVE_STANDALONE_MONGO_SOCKET_ERR_0x99_{pfx}_SECRET_CLUSTER_KEY"

        orig_insert_one = AsyncCollection.insert_one
        async def failing_insert_one(self, doc, *args, **kwargs):
            if "hashed_password" in doc and doc.get("email") == inv_info["email"]:
                raise PyMongoError(sensitive_error_string)
            return await orig_insert_one(self, doc, *args, **kwargs)

        with caplog.at_level(logging.ERROR, logger="multitenant_rag.auth"):
            with patch.object(AsyncCollection, "insert_one", failing_insert_one):
                res = await ac.post("/auth/invitations/accept", json={
                    "invitation_token": inv_info["invitation_token"],
                    "email": inv_info["email"],
                    "password": inv_info["password"]
                })

        # 1. Assert HTTP response status and stable generic message
        assert res.status_code == 500
        data = res.json()
        assert data["detail"] == INVITATION_ACCEPTANCE_ERROR_DETAIL

        # 2. Assert sensitive strings and internal database details are absent from response
        assert sensitive_error_string not in res.text
        assert "PyMongoError" not in res.text
        assert "Database error during user registration" not in res.text

        # 3. Assert server logs retain diagnostic error and identifiers
        assert sensitive_error_string in caplog.text
        assert inv_info["tenant_id"] in caplog.text
        assert inv_info["invitation_id"] in caplog.text
        assert inv_info["email"] in caplog.text

        # 4. Assert invitation rolled back to pending in MongoDB
        db = get_db()
        inv_doc = await db.invitations.find_one({"_id": ObjectId(inv_info["invitation_id"])})
        assert inv_doc["status"] == "pending"
        assert inv_doc["used_at"] is None

        # 5. Assert user can retry and succeed after database recovers
        retry_res = await ac.post("/auth/invitations/accept", json={
            "invitation_token": inv_info["invitation_token"],
            "email": inv_info["email"],
            "password": inv_info["password"]
        })
        assert retry_res.status_code == 200
        assert "access_token" in retry_res.json()

@pytest.mark.asyncio
async def test_invitation_acceptance_transaction_path_sanitizes_errors(caplog):
    """
    Transaction path: A failure during transaction execution must:
    1. Not leak the sensitive database error string or internals to the client.
    2. Return stable generic HTTP 500 error message.
    3. Retain diagnostic error and traceback in server logs with identifiers.
    4. Abort the transaction cleanly.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        pfx = uuid.uuid4().hex[:6]
        inv_info = await setup_test_invitation(ac, pfx)

        sensitive_error_string = f"SENSITIVE_TX_INTERNAL_CLUSTER_SHARD_CRASH_{pfx}_ROOT_KEY"

        mock_session = MockTxSession()
        client = get_mongo_client()

        orig_start_session = client.start_session
        def mocked_start_session(*args, **kwargs):
            return mock_session

        orig_find_one_and_update = AsyncCollection.find_one_and_update
        async def mock_tx_find_one_and_update(self, *args, **kwargs):
            if kwargs.get("session") == mock_session:
                return {
                    "_id": ObjectId(inv_info["invitation_id"]),
                    "tenant_id": inv_info["tenant_id"],
                    "email": inv_info["email"],
                    "role": "member",
                    "status": "accepted",
                    "used_at": "2026-10-18T00:00:00Z"
                }
            return await orig_find_one_and_update(self, *args, **kwargs)

        orig_insert_one = AsyncCollection.insert_one
        async def failing_insert_one(self, doc, *args, **kwargs):
            if "hashed_password" in doc and doc.get("email") == inv_info["email"]:
                raise PyMongoError(sensitive_error_string)
            return await orig_insert_one(self, doc, *args, **kwargs)

        with caplog.at_level(logging.ERROR, logger="multitenant_rag.auth"):
            with patch.object(client, "start_session", mocked_start_session):
                with patch.object(AsyncCollection, "find_one_and_update", mock_tx_find_one_and_update):
                    with patch.object(AsyncCollection, "insert_one", failing_insert_one):
                        res = await ac.post("/auth/invitations/accept", json={
                            "invitation_token": inv_info["invitation_token"],
                            "email": inv_info["email"],
                            "password": inv_info["password"]
                        })

        # 1. Assert HTTP response status and stable generic message
        assert res.status_code == 500
        data = res.json()
        assert data["detail"] == INVITATION_ACCEPTANCE_ERROR_DETAIL

        # 2. Assert sensitive strings and internal database details are absent from response
        assert sensitive_error_string not in res.text
        assert "PyMongoError" not in res.text
        assert "Transaction failure during invitation acceptance" not in res.text

        # 3. Assert server logs retain diagnostic error and identifiers
        assert sensitive_error_string in caplog.text
        assert inv_info["tenant_id"] in caplog.text
        assert inv_info["invitation_id"] in caplog.text
        assert inv_info["email"] in caplog.text

        # 4. Assert transaction was aborted
        assert mock_session.aborted is True

@pytest.mark.asyncio
async def test_invitation_acceptance_claim_failure_sanitizes_errors(caplog):
    """
    If the atomic claim operation itself fails in the database:
    1. HTTP response returns generic 500 message.
    2. Sensitive internal details do not leak to HTTP response.
    3. Server log retains diagnostic info.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        pfx = uuid.uuid4().hex[:6]
        inv_info = await setup_test_invitation(ac, pfx)

        sensitive_error_string = f"SENSITIVE_CLAIM_FAIL_CLUSTER_CORRUPT_{pfx}_KEY"

        orig_find_one_and_update = AsyncCollection.find_one_and_update
        async def failing_claim(self, *args, **kwargs):
            if kwargs.get("return_document") is not None and not kwargs.get("session"):
                raise PyMongoError(sensitive_error_string)
            return await orig_find_one_and_update(self, *args, **kwargs)

        with caplog.at_level(logging.ERROR, logger="multitenant_rag.auth"):
            with patch.object(AsyncCollection, "find_one_and_update", failing_claim):
                res = await ac.post("/auth/invitations/accept", json={
                    "invitation_token": inv_info["invitation_token"],
                    "email": inv_info["email"],
                    "password": inv_info["password"]
                })

        assert res.status_code == 500
        assert res.json()["detail"] == INVITATION_ACCEPTANCE_ERROR_DETAIL
        assert sensitive_error_string not in res.text
        assert sensitive_error_string in caplog.text
        assert inv_info["tenant_id"] in caplog.text

@pytest.mark.asyncio
async def test_invitation_acceptance_rollback_failure_logs_critical(caplog):
    """
    If the compensating rollback itself fails:
    1. HTTP response returns generic 500 message without leaking details.
    2. Server logs record a CRITICAL message with identifiers and the exception.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        pfx = uuid.uuid4().hex[:6]
        inv_info = await setup_test_invitation(ac, pfx)

        insert_err = f"PRIMARY_INSERT_FAIL_{pfx}"
        rollback_err = f"SENSITIVE_ROLLBACK_CRITICAL_FAIL_{pfx}_SECRET"

        orig_insert_one = AsyncCollection.insert_one
        async def failing_insert_one(self, doc, *args, **kwargs):
            if "hashed_password" in doc and doc.get("email") == inv_info["email"]:
                raise PyMongoError(insert_err)
            return await orig_insert_one(self, doc, *args, **kwargs)

        orig_update_one = AsyncCollection.update_one
        async def failing_update_one(self, filter_doc, update_doc, *args, **kwargs):
            if filter_doc.get("status") == "accepted":
                raise PyMongoError(rollback_err)
            return await orig_update_one(self, filter_doc, update_doc, *args, **kwargs)

        with caplog.at_level(logging.ERROR, logger="multitenant_rag.auth"):
            with patch.object(AsyncCollection, "insert_one", failing_insert_one):
                with patch.object(AsyncCollection, "update_one", failing_update_one):
                    res = await ac.post("/auth/invitations/accept", json={
                        "invitation_token": inv_info["invitation_token"],
                        "email": inv_info["email"],
                        "password": inv_info["password"]
                    })

        assert res.status_code == 500
        assert res.json()["detail"] == INVITATION_ACCEPTANCE_ERROR_DETAIL
        assert rollback_err not in res.text
        assert insert_err not in res.text

        # Both the original error and the critical rollback failure must be logged
        assert rollback_err in caplog.text
        assert "Compensating rollback failure" in caplog.text
        assert inv_info["invitation_id"] in caplog.text
        assert inv_info["tenant_id"] in caplog.text

