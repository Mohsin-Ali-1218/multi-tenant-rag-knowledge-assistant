import uuid
import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app

@pytest.mark.asyncio
async def test_conversation_ownership_and_user_privacy():
    """
    Regression test for conversation ownership and privacy within and across tenants:
    1. Authorized conversation owner access:
       - User 1 (Alice) creates a conversation and posts messages.
       - Alice can view messages (GET /conversations/{id}/messages).
       - Alice can continue the conversation (POST /chat with conversation_id).
    2. Same-tenant, different-user access:
       - User 2 (Bob) belongs to the same tenant as Alice.
       - Bob attempts to read Alice's conversation messages -> 403 Forbidden.
       - Bob attempts to post into Alice's conversation -> 403 Forbidden.
       - Bob attempts to delete Alice's conversation -> 403 Forbidden.
    3. Cross-tenant conversation access:
       - User 3 (Charlie) belongs to a different tenant.
       - Charlie attempts to read Alice's conversation messages -> 403 Forbidden.
       - Charlie attempts to post into Alice's conversation -> 403 Forbidden.
       - Charlie attempts to delete Alice's conversation -> 403 Forbidden.
    4. Deletion lifecycle & invalid IDs:
       - Alice deletes her conversation (DELETE /conversations/{id}) -> 204 No Content.
       - Subsequent access to deleted conversation returns 404 Not Found.
       - Invalid conversation ID format returns 404 Not Found.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        prefix = uuid.uuid4().hex[:6]

        # -------------------------------------------------------------
        # Setup Tenant A with Owner (Alice) and Member (Bob)
        # -------------------------------------------------------------
        # 1. Register Alice in Tenant A
        alice_email = f"alice_{prefix}@tenanta.com"
        reg_a = await ac.post("/auth/register", json={
            "email": alice_email,
            "password": "PasswordAlice123!",
            "tenant_name": f"Tenant A {prefix}"
        })
        assert reg_a.status_code == 200, f"Alice registration failed: {reg_a.text}"
        alice_data = reg_a.json()
        tenant_a_id = alice_data["tenant_id"]
        alice_token = alice_data["access_token"]
        alice_headers = {"Authorization": f"Bearer {alice_token}"}

        # 2. Alice invites Bob to Tenant A
        bob_email = f"bob_{prefix}@tenanta.com"
        inv_res = await ac.post("/auth/invitations", headers=alice_headers, json={
            "email": bob_email,
            "role": "member"
        })
        assert inv_res.status_code == 201, f"Bob invitation failed: {inv_res.text}"
        bob_token_raw = inv_res.json()["invitation_token"]

        # 3. Bob accepts invitation
        accept_res = await ac.post("/auth/invitations/accept", json={
            "invitation_token": bob_token_raw,
            "email": bob_email,
            "password": "PasswordBob123!"
        })
        assert accept_res.status_code == 200, f"Bob accept failed: {accept_res.text}"
        bob_data = accept_res.json()
        assert bob_data["tenant_id"] == tenant_a_id, "Bob must belong to Tenant A"
        assert bob_data["user_id"] != alice_data["user_id"], "Alice and Bob must have distinct user_ids"
        bob_headers = {"Authorization": f"Bearer {bob_data['access_token']}"}

        # -------------------------------------------------------------
        # Setup Tenant B with Owner (Charlie)
        # -------------------------------------------------------------
        charlie_email = f"charlie_{prefix}@tenantb.com"
        reg_b = await ac.post("/auth/register", json={
            "email": charlie_email,
            "password": "PasswordCharlie123!",
            "tenant_name": f"Tenant B {prefix}"
        })
        assert reg_b.status_code == 200, f"Charlie registration failed: {reg_b.text}"
        charlie_data = reg_b.json()
        tenant_b_id = charlie_data["tenant_id"]
        assert tenant_b_id != tenant_a_id, "Tenant B must be distinct from Tenant A"
        charlie_headers = {"Authorization": f"Bearer {charlie_data['access_token']}"}

        # -------------------------------------------------------------
        # Step 1: Alice creates a private conversation and sends a message
        # -------------------------------------------------------------
        alice_chat_1 = await ac.post("/chat", headers=alice_headers, json={
            "message": "Hello, this is Alice's confidential strategic planning."
        })
        assert alice_chat_1.status_code == 200, f"Alice chat failed: {alice_chat_1.text}"
        conv_id = alice_chat_1.json()["conversation_id"]
        assert conv_id is not None

        # -------------------------------------------------------------
        # Step 2: Same-tenant, different-user access checks (Bob -> Alice's conv)
        # -------------------------------------------------------------
        # Bob attempts to read messages of Alice's conversation
        bob_get_res = await ac.get(f"/conversations/{conv_id}/messages", headers=bob_headers)
        assert bob_get_res.status_code == 403, (
            f"Expected 403 when Bob attempts to read Alice's conversation, got {bob_get_res.status_code}"
        )
        assert "permission" in bob_get_res.json()["detail"].lower()
        assert "confidential strategic planning" not in bob_get_res.text

        # Bob attempts to post a message into Alice's conversation
        bob_post_res = await ac.post("/chat", headers=bob_headers, json={
            "conversation_id": conv_id,
            "message": "Imposter Bob injecting message into Alice's chat"
        })
        assert bob_post_res.status_code == 403, (
            f"Expected 403 when Bob attempts to post into Alice's conversation, got {bob_post_res.status_code}"
        )
        assert "permission" in bob_post_res.json()["detail"].lower()

        # Bob attempts to delete Alice's conversation
        bob_delete_res = await ac.delete(f"/conversations/{conv_id}", headers=bob_headers)
        assert bob_delete_res.status_code == 403, (
            f"Expected 403 when Bob attempts to delete Alice's conversation, got {bob_delete_res.status_code}"
        )

        # -------------------------------------------------------------
        # Step 3: Cross-tenant access checks (Charlie -> Alice's conv)
        # -------------------------------------------------------------
        # Charlie attempts to read messages of Alice's conversation
        charlie_get_res = await ac.get(f"/conversations/{conv_id}/messages", headers=charlie_headers)
        assert charlie_get_res.status_code == 403, (
            f"Expected 403 when Charlie attempts to read Alice's conversation, got {charlie_get_res.status_code}"
        )
        assert "different tenant" in charlie_get_res.json()["detail"].lower()
        assert "confidential strategic planning" not in charlie_get_res.text

        # Charlie attempts to post a message into Alice's conversation
        charlie_post_res = await ac.post("/chat", headers=charlie_headers, json={
            "conversation_id": conv_id,
            "message": "Cross-tenant intrusion attempt"
        })
        assert charlie_post_res.status_code == 403, (
            f"Expected 403 when Charlie attempts to post into Alice's conversation, got {charlie_post_res.status_code}"
        )
        assert "different tenant" in charlie_post_res.json()["detail"].lower()

        # Charlie attempts to delete Alice's conversation
        charlie_delete_res = await ac.delete(f"/conversations/{conv_id}", headers=charlie_headers)
        assert charlie_delete_res.status_code == 403, (
            f"Expected 403 when Charlie attempts to delete Alice's conversation, got {charlie_delete_res.status_code}"
        )
        assert "different tenant" in charlie_delete_res.json()["detail"].lower()

        # -------------------------------------------------------------
        # Step 4: Authorized conversation owner access (Alice)
        # -------------------------------------------------------------
        # Alice reads her conversation messages
        alice_get_res = await ac.get(f"/conversations/{conv_id}/messages", headers=alice_headers)
        assert alice_get_res.status_code == 200, f"Alice read failed: {alice_get_res.text}"
        messages = alice_get_res.json()["messages"]
        assert len(messages) >= 2, "Conversation should have user and assistant messages"
        assert any("confidential strategic planning" in m["content"] for m in messages)

        # Alice sends a follow-up message in her conversation
        alice_chat_2 = await ac.post("/chat", headers=alice_headers, json={
            "conversation_id": conv_id,
            "message": "Second message from Alice in the same conversation."
        })
        assert alice_chat_2.status_code == 200, f"Alice second chat failed: {alice_chat_2.text}"
        assert alice_chat_2.json()["conversation_id"] == conv_id

        # Verify updated message count
        alice_get_res_2 = await ac.get(f"/conversations/{conv_id}/messages", headers=alice_headers)
        assert alice_get_res_2.status_code == 200
        assert len(alice_get_res_2.json()["messages"]) >= 4

        # -------------------------------------------------------------
        # Step 5: Deletion by owner and subsequent access verification
        # -------------------------------------------------------------
        # Alice deletes her conversation
        alice_del_res = await ac.delete(f"/conversations/{conv_id}", headers=alice_headers)
        assert alice_del_res.status_code == 204

        # Subsequent GET by Alice returns 404
        alice_after_del = await ac.get(f"/conversations/{conv_id}/messages", headers=alice_headers)
        assert alice_after_del.status_code == 404

        # Subsequent POST by Alice returns 404
        alice_post_after_del = await ac.post("/chat", headers=alice_headers, json={
            "conversation_id": conv_id,
            "message": "Attempting to post to deleted conversation"
        })
        assert alice_post_after_del.status_code == 404

        # Invalid conversation ID returns 404
        invalid_id_res = await ac.get("/conversations/invalid-id-format/messages", headers=alice_headers)
        assert invalid_id_res.status_code == 404
