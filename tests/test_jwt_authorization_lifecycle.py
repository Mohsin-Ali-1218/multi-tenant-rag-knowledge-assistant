import uuid
from bson import ObjectId
import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.app.db.mongo import get_db

@pytest.mark.asyncio
async def test_jwt_authorization_and_lifecycle_state():
    """
    Regression tests for live database state enforcement of JWT claims:
    1. Disabled users with existing tokens:
       - Users marked is_active=False, disabled=True, or status='disabled' cannot
         access protected endpoints with existing JWTs (401 Unauthorized).
       - Disabled users cannot log in (401 Unauthorized).
    2. Removed users with existing tokens:
       - Users deleted from MongoDB or removed from tenant cannot access
         endpoints with existing JWTs (401 Unauthorized).
    3. Demoted administrators attempting privileged actions:
       - Administrator demoted to 'member' in DB is blocked from privileged endpoints
         (POST/GET/DELETE /auth/invitations) despite holding an old JWT with role='admin' (403 Forbidden).
       - Demoted user retains access to standard member endpoints (GET /documents).
    4. Valid users continuing to access authorized endpoints:
       - Active members access member endpoints.
       - Active admins/owners access both member and admin endpoints.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        db = get_db()
        prefix = uuid.uuid4().hex[:6]

        # =====================================================================
        # 1. Disabled users with existing tokens
        # =====================================================================
        email_disabled_1 = f"disabled1_{prefix}@test.com"
        reg_d1 = await ac.post("/auth/register", json={
            "email": email_disabled_1,
            "password": "Password123!",
            "tenant_name": f"Disabled Corp 1 {prefix}"
        })
        assert reg_d1.status_code == 200
        d1_data = reg_d1.json()
        d1_token = d1_data["access_token"]
        d1_headers = {"Authorization": f"Bearer {d1_token}"}
        d1_user_id = d1_data["user_id"]

        # Verify active access works initially
        doc_res = await ac.get("/documents", headers=d1_headers)
        assert doc_res.status_code == 200

        # Case A: is_active=False
        await db.users.update_one(
            {"_id": ObjectId(d1_user_id)},
            {"$set": {"is_active": False}}
        )

        res_after_disable = await ac.get("/documents", headers=d1_headers)
        assert res_after_disable.status_code == 401, (
            f"Expected 401 for is_active=False user with existing JWT, got {res_after_disable.status_code}"
        )
        assert "disabled" in res_after_disable.json()["detail"].lower()

        # Disabled user cannot login
        login_disabled = await ac.post("/auth/login", json={
            "email": email_disabled_1,
            "password": "Password123!",
            "tenant_id": d1_data["tenant_id"]
        })
        assert login_disabled.status_code == 401
        assert "disabled" in login_disabled.json()["detail"].lower()

        # Case B: disabled=True
        email_disabled_2 = f"disabled2_{prefix}@test.com"
        reg_d2 = await ac.post("/auth/register", json={
            "email": email_disabled_2,
            "password": "Password123!",
            "tenant_name": f"Disabled Corp 2 {prefix}"
        })
        assert reg_d2.status_code == 200
        d2_data = reg_d2.json()
        d2_headers = {"Authorization": f"Bearer {d2_data['access_token']}"}

        await db.users.update_one(
            {"_id": ObjectId(d2_data["user_id"])},
            {"$set": {"disabled": True}}
        )
        res_d2 = await ac.get("/documents", headers=d2_headers)
        assert res_d2.status_code == 401
        assert "disabled" in res_d2.json()["detail"].lower()

        # Case C: status='disabled'
        email_disabled_3 = f"disabled3_{prefix}@test.com"
        reg_d3 = await ac.post("/auth/register", json={
            "email": email_disabled_3,
            "password": "Password123!",
            "tenant_name": f"Disabled Corp 3 {prefix}"
        })
        assert reg_d3.status_code == 200
        d3_data = reg_d3.json()
        d3_headers = {"Authorization": f"Bearer {d3_data['access_token']}"}

        await db.users.update_one(
            {"_id": ObjectId(d3_data["user_id"])},
            {"$set": {"status": "disabled"}}
        )
        res_d3 = await ac.get("/documents", headers=d3_headers)
        assert res_d3.status_code == 401
        assert "disabled" in res_d3.json()["detail"].lower()

        # =====================================================================
        # 2. Removed users with existing tokens
        # =====================================================================
        email_removed = f"removed_{prefix}@test.com"
        reg_rem = await ac.post("/auth/register", json={
            "email": email_removed,
            "password": "Password123!",
            "tenant_name": f"Removed Corp {prefix}"
        })
        assert reg_rem.status_code == 200
        rem_data = reg_rem.json()
        rem_headers = {"Authorization": f"Bearer {rem_data['access_token']}"}
        rem_user_id = rem_data["user_id"]

        # Delete user from DB
        await db.users.delete_one({"_id": ObjectId(rem_user_id)})

        res_rem_doc = await ac.get("/documents", headers=rem_headers)
        assert res_rem_doc.status_code == 401, (
            f"Expected 401 for deleted user with existing JWT, got {res_rem_doc.status_code}"
        )
        assert "revoked" in res_rem_doc.json()["detail"].lower() or "not found" in res_rem_doc.json()["detail"].lower()

        # Case: User removed from tenant (membership changed to different tenant)
        email_member_removed = f"mem_rem_{prefix}@test.com"
        reg_mem = await ac.post("/auth/register", json={
            "email": email_member_removed,
            "password": "Password123!",
            "tenant_name": f"Mem Tenant {prefix}"
        })
        assert reg_mem.status_code == 200
        mem_data = reg_mem.json()
        mem_headers = {"Authorization": f"Bearer {mem_data['access_token']}"}

        # Change user's tenant in DB to simulate removal from that tenant
        other_tenant_id = str(ObjectId())
        await db.users.update_one(
            {"_id": ObjectId(mem_data["user_id"])},
            {"$set": {"tenant_id": other_tenant_id}}
        )

        res_tenant_revoked = await ac.get("/documents", headers=mem_headers)
        assert res_tenant_revoked.status_code == 401
        assert "revoked" in res_tenant_revoked.json()["detail"].lower() or "not found" in res_tenant_revoked.json()["detail"].lower()

        # =====================================================================
        # 3. Demoted administrators attempting privileged actions
        # =====================================================================
        owner_email = f"owner_admin_{prefix}@org.com"
        reg_owner = await ac.post("/auth/register", json={
            "email": owner_email,
            "password": "Password123!",
            "tenant_name": f"Admin Org {prefix}"
        })
        assert reg_owner.status_code == 200
        owner_data = reg_owner.json()
        owner_headers = {"Authorization": f"Bearer {owner_data['access_token']}"}

        # Invite an admin
        admin_email = f"eve_admin_{prefix}@org.com"
        inv_admin = await ac.post("/auth/invitations", headers=owner_headers, json={
            "email": admin_email,
            "role": "admin"
        })
        assert inv_admin.status_code == 201
        admin_inv_token = inv_admin.json()["invitation_token"]

        # Eve accepts invitation and becomes an admin
        accept_admin = await ac.post("/auth/invitations/accept", json={
            "invitation_token": admin_inv_token,
            "email": admin_email,
            "password": "PasswordEve123!"
        })
        assert accept_admin.status_code == 200
        eve_data = accept_admin.json()
        assert eve_data["role"] == "admin"
        eve_token = eve_data["access_token"]
        eve_headers = {"Authorization": f"Bearer {eve_token}"}
        eve_user_id = eve_data["user_id"]

        # As admin, Eve can create an invitation
        test_invitee = f"frank_{prefix}@org.com"
        eve_create_res = await ac.post("/auth/invitations", headers=eve_headers, json={
            "email": test_invitee,
            "role": "member"
        })
        assert eve_create_res.status_code == 201
        frank_inv_id = eve_create_res.json()["id"]

        # Eve can list invitations
        eve_list_res = await ac.get("/auth/invitations", headers=eve_headers)
        assert eve_list_res.status_code == 200

        # NOW: Demote Eve in MongoDB to 'member'
        await db.users.update_one(
            {"_id": ObjectId(eve_user_id)},
            {"$set": {"role": "member"}}
        )

        # Eve STILL holds her existing JWT with claim role='admin'
        # Attempting privileged actions MUST be rejected based on live DB state:
        # A) Creating invitations -> 403 Forbidden
        eve_demoted_create = await ac.post("/auth/invitations", headers=eve_headers, json={
            "email": f"unauthorized_invitee_{prefix}@org.com",
            "role": "member"
        })
        assert eve_demoted_create.status_code == 403, (
            f"Expected 403 for demoted admin creating invitations, got {eve_demoted_create.status_code}"
        )
        assert "insufficient permissions" in eve_demoted_create.json()["detail"].lower()

        # B) Listing invitations -> 403 Forbidden
        eve_demoted_list = await ac.get("/auth/invitations", headers=eve_headers)
        assert eve_demoted_list.status_code == 403
        assert "insufficient permissions" in eve_demoted_list.json()["detail"].lower()

        # C) Revoking invitations -> 403 Forbidden
        eve_demoted_revoke = await ac.delete(f"/auth/invitations/{frank_inv_id}", headers=eve_headers)
        assert eve_demoted_revoke.status_code == 403
        assert "insufficient permissions" in eve_demoted_revoke.json()["detail"].lower()

        # D) However, as a regular member, Eve CAN STILL access member endpoints!
        eve_member_docs = await ac.get("/documents", headers=eve_headers)
        assert eve_member_docs.status_code == 200

        # =====================================================================
        # 4. Valid users continuing to access authorized endpoints
        # =====================================================================
        # Owner can still create invitations and access documents
        owner_docs = await ac.get("/documents", headers=owner_headers)
        assert owner_docs.status_code == 200

        owner_create = await ac.post("/auth/invitations", headers=owner_headers, json={
            "email": f"valid_member_{prefix}@org.com",
            "role": "member"
        })
        assert owner_create.status_code == 201


def test_production_configuration_rejects_insecure_secrets(monkeypatch):
    """
    Verify production configuration strictly rejects default/placeholder secrets
    and enforces entropy requirements.
    """
    from backend.app.config import Settings

    s = Settings()

    # In production, default/placeholder secret must be rejected
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", "replace_with_a_random_secret")
    with pytest.raises(RuntimeError, match="Default or placeholder JWT_SECRET is prohibited"):
        _ = s.JWT_SECRET

    # In production, short secret (< 32 chars) must be rejected
    monkeypatch.setenv("JWT_SECRET", "short_secret_12345")
    with pytest.raises(RuntimeError, match="JWT_SECRET must be at least 32 characters"):
        _ = s.JWT_SECRET

    # In production, placeholder GEMINI_API_KEY must be rejected
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    with pytest.raises(RuntimeError, match="Valid GEMINI_API_KEY must be configured"):
        _ = s.GEMINI_API_KEY

    # In production, empty GEMINI_API_KEY must be rejected
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(RuntimeError, match="Valid GEMINI_API_KEY must be configured"):
        _ = s.GEMINI_API_KEY

    # In production, valid 32+ char secret and key succeed
    monkeypatch.setenv("JWT_SECRET", "a" * 32)
    monkeypatch.setenv("GEMINI_API_KEY", "valid_api_key_sample")
    assert s.JWT_SECRET == "a" * 32
    assert s.GEMINI_API_KEY == "valid_api_key_sample"
