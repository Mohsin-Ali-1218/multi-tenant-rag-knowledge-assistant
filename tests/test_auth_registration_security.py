import uuid
from datetime import datetime, timedelta, timezone
from bson import ObjectId
import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.app.db.mongo import get_db

@pytest.mark.asyncio
async def test_registration_rejects_client_chosen_role_and_slug():
    """
    OWASP API3: Mass Assignment Protection
    Clients MUST NOT be permitted to specify `role`, `tenant_slug`, or `tenant_id`
    on registration. Pydantic must reject with 422 Unprocessable Entity.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        unique_prefix = uuid.uuid4().hex[:6]

        # 1. Attempt to send role='admin'
        res_role = await ac.post("/auth/register", json={
            "email": f"attacker_{unique_prefix}_1@evil.com",
            "password": "password123",
            "tenant_name": "Infiltrate Org",
            "role": "admin"
        })
        assert res_role.status_code == 422, f"Expected 422 for client-chosen role, got {res_role.status_code}: {res_role.text}"

        # 2. Attempt to send tenant_slug
        res_slug = await ac.post("/auth/register", json={
            "email": f"attacker_{unique_prefix}_2@evil.com",
            "password": "password123",
            "tenant_name": "Infiltrate Org",
            "tenant_slug": "target-tenant"
        })
        assert res_slug.status_code == 422, f"Expected 422 for client-chosen tenant_slug, got {res_slug.status_code}: {res_slug.text}"

        # 3. Attempt to send tenant_id
        res_tid = await ac.post("/auth/register", json={
            "email": f"attacker_{unique_prefix}_3@evil.com",
            "password": "password123",
            "tenant_name": "Infiltrate Org",
            "tenant_id": "6ad04c0d9f72377f1c12f5ab"
        })
        assert res_tid.status_code == 422, f"Expected 422 for client-chosen tenant_id, got {res_tid.status_code}: {res_tid.text}"

@pytest.mark.asyncio
async def test_existing_slug_never_joins_that_tenant():
    """
    Vulnerability Demonstration & Fix Verification:
    Original vulnerability: Registering with an existing tenant slug reused the existing tenant_id,
    allowing unauthorized users to join an existing organization and read its documents.
    Fix: New registration ALWAYS creates a brand-new tenant with a unique slug and distinct tenant_id.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        unique_prefix = uuid.uuid4().hex[:6]
        org_name = f"Secure Enterprise {unique_prefix}"

        # Step 1: Legitimate Organization Owner registers
        res_org = await ac.post("/auth/register", json={
            "email": f"owner_{unique_prefix}@secure-org.com",
            "password": "password123",
            "tenant_name": org_name
        })
        assert res_org.status_code == 200, f"Org registration failed: {res_org.text}"
        org_data = res_org.json()
        org_tenant_id = org_data["tenant_id"]
        assert org_data["role"] == "owner", "New tenant creator must be assigned owner role"
        org_headers = {"Authorization": f"Bearer {org_data['access_token']}"}

        # Step 2: Org owner uploads a sensitive document
        doc_res = await ac.post("/documents", headers=org_headers, json={
            "title": "Secret Financial Projections",
            "raw_text": "Confidential financial quarterly projections: revenue $50M, profit $12M."
        })
        assert doc_res.status_code == 201
        doc_id = doc_res.json()["id"]

        # Step 3: Attacker attempts to register using the exact same tenant_name
        res_attacker = await ac.post("/auth/register", json={
            "email": f"attacker_{unique_prefix}@outside.com",
            "password": "password123",
            "tenant_name": org_name
        })
        assert res_attacker.status_code == 200, f"Attacker registration failed: {res_attacker.text}"
        attacker_data = res_attacker.json()
        attacker_tenant_id = attacker_data["tenant_id"]
        attacker_headers = {"Authorization": f"Bearer {attacker_data['access_token']}"}

        # Step 4: Verify Attacker is NOT joined to the legitimate tenant!
        assert attacker_tenant_id != org_tenant_id, (
            f"CRITICAL VULNERABILITY: Attacker joined existing tenant! Both have tenant_id: {org_tenant_id}"
        )

        # Step 5: Verify Attacker cannot access legitimate organization's documents
        forbidden_res = await ac.get(f"/documents/{doc_id}", headers=attacker_headers)
        assert forbidden_res.status_code == 403, (
            f"Expected 403 Forbidden for cross-tenant document access, got {forbidden_res.status_code}"
        )

        # Attacker's own document listing must be empty
        docs_list_res = await ac.get("/documents", headers=attacker_headers)
        assert docs_list_res.status_code == 200
        assert len(docs_list_res.json()) == 0, "Attacker should have no documents in their new isolated tenant"

@pytest.mark.asyncio
async def test_invitation_lifecycle_and_role_authorization():
    """
    Test legitimate tenant joining via cryptographically secure invitation:
    - Admin creates invitation with role='member'
    - Plain token returned in response, but SHA-256 hash stored in DB
    - Invitee accepts invitation and is enrolled in the tenant with role='member'
    - Member cannot choose or escalate role
    - Member cannot create invitations (403 Forbidden)
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        unique_prefix = uuid.uuid4().hex[:6]
        org_name = f"Collab Corp {unique_prefix}"

        # 1. Register Owner
        res_owner = await ac.post("/auth/register", json={
            "email": f"admin_{unique_prefix}@collab.com",
            "password": "password123",
            "tenant_name": org_name
        })
        assert res_owner.status_code == 200
        owner_data = res_owner.json()
        owner_token = owner_data["access_token"]
        owner_tenant_id = owner_data["tenant_id"]
        owner_headers = {"Authorization": f"Bearer {owner_token}"}

        # 2. Owner invites a member
        invite_email = f"employee_{unique_prefix}@collab.com"
        inv_res = await ac.post("/auth/invitations", headers=owner_headers, json={
            "email": invite_email,
            "role": "member"
        })
        assert inv_res.status_code == 201, f"Failed to create invitation: {inv_res.text}"
        inv_data = inv_res.json()
        raw_token = inv_data["invitation_token"]
        assert raw_token
        assert inv_data["role"] == "member"
        assert inv_data["tenant_id"] == owner_tenant_id

        # 3. Verify in MongoDB: plain token is NOT stored in DB, only token_hash
        db = get_db()
        inv_doc = await db.invitations.find_one({"_id": ObjectId(inv_data["id"])})
        assert inv_doc is not None
        assert "token_hash" in inv_doc
        assert "invitation_token" not in inv_doc
        assert inv_doc["token_hash"] != raw_token

        # 4. Invitee accepts invitation (role=admin attempt rejected)
        tamper_res = await ac.post("/auth/invitations/accept", json={
            "invitation_token": raw_token,
            "email": invite_email,
            "password": "newpassword123",
            "role": "admin"
        })
        assert tamper_res.status_code == 422, "Accept invitation must forbid client-specified role"

        # 5. Invitee accepts invitation properly
        accept_res = await ac.post("/auth/invitations/accept", json={
            "invitation_token": raw_token,
            "email": invite_email,
            "password": "newpassword123"
        })
        assert accept_res.status_code == 200, f"Accept invitation failed: {accept_res.text}"
        employee_data = accept_res.json()
        assert employee_data["tenant_id"] == owner_tenant_id, "Employee must join owner's tenant"
        assert employee_data["role"] == "member", "Employee role must be 'member' as specified by inviter"
        employee_headers = {"Authorization": f"Bearer {employee_data['access_token']}"}

        # 6. Verify non-admin (member) cannot create invitations
        unauthorized_invite = await ac.post("/auth/invitations", headers=employee_headers, json={
            "email": f"hacker_{unique_prefix}@test.com",
            "role": "member"
        })
        assert unauthorized_invite.status_code == 403, (
            f"Expected 403 for member creating invitation, got {unauthorized_invite.status_code}"
        )

@pytest.mark.asyncio
async def test_invitation_rejection_conditions():
    """
    Test invitation security conditions:
    1. Reused / single-use invitation is rejected
    2. Wrong email invitation is rejected
    3. Expired invitation is rejected
    4. Non-existent token is rejected
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        unique_prefix = uuid.uuid4().hex[:6]
        org_name = f"Security Checks {unique_prefix}"

        # 1. Register Owner
        res_owner = await ac.post("/auth/register", json={
            "email": f"owner_{unique_prefix}@checks.com",
            "password": "password123",
            "tenant_name": org_name
        })
        assert res_owner.status_code == 200
        owner_headers = {"Authorization": f"Bearer {res_owner.json()['access_token']}"}

        # 2. Test Single-Use (Reused token)
        email_single = f"single_{unique_prefix}@checks.com"
        inv1 = await ac.post("/auth/invitations", headers=owner_headers, json={
            "email": email_single,
            "role": "member"
        })
        assert inv1.status_code == 201
        token1 = inv1.json()["invitation_token"]

        accept1 = await ac.post("/auth/invitations/accept", json={
            "invitation_token": token1,
            "email": email_single,
            "password": "password123"
        })
        assert accept1.status_code == 200

        # Attempt to reuse the token
        reuse_res = await ac.post("/auth/invitations/accept", json={
            "invitation_token": token1,
            "email": email_single,
            "password": "password123"
        })
        assert reuse_res.status_code == 400, f"Expected 400 for reused invitation token, got {reuse_res.status_code}"
        assert "used" in reuse_res.json()["detail"].lower()

        # 3. Test Wrong Email Binding
        email_legit = f"legit_{unique_prefix}@checks.com"
        email_imposter = f"imposter_{unique_prefix}@checks.com"
        inv2 = await ac.post("/auth/invitations", headers=owner_headers, json={
            "email": email_legit,
            "role": "member"
        })
        assert inv2.status_code == 201
        token2 = inv2.json()["invitation_token"]

        wrong_email_res = await ac.post("/auth/invitations/accept", json={
            "invitation_token": token2,
            "email": email_imposter,
            "password": "password123"
        })
        assert wrong_email_res.status_code == 400
        assert "different email" in wrong_email_res.json()["detail"].lower()

        # 4. Test Expired Invitation
        email_expired = f"expired_{unique_prefix}@checks.com"
        inv3 = await ac.post("/auth/invitations", headers=owner_headers, json={
            "email": email_expired,
            "role": "member"
        })
        assert inv3.status_code == 201
        token3 = inv3.json()["invitation_token"]

        # Manually backdate expiration in DB to simulate expired invitation
        db = get_db()
        past_iso = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        await db.invitations.update_one(
            {"email": email_expired},
            {"$set": {"expires_at": past_iso}}
        )

        expired_res = await ac.post("/auth/invitations/accept", json={
            "invitation_token": token3,
            "email": email_expired,
            "password": "password123"
        })
        assert expired_res.status_code == 400
        assert "expired" in expired_res.json()["detail"].lower()

        # 5. Test Non-Existent Token
        fake_res = await ac.post("/auth/invitations/accept", json={
            "invitation_token": "non_existent_token_1234567890",
            "email": "someone@example.com",
            "password": "password123"
        })
        assert fake_res.status_code == 400
        assert "invalid" in fake_res.json()["detail"].lower()
