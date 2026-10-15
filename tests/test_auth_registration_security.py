import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from bson import ObjectId
from pymongo.errors import PyMongoError
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

@pytest.mark.asyncio
async def test_concurrent_invitation_acceptance_race_condition():
    """
    Concurrency Regression Test:
    Verify that when multiple concurrent requests attempt to accept the same invitation token,
    atomic claim via find_one_and_update guarantees that exactly ONE request succeeds (HTTP 200)
    and all other concurrent requests are rejected (HTTP 400).
    Under no circumstances should two requests accept the token, nor should unhandled server crashes (HTTP 500) occur.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        unique_prefix = uuid.uuid4().hex[:6]
        org_name = f"Concurrency Org {unique_prefix}"

        # 1. Register Owner
        res_owner = await ac.post("/auth/register", json={
            "email": f"concur_owner_{unique_prefix}@org.com",
            "password": "password123",
            "tenant_name": org_name
        })
        assert res_owner.status_code == 200
        owner_data = res_owner.json()
        owner_headers = {"Authorization": f"Bearer {owner_data['access_token']}"}

        # 2. Create single invitation token
        invite_email = f"concur_invitee_{unique_prefix}@org.com"
        inv_res = await ac.post("/auth/invitations", headers=owner_headers, json={
            "email": invite_email,
            "role": "member"
        })
        assert inv_res.status_code == 201
        invitation_token = inv_res.json()["invitation_token"]

        # 3. Fire multiple concurrent accept requests for the exact same token
        payload = {
            "invitation_token": invitation_token,
            "email": invite_email,
            "password": "password123"
        }

        concurrent_count = 5
        tasks = [ac.post("/auth/invitations/accept", json=payload) for _ in range(concurrent_count)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        status_codes = [r.status_code for r in responses if hasattr(r, "status_code")]
        assert len(status_codes) == concurrent_count

        # Exactly 1 request must succeed (200 OK)
        successes = [r for r in responses if getattr(r, "status_code", None) == 200]
        failures = [r for r in responses if getattr(r, "status_code", None) == 400]

        assert len(successes) == 1, (
            f"Expected exactly 1 successful accept, but got {len(successes)}. Status codes: {status_codes}"
        )
        assert len(failures) == concurrent_count - 1, (
            f"Expected {concurrent_count - 1} rejections with 400, but got {len(failures)}. Status codes: {status_codes}"
        )
        assert 500 not in status_codes, f"Unhandled server crash detected in concurrent requests: {status_codes}"

        # Verify failure error details
        for f_res in failures:
            detail = f_res.json().get("detail", "").lower()
            assert "used" in detail or "belongs" in detail

        # 4. Verify MongoDB state consistency: exactly ONE user created for this tenant/email
        db = get_db()
        users_count = await db.users.count_documents({
            "tenant_id": owner_data["tenant_id"],
            "email": invite_email
        })
        assert users_count == 1, f"Expected exactly 1 user record, found {users_count}"

        # 5. Verify invitation is marked accepted
        inv_doc = await db.invitations.find_one({"_id": ObjectId(inv_res.json()["id"])})
        assert inv_doc["status"] == "accepted"
        assert inv_doc["used_at"] is not None

@pytest.mark.asyncio
async def test_invitation_rollback_on_database_failure():
    """
    Test Failure Scenario 1:
    A transient database failure during user insertion must NOT leave an accepted invitation
    without an associated user.
    The recovery mechanism must safely roll back the invitation to 'pending' with used_at=None,
    allowing the user to retry and accept once the database recovers.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        unique_prefix = uuid.uuid4().hex[:6]
        org_name = f"DB Failure Org {unique_prefix}"

        # 1. Register Owner
        res_owner = await ac.post("/auth/register", json={
            "email": f"fail_owner_{unique_prefix}@org.com",
            "password": "password123",
            "tenant_name": org_name
        })
        assert res_owner.status_code == 200
        owner_data = res_owner.json()
        owner_headers = {"Authorization": f"Bearer {owner_data['access_token']}"}

        # 2. Create invitation
        invite_email = f"fail_invitee_{unique_prefix}@org.com"
        inv_res = await ac.post("/auth/invitations", headers=owner_headers, json={
            "email": invite_email,
            "role": "member"
        })
        assert inv_res.status_code == 201
        inv_id = inv_res.json()["id"]
        invitation_token = inv_res.json()["invitation_token"]

        payload = {
            "invitation_token": invitation_token,
            "email": invite_email,
            "password": "password123"
        }

        # 3. Simulate transient database write failure during users.insert_one
        db = get_db()
        from pymongo.asynchronous.collection import AsyncCollection
        orig_insert_one = AsyncCollection.insert_one

        async def failing_insert_one(self, doc, *args, **kwargs):
            if "hashed_password" in doc and doc.get("email") == invite_email:
                raise PyMongoError("Simulated database failure during user insert")
            return await orig_insert_one(self, doc, *args, **kwargs)

        with patch.object(AsyncCollection, "insert_one", failing_insert_one):
            fail_res = await ac.post("/auth/invitations/accept", json=payload)
            assert fail_res.status_code == 500, f"Expected 500 on database failure, got {fail_res.status_code}"

        # 4. Verify Invitation State: NOT left in 'accepted' state without a user!
        inv_doc = await db.invitations.find_one({"_id": ObjectId(inv_id)})
        assert inv_doc["status"] == "pending", "Invitation must be rolled back to pending after DB failure"
        assert inv_doc.get("used_at") is None, "used_at must be reset to None"

        # 5. Verify User State: No user record created
        user_doc = await db.users.find_one({"tenant_id": owner_data["tenant_id"], "email": invite_email})
        assert user_doc is None, "No user record should exist after failed acceptance"

        # 6. Retry: Once the DB failure is resolved, user must be able to accept successfully!
        retry_res = await ac.post("/auth/invitations/accept", json=payload)
        assert retry_res.status_code == 200, f"Retry accept failed: {retry_res.text}"
        assert "access_token" in retry_res.json()

        # Final check: Invitation is now accepted, user exists
        inv_doc_final = await db.invitations.find_one({"_id": ObjectId(inv_id)})
        assert inv_doc_final["status"] == "accepted"
        assert inv_doc_final["used_at"] is not None
        user_doc_final = await db.users.find_one({"tenant_id": owner_data["tenant_id"], "email": invite_email})
        assert user_doc_final is not None

@pytest.mark.asyncio
async def test_invitation_not_restored_to_pending_when_user_already_exists():
    """
    Test Failure Scenario 2:
    If an invitation is claimed for an email that already belongs to the tenant,
    the invitation must NOT be incorrectly restored to 'pending'.
    It must be finalized (marked accepted) so that stale invitations cannot be replayed.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        unique_prefix = uuid.uuid4().hex[:6]
        org_name = f"Existing User Org {unique_prefix}"

        # 1. Register Owner
        res_owner = await ac.post("/auth/register", json={
            "email": f"exist_owner_{unique_prefix}@org.com",
            "password": "password123",
            "tenant_name": org_name
        })
        assert res_owner.status_code == 200
        owner_data = res_owner.json()
        owner_headers = {"Authorization": f"Bearer {owner_data['access_token']}"}

        # 2. Pre-create a user in this tenant with a specific email
        existing_email = f"already_member_{unique_prefix}@org.com"
        db = get_db()
        await db.users.insert_one({
            "tenant_id": owner_data["tenant_id"],
            "email": existing_email,
            "hashed_password": "dummyhashedpassword",
            "role": "member",
            "created_at": datetime.now(timezone.utc).isoformat()
        })

        # 3. Create invitation for this already-existing email
        # Bypass admin check in test to insert invitation directly (simulating pre-existing invite)
        import hashlib
        raw_token = f"test_existing_token_{unique_prefix}_12345678"
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc)
        inv_insert = await db.invitations.insert_one({
            "tenant_id": owner_data["tenant_id"],
            "email": existing_email,
            "role": "member",
            "token_hash": token_hash,
            "invited_by": owner_data["user_id"],
            "status": "pending",
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(days=7)).isoformat(),
            "used_at": None
        })
        inv_id = inv_insert.inserted_id

        # 4. Attempt to accept invitation
        accept_res = await ac.post("/auth/invitations/accept", json={
            "invitation_token": raw_token,
            "email": existing_email,
            "password": "password123"
        })
        assert accept_res.status_code == 400
        assert "already belongs" in accept_res.json()["detail"].lower()

        # 5. Verify Invitation is NOT restored to pending!
        inv_doc = await db.invitations.find_one({"_id": inv_id})
        assert inv_doc["status"] != "pending", "Invitation must NOT be incorrectly restored to 'pending'"
        assert inv_doc["status"] == "accepted", "Invitation should be finalized as accepted"


@pytest.mark.asyncio
async def test_multitenant_email_login_and_tenant_selection():
    """
    Regression test for multi-tenant user authentication:
    1. Duplicate emails across tenants:
       - Same email registered as owner in Tenant Alpha and Tenant Beta with different passwords.
    2. Ambiguous login prevention:
       - When multiple accounts exist for an email, logging in without tenant_id or tenant_slug
         must return 400 Bad Request to prevent cross-tenant credential leakage.
    3. Correct tenant selection via tenant_slug:
       - Supplying Tenant Alpha's slug and password logs into Tenant Alpha.
       - Supplying Tenant Beta's slug and password logs into Tenant Beta.
    4. Correct tenant selection via tenant_id:
       - Supplying Tenant Alpha's id and password logs into Tenant Alpha.
       - Supplying Tenant Beta's id and password logs into Tenant Beta.
    5. Incorrect passwords:
       - Incorrect password with Tenant Alpha slug returns 401 Unauthorized.
       - Incorrect password with Tenant Beta id returns 401 Unauthorized.
    6. Cross-tenant authentication attempts:
       - Attempting to log into Tenant Beta with Tenant Alpha's password returns 401 Unauthorized.
       - Attempting to log into Tenant Gamma where the user has no account returns 401 Unauthorized.
       - Attempting to supply conflicting tenant_id and tenant_slug returns 401 Unauthorized.
    7. Single-tenant backward compatibility:
       - User registered with only one tenant account can log in without tenant identifiers.
       - Invalid password for single-tenant user returns 401 Unauthorized.
    8. Invitation flow with duplicate email across tenants:
       - Tenant Gamma invites the same shared email.
       - Acceptance succeeds with a distinct password.
       - User can authenticate into Tenant Gamma with member role.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        db = get_db()
        prefix = uuid.uuid4().hex[:6]
        shared_email = f"shared_{prefix}@domain.com"
        pass_alpha = "AlphaSecretPass123!"
        pass_beta = "BetaSecretPass456!"
        pass_gamma = "GammaSecretPass789!"

        # 1. Register Tenant Alpha
        res_alpha = await ac.post("/auth/register", json={
            "email": shared_email,
            "password": pass_alpha,
            "tenant_name": f"Alpha Corp {prefix}"
        })
        assert res_alpha.status_code == 200, f"Alpha registration failed: {res_alpha.text}"
        data_alpha = res_alpha.json()
        tenant_id_alpha = data_alpha["tenant_id"]
        tenant_alpha_doc = await db.tenants.find_one({"_id": ObjectId(tenant_id_alpha)})
        slug_alpha = tenant_alpha_doc["slug"]

        # 2. Register Tenant Beta with SAME email, DIFFERENT password
        res_beta = await ac.post("/auth/register", json={
            "email": shared_email,
            "password": pass_beta,
            "tenant_name": f"Beta Corp {prefix}"
        })
        assert res_beta.status_code == 200, f"Beta registration failed: {res_beta.text}"
        data_beta = res_beta.json()
        tenant_id_beta = data_beta["tenant_id"]
        tenant_beta_doc = await db.tenants.find_one({"_id": ObjectId(tenant_id_beta)})
        slug_beta = tenant_beta_doc["slug"]

        assert tenant_id_alpha != tenant_id_beta
        assert slug_alpha != slug_beta

        # Register a third independent Tenant Gamma with a different owner
        gamma_owner_email = f"owner_gamma_{prefix}@domain.com"
        res_gamma = await ac.post("/auth/register", json={
            "email": gamma_owner_email,
            "password": "GammaOwnerPass123!",
            "tenant_name": f"Gamma Corp {prefix}"
        })
        assert res_gamma.status_code == 200
        data_gamma = res_gamma.json()
        tenant_id_gamma = data_gamma["tenant_id"]
        tenant_gamma_doc = await db.tenants.find_one({"_id": ObjectId(tenant_id_gamma)})
        slug_gamma = tenant_gamma_doc["slug"]
        headers_gamma_owner = {"Authorization": f"Bearer {data_gamma['access_token']}"}

        # 3. Ambiguous login attempt without tenant identification -> 400 Bad Request
        res_ambiguous = await ac.post("/auth/login", json={
            "email": shared_email,
            "password": pass_alpha
        })
        assert res_ambiguous.status_code == 400, (
            f"Expected 400 for ambiguous login with duplicate email, got {res_ambiguous.status_code}"
        )
        assert "multiple tenant accounts" in res_ambiguous.json()["detail"].lower()

        # 4. Correct tenant selection via tenant_slug
        # Login to Alpha via slug
        res_login_alpha_slug = await ac.post("/auth/login", json={
            "email": shared_email,
            "password": pass_alpha,
            "tenant_slug": slug_alpha
        })
        assert res_login_alpha_slug.status_code == 200, f"Alpha slug login failed: {res_login_alpha_slug.text}"
        assert res_login_alpha_slug.json()["tenant_id"] == tenant_id_alpha

        # Login to Beta via slug
        res_login_beta_slug = await ac.post("/auth/login", json={
            "email": shared_email,
            "password": pass_beta,
            "tenant_slug": slug_beta
        })
        assert res_login_beta_slug.status_code == 200, f"Beta slug login failed: {res_login_beta_slug.text}"
        assert res_login_beta_slug.json()["tenant_id"] == tenant_id_beta

        # 5. Correct tenant selection via tenant_id
        # Login to Alpha via id
        res_login_alpha_id = await ac.post("/auth/login", json={
            "email": shared_email,
            "password": pass_alpha,
            "tenant_id": tenant_id_alpha
        })
        assert res_login_alpha_id.status_code == 200
        assert res_login_alpha_id.json()["tenant_id"] == tenant_id_alpha

        # Login to Beta via id
        res_login_beta_id = await ac.post("/auth/login", json={
            "email": shared_email,
            "password": pass_beta,
            "tenant_id": tenant_id_beta
        })
        assert res_login_beta_id.status_code == 200
        assert res_login_beta_id.json()["tenant_id"] == tenant_id_beta

        # 6. Incorrect password tests
        res_bad_pw_alpha = await ac.post("/auth/login", json={
            "email": shared_email,
            "password": "WrongPassword123!",
            "tenant_slug": slug_alpha
        })
        assert res_bad_pw_alpha.status_code == 401

        res_bad_pw_beta = await ac.post("/auth/login", json={
            "email": shared_email,
            "password": "WrongPassword123!",
            "tenant_id": tenant_id_beta
        })
        assert res_bad_pw_beta.status_code == 401

        # 7. Cross-tenant authentication attempts (credentials from one tenant used against another)
        # Attempting Beta login with Alpha password
        res_cross_tenant_1 = await ac.post("/auth/login", json={
            "email": shared_email,
            "password": pass_alpha,
            "tenant_slug": slug_beta
        })
        assert res_cross_tenant_1.status_code == 401, (
            f"Expected 401 when using Alpha password for Beta slug, got {res_cross_tenant_1.status_code}"
        )

        # Attempting Alpha login with Beta password
        res_cross_tenant_2 = await ac.post("/auth/login", json={
            "email": shared_email,
            "password": pass_beta,
            "tenant_id": tenant_id_alpha
        })
        assert res_cross_tenant_2.status_code == 401

        # Attempting to authenticate into Tenant Gamma where user does not have an account
        res_unauthorized_gamma_slug = await ac.post("/auth/login", json={
            "email": shared_email,
            "password": pass_alpha,
            "tenant_slug": slug_gamma
        })
        assert res_unauthorized_gamma_slug.status_code == 401

        res_unauthorized_gamma_id = await ac.post("/auth/login", json={
            "email": shared_email,
            "password": pass_alpha,
            "tenant_id": tenant_id_gamma
        })
        assert res_unauthorized_gamma_id.status_code == 401

        # Attempting with mismatched / conflicting tenant_id and tenant_slug
        res_conflict = await ac.post("/auth/login", json={
            "email": shared_email,
            "password": pass_alpha,
            "tenant_id": tenant_id_alpha,
            "tenant_slug": slug_beta
        })
        assert res_conflict.status_code == 401

        # 8. Single-tenant user backwards compatibility
        solo_email = f"solo_{prefix}@domain.com"
        solo_pass = "SoloPassSecret123!"
        res_solo_reg = await ac.post("/auth/register", json={
            "email": solo_email,
            "password": solo_pass,
            "tenant_name": f"Solo Corp {prefix}"
        })
        assert res_solo_reg.status_code == 200
        solo_tenant_id = res_solo_reg.json()["tenant_id"]

        # Solo user logs in WITHOUT tenant identifiers
        res_solo_login = await ac.post("/auth/login", json={
            "email": solo_email,
            "password": solo_pass
        })
        assert res_solo_login.status_code == 200
        assert res_solo_login.json()["tenant_id"] == solo_tenant_id

        # Solo user with wrong password
        res_solo_bad = await ac.post("/auth/login", json={
            "email": solo_email,
            "password": "WrongPassword!"
        })
        assert res_solo_bad.status_code == 401

        # 9. Duplicate email via Invitation Flow
        # Tenant Gamma invites shared_email
        inv_res = await ac.post("/auth/invitations", headers=headers_gamma_owner, json={
            "email": shared_email,
            "role": "member"
        })
        assert inv_res.status_code == 201, f"Invitation creation failed: {inv_res.text}"
        inv_token = inv_res.json()["invitation_token"]

        # Accept invitation into Tenant Gamma with distinct password
        accept_res = await ac.post("/auth/invitations/accept", json={
            "invitation_token": inv_token,
            "email": shared_email,
            "password": pass_gamma
        })
        assert accept_res.status_code == 200, f"Accept invitation failed: {accept_res.text}"
        assert accept_res.json()["tenant_id"] == tenant_id_gamma
        assert accept_res.json()["role"] == "member"

        # shared_email can now log into Tenant Gamma
        res_gamma_login = await ac.post("/auth/login", json={
            "email": shared_email,
            "password": pass_gamma,
            "tenant_slug": slug_gamma
        })
        assert res_gamma_login.status_code == 200
        assert res_gamma_login.json()["tenant_id"] == tenant_id_gamma
        assert res_gamma_login.json()["role"] == "member"

        # Alpha password does NOT work for Tenant Gamma
        res_gamma_wrong_pw = await ac.post("/auth/login", json={
            "email": shared_email,
            "password": pass_alpha,
            "tenant_slug": slug_gamma
        })
        assert res_gamma_wrong_pw.status_code == 401
