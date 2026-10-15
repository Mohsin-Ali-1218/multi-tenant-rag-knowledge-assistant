import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import List
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, status
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError, OperationFailure, ConfigurationError, PyMongoError
from backend.app.db.mongo import get_db, get_mongo_client
from backend.app.models.schemas import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    CreateInvitationRequest,
    InvitationResponse,
    InvitationItemResponse,
    AcceptInvitationRequest
)
from backend.app.services.auth import (
    hash_password,
    verify_password,
    create_access_token,
    require_admin
)

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    db = get_db()
    email = req.email.strip().lower()

    # Generate unique slug for tenant - new registration ALWAYS creates a new tenant
    base_slug = re.sub(r'[^a-z0-9]+', '-', req.tenant_name.lower()).strip('-') or "tenant"
    slug = base_slug
    while await db.tenants.find_one({"slug": slug}):
        slug = f"{base_slug}-{secrets.token_hex(4)}"

    tenant_doc = {
        "name": req.tenant_name.strip(),
        "slug": slug,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    res = await db.tenants.insert_one(tenant_doc)
    tenant_id = str(res.inserted_id)

    # New tenant creator always becomes the "owner"
    hashed_pw = hash_password(req.password)
    user_doc = {
        "tenant_id": tenant_id,
        "email": email,
        "hashed_password": hashed_pw,
        "role": "owner",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    user_res = await db.users.insert_one(user_doc)
    user_id = str(user_res.inserted_id)

    token = create_access_token(data={
        "user_id": user_id,
        "tenant_id": tenant_id,
        "role": "owner",
        "email": email
    })

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user_id=user_id,
        tenant_id=tenant_id,
        role="owner"
    )

@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    db = get_db()
    email = req.email.strip().lower()

    target_tenant_id = None
    req_tenant_slug = req.tenant_slug.strip().lower() if req.tenant_slug and req.tenant_slug.strip() else None
    req_tenant_id = req.tenant_id.strip() if req.tenant_id and req.tenant_id.strip() else None

    if req_tenant_slug:
        tenant = await db.tenants.find_one({"slug": req_tenant_slug})
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email, password, or tenant"
            )
        target_tenant_id = str(tenant["_id"])
        if req_tenant_id and req_tenant_id != target_tenant_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email, password, or tenant"
            )
    elif req_tenant_id:
        target_tenant_id = req_tenant_id

    if target_tenant_id:
        user = await db.users.find_one({"email": email, "tenant_id": target_tenant_id})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email, password, or tenant"
            )
    else:
        # No tenant specified: find all tenant accounts for this email
        matching_users = await db.users.find({"email": email}).to_list(length=10)
        if not matching_users:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        if len(matching_users) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Multiple tenant accounts found for this email. Please specify 'tenant_id' or 'tenant_slug' to choose your tenant."
            )
        user = matching_users[0]
        target_tenant_id = str(user["tenant_id"])

    if not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email, password, or tenant" if (req_tenant_id or req_tenant_slug) else "Invalid email or password"
        )

    # Check if user account is disabled or inactive
    is_disabled = (
        user.get("disabled") is True
        or user.get("is_active") is False
        or str(user.get("status", "")).lower() in ["disabled", "inactive", "suspended"]
    )
    if is_disabled:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled"
        )

    # Check if tenant is disabled
    try:
        tenant_doc = await db.tenants.find_one({"_id": ObjectId(target_tenant_id)})
        if tenant_doc and (
            tenant_doc.get("disabled") is True
            or tenant_doc.get("is_active") is False
            or str(tenant_doc.get("status", "")).lower() in ["disabled", "inactive", "suspended"]
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Tenant account is disabled"
            )
    except InvalidId:
        pass

    user_id = str(user["_id"])
    role = user.get("role", "member")

    token = create_access_token(data={
        "user_id": user_id,
        "tenant_id": target_tenant_id,
        "role": role,
        "email": email
    })

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user_id=user_id,
        tenant_id=target_tenant_id,
        role=role
    )

# --- Invitation Endpoints ---

@router.post("/invitations", response_model=InvitationResponse, status_code=status.HTTP_201_CREATED)
async def create_invitation(
    req: CreateInvitationRequest,
    current_user: dict = Depends(require_admin)
):
    db = get_db()
    tenant_id = current_user["tenant_id"]
    email = req.email.strip().lower()

    # Check if user already exists in this tenant
    existing_user = await db.users.find_one({"tenant_id": tenant_id, "email": email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already belongs to this tenant"
        )

    # Generate cryptographically secure random token (stored as SHA-256 hash)
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=7)
    now_iso = now.isoformat()
    expires_at_iso = expires_at.isoformat()

    invitation_doc = {
        "tenant_id": tenant_id,
        "email": email,
        "role": req.role or "member",
        "token_hash": token_hash,
        "invited_by": current_user["user_id"],
        "status": "pending",
        "created_at": now_iso,
        "expires_at": expires_at_iso,
        "used_at": None
    }
    res = await db.invitations.insert_one(invitation_doc)
    inv_id = str(res.inserted_id)

    return InvitationResponse(
        id=inv_id,
        tenant_id=tenant_id,
        email=email,
        role=req.role or "member",
        invitation_token=raw_token,
        expires_at=expires_at_iso,
        created_at=now_iso
    )

@router.get("/invitations", response_model=List[InvitationItemResponse])
async def list_invitations(current_user: dict = Depends(require_admin)):
    db = get_db()
    tenant_id = current_user["tenant_id"]
    cursor = db.invitations.find({"tenant_id": tenant_id}).sort("created_at", -1)
    invites = []
    async for doc in cursor:
        invites.append(InvitationItemResponse(
            id=str(doc["_id"]),
            tenant_id=doc["tenant_id"],
            email=doc["email"],
            role=doc["role"],
            status=doc["status"],
            created_at=doc["created_at"],
            expires_at=doc["expires_at"],
            used_at=doc.get("used_at")
        ))
    return invites

@router.delete("/invitations/{invitation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_invitation(
    invitation_id: str,
    current_user: dict = Depends(require_admin)
):
    db = get_db()
    tenant_id = current_user["tenant_id"]
    try:
        inv_obj_id = ObjectId(invitation_id)
    except InvalidId:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found")

    inv = await db.invitations.find_one({"_id": inv_obj_id, "tenant_id": tenant_id})
    if not inv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found")

    await db.invitations.update_one(
        {"_id": inv_obj_id},
        {"$set": {"status": "revoked"}}
    )
    return None

@router.post("/invitations/accept", response_model=TokenResponse)
async def accept_invitation(req: AcceptInvitationRequest):
    db = get_db()
    token_hash = hashlib.sha256(req.invitation_token.encode("utf-8")).hexdigest()

    inv = await db.invitations.find_one({"token_hash": token_hash})
    if not inv:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid invitation token"
        )

    # Status check (single-use enforcement)
    if inv.get("status") != "pending" or inv.get("used_at") is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invitation has already been used or revoked"
        )

    # Expiry check
    expires_at_dt = datetime.fromisoformat(inv["expires_at"])
    now = datetime.now(timezone.utc)
    if now > expires_at_dt:
        await db.invitations.update_one(
            {"_id": inv["_id"]},
            {"$set": {"status": "expired"}}
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invitation has expired"
        )

    # Email-bound check
    req_email = req.email.strip().lower()
    if req_email != inv["email"].strip().lower():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invitation is bound to a different email address"
        )

    tenant_id = inv["tenant_id"]

    # Verify tenant exists
    try:
        tenant_obj_id = ObjectId(tenant_id)
    except InvalidId:
        tenant_obj_id = None

    tenant = await db.tenants.find_one({"_id": tenant_obj_id}) if tenant_obj_id else await db.tenants.find_one({"_id": tenant_id})
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant associated with invitation not found"
        )

    # Check if user already exists in this tenant
    existing_user = await db.users.find_one({"tenant_id": tenant_id, "email": req_email})
    if existing_user:
        # Mark invitation as accepted/used so it does not incorrectly linger in pending
        await db.invitations.update_one(
            {"_id": inv["_id"], "status": "pending"},
            {"$set": {"status": "accepted", "used_at": now.isoformat()}}
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already belongs to this tenant"
        )

    now_iso = now.isoformat()
    role = inv.get("role", "member")
    hashed_pw = hash_password(req.password)

    user_doc = {
        "tenant_id": tenant_id,
        "email": req_email,
        "hashed_password": hashed_pw,
        "role": role,
        "created_at": now_iso
    }

    # Attempt execution with MongoDB Multi-Document Transaction if supported by deployment (e.g. replica set)
    client = get_mongo_client()
    tx_supported = False
    try:
        async with client.start_session() as session:
            try:
                await session.start_transaction()
                claimed_inv = await db.invitations.find_one_and_update(
                    {
                        "_id": inv["_id"],
                        "status": "pending",
                        "used_at": None
                    },
                    {
                        "$set": {
                            "status": "accepted",
                            "used_at": now_iso
                        }
                    },
                    return_document=ReturnDocument.AFTER,
                    session=session
                )
                if not claimed_inv:
                    await session.abort_transaction()
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invitation has already been used or revoked"
                    )

                user_res = await db.users.insert_one(user_doc, session=session)
                user_id = str(user_res.inserted_id)

                await session.commit_transaction()
                tx_supported = True
            except (OperationFailure, ConfigurationError) as op_err:
                await session.abort_transaction()
                # Standalone mongod does not allow transactions (code 20)
                if getattr(op_err, "code", None) == 20 or "replica set" in str(op_err).lower():
                    tx_supported = False
                else:
                    raise
            except DuplicateKeyError:
                await session.abort_transaction()
                await db.invitations.update_one(
                    {"_id": inv["_id"], "status": "pending"},
                    {"$set": {"status": "accepted", "used_at": now_iso}}
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User already belongs to this tenant"
                )
    except HTTPException:
        raise
    except Exception as e:
        if tx_supported:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transaction failure during invitation acceptance: {str(e)}"
            )

    # Safe atomic claim with recovery compensation for standalone MongoDB deployments
    if not tx_supported:
        claimed_inv = await db.invitations.find_one_and_update(
            {
                "_id": inv["_id"],
                "status": "pending",
                "used_at": None
            },
            {
                "$set": {
                    "status": "accepted",
                    "used_at": now_iso
                }
            },
            return_document=ReturnDocument.AFTER
        )
        if not claimed_inv:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invitation has already been used or revoked"
            )

        try:
            user_res = await db.users.insert_one(user_doc)
            user_id = str(user_res.inserted_id)
        except DuplicateKeyError:
            # User already belongs to this tenant!
            # The invitation is NOT restored to pending (doing so would incorrectly
            # allow replaying/reusing an invitation for an already registered user).
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already belongs to this tenant"
            )
        except Exception as db_err:
            # An unexpected database failure occurred during user creation.
            # Safely revert the invitation to 'pending' so the user is NOT left with
            # an orphan 'accepted' invitation without an associated user account.
            try:
                await db.invitations.update_one(
                    {"_id": inv["_id"], "status": "accepted"},
                    {"$set": {"status": "pending", "used_at": None}}
                )
            except Exception:
                pass
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error during user registration: {str(db_err)}"
            )

    token = create_access_token(data={
        "user_id": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "email": req_email
    })

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user_id=user_id,
        tenant_id=tenant_id,
        role=role
    )
