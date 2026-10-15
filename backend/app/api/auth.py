import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import List
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, status
from backend.app.db.mongo import get_db
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

    # Prevent duplicate user registration
    existing_user = await db.users.find_one({"email": email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )

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
    query = {"email": email}
    if req.tenant_id:
        query["tenant_id"] = req.tenant_id

    user = await db.users.find_one(query)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    user_id = str(user["_id"])
    tenant_id = str(user["tenant_id"])
    role = user.get("role", "member")

    token = create_access_token(data={
        "user_id": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "email": email
    })

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user_id=user_id,
        tenant_id=tenant_id,
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already belongs to this tenant"
        )

    # Role is strictly assigned from the invitation record - cannot be chosen by client
    role = inv.get("role", "member")
    now_iso = now.isoformat()
    hashed_pw = hash_password(req.password)

    user_doc = {
        "tenant_id": tenant_id,
        "email": req_email,
        "hashed_password": hashed_pw,
        "role": role,
        "created_at": now_iso
    }
    user_res = await db.users.insert_one(user_doc)
    user_id = str(user_res.inserted_id)

    # Atomically mark invitation as accepted
    await db.invitations.update_one(
        {"_id": inv["_id"]},
        {"$set": {"status": "accepted", "used_at": now_iso}}
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
