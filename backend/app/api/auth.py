from datetime import datetime, timezone
from bson import ObjectId
from fastapi import APIRouter, HTTPException, status
from backend.app.db.mongo import get_db
from backend.app.models.schemas import RegisterRequest, LoginRequest, TokenResponse
from backend.app.services.auth import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    db = get_db()
    slug = req.tenant_slug or req.tenant_name.lower().replace(" ", "-")

    # Find or create tenant
    tenant = await db.tenants.find_one({"slug": slug})
    if not tenant:
        tenant_doc = {
            "name": req.tenant_name,
            "slug": slug,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        res = await db.tenants.insert_one(tenant_doc)
        tenant_id = str(res.inserted_id)
    else:
        tenant_id = str(tenant["_id"])

    # Check if user already exists for this tenant & email
    existing_user = await db.users.find_one({"tenant_id": tenant_id, "email": req.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists in the specified tenant"
        )

    hashed_pw = hash_password(req.password)
    user_doc = {
        "tenant_id": tenant_id,
        "email": req.email,
        "hashed_password": hashed_pw,
        "role": req.role or "member",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    user_res = await db.users.insert_one(user_doc)
    user_id = str(user_res.inserted_id)

    token = create_access_token(data={
        "user_id": user_id,
        "tenant_id": tenant_id,
        "role": req.role or "member",
        "email": req.email
    })

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user_id=user_id,
        tenant_id=tenant_id,
        role=req.role or "member"
    )

@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    db = get_db()
    user = await db.users.find_one({"email": req.email})
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
        "email": req.email
    })

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user_id=user_id,
        tenant_id=tenant_id,
        role=role
    )
