import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
import jwt
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from backend.app.config import settings
from backend.app.db.mongo import get_db

security = HTTPBearer()

def hash_password(password: str) -> str:
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    plain_bytes = plain_password.encode('utf-8')
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(plain_bytes, hashed_bytes)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": now})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    token = credentials.credentials
    payload = decode_access_token(token)
    user_id = payload.get("user_id")
    tenant_id = payload.get("tenant_id")
    
    if not user_id or not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required claims (user_id/tenant_id)",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_obj_id = ObjectId(user_id)
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user identifier in token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    db = get_db()
    user = await db.users.find_one({"_id": user_obj_id})
    if not user or str(user.get("tenant_id")) != str(tenant_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or tenant membership revoked",
            headers={"WWW-Authenticate": "Bearer"},
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
            detail="User account is disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify tenant status
    try:
        tenant_obj_id = ObjectId(tenant_id)
        tenant = await db.tenants.find_one({"_id": tenant_obj_id})
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Tenant not found or deleted",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if (
            tenant.get("disabled") is True
            or tenant.get("is_active") is False
            or str(tenant.get("status", "")).lower() in ["disabled", "inactive", "suspended"]
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Tenant account is disabled",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid tenant identifier in token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Fetch live role from database state rather than trusting potentially outdated JWT claim
    current_role = user.get("role", "member")

    return {
        "user_id": str(user["_id"]),
        "tenant_id": str(user["tenant_id"]),
        "role": current_role,
        "email": user.get("email", payload.get("email", ""))
    }

def require_roles(allowed_roles: List[str]):
    async def dependency(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        if current_user.get("role") not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: Insufficient permissions (requires one of: {', '.join(allowed_roles)})"
            )
        return current_user
    return dependency

require_admin = require_roles(["owner", "admin"])
