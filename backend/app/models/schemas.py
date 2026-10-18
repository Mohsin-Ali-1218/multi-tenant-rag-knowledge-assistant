from typing import List, Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field

# --- Auth Schemas ---
class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: EmailStr
    password: str = Field(..., min_length=6)
    tenant_name: str = Field(..., min_length=1)

class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: EmailStr
    password: str
    tenant_id: Optional[str] = None
    tenant_slug: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    tenant_id: str
    role: str

# --- Invitation Schemas ---
class CreateInvitationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: EmailStr
    role: Optional[str] = Field("member", pattern="^(member|admin)$")

class InvitationResponse(BaseModel):
    id: str
    tenant_id: str
    email: str
    role: str
    invitation_token: str
    expires_at: str
    created_at: str

class InvitationItemResponse(BaseModel):
    id: str
    tenant_id: str
    email: str
    role: str
    status: str
    created_at: str
    expires_at: str
    used_at: Optional[str] = None

class AcceptInvitationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    invitation_token: str = Field(..., min_length=10)
    email: EmailStr
    password: str = Field(..., min_length=6)

# --- Document Schemas ---
class DocumentCreate(BaseModel):
    title: str
    raw_text: str

class DocumentResponse(BaseModel):
    id: str
    tenant_id: str
    title: str
    raw_text: str
    chunk_count: int
    uploaded_at: str
    filename: Optional[str] = None
    file_type: Optional[str] = None

# --- Chat & RAG Schemas ---
class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str

class SourceItem(BaseModel):
    document_title: str
    chunk_text: str
    score: float = 0.0

class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: List[SourceItem]

class MessageItem(BaseModel):
    id: str
    conversation_id: str
    tenant_id: str
    role: str
    content: str
    sources: List[SourceItem] = []
    created_at: str

class ConversationMessagesResponse(BaseModel):
    conversation_id: str
    tenant_id: str
    messages: List[MessageItem]
