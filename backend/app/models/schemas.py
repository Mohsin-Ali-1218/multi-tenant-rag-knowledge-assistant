from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field

# --- Auth Schemas ---
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    tenant_name: str
    tenant_slug: Optional[str] = None
    role: Optional[str] = "admin"

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    tenant_id: str
    role: str

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
