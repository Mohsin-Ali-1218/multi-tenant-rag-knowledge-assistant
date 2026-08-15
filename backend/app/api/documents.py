from datetime import datetime, timezone
from typing import List
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, status
from backend.app.db.mongo import get_db
from backend.app.db.vector import upsert_chunks
from backend.app.models.schemas import DocumentCreate, DocumentResponse
from backend.app.services.auth import get_current_user
from backend.app.services.embedding import chunk_text, get_embeddings

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    doc_input: DocumentCreate,
    current_user: dict = Depends(get_current_user)
):
    tenant_id = current_user["tenant_id"]
    db = get_db()

    chunks = chunk_text(doc_input.raw_text)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document text cannot be empty"
        )

    # Embed chunks
    embeddings = get_embeddings(chunks)

    # Save metadata to Mongo
    now_iso = datetime.now(timezone.utc).isoformat()
    mongo_doc = {
        "tenant_id": tenant_id,
        "title": doc_input.title,
        "raw_text": doc_input.raw_text,
        "chunk_count": len(chunks),
        "uploaded_at": now_iso
    }
    result = await db.documents.insert_one(mongo_doc)
    doc_id = str(result.inserted_id)

    # Upsert vector chunks to Qdrant collection for this tenant
    await upsert_chunks(
        tenant_id=tenant_id,
        doc_id=doc_id,
        doc_title=doc_input.title,
        chunks=chunks,
        embeddings=embeddings
    )

    return DocumentResponse(
        id=doc_id,
        tenant_id=tenant_id,
        title=doc_input.title,
        raw_text=doc_input.raw_text,
        chunk_count=len(chunks),
        uploaded_at=now_iso
    )

@router.get("", response_model=List[DocumentResponse])
async def list_documents(current_user: dict = Depends(get_current_user)):
    tenant_id = current_user["tenant_id"]
    db = get_db()

    cursor = db.documents.find({"tenant_id": tenant_id})
    documents = []
    async for doc in cursor:
        documents.append(DocumentResponse(
            id=str(doc["_id"]),
            tenant_id=doc["tenant_id"],
            title=doc["title"],
            raw_text=doc["raw_text"],
            chunk_count=doc["chunk_count"],
            uploaded_at=doc["uploaded_at"]
        ))
    return documents

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user)
):
    tenant_id = current_user["tenant_id"]
    db = get_db()

    try:
        obj_id = ObjectId(document_id)
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Search for document by ID regardless of tenant to distinguish 403 vs 404
    doc = await db.documents.find_one({"_id": obj_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Strict multi-tenant isolation check
    if doc.get("tenant_id") != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: Document belongs to a different tenant"
        )

    return DocumentResponse(
        id=str(doc["_id"]),
        tenant_id=doc["tenant_id"],
        title=doc["title"],
        raw_text=doc["raw_text"],
        chunk_count=doc["chunk_count"],
        uploaded_at=doc["uploaded_at"]
    )
