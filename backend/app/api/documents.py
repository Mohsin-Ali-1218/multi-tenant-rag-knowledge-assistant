import logging
from datetime import datetime, timezone
from typing import List, Optional
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from backend.app.config import settings
from backend.app.db.mongo import get_db
from backend.app.db.vector import upsert_chunks, delete_document_chunks
from backend.app.models.schemas import DocumentCreate, DocumentResponse
from backend.app.services.auth import get_current_user
from backend.app.services.embedding import chunk_text, get_embeddings
from backend.app.services.extractor import process_uploaded_document

logger = logging.getLogger("multitenant_rag.documents")

INDEXING_FAILURE_DETAIL = "Failed to index document due to an internal processing error."

router = APIRouter(prefix="/documents", tags=["Documents"])

async def rollback_document_indexing(
    db,
    tenant_id: str,
    doc_id: str,
    mongo_id: ObjectId,
    original_err: Exception,
) -> None:
    """
    Performs compensating rollback deletion across MongoDB and Qdrant.
    Preserves original exception and traceback in server logs with tenant and document identifiers.
    Logs cleanup failures separately and attempts both tenant-scoped cleanups independently.
    Returns a stable, generic HTTP 500 message without disclosing internal error details.
    """
    # 1. Preserve original exception and traceback in server logs with tenant and document identifiers
    logger.error(
        f"Document vector indexing failed for tenant {tenant_id}, doc {doc_id}: {original_err}",
        exc_info=True
    )

    # 2. Tenant-scoped MongoDB rollback
    try:
        await db.documents.delete_one({"_id": mongo_id, "tenant_id": tenant_id})
    except Exception as mongo_err:
        logger.critical(
            f"Compensating rollback failure (MongoDB) for tenant {tenant_id}, doc {doc_id}: {mongo_err}",
            exc_info=True
        )

    # 3. Tenant-scoped Qdrant rollback (attempted independently of MongoDB cleanup result)
    try:
        await delete_document_chunks(tenant_id=tenant_id, doc_id=doc_id)
    except Exception as qdrant_err:
        logger.critical(
            f"Compensating rollback failure (Qdrant) for tenant {tenant_id}, doc {doc_id}: {qdrant_err}",
            exc_info=True
        )

    # 4. Return stable, generic HTTP 500 message without disclosing internal error details
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=INDEXING_FAILURE_DETAIL
    )

@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    doc_input: DocumentCreate,
    current_user: dict = Depends(get_current_user)
):
    tenant_id = current_user["tenant_id"]
    db = get_db()

    clean_text = doc_input.raw_text.strip()
    if not clean_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document text cannot be empty"
        )

    if len(clean_text) > settings.MAX_EXTRACTED_TEXT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document text length ({len(clean_text)}) exceeds limit of {settings.MAX_EXTRACTED_TEXT_LENGTH} chars."
        )

    chunks = chunk_text(clean_text)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document text cannot be empty"
        )

    if len(chunks) > settings.MAX_DOCUMENT_CHUNKS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document chunk count ({len(chunks)}) exceeds limit of {settings.MAX_DOCUMENT_CHUNKS} chunks."
        )

    # Embed chunks
    embeddings = get_embeddings(chunks)

    # Save metadata to Mongo
    now_iso = datetime.now(timezone.utc).isoformat()
    mongo_doc = {
        "tenant_id": tenant_id,
        "title": doc_input.title,
        "filename": None,
        "file_type": "raw",
        "raw_text": clean_text,
        "chunk_count": len(chunks),
        "uploaded_at": now_iso
    }
    result = await db.documents.insert_one(mongo_doc)
    doc_id = str(result.inserted_id)

    # Upsert vector chunks to Qdrant collection for this tenant
    try:
        await upsert_chunks(
            tenant_id=tenant_id,
            doc_id=doc_id,
            doc_title=doc_input.title,
            chunks=chunks,
            embeddings=embeddings,
            metadatas=[{"page_number": None, "filename": None} for _ in chunks],
            filename=None
        )
    except Exception as e:
        await rollback_document_indexing(
            db=db,
            tenant_id=tenant_id,
            doc_id=doc_id,
            mongo_id=result.inserted_id,
            original_err=e
        )

    return DocumentResponse(
        id=doc_id,
        tenant_id=tenant_id,
        title=doc_input.title,
        filename=None,
        file_type="raw",
        raw_text=clean_text,
        chunk_count=len(chunks),
        uploaded_at=now_iso
    )

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    tenant_id = current_user["tenant_id"]
    db = get_db()

    # Read bytes with streaming size check
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=f"Uploaded file size ({len(content) / (1024 * 1024):.1f} MB) exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE_MB} MB."
        )
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    # Process and extract (validates extension, magic bytes, resource limits)
    extraction = process_uploaded_document(
        file_bytes=content,
        original_filename=file.filename,
        title=title
    )

    # Embed chunks
    embeddings = get_embeddings(extraction.chunks)

    # Save metadata to Mongo
    now_iso = datetime.now(timezone.utc).isoformat()
    mongo_doc = {
        "tenant_id": tenant_id,
        "title": extraction.title,
        "filename": extraction.filename,
        "file_type": extraction.file_type,
        "raw_text": extraction.raw_text,
        "chunk_count": len(extraction.chunks),
        "uploaded_at": now_iso
    }
    result = await db.documents.insert_one(mongo_doc)
    doc_id = str(result.inserted_id)

    # Upsert vector chunks to Qdrant collection for this tenant
    try:
        await upsert_chunks(
            tenant_id=tenant_id,
            doc_id=doc_id,
            doc_title=extraction.title,
            chunks=extraction.chunks,
            embeddings=embeddings,
            metadatas=extraction.chunk_metadatas,
            filename=extraction.filename
        )
    except Exception as e:
        await rollback_document_indexing(
            db=db,
            tenant_id=tenant_id,
            doc_id=doc_id,
            mongo_id=result.inserted_id,
            original_err=e
        )

    return DocumentResponse(
        id=doc_id,
        tenant_id=tenant_id,
        title=extraction.title,
        filename=extraction.filename,
        file_type=extraction.file_type,
        raw_text=extraction.raw_text,
        chunk_count=len(extraction.chunks),
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
            filename=doc.get("filename"),
            file_type=doc.get("file_type"),
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
        filename=doc.get("filename"),
        file_type=doc.get("file_type"),
        raw_text=doc["raw_text"],
        chunk_count=doc["chunk_count"],
        uploaded_at=doc["uploaded_at"]
    )
