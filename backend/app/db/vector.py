import uuid
import logging
from typing import Optional, List, Dict, Any
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from backend.app.config import settings

logger = logging.getLogger("multitenant_rag.vector")

_qdrant_client: Optional[AsyncQdrantClient] = None

def get_tenant_collection_name(tenant_id: str) -> str:
    safe_tenant = str(tenant_id).replace("-", "_")
    return f"tenant_{safe_tenant}_docs"

def get_qdrant_client() -> AsyncQdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL)
    return _qdrant_client

async def close_qdrant_client():
    global _qdrant_client
    if _qdrant_client is not None:
        await _qdrant_client.close()
        _qdrant_client = None

async def ensure_tenant_collection(tenant_id: str, vector_size: int = 1024):
    client = get_qdrant_client()
    collection_name = get_tenant_collection_name(tenant_id)
    collections = await client.get_collections()
    collection_names = [c.name for c in collections.collections]
    if collection_name not in collection_names:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

async def upsert_chunks(
    tenant_id: str,
    doc_id: str,
    doc_title: str,
    chunks: List[str],
    embeddings: List[List[float]],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    filename: Optional[str] = None,
):
    if not chunks or not embeddings:
        return
    client = get_qdrant_client()
    collection_name = get_tenant_collection_name(tenant_id)
    vector_size = len(embeddings[0])
    await ensure_tenant_collection(tenant_id, vector_size=vector_size)

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        point_id = str(uuid.uuid4())
        chunk_meta = metadatas[i] if metadatas and i < len(metadatas) else {}
        payload = {
            "document_id": str(doc_id),
            "tenant_id": str(tenant_id),
            "document_title": doc_title,
            "filename": chunk_meta.get("filename") or filename,
            "page_number": chunk_meta.get("page_number"),
            "chunk_index": i,
            "text": chunk,
        }
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    await client.upsert(collection_name=collection_name, points=points)

async def delete_document_chunks(tenant_id: str, doc_id: str):
    client = get_qdrant_client()
    collection_name = get_tenant_collection_name(tenant_id)
    collections = await client.get_collections()
    collection_names = [c.name for c in collections.collections]
    if collection_name not in collection_names:
        return

    from qdrant_client.models import Filter, FieldCondition, MatchValue
    try:
        await client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(key="document_id", match=MatchValue(value=str(doc_id))),
                    FieldCondition(key="tenant_id", match=MatchValue(value=str(tenant_id)))
                ]
            )
        )
    except Exception as e:
        logger.error(
            f"Failed to delete Qdrant points for doc {doc_id} in tenant {tenant_id}: {e}",
            exc_info=True
        )
        raise

async def search_tenant_docs(
    tenant_id: str,
    query_vector: List[float],
    limit: int = 4,
    score_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    client = get_qdrant_client()
    collection_name = get_tenant_collection_name(tenant_id)
    
    collections = await client.get_collections()
    collection_names = [c.name for c in collections.collections]
    if collection_name not in collection_names:
        return []

    threshold = score_threshold if score_threshold is not None else settings.RAG_SIMILARITY_THRESHOLD

    if hasattr(client, "query_points"):
        response = await client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=threshold,
            with_payload=True
        )
        results = response.points
    else:
        results = await client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=threshold,
            with_payload=True
        )

    hits = []
    for hit in results:
        score = float(hit.score)
        if score >= threshold:
            hits.append({
                "document_id": hit.payload.get("document_id"),
                "document_title": hit.payload.get("document_title"),
                "filename": hit.payload.get("filename"),
                "page_number": hit.payload.get("page_number"),
                "chunk_text": hit.payload.get("text"),
                "score": score,
                "chunk_index": hit.payload.get("chunk_index"),
            })
    return hits
