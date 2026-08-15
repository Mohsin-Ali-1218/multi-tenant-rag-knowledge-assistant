from typing import Optional, Any
from pymongo import AsyncMongoClient
from backend.app.config import settings

_mongo_client: Optional[AsyncMongoClient] = None

def get_mongo_client() -> AsyncMongoClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = AsyncMongoClient(settings.MONGODB_URI)
    return _mongo_client

def get_db() -> Any:
    client = get_mongo_client()
    return client[settings.MONGODB_DB_NAME]

async def close_mongo_client():
    global _mongo_client
    if _mongo_client is not None:
        await _mongo_client.close()
        _mongo_client = None

async def init_mongo_indexes():
    db = get_db()
    # Compound indexes for tenant isolation & performance
    await db.documents.create_index([("tenant_id", 1), ("_id", 1)])
    await db.conversations.create_index([("tenant_id", 1), ("_id", 1)])
    await db.messages.create_index([("tenant_id", 1), ("_id", 1)])
    await db.users.create_index([("tenant_id", 1), ("email", 1)], unique=True)
    await db.tenants.create_index("slug", unique=True)
