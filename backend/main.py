import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.auth import router as auth_router
from backend.app.api.documents import router as documents_router
from backend.app.api.chat import router as chat_router
from backend.app.db.mongo import init_mongo_indexes, close_mongo_client
from backend.app.db.vector import close_qdrant_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multitenant_rag")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing application resources and database indexes...")
    try:
        await init_mongo_indexes()
        logger.info("MongoDB indexes initialized successfully.")
    except Exception as e:
        logger.warning(f"Database index initialization deferred (containers may be starting): {e}")
    yield
    logger.info("Closing application connections...")
    await close_mongo_client()
    await close_qdrant_client()
    logger.info("Application shutdown complete.")

app = FastAPI(
    title="Multi-Tenant RAG Knowledge Assistant",
    description="Isolated multi-tenant RAG solution with FastAPI, PyMongo native async, Qdrant, BGE-M3 embeddings, and Gemini 3.5 Flash.",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(chat_router)

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Multi-Tenant RAG Knowledge Assistant",
        "version": "0.1.0"
    }
