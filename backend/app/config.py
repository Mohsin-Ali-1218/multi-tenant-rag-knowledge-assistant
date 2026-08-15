import os
from dotenv import load_dotenv

# Directly load environment variables from .env file at startup
load_dotenv()

class Settings:
    @property
    def MONGODB_URI(self) -> str:
        return os.getenv("MONGODB_URI", "mongodb://localhost:27017")

    @property
    def MONGODB_DB_NAME(self) -> str:
        return os.getenv("MONGODB_DB_NAME", "multitenant_rag")

    @property
    def QDRANT_URL(self) -> str:
        return os.getenv("QDRANT_URL", "http://localhost:6333")

    @property
    def GEMINI_API_KEY(self) -> str:
        return os.getenv("GEMINI_API_KEY", "")

    @property
    def JWT_SECRET(self) -> str:
        secret = os.getenv("JWT_SECRET", "").strip()
        if not secret:
            raise RuntimeError("JWT_SECRET must be set in the environment")
        return secret

    @property
    def JWT_ALGORITHM(self) -> str:
        return os.getenv("JWT_ALGORITHM", "HS256")

    @property
    def ACCESS_TOKEN_EXPIRE_MINUTES(self) -> int:
        return int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

    @property
    def EMBEDDING_MODEL_NAME(self) -> str:
        return os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

settings = Settings()
