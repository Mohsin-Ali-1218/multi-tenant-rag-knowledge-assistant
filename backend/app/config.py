import os
import warnings
from dotenv import load_dotenv

# Directly load environment variables from .env file at startup
load_dotenv()

INSECURE_SECRET_PLACEHOLDERS = {
    "replace_with_a_random_secret",
    "your_jwt_secret",
    "secret",
    "changeme",
    "default",
    "password",
    "admin",
    "test_secret"
}

class Settings:
    @property
    def ENVIRONMENT(self) -> str:
        return os.getenv("ENVIRONMENT", os.getenv("APP_ENV", "development")).lower()

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
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if self.ENVIRONMENT in ["production", "prod"] and (not key or "your_gemini_api_key_here" in key):
            raise RuntimeError("Valid GEMINI_API_KEY must be configured in production")
        return key

    @property
    def JWT_SECRET(self) -> str:
        secret = os.getenv("JWT_SECRET", "").strip()
        if not secret:
            raise RuntimeError("JWT_SECRET must be set in the environment")

        if secret.lower() in INSECURE_SECRET_PLACEHOLDERS:
            if self.ENVIRONMENT in ["production", "prod"]:
                raise RuntimeError("Default or placeholder JWT_SECRET is prohibited in production")
            warnings.warn("Insecure default JWT_SECRET is in use. Set a secure random secret.", UserWarning)

        if self.ENVIRONMENT in ["production", "prod"] and len(secret) < 32:
            raise RuntimeError("JWT_SECRET must be at least 32 characters long in production")

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
