"""
Unified Configuration System for Sheily AI Enterprise
Centralizes all configuration to avoid conflicts and duplication
"""

import os
from typing import List, Optional

from pydantic_settings import BaseSettings


class UnifiedSettings(BaseSettings):
    """Unified settings for the entire Sheily AI system"""

    # === BASIC APP SETTINGS ===
    app_name: str = "Sheily AI Enterprise"
    version: str = "3.1.0"
    environment: str = os.getenv("ENVIRONMENT", "development")

    # === SERVER SETTINGS ===
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    reload: bool = os.getenv("RELOAD", "true").lower() == "true"

    # === API SETTINGS ===
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]

    # === DATABASE SETTINGS ===
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/sheily_ai.db")
    connection_pool_size: int = 10
    connection_timeout: int = 30

    # === CACHE SETTINGS ===
    cache_enabled: bool = True
    cache_type: str = os.getenv("CACHE_TYPE", "file")
    cache_url: str = os.getenv("CACHE_URL", "redis://localhost:6379")
    cache_ttl: int = 3600

    # === REDIS SETTINGS ===
    redis_url: Optional[str] = os.getenv("REDIS_URL")

    # === AI/ML SETTINGS ===
    model_path: str = "models/llama-3.2-3b/Llama-3.2-3B-Instruct-f16.gguf"
    llama_binary_path: str = "llama-cli.exe"
    model_max_tokens: int = 512
    model_temperature: float = 0.7
    model_threads: int = 4
    model_timeout: int = 30

    # === RAG SETTINGS ===
    corpus_root: str = "data"
    context_max_length: int = 2000
    max_context_docs: int = 3
    rag_similarity_threshold: float = 0.7
    rag_top_k: int = 3

    # === SECURITY SETTINGS ===
    security_enabled: bool = True
    secret_key: str = os.getenv("SECRET_KEY", "")
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "")
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    max_queries_per_minute: int = 60
    max_query_length: int = 10000
    require_authentication: bool = False

    # === LOGGING SETTINGS ===
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = "logs/sheily_ai.log"
    log_max_size: int = 10485760  # 10MB
    log_backup_count: int = 5

    # === MONITORING SETTINGS ===
    monitoring_enabled: bool = True
    metrics_collection_interval: int = 30
    metrics_retention_hours: int = 24

    # === RESOURCE LIMITS ===
    max_memory_usage: int = 2048  # MB
    max_cpu_usage: int = 80  # %

    # === FEATURE FLAGS ===
    features: dict = {
        "chat": True,
        "rag": True,
        "memory": True,
        "monitoring": True,
        "security": True,
        "enterprise_mode": True,
    }

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Global unified settings instance
settings = UnifiedSettings()


# Validación y refuerzo automático de secretos
def _validate_security(_s: UnifiedSettings):
    # En producción no se permiten placeholders ni vacíos
    if _s.environment.lower() == "production":
        if (not _s.secret_key) or ("secret-key" in _s.secret_key):
            raise RuntimeError("SECRET_KEY inválida en producción")
        if (not _s.jwt_secret_key) or ("jwt-secret-key" in _s.jwt_secret_key):
            raise RuntimeError("JWT_SECRET_KEY inválida en producción")

    # Si faltan claves (dev), generar dinámicamente para evitar valores inseguros
    if not _s.secret_key:
        _s.secret_key = os.getenv("SECRET_KEY_GENERATED", "") or os.urandom(32).hex()
    if not _s.jwt_secret_key:
        _s.jwt_secret_key = (
            os.getenv("JWT_SECRET_KEY_GENERATED", "") or os.urandom(32).hex()
        )


_validate_security(settings)
