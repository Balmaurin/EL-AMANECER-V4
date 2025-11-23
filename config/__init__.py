"""
Sheily MCP Enterprise Configuration
====================================

Centralized configuration management for the entire enterprise system.
"""

import os
from pathlib import Path
from typing import List, Optional


class Settings:
    """Global application settings"""

    def __init__(self):
        # Environment
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
        self.DEBUG = self.ENVIRONMENT == "development"

        # API Configuration
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))
        self.API_WORKERS = int(os.getenv("API_WORKERS", "4"))

        # Database Configuration
        self.DATABASE_URL = os.getenv(
            "DATABASE_URL", "postgresql://user:password@localhost/sheily_mcp"
        )
        self.REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "http://localhost:6333")

        # Security Configuration
        self.SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
        self.JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
        self.JWT_EXPIRATION = int(os.getenv("JWT_EXPIRATION", "3600"))

        # SSL Configuration
        self.SSL_CERT_PATH = os.getenv("SSL_CERT_PATH")
        self.SSL_KEY_PATH = os.getenv("SSL_KEY_PATH")

        # CORS Configuration
        self.CORS_ORIGINS = os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        self.ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")

        # Frontend Configuration
        self.SERVE_FRONTEND = os.getenv("SERVE_FRONTEND", "false").lower() == "true"
        self.FRONTEND_PATH = os.getenv("FRONTEND_PATH", "Frontend/out")

        # ML Configuration
        self.ML_MODELS_PATH = os.getenv("ML_MODELS_PATH", "models")
        self.QLoRA_CHECKPOINT_DIR = os.getenv(
            "QLoRA_CHECKPOINT_DIR", "models/qlora_checkpoints"
        )

        # Logging Configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = os.getenv("LOG_FILE", "logs/sheily_mcp.log")

        # Enterprise Features
        self.ENABLE_ML = os.getenv("ENABLE_ML", "true").lower() == "true"
        self.ENABLE_AGENT_REGISTRY = (
            os.getenv("ENABLE_AGENT_REGISTRY", "true").lower() == "true"
        )
        self.ENABLE_ORCHESTRATING = (
            os.getenv("ENABLE_ORCHESTRATING", "true").lower() == "true"
        )
        self.ENABLE_MONITORING = (
            os.getenv("ENABLE_MONITORING", "true").lower() == "true"
        )

        # Monitoring and Analytics
        self.PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9090"))
        self.GRAFANA_PORT = int(os.getenv("GRAFANA_PORT", "3001"))

        # Agent Configuration
        self.MAX_AGENTS_PER_TYPE = int(os.getenv("MAX_AGENTS_PER_TYPE", "50"))
        self.AGENT_HEARTBEAT_INTERVAL = int(os.getenv("AGENT_HEARTBEAT_INTERVAL", "30"))
        self.AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "120"))

        # Task Configuration
        self.MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "100"))
        self.TASK_TIMEOUT_DEFAULT = int(os.getenv("TASK_TIMEOUT_DEFAULT", "300"))


# Global settings instance
settings = Settings()

# Export settings
__all__ = ["settings", "Settings"]
