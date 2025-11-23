"""
Modelos base de SQLAlchemy para Sheily AI
Configuración de base de datos y modelos comunes
"""

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from ..config.settings import settings

# Base para todos los modelos
Base = declarative_base()


class TimestampMixin:
    """Mixin para timestamps automáticos"""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class User(Base, TimestampMixin):
    """Modelo de usuario"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    username: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="user", nullable=False)
    preferences: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, default=dict)

    # Relaciones
    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"


class Conversation(Base, TimestampMixin):
    """Modelo de conversación"""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata", JSON, default=dict
    )

    # Relaciones
    user: Mapped["User"] = relationship("User", back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title={self.title[:50]}..., user_id={self.user_id})>"  # noqa: E501


class Message(Base, TimestampMixin):
    """Modelo de mensaje"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer)
    model_name: Mapped[Optional[str]] = mapped_column(String(100))
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata", JSON, default=dict
    )

    # Relaciones
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )

    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role={self.role}, tokens={self.tokens_used})>"  # noqa: E501


class UsageMetrics(Base, TimestampMixin):
    """Modelo para métricas de uso"""

    __tablename__ = "usage_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    session_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    method: Mapped[str] = mapped_column(String(10), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata", JSON, default=dict
    )

    def __repr__(self) -> str:
        return f"<UsageMetrics(id={self.id}, endpoint={self.endpoint}, status={self.status_code})>"  # noqa: E501


class APIToken(Base, TimestampMixin):
    """Modelo para tokens de API"""

    __tablename__ = "api_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    token_hash: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    permissions: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    # Relaciones
    user: Mapped["User"] = relationship("User")

    def __repr__(self) -> str:
        return f"<APIToken(id={self.id}, name={self.name}, active={self.is_active})>"  # noqa: E501


class Document(Base, TimestampMixin):
    """Modelo para documentos en el sistema RAG"""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    checksum: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True
    )
    status: Mapped[str] = mapped_column(
        String(50), default="processing", nullable=False
    )  # processing, indexed, failed
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata", JSON, default=dict
    )
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"  # noqa: E501


class DocumentChunk(Base, TimestampMixin):
    """Modelo para chunks de documentos indexados"""

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[list[float]]] = mapped_column(JSON)  # Vector embedding
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata", JSON, default=dict
    )

    # Relaciones
    document: Mapped["Document"] = relationship("Document")

    def __repr__(self) -> str:
        return f"<DocumentChunk(id={self.id}, doc_id={self.document_id}, chunk={self.chunk_index})>"  # noqa: E501


# Índices adicionales para performance
Index(
    "idx_messages_conversation_created",
    Message.conversation_id,
    Message.created_at,
)
Index(
    "idx_conversations_user_active",
    Conversation.user_id,
    Conversation.is_active,
)
Index(
    "idx_usage_metrics_user_created",
    UsageMetrics.user_id,
    UsageMetrics.created_at,
)
Index("idx_documents_status", Document.status)
Index("idx_document_chunks_document", DocumentChunk.document_id)


# Funciones de utilidad
def get_db_url() -> str:
    """Obtener URL de base de datos"""
    return settings.database.database_url


def get_redis_url() -> str:
    """Obtener URL de Redis"""
    return settings.database.redis_url


# Exportar modelos
__all__ = [
    "Base",
    "TimestampMixin",
    "User",
    "Conversation",
    "Message",
    "UsageMetrics",
    "APIToken",
    "Document",
    "DocumentChunk",
    "get_db_url",
    "get_redis_url",
]
