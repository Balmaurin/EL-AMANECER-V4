"""
Esquemas Pydantic completos para la API de Sheily AI
Define todos los modelos de request/response con validación automática
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, EmailStr, Field, HttpUrl, validator


# Enums
class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    ANALYST = "analyst"
    API_USER = "api_user"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# Base Models
class BaseResponse(BaseModel):
    """Modelo base para todas las respuestas"""

    success: bool = Field(
        default=True, description="Indica si la operación fue exitosa"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Timestamp de la respuesta"
    )
    request_id: Optional[str] = Field(None, description="ID único de la request")


class ErrorResponse(BaseModel):
    """Modelo para respuestas de error"""

    success: bool = Field(default=False)
    error: str = Field(..., description="Mensaje de error")
    error_code: str = Field(..., description="Código de error")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Detalles adicionales del error"
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class PaginationInfo(BaseModel):
    """Información de paginación"""

    page: int = Field(..., ge=1, description="Página actual")
    limit: int = Field(..., ge=1, le=100, description="Elementos por página")
    total: int = Field(..., ge=0, description="Total de elementos")
    total_pages: int = Field(..., ge=0, description="Total de páginas")
    has_next: bool = Field(default=False, description="Si hay página siguiente")
    has_prev: bool = Field(default=False, description="Si hay página anterior")


# Auth Schemas
class UserBase(BaseModel):
    """Información básica de usuario"""

    id: int = Field(..., description="ID único del usuario")
    email: EmailStr = Field(..., description="Email del usuario")
    username: str = Field(
        ..., min_length=3, max_length=50, description="Nombre de usuario"
    )
    full_name: str = Field(
        ..., min_length=1, max_length=100, description="Nombre completo"
    )
    is_active: bool = Field(default=True, description="Si el usuario está activo")
    role: UserRole = Field(default=UserRole.USER, description="Rol del usuario")
    created_at: datetime = Field(..., description="Fecha de creación")
    last_login: Optional[datetime] = Field(None, description="Último login")


class UserPreferences(BaseModel):
    """Preferencias del usuario"""

    theme: str = Field(default="dark", description="Tema de la interfaz")
    language: str = Field(default="es", description="Idioma preferido")
    notifications: bool = Field(default=True, description="Notificaciones activas")
    rag_enabled: bool = Field(default=True, description="RAG activado por defecto")


class UserResponse(UserBase):
    """Respuesta completa de usuario"""

    preferences: UserPreferences = Field(
        default_factory=UserPreferences, description="Preferencias del usuario"
    )


class RegisterRequest(BaseModel):
    """Request para registro de usuario"""

    email: EmailStr = Field(..., description="Email único del usuario")
    username: str = Field(
        ..., min_length=3, max_length=50, description="Nombre de usuario único"
    )
    password: str = Field(
        ..., min_length=8, max_length=128, description="Contraseña segura"
    )
    full_name: str = Field(
        ..., min_length=1, max_length=100, description="Nombre completo"
    )

    @validator("password")
    def validate_password(cls, v):
        """Validar complejidad de contraseña"""
        if not any(char.isdigit() for char in v):
            raise ValueError("La contraseña debe contener al menos un número")
        if not any(char.isupper() for char in v):
            raise ValueError("La contraseña debe contener al menos una mayúscula")
        if not any(char.islower() for char in v):
            raise ValueError("La contraseña debe contener al menos una minúscula")
        return v


class LoginRequest(BaseModel):
    """Request para login"""

    username: str = Field(..., description="Email o nombre de usuario")
    password: str = Field(..., description="Contraseña")


class TokenResponse(BaseModel):
    """Respuesta con tokens de autenticación"""

    access_token: str = Field(..., description="Token de acceso JWT")
    refresh_token: str = Field(..., description="Token de refresco JWT")
    token_type: str = Field(default="bearer", description="Tipo de token")
    expires_in: int = Field(
        ..., description="Segundos hasta expiración del access token"
    )
    user: UserResponse = Field(..., description="Información del usuario")


class RefreshTokenRequest(BaseModel):
    """Request para refrescar token"""

    refresh_token: str = Field(..., description="Token de refresco válido")


class APITokenCreateRequest(BaseModel):
    """Request para crear token de API"""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Nombre descriptivo del token",
    )
    permissions: Dict[str, bool] = Field(
        default_factory=lambda: {
            "chat": True,
            "rag": True,
            "conversations": False,
        },
        description="Permisos del token",
    )
    expires_days: Optional[int] = Field(
        None, ge=1, le=365, description="Días hasta expiración"
    )


class APITokenResponse(BaseModel):
    """Respuesta con token de API"""

    id: str = Field(..., description="ID del token")
    name: str = Field(..., description="Nombre del token")
    token: str = Field(..., description="Token de API (solo visible al crearlo)")
    permissions: Dict[str, bool] = Field(..., description="Permisos del token")
    created_at: datetime = Field(..., description="Fecha de creación")
    expires_at: Optional[datetime] = Field(None, description="Fecha de expiración")
    last_used: Optional[datetime] = Field(None, description="Último uso")


# Chat Schemas
class ChatRequest(BaseModel):
    """Request para enviar mensaje de chat"""

    message: str = Field(
        ..., min_length=1, max_length=10000, description="Mensaje del usuario"
    )
    system_prompt: Optional[str] = Field(
        None, max_length=2000, description="Prompt del sistema personalizado"
    )
    max_tokens: int = Field(
        512, ge=1, le=2048, description="Máximo tokens en respuesta"
    )
    temperature: float = Field(
        0.7, ge=0.0, le=2.0, description="Temperatura de generación"
    )
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling")
    rag_top_k: Optional[int] = Field(
        None, ge=1, le=20, description="Documentos RAG a recuperar"
    )
    rag_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Umbral de similitud RAG"
    )


class ModelInfo(BaseModel):
    """Información del modelo de IA"""

    name: str = Field(..., description="Nombre del modelo")
    size: str = Field(..., description="Tamaño del modelo")
    quantization: str = Field(..., description="Tipo de cuantización")
    max_tokens: int = Field(..., description="Máximo tokens del modelo")


class RAGInfo(BaseModel):
    """Información del sistema RAG"""

    documents_found: int = Field(..., description="Documentos encontrados")
    documents_used: int = Field(..., description="Documentos utilizados")
    avg_similarity: float = Field(..., description="Similitud promedio")
    search_time: float = Field(..., description="Tiempo de búsqueda en segundos")


class ChatResponse(BaseModel):
    """Respuesta completa del chat"""

    response: str = Field(..., description="Respuesta del asistente")
    conversation_id: str = Field(..., description="ID de la conversación")
    context_used: bool = Field(..., description="Si se utilizó contexto RAG")
    context_length: int = Field(..., description="Longitud del contexto utilizado")
    timestamp: float = Field(..., description="Timestamp de la respuesta")
    model_info: ModelInfo = Field(..., description="Información del modelo")
    rag_info: Optional[RAGInfo] = Field(None, description="Información del RAG")


class MessageValidationRequest(BaseModel):
    """Request para validar mensaje"""

    message: str = Field(
        ..., min_length=1, max_length=10000, description="Mensaje a validar"
    )


class MessageValidationResponse(BaseModel):
    """Respuesta de validación de mensaje"""

    valid: bool = Field(..., description="Si el mensaje es válido")
    message: str = Field(..., description="Mensaje original o sanitizado")
    length: int = Field(..., description="Longitud del mensaje")
    warnings: List[str] = Field(
        default_factory=list, description="Advertencias de validación"
    )


class ChatServiceInfo(BaseModel):
    """Información del servicio de chat"""

    status: ServiceStatus = Field(..., description="Estado del servicio")
    model_loaded: bool = Field(..., description="Si el modelo está cargado")
    rag_enabled: bool = Field(..., description="Si RAG está habilitado")
    concurrent_connections: int = Field(..., description="Conexiones concurrentes")
    max_concurrent_connections: int = Field(
        ..., description="Máximo conexiones concurrentes"
    )
    uptime_seconds: Optional[int] = Field(
        None, description="Tiempo de actividad en segundos"
    )


# Conversation Schemas
class ConversationBase(BaseModel):
    """Información básica de conversación"""

    id: str = Field(..., description="ID único de la conversación")
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Título de la conversación",
    )
    description: Optional[str] = Field(
        None, max_length=500, description="Descripción de la conversación"
    )
    tags: List[str] = Field(default_factory=list, description="Tags de la conversación")
    status: ConversationStatus = Field(
        default=ConversationStatus.ACTIVE,
        description="Estado de la conversación",
    )
    created_at: datetime = Field(..., description="Fecha de creación")
    updated_at: datetime = Field(..., description="Fecha de última actualización")


class ConversationSettings(BaseModel):
    """Configuración de conversación"""

    model: str = Field(default="default-model", description="Modelo a utilizar")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Temperatura de generación"
    )
    max_tokens: int = Field(default=512, ge=1, le=2048, description="Máximo tokens")
    rag_enabled: bool = Field(default=True, description="Si usar RAG")
    system_prompt: Optional[str] = Field(
        None, max_length=2000, description="Prompt del sistema"
    )


class ConversationCreateRequest(BaseModel):
    """Request para crear conversación"""

    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Título de la conversación",
    )
    description: Optional[str] = Field(None, max_length=500, description="Descripción")
    tags: List[str] = Field(default_factory=list, max_items=10, description="Tags")
    settings: ConversationSettings = Field(
        default_factory=ConversationSettings, description="Configuración"
    )


class ConversationUpdateRequest(BaseModel):
    """Request para actualizar conversación"""

    title: Optional[str] = Field(
        None, min_length=1, max_length=200, description="Nuevo título"
    )
    description: Optional[str] = Field(
        None, max_length=500, description="Nueva descripción"
    )
    tags: Optional[List[str]] = Field(None, max_items=10, description="Nuevos tags")
    settings: Optional[ConversationSettings] = Field(
        None, description="Nueva configuración"
    )


class ConversationResponse(ConversationBase):
    """Respuesta completa de conversación"""

    settings: ConversationSettings = Field(
        ..., description="Configuración de la conversación"
    )
    message_count: int = Field(default=0, description="Número de mensajes")
    last_message_at: Optional[datetime] = Field(None, description="Último mensaje")


class MessageInfo(BaseModel):
    """Información de un mensaje"""

    id: str = Field(..., description="ID único del mensaje")
    role: MessageRole = Field(..., description="Rol del mensaje")
    content: str = Field(..., description="Contenido del mensaje")
    context_used: Optional[bool] = Field(
        None, description="Si se usó contexto (solo assistant)"
    )
    context_length: Optional[int] = Field(
        None, description="Longitud del contexto (solo assistant)"
    )
    timestamp: datetime = Field(..., description="Timestamp del mensaje")


class ConversationDetailResponse(ConversationResponse):
    """Respuesta detallada de conversación con mensajes"""

    messages: List[MessageInfo] = Field(
        default_factory=list, description="Lista de mensajes"
    )


class ConversationListResponse(BaseModel):
    """Respuesta de lista de conversaciones"""

    conversations: List[ConversationResponse] = Field(
        ..., description="Lista de conversaciones"
    )
    pagination: PaginationInfo = Field(..., description="Información de paginación")


class ConversationListQuery(BaseModel):
    """Parámetros de query para listar conversaciones"""

    page: int = Field(default=1, ge=1, description="Página a obtener")
    limit: int = Field(default=20, ge=1, le=100, description="Elementos por página")
    tags: Optional[List[str]] = Field(None, description="Filtrar por tags")
    search: Optional[str] = Field(
        None,
        min_length=1,
        max_length=100,
        description="Buscar en título/descripción",
    )
    sort: Optional[str] = Field("updated_at", description="Campo de ordenamiento")


# RAG Schemas
class DocumentMetadata(BaseModel):
    """Metadata de documento RAG"""

    title: str = Field(
        ..., min_length=1, max_length=200, description="Título del documento"
    )
    author: Optional[str] = Field(
        None, max_length=100, description="Autor del documento"
    )
    category: str = Field(default="general", description="Categoría del documento")
    tags: List[str] = Field(default_factory=list, description="Tags del documento")
    source_url: Optional[HttpUrl] = Field(None, description="URL de origen")
    created_at: Optional[datetime] = Field(None, description="Fecha de creación")


class DocumentChunkInfo(BaseModel):
    """Información de chunk de documento"""

    chunk_id: int = Field(..., description="ID del chunk")
    total_chunks: int = Field(..., description="Total de chunks")
    position: str = Field(..., description="Posición del chunk")
    content_length: int = Field(..., description="Longitud del contenido")


class RAGSearchResult(BaseModel):
    """Resultado de búsqueda RAG"""

    id: str = Field(..., description="ID único del documento")
    content: str = Field(..., description="Contenido relevante")
    metadata: DocumentMetadata = Field(..., description="Metadata del documento")
    chunk_info: DocumentChunkInfo = Field(..., description="Información del chunk")
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Puntuación de similitud"
    )


class RAGSearchRequest(BaseModel):
    """Request para búsqueda RAG"""

    query: str = Field(
        ..., min_length=1, max_length=1000, description="Consulta de búsqueda"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Número de resultados")
    threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Umbral de similitud"
    )
    filters: Optional[Dict[str, Any]] = Field(None, description="Filtros adicionales")


class RAGSearchResponse(BaseModel):
    """Respuesta de búsqueda RAG"""

    query: str = Field(..., description="Consulta original")
    total_results: int = Field(..., description="Total de resultados encontrados")
    returned_results: int = Field(..., description="Resultados retornados")
    search_time: float = Field(..., description="Tiempo de búsqueda en segundos")
    results: List[RAGSearchResult] = Field(
        default_factory=list, description="Resultados de búsqueda"
    )


class DocumentUploadRequest(BaseModel):
    """Request para subir documento a RAG"""

    content: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="Contenido del documento",
    )
    metadata: DocumentMetadata = Field(..., description="Metadata del documento")
    chunk_size: int = Field(
        default=512, ge=100, le=2000, description="Tamaño de chunks"
    )
    chunk_overlap: int = Field(
        default=50, ge=0, le=500, description="Superposición de chunks"
    )


class DocumentUploadResponse(BaseModel):
    """Respuesta de subida de documento"""

    document_id: str = Field(..., description="ID del documento creado")
    chunks_created: int = Field(..., description="Número de chunks creados")
    total_tokens: int = Field(..., description="Total de tokens procesados")
    processing_time: float = Field(..., description="Tiempo de procesamiento")


class RAGCollectionInfo(BaseModel):
    """Información de colección RAG"""

    name: str = Field(..., description="Nombre de la colección")
    document_count: int = Field(..., ge=0, description="Número de documentos")
    chunk_count: int = Field(..., ge=0, description="Número de chunks")
    last_updated: datetime = Field(..., description="Última actualización")
    size_mb: float = Field(..., ge=0.0, description="Tamaño en MB")


class RAGInfoResponse(BaseModel):
    """Respuesta de información del sistema RAG"""

    collections: List[RAGCollectionInfo] = Field(
        ..., description="Colecciones disponibles"
    )
    embedding_model: str = Field(..., description="Modelo de embeddings utilizado")
    vector_dimension: int = Field(..., description="Dimensión de vectores")
    similarity_metric: str = Field(..., description="Métrica de similitud")


# Monitoring Schemas
class ServiceHealth(BaseModel):
    """Estado de salud de un servicio"""

    status: ServiceStatus = Field(..., description="Estado del servicio")
    response_time: Optional[float] = Field(
        None, description="Tiempo de respuesta en segundos"
    )
    message: Optional[str] = Field(None, description="Mensaje adicional")
    last_check: datetime = Field(
        default_factory=datetime.now, description="Última verificación"
    )


class SystemHealthResponse(BaseModel):
    """Respuesta completa de health check"""

    status: ServiceStatus = Field(..., description="Estado general del sistema")
    services: Dict[str, ServiceHealth] = Field(
        ..., description="Estado de cada servicio"
    )
    timestamp: float = Field(..., description="Timestamp de la verificación")


class SystemInfoResponse(BaseModel):
    """Respuesta de información del sistema"""

    name: str = Field(..., description="Nombre del sistema")
    version: str = Field(..., description="Versión del sistema")
    debug: bool = Field(..., description="Modo debug activado")
    services: Dict[str, Any] = Field(..., description="Información de servicios")
    system: Dict[str, Any] = Field(..., description="Información del sistema operativo")


class MetricsResponse(BaseModel):
    """Respuesta de métricas del sistema"""

    requests: Dict[str, Union[int, float]] = Field(
        ..., description="Métricas de requests"
    )
    performance: Dict[str, float] = Field(..., description="Métricas de performance")
    resources: Dict[str, float] = Field(..., description="Uso de recursos")
    ai_metrics: Dict[str, Union[int, float]] = Field(..., description="Métricas de IA")
    timestamp: float = Field(..., description="Timestamp de las métricas")


# Rate Limiting
class RateLimitInfo(BaseModel):
    """Información de rate limiting"""

    limit: int = Field(..., description="Límite de requests")
    remaining: int = Field(..., description="Requests restantes")
    reset_time: datetime = Field(..., description="Tiempo de reset")
    retry_after: Optional[int] = Field(None, description="Segundos para reintentar")


class RateLimitExceededResponse(BaseModel):
    """Respuesta cuando se excede el rate limit"""

    success: bool = Field(default=False)
    error: str = Field(default="Rate limit exceeded", description="Mensaje de error")
    retry_after: int = Field(..., description="Segundos para reintentar")
    limit: int = Field(..., description="Límite de requests")
    reset_time: datetime = Field(..., description="Tiempo de reset")


# WebSocket Messages
class WebSocketMessage(BaseModel):
    """Mensaje WebSocket base"""

    type: str = Field(..., description="Tipo de mensaje")
    conversation_id: str = Field(..., description="ID de conversación")


class WebSocketChatMessage(WebSocketMessage):
    """Mensaje de chat WebSocket"""

    type: str = Field(default="chat", description="Tipo de mensaje")
    message: str = Field(..., description="Mensaje del usuario")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Parámetros adicionales"
    )


class WebSocketResponse(WebSocketMessage):
    """Respuesta WebSocket"""

    type: str = Field(default="response", description="Tipo de mensaje")
    response: str = Field(..., description="Respuesta del asistente")
    context_used: bool = Field(..., description="Si se usó contexto")
    context_length: int = Field(..., description="Longitud del contexto")
    timestamp: float = Field(..., description="Timestamp")
    model_info: ModelInfo = Field(..., description="Información del modelo")
    rag_info: Optional[RAGInfo] = Field(None, description="Información RAG")


class WebSocketError(WebSocketMessage):
    """Mensaje de error WebSocket"""

    type: str = Field(default="error", description="Tipo de mensaje")
    error: str = Field(..., description="Mensaje de error")
    code: Optional[str] = Field(None, description="Código de error")
