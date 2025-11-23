"""
Knowledge API - Gestión de base de conocimientos y RAG
Extraído de backend/chat_server.py
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# Importar knowledge base del módulo websocket (compartido)
try:
    from .websocket import knowledge_base, CHROMA_AVAILABLE
except ImportError:
    # Fallback si no está disponible
    knowledge_base = None
    CHROMA_AVAILABLE = False


logger = logging.getLogger(__name__)

class KnowledgeItem(BaseModel):
    text: str
    metadata: Optional[Dict] = None

class KnowledgeSearchResult(BaseModel):
    document: str
    metadata: Dict
    distance: float
    score: float

class KnowledgeSearchResponse(BaseModel):
    results: List[KnowledgeSearchResult]
    query: str
    total_found: int

router = APIRouter()


@router.post("/add")
async def add_knowledge(item: KnowledgeItem):
    """Añadir conocimiento a la base RAG"""
    try:
        if not knowledge_base:
            raise HTTPException(status_code=503, detail="Sistema de conocimiento no disponible")

        result = await knowledge_base.add_knowledge(item.text, item.metadata)

        if not result.get("success"):
            error_msg = result.get("error", "Error desconocido")
            raise HTTPException(status_code=500, detail=f"Error añadiendo conocimiento: {error_msg}")

        return {
            "success": True,
            "message": "Conocimiento añadido correctamente",
            "document_id": result.get("document_id"),
            "chunks_added": result.get("chunks_added", 0),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error añadiendo conocimiento: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(
    query: str = Query(..., min_length=1, description="Término de búsqueda"),
    limit: int = Query(5, ge=1, le=20, description="Número máximo de resultados")
):
    """Buscar en la base de conocimientos"""
    try:
        if not knowledge_base:
            raise HTTPException(status_code=503, detail="Sistema de conocimiento no disponible")

        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query de búsqueda requerido")

        result = await knowledge_base.search_knowledge(query.strip(), limit)

        # Convertir a formato Pydantic
        search_results = []
        for item in result.get("results", []):
            search_results.append(KnowledgeSearchResult(
                document=item.get("document", ""),
                metadata=item.get("metadata", {}),
                distance=item.get("distance", 0.0),
                score=item.get("score", 0.0),
            ))

        return KnowledgeSearchResponse(
            results=search_results,
            query=result.get("query", query),
            total_found=result.get("total_found", 0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error buscando conocimiento: {e}")
        raise HTTPException(status_code=500, detail="Error en búsqueda")


@router.get("/stats")
async def get_knowledge_stats():
    """Obtener estadísticas de la base de conocimientos"""
    try:
        if not knowledge_base:
            return {
                "available": False,
                "reason": "Sistema de conocimiento no inicializado",
            }

        stats = {
            "available": True,
            "chroma_available": CHROMA_AVAILABLE,
            "collection_initialized": knowledge_base.collection is not None,
        }

        if knowledge_base.collection:
            try:
                stats["documents_count"] = knowledge_base.collection.count()
                stats["collection_name"] = knowledge_base.collection.name
                stats["collection_metadata"] = knowledge_base.collection.metadata
            except Exception as e:
                logger.warning(f"Error obteniendo stats detallados: {e}")
                stats["documents_count"] = 0
                stats["collection_error"] = str(e)

        return stats

    except Exception as e:
        logger.error(f"Error obteniendo estadísticas de conocimiento: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo estadísticas")


@router.delete("/documents/{document_id}")
async def delete_knowledge_document(document_id: str):
    """Eliminar documento específico de la base de conocimientos"""
    try:
        if not knowledge_base or not knowledge_base.collection:
            raise HTTPException(status_code=503, detail="Sistema de conocimiento no disponible")

        # ChromaDB delete operation
        try:
            knowledge_base.collection.delete(ids=[document_id])
            return {"success": True, "message": f"Documento {document_id} eliminado"}
        except Exception as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail="Documento no encontrado")
            else:
                raise HTTPException(status_code=500, detail=f"Error eliminando documento: {e}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando documento {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/documents")
async def list_knowledge_documents(limit: int = Query(50, ge=1, le=200)):
    """Listar documentos en la base de conocimientos"""
    try:
        if not knowledge_base or not knowledge_base.collection:
            raise HTTPException(status_code=503, detail="Sistema de conocimiento no disponible")

        # Obtener datos básicos (ChromaDB no tiene método directo para listar con metadata completa)
        try:
            # Intentar obtener información básica
            count = knowledge_base.collection.count()
            if count == 0:
                return {"documents": [], "total": 0}

            # Para obtener detalles completos necesitaríamos hacer queries específicas
            # Por simplicidad, devolver stats básicos
            return {
                "total": count,
                "documents": [],  # Lista vacía por simplicidad - sería expensive obtener todos
                "message": "Use search endpoint para buscar documentos específicos",
                "note": f"La colección contiene {count} documentos. Use search para contenido específico.",
            }

        except Exception as e:
            logger.error(f"Error listando documentos: {e}")
            raise HTTPException(status_code=500, detail=f"Error accediendo a colección: {e}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listando documentos: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.post("/initialize")
async def initialize_knowledge_base():
    """Inicializar o reinicializar la base de conocimientos"""
    try:
        if not knowledge_base:
            raise HTTPException(status_code=503, detail="Sistema de conocimiento no disponible")

        # Re-inicializar colección
        knowledge_base._initialize_collection()

        # Añadir conocimiento inicial básico
        initial_knowledge = [
            {
                "text": "Sheily AI es una plataforma avanzada de inteligencia artificial que combina múltiples técnicas de aprendizaje automático, procesamiento de lenguaje natural y sistemas de recomendación para proporcionar experiencias inteligentes y personalizadas.",
                "metadata": {"category": "general", "topic": "introduccion", "importance": "high"}
            },
            {
                "text": "El sistema utiliza RAG (Retrieval-Augmented Generation) para proporcionar respuestas contextualmente relevantes basadas en una base de conocimientos dinámica.",
                "metadata": {"category": "technical", "topic": "rag", "importance": "high"}
            }
        ]

        added_count = 0
        for item in initial_knowledge:
            try:
                result = await knowledge_base.add_knowledge(item["text"], item["metadata"])
                if result.get("success"):
                    added_count += 1
            except Exception as e:
                logger.warning(f"Error añadiendo conocimiento inicial: {e}")

        return {
            "success": True,
            "message": "Base de conocimientos inicializada",
            "initial_documents_added": added_count,
            "collection_ready": knowledge_base.collection is not None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inicializando base de conocimientos: {e}")
        raise HTTPException(status_code=500, detail="Error inicializando base de conocimientos")
