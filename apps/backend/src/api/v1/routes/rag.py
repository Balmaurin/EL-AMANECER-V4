"""
API endpoints para sistema RAG - FastAPI router
Basado en RealRAGService para funcionalidad vectorial real
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Crear router
router = APIRouter()


class KnowledgeItem(BaseModel):
    """Modelo para item de conocimiento"""

    text: str = Field(
        ..., min_length=1, max_length=10000, description="Texto del documento"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Metadatos adicionales"
    )


class SearchResult(BaseModel):
    """Modelo para resultado de búsqueda"""

    document: str
    metadata: Dict[str, Any]
    distance: float
    score: float


def get_rag_service():
    """Dependencia para obtener el servicio RAG"""
    from apps.backend.src.main import app

    return app.state.rag_service


@router.post("/knowledge")
async def add_knowledge(
    item: KnowledgeItem,
    doc_id: Optional[str] = Query(None, description="ID personalizado del documento"),
    rag_service: Any = Depends(get_rag_service),
):
    """
    Añadir conocimiento a la base RAG

    - **text**: Texto del documento
    - **metadata**: Metadatos adicionales (opcional)
    - **doc_id**: ID personalizado del documento (opcional)
    """
    try:
        if not hasattr(rag_service, 'index_documents'):
            raise HTTPException(status_code=503, detail="RAG service not available")

        # Usar el sistema real de RAG
        result = await rag_service.index_documents(
            docs=[item.text],
            ids=[doc_id] if doc_id else None,
            metadatas=[item.metadata]
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Error adding knowledge: {result['error']}")

        return {
            "status": "success",
            "message": "Conocimiento añadido correctamente",
            "indexed": result.get("indexed", 1),
            "dimensions": result.get("dimensions", 0),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error añadiendo conocimiento: {e}")
        raise HTTPException(status_code=500, detail="Error añadiendo conocimiento")


@router.get("/search", response_model=List[SearchResult])
async def search_knowledge(
    query: str = Query(
        ..., min_length=1, max_length=1000, description="Consulta de búsqueda"
    ),
    limit: int = Query(5, ge=1, le=20, description="Número máximo de resultados"),
    rag_service: Any = Depends(get_rag_service),
):
    """
    Buscar en la base de conocimientos usando TF-IDF + SVD

    - **query**: Término de búsqueda
    - **limit**: Número máximo de resultados
    """
    try:
        if not hasattr(rag_service, 'search'):
            raise HTTPException(status_code=503, detail="RAG service not available")

        results = await rag_service.search(query, limit)

        if "error" in results:
            raise HTTPException(status_code=500, detail=f"Error en búsqueda: {results['error']}")

        # Convertir al formato esperado
        formatted_results = []
        for result in results.get("results", []):
            formatted_results.append({
                "document": result.get("document", ""),
                "metadata": result.get("metadata", {}),
                "distance": 1 - result.get("similarity", 0),  # Convertir similitud a distancia
                "score": result.get("similarity", 0),
            })

        return formatted_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error buscando conocimiento: {e}")
        raise HTTPException(status_code=500, detail="Error en búsqueda")


@router.get("/context")
async def get_relevant_context(
    query: str = Query(
        ...,
        min_length=1,
        max_length=1000,
        description="Consulta para contexto",
    ),
    top_k: Optional[int] = Query(None, ge=1, le=20, description="Número de documentos"),
    threshold: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Umbral de similitud"
    ),
    rag_service: Any = Depends(get_rag_service),
):
    """
    Obtener contexto relevante para una consulta usando RAG real

    - **query**: Consulta del usuario
    - **top_k**: Número de documentos a recuperar
    - **threshold**: Umbral de similitud mínimo
    """
    try:
        if not hasattr(rag_service, 'search'):
            raise HTTPException(status_code=503, detail="RAG service not available")

        # Buscar documentos relevantes
        search_results = await rag_service.search(query, top_k or 5)

        if "error" in search_results:
            raise HTTPException(status_code=500, detail=f"Error obteniendo contexto: {search_results['error']}")

        # Filtrar por threshold si se especifica
        results = search_results.get("results", [])
        if threshold is not None:
            results = [r for r in results if r.get("similarity", 0) >= threshold]

        # Combinar el contexto relevante
        context_parts = []
        for result in results:
            doc_content = result.get("document", "")
            similarity = result.get("similarity", 0)

            # Solo incluir si es suficientemente relevante
            if similarity > 0.1:  # Threshold mínimo
                context_parts.append(f"[Relevance: {similarity:.3f}] {doc_content}")

        full_context = "\n\n".join(context_parts)

        return {
            "query": query,
            "context": full_context,
            "context_length": len(full_context),
            "documents_used": len(results),
            "method": search_results.get("method", "tfidf_svd_cosine"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo contexto: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo contexto")


@router.get("/info")
async def get_rag_info(rag_service: Any = Depends(get_rag_service)):
    """
    Obtener información del sistema RAG real
    """
    try:
        info = {
            "service_type": "RealRAGService",
            "method": "TF-IDF + SVD + Cosine Similarity",
            "indexed_documents": len(rag_service.documents) if hasattr(rag_service, 'documents') else 0,
            "features": []
        }

        # Detectar características disponibles
        if hasattr(rag_service, 'vectorizer') and rag_service.vectorizer:
            info["features"].append("TF-IDF Vectorization")
            if hasattr(rag_service.vectorizer, 'vocabulary_'):
                info["vocabulary_size"] = len(rag_service.vectorizer.vocabulary_)

        if hasattr(rag_service, 'svd') and rag_service.svd:
            info["features"].append("SVD Dimensionality Reduction")
            if hasattr(rag_service.svd, 'n_components'):
                info["n_components"] = rag_service.svd.n_components

        if hasattr(rag_service, 'tfidf_matrix') and rag_service.tfidf_matrix is not None:
            info["features"].append("TF-IDF Matrix")
            info["matrix_shape"] = rag_service.tfidf_matrix.shape

        info["features"].append("Cosine Similarity Search")
        info["ready"] = len(rag_service.documents) > 0 if hasattr(rag_service, 'documents') else False

        return info

    except Exception as e:
        logger.error(f"Error obteniendo info RAG: {e}")
        return {
            "service_type": "RealRAGService",
            "status": "error",
            "error": str(e),
            "ready": False,
        }


@router.delete("/collection")
async def clear_collection(rag_service: Any = Depends(get_rag_service)):
    """
    Limpiar toda la colección RAG (usar con cuidado)
    """
    try:
        # Reset del servicio RAG
        if hasattr(rag_service, '_save_data'):
            rag_service.documents = []
            rag_service.doc_ids = []
            rag_service.metadatas = []
            rag_service.vectorizer = None
            rag_service.svd = None
            rag_service.tfidf_matrix = None
            rag_service.reduced_matrix = None

            # Guardar estado vacío
            rag_service._save_data()

        return {"status": "cleared", "message": "Colección RAG limpiada"}
    except Exception as e:
        logger.error(f"Error limpiando colección: {e}")
        raise HTTPException(status_code=500, detail="Error limpiando colección")


@router.get("/health")
async def rag_health_check(rag_service: Any = Depends(get_rag_service)):
    """
    Verificar estado del sistema RAG real
    """
    try:
        # Verificar componentes críticos
        has_vectorizer = hasattr(rag_service, 'vectorizer') and rag_service.vectorizer is not None
        has_documents = hasattr(rag_service, 'documents') and len(rag_service.documents) > 0
        has_svd = hasattr(rag_service, 'svd') and rag_service.svd is not None

        health_score = sum([has_vectorizer, has_documents, has_svd])

        return {
            "healthy": health_score >= 2,  # Necesita al menos 2 componentes
            "document_count": len(rag_service.documents) if hasattr(rag_service, 'documents') else 0,
            "has_vectorizer": has_vectorizer,
            "has_svd": has_svd,
            "method": "tfidf_svd_cosine",
            "data_directory": str(rag_service.data_dir) if hasattr(rag_service, 'data_dir') else None,
            "vocabulary_size": len(rag_service.vectorizer.vocabulary_) if has_vectorizer else 0,
            "components": rag_service.svd.n_components if has_svd and hasattr(rag_service.svd, 'n_components') else 0,
        }
    except Exception as e:
        logger.error(f"Error en health check RAG: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "document_count": 0,
            "has_vectorizer": False,
            "has_svd": False,
        }


@router.post("/bulk-index")
async def bulk_index_documents(
    documents: List[KnowledgeItem],
    rag_service: Any = Depends(get_rag_service),
):
    """
    Indexar múltiples documentos en una sola operación

    - **documents**: Lista de documentos con texto y metadata
    """
    try:
        if not hasattr(rag_service, 'index_documents'):
            raise HTTPException(status_code=503, detail="RAG service not available")

        if not documents:
            raise HTTPException(status_code=400, detail="No documents provided")

        # Extraer textos e IDs
        texts = [doc.text for doc in documents]
        ids = [f"bulk_doc_{i}_{hash(doc.text) % 10000}" for i, doc in enumerate(documents)]
        metadatas = [doc.metadata or {} for doc in documents]

        # Indexar en lote
        result = await rag_service.index_documents(
            docs=texts,
            ids=ids,
            metadatas=metadatas
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Bulk indexing failed: {result['error']}")

        return {
            "status": "success",
            "message": f"Indexados {result.get('indexed', 0)} documentos correctamente",
            "indexed_count": result.get("indexed", 0),
            "dimensions": result.get("dimensions", 0),
            "document_ids": ids,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en bulk indexing: {e}")
        raise HTTPException(status_code=500, detail="Bulk indexing failed")


@router.get("/stats")
async def get_rag_statistics(rag_service: Any = Depends(get_rag_service)):
    """
    Obtener estadísticas detalladas del sistema RAG
    """
    try:
        stats = {
            "service_type": "RealRAGService",
            "document_count": len(rag_service.documents) if hasattr(rag_service, 'documents') else 0,
            "vectorization_ready": hasattr(rag_service, 'vectorizer') and rag_service.vectorizer is not None,
            "dimensionality_reduction_ready": hasattr(rag_service, 'svd') and rag_service.svd is not None,
            "method": "TF-IDF + SVD + Cosine Similarity",
        }

        # Estadísticas detalladas del vectorizador
        if stats["vectorization_ready"]:
            vectorizer = rag_service.vectorizer
            stats["vocabulary_size"] = len(vectorizer.vocabulary_)
            stats["max_features"] = getattr(vectorizer, 'max_features', None)
            stats["smooth_idf"] = getattr(vectorizer, 'smooth_idf', None)
            stats["sublinear_tf"] = getattr(vectorizer, 'sublinear_tf', None)

        # Estadísticas detalladas del SVD
        if stats["dimensionality_reduction_ready"]:
            svd = rag_service.svd
            stats["n_components"] = getattr(svd, 'n_components', None)
            if hasattr(svd, 'explained_variance_ratio_'):
                stats["explained_variance_ratio"] = svd.explained_variance_ratio_.tolist()
            if hasattr(svd, 'singular_values_'):
                stats["singular_values"] = svd.singular_values_.tolist()

        # Estadísticas de matrices
        if hasattr(rag_service, 'tfidf_matrix') and rag_service.tfidf_matrix is not None:
            stats["tfidf_matrix_shape"] = rag_service.tfidf_matrix.shape
            stats["tfidf_matrix_density"] = float(rag_service.tfidf_matrix.nnz) / (rag_service.tfidf_matrix.shape[0] * rag_service.tfidf_matrix.shape[1])

        if hasattr(rag_service, 'reduced_matrix') and rag_service.reduced_matrix is not None:
            stats["reduced_matrix_shape"] = rag_service.reduced_matrix.shape

        return stats

    except Exception as e:
        logger.error(f"Error obteniendo estadísticas RAG: {e}")
        logger.error(f"Error obteniendo estadísticas RAG: {e}")
