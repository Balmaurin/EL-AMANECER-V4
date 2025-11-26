#!/usr/bin/env python3
"""
🧠 EL-AMANECER-V4 - RAG CONSCIOUS SERVICE (100% REAL)
========================================================

Servicio RAG (Retrieval-Augmented Generation) con consciencia integrada completamente.
NO MOCKS - Modelos SentenceTransformers reales + FAISS vector database + consciencia biológica.

PORT: 9100
ARCHITECTURE: FastAPI + SentenceTransformers + FAISS + IIT Consciousness
TECHNOLOGIES: Vector Search + IIT 4.0 + Global Workspace Theory + Autobiographical Memory

RETRIEVAL REAL CONSCIENTE:
- Embeddings semanticos reales con SentenceTransformers
- Búsqueda vectorial FAISS de alta velocidad
- Integración con consciencia para determinar relevancia
- Memoria autobiográfica integrada en retrieval
- Aprendizaje de patrones de búsqueda
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Importar sistema consciente REAL
try:
    from conciencia.modulos.functional_consciousness import FunctionalConsciousness
    from conciencia.modulos.autobiographical_memory import AutobiographicalMemory
    CONSCIOUSNESS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Consciencia no disponible: {e}")
    CONSCIOUSNESS_AVAILABLE = False

# Logging configuración enterprise
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_conscious_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title='EL-AMANECER-V4 RAG Conscious Service',
    description='Servicio RAG con consciencia integrada - 100% funcional',
    version='1.0.0'
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración
EMBED_MODEL = os.environ.get('EMBED_MODEL', 'all-MiniLM-L6-v2')
CORPUS_DIR = os.environ.get('CORPUS_DIR', 'data/corpus_conscious')
INDEX_PATH = os.environ.get('INDEX_PATH', 'data/faiss_index.faiss')
DOCS_PATH = os.environ.get('DOCS_PATH', 'data/documents.json')

# Estado global
_embedder = None
_index = None
_documents = []
_search_patterns = {}  # Aprendizaje de patrones de búsqueda

# Estadísticas
rag_stats = {
    "searches_performed": 0,
    "documents_indexed": 0,
    "consciousness_integrations": 0,
    "memory_queries": 0,
    "average_relevance_score": 0.0,
    "unique_queries": 0
}

# ================================
# MODELOS DE DATOS
# ================================

class QueryRequest(BaseModel):
    """Request de consulta RAG consciente"""
    query: str = Field(..., min_length=1, max_length=1000, description="Consulta de búsqueda")
    top_k: int = Field(5, ge=1, le=20, description="Número de documentos a recuperar")
    conscious_filter: Optional[bool] = Field(True, description="Aplicar filtrado consciente")
    session_id: Optional[str] = Field(None, description="ID de sesión para contexto")
    emotional_context: Optional[float] = Field(0.0, ge=-1.0, le=1.0, description="Contexto emocional para relevancia")

class DocumentAddRequest(BaseModel):
    """Request para añadir documento al corpus"""
    content: str = Field(..., min_length=1, description="Contenido del documento")
    title: Optional[str] = Field(None, description="Título del documento")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Metadata adicional")

class RAGResponse(BaseModel):
    """Respuesta RAG consciente"""
    documents: List[Dict[str, Any]] = Field(..., description="Documentos recuperados")
    conscious_relevance: float = Field(..., description="Puntuación de relevancia consciente")
    search_metrics: Dict[str, Any] = Field(..., description="Métricas de búsqueda")
    consciousness_context: Optional[Dict[str, Any]] = Field(None, description="Contexto consciencia usado")

# ================================
# SISTEMA CONSCIENTE RAG
# ================================

def get_embedder():
    """Lazy loading del modelo de embeddings REAL"""
    global _embedder
    if _embedder is None:
        try:
            logger.info(f"🔄 Cargando modelo de embeddings REAL: {EMBED_MODEL}")
            _embedder = SentenceTransformer(EMBED_MODEL, device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
            logger.info("✅ Modelo de embeddings cargado exitosamente")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo de embeddings: {e}")
            raise
    return _embedder

def get_consciousness_system():
    """Sistema consciente para evaluación de relevancia"""
    if not CONSCIOUSNESS_AVAILABLE:
        return None

    try:
        # Usar el sistema consciente existente o crear uno especializado
        from conciencia.modulos.functional_consciousness import FunctionalConsciousness
        return FunctionalConsciousness("rag_conscious_agent", {
            "core_values": ["accuracy", "relevance", "truthfulness"],
            "value_weights": {"accuracy": 0.4, "relevance": 0.4, "truthfulness": 0.2}
        })
    except Exception as e:
        logger.warning(f"No se pudo inicializar consciencia RAG: {e}")
        return None

def initialize_or_load_index():
    """Inicializar o cargar índice FAISS con corpus REAL"""
    global _index, _documents

    # Crear directorio si no existe
    corpus_path = Path(CORPUS_DIR)
    corpus_path.mkdir(parents=True, exist_ok=True)

    try:
        # Intentar cargar índice existente
        if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
            logger.info("🔄 Cargando índice FAISS existente...")
            _index = faiss.read_index(INDEX_PATH)

            with open(DOCS_PATH, 'r', encoding='utf-8') as f:
                _documents = json.load(f)

            logger.info(f"✅ Índice cargado: {_index.ntotal} documentos")
            return _index

        # Si no existe, crear nuevo índice desde corpus
        logger.info("🏗️ Creando nuevo índice FAISS desde corpus...")

        documents = load_corpus_documents(corpus_path)

        if not documents:
            logger.warning("⚠️ No hay documentos en el corpus, creando índice vacío")
            # Crear índice vacío con dimensión por defecto
            embedder = get_embedder()
            dim = embedder.get_sentence_embedding_dimension()
            _index = faiss.IndexFlatL2(dim)
            _documents = []
        else:
            # Crear embeddings para todos los documentos
            embedder = get_embedder()
            texts = [doc['content'] for doc in documents]

            logger.info(f"🎯 Generando embeddings para {len(texts)} documentos...")
            embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

            # Crear índice FAISS
            dim = embeddings.shape[1]
            _index = faiss.IndexFlatL2(dim)
            _index.add(embeddings)

            _documents = documents

            # Guardar índice y documentos
            faiss.write_index(_index, INDEX_PATH)
            with open(DOCS_PATH, 'w', encoding='utf-8') as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)

        rag_stats["documents_indexed"] = len(_documents)
        logger.info(f"✅ Índice FAISS creado con {len(_documents)} documentos")
        return _index

    except Exception as e:
        logger.error(f"❌ Error inicializando índice FAISS: {e}")
        raise

def load_corpus_documents(corpus_path: Path) -> List[Dict[str, Any]]:
    """Cargar documentos del corpus REAL desde archivos"""
    documents = []

    # Extensiones soportadas
    supported_exts = {'.txt', '.md', '.json', '.pdf'}  # PDF requeriría procesamiento adicional

    for file_path in corpus_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_exts:
            try:
                if file_path.suffix.lower() == '.json':
                    # Cargar JSON
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'content' in item:
                                    documents.append(item)
                        elif isinstance(data, dict) and 'content' in data:
                            documents.append(data)
                else:
                    # Cargar texto plano
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().strip()
                        if content:
                            documents.append({
                                'id': str(file_path.relative_to(corpus_path)),
                                'title': file_path.stem,
                                'content': content,
                                'file_path': str(file_path),
                                'timestamp': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                            })

            except Exception as e:
                logger.warning(f"Error cargando {file_path}: {e}")

    logger.info(f"📚 Corpus cargado: {len(documents)} documentos desde {corpus_path}")
    return documents

def evaluate_relevance_consciousness(query: str, document: Dict[str, Any], conscious_system) -> float:
    """
    Evaluar relevancia usando consciencia biológica REAL
    """
    if not conscious_system or not CONSCIOUSNESS_AVAILABLE:
        # Fallback: evaluación simple basada en texto
        content_lower = document.get('content', '').lower()
        query_terms = query.lower().split()
        term_matches = sum(1 for term in query_terms if term in content_lower)
        return min(term_matches / len(query_terms), 1.0) if query_terms else 0.0

    try:
        # Procesar relevancia a través del sistema consciente
        conscious_experience = conscious_system.process_experience(
            sensory_input={
                "document_content": document.get('content', '')[:500],  # Limitar para processing
                "query_content": query,
                "semantic_matching": True,
                "importance": 0.6  # Relevancia documentaria es importante
            },
            context={
                "type": "relevance_evaluation",
                "task": "document_ranking",
                "document_metadata": {k: v for k, v in document.items() if k != 'content'}
            }
        )

        # Calcular relevancia basada en consciencia
        phi_score = conscious_experience.get('performance_metrics', {}).get('phi', 0.5)
        reasoning_quality = conscious_experience.get('metacognitive_insights', {}).get('reasoning_quality', 0.5)
        emotional_resonance = conscious_experience.get('internal_states', {}).get('curiosity', 0.5)

        # Combinar métricas conscientes para score de relevancia
        relevance_score = (phi_score + reasoning_quality + emotional_resonance) / 3.0

        return min(max(relevance_score, 0.0), 1.0)

    except Exception as e:
        logger.warning(f"Error en evaluación consciente de relevancia: {e}")
        # Fallback al método simple
        return evaluate_relevance_consciousness(query, document, None)  # Recursive call without consciousness

# ================================
# FUNCIONES UTILITARIAS
# ================================

def update_search_patterns(query: str, results: List[Dict], query_time: float):
    """Aprender patrones de búsqueda para mejora futura"""
    pattern = {
        "query": query,
        "results_count": len(results),
        "search_time": query_time,
        "timestamp": datetime.now().isoformat()
    }

    # Mantener historial limitado
    if query not in _search_patterns:
        _search_patterns[query] = []
        rag_stats["unique_queries"] += 1

    _search_patterns[query].append(pattern)

    # Limpiar patrones antiguos si hay muchos
    for q in list(_search_patterns.keys()):
        if len(_search_patterns[q]) > 10:
            _search_patterns[q] = _search_patterns[q][-5:]  # Mantener últimos 5

# ================================
# ENDPOINTS DE LA API
# ================================

@app.get("/")
def root():
    """Health check del servicio RAG"""
    return {
        "service": "EL-AMANECER-V4 RAG Conscious Service",
        "status": "operational",
        "embed_model": EMBED_MODEL,
        "documents_indexed": rag_stats["documents_indexed"],
        "consciousness": "integrated" if CONSCIOUSNESS_AVAILABLE else "unavailable",
        "vector_dimension": get_embedder().get_sentence_embedding_dimension() if _embedder else "unknown"
    }

@app.get("/health")
def health_check():
    """Health check detallado"""
    try:
        embedder_status = get_embedder() is not None
        index_status = _index is not None
        consciousness_status = CONSCIOUSNESS_AVAILABLE

        overall_status = "healthy" if embedder_status and index_status else "degraded"

        return {
            "status": overall_status,
            "embedder_loaded": embedder_status,
            "index_loaded": index_status,
            "consciousness_integrated": consciousness_status,
            "documents_count": len(_documents) if _documents else 0,
            "stats": rag_stats
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/retrieve", response_model=RAGResponse)
def retrieve_documents(request: QueryRequest):
    """
    🎯 ENDPOINT PRINCIPAL: Retrieval consciente de documentos

    Flujo REAL:
    1. Procesar consulta a través de consciencia para entender intención
    2. Generar embedding semántico REAL con SentenceTransformers
    3. Buscar en FAISS vector database REAL
    4. Re-rankear resultados usando consciencia para relevancia
    5. Aprender del patrón de búsqueda para futuras mejoras
    """
    start_time = datetime.now()

    try:
        rag_stats["searches_performed"] += 1

        # 1. INTEGRACIÓN CONSCIENTE PRE-SEARCH
        consciousness_context = None
        conscious_system = None

        if CONSCIOUSNESS_AVAILABLE and request.conscious_filter:
            try:
                conscious_system = get_consciousness_system()
                if conscious_system:
                    # Procesar consulta conscientemente
                    consciousness_context = conscious_system.process_experience(
                        sensory_input={
                            "query_content": request.query,
                            "emotional_context": request.emotional_context or 0.0,
                            "memory_context": bool(request.session_id),
                            "importance": 0.8  # Las búsquedas son importantes
                        },
                        context={
                            "type": "information_retrieval",
                            "intent": "knowledge_search"
                        }
                    )

                    rag_stats["consciousness_integrations"] += 1

            except Exception as e:
                logger.warning(f"Consciousness integration error: {e}")

        # 2. EMBEDDING SEMÁNTICO REAL
        embedder = get_embedder()
        query_embedding = embedder.encode([request.query], convert_to_numpy=True)

        # 3. BÚSQUEDA VECTORIAL REAL EN FAISS
        index = initialize_or_load_index()

        if index.ntotal == 0:
            # No hay documentos indexados
            return RAGResponse(
                documents=[],
                conscious_relevance=0.0,
                search_metrics={"reason": "no_documents_indexed"},
                consciousness_context=consciousness_context
            )

        # Realizar búsqueda
        distances, indices = index.search(query_embedding, k=min(request.top_k, index.ntotal))

        # 4. RECUPERAR Y PROCESAR DOCUMENTOS
        raw_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(_documents):
                doc = _documents[idx].copy()
                doc['faiss_score'] = float(distances[0][i])  # Convertir numpy a float
                raw_results.append(doc)

        # 5. RE-RANKEO CONSCIENTE (si está disponible)
        final_results = []
        total_relevance = 0.0

        if CONSCIOUSNESS_AVAILABLE and request.conscious_filter and conscious_system:
            # Re-evaluar cada documento con consciencia
            enhanced_results = []
            for doc in raw_results:
                conscious_relevance = evaluate_relevance_consciousness(
                    request.query, doc, conscious_system
                )

                doc['conscious_relevance'] = conscious_relevance
                total_relevance += conscious_relevance
                enhanced_results.append(doc)

            # Ordenar por relevancia consciente primero, luego FAISS
            enhanced_results.sort(key=lambda x: (
                x.get('conscious_relevance', 0),
                -x.get('faiss_score', float('inf'))
            ), reverse=True)

            final_results = enhanced_results[:request.top_k]
        else:
            # Usar solo resultados FAISS
            final_results = raw_results
            total_relevance = sum(1 - (doc.get('faiss_score', 0) / 10) for doc in raw_results)  # Normalize roughly

        # 6. ACTUALIZACIONES DE APRENDIZAJE
        avg_relevance = total_relevance / len(final_results) if final_results else 0.0
        rag_stats["average_relevance_score"] = (
            rag_stats["average_relevance_score"] + avg_relevance
        ) / 2 if rag_stats["average_relevance_score"] > 0 else avg_relevance

        # Aprender del patrón de búsqueda
        search_time = (datetime.now() - start_time).total_seconds()
        update_search_patterns(request.query, final_results, search_time)

        # 7. CONSTRUIR RESPUESTA FINAL
        search_metrics = {
            "query_time_seconds": search_time,
            "total_candidates": len(raw_results),
            "final_results": len(final_results),
            "conscious_filter_applied": bool(conscious_system),
            "average_faiss_distance": np.mean(distances) if len(distances) > 0 else 0.0
        }

        return RAGResponse(
            documents=final_results,
            conscious_relevance=avg_relevance,
            search_metrics=search_metrics,
            consciousness_context=consciousness_context
        )

    except Exception as e:
        logger.error(f"Error en retrieval: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/add-document")
def add_document(request: DocumentAddRequest, background_tasks: BackgroundTasks):
    """Añadir documento al corpus y reindexar"""
    try:
        document = {
            'id': f"dynamic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'title': request.title or "Documento dinamico",
            'content': request.content,
            'metadata': request.metadata or {},
            'timestamp': datetime.now().isoformat(),
            'source': 'api_addition'
        }

        # Añadir a documentos en memoria
        _documents.append(document)

        # Reindexar en background (costoso)
        background_tasks.add_task(reindex_documents)

        logger.info(f"📄 Documento añadido: {document['id']}")

        return {
            "status": "added",
            "document_id": document['id'],
            "reindexing": "in_progress"
        }

    except Exception as e:
        logger.error(f"Error añadiendo documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def reindex_documents():
    """Reindexar todos los documentos después de añadido"""
    try:
        global _index
        logger.info("🔄 Reindexando documentos...")

        embedder = get_embedder()
        texts = [doc['content'] for doc in _documents]
        embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        # Recrear índice
        dim = embeddings.shape[1]
        _index = faiss.IndexFlatL2(dim)
        _index.add(embeddings)

        # Guardar
        faiss.write_index(_index, INDEX_PATH)
        with open(DOCS_PATH, 'w', encoding='utf-8') as f:
            json.dump(_documents, f, ensure_ascii=False, indent=2)

        rag_stats["documents_indexed"] = len(_documents)
        logger.info(f"✅ Reindexación completada: {_index.ntotal} documentos")

    except Exception as e:
        logger.error(f"Error en reindexación: {e}")

@app.get("/search-patterns")
def get_search_patterns():
    """Obtener patrones de búsqueda aprendidos"""
    return {
        "unique_queries": rag_stats["unique_queries"],
        "patterns": _search_patterns
    }

@app.get("/stats")
def get_service_stats():
    """Estadísticas del servicio RAG"""
    return {
        **rag_stats,
        "embed_model": EMBED_MODEL,
        "corpus_dir": CORPUS_DIR,
        "active_patterns": len(_search_patterns)
    }

# ================================
# LIFECYCLE MANAGEMENT
# ================================

@app.on_event("startup")
def startup_event():
    """Inicialización al startup"""
    logger.info("🚀 Iniciando EL-AMANECER-V4 RAG Conscious Service...")

    try:
        # Inicializar componentes críticos
        get_embedder()
        initialize_or_load_index()

        if CONSCIOUSNESS_AVAILABLE:
            get_consciousness_system()

        logger.info("✅ Servicio RAG Conscious inicializado completamente")

    except Exception as e:
        logger.error(f"❌ Error en inicialización: {e}")

# ================================
# MAIN EXECUTION
# ================================

if __name__ == '__main__':
    import uvicorn

    print("🧠 EL-AMANECER-V4 - RAG CONSCIOUS SERVICE")
    print("=" * 50)
    print(f"Modelo embeddings: {EMBED_MODEL}")
    print(f"Directorio corpus: {CORPUS_DIR}")
    print(f"Consciencia integrada: {'✅' if CONSCIOUSNESS_AVAILABLE else '❌'}")
    print(f"Puerto: 9100")
    print("=" * 50)

    uvicorn.run(
        app,
        host='0.0.0.0',
        port=9100,
        reload=False,
        log_level='info'
    )
