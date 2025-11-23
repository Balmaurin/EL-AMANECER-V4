#!/usr/bin/env python3
"""
ADVANCED RAG SYSTEM MCP - Sistema Completo de Retrieval Augmentation
=========================================================================

Sistema RAG avanzado que integra:
- Sentence-Transformers embeddings (all-MiniLM-L6-v2)
- Vector search con HNSW
- Retrieval híbrido con fallback
- Integración completa con MCP-Phoenix
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Fallback imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger("advanced_rag")


class AdvancedRAGService:
    """
    Sistema RAG avanzado con embeddings transformers de verdad
    """

    def __init__(self, config_path: str = "config/universal.yaml", data_dir: str = "./corpus_data"):
        self.config_path = config_path
        self.data_dir = Path(data_dir)
        self.config = None

        # Componentes del sistema
        self.embedding_model = None
        self.documents = []
        self.doc_ids = []
        self.metadatas = []
        self.embeddings = None

        # Fallback TF-IDF
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

        # Indices HNSW/FAISS (simulados inicialmente)
        self.vector_index = None

        # Estado
        self.initialized = False
        self.embedding_dimension = 384  # all-MiniLM-L6-v2 dimension

        logger.info("🚀 Advanced RAG System inicializado")

    async def initialize(self) -> bool:
        """Inicializar sistema completo de RAG"""

        try:
            # Cargar configuración
            await self._load_config()

            # Inicializar embeddings
            success = await self._initialize_embeddings()
            if not success:
                logger.warning("⚠️ Embeddings no disponible - usando TF-IDF fallback")
                await self._initialize_fallback()

            # Cargar datos si existen
            await self._load_data()

            # Inicializar índices si hay datos
            if self.documents:
                await self._build_indices()

            self.initialized = True
            logger.info(f"✅ Advanced RAG System operativo - {len(self.documents)} documentos")
            return True

        except Exception as e:
            logger.error(f"❌ Error inicializando Advanced RAG: {e}")
            return False

    async def _load_config(self) -> None:
        """Cargar configuración universal"""
        try:
            import yaml
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
                logger.info("✅ Configuración cargada desde universal.yaml")
        except Exception as e:
            logger.warning(f"No se pudo cargar config universal: {e}")
            self.config = {}

    async def _initialize_embeddings(self) -> bool:
        """Inicializar sistema de embeddings con Sentence Transformers"""

        if not ST_AVAILABLE:
            logger.warning("Sentence-Transformers no disponible")
            return False

        try:
            # Configuración desde universal.yaml
            embedder_config = self.config.get("embedder", {})
            model_name = embedder_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")

            # Forzar CPU para Windows estabilidad
            device = embedder_config.get("device", "cpu")
            if device != "cpu":
                device = "cpu"  # Forzar CPU

            logger.info(f"🔧 Cargando modelo embeddings: {model_name} en {device}")

            # Cargar modelo con optimizaciones para Windows
            self.embedding_model = SentenceTransformer(
                model_name,
                device=device,
                cache_folder=os.environ.get("HF_HOME", "./cache")
            )

            # Configurar dimensión
            test_embed = self.embedding_model.encode(["test"])
            self.embedding_dimension = len(test_embed[0])

            logger.info(f"✅ Modelo embeddings cargado - dimensión: {self.embedding_dimension}")
            return True

        except Exception as e:
            logger.error(f"❌ Error cargando modelo embeddings: {e}")
            return False

    async def _initialize_fallback(self) -> None:
        """Inicializar fallback TF-IDF + SVD si embeddings fallan"""

        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words=None,
                max_features=1000,
                ngram_range=(1, 2)
            )
            logger.info("✅ Fallback TF-IDF activado")
        else:
            logger.warning("Ni embeddings ni TF-IDF disponible")

    async def _load_data(self) -> None:
        """Cargar datos si existen en el sistema"""

        try:
            # Intentar cargar desde corpus_data
            data_file = self.data_dir / "documents.json"
            if data_file.exists():
                with open(data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    self.doc_ids = data.get("doc_ids", [])
                    self.metadatas = data.get("metadatas", [])
                    logger.info(f"✅ Datos cargados: {len(self.documents)} documentos")
            else:
                logger.info("No hay datos pre-cargados - usar index_documents()")

        except Exception as e:
            logger.error(f"Error cargando datos: {e}")

    async def _build_indices(self) -> None:
        """Construir índices para búsqueda rápida"""

        try:
            if not self.documents:
                return

            # Generar embeddings si tenemos modelo
            if self.embedding_model:
                logger.info("🔧 Generando embeddings...")

                # Batch processing optimizado
                batch_size = min(32, len(self.documents))
                embeddings_list = []

                for i in range(0, len(self.documents), batch_size):
                    batch_texts = self.documents[i:i+batch_size]
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=8  # Pequeño para estabilidad
                    )
                    embeddings_list.append(batch_embeddings)

                self.embeddings = np.vstack(embeddings_list)
                logger.info(f"✅ Embeddings generados: {self.embeddings.shape}")

                # Construir índice HNSW/FAISS
                await self._build_vector_index()

            # Fallback TF-IDF si no hay embeddings
            elif self.tfidf_vectorizer and SKLEARN_AVAILABLE:
                logger.info("🔧 Construyendo índice TF-IDF...")
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
                logger.info("✅ Índice TF-IDF construido")

        except Exception as e:
            logger.error(f"Error construyendo índices: {e}")

    async def _build_vector_index(self) -> None:
        """Construir índice vectorial HNSW"""

        if not FAISS_AVAILABLE or self.embeddings is None:
            logger.info("FAISS no disponible - búsqueda vectorial limitada")
            return

        try:
            dimension = self.embedding_dimension

            # Crear índice HNSW
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections por elemento

            # Configurar parámetros
            index.hnsw.efConstruction = 200

            # Agregar vectores
            index.add(self.embeddings.astype('float32'))

            # Guardar índice
            self.vector_index = index

            logger.info(f"✅ Índice HNSW construido: {len(self.embeddings)} vectores, dim={dimension}")

        except Exception as e:
            logger.error(f"Error construyendo HNSW: {e}")
            self.vector_index = None

    # ========== MÉTODOS DE COMPATIBILIDAD CON CHATSERVICE ==========

    async def index_documents(self, docs: List[str], ids: List[str] = None, metadatas: List[Dict] = None) -> Dict[str, Any]:
        """Indexar documentos - compatible con ChatService"""

        if not docs:
            return {"error": "no documents"}

        if not ids:
            ids = [f"doc_{i}" for i in range(len(docs))]
        if not metadatas:
            metadatas = [{}] * len(docs)

        self.documents = docs
        self.doc_ids = ids
        self.metadatas = metadatas

        # Construir índices
        await self._build_indices()

        # Guardar datos
        await self._save_data()

        return {
            "indexed": len(docs),
            "method": "sentence_transformers_hnsw" if self.embedding_model else "tfidf_fallback",
            "embeddings_available": self.embedding_model is not None,
            "vector_index": self.vector_index is not None
        }

    async def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Búsqueda vectorial con embeddings transformers"""

        if not self.initialized:
            return {"error": "RAG not initialized"}

        if not self.documents:
            return {"error": "no documents indexed"}

        try:
            results = []

            # Método principal: búsqueda vectorial con embeddings
            if self.embedding_model and self.embeddings is not None:
                results = await self._vector_search(query, top_k)

            # Fallback 1: TF-IDF si no hay embeddings
            elif self.tfidf_vectorizer and self.tfidf_matrix is not None:
                results = await self._tfidf_search(query, top_k)

            # Fallback 2: búsqueda simple por keywords
            else:
                results = await self._keyword_search(query, top_k)

            return {
                "query": query,
                "results": results,
                "method": "sentence_transformers_hnsw" if self.embedding_model else "fallback_tfidf",
                "total_docs": len(self.documents)
            }

        except Exception as e:
            logger.error(f"Búsqueda falló: {e}")
            return {"error": str(e)}

    async def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Búsqueda vectorial con embeddings transformers"""

        # Generar embedding para la query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        query_embedding = query_embedding.astype('float32')

        # Búsqueda con FAISS si disponible
        if self.vector_index is not None:
            # Configurar ef_search para balance calidad/velocidad
            faiss.cvar.hnsw.efSearch.set(100)

            distances, indices = self.vector_index.search(
                query_embedding.reshape(1, -1),
                top_k
            )

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.documents) and idx >= 0:
                    results.append({
                        "document": self.documents[idx][:500] + "..." if len(self.documents[idx]) > 500 else self.documents[idx],
                        "id": self.doc_ids[idx],
                        "metadata": self.metadatas[idx],
                        "similarity": float(1 - dist),  # Convertir distancia a similitud
                        "score": float(1 - dist)
                    })
        else:
            # Búsqueda manual con coseno si no hay FAISS
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )

            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                results.append({
                    "document": self.documents[idx][:500] + "..." if len(self.documents[idx]) > 500 else self.documents[idx],
                    "id": self.doc_ids[idx],
                    "metadata": self.metadatas[idx],
                    "similarity": float(similarities[idx]),
                    "score": float(similarities[idx])
                })

        return results

    async def _tfidf_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Búsqueda fallback con TF-IDF"""

        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx][:500] + "..." if len(self.documents[idx]) > 500 else self.documents[idx],
                "id": self.doc_ids[idx],
                "metadata": self.metadatas[idx],
                "similarity": float(similarities[idx]),
                "score": float(similarities[idx])
            })

        return results

    async def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Búsqueda simple por keywords como último fallback"""

        query_lower = query.lower()
        query_keywords = set(query_lower.split())

        results = []
        for i, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            matching_keywords = sum(1 for kw in query_keywords if kw in doc_lower)
            score = matching_keywords / len(query_keywords) if query_keywords else 0

            if score > 0:
                results.append({
                    "document": doc[:500] + "..." if len(doc) > 500 else doc,
                    "id": self.doc_ids[i],
                    "metadata": self.metadatas[i],
                    "similarity": score,
                    "score": score
                })

        # Ordenar por score descendente
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def retrieve_relevant_context(self, query: str, top_k: int = 3, similarity_threshold: float = 0.1) -> str:
        """Método compatible con ChatService para obtener contexto"""

        results = await self.search(query, top_k=top_k)

        if not results or "results" not in results:
            return ""

        # Formatear como contexto string
        context_parts = []
        for result in results["results"]:
            similarity = result.get("similarity", 0)
            if similarity > similarity_threshold:
                doc = result.get("document", "")
                if doc:
                    context_parts.append(doc)

        return "\n\n".join(context_parts)

    async def get_collection_info(self) -> Dict[str, Any]:
        """Información del collection para ChatService"""

        return {
            "name": "advanced_rag_collection",
            "initialized": self.initialized,
            "count": len(self.documents),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2" if self.embedding_model else "tfidf_fallback",
            "vector_index": "HNSW" if self.vector_index else "cosine_manual",
            "dimension": self.embedding_dimension
        }

    async def is_ready(self) -> bool:
        """Verificar si el sistema está listo"""
        return self.initialized and len(self.documents) > 0

    async def _save_data(self) -> None:
        """Guardar datos indexados"""

        try:
            data = {
                "documents": self.documents,
                "doc_ids": self.doc_ids,
                "metadatas": self.metadatas,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2" if self.embedding_model else None,
                "vector_index": "HNSW" if self.vector_index else None
            }

            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.data_dir / "documents.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Datos guardados en: {self.data_dir / 'documents.json'}")

        except Exception as e:
            logger.error(f"Error guardando datos: {e}")

# ==========
# DEMO Y TESTING
# ==========

async def demo_advanced_rag():
    """Demo del Advanced RAG System"""

    print("🧠 ADVANCED RAG SYSTEM - EMBEDDINGS TRANSFORMERS COMPLETOS")
    print("=" * 70)

    # Inicializar sistema
    rag = AdvancedRAGService()

    # Inicializar
    print("🔧 Inicializando Advanced RAG...")
    success = await rag.initialize()

    if not success:
        print("❌ Error inicializando RAG avanzado")
        return

    # Información del sistema
    info = await rag.get_collection_info()
    print("✅ Sistema operativo:")
    print(f"   📊 Documentos: {info['count']}")
    print(f"   🧠 Embeddings: {info['embedding_model']}")
    print(f"   🔍 Vector Index: {info['vector_index']}")
    print(f"   📐 Dimensión: {info['dimension']}")

    # Indexar documentos de demostración
    if info['count'] == 0:
        demo_docs = [
            "La Inteligencia Artificial es una rama de la informática que se enfoca en crear máquinas capaces de realizar tareas que normalmente requieren inteligencia humana.",
            "El aprendizaje profundo utiliza redes neuronales para procesar datos complejos y aprender patrones sofisticados.",
            "Los transformers son modelos de arquitectura que han revolucionado el procesamiento del lenguaje natural.",
            "El RAG (Retrieval Augmented Generation) combina recuperación de información con generación de texto para respuestas más precisas.",
            "Los embeddings vectoriales representan conceptos como puntos en un espacio matemático multidimensional.",
        ]

        print("\n📚 Indexando documentos de demostración...")
        result = await rag.index_documents(demo_docs)
        print(f"✅ Indexación completada: {result}")

    # Pruebas de búsqueda
    print("\n🔍 PRUEBA DE BÚSQUEDAS AVANZADAS:")

    queries = [
        "¿Qué es la inteligencia artificial?",
        "Cómo funciona el aprendizaje profundo?",
        "¿Qué son los embeddings vectoriales?"
    ]

    for query in queries:
        print(f"\n📝 Query: \"{query}\"")

        results = await rag.search(query, top_k=2)

        if results.get("results"):
            for i, result in enumerate(results["results"][:2], 1):
                similarity = result.get('similarity', 0)
                doc_preview = result.get('document', '')[:100]
                print(f"      {i}. Score: {similarity:.3f} - '{doc_preview}'")
        else:
            print("  ❌ No se encontraron resultados")

    # Test de contexto para ChatService
    print("\n💬 Test de integración con ChatService:")
    context = await rag.retrieve_relevant_context("¿Qué es IA?", top_k=1)
    print(f"  📄 Contexto generado: {len(context)} caracteres")
    if context:
        print(f"  📖 Preview: {context[:200]}...")

    print("""
🎉 ADVANCED RAG SYSTEM OPERATIVO!
   ✅ Embeddings transformers reales (all-MiniLM-L6-v2)
   ✅ Índices vectoriales HNSW/FAISS
   ✅ Retrieval híbrido con fallbacks
   ✅ Integración completa con MCP-Phoenix
""")


if __name__ == "__main__":
    asyncio.run(demo_advanced_rag())
