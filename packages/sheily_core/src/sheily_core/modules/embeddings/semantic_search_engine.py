#!/usr/bin/env python3
"""
Semantic Search Engine - Motor de Búsqueda Semántica

Este módulo implementa búsqueda semántica avanzada con capacidades de:
- Búsqueda por similitud semántica
- Indexación de documentos
- Ranking inteligente
- Filtros avanzados
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """Motor de búsqueda semántica"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """Inicializar motor de búsqueda"""
        self.model_name = model_name
        self.documents = []  # Lista de documentos indexados
        self.embeddings = []  # Embeddings correspondientes

        # Simular modelo de embeddings
        self.model = "simulated_embeddings_model"

        self.initialized = True
        logger.info(f"SemanticSearchEngine inicializado con {model_name}")

    def index_document(
        self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Indexar un documento"""
        try:
            # Simular generación de embedding
            embedding = self._generate_embedding(content)

            document = {
                "id": doc_id,
                "content": content,
                "metadata": metadata or {},
                "indexed_at": time.time(),
                "embedding": embedding,
            }

            self.documents.append(document)
            self.embeddings.append(embedding)

            return True
        except Exception as e:
            logger.error(f"Error indexando documento {doc_id}: {e}")
            return False

    def search(
        self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Realizar búsqueda semántica"""
        try:
            # Generar embedding de la consulta
            query_embedding = self._generate_embedding(query)

            # Calcular similitudes
            similarities = []
            for i, doc_embedding in enumerate(self.embeddings):
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append((i, similarity))

            # Ordenar por similitud descendente
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Aplicar filtros si existen
            filtered_results = self._apply_filters(similarities, filters)

            # Obtener resultados top-k
            results = []
            for doc_idx, similarity in filtered_results[:top_k]:
                doc = self.documents[doc_idx]
                results.append(
                    {
                        "document_id": doc["id"],
                        "content": (
                            doc["content"][:200] + "..."
                            if len(doc["content"]) > 200
                            else doc["content"]
                        ),
                        "similarity": similarity,
                        "metadata": doc["metadata"],
                        "rank": len(results) + 1,
                    }
                )

            return {
                "query": query,
                "total_results": len(results),
                "results": results,
                "search_time": time.time(),
                "model_used": self.model_name,
            }

        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return {"error": str(e)}

    def _generate_embedding(self, text: str) -> List[float]:
        """Generar embedding para un texto (simulado)"""
        # Simulación simple de embedding basado en hash
        import hashlib

        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)

        # Generar vector de 384 dimensiones (típico para sentence-transformers)
        embedding = []
        for i in range(384):
            # Usar diferentes partes del hash para generar el vector
            value = (hash_int >> (i % 32)) & 0xFF
            # Normalizar a [-1, 1]
            normalized = (value / 127.5) - 1
            embedding.append(normalized)

        return embedding

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcular similitud coseno entre dos vectores"""
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)
        except:
            return 0.0

    def _apply_filters(
        self, similarities: List[tuple], filters: Optional[Dict[str, Any]]
    ) -> List[tuple]:
        """Aplicar filtros a los resultados"""
        if not filters:
            return similarities

        filtered = []
        for doc_idx, similarity in similarities:
            doc = self.documents[doc_idx]
            include = True

            # Aplicar filtros de metadata
            for key, value in filters.items():
                if key in doc["metadata"]:
                    if doc["metadata"][key] != value:
                        include = False
                        break

            if include:
                filtered.append((doc_idx, similarity))

        return filtered

    def remove_document(self, doc_id: str) -> bool:
        """Eliminar un documento del índice"""
        for i, doc in enumerate(self.documents):
            if doc["id"] == doc_id:
                del self.documents[i]
                del self.embeddings[i]
                return True
        return False

    def get_index_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del índice"""
        return {
            "total_documents": len(self.documents),
            "model_name": self.model_name,
            "embedding_dimension": 384,
            "last_updated": max(
                (doc["indexed_at"] for doc in self.documents), default=0
            ),
            "capabilities": ["semantic_search", "metadata_filtering", "ranking"],
        }

    def clear_index(self) -> bool:
        """Limpiar todo el índice"""
        try:
            self.documents.clear()
            self.embeddings.clear()
            return True
        except Exception as e:
            logger.error(f"Error limpiando índice: {e}")
            return False
