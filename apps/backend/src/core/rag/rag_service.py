"""
Servicio RAG - Retrieval-Augmented Generation usando ChromaDB
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from ...config.settings import settings
from ..embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class RAGService:
    """
    Servicio de Retrieval-Augmented Generation usando ChromaDB
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.chroma_client: Optional[Any] = None
        self.collection: Optional[Any] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Inicializar el sistema RAG

        Returns:
            bool: True si la inicialización fue exitosa
        """
        if self._initialized:
            return True

        try:
            # Inicializar servicio de embeddings
            if not await self.embedding_service.initialize():
                logger.error("Error inicializando servicio de embeddings")
                return False

            # Inicializar ChromaDB
            if not CHROMADB_AVAILABLE:
                logger.error("ChromaDB no está disponible")
                return False

            logger.info("Inicializando sistema RAG con ChromaDB...")

            # Configurar ChromaDB
            chroma_settings = ChromaSettings(
                persist_directory=settings.rag.chroma_db_path,
                is_persistent=True,
            )

            self.chroma_client = chromadb.PersistentClient(
                path=settings.rag.chroma_db_path, settings=chroma_settings
            )

            # Crear o obtener colección
            self.collection = self.chroma_client.get_or_create_collection(
                name="knowledge_base",
                metadata={
                    "description": "Base de conocimientos para RAG",
                    "created_at": datetime.now().isoformat(),
                    "embedding_dimension": (self.embedding_service.get_dimension()),
                },
            )

            # Configurar conocimiento inicial si está vacío
            if self.collection.count() == 0:
                await self._setup_initial_knowledge()

            self._initialized = True
            logger.info("Sistema RAG inicializado exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error inicializando RAG: {e}")
            return False

    async def _setup_initial_knowledge(self):
        """Configurar conocimiento inicial en la base RAG"""
        logger.info("Configurando conocimiento inicial...")

        initial_knowledge = [
            {
                "text": (
                    "Sheily AI es un sistema de inteligencia artificial "
                    "avanzado que combina chat inteligente, entrenamiento "
                    "automático LoRA, sistema RAG con mejora continua, "
                    "y una interfaz web completa. Todo containerizado "
                    "para facilitar el despliegue y escalabilidad."
                ),
                "metadata": {
                    "category": "general",
                    "topic": "introduccion",
                    "importance": "high",
                    "source": "system",
                },
            },
            {
                "text": (
                    "Gemma 2 es un modelo de lenguaje desarrollado "
                    "por Google DeepMind, optimizado para eficiencia "
                    "y rendimiento. Utiliza arquitectura de transformer "
                    "con mejoras en comprensión contextual y generación "
                    "de texto."
                ),
                "metadata": {
                    "category": "technical",
                    "topic": "modelo",
                    "importance": "high",
                    "source": "system",
                },
            },
            {
                "text": (
                    "El sistema RAG (Retrieval-Augmented Generation) mejora "
                    "las respuestas del modelo al recuperar información "
                    "relevante de una base de conocimientos antes de generar "
                    "la respuesta final."
                ),
                "metadata": {
                    "category": "technical",
                    "topic": "rag",
                    "importance": "high",
                    "source": "system",
                },
            },
            {
                "text": (
                    "Las conversaciones se mantienen en tiempo real mediante "
                    "WebSocket, permitiendo interacciones fluidas y continuas "
                    "con el sistema de IA."
                ),
                "metadata": {
                    "category": "technical",
                    "topic": "chat",
                    "importance": "medium",
                    "source": "system",
                },
            },
            {
                "text": (
                    "El sistema incluye medidas de seguridad como validación "
                    "de inputs, rate limiting y sanitización de datos para "
                    "prevenir ataques y abuso."
                ),
                "metadata": {
                    "category": "security",
                    "topic": "proteccion",
                    "importance": "high",
                    "source": "system",
                },
            },
        ]

        for item in initial_knowledge:
            await self.add_knowledge(text=item["text"], metadata=item["metadata"])

        logger.info(
            "Conocimiento inicial configurado: " f"{len(initial_knowledge)} documentos"
        )

    async def add_knowledge(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Añadir conocimiento a la base RAG

        Args:
            text: Texto del documento
            metadata: Metadatos adicionales
            doc_id: ID personalizado del documento

        Returns:
            str: ID del documento añadido
        """
        if not self.collection:
            raise RuntimeError("RAG service not initialized")

        try:
            # Generar embedding
            embedding = await self.embedding_service.generate_embedding(text)

            # Generar ID si no se proporciona
            if doc_id is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                doc_id = f"doc_{timestamp}_{hash(text) % 10000}"

            # Preparar metadatos
            if metadata is None:
                metadata = {}

            metadata.update(
                {
                    "added_at": datetime.now().isoformat(),
                    "text_length": len(text),
                }
            )

            # Añadir a ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id],
            )

            logger.info(f"Conocimiento añadido: {doc_id}")
            return doc_id

        except Exception as e:
            logger.error(f"Error añadiendo conocimiento: {e}")
            raise

    async def retrieve_relevant_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> str:
        """
        Recuperar contexto relevante para una consulta

        Args:
            query: Consulta del usuario
            top_k: Número de documentos a recuperar
            similarity_threshold: Umbral de similitud

        Returns:
            str: Contexto relevante concatenado
        """
        if not self.collection or self.collection.count() == 0:
            return ""

        try:
            # Usar configuración por defecto si no se especifica
            if top_k is None:
                top_k = settings.rag.top_k
            if similarity_threshold is None:
                similarity_threshold = settings.rag.similarity_threshold

            # Generar embedding de la consulta
            query_embedding = await self.embedding_service.generate_embedding(query)

            # Buscar documentos relevantes
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            # Filtrar y combinar contexto relevante
            context_parts = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = (
                        results["distances"][0][i] if results["distances"] else 1.0
                    )
                    # Solo incluir documentos con similitud razonable
                    if distance < (
                        1 - similarity_threshold
                    ):  # Convertir threshold a distancia
                        context_parts.append(doc)

            context = "\n\n".join(context_parts)

            if context:
                logger.info(
                    f"Contexto RAG recuperado: {len(context)} caracteres "
                    f"de {len(context_parts)} documentos"
                )

            return context

        except Exception as e:
            logger.error(f"Error en recuperación RAG: {e}")
            return ""

    async def search_knowledge(
        self, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Buscar en la base de conocimientos

        Args:
            query: Consulta de búsqueda
            limit: Número máximo de resultados

        Returns:
            List[Dict]: Resultados de búsqueda con scores
        """
        if not self.collection:
            raise RuntimeError("RAG service not initialized")

        try:
            if not query.strip():
                return []

            # Generar embedding de la consulta
            query_embedding = await self.embedding_service.generate_embedding(query)

            # Buscar documentos relevantes
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )

            # Formatear resultados
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = (
                        results["distances"][0][i] if results["distances"] else 1.0
                    )
                    score = 1 - distance  # Convertir distancia a score de similitud

                    formatted_results.append(
                        {
                            "document": doc,
                            "metadata": (
                                results["metadatas"][0][i]
                                if results["metadatas"]
                                else {}
                            ),
                            "distance": distance,
                            "score": score,
                        }
                    )

            return formatted_results

        except Exception as e:
            logger.error(f"Error buscando conocimiento: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Obtener información de la colección

        Returns:
            Dict con información de la colección
        """
        if not self.collection:
            return {"initialized": False}

        try:
            return {
                "initialized": True,
                "count": self.collection.count(),
                "name": self.collection.name,
                "metadata": self.collection.metadata,
            }
        except Exception as e:
            logger.error(f"Error obteniendo info de colección: {e}")
            return {"initialized": False, "error": str(e)}

    async def is_ready(self) -> bool:
        """
        Verificar si el servicio RAG está listo

        Returns:
            bool: True si está inicializado y tiene documentos
        """
        return (
            self._initialized
            and self.collection is not None
            and self.collection.count() > 0
        )

    async def clear_collection(self):
        """
        Limpiar toda la colección (usar con cuidado)
        """
        if self.collection:
            try:
                # ChromaDB no tiene método directo para limpiar,
                # recrear colección
                collection_name = self.collection.name
                self.chroma_client.delete_collection(collection_name)

                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Base de conocimientos para RAG"},
                )

                logger.info("Colección RAG limpiada")
            except Exception as e:
                logger.error(f"Error limpiando colección: {e}")
                raise
