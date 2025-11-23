"""
Servicio de embeddings para el sistema RAG
Implementación simplificada que puede ser reemplazada por modelos más avanzados
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional

import numpy as np

try:
    # Intentar importar sentence-transformers para embeddings de producción
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Servicio para generar embeddings de texto
    """

    def __init__(self):
        self.model: Optional[Any] = None
        self.dimension: int = 384  # Dimensión por defecto (similar a MiniLM)
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Inicializar el modelo de embeddings

        Returns:
            bool: True si la inicialización fue exitosa
        """
        if self._initialized:
            return True

        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.info("Inicializando SentenceTransformer para embeddings...")
                # Usar un modelo de embeddings de producción
                self.model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Modelo de embeddings cargado: dimensión {self.dimension}")
            else:
                logger.warning(
                    "SentenceTransformers no disponible, usando implementación simplificada"  # noqa: E501
                )
                # Implementación simplificada para desarrollo
                self.model = None
                self.dimension = 384

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Error inicializando embeddings: {e}")
            return False

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generar embedding para un texto

        Args:
            text: Texto a convertir en embedding

        Returns:
            List[float]: Vector de embedding
        """
        if not self._initialized:
            if not await self.initialize():
                # Fallback a implementación simplificada
                return self._generate_simple_embedding(text)

        try:
            if self.model:
                # Usar modelo de producción
                embedding = self.model.encode(text, convert_to_numpy=True)
                # Normalizar para mejor similitud coseno
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding.tolist()
            else:
                # Usar implementación simplificada
                return self._generate_simple_embedding(text)

        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            return self._generate_simple_embedding(text)

    def _generate_simple_embedding(self, text: str) -> List[float]:
        """
        Generar embedding simplificado usando hash determinista
        Solo para desarrollo - reemplazar con modelo real en producción

        Args:
            text: Texto a convertir

        Returns:
            List[float]: Vector pseudo-aleatorio determinista
        """
        # Crear hash determinista del texto
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)

        # Generar vector pseudo-aleatorio determinista
        np.random.seed(hash_int % 2**32)

        # Generar embedding de la dimensión configurada
        embedding = np.random.normal(0, 1, self.dimension).tolist()

        # Normalizar para mejor similitud coseno
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = (np.array(embedding) / norm).tolist()

        return embedding

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generar embeddings para múltiples textos

        Args:
            texts: Lista de textos

        Returns:
            List[List[float]]: Lista de vectores de embedding
        """
        embeddings = []
        for text in texts:
            embedding = await self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def get_dimension(self) -> int:
        """
        Obtener la dimensión de los embeddings

        Returns:
            int: Dimensión del vector de embedding
        """
        return self.dimension

    async def is_ready(self) -> bool:
        """
        Verificar si el servicio está listo

        Returns:
            bool: True si está inicializado
        """
        return self._initialized

    def get_info(self) -> Dict[str, Any]:
        """
        Obtener información del servicio de embeddings

        Returns:
            Dict con información del servicio
        """
        return {
            "dimension": self.dimension,
            "initialized": self._initialized,
            "model_type": ("SentenceTransformer" if self.model else "SimpleHash"),
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
        }
