#!/usr/bin/env python3
"""
TEMPORAL MEMORY - Sistema de Memoria Temporal y Contextual Avanzada
==================================================================

Sistema de memoria temporal y contextual avanzada que mantiene comprensión
holística del tiempo, contexto histórico y evolución de conversaciones.
"""

import asyncio
import hashlib
import json
import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


@dataclass
class TemporalMemoryNode:
    """Nodo de memoria temporal con contexto completo"""

    node_id: str
    timestamp: datetime
    content: Any
    context_vector: np.ndarray
    emotional_state: Dict[str, float]
    importance_score: float
    connections: List[str] = field(default_factory=list)
    decay_factor: float = 1.0
    access_count: int = 0
    last_accessed: datetime = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationEpisode:
    """Episodio completo de conversación"""

    episode_id: str
    start_time: datetime
    end_time: datetime
    participants: List[str]
    topic_evolution: List[str]
    emotional_trajectory: List[Dict[str, float]]
    key_insights: List[str]
    resolution_status: str
    memory_nodes: List[str]
    context_summary: str
    importance_score: float


@dataclass
class ContextualPattern:
    """Patrón contextual identificado"""

    pattern_id: str
    pattern_type: str  # 'temporal', 'emotional', 'thematic', 'relational'
    pattern_vector: np.ndarray
    occurrences: List[datetime]
    strength: float
    prediction_power: float
    associated_memories: List[str]


@dataclass
class TemporalContext:
    """Contexto temporal completo"""

    current_time: datetime
    time_since_last_interaction: timedelta
    conversation_duration: timedelta
    temporal_patterns: List[str]
    seasonal_context: str
    circadian_rhythm: str
    temporal_importance: float
    memory_relevance: float


class AdvancedTemporalMemorySystem:
    """
    Sistema de memoria temporal y contextual avanzada

    Características principales:
    - Memoria episódica con contexto temporal completo
    - Reconocimiento de patrones temporales
    - Evolución de conversaciones a largo plazo
    - Adaptación contextual inteligente
    - Predicción temporal y proyección futura
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()

        # Memoria temporal
        self.memory_nodes: Dict[str, TemporalMemoryNode] = {}
        self.memory_timeline: List[str] = []  # Orden temporal de nodos

        # Episodios de conversación
        self.conversation_episodes: Dict[str, ConversationEpisode] = {}
        self.active_episodes: Dict[str, ConversationEpisode] = {}

        # Patrones contextuales
        self.contextual_patterns: Dict[str, ContextualPattern] = {}

        # Sistema de predicción temporal
        self.temporal_predictor = TemporalPatternPredictor(
            self.config["memory_dimension"]
        )

        # Memoria de trabajo
        self.working_memory: deque = deque(maxlen=self.config["working_memory_size"])

        logger.info("⏰ Advanced Temporal Memory System initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Configuración por defecto"""
        return {
            "memory_dimension": 512,
            "max_memory_nodes": 10000,
            "decay_rate": 0.95,
            "importance_threshold": 0.3,
            "episode_timeout": 3600,  # 1 hora
            "pattern_recognition_window": 7 * 24 * 3600,  # 7 días
            "working_memory_size": 50,
            "temporal_relevance_decay": 0.9,
        }

    async def initialize_temporal_memory(self) -> bool:
        """Inicializar sistema de memoria temporal"""
        try:
            logger.info("⏰ Initializing Advanced Temporal Memory System...")

            # Inicializar componentes
            init_tasks = [self.temporal_predictor.initialize()]

            results = await asyncio.gather(*init_tasks, return_exceptions=True)

            success_count = sum(1 for r in results if not isinstance(r, Exception))
            if success_count >= 1:
                logger.info(
                    "✅ Advanced Temporal Memory System initialized successfully"
                )
                return True
            else:
                logger.error(
                    f"❌ Temporal memory initialization failed: {success_count}/1 components"
                )
                return False

        except Exception as e:
            logger.error(f"❌ Temporal memory initialization error: {e}")
            return False

    async def store_temporal_memory(
        self,
        content: Any,
        context: Dict[str, Any],
        user_id: str,
        importance: float = None,
    ) -> str:
        """
        Almacenar memoria temporal con contexto completo

        Args:
            content: Contenido a almacenar
            context: Contexto de la memoria
            user_id: ID del usuario
            importance: Puntaje de importancia (opcional)

        Returns:
            ID del nodo de memoria creado

        Raises:
            ValueError: Si los parámetros son inválidos
            RuntimeError: Si hay error interno en el procesamiento
        """
        try:
            # Validar parámetros
            if not user_id or not isinstance(user_id, str):
                raise ValueError("user_id must be a non-empty string")

            if content is None:
                raise ValueError("content cannot be None")

            if not isinstance(context, dict):
                context = {}

            # Generar ID único de nodo
            node_id = f"mem_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}_{len(self.memory_nodes)}"

            # Calcular importancia si no se proporciona
            if importance is None:
                importance = await self._calculate_importance(content, context)
            else:
                importance = max(0.0, min(1.0, importance))  # Clamp to [0,1]

            # Crear vector de contexto
            context_vector = await self._create_context_vector(content, context)

            # Crear nodo de memoria temporal
            memory_node = TemporalMemoryNode(
                node_id=node_id,
                timestamp=datetime.now(),
                content=content,
                context_vector=context_vector,
                emotional_state=context.get("emotional_state", {}),
                importance_score=importance,
                last_accessed=datetime.now(),
                metadata={**context, "user_id": user_id},  # Añadir user_id a metadata
            )

            # Conectar con memorias relacionadas
            await self._connect_related_memories(memory_node)

            # Gestionar tamaño de memoria
            await self._manage_memory_size()

            # Almacenar nodo
            self.memory_nodes[node_id] = memory_node
            self.memory_timeline.append(node_id)

            logger.info(f"⏰ Stored temporal memory node: {node_id}")

            return node_id

        except Exception as e:
            logger.error(f"❌ Error storing temporal memory: {e}")
            raise RuntimeError(f"Failed to store temporal memory: {e}") from e

    async def _create_context_vector(
        self, content: Any, context: Dict[str, Any]
    ) -> np.ndarray:
        """Crear vector de contexto combinando características de texto, temporales y contextuales"""
        context_features = []

        # Extraer características de texto
        text_features = await self._extract_text_features(content, context)
        context_features.extend(text_features)

        # Calcular factores temporales
        temporal_factors = await self._calculate_temporal_factors(context)
        context_features.extend(temporal_factors)

        # Asegurar que tengamos al menos algunas características básicas
        if not context_features:
            context_features = [0.0] * 10  # Características básicas por defecto

        # Crear vector del tamaño correcto
        if len(context_features) >= self.config["memory_dimension"]:
            context_vector = np.array(
                context_features[: self.config["memory_dimension"]]
            )
        else:
            # Rellenar duplicando hasta alcanzar la dimensión
            context_vector = np.array(context_features)
            while len(context_vector) < self.config["memory_dimension"]:
                context_vector = np.concatenate([context_vector, context_vector])
            context_vector = context_vector[: self.config["memory_dimension"]]

        return context_vector

    async def _extract_text_features(
        self, content: Any, context: Dict[str, Any]
    ) -> List[float]:
        """Extraer características de texto del contenido y contexto"""
        features = []

        # Convertir contenido a texto
        content_text = str(content) if content else ""
        context_text = (
            str(context.get("query", "")) + " " + str(context.get("response", ""))
        )

        combined_text = content_text + " " + context_text
        words = combined_text.split()

        # Características básicas de texto
        features.extend(
            [
                len(combined_text) / 1000.0,  # Longitud normalizada
                combined_text.count("?") / 10.0,  # Número de preguntas
                combined_text.count("!") / 10.0,  # Número de exclamaciones
                len(words) / 100.0,  # Número de palabras
            ]
        )

        # Características léxicas mejoradas
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            unique_words = len(set(words))
            lexical_diversity = unique_words / len(words) if words else 0.0

            features.extend(
                [
                    avg_word_length / 10.0,  # Longitud promedio de palabras
                    lexical_diversity,  # Diversidad léxica
                    sum(1 for word in words if word.isupper())
                    / len(words),  # Proporción de mayúsculas
                    sum(1 for word in words if word.isdigit())
                    / len(words),  # Proporción de números
                ]
            )

            # Características de frecuencia de palabras comunes (TF-like)
            common_words = [
                "the",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
            ]
            for word in common_words:
                features.append(
                    combined_text.lower().count(word) / len(words) if words else 0.0
                )
        else:
            # Características por defecto si no hay palabras
            features.extend([0.0] * (4 + 12))  # 4 léxicas + 12 palabras comunes

        return features

    async def _calculate_temporal_factors(self, context: Dict[str, Any]) -> List[float]:
        """Calcular factores temporales"""
        current_time = datetime.now()
        factors = []

        # Factores temporales básicos
        factors.extend(
            [
                current_time.hour / 24.0,  # Hora del día
                current_time.weekday() / 7.0,  # Día de la semana
                current_time.month / 12.0,  # Mes del año
            ]
        )

        # Factores de interacción
        factors.extend(
            [
                context.get("interaction_type", "unknown") == "question",  # Es pregunta
                context.get("urgency_level", 0.5),  # Nivel de urgencia
                context.get("complexity_level", 0.5),  # Nivel de complejidad
                len(context.get("participants", [])) / 10.0,  # Número de participantes
            ]
        )

        return factors

    async def _add_to_active_episode(self, memory_node: TemporalMemoryNode):
        relevant_episode = None

        for episode_id, episode in self.active_episodes.items():
            # Verificar si el episodio está relacionado (simplificado)
            time_diff = (memory_node.timestamp - episode.start_time).total_seconds()
            if time_diff < self.config["episode_timeout"]:
                relevant_episode = episode
                break

        if not relevant_episode:
            # Crear nuevo episodio
            episode_id = (
                f"episode_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
            )
            relevant_episode = ConversationEpisode(
                episode_id=episode_id,
                start_time=memory_node.timestamp,
                end_time=memory_node.timestamp,
                participants=memory_node.metadata.get("participants", []),
                topic_evolution=[],
                emotional_trajectory=[],
                key_insights=[],
                resolution_status="active",
                memory_nodes=[],
                context_summary="",
                importance_score=0.5,
            )
            self.active_episodes[episode_id] = relevant_episode

        # Añadir nodo al episodio
        relevant_episode.memory_nodes.append(memory_node.node_id)
        relevant_episode.end_time = memory_node.timestamp

        # Actualizar trayectoria emocional
        if memory_node.emotional_state:
            relevant_episode.emotional_trajectory.append(memory_node.emotional_state)

    async def _calculate_importance(
        self, content: Any, context: Dict[str, Any]
    ) -> float:
        """Calcular importancia de la memoria basada en contenido y contexto"""
        importance = 0.5  # Base

        # Factores de contenido
        content_str = str(content).lower()
        if any(
            keyword in content_str
            for keyword in ["importante", "crucial", "urgente", "critical"]
        ):
            importance += 0.2

        # Factores de contexto
        if context.get("interaction_type") == "question":
            importance += 0.1
        if context.get("urgency_level", 0) > 0.7:
            importance += 0.15
        if len(context.get("participants", [])) > 3:
            importance += 0.1

        # Factores emocionales
        emotional_state = context.get("emotional_state", {})
        if emotional_state.get("arousal", 0) > 0.7:
            importance += 0.1

        return min(1.0, importance)

    async def _connect_related_memories(self, memory_node: TemporalMemoryNode):
        """Conectar memoria con nodos relacionados basados en similitud"""
        if len(self.memory_nodes) < 2:
            return

        # Encontrar nodos similares
        similar_nodes = []
        for node_id, existing_node in self.memory_nodes.items():
            if node_id == memory_node.node_id:
                continue

            # Calcular similitud semántica (usando distancia coseno)
            similarity = 1 - cosine(
                memory_node.context_vector, existing_node.context_vector
            )
            if similarity > 0.7:  # Umbral de similitud
                similar_nodes.append((node_id, similarity))

        # Ordenar por similitud y tomar top 5
        similar_nodes.sort(key=lambda x: x[1], reverse=True)
        top_similar = similar_nodes[:5]

        # Crear enlaces bidireccionales
        for similar_node_id, similarity_score in top_similar:
            if similar_node_id not in memory_node.connections:
                memory_node.connections.append(similar_node_id)
            if (
                memory_node.node_id
                not in self.memory_nodes[similar_node_id].connections
            ):
                self.memory_nodes[similar_node_id].connections.append(
                    memory_node.node_id
                )

            logger.debug(
                f"Connected memory {memory_node.node_id} to {similar_node_id} (similarity: {similarity_score:.3f})"
            )

    async def _manage_memory_size(self):
        """Gestionar tamaño de memoria eliminando nodos menos relevantes de forma eficiente"""
        if len(self.memory_nodes) <= self.config["max_memory_nodes"]:
            return

        current_time = datetime.now()
        nodes_to_evaluate = len(self.memory_nodes) - self.config["max_memory_nodes"]

        # Calcular puntuaciones de relevancia para los nodos candidatos a eliminación
        node_scores = {}
        candidate_nodes = list(self.memory_nodes.keys())[
            -nodes_to_evaluate:
        ]  # Más recientes primero

        for node_id in candidate_nodes:
            node = self.memory_nodes[node_id]

            # Factor de decaimiento temporal optimizado
            time_since_creation = (current_time - node.timestamp).total_seconds() / (
                24 * 3600
            )  # días
            temporal_decay = (
                self.config["temporal_relevance_decay"] ** time_since_creation
            )

            # Puntuación combinada simplificada
            relevance_score = (
                node.importance_score * 0.5
                + temporal_decay * 0.3
                + node.decay_factor * 0.15
                + min(node.access_count * 0.05, 1.0)  # Cap access bonus
            )
            node_scores[node_id] = relevance_score

        # Mantener solo los más relevantes
        sorted_candidates = sorted(
            node_scores.items(), key=lambda x: x[1], reverse=True
        )
        nodes_to_keep = [
            node_id
            for node_id, _ in sorted_candidates[: self.config["max_memory_nodes"] // 2]
        ]

        # Si aún tenemos demasiados nodos, eliminar los menos relevantes globalmente
        if len(self.memory_nodes) > self.config["max_memory_nodes"]:
            all_scores = {}
            for node_id, node in self.memory_nodes.items():
                time_since_creation = (
                    current_time - node.timestamp
                ).total_seconds() / (24 * 3600)
                temporal_decay = (
                    self.config["temporal_relevance_decay"] ** time_since_creation
                )
                all_scores[node_id] = (
                    node.importance_score * 0.5
                    + temporal_decay * 0.3
                    + node.decay_factor * 0.2
                )

            sorted_all = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            nodes_to_keep = [
                node_id for node_id, _ in sorted_all[: self.config["max_memory_nodes"]]
            ]

        # Eliminar nodos menos relevantes
        nodes_to_remove = set(self.memory_nodes.keys()) - set(nodes_to_keep)
        for node_id in nodes_to_remove:
            del self.memory_nodes[node_id]

        # Actualizar timeline de forma eficiente
        self.memory_timeline = [
            nid for nid in self.memory_timeline if nid in self.memory_nodes
        ]

        if nodes_to_remove:
            logger.info(
                f"🗑️ Managed memory size: removed {len(nodes_to_remove)} nodes, kept {len(self.memory_nodes)}"
            )

    async def _consolidate_memory(self):
        """Consolidar memoria aplicando decaimiento y refuerzo"""
        current_time = datetime.now()

        for node in self.memory_nodes.values():
            # Calcular tiempo desde último acceso
            time_since_access = (current_time - node.last_accessed).total_seconds()
            hours_since_access = time_since_access / 3600.0

            # Aplicar decaimiento
            decay_factor = self.config["decay_rate"] ** hours_since_access
            node.decay_factor *= decay_factor

            # Refuerzo por importancia y acceso frecuente
            reinforcement = node.importance_score * 0.3 + node.access_count * 0.1
            node.decay_factor = min(1.0, node.decay_factor + reinforcement * 0.1)

    async def _prune_memory(self):
        """Eliminar memorias menos relevantes para mantener límites"""
        if len(self.memory_nodes) <= self.config["max_memory_nodes"]:
            return

        # Ordenar nodos por relevancia (importancia * decaimiento * acceso)
        node_scores = {}
        for node_id, node in self.memory_nodes.items():
            relevance_score = (
                node.importance_score
                * node.decay_factor
                * (1 + node.access_count * 0.1)
            )
            node_scores[node_id] = relevance_score

        # Mantener solo los más relevantes
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        nodes_to_keep = [
            node_id for node_id, _ in sorted_nodes[: self.config["max_memory_nodes"]]
        ]

        # Eliminar nodos menos relevantes
        nodes_to_remove = set(self.memory_nodes.keys()) - set(nodes_to_keep)
        for node_id in nodes_to_remove:
            del self.memory_nodes[node_id]

        # Actualizar timeline
        self.memory_timeline = [
            nid for nid in self.memory_timeline if nid in self.memory_nodes
        ]

        logger.info(f"🗑️ Pruned {len(nodes_to_remove)} low-relevance memory nodes")

    async def retrieve_temporal_context(
        self, query_time: datetime = None, context_window: timedelta = None
    ) -> TemporalContext:
        """
        Recuperar contexto temporal completo

        Args:
            query_time: Tiempo de consulta (por defecto ahora)
            context_window: Ventana de contexto temporal

        Returns:
            Contexto temporal completo
        """
        query_time = query_time or datetime.now()
        context_window = context_window or timedelta(hours=1)

        # Calcular tiempo desde última interacción
        last_interaction_time = await self._get_last_interaction_time()
        time_since_last = (
            query_time - last_interaction_time
            if last_interaction_time
            else timedelta.max
        )

        # Calcular duración de conversación actual
        conversation_duration = await self._get_current_conversation_duration(
            query_time
        )

        # Identificar patrones temporales relevantes
        temporal_patterns = await self._identify_temporal_patterns(
            query_time, context_window
        )

        # Determinar contexto estacional y circadiano
        seasonal_context = self._get_seasonal_context(query_time)
        circadian_rhythm = self._get_circadian_rhythm(query_time)

        # Calcular importancia temporal
        temporal_importance = await self._calculate_temporal_importance(
            query_time, context_window
        )

        # Calcular relevancia de memoria
        memory_relevance = await self._calculate_memory_relevance(query_time)

        return TemporalContext(
            current_time=query_time,
            time_since_last_interaction=time_since_last,
            conversation_duration=conversation_duration,
            temporal_patterns=temporal_patterns,
            seasonal_context=seasonal_context,
            circadian_rhythm=circadian_rhythm,
            temporal_importance=temporal_importance,
            memory_relevance=memory_relevance,
        )

    async def _get_last_interaction_time(self) -> Optional[datetime]:
        """Obtener tiempo de última interacción"""
        if not self.memory_timeline:
            return None

        last_node_id = self.memory_timeline[-1]
        return self.memory_nodes[last_node_id].timestamp

    async def _get_current_conversation_duration(
        self, current_time: datetime
    ) -> timedelta:
        """Calcular duración de conversación actual"""
        if not self.active_episodes:
            return timedelta(0)

        # Tomar el episodio más reciente
        latest_episode = max(self.active_episodes.values(), key=lambda x: x.start_time)
        return current_time - latest_episode.start_time

    async def _identify_temporal_patterns(
        self, query_time: datetime, context_window: timedelta
    ) -> List[str]:
        """Identificar patrones temporales relevantes"""
        patterns = []

        # Buscar nodos en ventana de contexto
        window_start = query_time - context_window
        relevant_nodes = [
            node
            for node in self.memory_nodes.values()
            if window_start <= node.timestamp <= query_time
        ]

        if len(relevant_nodes) < 3:
            return patterns

        # Analizar patrones de frecuencia
        timestamps = [node.timestamp for node in relevant_nodes]
        intervals = []

        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i - 1]).total_seconds()
            intervals.append(interval)

        if intervals:
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)

            if std_interval / avg_interval < 0.5:  # Patrón regular
                patterns.append("interacciones_regulares")
            else:
                patterns.append("interacciones_irregulares")

        # Analizar patrones de importancia
        importance_trend = [node.importance_score for node in relevant_nodes[-5:]]
        if len(importance_trend) >= 3:
            if importance_trend[-1] > importance_trend[0]:
                patterns.append("aumento_importancia")
            elif importance_trend[-1] < importance_trend[0]:
                patterns.append("disminución_importancia")

        return patterns

    def _get_seasonal_context(self, query_time: datetime) -> str:
        """Determinar contexto estacional"""
        month = query_time.month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"

    def _get_circadian_rhythm(self, query_time: datetime) -> str:
        """Determinar ritmo circadiano"""
        hour = query_time.hour
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"

    async def _calculate_temporal_importance(
        self, query_time: datetime, context_window: timedelta
    ) -> float:
        """Calcular importancia temporal del momento actual"""
        # Buscar nodos recientes
        window_start = query_time - context_window
        recent_nodes = [
            node
            for node in self.memory_nodes.values()
            if window_start <= node.timestamp <= query_time
        ]

        if not recent_nodes:
            return 0.5

        # Importancia basada en promedio de importancia de nodos recientes
        avg_importance = np.mean([node.importance_score for node in recent_nodes])

        # Bonus por frecuencia de interacciones recientes
        interaction_frequency = (
            len(recent_nodes) / context_window.total_seconds() * 3600
        )  # por hora
        frequency_bonus = min(0.3, interaction_frequency * 0.1)

        return min(1.0, avg_importance + frequency_bonus)

    async def _calculate_memory_relevance(self, query_time: datetime) -> float:
        """Calcular relevancia de la memoria en el momento actual"""
        if not self.memory_nodes:
            return 0.0

        # Calcular decaimiento promedio de memoria
        total_decay = sum(node.decay_factor for node in self.memory_nodes.values())
        avg_decay = total_decay / len(self.memory_nodes)

        # Relevancia basada en frescura y acceso
        recent_accesses = sum(
            1
            for node in self.memory_nodes.values()
            if node.last_accessed
            and (query_time - node.last_accessed).total_seconds() < 3600
        )  # Última hora

        access_ratio = recent_accesses / len(self.memory_nodes)

        return avg_decay * 0.7 + access_ratio * 0.3

    async def search_temporal_memories(
        self, query: str, temporal_filters: Dict[str, Any] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Buscar memorias con filtros temporales

        Args:
            query: Consulta de búsqueda
            temporal_filters: Filtros temporales
            limit: Número máximo de resultados

        Returns:
            Lista de memorias relevantes
        """
        # Crear vector de consulta
        query_vector = await self._create_context_vector({"query": query})

        results = []

        for node_id, node in self.memory_nodes.items():
            # Verificar filtros temporales
            if temporal_filters:
                if not await self._matches_temporal_filters(node, temporal_filters):
                    continue

            # Calcular similitud
            similarity = 1 - cosine(query_vector, node.context_vector)

            if similarity > 0.3:  # Umbral de relevancia
                # Actualizar acceso
                node.access_count += 1
                node.last_accessed = datetime.now()

                results.append(
                    {
                        "node_id": node_id,
                        "content": node.content,
                        "similarity": similarity,
                        "timestamp": node.timestamp,
                        "importance": node.importance_score,
                        "emotional_state": node.emotional_state,
                        "decay_factor": node.decay_factor,
                        "metadata": node.metadata,
                    }
                )

        # Ordenar por similitud y recencia
        results.sort(
            key=lambda x: (x["similarity"] * 0.7 + x["decay_factor"] * 0.3),
            reverse=True,
        )

        return results[:limit]

    async def retrieve_contextual_memories(
        self,
        query: Any,
        user_id: str,
        temporal_context: TemporalContext,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Recuperar memorias contextuales relevantes para una consulta

        Args:
            query: Consulta de búsqueda
            user_id: ID del usuario
            temporal_context: Contexto temporal actual
            limit: Número máximo de resultados

        Returns:
            Lista de memorias relevantes con metadatos
        """
        # Filtrar memorias del usuario
        user_memories = {
            node_id: node
            for node_id, node in self.memory_nodes.items()
            if node.metadata.get("user_id") == user_id
        }

        if not user_memories:
            return []

        # Crear vector de consulta
        query_vector = await self._create_context_vector(
            query, {"temporal_context": temporal_context}
        )

        # Calcular relevancia para cada nodo
        memory_relevance = []
        for node_id, node in user_memories.items():
            relevance_score = await self._calculate_memory_relevance(
                query_vector, node, temporal_context
            )
            memory_relevance.append((node_id, relevance_score))

        # Ordenar por puntuación de relevancia
        memory_relevance.sort(key=lambda x: x[1], reverse=True)

        # Devolver top memories con metadatos
        results = []
        for node_id, relevance_score in memory_relevance[:limit]:
            node = user_memories[node_id]
            # Actualizar acceso
            node.access_count += 1
            node.last_accessed = datetime.now()

            results.append(
                {
                    "node_id": node_id,
                    "content": node.content,
                    "relevance_score": relevance_score,
                    "timestamp": node.timestamp,
                    "importance": node.importance_score,
                    "emotional_state": node.emotional_state,
                    "decay_factor": node.decay_factor,
                    "connections": node.connections,
                    "metadata": node.metadata,
                }
            )

        return results

    async def _calculate_memory_relevance(
        self,
        query_vector: np.ndarray,
        node: TemporalMemoryNode,
        temporal_context: TemporalContext,
    ) -> float:
        """Calcular puntuación de relevancia multi-factor para una memoria"""
        # Similitud semántica
        semantic_similarity = 1 - cosine(query_vector, node.context_vector)

        # Factor de decaimiento temporal
        time_diff_hours = (datetime.now() - node.timestamp).total_seconds() / 3600.0
        temporal_decay_factor = (
            self.config["temporal_relevance_decay"] ** time_diff_hours
        )

        # Ponderación de importancia
        importance_weighting = node.importance_score

        # Relevancia contextual (basada en contexto temporal actual)
        context_relevance = self._calculate_context_relevance(node, temporal_context)

        # Puntuación combinada
        relevance_score = (
            semantic_similarity * 0.4
            + temporal_decay_factor * 0.3
            + importance_weighting * 0.2
            + context_relevance * 0.1
        )

        return relevance_score

    def _calculate_context_relevance(
        self, node: TemporalMemoryNode, temporal_context: TemporalContext
    ) -> float:
        """Calcular relevancia basada en contexto temporal"""
        relevance = 0.0

        # Relevancia por patrón temporal
        if temporal_context.temporal_patterns:
            node_patterns = node.metadata.get("temporal_patterns", [])
            pattern_overlap = len(
                set(temporal_context.temporal_patterns) & set(node_patterns)
            )
            relevance += pattern_overlap * 0.2

        # Relevancia por ritmo circadiano
        if temporal_context.circadian_rhythm == node.metadata.get("circadian_rhythm"):
            relevance += 0.1

        # Relevancia por contexto estacional
        if temporal_context.seasonal_context == node.metadata.get("seasonal_context"):
            relevance += 0.1

        return min(1.0, relevance)

    async def _matches_temporal_filters(
        self, node: TemporalMemoryNode, filters: Dict[str, Any]
    ) -> bool:
        """Verificar si un nodo cumple con los filtros temporales"""
        # Filtro de tiempo mínimo
        if "min_time" in filters:
            if node.timestamp < filters["min_time"]:
                return False

        # Filtro de tiempo máximo
        if "max_time" in filters:
            if node.timestamp > filters["max_time"]:
                return False

        # Filtro de importancia mínima
        if "min_importance" in filters:
            if node.importance_score < filters["min_importance"]:
                return False

        # Filtro de emoción
        if "emotion" in filters:
            required_emotion = filters["emotion"]
            if (
                required_emotion not in node.emotional_state
                or node.emotional_state[required_emotion] < 0.5
            ):
                return False

        return True

    async def predict_temporal_patterns(
        self, prediction_horizon: timedelta
    ) -> Dict[str, Any]:
        """
        Predecir patrones temporales futuros

        Args:
            prediction_horizon: Horizonte de predicción

        Returns:
            Predicciones de patrones temporales
        """
        # Usar predictor temporal para hacer predicciones
        current_patterns = await self._identify_temporal_patterns(
            datetime.now(), timedelta(hours=24)
        )

        predictions = await self.temporal_predictor.predict_patterns(
            current_patterns, prediction_horizon
        )

        return {
            "current_patterns": current_patterns,
            "predicted_patterns": predictions.get("patterns", []),
            "confidence": predictions.get("confidence", 0.0),
            "prediction_horizon": prediction_horizon.total_seconds(),
            "timestamp": datetime.now(),
        }

    async def get_temporal_memory_status(self) -> Dict[str, Any]:
        """Obtener estado del sistema de memoria temporal"""
        total_nodes = len(self.memory_nodes)
        total_episodes = len(self.conversation_episodes)
        active_episodes = len(self.active_episodes)

        # Estadísticas de memoria
        if total_nodes > 0:
            avg_importance = np.mean(
                [node.importance_score for node in self.memory_nodes.values()]
            )
            avg_decay = np.mean(
                [node.decay_factor for node in self.memory_nodes.values()]
            )
            total_accesses = sum(
                node.access_count for node in self.memory_nodes.values()
            )
        else:
            avg_importance = avg_decay = total_accesses = 0.0

        return {
            "total_memory_nodes": total_nodes,
            "total_episodes": total_episodes,
            "active_episodes": active_episodes,
            "average_importance": avg_importance,
            "average_decay": avg_decay,
            "total_accesses": total_accesses,
            "memory_utilization": total_nodes / self.config["max_memory_nodes"],
            "working_memory_size": len(self.working_memory),
            "contextual_patterns": len(self.contextual_patterns),
            "config": self.config,
        }


# Componentes auxiliares


class TemporalPatternPredictor:
    """Predictor de patrones temporales"""

    def __init__(self, dimension: int):
        self.dimension = dimension

    async def initialize(self):
        pass

    async def predict_patterns(
        self, current_patterns: List[str], prediction_horizon: timedelta
    ) -> Dict[str, Any]:
        """Predecir patrones temporales futuros"""
        # Predicción simplificada basada en patrones actuales
        predictions = []

        if "interacciones_regulares" in current_patterns:
            predictions.append("continuidad_patron_regular")
        if "aumento_importancia" in current_patterns:
            predictions.append("escalada_importancia")
        if "interacciones_irregulares" in current_patterns:
            predictions.append("posible_cambio_patron")

        # Confianza basada en claridad de patrones
        confidence = min(1.0, len(current_patterns) * 0.2 + 0.5)

        return {
            "patterns": predictions,
            "confidence": confidence,
            "horizon_seconds": prediction_horizon.total_seconds(),
        }


# Instancia global
advanced_temporal_memory_system = AdvancedTemporalMemorySystem()


async def store_temporal_memory(
    content: Any, context: Dict[str, Any], user_id: str, importance: float = None
) -> str:
    """Función pública para almacenar memoria temporal"""
    return await advanced_temporal_memory_system.store_temporal_memory(
        content, context, user_id, importance
    )


async def retrieve_temporal_context(
    query_time: datetime = None, context_window: timedelta = None
) -> TemporalContext:
    """Función pública para recuperar contexto temporal"""
    return await advanced_temporal_memory_system.retrieve_temporal_context(
        query_time, context_window
    )


async def search_temporal_memories(
    query: str, temporal_filters: Dict[str, Any] = None, limit: int = 10
) -> List[Dict[str, Any]]:
    """Función pública para buscar memorias temporales"""
    return await advanced_temporal_memory_system.search_temporal_memories(
        query, temporal_filters, limit
    )


async def retrieve_contextual_memories(
    query: Any, user_id: str, temporal_context: TemporalContext, limit: int = 10
) -> List[Dict[str, Any]]:
    """Función pública para recuperar memorias contextuales"""
    return await advanced_temporal_memory_system.retrieve_contextual_memories(
        query, user_id, temporal_context, limit
    )


async def predict_temporal_patterns(prediction_horizon: timedelta) -> Dict[str, Any]:
    """Función pública para predecir patrones temporales"""
    return await advanced_temporal_memory_system.predict_temporal_patterns(
        prediction_horizon
    )


async def get_temporal_memory_status() -> Dict[str, Any]:
    """Función pública para estado de memoria temporal"""
    return await advanced_temporal_memory_system.get_temporal_memory_status()


async def initialize_temporal_memory() -> bool:
    """Función pública para inicializar memoria temporal"""
    return await advanced_temporal_memory_system.initialize_temporal_memory()


# Información del módulo
__version__ = "1.0.0"
__author__ = "Sheily AI Team - Advanced Temporal Memory System"
__description__ = "Sistema de memoria temporal y contextual avanzada"
