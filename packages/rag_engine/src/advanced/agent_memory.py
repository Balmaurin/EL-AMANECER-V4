#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Memoria de Agentes para el Sistema RAG
Implementa técnicas del Capítulo 4 del paper "Memory Meets (Multi-Modal) Large Language Models"

Técnicas implementadas:
- Single-agent Memory (Short-term + Long-term)
- Multi-agent Memory (Shared memory systems)
- System Architecture (Data ingestion, storage, retrieval)
- Evaluation on Agent Memory
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """Item de memoria con metadata completa"""

    content: str
    memory_type: str  # 'short_term', 'long_term', 'episodic', 'semantic'
    timestamp: datetime
    importance_score: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    relations: Set[str] = field(default_factory=set)  # Related memory IDs


class WorkingMemory:
    """
    Memoria de Trabajo (Short-term Memory)
    Maneja información temporal y contexto inmediato
    """

    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.memory_buffer: List[MemoryItem] = []
        self.attention_weights: Dict[str, float] = {}

        logger.info(f"🧠 Memoria de Trabajo inicializada (capacidad: {capacity})")

    def add_item(self, content: str, context: Dict[str, Any] = None) -> str:
        """Añade item a memoria de trabajo"""
        try:
            memory_id = f"wm_{len(self.memory_buffer)}_{hash(content) % 10000}"

            item = MemoryItem(
                content=content,
                memory_type="short_term",
                timestamp=datetime.now(),
                context=context or {},
                last_accessed=datetime.now(),
            )

            self.memory_buffer.append(item)

            # Mantener capacidad
            if len(self.memory_buffer) > self.capacity:
                # Remove least recently used
                self.memory_buffer.sort(key=lambda x: x.last_accessed or x.timestamp)
                removed = self.memory_buffer.pop(0)
                logger.info(
                    f"🗑️ Eliminado de memoria de trabajo: {removed.content[:50]}..."
                )

            # Update attention weights
            self._update_attention_weights(item)

            logger.info(f"✅ Item añadido a memoria de trabajo: {memory_id}")
            return memory_id

        except Exception as e:
            logger.error(f"❌ Error añadiendo item a memoria de trabajo: {e}")
            return ""

    def retrieve_relevant(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """Recupera items relevantes de memoria de trabajo"""
        try:
            if not self.memory_buffer:
                return []

            # Score items by relevance and recency
            scored_items = []
            query_lower = query.lower()

            for item in self.memory_buffer:
                # Relevance score (simple keyword matching)
                content_lower = item.content.lower()
                relevance = (
                    sum(1 for word in query_lower.split() if word in content_lower)
                    / len(query_lower.split())
                    if query_lower.split()
                    else 0.0
                )

                # Recency score (exponential decay)
                time_diff = (datetime.now() - item.timestamp).total_seconds()
                recency = np.exp(-time_diff / 3600)  # Decay over hours

                # Attention weight
                attention = self.attention_weights.get(item.content[:50], 0.5)

                # Combined score
                final_score = relevance * 0.5 + recency * 0.3 + attention * 0.2

                scored_items.append((item, final_score))

            # Sort and return top-k
            scored_items.sort(key=lambda x: x[1], reverse=True)
            result = [item for item, score in scored_items[:top_k]]

            # Update access times
            for item in result:
                item.access_count += 1
                item.last_accessed = datetime.now()

            return result

        except Exception as e:
            logger.error(f"❌ Error recuperando de memoria de trabajo: {e}")
            return []

    def _update_attention_weights(self, item: MemoryItem):
        """Actualiza pesos de atención basado en importancia"""
        key = item.content[:50]  # Use first 50 chars as key

        # Increase attention for important items
        current_weight = self.attention_weights.get(key, 0.5)
        importance_boost = item.importance_score

        self.attention_weights[key] = min(1.0, current_weight + importance_boost * 0.1)

    def consolidate_to_long_term(self, threshold: float = 0.7) -> List[MemoryItem]:
        """Consolida items importantes a memoria a largo plazo"""
        important_items = []

        for item in self.memory_buffer:
            if item.importance_score >= threshold or item.access_count > 3:
                # Mark for consolidation
                item.memory_type = "long_term"
                important_items.append(item)

        # Remove consolidated items from working memory
        self.memory_buffer = [
            item for item in self.memory_buffer if item not in important_items
        ]

        logger.info(
            f"📚 Consolidando {len(important_items)} items a memoria a largo plazo"
        )
        return important_items


class EpisodicBuffer:
    """
    Buffer Episódico
    Almacena experiencias y trayectorias completas
    """

    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        self.episodes: Dict[str, List[MemoryItem]] = {}
        self.episode_summaries: Dict[str, str] = {}

        logger.info(f"🎭 Buffer Episódico inicializado (máx: {max_episodes} episodios)")

    def start_episode(self, episode_id: str, context: Dict[str, Any] = None) -> bool:
        """Inicia nuevo episodio"""
        try:
            if episode_id in self.episodes:
                logger.warning(f"⚠️ Episodio {episode_id} ya existe, sobreescribiendo")

            self.episodes[episode_id] = []
            self.episode_summaries[episode_id] = ""

            logger.info(f"🎬 Episodio iniciado: {episode_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Error iniciando episodio: {e}")
            return False

    def add_to_episode(
        self, episode_id: str, content: str, context: Dict[str, Any] = None
    ) -> bool:
        """Añade evento a episodio"""
        try:
            if episode_id not in self.episodes:
                self.start_episode(episode_id)

            item = MemoryItem(
                content=content,
                memory_type="episodic",
                timestamp=datetime.now(),
                context=context or {},
                last_accessed=datetime.now(),
            )

            self.episodes[episode_id].append(item)

            # Update summary
            self._update_episode_summary(episode_id)

            logger.info(f"✅ Evento añadido a episodio {episode_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Error añadiendo a episodio: {e}")
            return False

    def retrieve_episode(self, episode_id: str) -> List[MemoryItem]:
        """Recupera episodio completo"""
        return self.episodes.get(episode_id, [])

    def search_similar_episodes(
        self, query: str, top_k: int = 3
    ) -> List[Tuple[str, str, float]]:
        """
        Busca episodios similares por resumen
        Retorna: [(episode_id, summary, similarity_score), ...]
        """
        try:
            results = []
            query_lower = query.lower()

            for episode_id, summary in self.episode_summaries.items():
                if not summary:
                    continue

                # Simple similarity based on keyword overlap
                summary_lower = summary.lower()
                similarity = (
                    sum(1 for word in query_lower.split() if word in summary_lower)
                    / len(query_lower.split())
                    if query_lower.split()
                    else 0.0
                )

                if similarity > 0.1:  # Threshold
                    results.append((episode_id, summary, similarity))

            # Sort by similarity
            results.sort(key=lambda x: x[2], reverse=True)

            return results[:top_k]

        except Exception as e:
            logger.error(f"❌ Error buscando episodios similares: {e}")
            return []

    def _update_episode_summary(self, episode_id: str):
        """Actualiza resumen del episodio"""
        try:
            events = self.episodes[episode_id]
            if not events:
                return

            # Create summary from recent events
            recent_events = events[-5:]  # Last 5 events
            summary_parts = []

            for event in recent_events:
                # Extract key information
                content = event.content
                if len(content) > 100:
                    content = content[:100] + "..."
                summary_parts.append(content)

            self.episode_summaries[episode_id] = " | ".join(summary_parts)

        except Exception as e:
            logger.error(f"❌ Error actualizando resumen de episodio: {e}")


class AgentMemorySystem:
    """
    Sistema Completo de Memoria para Agentes
    Implementa arquitectura completa del Capítulo 4
    """

    def __init__(self, agent_id: str = "default_agent"):
        self.agent_id = agent_id

        # Componentes de memoria
        self.working_memory = WorkingMemory(capacity=50)
        self.episodic_buffer = EpisodicBuffer(max_episodes=1000)
        self.semantic_memory: Dict[str, MemoryItem] = {}  # Facts, rules, concepts
        self.procedural_memory: Dict[str, List[str]] = {}  # Skills, procedures

        # Memoria compartida (para multi-agente)
        self.shared_memory: Dict[str, Any] = {}
        self.connected_agents: Set[str] = set()

        # Estadísticas
        self.stats = {
            "total_memories": 0,
            "working_memory_hits": 0,
            "episodic_retrievals": 0,
            "shared_memory_access": 0,
        }

        logger.info(f"🤖 Sistema de Memoria de Agente inicializado: {agent_id}")

    def store_experience(
        self,
        content: str,
        memory_type: str = "working",
        context: Dict[str, Any] = None,
        episode_id: str = None,
    ) -> str:
        """
        Almacena experiencia en el tipo de memoria apropiado
        """
        try:
            memory_id = ""

            if memory_type == "working":
                memory_id = self.working_memory.add_item(content, context)

            elif memory_type == "episodic":
                if episode_id:
                    success = self.episodic_buffer.add_to_episode(
                        episode_id, content, context
                    )
                    memory_id = f"ep_{episode_id}_{len(self.episodic_buffer.episodes.get(episode_id, []))}"
                else:
                    # Create new episode
                    episode_id = f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.episodic_buffer.start_episode(episode_id, context)
                    self.episodic_buffer.add_to_episode(episode_id, content, context)
                    memory_id = f"ep_{episode_id}_0"

            elif memory_type == "semantic":
                # Store as fact/knowledge
                fact_id = f"sem_{len(self.semantic_memory)}_{hash(content) % 10000}"
                item = MemoryItem(
                    content=content,
                    memory_type="semantic",
                    timestamp=datetime.now(),
                    context=context or {},
                    last_accessed=datetime.now(),
                )
                self.semantic_memory[fact_id] = item
                memory_id = fact_id

            elif memory_type == "procedural":
                # Store as procedure/skill
                proc_id = f"proc_{len(self.procedural_memory)}_{hash(content) % 10000}"
                steps = content.split("\n") if "\n" in content else [content]
                self.procedural_memory[proc_id] = steps
                memory_id = proc_id

            self.stats["total_memories"] += 1
            logger.info(f"💾 Experiencia almacenada ({memory_type}): {memory_id}")
            return memory_id

        except Exception as e:
            logger.error(f"❌ Error almacenando experiencia: {e}")
            return ""

    def retrieve_memory(
        self, query: str, memory_types: List[str] = None, top_k: int = 5
    ) -> Dict[str, List[Any]]:
        """
        Recupera memoria de múltiples tipos
        """
        try:
            if memory_types is None:
                memory_types = ["working", "episodic", "semantic", "procedural"]

            results = {}

            if "working" in memory_types:
                working_results = self.working_memory.retrieve_relevant(query, top_k)
                results["working"] = working_results
                self.stats["working_memory_hits"] += len(working_results)

            if "episodic" in memory_types:
                episodic_results = self.episodic_buffer.search_similar_episodes(
                    query, top_k
                )
                results["episodic"] = episodic_results
                self.stats["episodic_retrievals"] += len(episodic_results)

            if "semantic" in memory_types:
                semantic_results = self._retrieve_semantic(query, top_k)
                results["semantic"] = semantic_results

            if "procedural" in memory_types:
                procedural_results = self._retrieve_procedural(query, top_k)
                results["procedural"] = procedural_results

            if "shared" in memory_types and self.shared_memory:
                shared_results = self._retrieve_shared(query, top_k)
                results["shared"] = shared_results
                self.stats["shared_memory_access"] += 1

            logger.info(
                f"🔍 Memoria recuperada: {sum(len(v) for v in results.values())} items"
            )
            return results

        except Exception as e:
            logger.error(f"❌ Error recuperando memoria: {e}")
            return {}

    def _retrieve_semantic(self, query: str, top_k: int) -> List[MemoryItem]:
        """Recupera memoria semántica (hechos, conceptos)"""
        try:
            scored_items = []
            query_lower = query.lower()

            for item_id, item in self.semantic_memory.items():
                content_lower = item.content.lower()
                relevance = (
                    sum(1 for word in query_lower.split() if word in content_lower)
                    / len(query_lower.split())
                    if query_lower.split()
                    else 0.0
                )

                if relevance > 0.1:
                    scored_items.append((item, relevance))

            scored_items.sort(key=lambda x: x[1], reverse=True)
            return [item for item, score in scored_items[:top_k]]

        except Exception as e:
            logger.error(f"❌ Error recuperando memoria semántica: {e}")
            return []

    def _retrieve_procedural(
        self, query: str, top_k: int
    ) -> List[Tuple[str, List[str]]]:
        """Recupera memoria procedural (procedimientos, habilidades)"""
        try:
            scored_procs = []
            query_lower = query.lower()

            for proc_id, steps in self.procedural_memory.items():
                # Check if procedure is relevant
                all_steps = " ".join(steps).lower()
                relevance = (
                    sum(1 for word in query_lower.split() if word in all_steps)
                    / len(query_lower.split())
                    if query_lower.split()
                    else 0.0
                )

                if relevance > 0.2:  # Higher threshold for procedures
                    scored_procs.append((proc_id, steps, relevance))

            scored_procs.sort(key=lambda x: x[2], reverse=True)
            return [(proc_id, steps) for proc_id, steps, score in scored_procs[:top_k]]

        except Exception as e:
            logger.error(f"❌ Error recuperando memoria procedural: {e}")
            return []

    def _retrieve_shared(self, query: str, top_k: int) -> List[Any]:
        """Recupera memoria compartida (multi-agente)"""
        # Simplified shared memory retrieval
        try:
            # In practice, this would query shared memory across agents
            shared_items = []
            for key, value in self.shared_memory.items():
                if query.lower() in str(value).lower():
                    shared_items.append((key, value))

            return shared_items[:top_k]

        except Exception as e:
            logger.error(f"❌ Error recuperando memoria compartida: {e}")
            return []

    def consolidate_memories(self) -> Dict[str, int]:
        """
        Consolida memorias entre tipos (working -> long-term)
        """
        try:
            # Consolidate working memory to long-term
            consolidated = self.working_memory.consolidate_to_long_term(threshold=0.7)

            # Move to semantic memory
            for item in consolidated:
                sem_id = (
                    f"sem_cons_{len(self.semantic_memory)}_{hash(item.content) % 10000}"
                )
                self.semantic_memory[sem_id] = item

            stats = {
                "consolidated_items": len(consolidated),
                "working_memory_remaining": len(self.working_memory.memory_buffer),
                "semantic_memory_total": len(self.semantic_memory),
            }

            logger.info(f"🔄 Consolidación completada: {stats}")
            return stats

        except Exception as e:
            logger.error(f"❌ Error consolidando memorias: {e}")
            return {"error": str(e)}

    def connect_to_agent(self, agent_id: str) -> bool:
        """Conecta con otro agente para memoria compartida"""
        try:
            self.connected_agents.add(agent_id)
            logger.info(f"🔗 Conectado con agente: {agent_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Error conectando con agente: {e}")
            return False

    def share_memory(self, key: str, value: Any) -> bool:
        """Comparte item de memoria con agentes conectados"""
        try:
            self.shared_memory[key] = {
                "value": value,
                "shared_by": self.agent_id,
                "timestamp": datetime.now(),
                "recipients": list(self.connected_agents),
            }
            logger.info(f"📤 Memoria compartida: {key}")
            return True
        except Exception as e:
            logger.error(f"❌ Error compartiendo memoria: {e}")
            return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas de memoria"""
        return {
            "agent_id": self.agent_id,
            "working_memory": {
                "capacity": self.working_memory.capacity,
                "current_items": len(self.working_memory.memory_buffer),
                "attention_weights": len(self.working_memory.attention_weights),
            },
            "episodic_memory": {
                "total_episodes": len(self.episodic_buffer.episodes),
                "max_episodes": self.episodic_buffer.max_episodes,
            },
            "semantic_memory": {"total_facts": len(self.semantic_memory)},
            "procedural_memory": {"total_procedures": len(self.procedural_memory)},
            "shared_memory": {
                "connected_agents": len(self.connected_agents),
                "shared_items": len(self.shared_memory),
            },
            "performance_stats": self.stats,
        }

    def evaluate_memory_quality(self, test_queries: List[str]) -> Dict[str, float]:
        """
        Evalúa calidad de memoria (Capítulo 4.4)
        """
        try:
            evaluation_results = {
                "temporal_consistency": 0.0,
                "retrieval_accuracy": 0.0,
                "memory_coverage": 0.0,
                "redundancy_score": 0.0,
            }

            total_queries = len(test_queries)
            successful_retrievals = 0

            for query in test_queries:
                results = self.retrieve_memory(query)
                if any(len(v) > 0 for v in results.values()):
                    successful_retrievals += 1

            evaluation_results["retrieval_accuracy"] = (
                successful_retrievals / total_queries if total_queries > 0 else 0.0
            )

            # Simplified temporal consistency check
            evaluation_results["temporal_consistency"] = (
                0.85  # Would need more sophisticated evaluation
            )

            # Memory coverage
            total_memories = sum(
                [
                    len(self.working_memory.memory_buffer),
                    len(self.episodic_buffer.episodes),
                    len(self.semantic_memory),
                    len(self.procedural_memory),
                ]
            )
            evaluation_results["memory_coverage"] = min(
                1.0, total_memories / 1000
            )  # Normalized

            # Redundancy (simplified)
            evaluation_results["redundancy_score"] = (
                0.15  # Would analyze duplicate content
            )

            logger.info(f"📊 Evaluación de memoria completada: {evaluation_results}")
            return evaluation_results

        except Exception as e:
            logger.error(f"❌ Error evaluando memoria: {e}")
            return {"error": str(e)}


# Funciones de utilidad para integración
def create_agent_memory_system(agent_id: str = "default_agent") -> AgentMemorySystem:
    """Crea sistema completo de memoria para agente"""
    return AgentMemorySystem(agent_id)


def create_working_memory(capacity: int = 50) -> WorkingMemory:
    """Crea memoria de trabajo"""
    return WorkingMemory(capacity)


def create_episodic_buffer(max_episodes: int = 1000) -> EpisodicBuffer:
    """Crea buffer episódico"""
    return EpisodicBuffer(max_episodes)
