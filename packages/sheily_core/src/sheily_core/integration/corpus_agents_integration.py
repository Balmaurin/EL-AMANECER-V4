#!/usr/bin/env python3
"""
Corpus-Agents Integration System
================================

Sistema unificado que conecta el sistema de corpus/RAG con los agentes MCP,
permitiendo que los agentes accedan y actualicen conocimientos corporativos
en tiempo real.

Características principales:
- Unified knowledge retrieval para agentes
- Dynamic corpus updates desde interacciones de agentes
- Knowledge graph expansion automático
- Cross-agent knowledge sharing
- Context-aware responses using enterprise knowledge

Integra con:
- MemoryCore vector database
- MCP Agent coordination system
- Enterprise knowledge base
- Continuous learning pipeline

Author: MCP Enterprise Integration System
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Corpus-Agents-Integration")


class CorpusAgentsIntegration:
    """
    Sistema unificado de integración entre corpus/RAG y agentes MCP

    Esta clase actúa como puente entre el sistema de conocimientos
    corporativos y los agentes inteligentes, permitiendo:
    - Búsqueda contextualizada de conocimientos
    - Actualización dinámica del corpus
    - Aprendizaje colaborativo entre agentes
    - Knowledge discovery automatizado
    """

    def __init__(self):
        # Estado del sistema de integración
        self.corpus_index = {}  # Index del corpus por dominio
        self.agent_knowledge_graph = {}  # Grafo de conocimientos por agente
        self.cross_agent_learnings = []  # Aprendizajes compartidos
        self.knowledge_updates = []  # Actualizaciones pendientes al corpus

        # Configuración de integración
        self.similarity_threshold = 0.85  # Threshold para matching de conocimientos
        self.max_context_docs = 5  # Documentos máximos de contexto
        self.knowledge_decay_factor = 0.95  # Factor de decaimiento de relevancia

        # Integraciones con subsistemas
        self.memory_core_client = None
        self.agent_coordinator = None
        self.audit_logger = None

        self._initialize_integration()

    def _initialize_integration(self):
        """Inicializar todas las integraciones del sistema"""
        try:
            # Configurar directorios
            Path("data/corpus_agents").mkdir(exist_ok=True)
            Path("data/knowledge_graph").mkdir(exist_ok=True)
            Path("logs/corpus_agents").mkdir(exist_ok=True)

            logger.info("✅ Sistema de integración Corpus-Agents inicializado")

        except Exception as e:
            logger.error(f"Error inicializando integración: {e}")

    async def retrieve_contextual_knowledge(
        self,
        agent_id: str,
        query: str,
        domain: str = "general",
        context_history: List[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Recuperar conocimientos contextuales para un agente

        Args:
            agent_id: ID del agente solicitante
            query: Consulta o pregunta del agente
            domain: Dominio de especialización
            context_history: Historial de conversación (opcional)

        Returns:
            Conocimientos contextuales encontrados
        """
        try:
            logger.info(
                f"🔍 Recuperando conocimientos para {agent_id} - Query: {query[:50]}..."
            )

            # 1. Preparar query enriquecida con contexto
            enriched_query = await self._enrich_query_with_context(
                agent_id, query, domain, context_history
            )

            # 2. Buscar en múltiples fuentes de conocimiento
            knowledge_sources = await self._multi_source_knowledge_retrieval(
                enriched_query, domain
            )

            # 3. Filtrar y rankear resultados
            filtered_knowledge = await self._filter_and_rank_knowledge(
                knowledge_sources, agent_id, query
            )

            # 4. Generar metadatos de relevancia
            relevance_metadata = await self._generate_relevance_metadata(
                filtered_knowledge, query, agent_id
            )

            contextual_knowledge = {
                "query": query,
                "enriched_query": enriched_query,
                "agent_id": agent_id,
                "domain": domain,
                "knowledge_sources": len(knowledge_sources),
                "filtered_results": len(filtered_knowledge),
                "knowledge_chunks": filtered_knowledge,
                "relevance_metadata": relevance_metadata,
                "retrieval_timestamp": datetime.now().isoformat(),
                "confidence_score": self._calculate_knowledge_confidence(
                    filtered_knowledge
                ),
            }

            # 5. Registrar uso de conocimientos
            await self._log_knowledge_usage(agent_id, contextual_knowledge)

            logger.info(
                f"✅ Conocimientos recuperados: {len(filtered_knowledge)} chunks para {agent_id}"
            )
            return contextual_knowledge

        except Exception as e:
            logger.error(f"Error recuperando conocimientos contextuales: {e}")
            return {
                "error": str(e),
                "query": query,
                "agent_id": agent_id,
                "knowledge_chunks": [],
                "retrieval_timestamp": datetime.now().isoformat(),
            }

    async def update_corpus_from_agent_interaction(
        self, agent_id: str, interaction_data: Dict[str, Any]
    ) -> bool:
        """
        Actualizar corpus desde interacciones de agentes

        Args:
            agent_id: ID del agente que generó la interacción
            interaction_data: Datos de la interacción (query, response, etc.)

        Returns:
            True si se actualizó correctamente
        """
        try:
            logger.info(f"📚 Actualizando corpus desde interacción de {agent_id}")

            # 1. Analizar calidad de la interacción
            quality_score = self._assess_interaction_quality(interaction_data)

            if quality_score < 0.7:
                logger.info(
                    f"⚠️ Interacción de baja calidad ({quality_score:.2f}) - no actualizada"
                )
                return False

            # 2. Extraer conocimientos nuevos
            new_knowledge = await self._extract_knowledge_from_interaction(
                agent_id, interaction_data, quality_score
            )

            if not new_knowledge:
                logger.info("ℹ️ No se encontraron conocimientos nuevos")
                return True

            # 3. Integrar en knowledge graph
            await self._integrate_knowledge_in_graph(agent_id, new_knowledge)

            # 4. Queue update to corpus
            corpus_update = {
                "agent_id": agent_id,
                "knowledge": new_knowledge,
                "source": "agent_interaction",
                "quality_score": quality_score,
                "timestamp": datetime.now().isoformat(),
                "status": "pending",
            }

            self.knowledge_updates.append(corpus_update)

            # 5. Trigger batch update si es necesario
            if len(self.knowledge_updates) >= 10:  # Batch size
                await self._execute_corpus_batch_update()

            logger.info(
                f"✅ Corpus actualizado con {len(new_knowledge)} nuevos conocimientos"
            )
            return True

        except Exception as e:
            logger.error(f"Error actualizando corpus: {e}")
            return False

    async def share_knowledge_between_agents(
        self, source_agent: str, target_agent: str, knowledge_topic: str
    ) -> Dict[str, Any]:
        """
        Compartir conocimientos entre agentes

        Args:
            source_agent: Agente que comparte conocimiento
            target_agent: Agente que recibe conocimiento
            knowledge_topic: Tema de conocimiento a compartir

        Returns:
            Resultado del intercambio de conocimientos
        """
        try:
            logger.info(
                f"🤝 Compartiendo conocimiento: {source_agent} → {target_agent} ({knowledge_topic})"
            )

            # 1. Extraer conocimientos relevantes
            relevant_knowledge = await self._extract_agent_knowledge(
                source_agent, knowledge_topic
            )

            if not relevant_knowledge:
                return {"shared": False, "reason": "No relevant knowledge found"}

            # 2. Adaptar conocimiento para agente destino
            adapted_knowledge = await self._adapt_knowledge_for_agent(
                relevant_knowledge, target_agent
            )

            # 3. Transferir conocimiento
            transfer_result = await self._transfer_knowledge_to_agent(
                source_agent, target_agent, adapted_knowledge
            )

            # 4. Registrar cross-agent learning
            cross_learning = {
                "source_agent": source_agent,
                "target_agent": target_agent,
                "topic": knowledge_topic,
                "knowledge_chunks": len(relevant_knowledge),
                "adapted_chunks": len(adapted_knowledge),
                "transfer_success": transfer_result["success"],
                "timestamp": datetime.now().isoformat(),
            }

            self.cross_agent_learnings.append(cross_learning)

            logger.info(
                f"✅ Intercambio completado - {len(adapted_knowledge)} chunks compartidos"
            )
            return {
                "shared": True,
                "source_agent": source_agent,
                "target_agent": target_agent,
                "knowledge_chunks_shared": len(adapted_knowledge),
                "transfer_details": transfer_result,
                "cross_learning_id": len(self.cross_agent_learnings) - 1,
            }

        except Exception as e:
            logger.error(f"Error compartiendo conocimientos: {e}")
            return {"shared": False, "error": str(e)}

    async def discover_new_knowledge_patterns(self) -> List[Dict[str, Any]]:
        """
        Descubrir nuevos patrones de conocimiento desde interacciones

        Returns:
            Lista de patrones de conocimiento descubiertos
        """
        try:
            logger.info("🔍 Descubriendo nuevos patrones de conocimiento...")

            # 1. Analizar interacciones recientes
            recent_interactions = await self._get_recent_interactions()

            # 2. Identificar patrones emergentes
            emerging_patterns = await self._identify_emerging_patterns(
                recent_interactions
            )

            # 3. Validar patrones
            validated_patterns = await self._validate_knowledge_patterns(
                emerging_patterns
            )

            # 4. Integrar patrones validados en corpus
            for pattern in validated_patterns:
                await self._integrate_pattern_in_corpus(pattern)

            logger.info(
                f"✅ Descubiertos {len(validated_patterns)} nuevos patrones de conocimiento"
            )
            return validated_patterns

        except Exception as e:
            logger.error(f"Error descubriendo patrones: {e}")
            return []

    def get_integration_status(self) -> Dict[str, Any]:
        """Obtener estado completo del sistema de integración"""
        return {
            "corpus_domains": list(self.corpus_index.keys()),
            "registered_agents": list(self.agent_knowledge_graph.keys()),
            "pending_updates": len(self.knowledge_updates),
            "cross_agent_learnings": len(self.cross_agent_learnings),
            "total_knowledge_chunks": sum(
                len(chunks) for chunks in self.agent_knowledge_graph.values()
            ),
            "integration_health": "healthy",  # Implementar health checks
            "last_update": datetime.now().isoformat(),
        }

    # ========== MÉTODOS PRIVADOS ==========

    async def _enrich_query_with_context(
        self, agent_id: str, query: str, domain: str, context_history: List[Dict] = None
    ) -> str:
        """Enriquecer query con contexto del agente"""
        enriched_parts = [query]

        # Agregar especialización del agente
        if agent_id in self.agent_knowledge_graph:
            agent_expertise = list(self.agent_knowledge_graph[agent_id].keys())[
                :3
            ]  # Top 3 topics
            if agent_expertise:
                enriched_parts.append(f"Agent expertise: {', '.join(agent_expertise)}")

        # Agregar dominio
        if domain != "general":
            enriched_parts.append(f"Domain: {domain}")

        # Agregar contexto histórico relevante
        if context_history:
            recent_context = context_history[-3:]  # Últimas 3 interacciones
            context_summary = self._summarize_context_history(recent_context)
            if context_summary:
                enriched_parts.append(f"Recent context: {context_summary}")

        return " | ".join(enriched_parts)

    async def _multi_source_knowledge_retrieval(
        self, enriched_query: str, domain: str
    ) -> List[Dict[str, Any]]:
        """Recuperar conocimientos desde múltiples fuentes"""
        knowledge_sources = []

        # 1. Buscar en corpus principal por dominio
        if domain in self.corpus_index:
            domain_knowledge = self._search_domain_corpus(enriched_query, domain)
            knowledge_sources.extend(domain_knowledge)

        # 2. Buscar en conocimientos compartidos entre agentes
        shared_knowledge = self._search_shared_knowledge(enriched_query, domain)
        knowledge_sources.extend(shared_knowledge)

        # 3. Buscar en patrones descubiertos
        pattern_knowledge = self._search_knowledge_patterns(enriched_query, domain)
        knowledge_sources.extend(pattern_knowledge)

        return knowledge_sources

    async def _filter_and_rank_knowledge(
        self, knowledge_sources: List[Dict], agent_id: str, original_query: str
    ) -> List[Dict[str, Any]]:
        """Filtrar y rankear conocimientos por relevancia"""
        filtered_chunks = []

        for chunk in knowledge_sources:
            # Calcular similitud con query
            similarity = self._calculate_query_similarity(chunk, original_query)

            if similarity >= self.similarity_threshold:
                # Verificar si el agente ya conoce este conocimiento
                familiarity_score = self._assess_agent_familiarity(agent_id, chunk)

                # Calcular score compuesto
                relevance_score = (similarity * 0.7) + ((1 - familiarity_score) * 0.3)

                if relevance_score >= 0.6:  # Threshold adicional
                    chunk_with_score = chunk.copy()
                    chunk_with_score.update(
                        {
                            "relevance_score": relevance_score,
                            "similarity_score": similarity,
                            "familiarity_score": familiarity_score,
                            "ranking_timestamp": datetime.now().isoformat(),
                        }
                    )
                    filtered_chunks.append(chunk_with_score)

        # Rankear por score de relevancia
        filtered_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Limitar a top N
        return filtered_chunks[: self.max_context_docs]

    async def _generate_relevance_metadata(
        self, knowledge_chunks: List[Dict], original_query: str, agent_id: str
    ) -> Dict[str, Any]:
        """Generar metadatos de relevancia para los conocimientos"""
        if not knowledge_chunks:
            return {
                "confidence_level": 0.0,
                "diversity_score": 0.0,
                "coverage_score": 0.0,
            }

        relevance_scores = [chunk["relevance_score"] for chunk in knowledge_chunks]
        similarity_scores = [chunk["similarity_score"] for chunk in knowledge_chunks]

        # Calcular métricas de calidad
        avg_relevance = np.mean(relevance_scores)
        avg_similarity = np.mean(similarity_scores)

        # Diversidad de fuentes
        sources = list(
            set(chunk.get("source", "unknown") for chunk in knowledge_chunks)
        )
        diversity_score = len(sources) / max(1, len(knowledge_chunks))

        # Coverage score (qué tanto cubre la query)
        coverage_score = min(avg_similarity * avg_relevance * diversity_score, 1.0)

        return {
            "average_relevance": float(avg_relevance),
            "average_similarity": float(avg_similarity),
            "diversity_score": float(diversity_score),
            "coverage_score": float(coverage_score),
            "confidence_level": float(avg_relevance * 0.8 + coverage_score * 0.2),
            "top_sources": sources[:3],
            "knowledge_chunks": len(knowledge_chunks),
        }

    def _calculate_knowledge_confidence(self, knowledge_chunks: List[Dict]) -> float:
        """Calcular confianza general en los conocimientos"""
        if not knowledge_chunks:
            return 0.0

        scores = [chunk.get("relevance_score", 0) for chunk in knowledge_chunks]
        avg_score = np.mean(scores)
        consistency = 1 - np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0

        return float(avg_score * 0.7 + consistency * 0.3)

    async def _log_knowledge_usage(
        self, agent_id: str, knowledge_result: Dict[str, Any]
    ):
        """Registrar uso de conocimientos para analytics"""
        try:
            usage_log = {
                "agent_id": agent_id,
                "query": knowledge_result["query"],
                "knowledge_chunks_used": len(knowledge_result["knowledge_chunks"]),
                "confidence_score": knowledge_result.get("confidence_score", 0),
                "domain": knowledge_result.get("domain", "unknown"),
                "timestamp": datetime.now().isoformat(),
            }

            # Aquí se implementaría persistencia real
            logger.debug(f"📊 Knowledge usage logged for {agent_id}")

        except Exception as e:
            logger.debug(f"Error logging knowledge usage: {e}")

    def _assess_interaction_quality(self, interaction_data: Dict) -> float:
        """Evaluar calidad de una interacción para extracción de conocimientos"""
        quality_factors = {
            "response_length": 0.2,
            "technical_terms": 0.3,
            "specificity": 0.2,
            "coherence": 0.15,
            "value_added": 0.15,
        }

        response = interaction_data.get(
            "response", interaction_data.get("final_response", "")
        )

        # Factor 1: Longitud apropiada
        length_score = min(len(response) / 500, 1.0) if len(response) >= 50 else 0

        # Factor 2: Contenido técnico
        tech_keywords = [
            "analysis",
            "system",
            "process",
            "method",
            "framework",
            "architecture",
        ]
        tech_count = sum(
            1 for keyword in tech_keywords if keyword.lower() in response.lower()
        )
        tech_score = min(tech_count / 5, 1.0)

        # Factor 3: Especificidad
        specific_indicators = [
            "specifically",
            "particularly",
            "according to",
            "based on",
            "as per",
        ]
        specific_score = min(
            sum(1 for ind in specific_indicators if ind in response.lower()) / 3, 1.0
        )

        # Factor 4: Coherencia (simplificada)
        coherence_score = 0.8 if len(response.split(".")) >= 3 else 0.5

        # Factor 5: Valor agregado
        value_indicators = [
            "recommendation",
            "consider",
            "suggest",
            "improve",
            "optimize",
        ]
        value_score = min(
            sum(1 for ind in value_indicators if ind in response.lower()) / 3, 1.0
        )

        total_score = (
            length_score * quality_factors["response_length"]
            + tech_score * quality_factors["technical_terms"]
            + specific_score * quality_factors["specificity"]
            + coherence_score * quality_factors["coherence"]
            + value_score * quality_factors["value_added"]
        )

        return min(total_score, 1.0)

    async def _extract_knowledge_from_interaction(
        self, agent_id: str, interaction_data: Dict, quality_score: float
    ) -> List[Dict[str, Any]]:
        """Extraer conocimientos desde una interacción"""
        query = interaction_data.get("query", interaction_data.get("prompt", ""))
        response = interaction_data.get(
            "response", interaction_data.get("final_response", "")
        )

        if not query or not response or len(response) < 100:
            return []

        # Dividir respuesta en chunks significativos
        knowledge_chunks = self._split_response_into_chunks(response)

        extracted_knowledge = []

        for i, chunk in enumerate(knowledge_chunks):
            if len(chunk) >= 50:  # Mínimo tamaño de chunk
                knowledge_id = hashlib.md5(f"{query}_{chunk}_{i}".encode()).hexdigest()[
                    :16
                ]

                knowledge_chunk = {
                    "id": knowledge_id,
                    "content": chunk,
                    "source_query": query,
                    "agent_id": agent_id,
                    "extracted_from": "agent_interaction",
                    "quality_score": quality_score,
                    "chunk_index": i,
                    "total_chunks": len(knowledge_chunks),
                    "embedding_ready": True,  # Flag para vectorization
                    "extraction_timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "interaction_type": interaction_data.get(
                            "task_type", "unknown"
                        ),
                        "response_length": len(response),
                        "technical_density": self._calculate_technical_density(chunk),
                    },
                }

                extracted_knowledge.append(knowledge_chunk)

        return extracted_knowledge

    async def _integrate_knowledge_in_graph(self, agent_id: str, knowledge: List[Dict]):
        """Integrar conocimientos en el knowledge graph del agente"""
        if agent_id not in self.agent_knowledge_graph:
            self.agent_knowledge_graph[agent_id] = {}

        # Categorizar conocimiento por temas
        for chunk in knowledge:
            topics = self._identify_knowledge_topics(chunk["content"])

            for topic in topics:
                if topic not in self.agent_knowledge_graph[agent_id]:
                    self.agent_knowledge_graph[agent_id][topic] = []

                # Evitar duplicados con similitud
                is_duplicate = False
                for existing in self.agent_knowledge_graph[agent_id][topic]:
                    if (
                        self._calculate_text_similarity(
                            chunk["content"], existing["content"]
                        )
                        > 0.9
                    ):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    self.agent_knowledge_graph[agent_id][topic].append(chunk)
                    # Decay old knowledge
                    self._apply_knowledge_decay(
                        self.agent_knowledge_graph[agent_id][topic]
                    )

    async def _execute_corpus_batch_update(self):
        """Ejecutar actualización batch del corpus principal"""
        try:
            logger.info(
                f"🔄 Ejecutando batch update del corpus ({len(self.knowledge_updates)} actualizaciones)"
            )

            # Preparar datos para actualización del corpus
            corpus_update_data = {
                "updates": self.knowledge_updates,
                "batch_id": f"batch_{int(datetime.now().timestamp())}",
                "total_updates": len(self.knowledge_updates),
                "timestamp": datetime.now().isoformat(),
            }

            # Aquí se integraría con MemoryCore para actualización del vector store
            # Por ahora solo logging

            # Limpiar updates procesados
            self.knowledge_updates = []

            logger.info("✅ Corpus batch update completado")

        except Exception as e:
            logger.error(f"Error en corpus batch update: {e}")

    # ========== MÉTODOS HELPER ==========

    def _search_domain_corpus(self, query: str, domain: str) -> List[Dict[str, Any]]:
        """Buscar en corpus específico del dominio"""
        # Implementación simplificada - en producción buscaría en vector DB
        if domain not in self.corpus_index:
            return []

        domain_docs = self.corpus_index[domain]
        matching_docs = []

        for doc in domain_docs:
            similarity = self._calculate_query_similarity(doc, query)
            if similarity > 0.7:  # Threshold simplificado
                doc_copy = doc.copy()
                doc_copy["similarity"] = similarity
                matching_docs.append(doc_copy)

        return matching_docs[: self.max_context_docs]

    def _search_shared_knowledge(self, query: str, domain: str) -> List[Dict[str, Any]]:
        """Buscar en conocimientos compartidos entre agentes"""
        shared_knowledge = []

        for agent_knowledge in self.agent_knowledge_graph.values():
            for topic_docs in agent_knowledge.values():
                for doc in topic_docs:
                    if doc.get("is_shared", False):
                        similarity = self._calculate_query_similarity(doc, query)
                        if similarity > 0.75:
                            doc_copy = doc.copy()
                            doc_copy["similarity"] = similarity
                            doc_copy["source"] = "shared_agent_knowledge"
                            shared_knowledge.append(doc_copy)

        return shared_knowledge[:3]  # Limitar shared knowledge

    def _search_knowledge_patterns(
        self, query: str, domain: str
    ) -> List[Dict[str, Any]]:
        """Buscar en patrones de conocimiento descubiertos"""
        pattern_knowledge = []

        # Aquí se implementaría búsqueda en patrones descubiertos
        # Por simplicidad, retornamos lista vacía
        return pattern_knowledge

    def _calculate_query_similarity(self, doc: Dict, query: str) -> float:
        """Calcular similitud entre documento y query"""
        # Implementación simplificada - en producción usar embeddings
        doc_text = doc.get("content", doc.get("text", "")).lower()
        query_lower = query.lower()

        # Simple word overlap
        query_words = set(query_lower.split())
        doc_words = set(doc_text.split())

        intersection = len(query_words.intersection(doc_words))
        union = len(query_words.union(doc_words))

        return intersection / union if union > 0 else 0

    def _assess_agent_familiarity(self, agent_id: str, knowledge_chunk: Dict) -> float:
        """Evaluar que tan familiar es un conocimiento para el agente"""
        if agent_id not in self.agent_knowledge_graph:
            return 0.0  # No familiar

        # Verificar similitud con conocimientos existentes
        max_similarity = 0.0

        for topic_docs in self.agent_knowledge_graph[agent_id].values():
            for existing_doc in topic_docs:
                similarity = self._calculate_text_similarity(
                    knowledge_chunk.get("content", ""), existing_doc.get("content", "")
                )
                max_similarity = max(max_similarity, similarity)

        return min(max_similarity, 1.0)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud entre dos textos"""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0

    def _summarize_context_history(self, context_history: List[Dict]) -> str:
        """Resumir historial de contexto"""
        if not context_history:
            return ""

        summaries = []
        for ctx in context_history[-2:]:  # Últimas 2 interacciones
            query = ctx.get("query", "")[:30]
            if query:
                summaries.append(query)

        return " → ".join(summaries)

    def _split_response_into_chunks(self, response: str) -> List[str]:
        """Dividir respuesta en chunks significativos"""
        # Dividir por oraciones y combinar en chunks
        sentences = response.split(".")

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip() + "."
            if len(current_chunk + sentence) <= 400:  # Tamaño máximo de chunk
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _calculate_technical_density(self, text: str) -> float:
        """Calcular densidad técnica de un texto"""
        technical_terms = [
            "system",
            "process",
            "framework",
            "architecture",
            "implementation",
            "optimization",
            "analysis",
            "evaluation",
            "integration",
            "automation",
        ]

        words = text.lower().split()
        technical_count = sum(
            1 for word in words if any(term in word for term in technical_terms)
        )

        return technical_count / len(words) if words else 0

    def _identify_knowledge_topics(self, content: str) -> List[str]:
        """Identificar temas/topics en contenido"""
        topics = []

        topic_keywords = {
            "finance": ["financial", "risk", "portfolio", "investment", "compliance"],
            "technical": [
                "system",
                "architecture",
                "code",
                "implementation",
                "debugging",
            ],
            "business": ["strategy", "market", "customer", "revenue", "growth"],
            "operations": [
                "process",
                "workflow",
                "automation",
                "monitoring",
                "performance",
            ],
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in content.lower() for keyword in keywords):
                topics.append(topic)

        return topics if topics else ["general"]

    def _apply_knowledge_decay(self, knowledge_list: List[Dict]):
        """Aplicar factor de decaimiento a conocimientos antiguos"""
        current_time = datetime.now()

        for knowledge in knowledge_list:
            if "extraction_timestamp" in knowledge:
                timestamp = datetime.fromisoformat(knowledge["extraction_timestamp"])
                age_days = (current_time - timestamp).days

                # Aplicar decaimiento exponencial
                decay_factor = self.knowledge_decay_factor ** (
                    age_days / 30
                )  # Decay monthly

                if "quality_score" in knowledge:
                    knowledge["quality_score"] *= decay_factor

        # Remover conocimientos muy antiguos/decayed
        knowledge_list[:] = [
            k for k in knowledge_list if k.get("quality_score", 1.0) > 0.1
        ]

    # Placeholder methods para completar la interfaz
    async def _extract_agent_knowledge(self, agent_id: str, topic: str) -> List[Dict]:
        """Extraer conocimientos de un agente sobre un tema"""
        return []

    async def _adapt_knowledge_for_agent(
        self, knowledge: List[Dict], target_agent: str
    ) -> List[Dict]:
        """Adaptar conocimientos para un agente destino"""
        return knowledge

    async def _transfer_knowledge_to_agent(
        self, source_agent: str, target_agent: str, knowledge: List[Dict]
    ) -> Dict[str, Any]:
        """Transferir conocimientos a agente destino"""
        return {"success": True, "transferred_chunks": len(knowledge)}

    async def _get_recent_interactions(self) -> List[Dict]:
        """Obtener interacciones recientes"""
        return []

    async def _identify_emerging_patterns(self, interactions: List[Dict]) -> List[Dict]:
        """Identificar patrones emergentes"""
        return []

    async def _validate_knowledge_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Validar patrones de conocimiento"""
        return patterns

    async def _integrate_pattern_in_corpus(self, pattern: Dict):
        """Integrar patrón en corpus"""
        pass


# Backwards compatibility alias (singular name used elsewhere)
CorpusAgentIntegration = CorpusAgentsIntegration

# ========== DEMO E INTEGRACIÓN ==========


async def demo_corpus_agents_integration():
    """Demostración del sistema de integración Corpus-Agents"""
    print("🔗 Demo: Corpus-Agents Integration System")
    print("=" * 50)

    try:
        # Inicializar sistema
        print("🚀 Inicializando Corpus-Agents Integration...")
        integration_system = CorpusAgentsIntegration()

        # Demo 1: Recuperación de conocimientos contextuales
        print("\n🎯 Demo 1: Contextual Knowledge Retrieval")
        knowledge_result = await integration_system.retrieve_contextual_knowledge(
            agent_id="finance_agent",
            query="¿Cómo mejorar la evaluación de riesgo de portafolio?",
            domain="finance",
        )

        print(
            f"✅ Conocimientos encontrados: {knowledge_result.get('filtered_results', 0)}"
        )
        print(f"🎯 Confidence Score: {knowledge_result.get('confidence_score', 0):.2f}")

        # Demo 2: Actualización desde interacción
        print("\n📚 Demo 2: Corpus Update from Agent Interaction")
        sample_interaction = {
            "query": "Optimizar portafolio de inversiones",
            "response": "Para optimizar un portafolio se recomienda usar teoría moderna de portafolios. Diversificar activos, minimizar volatilidad, maximizar ratio Sharpe. Considerar correlaciones entre activos.",
            "agent_id": "finance_agent",
            "task_type": "portfolio_optimization",
        }

        update_success = await integration_system.update_corpus_from_agent_interaction(
            "finance_agent", sample_interaction
        )
        print(f"✅ Corpus Update: {'Successful' if update_success else 'Failed'}")

        # Demo 3: Compartir conocimientos entre agentes
        print("\n🤝 Demo 3: Cross-Agent Knowledge Sharing")
        share_result = await integration_system.share_knowledge_between_agents(
            source_agent="finance_agent",
            target_agent="technical_agent",
            knowledge_topic="finance",
        )

        print(f"✅ Knowledge Sharing: {share_result.get('shared', False)}")
        if share_result.get("shared", False):
            print(
                f"📦 Chunks compartidos: {share_result.get('knowledge_chunks_shared', 0)}"
            )

        # Status final
        status = integration_system.get_integration_status()
        print("\n📊 Final Status:")
        print(f"   • Agent Domains: {len(status['corpus_domains'])}")
        print(f"   • Registered Agents: {len(status['registered_agents'])}")
        print(f"   • Pending Updates: {status['pending_updates']}")

        print("\n✅ Corpus-Agents Integration demo completada!")

    except Exception as e:
        print(f"❌ Error en demo: {e}")


if __name__ == "__main__":
    print("Corpus-Agents Integration System")
    print("=" * 40)
    asyncio.run(demo_corpus_agents_integration())
