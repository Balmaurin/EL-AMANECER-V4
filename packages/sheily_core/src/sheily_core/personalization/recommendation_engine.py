#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Recomendaciones Personalizadas - Sheily AI
====================================================

Motor inteligente de recomendaciones basado en:
- Historial de interacciones del usuario
- Análisis de patrones de comportamiento
- Aprendizaje continuo de preferencias
- Recomendaciones contextuales en tiempo real
- Sistema de retroalimentación para mejorar precisión
"""

import asyncio
import heapq
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from sheily_core.cache.smart_cache import cached, get_cache
from sheily_core.sentiment.sentiment_analysis import get_sentiment_api

logger = logging.getLogger(__name__)


@dataclass
class UserInteraction:
    """Interacción del usuario"""

    user_id: str
    timestamp: datetime
    interaction_type: str  # 'query', 'response', 'feedback', 'click'
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None


@dataclass
class UserProfile:
    """Perfil del usuario para recomendaciones"""

    user_id: str
    preferences: Dict[str, float] = field(default_factory=dict)  # tópico -> score
    interaction_history: List[UserInteraction] = field(default_factory=list)
    favorite_topics: List[str] = field(default_factory=list)
    avoided_topics: List[str] = field(default_factory=list)
    avg_sentiment: float = 0.0
    total_interactions: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    def update_preferences(self, topic: str, score: float):
        """Actualizar preferencias por tópico"""
        if topic not in self.preferences:
            self.preferences[topic] = score
        else:
            # Media ponderada
            self.preferences[topic] = (self.preferences[topic] * 0.7) + (score * 0.3)

        self.last_updated = datetime.now()

    def get_top_topics(self, limit: int = 5) -> List[Tuple[str, float]]:
        """Obtener tópicos más preferidos"""
        return sorted(self.preferences.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]


@dataclass
class Recommendation:
    """Recomendación personalizada"""

    id: str
    user_id: str
    content: str
    reason: str
    confidence: float
    category: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PersonalizationEngine:
    """Motor de recomendaciones personalizadas"""

    def __init__(self):
        self.cache = get_cache()
        self.sentiment_api = get_sentiment_api()
        self.user_profiles: Dict[str, UserProfile] = {}
        self.topic_keywords: Dict[str, Set[str]] = self._load_topic_keywords()
        self.recommendation_history: Dict[str, List[Recommendation]] = defaultdict(list)

        # Pesos para scoring de recomendaciones
        self.weights = {
            "topic_relevance": 0.4,
            "sentiment_history": 0.3,
            "temporal_recency": 0.2,
            "diversity": 0.1,
        }

    def _load_topic_keywords(self) -> Dict[str, Set[str]]:
        """Cargar palabras clave por tópico"""
        return {
            "tecnologia": {
                "programacion",
                "codigo",
                "desarrollo",
                "software",
                "hardware",
                "inteligencia artificial",
                "machine learning",
                "deep learning",
                "python",
                "javascript",
                "web",
                "mobile",
                "cloud",
                "devops",
            },
            "ciencia": {
                "fisica",
                "quimica",
                "biologia",
                "matematicas",
                "astronomia",
                "investigacion",
                "experimento",
                "teoria",
                "hipotesis",
            },
            "negocios": {
                "empresa",
                "marketing",
                "ventas",
                "finanzas",
                "estrategia",
                "liderazgo",
                "emprendimiento",
                "startup",
                "cliente",
                "producto",
            },
            "salud": {
                "medicina",
                "bienestar",
                "nutricion",
                "ejercicio",
                "mental",
                "enfermedad",
                "prevencion",
                "diagnostico",
                "tratamiento",
            },
            "educacion": {
                "aprendizaje",
                "estudio",
                "universidad",
                "curso",
                "certificacion",
                "habilidad",
                "conocimiento",
                "ensenanza",
                "formacion",
            },
            "entretenimiento": {
                "musica",
                "pelicula",
                "juego",
                "deporte",
                "arte",
                "cultura",
                "ocio",
                "diversion",
                "cine",
                "television",
                "libro",
            },
        }

    def _extract_topics(self, text: str) -> List[Tuple[str, float]]:
        """Extraer tópicos de un texto con scores de relevancia"""
        text_lower = text.lower()
        topics_scores = {}

        for topic, keywords in self.topic_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                # Score basado en cantidad de matches y longitud del texto
                score = min(matches / len(keywords.split()), 1.0)
                topics_scores[topic] = score

        return sorted(topics_scores.items(), key=lambda x: x[1], reverse=True)

    async def record_interaction(self, interaction: UserInteraction):
        """Registrar una interacción del usuario"""
        # Obtener perfil del usuario
        if interaction.user_id not in self.user_profiles:
            self.user_profiles[interaction.user_id] = UserProfile(
                user_id=interaction.user_id
            )

        profile = self.user_profiles[interaction.user_id]

        # Analizar sentimiento si no está presente
        if interaction.sentiment_score is None:
            try:
                sentiment_result = await self.sentiment_api.analyze_sentiment(
                    interaction.content
                )
                interaction.sentiment_score = (
                    sentiment_result.confidence
                    if sentiment_result.label.value == "positive"
                    else -sentiment_result.confidence
                )
            except Exception as e:
                logger.error(f"Error analizando sentimiento: {e}")
                interaction.sentiment_score = 0.0

        # Extraer tópicos y actualizar preferencias
        topics = self._extract_topics(interaction.content)
        for topic, score in topics:
            # Modificar score basado en sentimiento
            adjusted_score = score * (1 + interaction.sentiment_score)
            profile.update_preferences(topic, adjusted_score)

        # Agregar a historial
        profile.interaction_history.append(interaction)
        profile.total_interactions += 1

        # Mantener solo últimas 1000 interacciones
        if len(profile.interaction_history) > 1000:
            profile.interaction_history = profile.interaction_history[-1000:]

        # Actualizar estadísticas
        sentiments = [
            i.sentiment_score
            for i in profile.interaction_history
            if i.sentiment_score is not None
        ]
        profile.avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        # Cachear perfil actualizado
        await self.cache.set(f"user_profile:{interaction.user_id}", profile, ttl=3600)

        logger.debug(f"Interacción registrada para usuario {interaction.user_id}")

    @cached("user_recommendations", ttl=1800)
    async def generate_recommendations(
        self, user_id: str, context: Optional[str] = None, limit: int = 5
    ) -> List[Recommendation]:
        """Generar recomendaciones personalizadas para un usuario"""
        if user_id not in self.user_profiles:
            # Usuario nuevo - recomendaciones genéricas
            return await self._generate_default_recommendations(user_id, limit)

        profile = self.user_profiles[user_id]

        if not profile.interaction_history:
            return await self._generate_default_recommendations(user_id, limit)

        recommendations = []

        # 1. Recomendaciones basadas en tópicos preferidos
        top_topics = profile.get_top_topics(3)
        for topic, score in top_topics:
            recs = await self._generate_topic_recommendations(
                user_id, topic, score, context
            )
            recommendations.extend(recs)

        # 2. Recomendaciones basadas en patrones temporales
        temporal_recs = await self._generate_temporal_recommendations(user_id, profile)
        recommendations.extend(temporal_recs)

        # 3. Recomendaciones basadas en similitud con otros usuarios
        similar_recs = await self._generate_collaborative_recommendations(
            user_id, profile
        )
        recommendations.extend(similar_recs)

        # 4. Recomendaciones de diversidad
        diversity_recs = await self._generate_diversity_recommendations(
            user_id, profile
        )
        recommendations.extend(diversity_recs)

        # Filtrar y rankear recomendaciones
        filtered_recs = await self._filter_and_rank_recommendations(
            recommendations, profile, limit
        )

        # Registrar recomendaciones generadas
        self.recommendation_history[user_id].extend(filtered_recs)

        return filtered_recs

    async def _generate_topic_recommendations(
        self, user_id: str, topic: str, topic_score: float, context: Optional[str]
    ) -> List[Recommendation]:
        """Generar recomendaciones basadas en tópicos"""
        recommendations = []

        # Recomendaciones específicas por tópico
        topic_templates = {
            "tecnologia": [
                "Aprende sobre las últimas tendencias en inteligencia artificial",
                "¿Te gustaría explorar frameworks de desarrollo web modernos?",
                "Descubre mejores prácticas en desarrollo de software",
                "Explora tecnologías emergentes como blockchain o IoT",
            ],
            "ciencia": [
                "Mantente al día con los últimos descubrimientos científicos",
                "¿Quieres aprender sobre avances en biotecnología?",
                "Explora conceptos fascinantes de física cuántica",
                "Descubre investigaciones revolucionarias en medicina",
            ],
            "negocios": [
                "Desarrolla estrategias efectivas de marketing digital",
                "¿Te interesa aprender sobre emprendimiento innovador?",
                "Explora técnicas avanzadas de gestión empresarial",
                "Descubre tendencias en transformación digital",
            ],
            "salud": [
                "Aprende sobre nutrición y hábitos saludables",
                "¿Quieres conocer técnicas de mindfulness y bienestar?",
                "Explora avances en medicina preventiva",
                "Descubre ejercicios y rutinas para mantenerte en forma",
            ],
            "educacion": [
                "Amplía tus conocimientos con cursos especializados",
                "¿Te gustaría obtener nuevas certificaciones profesionales?",
                "Explora metodologías innovadoras de aprendizaje",
                "Descubre recursos para desarrollo personal",
            ],
        }

        if topic in topic_templates:
            templates = topic_templates[topic][:2]  # Máximo 2 por tópico
            for i, template in enumerate(templates):
                rec = Recommendation(
                    id=f"{user_id}_{topic}_topic_{i}_{int(datetime.now().timestamp())}",
                    user_id=user_id,
                    content=template,
                    reason=f"Basado en tu interés en {topic}",
                    confidence=min(topic_score, 0.9),
                    category="topic_based",
                    metadata={"topic": topic, "topic_score": topic_score},
                )
                recommendations.append(rec)

        return recommendations

    async def _generate_temporal_recommendations(
        self, user_id: str, profile: UserProfile
    ) -> List[Recommendation]:
        """Generar recomendaciones basadas en patrones temporales"""
        recommendations = []

        # Analizar interacciones recientes (últimas 24 horas)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_interactions = [
            i for i in profile.interaction_history if i.timestamp >= recent_cutoff
        ]

        if recent_interactions:
            # Encontrar tópicos más activos recientemente
            recent_topics = Counter()
            for interaction in recent_interactions:
                topics = self._extract_topics(interaction.content)
                for topic, score in topics:
                    recent_topics[topic] += score

            if recent_topics:
                top_recent_topic = recent_topics.most_common(1)[0][0]
                rec = Recommendation(
                    id=f"{user_id}_temporal_{int(datetime.now().timestamp())}",
                    user_id=user_id,
                    content=f"¿Quieres profundizar más en {top_recent_topic} basado en tus consultas recientes?",
                    reason="Patrón de actividad reciente detectado",
                    confidence=0.7,
                    category="temporal",
                    metadata={"recent_topic": top_recent_topic},
                )
                recommendations.append(rec)

        return recommendations

    async def _generate_collaborative_recommendations(
        self, user_id: str, profile: UserProfile
    ) -> List[Recommendation]:
        """Generar recomendaciones colaborativas (simplificadas)"""
        recommendations = []

        # En una implementación completa, esto buscaría usuarios similares
        # Por simplicidad, generamos recomendaciones basadas en tópicos populares globales
        global_popular_topics = ["tecnologia", "ciencia", "negocios"]
        user_topics = set(profile.preferences.keys())

        for topic in global_popular_topics:
            if topic not in user_topics:
                rec = Recommendation(
                    id=f"{user_id}_collaborative_{topic}_{int(datetime.now().timestamp())}",
                    user_id=user_id,
                    content=f"Otros usuarios interesados en tus tópicos también exploran {topic}",
                    reason="Recomendación colaborativa",
                    confidence=0.5,
                    category="collaborative",
                    metadata={"topic": topic},
                )
                recommendations.append(rec)
                break  # Solo una recomendación colaborativa

        return recommendations

    async def _generate_diversity_recommendations(
        self, user_id: str, profile: UserProfile
    ) -> List[Recommendation]:
        """Generar recomendaciones para aumentar diversidad"""
        recommendations = []

        # Encontrar tópicos poco explorados
        all_topics = set(self.topic_keywords.keys())
        explored_topics = set(profile.preferences.keys())

        unexplored_topics = list(all_topics - explored_topics)
        if unexplored_topics:
            topic = unexplored_topics[0]  # Tomar el primero
            rec = Recommendation(
                id=f"{user_id}_diversity_{topic}_{int(datetime.now().timestamp())}",
                user_id=user_id,
                content=f"¿Por qué no explorar algo nuevo? Descubre {topic}",
                reason="Para diversificar tus intereses",
                confidence=0.4,
                category="diversity",
                metadata={"topic": topic},
            )
            recommendations.append(rec)

        return recommendations

    async def _generate_default_recommendations(
        self, user_id: str, limit: int
    ) -> List[Recommendation]:
        """Generar recomendaciones por defecto para usuarios nuevos"""
        default_recs = [
            Recommendation(
                id=f"{user_id}_default_{i}_{int(datetime.now().timestamp())}",
                user_id=user_id,
                content=content,
                reason="Recomendación inicial para nuevos usuarios",
                confidence=0.5,
                category="default",
                metadata={"new_user": True},
            )
            for i, content in enumerate(
                [
                    "¿Te gustaría aprender sobre inteligencia artificial?",
                    "Explora consejos útiles para el desarrollo personal",
                    "¿Quieres conocer las últimas tendencias tecnológicas?",
                    "Descubre estrategias efectivas para el bienestar",
                    "Aprende sobre emprendimiento e innovación",
                ]
            )
        ]

        return default_recs[:limit]

    async def _filter_and_rank_recommendations(
        self, recommendations: List[Recommendation], profile: UserProfile, limit: int
    ) -> List[Recommendation]:
        """Filtrar y rankear recomendaciones"""

        # Eliminar duplicados y recomendaciones ya vistas recientemente
        recent_recs = self.recommendation_history[profile.user_id][-20:]  # Últimas 20
        recent_content = {rec.content for rec in recent_recs}

        filtered = []
        for rec in recommendations:
            if rec.content not in recent_content:
                # Ajustar confianza basada en historial del usuario
                if profile.avg_sentiment > 0.1:
                    rec.confidence *= 1.1  # Boost para usuarios positivos
                elif profile.avg_sentiment < -0.1:
                    rec.confidence *= 0.9  # Reduce para usuarios negativos

                filtered.append(rec)

        # Rankear por confianza
        filtered.sort(key=lambda x: x.confidence, reverse=True)

        return filtered[:limit]

    async def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Obtener insights sobre el usuario para análisis"""
        if user_id not in self.user_profiles:
            return {"error": "Usuario no encontrado"}

        profile = self.user_profiles[user_id]

        return {
            "user_id": user_id,
            "total_interactions": profile.total_interactions,
            "avg_sentiment": profile.avg_sentiment,
            "top_topics": profile.get_top_topics(5),
            "interaction_trends": await self._analyze_interaction_trends(profile),
            "recommendation_stats": {
                "total_generated": len(self.recommendation_history[user_id]),
                "categories": Counter(
                    rec.category for rec in self.recommendation_history[user_id]
                ),
            },
        }

    async def _analyze_interaction_trends(self, profile: UserProfile) -> Dict[str, Any]:
        """Analizar tendencias de interacciones"""
        if not profile.interaction_history:
            return {}

        # Agrupar por día
        daily_interactions = defaultdict(int)
        daily_sentiment = defaultdict(list)

        for interaction in profile.interaction_history[-100:]:  # Últimas 100
            day = interaction.timestamp.date()
            daily_interactions[day] += 1
            if interaction.sentiment_score is not None:
                daily_sentiment[day].append(interaction.sentiment_score)

        # Calcular promedios
        avg_daily_interactions = (
            sum(daily_interactions.values()) / len(daily_interactions)
            if daily_interactions
            else 0
        )
        avg_daily_sentiment = {
            day: sum(scores) / len(scores) for day, scores in daily_sentiment.items()
        }

        return {
            "avg_daily_interactions": avg_daily_interactions,
            "sentiment_trend": avg_daily_sentiment,
            "most_active_day": (
                max(daily_interactions.items(), key=lambda x: x[1])[0]
                if daily_interactions
                else None
            ),
        }

    async def feedback_recommendation(
        self, user_id: str, recommendation_id: str, feedback: str
    ):
        """Registrar feedback sobre una recomendación"""
        # Buscar la recomendación
        user_recs = self.recommendation_history[user_id]
        rec = next((r for r in user_recs if r.id == recommendation_id), None)

        if rec:
            rec.metadata["feedback"] = feedback
            rec.metadata["feedback_timestamp"] = datetime.now().isoformat()

            # Ajustar futuras recomendaciones basadas en feedback
            if feedback == "positive":
                # Boost recomendaciones similares
                if user_id in self.user_profiles:
                    profile = self.user_profiles[user_id]
                    if rec.category == "topic_based" and "topic" in rec.metadata:
                        profile.update_preferences(rec.metadata["topic"], 0.8)
            elif feedback == "negative":
                # Reduce recomendaciones similares
                if user_id in self.user_profiles:
                    profile = self.user_profiles[user_id]
                    if rec.category == "topic_based" and "topic" in rec.metadata:
                        profile.update_preferences(rec.metadata["topic"], 0.2)

            logger.info(
                f"Feedback registrado para recomendación {recommendation_id}: {feedback}"
            )


# Instancia global
_personalization_instance = None


def get_personalization_engine() -> PersonalizationEngine:
    """Obtener instancia global del motor de personalización"""
    global _personalization_instance
    if _personalization_instance is None:
        _personalization_instance = PersonalizationEngine()
    return _personalization_instance


# Funciones de utilidad para integración
async def record_user_interaction(
    user_id: str,
    interaction_type: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Función de utilidad para registrar interacciones"""
    engine = get_personalization_engine()
    interaction = UserInteraction(
        user_id=user_id,
        timestamp=datetime.now(),
        interaction_type=interaction_type,
        content=content,
        metadata=metadata or {},
    )
    await engine.record_interaction(interaction)


async def get_user_recommendations(
    user_id: str, context: Optional[str] = None, limit: int = 5
) -> List[Recommendation]:
    """Función de utilidad para obtener recomendaciones"""
    engine = get_personalization_engine()
    return await engine.generate_recommendations(user_id, context, limit)


async def provide_recommendation_feedback(
    user_id: str, recommendation_id: str, feedback: str
):
    """Función de utilidad para feedback de recomendaciones"""
    engine = get_personalization_engine()
    await engine.feedback_recommendation(user_id, recommendation_id, feedback)
