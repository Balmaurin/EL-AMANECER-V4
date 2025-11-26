"""
Sistema de Analytics Educativos para Sheily AI
Análisis avanzado de datos educativos y métricas de aprendizaje
Basado en investigación: Modelos económicos, QCoin analytics, investigación Web3

Características:
- Análisis predictivo de engagement estudiantil
- Métricas de calidad de aprendizaje
- Recomendaciones personalizadas
- Dashboards en tiempo real
"""

import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EducationalAnalytics:
    """
    Sistema de analytics educativos avanzados
    Proporciona insights y recomendaciones basadas en datos
    """

    def __init__(self):
        self.learning_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.prediction_models: Dict[str, Any] = {}
        self.analytics_cache: Dict[str, Any] = {}

        logger.info("📊 Educational Analytics system initialized")

    async def record_learning_event(self, user_id: str, event_data: Dict[str, Any]):
        """
        Registrar evento de aprendizaje para análisis
        """
        try:
            event = {"timestamp": datetime.now(), "user_id": user_id, **event_data}

            self.learning_data[user_id].append(event)

            # Limpiar cache para forzar recálculo
            if user_id in self.analytics_cache:
                del self.analytics_cache[user_id]

            logger.debug(f"📝 Learning event recorded for user {user_id}")

        except Exception as e:
            logger.error(f"Error recording learning event: {e}")

    async def get_user_learning_analytics(self, user_id: str) -> Dict[str, Any]:
        """
        Obtener analytics completos de aprendizaje del usuario
        """
        try:
            if user_id in self.analytics_cache:
                return self.analytics_cache[user_id]

            user_events = self.learning_data.get(user_id, [])
            if not user_events:
                return {"user_id": user_id, "message": "No learning data available"}

            # Análisis temporal
            time_analytics = self._analyze_temporal_patterns(user_events)

            # Análisis de rendimiento
            performance_analytics = self._analyze_performance_patterns(user_events)

            # Análisis de engagement
            engagement_analytics = self._analyze_engagement_patterns(user_events)

            # Predicciones y recomendaciones
            predictions = await self._generate_predictions(user_id, user_events)

            analytics_result = {
                "user_id": user_id,
                "total_events": len(user_events),
                "date_range": {
                    "start": min(e["timestamp"] for e in user_events).isoformat(),
                    "end": max(e["timestamp"] for e in user_events).isoformat(),
                },
                "temporal_analytics": time_analytics,
                "performance_analytics": performance_analytics,
                "engagement_analytics": engagement_analytics,
                "predictions": predictions,
                "recommendations": self._generate_recommendations(
                    performance_analytics, engagement_analytics
                ),
                "generated_at": datetime.now().isoformat(),
            }

            # Cachear resultado
            self.analytics_cache[user_id] = analytics_result

            return analytics_result

        except Exception as e:
            logger.error(f"Error getting user learning analytics: {e}")
            return {
                "user_id": user_id,
                "error": str(e),
                "generated_at": datetime.now().isoformat(),
            }

    def _analyze_temporal_patterns(
        self, events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analizar patrones temporales de aprendizaje
        """
        try:
            if not events:
                return {}

            # Eventos por día de la semana
            weekday_counts = defaultdict(int)
            hour_counts = defaultdict(int)

            for event in events:
                dt = event["timestamp"]
                weekday_counts[dt.weekday()] += 1
                hour_counts[dt.hour] += 1

            # Calcular día más activo
            most_active_day = max(weekday_counts.items(), key=lambda x: x[1])[0]
            day_names = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]

            # Calcular hora más activa
            most_active_hour = max(hour_counts.items(), key=lambda x: x[1])[0]

            # Calcular consistencia (días con actividad / días totales)
            event_dates = set(event["timestamp"].date() for event in events)
            total_days = (max(event_dates) - min(event_dates)).days + 1
            consistency_rate = len(event_dates) / max(total_days, 1)

            # Calcular racha actual de aprendizaje
            sorted_dates = sorted(event_dates)
            current_streak = 0
            max_streak = 0
            temp_streak = 0

            for i, date in enumerate(sorted_dates):
                if i == 0 or (date - sorted_dates[i - 1]).days == 1:
                    temp_streak += 1
                    current_streak = temp_streak
                else:
                    max_streak = max(max_streak, temp_streak)
                    temp_streak = 1

            max_streak = max(max_streak, temp_streak)

            return {
                "most_active_day": day_names[most_active_day],
                "most_active_hour": most_active_hour,
                "consistency_rate": round(consistency_rate * 100, 1),
                "current_streak_days": current_streak,
                "max_streak_days": max_streak,
                "avg_events_per_day": round(len(events) / max(len(event_dates), 1), 2),
                "weekday_distribution": dict(weekday_counts),
                "hour_distribution": dict(hour_counts),
            }

        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {e}")
            return {"error": str(e)}

    def _analyze_performance_patterns(
        self, events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analizar patrones de rendimiento
        """
        try:
            # Filtrar eventos con scores de calidad
            scored_events = [e for e in events if "quality_score" in e]

            if not scored_events:
                return {"message": "No performance data available"}

            quality_scores = [e["quality_score"] for e in scored_events]

            # Estadísticas básicas
            avg_quality = statistics.mean(quality_scores)
            median_quality = statistics.median(quality_scores)

            # Tendencia (últimos 10 eventos vs promedio general)
            recent_scores = (
                quality_scores[-10:] if len(quality_scores) >= 10 else quality_scores
            )
            recent_avg = statistics.mean(recent_scores)
            trend = (
                "improving"
                if recent_avg > avg_quality
                else "declining" if recent_avg < avg_quality else "stable"
            )

            # Variabilidad
            if len(quality_scores) > 1:
                std_dev = statistics.stdev(quality_scores)
                cv = std_dev / avg_quality  # Coeficiente de variación
            else:
                std_dev = 0
                cv = 0

            # Distribución por rangos
            ranges = {
                "excellent": sum(1 for s in quality_scores if s >= 0.9),
                "good": sum(1 for s in quality_scores if 0.8 <= s < 0.9),
                "average": sum(1 for s in quality_scores if 0.7 <= s < 0.8),
                "needs_improvement": sum(1 for s in quality_scores if s < 0.7),
            }

            # Mejores y peores desempeños
            best_performance = max(scored_events, key=lambda x: x["quality_score"])
            worst_performance = min(scored_events, key=lambda x: x["quality_score"])

            return {
                "average_quality_score": round(avg_quality, 3),
                "median_quality_score": round(median_quality, 3),
                "quality_trend": trend,
                "quality_variability": round(cv, 3),
                "performance_distribution": ranges,
                "total_scored_events": len(scored_events),
                "best_performance": {
                    "score": best_performance["quality_score"],
                    "activity": best_performance.get("activity_type", "unknown"),
                    "date": best_performance["timestamp"].isoformat(),
                },
                "worst_performance": {
                    "score": worst_performance["quality_score"],
                    "activity": worst_performance.get("activity_type", "unknown"),
                    "date": worst_performance["timestamp"].isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"Error analyzing performance patterns: {e}")
            return {"error": str(e)}

    def _analyze_engagement_patterns(
        self, events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analizar patrones de engagement
        """
        try:
            # Análisis de tipos de actividades
            activity_counts = defaultdict(int)
            engagement_levels = defaultdict(int)

            for event in events:
                if "activity_type" in event:
                    activity_counts[event["activity_type"]] += 1
                if "engagement_level" in event:
                    engagement_levels[event["engagement_level"]] += 1

            # Actividad más frecuente
            most_frequent_activity = (
                max(activity_counts.items(), key=lambda x: x[1])
                if activity_counts
                else ("none", 0)
            )

            # Nivel de engagement predominante
            dominant_engagement = (
                max(engagement_levels.items(), key=lambda x: x[1])
                if engagement_levels
                else ("unknown", 0)
            )

            # Duración promedio de sesiones
            durations = []
            for event in events:
                if "duration_minutes" in event and event["duration_minutes"] > 0:
                    durations.append(event["duration_minutes"])

            avg_duration = statistics.mean(durations) if durations else 0

            # Análisis de frecuencia
            event_dates = [e["timestamp"].date() for e in events]
            unique_days = len(set(event_dates))
            total_days = (
                (max(event_dates) - min(event_dates)).days + 1 if event_dates else 1
            )
            activity_frequency = len(events) / max(total_days, 1)

            return {
                "most_frequent_activity": {
                    "type": most_frequent_activity[0],
                    "count": most_frequent_activity[1],
                },
                "dominant_engagement_level": {
                    "level": dominant_engagement[0],
                    "count": dominant_engagement[1],
                },
                "activity_distribution": dict(activity_counts),
                "engagement_distribution": dict(engagement_levels),
                "avg_session_duration_minutes": round(avg_duration, 2),
                "activity_frequency_per_day": round(activity_frequency, 2),
                "active_days": unique_days,
                "total_engagement_events": len(events),
            }

        except Exception as e:
            logger.error(f"Error analyzing engagement patterns: {e}")
            return {"error": str(e)}

    async def _generate_predictions(
        self, user_id: str, events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generar predicciones sobre el rendimiento futuro del usuario
        """
        try:
            if len(events) < 5:
                return {"message": "Insufficient data for predictions"}

            # Predicción de score de calidad futuro
            quality_scores = [
                e.get("quality_score", 0) for e in events if "quality_score" in e
            ]
            if len(quality_scores) >= 3:
                # Predicción simple basada en tendencia
                recent_avg = statistics.mean(quality_scores[-3:])
                overall_avg = statistics.mean(quality_scores)
                predicted_quality = min(
                    1.0, max(0.0, recent_avg + (recent_avg - overall_avg) * 0.1)
                )
            else:
                predicted_quality = 0.7  # Valor por defecto

            # Predicción de engagement futuro
            engagement_scores = []
            for event in events:
                level = event.get("engagement_level", "medium")
                score_map = {"low": 1, "medium": 2, "high": 3, "exceptional": 4}
                engagement_scores.append(score_map.get(level, 2))

            if engagement_scores:
                avg_engagement = statistics.mean(engagement_scores)
                predicted_engagement_score = min(4, max(1, avg_engagement))
                engagement_map = {1: "low", 2: "medium", 3: "high", 4: "exceptional"}
                predicted_engagement = engagement_map.get(
                    int(predicted_engagement_score), "medium"
                )
            else:
                predicted_engagement = "medium"

            # Predicción de retención (probabilidad de continuar aprendiendo)
            # Basado en consistencia y engagement reciente
            recent_events = events[-10:]  # Últimos 10 eventos
            recent_dates = [e["timestamp"].date() for e in recent_events]
            unique_recent_days = len(set(recent_dates))

            retention_score = min(1.0, unique_recent_days / 7)  # Normalizado por semana
            retention_probability = round(retention_score * 100, 1)

            return {
                "predicted_quality_score": round(predicted_quality, 3),
                "predicted_engagement_level": predicted_engagement,
                "retention_probability_percent": retention_probability,
                "confidence_level": "medium",  # low, medium, high
                "prediction_basis": f"Based on {len(events)} learning events",
            }

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return {"error": str(e)}

    def _generate_recommendations(
        self, performance: Dict[str, Any], engagement: Dict[str, Any]
    ) -> List[str]:
        """
        Generar recomendaciones personalizadas basadas en analytics
        """
        recommendations = []

        try:
            # Recomendaciones basadas en rendimiento
            if "average_quality_score" in performance:
                avg_score = performance["average_quality_score"]
                if avg_score < 0.7:
                    recommendations.append(
                        "Consider reviewing fundamental concepts in challenging areas"
                    )
                elif avg_score > 0.9:
                    recommendations.append(
                        "Great performance! Consider taking on advanced challenges"
                    )

            # Recomendaciones basadas en engagement
            if "activity_frequency_per_day" in engagement:
                frequency = engagement["activity_frequency_per_day"]
                if frequency < 1:
                    recommendations.append(
                        "Try to engage in learning activities at least once daily"
                    )
                elif frequency > 3:
                    recommendations.append(
                        "Excellent consistency! Consider balancing with reflection time"
                    )

            # Recomendaciones basadas en consistencia
            if "consistency_rate" in engagement.get("temporal_analytics", {}):
                consistency = engagement["temporal_analytics"]["consistency_rate"]
                if consistency < 50:
                    recommendations.append(
                        "Try establishing a regular learning schedule"
                    )
                elif consistency > 80:
                    recommendations.append(
                        "Strong consistency! Keep up the excellent habit"
                    )

            # Recomendaciones por defecto si no hay datos específicos
            if not recommendations:
                recommendations.extend(
                    [
                        "Continue building on your current learning momentum",
                        "Consider exploring new learning activities to broaden your skills",
                        "Regular self-assessment can help track your progress",
                    ]
                )

            return recommendations[:3]  # Máximo 3 recomendaciones

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Continue your learning journey with consistent practice"]

    async def get_system_wide_analytics(self) -> Dict[str, Any]:
        """
        Obtener analytics a nivel del sistema completo
        """
        try:
            all_users = list(self.learning_data.keys())
            if not all_users:
                return {"message": "No learning data available"}

            # Agregar analytics de todos los usuarios
            system_performance = []
            system_engagement = []
            activity_types = defaultdict(int)

            for user_id in all_users:
                user_analytics = await self.get_user_learning_analytics(user_id)

                if "performance_analytics" in user_analytics:
                    perf = user_analytics["performance_analytics"]
                    if "average_quality_score" in perf:
                        system_performance.append(perf["average_quality_score"])

                if "engagement_analytics" in user_analytics:
                    eng = user_analytics["engagement_analytics"]
                    if "total_engagement_events" in eng:
                        system_engagement.append(eng["total_engagement_events"])

                    # Contar tipos de actividades
                    for activity, count in eng.get("activity_distribution", {}).items():
                        activity_types[activity] += count

            # Calcular métricas del sistema
            avg_system_performance = (
                statistics.mean(system_performance) if system_performance else 0
            )
            avg_system_engagement = (
                statistics.mean(system_engagement) if system_engagement else 0
            )

            # Usuarios más activos
            user_activity = []
            for user_id in all_users:
                events = self.learning_data[user_id]
                user_activity.append(
                    {
                        "user_id": user_id,
                        "total_events": len(events),
                        "avg_quality": (
                            statistics.mean(
                                [
                                    e.get("quality_score", 0)
                                    for e in events
                                    if "quality_score" in e
                                ]
                            )
                            if events
                            else 0
                        ),
                    }
                )

            top_performers = sorted(
                user_activity, key=lambda x: x["avg_quality"], reverse=True
            )[:5]
            most_active = sorted(
                user_activity, key=lambda x: x["total_events"], reverse=True
            )[:5]

            return {
                "total_users": len(all_users),
                "total_learning_events": sum(
                    len(events) for events in self.learning_data.values()
                ),
                "system_avg_performance": round(avg_system_performance, 3),
                "system_avg_engagement": round(avg_system_engagement, 2),
                "most_popular_activities": dict(
                    sorted(activity_types.items(), key=lambda x: x[1], reverse=True)[:5]
                ),
                "top_performers": top_performers,
                "most_active_users": most_active,
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting system-wide analytics: {e}")
            return {"error": str(e), "generated_at": datetime.now().isoformat()}


# Instancia global (singleton)
_educational_analytics: Optional[EducationalAnalytics] = None


def get_educational_analytics() -> EducationalAnalytics:
    """Obtener instancia singleton del sistema de analytics educativos"""
    global _educational_analytics
    if _educational_analytics is None:
        _educational_analytics = EducationalAnalytics()
    return _educational_analytics
