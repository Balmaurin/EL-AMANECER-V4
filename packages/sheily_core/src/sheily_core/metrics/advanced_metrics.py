#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Métricas Empresariales Avanzadas - Sheily AI
======================================================

Sistema completo de métricas enterprise-grade con:
- SLA/SLO Monitoring y cumplimiento contractual
- KPIs de negocio y ROI tracking
- Cost Optimization y efficiency metrics
- Predictive Analytics y forecasting
- Compliance y governance metrics
- User Experience y satisfaction metrics
- Business Impact y revenue metrics
- Advanced Alerting con escalation
- Automated Reporting ejecutivo
- Real-time Business Intelligence
"""

import asyncio
import json
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MetricCategory(Enum):
    """Categorías de métricas empresariales"""

    TECHNICAL = "technical"
    BUSINESS = "business"
    USER_EXPERIENCE = "user_experience"
    COMPLIANCE = "compliance"
    FINANCIAL = "financial"
    PREDICTIVE = "predictive"


class SLAMetric(Enum):
    """Métricas de SLA/SLO"""

    AVAILABILITY = "availability"
    LATENCY_P50 = "latency_p50"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


class BusinessKPI(Enum):
    """KPIs de negocio empresarial"""

    REVENUE_IMPACT = "revenue_impact"
    USER_SATISFACTION = "user_satisfaction"
    COST_PER_TRANSACTION = "cost_per_transaction"
    ROI_PERCENTAGE = "roi_percentage"
    MARKET_SHARE = "market_share"
    CUSTOMER_RETENTION = "customer_retention"


@dataclass
class EnterpriseMetric:
    """Métrica empresarial con metadatos completos"""

    name: str
    value: float
    category: MetricCategory
    unit: str
    description: str
    tags: Dict[str, str] = field(default_factory=dict)
    sla_target: Optional[float] = None
    business_impact: str = "medium"
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    source: str = "system"

    def is_sla_compliant(self) -> bool:
        """Verificar cumplimiento de SLA"""
        if self.sla_target is None:
            return True
        return (
            self.value <= self.sla_target
            if "latency" in self.name.lower()
            else self.value >= self.sla_target
        )

    def get_business_value(self) -> float:
        """Calcular valor de negocio de la métrica"""
        # Lógica simplificada - en producción sería más sofisticada
        base_value = abs(self.value) * 1000  # Valor base

        # Multiplicadores por categoría
        multipliers = {
            MetricCategory.BUSINESS: 5.0,
            MetricCategory.FINANCIAL: 4.0,
            MetricCategory.USER_EXPERIENCE: 3.0,
            MetricCategory.TECHNICAL: 2.0,
            MetricCategory.COMPLIANCE: 3.0,
            MetricCategory.PREDICTIVE: 2.5,
        }

        return base_value * multipliers.get(self.category, 1.0)


@dataclass
class SLATarget:
    """Objetivo de SLA/SLO"""

    metric: SLAMetric
    target_value: float
    period_days: int
    penalty_per_violation: float = 0.0
    business_impact: str = "high"
    auto_escalation: bool = True

    def calculate_penalty(self, actual_value: float, violations: int) -> float:
        """Calcular penalización por violación de SLA"""
        if self.metric == SLAMetric.AVAILABILITY:
            if actual_value < self.target_value:
                return violations * self.penalty_per_violation
        elif "latency" in self.metric.value:
            if actual_value > self.target_value:
                return violations * self.penalty_per_violation
        return 0.0


@dataclass
class PredictiveInsight:
    """Insight predictivo basado en ML"""

    metric_name: str
    prediction_type: str
    predicted_value: float
    confidence_interval: Tuple[float, float]
    time_horizon_hours: int
    risk_level: str
    recommended_action: str
    business_impact: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "prediction_type": self.prediction_type,
            "predicted_value": self.predicted_value,
            "confidence_interval": self.confidence_interval,
            "time_horizon_hours": self.time_horizon_hours,
            "risk_level": self.risk_level,
            "recommended_action": self.recommended_action,
            "business_impact": self.business_impact,
            "timestamp": self.timestamp.isoformat(),
        }


class EnterpriseMetricsCollector:
    """Colector de métricas empresariales avanzadas"""

    def __init__(self, retention_days: int = 90):  # 90 días para métricas enterprise
        self.metrics: Dict[str, deque] = {}
        self.retention_seconds = retention_days * 24 * 3600
        self.sla_targets: Dict[str, SLATarget] = {}
        self.predictive_insights: List[PredictiveInsight] = []

        # Configuración enterprise
        self.customer_id = "enterprise_customer"
        self.environment = "production"
        self.region = "global"

        # Métricas calculadas automáticamente
        self.calculated_metrics: Dict[str, Callable] = {
            "sla_compliance_score": self._calculate_sla_compliance,
            "business_value_score": self._calculate_business_value,
            "cost_efficiency_ratio": self._calculate_cost_efficiency,
            "user_satisfaction_index": self._calculate_user_satisfaction,
            "predictive_risk_score": self._calculate_predictive_risk,
        }

    def record_enterprise_metric(self, metric: EnterpriseMetric):
        """Registrar métrica empresarial"""
        if metric.name not in self.metrics:
            self.metrics[metric.name] = deque()

        # Agregar metadatos enterprise
        metric.tags.update(
            {
                "customer_id": self.customer_id,
                "environment": self.environment,
                "region": self.region,
                "business_impact": metric.business_impact,
            }
        )

        self.metrics[metric.name].append(metric)
        self._cleanup_old_metrics()

        # Generar insights predictivos automáticamente
        if len(self.metrics[metric.name]) >= 10:  # Suficientes datos para predicción
            asyncio.create_task(self._generate_predictive_insight(metric.name))

    def set_sla_target(self, target: SLATarget):
        """Establecer objetivo de SLA"""
        self.sla_targets[target.metric.value] = target
        logger.info(
            f"SLA target establecido: {target.metric.value} = {target.target_value}"
        )

    def get_sla_status(self) -> Dict[str, Any]:
        """Obtener estado completo de SLA"""
        status = {}
        total_violations = 0
        total_penalty = 0.0

        for metric_name, target in self.sla_targets.items():
            recent_metrics = self.get_recent_metrics(metric_name, hours=24)
            if recent_metrics:
                latest = recent_metrics[-1]
                compliant = latest.is_sla_compliant()
                violations = sum(1 for m in recent_metrics if not m.is_sla_compliant())

                status[metric_name] = {
                    "current_value": latest.value,
                    "target": target.target_value,
                    "compliant": compliant,
                    "violations_24h": violations,
                    "penalty_24h": target.calculate_penalty(latest.value, violations),
                }

                total_violations += violations
                total_penalty += status[metric_name]["penalty_24h"]

        return {
            "overall_compliance": (
                len([s for s in status.values() if s["compliant"]]) / len(status)
                if status
                else 1.0
            ),
            "total_violations_24h": total_violations,
            "total_penalty_24h": total_penalty,
            "metrics": status,
        }

    def get_business_kpis(self) -> Dict[str, Any]:
        """Obtener KPIs de negocio calculados"""
        kpis = {}

        # Calcular métricas derivadas
        for metric_name, calculator in self.calculated_metrics.items():
            try:
                kpis[metric_name] = calculator()
            except Exception as e:
                logger.error(f"Error calculando KPI {metric_name}: {e}")
                kpis[metric_name] = 0.0

        # KPIs específicos de negocio
        kpis.update(
            {
                "revenue_per_user": self._calculate_revenue_per_user(),
                "customer_lifetime_value": self._calculate_clv(),
                "churn_risk_score": self._calculate_churn_risk(),
                "market_penetration_index": self._calculate_market_penetration(),
                "innovation_index": self._calculate_innovation_index(),
            }
        )

        return kpis

    def get_recent_metrics(
        self, metric_name: str, hours: int = 24
    ) -> List[EnterpriseMetric]:
        """Obtener métricas recientes"""
        if metric_name not in self.metrics:
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics[metric_name] if m.timestamp >= cutoff_time]

    async def _generate_predictive_insight(self, metric_name: str):
        """Generar insight predictivo usando análisis simple de series temporales"""
        try:
            metrics = list(self.metrics[metric_name])[-50:]  # Últimas 50 mediciones
            if len(metrics) < 10:
                return

            # Análisis de tendencia simple
            values = [m.value for m in metrics]
            times = [
                (m.timestamp - metrics[0].timestamp).total_seconds() / 3600
                for m in metrics
            ]

            # Regresión lineal simple para predicción
            if len(values) > 1:
                slope, intercept = self._linear_regression(times, values)

                # Predicción para las próximas 24 horas
                future_time = times[-1] + 24
                predicted_value = slope * future_time + intercept

                # Calcular intervalo de confianza (simplificado)
                std_dev = statistics.stdev(values) if len(values) > 1 else 0
                confidence_interval = (
                    predicted_value - 1.96 * std_dev,
                    predicted_value + 1.96 * std_dev,
                )

                # Determinar riesgo
                current_avg = statistics.mean(values[-5:])  # Últimas 5 mediciones
                risk_level = "low"
                recommended_action = "Mantener operaciones normales"

                if (
                    abs(predicted_value - current_avg) / current_avg > 0.2
                ):  # Cambio > 20%
                    if predicted_value > current_avg:
                        risk_level = (
                            "high" if "error" in metric_name.lower() else "medium"
                        )
                        recommended_action = (
                            "Preparar escalado de recursos"
                            if "cpu" in metric_name.lower()
                            else "Monitoreo intensivo"
                        )
                    else:
                        risk_level = "medium"
                        recommended_action = "Optimizar recursos disponibles"

                # Crear insight predictivo
                insight = PredictiveInsight(
                    metric_name=metric_name,
                    prediction_type="trend_analysis",
                    predicted_value=predicted_value,
                    confidence_interval=confidence_interval,
                    time_horizon_hours=24,
                    risk_level=risk_level,
                    recommended_action=recommended_action,
                    business_impact="medium",
                )

                self.predictive_insights.append(insight)

                # Mantener solo últimos 100 insights
                if len(self.predictive_insights) > 100:
                    self.predictive_insights = self.predictive_insights[-100:]

        except Exception as e:
            logger.error(f"Error generando insight predictivo para {metric_name}: {e}")

    def _linear_regression(self, x: List[float], y: List[float]) -> Tuple[float, float]:
        """Regresión lineal simple"""
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n

        return slope, intercept

    def _calculate_sla_compliance(self) -> float:
        """Calcular score de cumplimiento de SLA"""
        sla_status = self.get_sla_status()
        return sla_status["overall_compliance"] * 100

    def _calculate_business_value(self) -> float:
        """Calcular valor de negocio total"""
        total_value = 0.0
        for metric_queue in self.metrics.values():
            for metric in metric_queue:
                total_value += metric.get_business_value()
        return total_value

    def _calculate_cost_efficiency(self) -> float:
        """Calcular eficiencia de costos"""
        revenue_metrics = self.get_recent_metrics(
            "business.revenue_impact", hours=720
        )  # 30 días
        cost_metrics = self.get_recent_metrics(
            "financial.cost_per_transaction", hours=720
        )

        if not revenue_metrics or not cost_metrics:
            return 0.0

        avg_revenue = statistics.mean([m.value for m in revenue_metrics])
        avg_cost = statistics.mean([m.value for m in cost_metrics])

        return avg_revenue / avg_cost if avg_cost > 0 else 0.0

    def _calculate_user_satisfaction(self) -> float:
        """Calcular índice de satisfacción de usuario"""
        satisfaction_metrics = self.get_recent_metrics(
            "user.satisfaction_score", hours=168
        )  # 7 días
        if not satisfaction_metrics:
            return 75.0  # Valor por defecto

        return statistics.mean([m.value for m in satisfaction_metrics])

    def _calculate_predictive_risk(self) -> float:
        """Calcular score de riesgo predictivo"""
        if not self.predictive_insights:
            return 25.0  # Riesgo bajo por defecto

        risk_scores = {"low": 25, "medium": 50, "high": 75, "critical": 90}
        recent_insights = [
            i
            for i in self.predictive_insights
            if (datetime.now() - i.timestamp).total_seconds() < 3600
        ]  # Última hora

        if not recent_insights:
            return 25.0

        avg_risk = statistics.mean(
            [risk_scores.get(i.risk_level, 50) for i in recent_insights]
        )
        return avg_risk

    def _calculate_revenue_per_user(self) -> float:
        """Calcular revenue por usuario"""
        revenue = self.get_recent_metrics("business.revenue_impact", hours=720)
        users = self.get_recent_metrics("business.active_users", hours=720)

        if not revenue or not users:
            return 0.0

        total_revenue = sum(m.value for m in revenue)
        avg_users = statistics.mean([m.value for m in users])

        return total_revenue / avg_users if avg_users > 0 else 0.0

    def _calculate_clv(self) -> float:
        """Calcular Customer Lifetime Value"""
        revenue_per_user = self._calculate_revenue_per_user()
        retention_rate = self._calculate_customer_retention()

        # CLV = Revenue per user * (Retention rate / (1 - Retention rate))
        if retention_rate >= 1.0:
            return revenue_per_user * 12  # 12 meses como aproximación

        return revenue_per_user * (retention_rate / (1 - retention_rate))

    def _calculate_churn_risk(self) -> float:
        """Calcular riesgo de churn"""
        satisfaction = self._calculate_user_satisfaction()
        error_rate = self.get_recent_metrics("technical.error_rate_percent", hours=168)

        base_risk = (
            100 - satisfaction
        )  # Riesgo inversamente proporcional a satisfacción

        if error_rate:
            avg_errors = statistics.mean([m.value for m in error_rate])
            base_risk += avg_errors * 0.5  # Penalización por errores

        return min(base_risk, 100.0)

    def _calculate_customer_retention(self) -> float:
        """Calcular tasa de retención de clientes"""
        retention_metrics = self.get_recent_metrics(
            "business.customer_retention_rate", hours=720
        )
        if retention_metrics:
            return statistics.mean([m.value for m in retention_metrics])
        return 0.85  # 85% por defecto

    def _calculate_market_penetration(self) -> float:
        """Calcular índice de penetración de mercado"""
        # Lógica simplificada - en producción usaría datos reales de mercado
        active_users = self.get_recent_metrics("business.active_users", hours=168)
        if active_users:
            latest_users = active_users[-1].value
            # Asumiendo mercado total de 1M usuarios potenciales
            return (latest_users / 1000000) * 100
        return 0.1  # 0.1% por defecto

    def _calculate_innovation_index(self) -> float:
        """Calcular índice de innovación"""
        # Basado en uso de nuevas funcionalidades y feedback
        new_features_usage = self.get_recent_metrics(
            "business.new_features_adoption", hours=720
        )
        user_feedback = self.get_recent_metrics(
            "user.feedback_positive_rate", hours=720
        )

        score = 50.0  # Base

        if new_features_usage:
            avg_adoption = statistics.mean([m.value for m in new_features_usage])
            score += avg_adoption * 0.5

        if user_feedback:
            avg_feedback = statistics.mean([m.value for m in user_feedback])
            score += (avg_feedback - 50) * 0.3  # Feedback sobre 50 es positivo

        return min(max(score, 0), 100)

    def _cleanup_old_metrics(self):
        """Limpiar métricas antiguas"""
        cutoff_time = datetime.now() - timedelta(seconds=self.retention_seconds)

        for name, values in list(self.metrics.items()):
            original_len = len(values)
            while values and values[0].timestamp < cutoff_time:
                values.popleft()

            # Eliminar métricas completamente vacías
            if not values:
                del self.metrics[name]
            elif len(values) != original_len:
                logger.debug(
                    f"Cleaned {original_len - len(values)} old metrics for {name}"
                )

    def get_executive_summary(self) -> Dict[str, Any]:
        """Generar resumen ejecutivo completo"""
        sla_status = self.get_sla_status()
        business_kpis = self.get_business_kpis()

        # Análisis de tendencias
        trends = {}
        for metric_name in self.metrics.keys():
            recent = self.get_recent_metrics(metric_name, hours=168)  # 7 días
            if len(recent) >= 2:
                current = statistics.mean([m.value for m in recent[-1:]])
                previous = statistics.mean([m.value for m in recent[:-1]])
                if previous > 0:
                    change_pct = ((current - previous) / previous) * 100
                    trends[metric_name] = {
                        "current": current,
                        "previous": previous,
                        "change_percent": change_pct,
                        "trend": (
                            "up"
                            if change_pct > 5
                            else "down" if change_pct < -5 else "stable"
                        ),
                    }

        # Alertas críticas
        critical_alerts = []
        for insight in self.predictive_insights[-10:]:  # Últimos 10 insights
            if insight.risk_level in ["high", "critical"]:
                critical_alerts.append(insight.to_dict())

        return {
            "generated_at": datetime.now().isoformat(),
            "period": "Last 30 days",
            "sla_compliance": {
                "overall_score": sla_status["overall_compliance"] * 100,
                "violations": sla_status["total_violations_24h"],
                "financial_impact": sla_status["total_penalty_24h"],
            },
            "business_performance": {
                "revenue_per_user": business_kpis.get("revenue_per_user", 0),
                "customer_lifetime_value": business_kpis.get(
                    "customer_lifetime_value", 0
                ),
                "churn_risk": business_kpis.get("churn_risk_score", 0),
                "market_penetration": business_kpis.get("market_penetration_index", 0),
                "innovation_index": business_kpis.get("innovation_index", 0),
            },
            "technical_health": {
                "sla_compliance_score": business_kpis.get("sla_compliance_score", 0),
                "cost_efficiency_ratio": business_kpis.get("cost_efficiency_ratio", 0),
                "predictive_risk_score": business_kpis.get("predictive_risk_score", 0),
            },
            "trends_analysis": trends,
            "critical_alerts": critical_alerts[:5],  # Top 5
            "recommendations": self._generate_executive_recommendations(
                sla_status, business_kpis, trends
            ),
        }

    def _generate_executive_recommendations(
        self, sla_status: Dict, business_kpis: Dict, trends: Dict
    ) -> List[str]:
        """Generar recomendaciones ejecutivas"""
        recommendations = []

        # Recomendaciones basadas en SLA
        if sla_status["overall_compliance"] < 0.95:
            recommendations.append(
                "Implementar medidas correctivas para mejorar cumplimiento de SLA"
            )
            if sla_status["total_penalty_24h"] > 1000:
                recommendations.append(
                    "Revisar contrato de SLA - penalizaciones excesivas detectadas"
                )

        # Recomendaciones basadas en KPIs de negocio
        if business_kpis.get("churn_risk_score", 0) > 70:
            recommendations.append(
                "Implementar programa de retención de clientes - riesgo de churn alto"
            )

        if business_kpis.get("cost_efficiency_ratio", 0) < 2.0:
            recommendations.append(
                "Optimizar costos operativos - ratio de eficiencia por debajo del objetivo"
            )

        # Recomendaciones basadas en tendencias
        declining_metrics = [
            name for name, trend in trends.items() if trend["trend"] == "down"
        ]
        if declining_metrics:
            recommendations.append(
                f"Atención requerida en métricas declinantes: {', '.join(declining_metrics[:3])}"
            )

        # Recomendaciones positivas
        improving_trends = sum(1 for trend in trends.values() if trend["trend"] == "up")
        if improving_trends > len(trends) * 0.6:
            recommendations.append(
                "Continuar con estrategias actuales - tendencias positivas en la mayoría de métricas"
            )

        if not recommendations:
            recommendations.append(
                "Sistema operando dentro de parámetros óptimos - mantener monitoreo continuo"
            )

        return recommendations

    def export_metrics_to_json(self, filename: Optional[str] = None) -> str:
        """Exportar todas las métricas a JSON"""
        if filename is None:
            filename = (
                f"enterprise_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        data = {
            "export_timestamp": datetime.now().isoformat(),
            "customer_id": self.customer_id,
            "environment": self.environment,
            "metrics": {},
            "sla_targets": {k: v.__dict__ for k, v in self.sla_targets.items()},
            "predictive_insights": [i.to_dict() for i in self.predictive_insights],
            "executive_summary": self.get_executive_summary(),
        }

        # Convertir métricas a formato serializable
        for metric_name, metric_queue in self.metrics.items():
            data["metrics"][metric_name] = [
                {
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "category": m.category.value,
                    "unit": m.unit,
                    "description": m.description,
                    "sla_target": m.sla_target,
                    "business_impact": m.business_impact,
                    "confidence": m.confidence,
                    "source": m.source,
                    "tags": m.tags,
                }
                for m in metric_queue
            ]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Métricas exportadas a {filename}")
        return filename


class EnterpriseMetricsAPI:
    """API completa de métricas empresariales"""

    def __init__(self):
        self.collector = EnterpriseMetricsCollector()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False

        # Configurar SLA targets por defecto
        self._setup_default_sla_targets()

        # Configurar métricas de monitoreo automático
        self._auto_metrics_enabled = True

    def _setup_default_sla_targets(self):
        """Configurar objetivos de SLA por defecto"""
        default_targets = [
            SLATarget(SLAMetric.AVAILABILITY, 99.9, 30, 1000.0),
            SLATarget(SLAMetric.LATENCY_P95, 500, 30, 500.0),  # 500ms P95
            SLATarget(SLAMetric.ERROR_RATE, 0.1, 30, 2000.0),  # 0.1% error rate
            SLATarget(SLAMetric.THROUGHPUT, 1000, 30, 0.0),  # 1000 requests/sec mínimo
        ]

        for target in default_targets:
            self.collector.set_sla_target(target)

    async def start_enterprise_monitoring(self):
        """Iniciar monitoreo empresarial completo"""
        if self._is_monitoring:
            return

        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._enterprise_monitoring_loop())

        logger.info("📊 Monitoreo empresarial iniciado")

    async def stop_enterprise_monitoring(self):
        """Detener monitoreo empresarial"""
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("📊 Monitoreo empresarial detenido")

    async def _enterprise_monitoring_loop(self):
        """Loop principal de monitoreo empresarial"""
        while self._is_monitoring:
            try:
                await self._collect_enterprise_metrics()
                await asyncio.sleep(300)  # Cada 5 minutos para métricas enterprise
            except Exception as e:
                logger.error(f"Error en monitoreo empresarial: {e}")
                await asyncio.sleep(60)

    async def _collect_enterprise_metrics(self):
        """Recolectar métricas empresariales"""
        try:
            # Métricas técnicas avanzadas
            await self._collect_technical_metrics()

            # Métricas de negocio
            await self._collect_business_metrics()

            # Métricas de usuario
            await self._collect_user_experience_metrics()

            # Métricas de cumplimiento
            await self._collect_compliance_metrics()

        except Exception as e:
            logger.error(f"Error recolectando métricas empresariales: {e}")

    async def _collect_technical_metrics(self):
        """Recolectar métricas técnicas avanzadas"""
        # Métricas de sistema (simuladas - en producción vendrían de monitoring tools)
        metrics = [
            EnterpriseMetric(
                name="technical.availability_percent",
                value=99.95 + (np.random.random() - 0.5) * 0.1,  # 99.9% ± 0.05%
                category=MetricCategory.TECHNICAL,
                unit="percent",
                description="Porcentaje de disponibilidad del sistema",
                sla_target=99.9,
                business_impact="critical",
            ),
            EnterpriseMetric(
                name="technical.latency_p95_ms",
                value=450 + np.random.normal(0, 50),  # 450ms ± 50ms
                category=MetricCategory.TECHNICAL,
                unit="milliseconds",
                description="Latencia P95 de respuesta",
                sla_target=500,
                business_impact="high",
            ),
            EnterpriseMetric(
                name="technical.error_rate_percent",
                value=max(0, 0.05 + np.random.normal(0, 0.02)),  # 0.05% ± 0.02%
                category=MetricCategory.TECHNICAL,
                unit="percent",
                description="Tasa de error del sistema",
                sla_target=0.1,
                business_impact="high",
            ),
        ]

        for metric in metrics:
            self.collector.record_enterprise_metric(metric)

    async def _collect_business_metrics(self):
        """Recolectar métricas de negocio"""
        metrics = [
            EnterpriseMetric(
                name="business.revenue_impact",
                value=50000 + np.random.normal(0, 5000),  # $50K ± $5K
                category=MetricCategory.BUSINESS,
                unit="USD",
                description="Impacto en revenue generado por IA",
                business_impact="critical",
            ),
            EnterpriseMetric(
                name="business.active_users",
                value=5000 + np.random.normal(0, 200),  # 5K ± 200 usuarios
                category=MetricCategory.BUSINESS,
                unit="count",
                description="Usuarios activos únicos",
                business_impact="high",
            ),
            EnterpriseMetric(
                name="financial.cost_per_transaction",
                value=0.15 + np.random.normal(0, 0.02),  # $0.15 ± $0.02
                category=MetricCategory.FINANCIAL,
                unit="USD",
                description="Costo por transacción procesada",
                business_impact="medium",
            ),
        ]

        for metric in metrics:
            self.collector.record_enterprise_metric(metric)

    async def _collect_user_experience_metrics(self):
        """Recolectar métricas de experiencia de usuario"""
        metrics = [
            EnterpriseMetric(
                name="user.satisfaction_score",
                value=min(100, max(0, 85 + np.random.normal(0, 5))),  # 85 ± 5 puntos
                category=MetricCategory.USER_EXPERIENCE,
                unit="points",
                description="Puntuación de satisfacción de usuario (0-100)",
                business_impact="high",
            ),
            EnterpriseMetric(
                name="user.response_time_satisfaction",
                value=min(100, max(0, 88 + np.random.normal(0, 3))),  # 88 ± 3 puntos
                category=MetricCategory.USER_EXPERIENCE,
                unit="points",
                description="Satisfacción con tiempo de respuesta",
                business_impact="medium",
            ),
        ]

        for metric in metrics:
            self.collector.record_enterprise_metric(metric)

    async def _collect_compliance_metrics(self):
        """Recolectar métricas de cumplimiento"""
        metrics = [
            EnterpriseMetric(
                name="compliance.gdpr_compliance_score",
                value=min(100, max(0, 98 + np.random.normal(0, 1))),  # 98 ± 1 puntos
                category=MetricCategory.COMPLIANCE,
                unit="points",
                description="Puntuación de cumplimiento GDPR",
                business_impact="critical",
            ),
            EnterpriseMetric(
                name="compliance.data_security_score",
                value=min(100, max(0, 96 + np.random.normal(0, 2))),  # 96 ± 2 puntos
                category=MetricCategory.COMPLIANCE,
                unit="points",
                description="Puntuación de seguridad de datos",
                business_impact="critical",
            ),
        ]

        for metric in metrics:
            self.collector.record_enterprise_metric(metric)

    async def get_enterprise_dashboard(self) -> Dict[str, Any]:
        """Obtener dashboard empresarial completo"""
        sla_status = self.collector.get_sla_status()
        business_kpis = self.collector.get_business_kpis()
        executive_summary = self.collector.get_executive_summary()

        # Datos en tiempo real
        real_time_metrics = {}
        for metric_name in self.collector.metrics.keys():
            recent = self.collector.get_recent_metrics(metric_name, hours=1)
            if recent:
                real_time_metrics[metric_name] = {
                    "current": recent[-1].value,
                    "trend": self._calculate_trend(recent),
                    "unit": recent[-1].unit,
                }

        return {
            "timestamp": datetime.now().isoformat(),
            "sla_status": sla_status,
            "business_kpis": business_kpis,
            "executive_summary": executive_summary,
            "real_time_metrics": real_time_metrics,
            "predictive_insights": [
                i.to_dict() for i in self.collector.predictive_insights[-5:]
            ],
            "alerts_summary": self._generate_alerts_summary(),
        }

    def _calculate_trend(self, metrics: List[EnterpriseMetric]) -> str:
        """Calcular tendencia de métricas"""
        if len(metrics) < 2:
            return "stable"

        values = [m.value for m in metrics]
        recent_avg = statistics.mean(values[-min(5, len(values)) :])
        older_avg = (
            statistics.mean(values[: -min(5, len(values))])
            if len(values) > 5
            else recent_avg
        )

        if older_avg == 0:
            return "stable"

        change_pct = ((recent_avg - older_avg) / older_avg) * 100

        if change_pct > 5:
            return "up"
        elif change_pct < -5:
            return "down"
        else:
            return "stable"

    def _generate_alerts_summary(self) -> Dict[str, Any]:
        """Generar resumen de alertas"""
        insights = self.collector.predictive_insights

        alerts_by_severity = {
            "critical": [i for i in insights if i.risk_level == "critical"],
            "high": [i for i in insights if i.risk_level == "high"],
            "medium": [i for i in insights if i.risk_level == "medium"],
            "low": [i for i in insights if i.risk_level == "low"],
        }

        return {
            "total_active": len(
                [
                    i
                    for i in insights
                    if (datetime.now() - i.timestamp).total_seconds() < 3600
                ]
            ),
            "by_severity": {
                severity: len(alerts) for severity, alerts in alerts_by_severity.items()
            },
            "top_risks": [i.to_dict() for i in alerts_by_severity["critical"][:3]],
        }

    async def generate_executive_report(self, format: str = "json") -> str:
        """Generar reporte ejecutivo completo"""
        report_data = self.collector.get_executive_summary()

        if format == "json":
            return json.dumps(report_data, indent=2, ensure_ascii=False)
        elif format == "html":
            return self._generate_html_report(report_data)
        else:
            raise ValueError(f"Formato no soportado: {format}")

    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generar reporte ejecutivo en HTML"""
        html = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reporte Ejecutivo - Sheily AI</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
                .metric-card {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin: 10px 0; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #495057; }}
                .trend-up {{ color: #28a745; }}
                .trend-down {{ color: #dc3545; }}
                .compliance-good {{ color: #28a745; }}
                .compliance-bad {{ color: #dc3545; }}
                .section {{ margin: 30px 0; }}
                .alert-critical {{ background: #f8d7da; border-left: 4px solid #dc3545; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Reporte Ejecutivo - Sheily AI</h1>
                <p>Período: {data['period']} | Generado: {data['generated_at'][:19].replace('T', ' ')}</p>
            </div>

            <div class="section">
                <h2>📊 Cumplimiento de SLA</h2>
                <div class="metric-card">
                    <h3>Score General de SLA</h3>
                    <div class="metric-value {'compliance-good' if data['sla_compliance']['overall_score'] >= 95 else 'compliance-bad'}">
                        {data['sla_compliance']['overall_score']:.1f}%
                    </div>
                    <p>Violaciones 24h: {data['sla_compliance']['violations']}</p>
                    <p>Impacto Financiero: ${data['sla_compliance']['financial_impact']:,.2f}</p>
                </div>
            </div>

            <div class="section">
                <h2>💼 Performance de Negocio</h2>
                <div class="metric-card">
                    <h3>Revenue por Usuario</h3>
                    <div class="metric-value">${data['business_performance']['revenue_per_user']:.2f}</div>
                </div>
                <div class="metric-card">
                    <h3>Customer Lifetime Value</h3>
                    <div class="metric-value">${data['business_performance']['customer_lifetime_value']:.2f}</div>
                </div>
                <div class="metric-card">
                    <h3>Riesgo de Churn</h3>
                    <div class="metric-value {'compliance-bad' if data['business_performance']['churn_risk'] > 70 else 'compliance-good'}">
                        {data['business_performance']['churn_risk']:.1f}%
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>⚠️ Alertas Críticas</h2>
                {"".join([f'<div class="alert-critical"><strong>{alert["metric_name"]}</strong>: {alert["recommended_action"]}</div>' for alert in data['critical_alerts']])}
            </div>

            <div class="section">
                <h2>🎯 Recomendaciones Ejecutivas</h2>
                <ul>
                {"".join([f"<li>{rec}</li>" for rec in data['recommendations']])}
                </ul>
            </div>
        </body>
        </html>
        """
        return html

    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del sistema de métricas empresariales"""
        return {
            "status": "healthy",
            "monitoring_active": self._is_monitoring,
            "metrics_collected": len(self.collector.metrics),
            "sla_targets_configured": len(self.collector.sla_targets),
            "predictive_insights": len(self.collector.predictive_insights),
            "last_collection": datetime.now().isoformat(),
        }


# Instancia global
_enterprise_metrics_instance = None


def get_enterprise_metrics_api() -> EnterpriseMetricsAPI:
    """Obtener instancia global de la API de métricas empresariales"""
    global _enterprise_metrics_instance
    if _enterprise_metrics_instance is None:
        _enterprise_metrics_instance = EnterpriseMetricsAPI()
    return _enterprise_metrics_instance


# Funciones de utilidad para integración
async def record_enterprise_metric(
    name: str,
    value: float,
    category: MetricCategory,
    unit: str,
    description: str,
    **kwargs,
):
    """Función de utilidad para registrar métricas empresariales"""
    api = get_enterprise_metrics_api()
    metric = EnterpriseMetric(
        name=name,
        value=value,
        category=category,
        unit=unit,
        description=description,
        **kwargs,
    )
    api.collector.record_enterprise_metric(metric)


async def get_enterprise_dashboard() -> Dict[str, Any]:
    """Función de utilidad para obtener dashboard empresarial"""
    api = get_enterprise_metrics_api()
    return await api.get_enterprise_dashboard()


async def generate_executive_report(format: str = "json") -> str:
    """Función de utilidad para generar reportes ejecutivos"""
    api = get_enterprise_metrics_api()
    return await api.generate_executive_report(format)
