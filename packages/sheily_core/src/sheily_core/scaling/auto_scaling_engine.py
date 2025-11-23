#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Auto-Escalado Inteligente - Sheily AI
================================================

Sistema automático para escalado dinámico de recursos basado en:
- Monitoreo de carga de trabajo en tiempo real
- Escalado horizontal de contenedores y servicios
- Optimización de recursos basada en patrones de uso
- Escalado predictivo usando machine learning
- Gestión automática de costos y eficiencia
- Integración con Kubernetes y Docker Swarm
"""

import asyncio
import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Acciones de escalado"""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


class ScalingStrategy(Enum):
    """Estrategias de escalado"""

    REACTIVE = "reactive"  # Basado en métricas actuales
    PREDICTIVE = "predictive"  # Basado en predicciones
    HYBRID = "hybrid"  # Combinación de ambas


@dataclass
class ScalingRule:
    """Regla de escalado"""

    name: str
    metric_name: str
    threshold_high: float
    threshold_low: float
    cooldown_minutes: int = 5
    scale_factor: float = 1.5  # Factor de escalado (1.5 = 50% más)
    max_instances: int = 10
    min_instances: int = 1
    enabled: bool = True

    def evaluate(self, current_value: float, current_instances: int) -> ScalingAction:
        """Evaluar si se debe escalar"""
        if not self.enabled:
            return ScalingAction.NO_ACTION

        if (
            current_value >= self.threshold_high
            and current_instances < self.max_instances
        ):
            return ScalingAction.SCALE_UP
        elif (
            current_value <= self.threshold_low
            and current_instances > self.min_instances
        ):
            return ScalingAction.SCALE_DOWN

        return ScalingAction.NO_ACTION


@dataclass
class ScalingDecision:
    """Decisión de escalado"""

    service_name: str
    action: ScalingAction
    current_instances: int
    target_instances: int
    reason: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class ServiceState:
    """Estado de un servicio"""

    name: str
    current_instances: int
    target_instances: int
    last_scaling: Optional[datetime] = None
    scaling_cooldown_until: Optional[datetime] = None
    health_status: str = "healthy"
    metrics: Dict[str, float] = field(default_factory=dict)


class AutoScalingEngine:
    """Motor de auto-escalado inteligente"""

    def __init__(self):
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.service_states: Dict[str, ServiceState] = {}
        self.scaling_history: List[ScalingDecision] = []

        # Configuración
        self.strategy = ScalingStrategy.HYBRID
        self.check_interval_seconds = 30
        self.max_scaling_history = 1000

        # Callbacks para ejecutar acciones de escalado
        self.scaling_callbacks: Dict[str, Callable] = {}

        # Sistema de predicción simple
        self.metric_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.prediction_window_hours = 2

        # Estadísticas
        self.stats = {
            "total_decisions": 0,
            "scale_up_actions": 0,
            "scale_down_actions": 0,
            "no_action_decisions": 0,
            "prediction_accuracy": 0.0,
        }

    def add_scaling_rule(self, rule: ScalingRule):
        """Agregar regla de escalado"""
        self.scaling_rules[rule.name] = rule
        logger.info(f"Regla de escalado agregada: {rule.name}")

    def remove_scaling_rule(self, rule_name: str):
        """Remover regla de escalado"""
        if rule_name in self.scaling_rules:
            del self.scaling_rules[rule_name]
            logger.info(f"Regla de escalado removida: {rule_name}")

    def register_service(
        self,
        service_name: str,
        initial_instances: int = 1,
        scaling_callback: Optional[Callable] = None,
    ):
        """Registrar un servicio para auto-escalado"""
        self.service_states[service_name] = ServiceState(
            name=service_name,
            current_instances=initial_instances,
            target_instances=initial_instances,
        )

        if scaling_callback:
            self.scaling_callbacks[service_name] = scaling_callback

        logger.info(
            f"Servicio registrado para auto-escalado: {service_name} ({initial_instances} instancias)"
        )

    def update_service_metrics(self, service_name: str, metrics: Dict[str, float]):
        """Actualizar métricas de un servicio"""
        if service_name not in self.service_states:
            logger.warning(f"Servicio no registrado: {service_name}")
            return

        service = self.service_states[service_name]
        service.metrics.update(metrics)

        # Almacenar historial para predicciones
        now = datetime.now()
        for metric_name, value in metrics.items():
            key = f"{service_name}:{metric_name}"
            if key not in self.metric_history:
                self.metric_history[key] = []

            self.metric_history[key].append((now, value))

            # Mantener solo últimas 1000 mediciones
            if len(self.metric_history[key]) > 1000:
                self.metric_history[key] = self.metric_history[key][-1000:]

    def _predict_future_load(
        self, service_name: str, metric_name: str, hours_ahead: int = 1
    ) -> Optional[float]:
        """Predecir carga futura usando análisis de tendencias simple"""
        key = f"{service_name}:{metric_name}"
        if key not in self.metric_history:
            return None

        history = self.metric_history[key]
        if len(history) < 10:  # Necesitamos al menos 10 puntos de datos
            return None

        # Usar solo datos de las últimas horas
        cutoff = datetime.now() - timedelta(hours=self.prediction_window_hours)
        recent_data = [(dt, val) for dt, val in history if dt >= cutoff]

        if len(recent_data) < 5:
            return None

        # Calcular tendencia lineal simple
        values = [val for _, val in recent_data]
        n = len(values)

        if n < 2:
            return values[0]

        # Pendiente de la línea de tendencia
        x = list(range(n))
        slope = statistics.linear_regression(x, values).slope

        # Predecir valor futuro
        predicted = values[-1] + (
            slope * hours_ahead * (n / self.prediction_window_hours)
        )

        return max(0, predicted)  # No permitir valores negativos

    def _is_in_cooldown(self, service: ServiceState) -> bool:
        """Verificar si el servicio está en período de cooldown"""
        if service.scaling_cooldown_until is None:
            return False

        return datetime.now() < service.scaling_cooldown_until

    def _calculate_target_instances(
        self,
        service: ServiceState,
        rule: ScalingRule,
        current_value: float,
        predicted_value: Optional[float] = None,
    ) -> int:
        """Calcular número objetivo de instancias"""
        # Usar valor predicho si está disponible y la estrategia lo permite
        effective_value = (
            predicted_value
            if (
                predicted_value is not None
                and self.strategy
                in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID]
            )
            else current_value
        )

        action = rule.evaluate(effective_value, service.current_instances)

        if action == ScalingAction.SCALE_UP:
            # Escalar hacia arriba
            target = min(
                int(service.current_instances * rule.scale_factor), rule.max_instances
            )
            if target == service.current_instances:
                target = min(service.current_instances + 1, rule.max_instances)
        elif action == ScalingAction.SCALE_DOWN:
            # Escalar hacia abajo
            target = max(
                int(service.current_instances / rule.scale_factor), rule.min_instances
            )
            if target == service.current_instances:
                target = max(service.current_instances - 1, rule.min_instances)
        else:
            target = service.current_instances

        return target

    async def evaluate_scaling_decisions(self) -> List[ScalingDecision]:
        """Evaluar decisiones de escalado para todos los servicios"""
        decisions = []

        for service_name, service in self.service_states.items():
            if self._is_in_cooldown(service):
                continue

            # Evaluar cada regla aplicable
            for rule_name, rule in self.scaling_rules.items():
                if rule.metric_name not in service.metrics:
                    continue

                current_value = service.metrics[rule.metric_name]

                # Obtener predicción si está habilitada
                predicted_value = None
                if self.strategy in [
                    ScalingStrategy.PREDICTIVE,
                    ScalingStrategy.HYBRID,
                ]:
                    predicted_value = self._predict_future_load(
                        service_name, rule.metric_name
                    )

                # Calcular instancias objetivo
                target_instances = self._calculate_target_instances(
                    service, rule, current_value, predicted_value
                )

                if target_instances != service.current_instances:
                    # Crear decisión de escalado
                    action = (
                        ScalingAction.SCALE_UP
                        if target_instances > service.current_instances
                        else ScalingAction.SCALE_DOWN
                    )

                    decision = ScalingDecision(
                        service_name=service_name,
                        action=action,
                        current_instances=service.current_instances,
                        target_instances=target_instances,
                        reason=f"Regla '{rule_name}': {rule.metric_name} = {current_value:.2f}",
                        confidence=0.8,  # Confianza base, podría ser más sofisticado
                        metadata={
                            "rule": rule_name,
                            "metric_value": current_value,
                            "predicted_value": predicted_value,
                            "threshold_high": rule.threshold_high,
                            "threshold_low": rule.threshold_low,
                        },
                    )

                    decisions.append(decision)
                    break  # Solo una decisión por servicio por evaluación

        return decisions

    async def execute_scaling_decisions(self, decisions: List[ScalingDecision]):
        """Ejecutar decisiones de escalado"""
        for decision in decisions:
            service = self.service_states[decision.service_name]

            try:
                # Ejecutar callback si está disponible
                if decision.service_name in self.scaling_callbacks:
                    callback = self.scaling_callbacks[decision.service_name]
                    await callback(decision)
                else:
                    # Simular escalado (en producción se integraría con K8s, Docker, etc.)
                    logger.info(
                        f"Simulando escalado: {decision.service_name} "
                        f"{decision.current_instances} -> {decision.target_instances} instancias"
                    )

                # Actualizar estado del servicio
                service.current_instances = decision.target_instances
                service.last_scaling = decision.timestamp
                service.scaling_cooldown_until = decision.timestamp + timedelta(
                    minutes=5
                )  # Cooldown de 5 minutos

                # Registrar en historial
                self.scaling_history.append(decision)

                # Actualizar estadísticas
                self.stats["total_decisions"] += 1
                if decision.action == ScalingAction.SCALE_UP:
                    self.stats["scale_up_actions"] += 1
                elif decision.action == ScalingAction.SCALE_DOWN:
                    self.stats["scale_down_actions"] += 1
                else:
                    self.stats["no_action_decisions"] += 1

                logger.info(
                    f"✅ Escalado ejecutado: {decision.service_name} "
                    f"{decision.action.value} a {decision.target_instances} instancias"
                )

            except Exception as e:
                logger.error(
                    f"❌ Error ejecutando escalado para {decision.service_name}: {e}"
                )

        # Mantener historial limitado
        if len(self.scaling_history) > self.max_scaling_history:
            self.scaling_history = self.scaling_history[-self.max_scaling_history :]

    async def run_auto_scaling_loop(self):
        """Loop principal de auto-escalado"""
        logger.info("🚀 Iniciando loop de auto-escalado")

        while True:
            try:
                # Evaluar decisiones
                decisions = await self.evaluate_scaling_decisions()

                if decisions:
                    logger.info(
                        f"📊 {len(decisions)} decisiones de escalado pendientes"
                    )
                    await self.execute_scaling_decisions(decisions)
                else:
                    logger.debug("📊 No hay decisiones de escalado pendientes")

            except Exception as e:
                logger.error(f"Error en loop de auto-escalado: {e}")

            # Esperar al siguiente intervalo
            await asyncio.sleep(self.check_interval_seconds)

    def get_scaling_stats(self) -> Dict[str, any]:
        """Obtener estadísticas de escalado"""
        return {
            "total_decisions": self.stats["total_decisions"],
            "scale_up_actions": self.stats["scale_up_actions"],
            "scale_down_actions": self.stats["scale_down_actions"],
            "no_action_decisions": self.stats["no_action_decisions"],
            "active_services": len(self.service_states),
            "active_rules": len(self.scaling_rules),
            "strategy": self.strategy.value,
            "check_interval_seconds": self.check_interval_seconds,
        }

    def get_service_states(self) -> Dict[str, Dict[str, any]]:
        """Obtener estados de todos los servicios"""
        return {
            name: {
                "current_instances": service.current_instances,
                "target_instances": service.target_instances,
                "last_scaling": (
                    service.last_scaling.isoformat() if service.last_scaling else None
                ),
                "health_status": service.health_status,
                "metrics": service.metrics,
                "in_cooldown": self._is_in_cooldown(service),
            }
            for name, service in self.service_states.items()
        }

    async def health_check(self) -> Dict[str, any]:
        """Verificar salud del sistema de auto-escalado"""
        return {
            "status": "healthy",
            "services_registered": len(self.service_states),
            "rules_active": len(self.scaling_rules),
            "scaling_history_size": len(self.scaling_history),
            "metric_history_size": sum(
                len(history) for history in self.metric_history.values()
            ),
            "last_evaluation": datetime.now().isoformat(),
        }


# Funciones de integración con Kubernetes (simplificadas)
class KubernetesScaler:
    """Integración básica con Kubernetes para escalado"""

    def __init__(self, namespace: str = "default"):
        self.namespace = namespace

    async def scale_deployment(self, deployment_name: str, replicas: int) -> bool:
        """Escalar un deployment de Kubernetes"""
        try:
            # En un entorno real, esto usaría la API de Kubernetes
            logger.info(
                f"Simulando escalado K8s: {deployment_name} -> {replicas} replicas"
            )

            # Simular comando kubectl
            import subprocess

            result = subprocess.run(
                [
                    "kubectl",
                    "scale",
                    "deployment",
                    deployment_name,
                    f"--replicas={replicas}",
                    f"--namespace={self.namespace}",
                ],
                capture_output=True,
                text=True,
            )

            return result.returncode == 0

        except Exception as e:
            logger.error(f"Error escalando deployment {deployment_name}: {e}")
            return False

    async def get_deployment_replicas(self, deployment_name: str) -> Optional[int]:
        """Obtener número actual de replicas"""
        try:
            import subprocess

            result = subprocess.run(
                [
                    "kubectl",
                    "get",
                    "deployment",
                    deployment_name,
                    "-o",
                    "jsonpath={.spec.replicas}",
                    f"--namespace={self.namespace}",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                return int(result.stdout.strip())

        except Exception as e:
            logger.error(f"Error obteniendo replicas de {deployment_name}: {e}")

        return None


# Funciones de integración con Docker Swarm (simplificadas)
class DockerSwarmScaler:
    """Integración básica con Docker Swarm"""

    async def scale_service(self, service_name: str, replicas: int) -> bool:
        """Escalar un servicio de Docker Swarm"""
        try:
            logger.info(
                f"Simulando escalado Docker Swarm: {service_name} -> {replicas} replicas"
            )

            import subprocess

            result = subprocess.run(
                ["docker", "service", "scale", f"{service_name}={replicas}"],
                capture_output=True,
                text=True,
            )

            return result.returncode == 0

        except Exception as e:
            logger.error(f"Error escalando servicio {service_name}: {e}")
            return False


# Instancia global y funciones de utilidad
_auto_scaling_instance = None


def get_auto_scaling_engine() -> AutoScalingEngine:
    """Obtener instancia global del motor de auto-escalado"""
    global _auto_scaling_instance
    if _auto_scaling_instance is None:
        _auto_scaling_instance = AutoScalingEngine()
    return _auto_scaling_instance


async def initialize_auto_scaling():
    """Inicializar sistema de auto-escalado con configuración por defecto"""
    engine = get_auto_scaling_engine()

    # Registrar reglas por defecto
    default_rules = [
        ScalingRule(
            name="cpu_high",
            metric_name="cpu_usage_percent",
            threshold_high=70.0,
            threshold_low=30.0,
            scale_factor=1.5,
            max_instances=10,
            min_instances=1,
        ),
        ScalingRule(
            name="memory_high",
            metric_name="memory_usage_percent",
            threshold_high=75.0,
            threshold_low=40.0,
            scale_factor=1.3,
            max_instances=8,
            min_instances=1,
        ),
        ScalingRule(
            name="requests_high",
            metric_name="requests_per_minute",
            threshold_high=1000.0,
            threshold_low=200.0,
            scale_factor=2.0,
            max_instances=15,
            min_instances=1,
        ),
    ]

    for rule in default_rules:
        engine.add_scaling_rule(rule)

    # Registrar servicios comunes
    engine.register_service("rag-service", initial_instances=2)
    engine.register_service("api-gateway", initial_instances=1)
    engine.register_service("worker-pool", initial_instances=3)

    logger.info("✅ Sistema de auto-escalado inicializado")


async def start_auto_scaling_background():
    """Iniciar auto-escalado en segundo plano"""
    engine = get_auto_scaling_engine()
    asyncio.create_task(engine.run_auto_scaling_loop())
    logger.info("🚀 Auto-escalado iniciado en segundo plano")


def create_kubernetes_callback(deployment_name: str, namespace: str = "default"):
    """Crear callback para escalado de Kubernetes"""
    scaler = KubernetesScaler(namespace)

    async def callback(decision: ScalingDecision):
        success = await scaler.scale_deployment(
            deployment_name, decision.target_instances
        )
        if not success:
            raise Exception(f"Failed to scale {deployment_name}")

    return callback


def create_docker_callback(service_name: str):
    """Crear callback para escalado de Docker Swarm"""
    scaler = DockerSwarmScaler()

    async def callback(decision: ScalingDecision):
        success = await scaler.scale_service(service_name, decision.target_instances)
        if not success:
            raise Exception(f"Failed to scale {service_name}")

    return callback
