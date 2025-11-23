#!/usr/bin/env python3
"""
Advanced Policy Engines - Sistema de Políticas Avanzadas
Motores de políticas inteligentes con aprendizaje automático y razonamiento ético
"""

import asyncio
import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# =============================================================================
# MODELOS DE DATOS PARA POLÍTICAS
# =============================================================================


class PolicyType(Enum):
    """Tipos de políticas"""

    SECURITY = "security"
    ETHICAL = "ethical"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    BUSINESS = "business"
    TECHNICAL = "technical"


class PolicyScope(Enum):
    """Alcance de las políticas"""

    GLOBAL = "global"
    ORGANIZATION = "organization"
    TEAM = "team"
    AGENT = "agent"
    TASK = "task"
    RESOURCE = "resource"


class PolicyEffect(Enum):
    """Efectos de las políticas"""

    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"
    AUDIT = "audit"
    MODIFY = "modify"
    REDIRECT = "redirect"


class PolicyPriority(Enum):
    """Prioridades de políticas"""

    CRITICAL = 100
    HIGH = 75
    MEDIUM = 50
    LOW = 25
    INFO = 10


@dataclass
class PolicyCondition:
    """Condición de política"""

    field: str
    operator: str  # equals, contains, regex, gt, lt, etc.
    value: Any
    negate: bool = False

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evalúa la condición contra un contexto"""
        try:
            actual_value = self._get_nested_value(context, self.field)

            result = self._compare_values(actual_value, self.value, self.operator)

            return not result if self.negate else result

        except Exception as e:
            logger.warning(f"Error evaluating condition {self.field}: {e}")
            return False

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Obtiene valor anidado usando dot notation"""
        keys = path.split(".")
        current = data

        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, list) and key.isdigit():
                current = current[int(key)] if int(key) < len(current) else None
            else:
                return None

        return current

    def _compare_values(self, actual: Any, expected: Any, operator: str) -> bool:
        """Compara valores según operador"""
        if operator == "equals":
            return actual == expected
        elif operator == "not_equals":
            return actual != expected
        elif operator == "contains":
            return expected in actual if isinstance(actual, (str, list)) else False
        elif operator == "not_contains":
            return expected not in actual if isinstance(actual, (str, list)) else True
        elif operator == "regex":
            return bool(re.search(expected, str(actual)))
        elif operator == "gt":
            return actual > expected if isinstance(actual, (int, float)) else False
        elif operator == "gte":
            return actual >= expected if isinstance(actual, (int, float)) else False
        elif operator == "lt":
            return actual < expected if isinstance(actual, (int, float)) else False
        elif operator == "lte":
            return actual <= expected if isinstance(actual, (int, float)) else False
        elif operator == "in":
            return actual in expected if isinstance(expected, (list, tuple)) else False
        elif operator == "not_in":
            return (
                actual not in expected if isinstance(expected, (list, tuple)) else True
            )

        return False


@dataclass
class PolicyAction:
    """Acción de política"""

    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta la acción"""
        if self.action_type == "log":
            return self._log_action(context)
        elif self.action_type == "alert":
            return self._alert_action(context)
        elif self.action_type == "modify":
            return self._modify_action(context)
        elif self.action_type == "redirect":
            return self._redirect_action(context)
        elif self.action_type == "block":
            return self._block_action(context)

        return {"status": "unknown_action"}

    def _log_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Acción de logging"""
        level = self.parameters.get("level", "info")
        message = self.parameters.get("message", "Policy action triggered")

        logger.log(
            getattr(logging, level.upper(), logging.INFO),
            f"Policy: {message} - Context: {context}",
        )

        return {"status": "logged", "level": level}

    def _alert_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Acción de alerta"""
        # Aquí se integraría con sistema de alertas
        alert_message = self.parameters.get("message", "Policy violation detected")
        severity = self.parameters.get("severity", "medium")

        logger.warning(f"ALERT [{severity}]: {alert_message}")

        # En producción, enviaría a sistema de alertas
        return {"status": "alerted", "severity": severity}

    def _modify_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Acción de modificación"""
        modifications = self.parameters.get("modifications", {})

        for key, value in modifications.items():
            keys = key.split(".")
            self._set_nested_value(context, keys, value)

        return {"status": "modified", "modifications": modifications}

    def _redirect_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Acción de redirección"""
        target = self.parameters.get("target", "default")
        reason = self.parameters.get("reason", "Policy redirect")

        context["redirected_to"] = target
        context["redirect_reason"] = reason

        return {"status": "redirected", "target": target}

    def _block_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Acción de bloqueo"""
        reason = self.parameters.get("reason", "Policy violation")

        context["blocked"] = True
        context["block_reason"] = reason

        return {"status": "blocked", "reason": reason}

    def _set_nested_value(self, data: Dict[str, Any], keys: List[str], value: Any):
        """Establece valor anidado usando dot notation"""
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value


@dataclass
class Policy:
    """Política completa"""

    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    policy_type: PolicyType = PolicyType.SECURITY
    scope: PolicyScope = PolicyScope.GLOBAL
    priority: PolicyPriority = PolicyPriority.MEDIUM
    conditions: List[PolicyCondition] = field(default_factory=list)
    actions: List[PolicyAction] = field(default_factory=list)
    effect: PolicyEffect = PolicyEffect.ALLOW
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"

    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        """Evalúa la política contra un contexto"""
        if not self.enabled:
            return False, []

        # Verificar todas las condiciones
        for condition in self.conditions:
            if not condition.evaluate(context):
                return False, []  # No aplica

        # Ejecutar acciones
        action_results = []
        for action in self.actions:
            result = action.execute(context)
            action_results.append(result)

        return True, action_results

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "policy_type": self.policy_type.value,
            "scope": self.scope.value,
            "priority": self.priority.value,
            "conditions": [
                {
                    "field": c.field,
                    "operator": c.operator,
                    "value": c.value,
                    "negate": c.negate,
                }
                for c in self.conditions
            ],
            "actions": [
                {"action_type": a.action_type, "parameters": a.parameters}
                for a in self.actions
            ],
            "effect": self.effect.value,
            "enabled": self.enabled,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
        }


# =============================================================================
# MOTOR DE POLÍTICAS INTELIGENTE
# =============================================================================


class IntelligentPolicyEngine:
    """Motor de políticas con aprendizaje automático"""

    def __init__(self):
        self.policies: Dict[str, Policy] = {}
        self.policy_cache: Dict[str, List[Policy]] = {}
        self.learning_engine = PolicyLearningEngine()
        self.violation_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}

    async def initialize_engine(self):
        """Inicializa el motor de políticas"""
        await self.learning_engine.initialize()
        await self._load_default_policies()
        logger.info("Intelligent Policy Engine initialized")

    async def evaluate_request(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa una solicitud contra todas las políticas aplicables"""
        start_time = datetime.now()

        # Enriquecer contexto con metadatos
        enriched_context = await self._enrich_context(request_context)

        # Obtener políticas aplicables
        applicable_policies = self._get_applicable_policies(enriched_context)

        # Evaluar políticas en orden de prioridad
        evaluation_results = []
        final_decision = "allow"
        triggered_actions = []

        for policy in sorted(
            applicable_policies, key=lambda p: p.priority.value, reverse=True
        ):
            try:
                applies, actions = policy.evaluate(enriched_context)

                if applies:
                    evaluation_results.append(
                        {
                            "policy_id": policy.policy_id,
                            "policy_name": policy.name,
                            "effect": policy.effect.value,
                            "actions": actions,
                        }
                    )

                    # Aplicar efecto de la política
                    if policy.effect == PolicyEffect.DENY:
                        final_decision = "deny"
                        triggered_actions.extend(actions)
                        break  # Deny tiene precedencia
                    elif policy.effect == PolicyEffect.WARN:
                        final_decision = "warn"
                        triggered_actions.extend(actions)
                    elif policy.effect == PolicyEffect.AUDIT:
                        triggered_actions.extend(actions)
                    else:
                        triggered_actions.extend(actions)

            except Exception as e:
                logger.error(f"Error evaluating policy {policy.policy_id}: {e}")
                evaluation_results.append(
                    {"policy_id": policy.policy_id, "error": str(e)}
                )

        # Registrar resultado
        await self._record_evaluation_result(
            enriched_context, evaluation_results, final_decision
        )

        # Calcular métricas de performance
        end_time = datetime.now()
        evaluation_time = (end_time - start_time).total_seconds()

        result = {
            "decision": final_decision,
            "evaluation_time_seconds": evaluation_time,
            "policies_evaluated": len(applicable_policies),
            "policies_triggered": len(evaluation_results),
            "actions_triggered": triggered_actions,
            "enriched_context": enriched_context,
            "evaluation_results": evaluation_results,
        }

        # Aprender de la evaluación
        await self.learning_engine.learn_from_evaluation(result)

        return result

    async def add_policy(self, policy: Policy) -> str:
        """Añade una nueva política"""
        self.policies[policy.policy_id] = policy

        # Limpiar cache
        self.policy_cache.clear()

        # Registrar con learning engine
        await self.learning_engine.register_policy(policy)

        logger.info(f"Added policy: {policy.name} ({policy.policy_id})")
        return policy.policy_id

    async def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:
        """Actualiza una política existente"""
        if policy_id not in self.policies:
            return False

        policy = self.policies[policy_id]

        # Aplicar actualizaciones
        for key, value in updates.items():
            if hasattr(policy, key):
                setattr(policy, key, value)

        policy.updated_at = datetime.now()
        policy.version = self._increment_version(policy.version)

        # Limpiar cache
        self.policy_cache.clear()

        # Actualizar learning engine
        await self.learning_engine.update_policy(policy)

        logger.info(f"Updated policy: {policy.name} ({policy.policy_id})")
        return True

    async def remove_policy(self, policy_id: str) -> bool:
        """Elimina una política"""
        if policy_id not in self.policies:
            return False

        policy = self.policies.pop(policy_id)

        # Limpiar cache
        self.policy_cache.clear()

        # Remover de learning engine
        await self.learning_engine.remove_policy(policy_id)

        logger.info(f"Removed policy: {policy.name} ({policy.policy_id})")
        return True

    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Obtiene una política por ID"""
        return self.policies.get(policy_id)

    def list_policies(
        self,
        policy_type: Optional[PolicyType] = None,
        scope: Optional[PolicyScope] = None,
    ) -> List[Policy]:
        """Lista políticas con filtros opcionales"""
        policies = list(self.policies.values())

        if policy_type:
            policies = [p for p in policies if p.policy_type == policy_type]

        if scope:
            policies = [p for p in policies if p.scope == scope]

        return sorted(policies, key=lambda p: p.priority.value, reverse=True)

    async def _enrich_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enriquece el contexto con información adicional"""
        enriched = dict(context)  # Copia

        # Añadir metadatos temporales
        enriched["timestamp"] = datetime.now().isoformat()
        enriched["request_id"] = enriched.get("request_id", str(uuid.uuid4()))

        # Añadir información de riesgo
        enriched["risk_score"] = await self._calculate_risk_score(enriched)

        # Añadir patrones históricos
        enriched["historical_patterns"] = await self._get_historical_patterns(enriched)

        return enriched

    def _get_applicable_policies(self, context: Dict[str, Any]) -> List[Policy]:
        """Obtiene políticas aplicables para el contexto"""
        cache_key = self._generate_cache_key(context)

        if cache_key in self.policy_cache:
            return self.policy_cache[cache_key]

        applicable = []

        for policy in self.policies.values():
            if policy.enabled and self._policy_applies_to_context(policy, context):
                applicable.append(policy)

        # Cachear resultado
        self.policy_cache[cache_key] = applicable

        return applicable

    def _policy_applies_to_context(
        self, policy: Policy, context: Dict[str, Any]
    ) -> bool:
        """Verifica si una política aplica al contexto"""
        # Verificar alcance
        if policy.scope == PolicyScope.GLOBAL:
            return True
        elif policy.scope == PolicyScope.AGENT:
            return context.get("agent_id") is not None
        elif policy.scope == PolicyScope.TASK:
            return context.get("task_id") is not None
        elif policy.scope == PolicyScope.RESOURCE:
            return context.get("resource_type") is not None

        return True  # Default

    def _generate_cache_key(self, context: Dict[str, Any]) -> str:
        """Genera clave de cache para el contexto"""
        key_components = [
            context.get("agent_id", ""),
            context.get("task_type", ""),
            context.get("resource_type", ""),
            str(context.get("risk_score", 0)),
        ]

        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def _calculate_risk_score(self, context: Dict[str, Any]) -> float:
        """Calcula puntuación de riesgo del contexto"""
        risk_factors = []

        # Factor de agente desconocido
        if not context.get("agent_verified", False):
            risk_factors.append(0.3)

        # Factor de tarea compleja
        task_complexity = context.get("task_complexity", 0.5)
        if task_complexity > 0.8:
            risk_factors.append(0.2)

        # Factor de recursos sensibles
        if context.get("sensitive_data", False):
            risk_factors.append(0.4)

        # Factor histórico de violaciones
        violation_history = await self._get_violation_history(context)
        if violation_history:
            risk_factors.append(min(len(violation_history) * 0.1, 0.3))

        return min(sum(risk_factors), 1.0)

    async def _get_historical_patterns(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Obtiene patrones históricos para el contexto"""
        # En implementación real, consultaría base de datos
        return {
            "similar_requests_count": 0,
            "average_processing_time": 0.0,
            "success_rate": 1.0,
        }

    async def _get_violation_history(
        self, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Obtiene historial de violaciones para el contexto"""
        # En implementación real, consultaría base de datos
        return []

    async def _record_evaluation_result(
        self,
        context: Dict[str, Any],
        evaluation_results: List[Dict[str, Any]],
        decision: str,
    ):
        """Registra resultado de evaluación"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "request_id": context.get("request_id"),
            "decision": decision,
            "risk_score": context.get("risk_score", 0),
            "policies_triggered": len(evaluation_results),
            "evaluation_results": evaluation_results,
        }

        self.violation_history.append(record)

        # Mantener solo últimas 1000 entradas
        if len(self.violation_history) > 1000:
            self.violation_history = self.violation_history[-1000:]

    def _increment_version(self, version: str) -> str:
        """Incrementa versión semántica"""
        parts = version.split(".")
        if len(parts) >= 3:
            parts[2] = str(int(parts[2]) + 1)
        return ".".join(parts)

    async def _load_default_policies(self):
        """Carga políticas por defecto"""
        default_policies = [
            # Política de seguridad básica
            Policy(
                name="Basic Security Policy",
                description="Política básica de seguridad para todas las solicitudes",
                policy_type=PolicyType.SECURITY,
                scope=PolicyScope.GLOBAL,
                priority=PolicyPriority.HIGH,
                conditions=[
                    PolicyCondition(
                        field="agent_verified", operator="equals", value=True
                    )
                ],
                actions=[
                    PolicyAction(
                        action_type="log",
                        parameters={
                            "level": "info",
                            "message": "Request from verified agent",
                        },
                    )
                ],
                effect=PolicyEffect.ALLOW,
            ),
            # Política de riesgo alto
            Policy(
                name="High Risk Policy",
                description="Bloquea solicitudes de alto riesgo",
                policy_type=PolicyType.SECURITY,
                scope=PolicyScope.GLOBAL,
                priority=PolicyPriority.CRITICAL,
                conditions=[
                    PolicyCondition(field="risk_score", operator="gt", value=0.8)
                ],
                actions=[
                    PolicyAction(
                        action_type="alert",
                        parameters={
                            "severity": "high",
                            "message": "High risk request blocked",
                        },
                    ),
                    PolicyAction(
                        action_type="block", parameters={"reason": "High risk score"}
                    ),
                ],
                effect=PolicyEffect.DENY,
            ),
            # Política ética
            Policy(
                name="Ethical AI Policy",
                description="Asegura uso ético de la IA",
                policy_type=PolicyType.ETHICAL,
                scope=PolicyScope.GLOBAL,
                priority=PolicyPriority.HIGH,
                conditions=[
                    PolicyCondition(field="ethical_score", operator="lt", value=0.7)
                ],
                actions=[
                    PolicyAction(
                        action_type="warn",
                        parameters={"message": "Low ethical score detected"},
                    ),
                    PolicyAction(
                        action_type="log",
                        parameters={"level": "warning", "message": "Ethical concern"},
                    ),
                ],
                effect=PolicyEffect.WARN,
            ),
        ]

        for policy in default_policies:
            await self.add_policy(policy)


# =============================================================================
# MOTOR DE APRENDIZAJE DE POLÍTICAS
# =============================================================================


class PolicyLearningEngine:
    """Motor de aprendizaje para políticas"""

    def __init__(self):
        self.policy_performance: Dict[str, Dict[str, Any]] = {}
        self.learning_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.adaptation_rules: Dict[str, Callable] = {}

    async def initialize(self):
        """Inicializa el motor de aprendizaje"""
        # Configurar reglas de adaptación
        self.adaptation_rules = {
            "violation_threshold": self._adapt_violation_threshold,
            "risk_tolerance": self._adapt_risk_tolerance,
            "performance_optimization": self._adapt_performance_optimization,
        }

        logger.info("Policy Learning Engine initialized")

    async def learn_from_evaluation(self, evaluation_result: Dict[str, Any]):
        """Aprende de un resultado de evaluación"""
        # Analizar patrones
        await self._analyze_patterns(evaluation_result)

        # Actualizar métricas de performance
        await self._update_performance_metrics(evaluation_result)

        # Aplicar adaptaciones si es necesario
        await self._apply_adaptations(evaluation_result)

    async def register_policy(self, policy: Policy):
        """Registra una política para aprendizaje"""
        self.policy_performance[policy.policy_id] = {
            "evaluations": 0,
            "violations": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "avg_evaluation_time": 0.0,
            "last_updated": datetime.now(),
        }

    async def update_policy(self, policy: Policy):
        """Actualiza información de aprendizaje de política"""
        if policy.policy_id in self.policy_performance:
            self.policy_performance[policy.policy_id]["last_updated"] = datetime.now()

    async def remove_policy(self, policy_id: str):
        """Remueve política del aprendizaje"""
        if policy_id in self.policy_performance:
            del self.policy_performance[policy_id]

    async def _analyze_patterns(self, evaluation_result: Dict[str, Any]):
        """Analiza patrones en resultados de evaluación"""
        context = evaluation_result.get("enriched_context", {})
        decision = evaluation_result.get("decision", "allow")

        # Crear patrón
        pattern = {
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "risk_score": context.get("risk_score", 0),
            "agent_type": context.get("agent_type", "unknown"),
            "task_type": context.get("task_type", "unknown"),
            "policies_triggered": evaluation_result.get("policies_triggered", 0),
        }

        # Categorizar patrón
        pattern_key = f"{decision}_{context.get('agent_type', 'unknown')}"

        if pattern_key not in self.learning_patterns:
            self.learning_patterns[pattern_key] = []

        self.learning_patterns[pattern_key].append(pattern)

        # Mantener solo últimos 100 patrones por categoría
        if len(self.learning_patterns[pattern_key]) > 100:
            self.learning_patterns[pattern_key] = self.learning_patterns[pattern_key][
                -100:
            ]

    async def _update_performance_metrics(self, evaluation_result: Dict[str, Any]):
        """Actualiza métricas de performance"""
        evaluation_time = evaluation_result.get("evaluation_time_seconds", 0)
        policies_evaluated = evaluation_result.get("policies_evaluated", 0)

        # Actualizar métricas globales
        if "global" not in self.policy_performance:
            self.policy_performance["global"] = {
                "total_evaluations": 0,
                "total_evaluation_time": 0.0,
                "avg_evaluation_time": 0.0,
                "avg_policies_evaluated": 0.0,
            }

        global_metrics = self.policy_performance["global"]
        global_metrics["total_evaluations"] += 1
        global_metrics["total_evaluation_time"] += evaluation_time
        global_metrics["avg_evaluation_time"] = (
            global_metrics["total_evaluation_time"]
            / global_metrics["total_evaluations"]
        )
        global_metrics["avg_policies_evaluated"] = (
            (
                global_metrics["avg_policies_evaluated"]
                * (global_metrics["total_evaluations"] - 1)
            )
            + policies_evaluated
        ) / global_metrics["total_evaluations"]

    async def _apply_adaptations(self, evaluation_result: Dict[str, Any]):
        """Aplica adaptaciones basadas en aprendizaje"""
        # Verificar si se necesitan adaptaciones
        global_metrics = self.policy_performance.get("global", {})

        # Adaptar si el tiempo de evaluación es muy alto
        if global_metrics.get("avg_evaluation_time", 0) > 1.0:  # Más de 1 segundo
            await self._adapt_performance_optimization()

        # Adaptar si hay muchas violaciones
        violation_rate = await self._calculate_violation_rate()
        if violation_rate > 0.1:  # Más del 10%
            await self._adapt_violation_threshold()

    async def _calculate_violation_rate(self) -> float:
        """Calcula tasa de violaciones"""
        # En implementación real, calcularía basado en historial
        return 0.05  # 5% placeholder

    async def _adapt_violation_threshold(self):
        """Adapta umbral de violaciones"""
        logger.info("Adapting violation threshold based on learning patterns")

        # Lógica de adaptación (simplificada)
        # En producción, ajustaría parámetros de políticas basado en patrones

    async def _adapt_risk_tolerance(self):
        """Adapta tolerancia al riesgo"""
        logger.info("Adapting risk tolerance based on learning patterns")

    async def _adapt_performance_optimization(self):
        """Adapta optimización de performance"""
        logger.info("Adapting performance optimization based on learning patterns")

        # Podría reordenar políticas, optimizar cache, etc.


# =============================================================================
# REGISTRO DE GOBERNANZA DE AGENTES
# =============================================================================


class AgentGovernanceRegistry:
    """Registro de gobernanza para agentes"""

    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.agent_lineage: Dict[str, List[str]] = {}
        self.governance_policies: Dict[str, Dict[str, Any]] = {}
        self.audit_trail: List[Dict[str, Any]] = []

    async def register_agent(self, agent_info: Dict[str, Any]) -> str:
        """Registra un nuevo agente en el registro de gobernanza"""
        agent_id = agent_info.get("agent_id", str(uuid.uuid4()))

        # Información completa del agente
        governance_record = {
            "agent_id": agent_id,
            "name": agent_info.get("name", ""),
            "type": agent_info.get("type", "unknown"),
            "capabilities": agent_info.get("capabilities", []),
            "permissions": agent_info.get("permissions", []),
            "trust_score": agent_info.get("trust_score", 0.5),
            "risk_level": agent_info.get("risk_level", "medium"),
            "created_by": agent_info.get("created_by", "system"),
            "parent_agent": agent_info.get("parent_agent"),
            "creation_timestamp": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "status": "active",
            "compliance_status": "pending_review",
            "audit_required": True,
        }

        self.agents[agent_id] = governance_record

        # Registrar linaje si tiene padre
        if governance_record["parent_agent"]:
            if governance_record["parent_agent"] not in self.agent_lineage:
                self.agent_lineage[governance_record["parent_agent"]] = []
            self.agent_lineage[governance_record["parent_agent"]].append(agent_id)

        # Registrar en audit trail
        await self._audit_event(
            "agent_registered", {"agent_id": agent_id, "agent_info": governance_record}
        )

        logger.info(f"Registered agent in governance registry: {agent_id}")
        return agent_id

    async def update_agent_status(self, agent_id: str, status: str, reason: str = ""):
        """Actualiza el estado de un agente"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found in registry")

        old_status = self.agents[agent_id]["status"]
        self.agents[agent_id]["status"] = status
        self.agents[agent_id]["last_activity"] = datetime.now().isoformat()

        # Registrar cambio
        await self._audit_event(
            "agent_status_changed",
            {
                "agent_id": agent_id,
                "old_status": old_status,
                "new_status": status,
                "reason": reason,
            },
        )

        logger.info(f"Updated agent {agent_id} status: {old_status} -> {status}")

    async def evaluate_agent_compliance(self, agent_id: str) -> Dict[str, Any]:
        """Evalúa el cumplimiento de un agente"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found in registry")

        agent = self.agents[agent_id]

        compliance_result = {
            "agent_id": agent_id,
            "overall_compliance": "compliant",
            "checks": {},
            "violations": [],
            "recommendations": [],
        }

        # Verificar trust score
        if agent["trust_score"] < 0.7:
            compliance_result["checks"]["trust_score"] = "failed"
            compliance_result["violations"].append("Low trust score")
            compliance_result["recommendations"].append(
                "Improve trust score through positive interactions"
            )

        # Verificar permisos
        if not agent["permissions"]:
            compliance_result["checks"]["permissions"] = "failed"
            compliance_result["violations"].append("No permissions assigned")
            compliance_result["recommendations"].append(
                "Assign appropriate permissions"
            )

        # Verificar actividad reciente
        last_activity = datetime.fromisoformat(agent["last_activity"])
        if (datetime.now() - last_activity) > timedelta(days=30):
            compliance_result["checks"]["activity"] = "warning"
            compliance_result["recommendations"].append(
                "Agent has been inactive for 30+ days"
            )

        # Determinar cumplimiento general
        if compliance_result["violations"]:
            compliance_result["overall_compliance"] = "non_compliant"
        elif any(check == "warning" for check in compliance_result["checks"].values()):
            compliance_result["overall_compliance"] = "warning"

        # Actualizar registro
        agent["compliance_status"] = compliance_result["overall_compliance"]
        agent["last_compliance_check"] = datetime.now().isoformat()

        # Registrar evaluación
        await self._audit_event(
            "compliance_evaluated",
            {"agent_id": agent_id, "compliance_result": compliance_result},
        )

        return compliance_result

    async def get_agent_lineage(self, agent_id: str) -> Dict[str, Any]:
        """Obtiene el linaje completo de un agente"""
        if agent_id not in self.agents:
            return {"error": "Agent not found"}

        lineage_info = {
            "agent_id": agent_id,
            "ancestors": [],
            "descendants": self.agent_lineage.get(agent_id, []),
            "generation": 0,
        }

        # Encontrar ancestros
        current = agent_id
        generation = 0

        while current and generation < 10:  # Evitar loops infinitos
            agent = self.agents.get(current)
            if agent and agent.get("parent_agent"):
                lineage_info["ancestors"].append(agent["parent_agent"])
                current = agent["parent_agent"]
                generation += 1
            else:
                break

        lineage_info["generation"] = generation
        return lineage_info

    async def audit_agent_activity(self, agent_id: str, activity: Dict[str, Any]):
        """Audita actividad de un agente"""
        if agent_id not in self.agents:
            logger.warning(f"Activity audit for unknown agent: {agent_id}")
            return

        # Actualizar última actividad
        self.agents[agent_id]["last_activity"] = datetime.now().isoformat()

        # Evaluar actividad contra políticas
        risk_assessment = await self._assess_activity_risk(activity)

        # Registrar en audit trail
        await self._audit_event(
            "agent_activity",
            {
                "agent_id": agent_id,
                "activity": activity,
                "risk_assessment": risk_assessment,
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Verificar si requiere intervención
        if risk_assessment["risk_level"] == "high":
            await self._handle_high_risk_activity(agent_id, activity, risk_assessment)

    async def _assess_activity_risk(self, activity: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa el riesgo de una actividad"""
        risk_score = 0.0
        risk_factors = []

        # Factor de tipo de actividad
        activity_type = activity.get("type", "")
        if activity_type in ["delete", "modify_system", "access_sensitive"]:
            risk_score += 0.4
            risk_factors.append(f"High-risk activity type: {activity_type}")

        # Factor de recursos involucrados
        if activity.get("sensitive_data", False):
            risk_score += 0.3
            risk_factors.append("Sensitive data involved")

        # Factor de frecuencia
        # En implementación real, verificar frecuencia histórica

        # Determinar nivel de riesgo
        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
        }

    async def _handle_high_risk_activity(
        self, agent_id: str, activity: Dict[str, Any], risk_assessment: Dict[str, Any]
    ):
        """Maneja actividad de alto riesgo"""
        logger.warning(f"High-risk activity detected for agent {agent_id}: {activity}")

        # Podría suspender al agente, enviar alertas, etc.
        # Por ahora, solo registrar
        await self._audit_event(
            "high_risk_activity_handled",
            {
                "agent_id": agent_id,
                "activity": activity,
                "risk_assessment": risk_assessment,
                "action_taken": "logged_and_monitored",
            },
        )

    async def _audit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Registra evento en audit trail"""
        audit_entry = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": event_data,
        }

        self.audit_trail.append(audit_entry)

        # Mantener solo últimas 5000 entradas
        if len(self.audit_trail) > 5000:
            self.audit_trail = self.audit_trail[-5000:]

    def get_agents_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Obtiene agentes por estado"""
        return [agent for agent in self.agents.values() if agent["status"] == status]

    def get_agents_by_risk_level(self, risk_level: str) -> List[Dict[str, Any]]:
        """Obtiene agentes por nivel de riesgo"""
        return [
            agent for agent in self.agents.values() if agent["risk_level"] == risk_level
        ]

    def get_audit_trail(
        self, agent_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtiene audit trail con filtros opcionales"""
        trail = self.audit_trail

        if agent_id:
            trail = [
                entry for entry in trail if entry["data"].get("agent_id") == agent_id
            ]

        return trail[-limit:]  # Últimas N entradas


# =============================================================================
# INTEGRACIÓN CON SISTEMA EXISTENTE
# =============================================================================

# Instancias globales
policy_engine = IntelligentPolicyEngine()
governance_registry = AgentGovernanceRegistry()


async def initialize_advanced_policy_system():
    """Inicializa el sistema avanzado de políticas"""
    await policy_engine.initialize_engine()
    logger.info("Advanced Policy System initialized")


async def evaluate_request_with_policies(
    request_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Evalúa una solicitud con el sistema de políticas"""
    return await policy_engine.evaluate_request(request_context)


async def register_agent_governance(agent_info: Dict[str, Any]) -> str:
    """Registra agente en sistema de gobernanza"""
    return await governance_registry.register_agent(agent_info)


# Funciones de utilidad
__all__ = [
    # Clases principales
    "IntelligentPolicyEngine",
    "AgentGovernanceRegistry",
    "Policy",
    "PolicyCondition",
    "PolicyAction",
    # Enums
    "PolicyType",
    "PolicyScope",
    "PolicyEffect",
    "PolicyPriority",
    # Instancias globales
    "policy_engine",
    "governance_registry",
    # Funciones de utilidad
    "initialize_advanced_policy_system",
    "evaluate_request_with_policies",
    "register_agent_governance",
]
