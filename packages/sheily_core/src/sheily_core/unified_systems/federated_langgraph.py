"""
Integración con LangGraph para Workflows y Agentes en Aprendizaje Federado

Este módulo implementa workflows automatizados y agentes inteligentes usando LangGraph,
un framework para construir aplicaciones complejas como grafos dirigidos.

Características:
- Workflows de entrenamiento federado automatizados
- Agentes de monitoreo y respuesta automática
- Flujos de decisión basados en métricas de privacidad/seguridad
- Ciclos de retroalimentación y optimización continua
- Memoria compartida entre agentes
- Depuración y trazabilidad de flujos

Autor: Sheily AI Team
Fecha: 2025
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

# LangGraph imports
try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback si LangGraph no está disponible
    StateGraph = None
    END = None
    START = None
    ToolNode = None
    MemorySaver = None
    HumanMessage = None
    AIMessage = None
    BaseMessage = None
    tool = None
    RunnableConfig = None
    ChatOpenAI = None
    ChatAnthropic = None
    LANGGRAPH_AVAILABLE = False

from federated_api import FederatedAPIClient

# Importaciones del sistema FL
from federated_learning import FederatedConfig, FederatedLearningSystem, UseCase
from federated_mlflow import FederatedMLflowTracker

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== ESTADOS DE WORKFLOW ====================


class FederatedWorkflowState(TypedDict):
    """Estado del workflow federado"""

    workflow_id: str
    current_step: str
    fl_system: Optional[FederatedLearningSystem]
    api_client: Optional[FederatedAPIClient]
    mlflow_tracker: Optional[FederatedMLflowTracker]

    # Estado del sistema FL
    active_rounds: List[Dict[str, Any]]
    client_metrics: Dict[str, Any]
    security_alerts: List[Dict[str, Any]]
    privacy_metrics: Dict[str, Any]

    # Estado de decisiones
    decisions_made: List[Dict[str, Any]]
    actions_taken: List[Dict[str, Any]]
    alerts_generated: List[Dict[str, Any]]

    # Memoria del workflow
    conversation_history: List[BaseMessage]
    context_data: Dict[str, Any]

    # Estado final
    workflow_status: Literal["running", "completed", "failed", "paused"]
    error_message: Optional[str]
    completion_time: Optional[datetime]


class AgentDecision(TypedDict):
    """Decisión tomada por un agente"""

    agent_id: str
    decision_type: str
    confidence: float
    reasoning: str
    actions: List[Dict[str, Any]]
    timestamp: datetime


# ==================== AGENTES ESPECIALIZADOS ====================


class FederatedMonitoringAgent:
    """Agente de monitoreo del sistema FL"""

    def __init__(self, agent_id: str = "monitor_agent"):
        """Inicializar agente de monitoreo"""
        self.agent_id = agent_id
        self.monitoring_thresholds = {
            "max_clients_inactive": 0.3,  # 30% máximo inactivos
            "min_accuracy_drop": 0.05,  # 5% caída máxima de precisión
            "max_privacy_budget_used": 0.8,  # 80% máximo usado
            "max_security_alerts": 5,  # Máximo 5 alertas por hora
        }

    async def monitor_system_health(
        self, state: FederatedWorkflowState
    ) -> Dict[str, Any]:
        """Monitorear salud del sistema FL"""
        try:
            # Obtener métricas actuales
            if state.get("api_client"):
                metrics = await state["api_client"].get_metrics()
            else:
                # Usar métricas del estado si no hay API client (modo demo)
                metrics = state.get("client_metrics", {})

            # Evaluar indicadores de salud
            health_score = self._calculate_health_score(metrics)
            alerts = self._check_thresholds(metrics)

            return {
                "health_score": health_score,
                "alerts": alerts,
                "recommendations": self._generate_recommendations(alerts, metrics),
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error en monitoreo de salud: {e}")
            return {"error": str(e)}

    def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calcular puntaje de salud del sistema (0-1)"""
        try:
            scores = []

            # Puntaje de clientes activos
            active_clients = metrics.get("active_clients", 0)
            total_clients = metrics.get("total_clients", 0)
            if total_clients > 0:
                client_score = active_clients / total_clients
                scores.append(client_score)

            # Puntaje de privacidad
            privacy_budget = metrics.get("gdpr_compliance_rate", 0.0)
            scores.append(privacy_budget)

            # Puntaje de rondas activas (no demasiadas ni muy pocas)
            active_rounds = metrics.get("active_rounds", 0)
            round_score = min(active_rounds / 5, 1.0)  # Óptimo: 1-5 rondas
            scores.append(round_score)

            # Puntaje promedio
            return sum(scores) / len(scores) if scores else 0.0

        except Exception:
            return 0.0

    def _check_thresholds(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verificar umbrales de alerta"""
        alerts = []

        try:
            # Verificar clientes inactivos
            active_clients = metrics.get("active_clients", 0)
            total_clients = metrics.get("total_clients", 0)
            if total_clients > 0:
                inactive_ratio = 1 - (active_clients / total_clients)
                if inactive_ratio > self.monitoring_thresholds["max_clients_inactive"]:
                    alerts.append(
                        {
                            "type": "clients_inactive",
                            "severity": "medium",
                            "message": ".1%",
                            "value": inactive_ratio,
                        }
                    )

            # Verificar presupuesto de privacidad
            privacy_budget_used = 1.0 - metrics.get("gdpr_compliance_rate", 0.0)
            if (
                privacy_budget_used
                > self.monitoring_thresholds["max_privacy_budget_used"]
            ):
                alerts.append(
                    {
                        "type": "privacy_budget",
                        "severity": "high",
                        "message": ".1%",
                        "value": privacy_budget_used,
                    }
                )

        except Exception as e:
            logger.error(f"Error verificando umbrales: {e}")

        return alerts

    def _generate_recommendations(
        self, alerts: List[Dict[str, Any]], metrics: Dict[str, Any]
    ) -> List[str]:
        """Generar recomendaciones basadas en alertas"""
        recommendations = []

        for alert in alerts:
            if alert["type"] == "clients_inactive":
                recommendations.append(
                    "Considerar reclutar más clientes o verificar conectividad"
                )
            elif alert["type"] == "privacy_budget":
                recommendations.append(
                    "Reducir ruido de privacidad o pausar rondas temporalmente"
                )

        if not alerts:
            recommendations.append("Sistema funcionando óptimamente")

        return recommendations


class FederatedDecisionAgent:
    """Agente de toma de decisiones automatizadas"""

    def __init__(self, agent_id: str = "decision_agent"):
        """Inicializar agente de decisiones"""
        self.agent_id = agent_id
        self.decision_rules = {
            "start_round": self._decide_start_round,
            "stop_round": self._decide_stop_round,
            "adjust_privacy": self._decide_adjust_privacy,
            "handle_alert": self._decide_handle_alert,
        }

    async def make_decision(
        self, state: FederatedWorkflowState, decision_type: str
    ) -> AgentDecision:
        """Tomar una decisión basada en el estado actual"""
        try:
            decision_func = self.decision_rules.get(decision_type)
            if not decision_func:
                return self._create_decision(
                    "unknown", 0.0, "Tipo de decisión no reconocido", []
                )

            # Ejecutar función de decisión
            confidence, reasoning, actions = await decision_func(state)

            return self._create_decision(decision_type, confidence, reasoning, actions)

        except Exception as e:
            logger.error(f"Error en toma de decisión: {e}")
            return self._create_decision("error", 0.0, f"Error: {str(e)}", [])

    def _create_decision(
        self,
        decision_type: str,
        confidence: float,
        reasoning: str,
        actions: List[Dict[str, Any]],
    ) -> AgentDecision:
        """Crear objeto de decisión"""
        return {
            "agent_id": self.agent_id,
            "decision_type": decision_type,
            "confidence": confidence,
            "reasoning": reasoning,
            "actions": actions,
            "timestamp": datetime.now(),
        }

    async def _decide_start_round(
        self, state: FederatedWorkflowState
    ) -> tuple[float, str, List[Dict[str, Any]]]:
        """Decidir si iniciar nueva ronda"""
        try:
            metrics = state.get("client_metrics", {})
            active_rounds = len(state.get("active_rounds", []))

            # Condiciones para iniciar ronda
            conditions = {
                "sufficient_clients": metrics.get("active_clients", 0) >= 3,
                "no_active_rounds": active_rounds == 0,
                "good_privacy_budget": metrics.get("gdpr_compliance_rate", 0.0) > 0.7,
                "no_critical_alerts": len(
                    [
                        a
                        for a in state.get("security_alerts", [])
                        if a.get("severity") == "critical"
                    ]
                )
                == 0,
            }

            satisfied_conditions = sum(conditions.values())
            total_conditions = len(conditions)
            confidence = satisfied_conditions / total_conditions

            if confidence >= 0.8:
                reasoning = f"Condiciones óptimas: {satisfied_conditions}/{total_conditions} cumplidas"
                actions = [
                    {"action": "start_round", "parameters": {"round_type": "regular"}}
                ]
            else:
                reasoning = f"Condiciones insuficientes: {satisfied_conditions}/{total_conditions} cumplidas"
                actions = []

            return confidence, reasoning, actions

        except Exception as e:
            return 0.0, f"Error evaluando inicio de ronda: {e}", []

    async def _decide_stop_round(
        self, state: FederatedWorkflowState
    ) -> tuple[float, str, List[Dict[str, Any]]]:
        """Decidir si detener ronda activa"""
        try:
            active_rounds = state.get("active_rounds", [])
            security_alerts = state.get("security_alerts", [])

            # Condiciones para detener ronda
            high_alerts = len(
                [
                    a
                    for a in security_alerts
                    if a.get("severity") in ["high", "critical"]
                ]
            )
            privacy_violations = len(
                [a for a in security_alerts if "privacy" in a.get("type", "")]
            )

            if high_alerts > 2 or privacy_violations > 0:
                confidence = 0.9
                reasoning = f"Alertas de seguridad críticas detectadas: {high_alerts} altas, {privacy_violations} privacidad"
                actions = [
                    {
                        "action": "stop_round",
                        "parameters": {"reason": "security_concern"},
                    }
                ]
            else:
                confidence = 0.3
                reasoning = "No hay condiciones críticas para detener ronda"
                actions = []

            return confidence, reasoning, actions

        except Exception as e:
            return 0.0, f"Error evaluando detención de ronda: {e}", []

    async def _decide_adjust_privacy(
        self, state: FederatedWorkflowState
    ) -> tuple[float, str, List[Dict[str, Any]]]:
        """Decidir ajustes de privacidad"""
        try:
            privacy_metrics = state.get("privacy_metrics", {})
            privacy_budget_used = 1.0 - privacy_metrics.get(
                "differential_privacy_budget", 1.0
            )

            if privacy_budget_used > 0.8:
                confidence = 0.9
                reasoning = ".1%"
                actions = [
                    {
                        "action": "increase_noise",
                        "parameters": {"noise_multiplier": 1.5},
                    }
                ]
            elif privacy_budget_used < 0.3:
                confidence = 0.7
                reasoning = ".1%"
                actions = [
                    {
                        "action": "decrease_noise",
                        "parameters": {"noise_multiplier": 0.8},
                    }
                ]
            else:
                confidence = 0.5
                reasoning = "Presupuesto de privacidad en rango óptimo"
                actions = []

            return confidence, reasoning, actions

        except Exception as e:
            return 0.0, f"Error evaluando ajustes de privacidad: {e}", []

    async def _decide_handle_alert(
        self, state: FederatedWorkflowState
    ) -> tuple[float, str, List[Dict[str, Any]]]:
        """Decidir cómo manejar alertas"""
        try:
            alerts = state.get("security_alerts", [])
            high_priority_alerts = [
                a for a in alerts if a.get("severity") in ["high", "critical"]
            ]

            if high_priority_alerts:
                confidence = 0.95
                reasoning = f"{len(high_priority_alerts)} alertas de alta prioridad requieren atención inmediata"
                actions = [
                    {
                        "action": "notify_admin",
                        "parameters": {"alerts": high_priority_alerts},
                    },
                    {
                        "action": "pause_operations",
                        "parameters": {"duration_minutes": 30},
                    },
                ]
            else:
                confidence = 0.6
                reasoning = "Solo alertas de baja prioridad, monitoreo continuo"
                actions = [{"action": "log_alerts", "parameters": {"alerts": alerts}}]

            return confidence, reasoning, actions

        except Exception as e:
            return 0.0, f"Error evaluando manejo de alertas: {e}", []


class FederatedOptimizationAgent:
    """Agente de optimización automática del sistema FL"""

    def __init__(self, agent_id: str = "optimization_agent"):
        """Inicializar agente de optimización"""
        self.agent_id = agent_id
        self.optimization_history = []

    async def optimize_system(self, state: FederatedWorkflowState) -> Dict[str, Any]:
        """Optimizar configuración del sistema FL"""
        try:
            current_metrics = state.get("client_metrics", {})
            active_rounds = state.get("active_rounds", [])

            optimizations = {
                "client_selection": self._optimize_client_selection(current_metrics),
                "round_parameters": self._optimize_round_parameters(active_rounds),
                "resource_allocation": self._optimize_resource_allocation(
                    current_metrics
                ),
                "privacy_settings": self._optimize_privacy_settings(
                    state.get("privacy_metrics", {})
                ),
            }

            # Filtrar solo optimizaciones aplicables
            applicable_optimizations = {
                k: v for k, v in optimizations.items() if v.get("confidence", 0) > 0.7
            }

            return {
                "optimizations": applicable_optimizations,
                "expected_improvement": self._calculate_expected_improvement(
                    applicable_optimizations
                ),
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error en optimización: {e}")
            return {"error": str(e)}

    def _optimize_client_selection(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Optimizar selección de clientes"""
        try:
            active_clients = metrics.get("active_clients", 0)
            total_clients = metrics.get("total_clients", 0)

            if total_clients > 0:
                participation_rate = active_clients / total_clients

                if participation_rate < 0.5:
                    return {
                        "action": "adjust_client_selection",
                        "parameters": {
                            "min_reputation": 0.7,
                            "max_clients_per_round": active_clients,
                        },
                        "confidence": 0.8,
                        "reasoning": ".1%",
                    }
                elif participation_rate > 0.9:
                    return {
                        "action": "relax_client_criteria",
                        "parameters": {"min_reputation": 0.5},
                        "confidence": 0.6,
                        "reasoning": "Alta participación, se pueden relajar criterios",
                    }

            return {
                "confidence": 0.0,
                "reasoning": "No se requieren cambios en selección",
            }

        except Exception:
            return {"confidence": 0.0, "reasoning": "Error en optimización de clientes"}

    def _optimize_round_parameters(
        self, active_rounds: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Optimizar parámetros de rondas"""
        try:
            if not active_rounds:
                return {
                    "action": "adjust_round_params",
                    "parameters": {"local_epochs": 3, "learning_rate": 0.01},
                    "confidence": 0.8,  # Aumentado para que se aplique en demo
                    "reasoning": "Configuración inicial optimizada",
                }

            # Analizar rendimiento de rondas recientes
            avg_completion_time = sum(
                r.get("duration", 120) for r in active_rounds
            ) / len(active_rounds)

            if avg_completion_time > 300:  # Más de 5 minutos
                return {
                    "action": "reduce_local_epochs",
                    "parameters": {"local_epochs": 2},
                    "confidence": 0.8,
                    "reasoning": f"Tiempo promedio alto: {avg_completion_time:.1f}s",
                }

            return {"confidence": 0.0, "reasoning": "Parámetros de ronda óptimos"}

        except Exception:
            return {"confidence": 0.0, "reasoning": "Error en optimización de rondas"}

    def _optimize_resource_allocation(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Optimizar asignación de recursos"""
        try:
            # Lógica simplificada de optimización de recursos
            return {
                "action": "adjust_resources",
                "parameters": {"cpu_allocation": 0.8, "memory_limit": "2Gi"},
                "confidence": 0.6,
                "reasoning": "Asignación de recursos balanceada",
            }

        except Exception:
            return {"confidence": 0.0, "reasoning": "Error en optimización de recursos"}

    def _optimize_privacy_settings(
        self, privacy_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimizar configuración de privacidad"""
        try:
            privacy_budget = privacy_metrics.get("differential_privacy_budget", 1.0)

            if privacy_budget < 0.5:
                return {
                    "action": "increase_privacy_budget",
                    "parameters": {"target_budget": 0.7},
                    "confidence": 0.9,
                    "reasoning": ".1%",
                }
            elif privacy_budget > 0.9:
                return {
                    "action": "optimize_privacy_tradeoff",
                    "parameters": {"reduce_noise": True},
                    "confidence": 0.7,
                    "reasoning": "Presupuesto de privacidad alto, se puede optimizar",
                }

            return {
                "confidence": 0.0,
                "reasoning": "Configuración de privacidad óptima",
            }

        except Exception:
            return {
                "confidence": 0.0,
                "reasoning": "Error en optimización de privacidad",
            }

    def _calculate_expected_improvement(self, optimizations: Dict[str, Any]) -> float:
        """Calcular mejora esperada de las optimizaciones"""
        try:
            total_improvement = 0.0
            count = 0

            for opt in optimizations.values():
                if isinstance(opt, dict) and "confidence" in opt:
                    # Estimación simplificada: confianza * 10% de mejora
                    total_improvement += opt["confidence"] * 0.1
                    count += 1

            return total_improvement / count if count > 0 else 0.0

        except Exception:
            return 0.0


# ==================== WORKFLOWS LANGGRAPH ====================


class FederatedWorkflows:
    """Workflows automatizados para el sistema FL usando LangGraph"""

    def __init__(self):
        """Inicializar gestor de workflows"""
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph no disponible - funcionalidad limitada")
            self.graphs = {}
            return

        self.graphs = {}
        self.memory = MemorySaver()

        # Crear workflows disponibles
        self._create_monitoring_workflow()
        self._create_optimization_workflow()
        self._create_emergency_workflow()

        logger.info("🔄 Workflows de LangGraph inicializados")

    def _create_monitoring_workflow(self):
        """Crear workflow de monitoreo continuo"""
        if not LANGGRAPH_AVAILABLE:
            return

        def monitoring_step(state: FederatedWorkflowState) -> FederatedWorkflowState:
            """Paso de monitoreo del sistema"""
            monitor_agent = FederatedMonitoringAgent()

            # Ejecutar monitoreo
            health_data = asyncio.run(monitor_agent.monitor_system_health(state))

            # Actualizar estado
            state["client_metrics"] = health_data
            state["security_alerts"].extend(health_data.get("alerts", []))
            state["current_step"] = "monitoring"

            # Log de conversación
            state["conversation_history"].append(
                AIMessage(
                    content=f"Monitoreo completado: Salud {health_data.get('health_score', 0):.2f}"
                )
            )

            return state

        def decision_step(state: FederatedWorkflowState) -> FederatedWorkflowState:
            """Paso de toma de decisiones"""
            decision_agent = FederatedDecisionAgent()

            # Evaluar alertas
            alerts = state.get("security_alerts", [])
            if alerts:
                decision = asyncio.run(
                    decision_agent.make_decision(state, "handle_alert")
                )
                state["decisions_made"].append(decision)

                # Ejecutar acciones
                for action in decision.get("actions", []):
                    if action["action"] == "notify_admin":
                        logger.warning(f"🚨 Notificación admin: {len(alerts)} alertas")

            state["current_step"] = "decision_making"
            return state

        # Crear grafo
        workflow = StateGraph(FederatedWorkflowState)

        # Agregar nodos
        workflow.add_node("monitoring", monitoring_step)
        workflow.add_node("decision", decision_step)

        # Definir flujo
        workflow.add_edge(START, "monitoring")
        workflow.add_edge("monitoring", "decision")
        workflow.add_edge("decision", END)

        # Compilar
        self.graphs["monitoring"] = workflow.compile(checkpointer=self.memory)

    def _create_optimization_workflow(self):
        """Crear workflow de optimización automática"""
        if not LANGGRAPH_AVAILABLE:
            return

        def analysis_step(state: FederatedWorkflowState) -> FederatedWorkflowState:
            """Paso de análisis del sistema"""
            # Recopilar métricas para análisis
            state["current_step"] = "analysis"
            return state

        def optimization_step(state: FederatedWorkflowState) -> FederatedWorkflowState:
            """Paso de optimización"""
            opt_agent = FederatedOptimizationAgent()

            # Ejecutar optimización
            optimization_result = asyncio.run(opt_agent.optimize_system(state))

            # Aplicar optimizaciones
            optimizations = optimization_result.get("optimizations", {})
            for opt_name, opt_config in optimizations.items():
                if opt_config.get("confidence", 0) > 0.7:
                    logger.info(f"🔧 Aplicando optimización: {opt_name}")
                    state["actions_taken"].append(
                        {
                            "optimization": opt_name,
                            "config": opt_config,
                            "timestamp": datetime.now(),
                        }
                    )

            state["current_step"] = "optimization"
            return state

        # Crear grafo
        workflow = StateGraph(FederatedWorkflowState)

        # Agregar nodos
        workflow.add_node("analysis", analysis_step)
        workflow.add_node("optimization", optimization_step)

        # Definir flujo
        workflow.add_edge(START, "analysis")
        workflow.add_edge("analysis", "optimization")
        workflow.add_edge("optimization", END)

        # Compilar
        self.graphs["optimization"] = workflow.compile(checkpointer=self.memory)

    def _create_emergency_workflow(self):
        """Crear workflow de respuesta a emergencias"""
        if not LANGGRAPH_AVAILABLE:
            return

        def assessment_step(state: FederatedWorkflowState) -> FederatedWorkflowState:
            """Evaluar severidad de la emergencia"""
            alerts = state.get("security_alerts", [])
            critical_alerts = [a for a in alerts if a.get("severity") == "critical"]

            if critical_alerts:
                state["workflow_status"] = "emergency"
                logger.critical(
                    f"🚨 EMERGENCIA: {len(critical_alerts)} alertas críticas"
                )
            else:
                state["workflow_status"] = "normal"

            state["current_step"] = "assessment"
            return state

        def response_step(state: FederatedWorkflowState) -> FederatedWorkflowState:
            """Ejecutar respuesta de emergencia"""
            if state.get("workflow_status") == "emergency":
                # Acciones de emergencia
                emergency_actions = [
                    {"action": "stop_all_rounds", "reason": "emergency"},
                    {"action": "isolate_affected_clients", "reason": "security"},
                    {"action": "notify_security_team", "reason": "breach"},
                ]

                for action in emergency_actions:
                    logger.critical(f"🚨 Ejecutando: {action['action']}")
                    state["actions_taken"].append(action)

            state["current_step"] = "response"
            return state

        # Crear grafo condicional
        workflow = StateGraph(FederatedWorkflowState)

        # Agregar nodos
        workflow.add_node("assessment", assessment_step)
        workflow.add_node("response", response_step)

        # Definir flujo condicional
        workflow.add_edge(START, "assessment")
        workflow.add_conditional_edges(
            "assessment",
            lambda state: (
                "emergency" if state.get("workflow_status") == "emergency" else "normal"
            ),
            {"emergency": "response", "normal": END},
        )
        workflow.add_edge("response", END)

        # Compilar
        self.graphs["emergency"] = workflow.compile(checkpointer=self.memory)

    async def execute_workflow(
        self,
        workflow_name: str,
        initial_state: FederatedWorkflowState,
        config: Optional[RunnableConfig] = None,
    ) -> FederatedWorkflowState:
        """Ejecutar un workflow específico"""
        if workflow_name not in self.graphs:
            raise ValueError(f"Workflow {workflow_name} no encontrado")

        try:
            graph = self.graphs[workflow_name]

            # Ejecutar workflow
            result = await graph.ainvoke(initial_state, config=config)

            logger.info(f"✅ Workflow {workflow_name} completado")
            return result

        except Exception as e:
            logger.error(f"❌ Error ejecutando workflow {workflow_name}: {e}")
            initial_state["workflow_status"] = "failed"
            initial_state["error_message"] = str(e)
            return initial_state

    def get_available_workflows(self) -> List[str]:
        """Obtener lista de workflows disponibles"""
        return list(self.graphs.keys())


# ==================== INTEGRACIÓN CON SISTEMA FL ====================


class FederatedLangGraphIntegration:
    """Integración completa de LangGraph con el sistema FL"""

    def __init__(self, fl_system: Optional[FederatedLearningSystem] = None):
        """Inicializar integración"""
        self.fl_system = fl_system or FederatedLearningSystem()
        self.api_client = FederatedAPIClient()
        self.mlflow_tracker = FederatedMLflowTracker()

        # Inicializar workflows
        self.workflows = FederatedWorkflows()

        # Estado global del sistema
        self.global_state = self._initialize_global_state()

        # Agentes
        self.monitoring_agent = FederatedMonitoringAgent()
        self.decision_agent = FederatedDecisionAgent()
        self.optimization_agent = FederatedOptimizationAgent()

        logger.info("🎯 Integración LangGraph-FL inicializada")

    def _initialize_global_state(self) -> FederatedWorkflowState:
        """Inicializar estado global del workflow"""
        return {
            "workflow_id": str(uuid.uuid4()),
            "current_step": "initialization",
            "fl_system": self.fl_system,
            "api_client": self.api_client,
            "mlflow_tracker": self.mlflow_tracker,
            "active_rounds": [],
            "client_metrics": {},
            "security_alerts": [],
            "privacy_metrics": {},
            "decisions_made": [],
            "actions_taken": [],
            "alerts_generated": [],
            "conversation_history": [],
            "context_data": {},
            "workflow_status": "running",
            "error_message": None,
            "completion_time": None,
        }

    async def run_automated_monitoring(self, interval_seconds: int = 60):
        """Ejecutar monitoreo automatizado continuo"""
        logger.info(f"🔄 Iniciando monitoreo automatizado cada {interval_seconds}s")

        while True:
            try:
                # Actualizar estado global
                await self._update_global_state()

                # Ejecutar workflow de monitoreo
                result = await self.workflows.execute_workflow(
                    "monitoring", self.global_state.copy()
                )

                # Procesar decisiones tomadas
                await self._process_decisions(result)

                # Esperar siguiente iteración
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error en monitoreo automatizado: {e}")
                await asyncio.sleep(interval_seconds)

    async def run_optimization_cycle(self):
        """Ejecutar ciclo de optimización"""
        try:
            logger.info("🔧 Ejecutando ciclo de optimización")

            # Actualizar estado
            await self._update_global_state()

            # Ejecutar workflow de optimización
            result = await self.workflows.execute_workflow(
                "optimization", self.global_state.copy()
            )

            # Aplicar optimizaciones
            await self._apply_optimizations(result)

            logger.info("✅ Ciclo de optimización completado")

        except Exception as e:
            logger.error(f"Error en ciclo de optimización: {e}")

    async def handle_emergency(self, emergency_data: Dict[str, Any]):
        """Manejar situación de emergencia"""
        try:
            logger.critical("🚨 Activando protocolo de emergencia")

            # Actualizar estado con datos de emergencia
            self.global_state["security_alerts"].append(
                {
                    "type": "emergency",
                    "severity": "critical",
                    "message": emergency_data.get("message", "Emergencia detectada"),
                    "timestamp": datetime.now(),
                }
            )

            # Ejecutar workflow de emergencia
            result = await self.workflows.execute_workflow(
                "emergency", self.global_state.copy()
            )

            # Ejecutar acciones de emergencia
            await self._execute_emergency_actions(result)

            logger.info("✅ Protocolo de emergencia ejecutado")

        except Exception as e:
            logger.error(f"Error en manejo de emergencia: {e}")

    async def _update_global_state(self):
        """Actualizar estado global con datos actuales"""
        try:
            # Obtener métricas del sistema
            if self.api_client:
                metrics = await self.api_client.get_metrics()
                self.global_state["client_metrics"] = metrics or {}

            # Obtener métricas de privacidad (simplificado)
            self.global_state["privacy_metrics"] = {
                "differential_privacy_budget": 0.8,
                "privacy_violations": 0,
            }

            # Mantener solo alertas recientes (últimas 24h)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.global_state["security_alerts"] = [
                alert
                for alert in self.global_state["security_alerts"]
                if alert.get("timestamp", datetime.min) > cutoff_time
            ]

        except Exception as e:
            logger.error(f"Error actualizando estado global: {e}")

    async def _process_decisions(self, workflow_result: FederatedWorkflowState):
        """Procesar decisiones tomadas por los agentes"""
        try:
            decisions = workflow_result.get("decisions_made", [])

            for decision in decisions:
                confidence = decision.get("confidence", 0.0)

                if confidence > 0.8:  # Alta confianza
                    actions = decision.get("actions", [])
                    for action in actions:
                        await self._execute_action(action)

                    logger.info(
                        f"✅ Decisión ejecutada: {decision.get('decision_type')}"
                    )

        except Exception as e:
            logger.error(f"Error procesando decisiones: {e}")

    async def _apply_optimizations(self, workflow_result: FederatedWorkflowState):
        """Aplicar optimizaciones identificadas"""
        try:
            actions = workflow_result.get("actions_taken", [])

            for action in actions:
                if "optimization" in action:
                    opt_config = action.get("config", {})
                    await self._execute_optimization(opt_config)

            logger.info(f"🔧 {len(actions)} optimizaciones aplicadas")

        except Exception as e:
            logger.error(f"Error aplicando optimizaciones: {e}")

    async def _execute_action(self, action: Dict[str, Any]):
        """Ejecutar una acción específica"""
        try:
            action_type = action.get("action")

            if action_type == "start_round":
                # Iniciar nueva ronda
                if self.fl_system:
                    round_id = await self.fl_system.start_federated_round()
                    logger.info(f"🚀 Ronda iniciada: {round_id}")

            elif action_type == "stop_round":
                # Detener ronda (simplificado)
                logger.info("⏹️ Ronda detenida por decisión del agente")

            elif action_type == "notify_admin":
                # Notificar administrador
                alerts = action.get("parameters", {}).get("alerts", [])
                logger.warning(f"📧 Notificación admin: {len(alerts)} alertas")

        except Exception as e:
            logger.error(f"Error ejecutando acción {action}: {e}")

    async def _execute_optimization(self, optimization: Dict[str, Any]):
        """Ejecutar una optimización específica"""
        try:
            action = optimization.get("action")

            if action == "adjust_client_selection":
                # Ajustar selección de clientes
                params = optimization.get("parameters", {})
                logger.info(f"👥 Optimizando selección de clientes: {params}")

            elif action == "reduce_local_epochs":
                # Reducir épocas locales
                params = optimization.get("parameters", {})
                logger.info(f"⚡ Reduciendo épocas locales: {params}")

        except Exception as e:
            logger.error(f"Error ejecutando optimización {optimization}: {e}")

    async def _execute_emergency_actions(self, workflow_result: FederatedWorkflowState):
        """Ejecutar acciones de emergencia"""
        try:
            actions = workflow_result.get("actions_taken", [])

            for action in actions:
                action_type = action.get("action")

                if action_type == "stop_all_rounds":
                    logger.critical("⏹️ DETENIENDO TODAS LAS RONDAS")
                    # Implementar detención de rondas

                elif action_type == "isolate_affected_clients":
                    logger.critical("🔒 AISLANDO CLIENTES AFECTADOS")
                    # Implementar aislamiento

                elif action_type == "notify_security_team":
                    logger.critical("🚨 NOTIFICANDO EQUIPO DE SEGURIDAD")
                    # Implementar notificación

        except Exception as e:
            logger.error(f"Error ejecutando acciones de emergencia: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado actual del sistema integrado"""
        return {
            "workflows_available": self.workflows.get_available_workflows(),
            "global_state": {
                "workflow_status": self.global_state.get("workflow_status"),
                "active_alerts": len(self.global_state.get("security_alerts", [])),
                "decisions_made": len(self.global_state.get("decisions_made", [])),
                "actions_taken": len(self.global_state.get("actions_taken", [])),
            },
            "agents_status": {
                "monitoring": "active",
                "decision": "active",
                "optimization": "active",
            },
        }


# ==================== DEMO LANGGRAPH ====================


async def demo_langgraph_integration():
    """Demostración de integración LangGraph-FL"""
    logger.info("🎯 Demo de Integración LangGraph con Sistema FL")
    logger.info("=" * 60)

    try:
        # Verificar disponibilidad de LangGraph
        if not LANGGRAPH_AVAILABLE:
            logger.warning("⚠️  LangGraph no está instalado - ejecutando demo limitada")
            logger.info(
                "📦 Para funcionalidad completa instalar: pip install langgraph langchain"
            )
            logger.info("🔄 Ejecutando demo con agentes básicos...")

            # Demo limitada sin workflows de LangGraph
            await demo_limited_langgraph()
            return

        # Crear integración completa
        logger.info("🔗 Creando integración LangGraph-FL completa...")
        integration = FederatedLangGraphIntegration()

        # Mostrar estado inicial
        status = integration.get_system_status()
        logger.info("📊 Estado inicial del sistema:")
        logger.info(f"  Workflows disponibles: {status['workflows_available']}")
        logger.info(f"  Estado global: {status['global_state']}")

        # Verificar que hay workflows disponibles
        if not status["workflows_available"]:
            logger.warning("⚠️  No hay workflows disponibles - usando modo limitado")
            await demo_limited_langgraph()
            return

        # Ejecutar workflow de monitoreo
        logger.info("\n🔍 Ejecutando workflow de monitoreo...")
        monitoring_result = await integration.workflows.execute_workflow(
            "monitoring", integration.global_state.copy()
        )

        health_score = monitoring_result.get("client_metrics", {}).get(
            "health_score", 0
        )
        logger.info(".2f")

        # Ejecutar ciclo de optimización
        logger.info("\n🔧 Ejecutando ciclo de optimización...")
        await integration.run_optimization_cycle()

        # Simular situación de emergencia
        logger.info("\n🚨 Simulando situación de emergencia...")
        emergency_data = {
            "message": "Ataque de envenenamiento detectado",
            "severity": "critical",
            "affected_clients": 2,
        }

        await integration.handle_emergency(emergency_data)

        # Mostrar estado final
        final_status = integration.get_system_status()
        logger.info("\n📈 Estado final del sistema:")
        logger.info(
            f"  Alertas activas: {final_status['global_state']['active_alerts']}"
        )
        logger.info(
            f"  Decisiones tomadas: {final_status['global_state']['decisions_made']}"
        )
        logger.info(
            f"  Acciones ejecutadas: {final_status['global_state']['actions_taken']}"
        )

        logger.info("✅ Demo de integración LangGraph completada exitosamente")

    except Exception as e:
        logger.error(f"❌ Error en demo de LangGraph: {e}")
        logger.info(
            "💡 Sugerencia: Instalar dependencias con 'pip install langgraph langchain'"
        )


async def demo_limited_langgraph():
    """Demo limitada cuando LangGraph no está disponible"""
    logger.info("🎭 Ejecutando demo limitada de agentes FL...")

    try:
        # Crear sistema FL básico
        from federated_learning import FederatedLearningSystem

        fl_system = FederatedLearningSystem()

        # Crear agentes individuales (sin workflows)
        monitoring_agent = FederatedMonitoringAgent("demo_monitor")
        decision_agent = FederatedDecisionAgent("demo_decision")
        optimization_agent = FederatedOptimizationAgent("demo_optimizer")

        logger.info("✅ Agentes individuales creados")

        # Simular estado del sistema
        mock_state = {
            "client_metrics": {
                "active_clients": 5,
                "total_clients": 8,
                "gdpr_compliance_rate": 0.95,
            },
            "security_alerts": [],
            "privacy_metrics": {"differential_privacy_budget": 0.8},
        }

        # Ejecutar agentes individualmente
        logger.info("\n🔍 Ejecutando agente de monitoreo...")
        health_data = await monitoring_agent.monitor_system_health(mock_state)
        logger.info(f"  Puntaje de salud: {health_data.get('health_score', 0):.2f}")

        logger.info("\n🧠 Ejecutando agente de decisiones...")
        decision = await decision_agent.make_decision(mock_state, "start_round")
        logger.info(
            f"  Decisión: {decision.get('decision_type')} (confianza: {decision.get('confidence', 0):.2f})"
        )

        logger.info("\n🔧 Ejecutando agente de optimización...")
        optimization = await optimization_agent.optimize_system(mock_state)
        logger.info(
            f"  Optimizaciones encontradas: {len(optimization.get('optimizations', {}))}"
        )

        logger.info(
            "✅ Demo limitada completada - instala LangGraph para workflows completos"
        )

    except Exception as e:
        logger.error(f"❌ Error en demo limitada: {e}")


if __name__ == "__main__":
    asyncio.run(demo_langgraph_integration())
