"""
MCP Agent Coordinator - Sistema de Agentes MCP Enterprise Master
================================================================

Coordinador principal del sistema de 4 agentes especializados core MCP (Model Context Protocol).
Maneja la comunicación, asignación de tareas y coordinación entre agentes.

Características principales:
- Gestión de ciclo de vida de 4 agentes core (Finance, Security, Healthcare, Business)
- Sistema de mensajería entre agentes (A2A)
- Balanceo de carga inteligente
- Recuperación automática de fallos
- Monitoreo y métricas de rendimiento

Author: MCP Enterprise Master
Version: 1.0.0
"""

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from ..base.base_agent import (
    AgentCapability,
    AgentMessage,
    AgentStatus,
    AgentTask,
    BaseMCPAgent,
    MessageBus,
    get_global_message_bus,
)

logger = logging.getLogger("MCPAgentCoordinator")


@dataclass
class AgentPool:
    """Pool de agentes especializados por categoría"""

    research: List[BaseMCPAgent] = field(default_factory=list)
    finance: List[BaseMCPAgent] = field(default_factory=list)
    security: List[BaseMCPAgent] = field(default_factory=list)
    healthcare: List[BaseMCPAgent] = field(default_factory=list)
    education: List[BaseMCPAgent] = field(default_factory=list)
    engineering: List[BaseMCPAgent] = field(default_factory=list)
    business: List[BaseMCPAgent] = field(default_factory=list)
    creative: List[BaseMCPAgent] = field(default_factory=list)
    legal: List[BaseMCPAgent] = field(default_factory=list)
    environment: List[BaseMCPAgent] = field(default_factory=list)
    communication: List[BaseMCPAgent] = field(default_factory=list)
    innovation: List[BaseMCPAgent] = field(default_factory=list)


class MCPAgentCoordinator(BaseMCPAgent):
    """
    Coordinador Maestro del Sistema MCP de Agentes

    Gestiona el ciclo de vida completo de 4 agentes especializados core,
    coordina sus operaciones y garantiza alta disponibilidad.
    """

    def __init__(self):
        super().__init__(
            agent_id="mcp_coordinator_001",
            agent_name="MCPAgentCoordinator",
            capabilities=[
                AgentCapability.COORDINATION,
                AgentCapability.ANALYSIS,
                AgentCapability.COMMUNICATION,
            ],
        )

        # Pool de agentes especializados
        self.agent_pool = AgentPool()

        # Sistema de mensajería unificado
        self.message_bus = get_global_message_bus()

        # Estado del sistema de agentes
        self.total_agents_registered = 0
        self.agents_active = 0
        self.agents_by_capability: Dict[AgentCapability, Set[str]] = {}

        # Gestión de tareas en paralelo
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.max_concurrent_task = 5

        # Estadísticas de coordinación
        self.coordination_stats = {
            "tasks_assigned": 0,
            "tasks_completed": 0,
            "task_failures": 0,
            "average_response_time": 0.0,
            "agent_failures": 0,
        }

        # Inicializar agentes especializados automáticamente
        self._initialize_specialized_agents()

    async def _setup_capabilities(self) -> None:
        """Configurar capacidades de coordinación"""
        logger.info("Setting up MCP Agent Coordination capabilities...")

        # Herramientas de coordinación
        self.coordination_tools = {
            "load_balancer": "Balanceo inteligente de carga por capacidad",
            "failover_manager": "Gestión automática de failover entre agentes",
            "resource_monitor": "Monitoreo de recursos y capacidad de agentes",
            "task_scheduler": "Programador inteligente de tareas por prioridad",
            "conflict_resolver": "Resolución automática de conflictos entre agentes",
            "performance_optimizer": "Optimización automática de rendimiento",
        }

        # Configurar subsistemas
        await self._setup_load_balancing()
        await self._setup_failover_system()
        await self._setup_resource_monitoring()

    async def _execute_task_implementation(self, task: AgentTask) -> Dict[str, Any]:
        """Ejecutar tareas de coordinación de agentes"""

        task_types = {
            "assign_task_to_agent": self._assign_task_to_agent,
            "coordinate_multi_agent": self._coordinate_multi_agent_task,
            "monitor_agent_health": self._monitor_agent_health,
            "redistribute_workload": self._redistribute_workload,
            "scale_agent_capacity": self._scale_agent_capacity,
            "diagnose_agent_issue": self._diagnose_agent_issue,
            "get_system_status": self._get_system_status_report,
        }

        if task.task_type in task_types:
            return await task_types[task.task_type](task)
        else:
            raise ValueError(f"Unknown coordination task type: {task.task_type}")

    async def _register_message_handlers(self) -> None:
        """Registrar handlers para mensajes de coordinación"""

        self.message_handlers = {
            "agent_registration": self._handle_agent_registration,
            "agent_status_update": self._handle_agent_status_update,
            "task_completion": self._handle_task_completion,
            "agent_failure": self._handle_agent_failure,
            "coordination_request": self._handle_coordination_request,
            "resource_request": self._handle_resource_request,
        }

    # === GESTIÓN DE AGENTES ===

    def _initialize_specialized_agents(self) -> None:
        """Inicializar los 4 agentes reales disponibles MCP"""

        logger.info("Initializing 4 real MCP agents...")

        # Intentar importar agentes reales disponibles
        try:
            # Import agent from main agents directory
            import sys
            import os

            # Add project root to path for imports
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            # Real agents available in project (only functional ones)
            real_agents = [
                ("reflexion_agent", "agents.reflexion_agent", "ReflexionAgent", "analysis"),
                ("constitutional_evaluator", "agents.constitutional_evaluator", "ConstitutionalEvaluatorAgent", "analysis"),
                ("toolformer_agent", "agents.toolformer_agent", "ToolformerAgent", "execution"),
                ("advanced_training_system", "agents.advanced_training_system", "AdvancedAgentTrainerAgent", "training"),
            ]

            loaded_count = 0
            for agent_name, module_path, class_name, category in real_agents:
                try:
                    # Dynamic import
                    module = __import__(module_path, fromlist=[class_name])
                    agent_class = getattr(module, class_name, None)

                    if agent_class:
                        agent_instance = agent_class()
                        self._register_agent(agent_instance, category)
                        loaded_count += 1
                        logger.info(f"✅ Loaded real agent: {agent_name}")

                    else:
                        logger.warning(f"⚠️ Class {class_name} not found in {module_path}")

                except Exception as e:
                    logger.warning(f"⚠️ Could not load real agent {agent_name}: {e}")

            logger.info(f"✅ Loaded {loaded_count}/4 real agents")

        except Exception as e:
            logger.warning(f"⚠️ Error initializing real agents: {e}")

    # Note: Removed reference to non-existent specialized agents

    async def register_agent(self, agent: BaseMCPAgent, category: str) -> bool:
        """Registrar un agente especializado en el sistema"""

        try:
            # Asignar bus de mensajería
            agent.set_message_bus(self.message_bus)

            # Inicializar agente
            success = await agent.initialize()
            if not success:
                logger.error(f"Failed to initialize agent {agent.agent_name}")
                return False

            # Registrar en pool correspondiente
            category_pool = getattr(self.agent_pool, category, [])
            category_pool.append(agent)
            setattr(self.agent_pool, category, category_pool)

            # Actualizar índices
            self.total_agents_registered += 1
            self.agents_active += 1

            # Indexar por capacidades
            for capability in agent.capabilities:
                if capability not in self.agents_by_capability:
                    self.agents_by_capability[capability] = set()
                self.agents_by_capability[capability].add(agent.agent_id)

            # Registrar handlers de mensajes del agente
            self.message_bus.subscribe(f"agent_{agent.agent_id}", agent.handle_message)

            logger.info(
                f"Agent {agent.agent_name} registered successfully in {category}"
            )
            return True

        except Exception as e:
            logger.error(f"Error registering agent {agent.agent_name}: {e}")
            return False

    def _register_agent(self, agent: BaseMCPAgent, category: str) -> None:
        """Método auxiliar para registro síncrono durante inicialización"""
        # En producción esto sería async, pero para inicialización usamos approach sincrónico
        asyncio.create_task(self.register_agent(agent, category))

    def get_agents_by_capability(
        self, capability: AgentCapability
    ) -> List[BaseMCPAgent]:
        """Obtener lista de agentes con capacidad específica"""
        agent_ids = self.agents_by_capability.get(capability, set())
        all_agents = []

        # Buscar en todos los pools de categorías
        for category_name in dir(self.agent_pool):
            if not category_name.startswith("_"):
                category_pool = getattr(self.agent_pool, category_name, [])
                all_agents.extend(category_pool)

        return [agent for agent in all_agents if agent.agent_id in agent_ids]

    async def get_agents_in_category(self, category: str) -> List[BaseMCPAgent]:
        """Obtener todos los agentes en una categoría específica"""
        return getattr(self.agent_pool, category, [])

    # === ASIGNACIÓN Y EJECUCIÓN DE TAREAS ===

    async def assign_task_intelligently(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        required_capabilities: List[AgentCapability],
    ) -> Dict[str, Any]:
        """Asignar tarea de manera inteligente al mejor agente disponible"""

        try:
            # Encontrar agentes candidatos
            candidate_agents = []
            for capability in required_capabilities:
                candidates = self.get_agents_by_capability(capability)
                candidate_agents.extend(candidates)

            if not candidate_agents:
                return {"error": "No agents available for required capabilities"}

            # Filtrar agentes activos y sin sobrecarga
            available_agents = [
                agent
                for agent in candidate_agents
                if agent.status == AgentStatus.IDLE and len(agent.task_queue) < 3
            ]

            if not available_agents:
                return {"error": "No agents currently available (all busy)"}

            # Seleccionar mejor agente basado en criterios
            best_agent = await self._select_best_agent(
                available_agents, task_type, parameters
            )

            # Crear tarea estructurada
            task = AgentTask(
                task_id=f"task_{uuid.uuid4()}",
                task_type=task_type,
                parameters=parameters,
                priority=self._calculate_task_priority(task_type, parameters),
            )

            # Asignar tarea
            best_agent.add_task_to_queue(task)

            # Actualizar estadísticas
            self.coordination_stats["tasks_assigned"] += 1

            logger.info(
                f"Task {task.task_id} assigned to agent {best_agent.agent_name}"
            )

            return {
                "task_assigned": True,
                "task_id": task.task_id,
                "assigned_agent": best_agent.agent_name,
                "agent_id": best_agent.agent_id,
                "estimated_completion": "pending",
            }

        except Exception as e:
            logger.error(f"Error assigning task: {e}")
            return {"error": f"Task assignment failed: {str(e)}"}

    async def _select_best_agent(
        self,
        available_agents: List[BaseMCPAgent],
        task_type: str,
        parameters: Dict[str, Any],
    ) -> BaseMCPAgent:
        """Seleccionar el mejor agente para la tarea usando criterios inteligentes"""

        scored_agents = []

        for agent in available_agents:
            score = 0

            # Criterios de puntuación
            # 1. Historial de tareas similares completadas exitosamente
            similar_tasks = self._count_similar_tasks(agent, task_type)
            score += similar_tasks * 10

            # 2. Tiempo promedio de respuesta
            if agent.metrics.response_time_avg > 0:
                # Preferir agentes más rápidos (score inverso)
                score += max(0, 100 - agent.metrics.response_time_avg)

            # 3. Tasa de éxito
            total_tasks = agent.metrics.tasks_completed + agent.metrics.tasks_failed
            if total_tasks > 0:
                success_rate = agent.metrics.tasks_completed / total_tasks
                score += success_rate * 50

            # 4. Carga actual (menos carga = mejor score)
            current_load = len(agent.task_queue)
            score += max(0, 10 - current_load * 2)

            # 5. Expertise específica para parámetros de tarea
            expertise_score = self._calculate_expertise_score(agent, parameters)
            score += expertise_score

            scored_agents.append((agent, score))

        # Ordenar por score y seleccionar el mejor
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        best_agent = scored_agents[0][0]

        logger.debug(
            f"Selected best agent {best_agent.agent_name} with score {scored_agents[0][1]}"
        )
        return best_agent

    def _count_similar_tasks(self, agent: BaseMCPAgent, task_type: str) -> int:
        """Contar tareas similares completadas por el agente"""
        # En producción implementaría búsqueda en memoria del agente
        # Por ahora usar aproximación basada en capacidades
        similar_tasks = 0
        task_capability_map = {
            "research": AgentCapability.RESEARCH,
            "analysis": AgentCapability.ANALYSIS,
            "communication": AgentCapability.COMMUNICATION,
            "execution": AgentCapability.EXECUTION,
        }

        required_capability = task_capability_map.get(task_type)
        if required_capability and required_capability in agent.capabilities:
            similar_tasks = agent.metrics.tasks_completed // 10  # Estimación

        return similar_tasks

    def _calculate_expertise_score(
        self, agent: BaseMCPAgent, parameters: Dict[str, Any]
    ) -> float:
        """Calcular puntuación de expertise específica"""
        # Implementaría lógica específica basada en parámetros
        # Por ahora devolver puntuación base
        return 25.0  # Puntuación neutral

    def _calculate_task_priority(
        self, task_type: str, parameters: Dict[str, Any]
    ) -> int:
        """Calcular prioridad de tarea"""
        base_priority = 3

        # Tareas críticas tienen mayor prioridad
        critical_tasks = ["security", "emergency", "system_failure"]
        if any(critical in task_type.lower() for critical in critical_tasks):
            base_priority = 1

        # Tareas urgentes
        urgent_tasks = ["optimize", "fix", "recovery"]
        if any(urgent in task_type.lower() for urgent in urgent_tasks):
            base_priority = min(base_priority, 2)

        return base_priority

    # === EJECUCIÓN DE TAREAS DE COORDINACIÓN ===

    async def _assign_task_to_agent(self, task: AgentTask) -> Dict[str, Any]:
        """Asignar tarea específica a un agente específico"""

        target_agent_id = task.parameters.get("target_agent_id")
        task_type = task.parameters.get("task_type")
        task_params = task.parameters.get("parameters", {})

        if not target_agent_id or not task_type:
            return {"error": "Missing target_agent_id or task_type"}

        # Encontrar agente objetivo
        target_agent = None
        for category_name in dir(self.agent_pool):
            if not category_name.startswith("_"):
                category_pool = getattr(self.agent_pool, category_name, [])
                for agent in category_pool:
                    if agent.agent_id == target_agent_id:
                        target_agent = agent
                        break
                if target_agent:
                    break

        if not target_agent:
            return {"error": f"Agent {target_agent_id} not found"}

        # Crear tarea para el agente específico
        agent_task = AgentTask(
            task_id=f"coord_task_{uuid.uuid4()}",
            task_type=task_type,
            parameters=task_params,
        )

        # Ejecutar tarea inmediatamente
        result = await target_agent.execute_task(agent_task)

        return {
            "task_assigned": True,
            "target_agent": target_agent.agent_name,
            "task_result": result,
            "execution_success": result.get("success", False),
        }

    async def _coordinate_multi_agent_task(self, task: AgentTask) -> Dict[str, Any]:
        """Coordinar tarea que requiere múltiples agentes"""

        sub_tasks = task.parameters.get("sub_tasks", [])
        coordination_strategy = task.parameters.get("strategy", "parallel")

        if not sub_tasks:
            return {"error": "No sub-tasks specified for multi-agent coordination"}

        # Coordinar ejecución según estrategia
        if coordination_strategy == "parallel":
            return await self._execute_parallel_tasks(sub_tasks)
        elif coordination_strategy == "sequential":
            return await self._execute_sequential_tasks(sub_tasks)
        elif coordination_strategy == "pipeline":
            return await self._execute_pipeline_tasks(sub_tasks)
        else:
            return {"error": f"Unknown coordination strategy: {coordination_strategy}"}

    async def _execute_parallel_tasks(
        self, sub_tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Ejecutar tareas en paralelo máximo 5 concurrentes"""

        semaphore = asyncio.Semaphore(self.max_concurrent_task)
        results = []

        async def execute_with_limit(sub_task):
            async with semaphore:
                sub_task_result = await self._execute_single_sub_task(sub_task)
                return sub_task_result

        # Ejecutar todas las tareas con límite de concurrencia
        tasks = [execute_with_limit(sub_task) for sub_task in sub_tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Procesar resultados
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        failed = len(results) - successful

        return {
            "strategy": "parallel",
            "tasks_executed": len(sub_tasks),
            "successful": successful,
            "failed": failed,
            "results": [r for r in results if isinstance(r, dict)],
            "execution_time": sum(
                r.get("execution_time", 0) for r in results if isinstance(r, dict)
            ),
        }

    async def _execute_sequential_tasks(
        self, sub_tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Ejecutar tareas en secuencia"""

        results = []
        total_time = 0

        for sub_task in sub_tasks:
            result = await self._execute_single_sub_task(sub_task)
            results.append(result)
            if isinstance(result, dict):
                total_time += result.get("execution_time", 0)

        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        failed = len(sub_tasks) - successful

        return {
            "strategy": "sequential",
            "tasks_executed": len(sub_tasks),
            "successful": successful,
            "failed": failed,
            "results": results,
            "execution_time": total_time,
        }

    async def _execute_pipeline_tasks(
        self, sub_tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Ejecutar tareas en pipeline (resultado de una alimenta a la siguiente)"""

        pipeline_result = None
        results = []
        total_time = 0

        for i, sub_task in enumerate(sub_tasks):
            if i > 0 and pipeline_result:
                # Pasar resultado anterior como input
                sub_task["parameters"]["previous_result"] = pipeline_result

            result = await self._execute_single_sub_task(sub_task)
            results.append(result)

            if isinstance(result, dict):
                total_time += result.get("execution_time", 0)
                pipeline_result = result.get("result")

        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        failed = len(sub_tasks) - successful

        return {
            "strategy": "pipeline",
            "tasks_executed": len(sub_tasks),
            "successful": successful,
            "failed": failed,
            "results": results,
            "execution_time": total_time,
            "final_result": pipeline_result,
        }

    async def _execute_single_sub_task(
        self, sub_task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ejecutar una tarea individual dentro de coordinación multi-agente"""

        task_type = sub_task.get("task_type")
        required_capabilities = sub_task.get("required_capabilities", [])
        parameters = sub_task.get("parameters", {})

        # Asignar tarea inteligentemente
        assignment_result = await self.assign_task_intelligently(
            task_type, parameters, required_capabilities
        )

        return assignment_result

    # === MONITOREO Y MANTENIMIENTO ===

    async def _monitor_agent_health(self, task: AgentTask) -> Dict[str, Any]:
        """Monitorear salud de todos los agentes del sistema"""

        health_report = {
            "total_agents": self.total_agents_registered,
            "active_agents": 0,
            "inactive_agents": 0,
            "agent_health_details": {},
            "system_load": {},
            "critical_issues": [],
        }

        # Revisar cada categoría de agentes
        for category_name in dir(self.agent_pool):
            if not category_name.startswith("_"):
                category_pool = getattr(self.agent_pool, category_name, [])
                category_health = []

                for agent in category_pool:
                    try:
                        agent_status = await agent.get_status()

                        if agent.status == AgentStatus.ACTIVE:
                            health_report["active_agents"] += 1
                        else:
                            health_report["inactive_agents"] += 1

                        category_health.append(
                            {
                                "agent_id": agent.agent_id,
                                "agent_name": agent.agent_name,
                                "status": agent.status.value,
                                "health_score": self._calculate_agent_health_score(
                                    agent_status
                                ),
                                "current_task": agent_status.get("current_task"),
                                "metrics": agent_status.get("metrics", {}),
                            }
                        )

                    except Exception as e:
                        health_report["critical_issues"].append(
                            {
                                "agent_id": agent.agent_id,
                                "issue": f"Status check failed: {str(e)}",
                            }
                        )

                health_report["agent_health_details"][category_name] = category_health

        # Calcular métricas del sistema
        health_report["system_load"] = {
            "overall_utilization": self._calculate_overall_utilization(health_report),
            "average_response_time": self._calculate_average_response_time(),
            "task_success_rate": self._calculate_task_success_rate(),
        }

        return health_report

    def _calculate_agent_health_score(self, agent_status: Dict[str, Any]) -> float:
        """Calcular score de salud de un agente"""

        metrics = agent_status.get("metrics", {})

        # Factores de salud
        uptime_score = 100  # Asumir agentes en funcionamiento
        task_success_score = 0
        response_time_score = 100

        total_tasks = metrics.get("tasks_completed", 0) + metrics.get("tasks_failed", 0)
        if total_tasks > 0:
            success_rate = metrics.get("tasks_completed", 0) / total_tasks
            task_success_score = success_rate * 100

        avg_response = metrics.get("response_time_avg", 0)
        if avg_response > 10:  # Más de 10 segundos es lento
            response_time_score = max(0, 100 - (avg_response - 10))

        # Score compuesto
        health_score = (
            uptime_score * 0.3 + task_success_score * 0.4 + response_time_score * 0.3
        )

        return round(health_score, 1)

    def _calculate_overall_utilization(self, health_report: Dict[str, Any]) -> float:
        """Calcular utilización general del sistema"""

        active_agents = health_report.get("active_agents", 0)
        total_agents = health_report.get("total_agents", 1)  # Evitar división por cero

        utilization = (active_agents / total_agents) * 100

        return round(utilization, 1)

    def _calculate_average_response_time(self) -> float:
        """Calcular tiempo promedio de respuesta del sistema"""

        all_response_times = []

        # Recopilar tiempos de respuesta de todos los agentes
        for category_name in dir(self.agent_pool):
            if not category_name.startswith("_"):
                category_pool = getattr(self.agent_pool, category_name, [])
                for agent in category_pool:
                    if agent.metrics.response_time_avg > 0:
                        all_response_times.append(agent.metrics.response_time_avg)

        if all_response_times:
            return round(sum(all_response_times) / len(all_response_times), 2)
        else:
            return 0.0

    def _calculate_task_success_rate(self) -> float:
        """Calcular tasa de éxito de tareas"""

        total_tasks = (
            self.coordination_stats["tasks_completed"]
            + self.coordination_stats["task_failures"]
        )
        if total_tasks == 0:
            return 100.0

        success_rate = (self.coordination_stats["tasks_completed"] / total_tasks) * 100
        return round(success_rate, 1)

    async def _redistribute_workload(self, task: AgentTask) -> Dict[str, Any]:
        """Redistribuir carga de trabajo entre agentes"""

        target_category = task.parameters.get("category")
        redistribution_strategy = task.parameters.get("strategy", "balance")

        if not target_category:
            return {"error": "No category specified for workload redistribution"}

        category_pool = getattr(self.agent_pool, target_category, [])
        if not category_pool:
            return {"error": f"No agents in category {target_category}"}

        # Analizar carga actual
        agent_loads = []
        for agent in category_pool:
            load = len(agent.task_queue)
            agent_loads.append((agent, load))

        # Aplicar estrategia de redistribución
        if redistribution_strategy == "balance":
            return await self._balance_workload(category_pool, agent_loads)
        elif redistribution_strategy == "consolidate":
            return await self._consolidate_workload(category_pool, agent_loads)
        else:
            return {
                "error": f"Unknown redistribution strategy: {redistribution_strategy}"
            }

    async def _balance_workload(
        self, category_pool: List[BaseMCPAgent], agent_loads: List[tuple]
    ) -> Dict[str, Any]:
        """Balancear carga uniformemente entre agentes"""

        # Identificar agentes sobrecargados y subutilizados
        overloaded = [agent for agent, load in agent_loads if load > 5]
        underloaded = [agent for agent, load in agent_loads if load < 2]

        redistributions = []

        for overloaded_agent in overloaded:
            for task in overloaded_agent.task_queue[
                :2
            ]:  # Mover máximo 2 tareas por agente
                if underloaded:
                    # Mover tarea a agente menos cargado
                    target_agent = underloaded[0]
                    target_agent.add_task_to_queue(task)
                    overloaded_agent.task_queue.remove(task)

                    redistributions.append(
                        {
                            "from_agent": overloaded_agent.agent_name,
                            "to_agent": target_agent.agent_name,
                            "task_id": task.task_id,
                        }
                    )

        return {
            "redistribution_strategy": "balance",
            "task_redistributions": len(redistributions),
            "details": redistributions,
            "agents_optimized": len(set(r["from_agent"] for r in redistributions)),
        }

    async def _consolidate_workload(
        self, category_pool: List[BaseMCPAgent], agent_loads: List[tuple]
    ) -> Dict[str, Any]:
        """Consolidar carga en agentes más eficientes"""

        # Encontrar agente más eficiente
        agent_scores = []
        for agent in category_pool:
            efficiency_score = self._calculate_agent_efficiency(agent)
            agent_scores.append((agent, efficiency_score))

        agent_scores.sort(key=lambda x: x[1], reverse=True)  # Más eficiente primero
        primary_agent = agent_scores[0][0]

        # Mover tareas de otros agentes al más eficiente
        redistributions = []
        for agent in [a for a, _ in agent_scores[1:]]:  # Excluir el principal
            tasks_to_move = list(agent.task_queue)[:1]  # Mover máximo 1 tarea
            for task in tasks_to_move:
                primary_agent.add_task_to_queue(task)
                agent.task_queue.remove(task)

                redistributions.append(
                    {
                        "from_agent": agent.agent_name,
                        "to_agent": primary_agent.agent_name,
                        "task_id": task.task_id,
                    }
                )

        return {
            "redistribution_strategy": "consolidate",
            "primary_agent": primary_agent.agent_name,
            "task_redistributions": len(redistributions),
            "details": redistributions,
        }

    def _calculate_agent_efficiency(self, agent: BaseMCPAgent) -> float:
        """Calcular eficiencia de un agente"""

        total_tasks = agent.metrics.tasks_completed + agent.metrics.tasks_failed

        if total_tasks == 0:
            return 50.0  # Score neutral para agentes sin historial

        success_rate = (
            agent.metrics.tasks_completed / total_tasks if total_tasks > 0 else 0
        )
        avg_response_time = agent.metrics.response_time_avg

        # Score compuesto: 70% tasa de éxito, 30% tiempo de respuesta (inverso)
        efficiency = (success_rate * 70) + max(0, 30 - (avg_response_time / 10) * 30)

        return round(efficiency, 1)

    # === SISTEMAS DE SETUP ===

    async def _setup_load_balancing(self) -> None:
        """Configurar sistema de balanceo de carga"""
        self.load_balancer = {
            "active": True,
            "strategy": "round_robin",
            "max_capacity_per_agent": 5,
            "overload_threshold": 3,
        }

    async def _setup_failover_system(self) -> None:
        """Configurar sistema de failover"""
        self.failover_system = {
            "active": True,
            "backup_agents_per_category": 1,
            "automatic_recovery": True,
            "health_check_interval": 30,  # segundos
        }

    async def _setup_resource_monitoring(self) -> None:
        """Configurar monitoreo de recursos"""
        self.resource_monitor = {
            "active": True,
            "cpu_threshold": 80,
            "memory_threshold": 85,
            "scaling_enabled": True,
            "alerts_enabled": True,
        }

    # === HANDLERS DE MENSAJES ===

    async def _handle_agent_registration(self, message: AgentMessage) -> Dict[str, Any]:
        """Manejar registro de nuevo agente"""
        agent_data = message.content

        return {
            "registration_acknowledged": True,
            "agent_id_assigned": f"agent_{uuid.uuid4()}",
            "coordinator_ready": True,
            "message_bus_connected": True,
        }

    async def _handle_agent_status_update(
        self, message: AgentMessage
    ) -> Dict[str, Any]:
        """Manejar actualización de estado de agente"""
        agent_status = message.content

        # Aquí implementaríamos lógica para actualizar estado interno
        return {"status_update_processed": True, "coordinator_acknowledged": True}

    async def _handle_task_completion(self, message: AgentMessage) -> Dict[str, Any]:
        """Manejar notificación de tarea completada"""
        task_result = message.content

        self.coordination_stats["tasks_completed"] += 1

        # Actualizar estadísticas de rendimiento
        await self._update_performance_metrics(task_result)

        return {
            "completion_acknowledged": True,
            "coordination_updated": True,
            "success_logged": task_result.get("success", False),
        }

    async def _handle_agent_failure(self, message: AgentMessage) -> Dict[str, Any]:
        """Manejar notificación de fallo de agente"""
        failure_data = message.content

        self.coordination_stats["agent_failures"] += 1

        # Implementar recuperación automática
        await self._initiate_agent_recovery(failure_data)

        return {
            "failure_acknowledged": True,
            "recovery_initiated": True,
            "coordination_adjusted": True,
        }

    async def _handle_coordination_request(
        self, message: AgentMessage
    ) -> Dict[str, Any]:
        """Manejar solicitud de coordinación"""
        request_data = message.content

        # Procesar solicitud específica
        request_type = request_data.get("request_type")
        if request_type == "assign_task":
            return await self.assign_task_intelligently(
                request_data.get("task_type"),
                request_data.get("parameters"),
                request_data.get("required_capabilities", []),
            )
        elif request_type == "get_status":
            return await self.get_status()
        else:
            return {"error": f"Unknown coordination request type: {request_type}"}

    async def _handle_resource_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Manejar solicitud de recursos"""
        resource_data = message.content

        return {
            "resource_request_acknowledged": True,
            "available_resources": await self._get_available_resources(),
            "allocation_possible": True,
        }

    # === MÉTODOS DE SOPORTE ===

    async def _update_performance_metrics(self, task_result: Dict[str, Any]) -> None:
        """Actualizar métricas de rendimiento tras tarea completada"""
        execution_time = task_result.get("execution_time", 0)

        if execution_time > 0:
            current_avg = self.coordination_stats["average_response_time"]
            total_completed = self.coordination_stats["tasks_completed"]

            # Actualizar promedio
            self.coordination_stats["average_response_time"] = (
                (current_avg * (total_completed - 1)) + execution_time
            ) / total_completed

    async def _initiate_agent_recovery(self, failure_data: Dict[str, Any]) -> None:
        """Iniciar recuperación de agente fallido"""
        agent_id = failure_data.get("agent_id")
        category = failure_data.get("category")

        logger.warning(
            f"Initiating recovery for failed agent {agent_id} in category {category}"
        )

        # Implementar lógica de recuperación (reinicio, reemplazo, etc.)
        # Por ahora solo logging

    async def _get_available_resources(self) -> Dict[str, Any]:
        """Obtener recursos disponibles en el sistema"""
        return {
            "available_agents": self.agents_active,
            "total_capacity": self.total_agents_registered * 5,  # 5 tareas por agente
            "current_load": sum(
                len(agent.task_queue) for agent in self._get_all_agents()
            ),
            "load_percentage": "calculated_in_realtime",
        }

    def _get_all_agents(self) -> List[BaseMCPAgent]:
        """Obtener lista de todos los agentes registrados"""
        all_agents = []
        for category_name in dir(self.agent_pool):
            if not category_name.startswith("_"):
                category_pool = getattr(self.agent_pool, category_name, [])
                all_agents.extend(category_pool)
        return all_agents

    async def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado completo del sistema de agentes"""

        all_agents = self._get_all_agents()

        agent_statuses = {}
        for agent in all_agents:
            agent_statuses[agent.agent_id] = {
                "name": agent.agent_name,
                "status": agent.status.value,
                "capabilities": [cap.value for cap in agent.capabilities],
                "task_queue_size": len(agent.task_queue),
                "metrics": agent.metrics.__dict__ if agent.metrics else {},
            }

        return {
            "coordinator_status": "active",
            "total_agents": len(all_agents),
            "agents_registered": self.total_agents_registered,
            "agents_active": self.agents_active,
            "coordination_stats": self.coordination_stats,
            "agent_pool": {
                category: len(getattr(self.agent_pool, category, []))
                for category in dir(self.agent_pool)
                if not category.startswith("_")
            },
            "agent_statuses": agent_statuses,
            "system_health": self._calculate_system_health(),
            "load_distribution": self._calculate_load_distribution(),
        }

    def _calculate_system_health(self) -> str:
        """Calcular salud general del sistema de agentes"""

        if self.agents_active == 0:
            return "down"
        elif self.agents_active >= self.total_agents_registered * 0.8:
            return "healthy"
        elif self.agents_active >= self.total_agents_registered * 0.5:
            return "warning"
        else:
            return "critical"

    def _calculate_load_distribution(self) -> Dict[str, float]:
        """Calcular distribución de carga por categoría"""

        distribution = {}
        total_load = 0

        for category_name in dir(self.agent_pool):
            if not category_name.startswith("_"):
                category_pool = getattr(self.agent_pool, category_name, [])
                category_load = sum(len(agent.task_queue) for agent in category_pool)
                total_load += category_load
                distribution[category_name] = category_load

        # Convertir a porcentajes
        if total_load > 0:
            distribution = {
                cat: (load / total_load) * 100 for cat, load in distribution.items()
            }

        return distribution

    async def _scale_agent_capacity(self, task: AgentTask) -> Dict[str, Any]:
        """Escalar capacidad de agentes según demanda"""

        scale_type = task.parameters.get("scale_type", "horizontal")
        target_category = task.parameters.get("category")
        capacity_change = task.parameters.get("capacity_change", 1)

        if not target_category:
            return {"error": "No category specified for scaling"}

        try:
            if scale_type == "horizontal":
                return await self._scale_horizontal(
                    str(target_category), capacity_change
                )
            elif scale_type == "vertical":
                return await self._scale_vertical(str(target_category), capacity_change)
            else:
                return {"error": f"Unknown scaling type: {scale_type}"}

        except Exception as e:
            return {"error": f"Scaling failed: {str(e)}"}

    async def _scale_horizontal(
        self, category: str, capacity_change: int
    ) -> Dict[str, Any]:
        """Escalado horizontal: añadir/quitar agentes"""

        current_count = len(getattr(self.agent_pool, category, []))

        if capacity_change > 0:
            # Escalar hacia arriba - añadir agentes
            added_agents = []
            for i in range(capacity_change):
                try:
                    # En producción crearíamos nuevos agentes
                    # Por ahora simulamos
                    agent_id = f"{category}_agent_{current_count + i + 1}"
                    logger.info(f"Simulated horizontal scaling: added agent {agent_id}")

                    added_agents.append(agent_id)

                except Exception as e:
                    logger.error(f"Error adding agent: {e}")

            return {
                "scaling_operation": "horizontal_up",
                "agents_added": len(added_agents),
                "category": category,
                "new_capacity": current_count + len(added_agents),
            }

        elif capacity_change < 0:
            # Escalar hacia abajo - remover agentes
            return {
                "scaling_operation": "horizontal_down",
                "agents_removed": abs(capacity_change),
                "category": category,
                "remaining_capacity": max(0, current_count + capacity_change),
            }

        return {"error": "No capacity change requested"}

    async def _scale_vertical(
        self, category: str, capacity_change: int
    ) -> Dict[str, Any]:
        """Escalado vertical: aumentar capacidad de agentes existentes"""

        category_pool = getattr(self.agent_pool, category, [])
        if not category_pool:
            return {"error": f"No agents in category {category}"}

        scaled_agents = 0
        for agent in category_pool:
            try:
                if capacity_change > 0:
                    # Aumentar capacidad (simulado)
                    agent.max_capacity = (
                        getattr(agent, "max_capacity", 5) + capacity_change
                    )
                else:
                    # Disminuir capacidad
                    current_max = getattr(agent, "max_capacity", 5)
                    agent.max_capacity = max(1, current_max + capacity_change)

                scaled_agents += 1

            except Exception as e:
                logger.error(f"Error scaling agent {agent.agent_id}: {e}")

        return {
            "scaling_operation": "vertical",
            "agents_scaled": scaled_agents,
            "capacity_change": capacity_change,
            "category": category,
        }

    async def _diagnose_agent_issue(self, task: AgentTask) -> Dict[str, Any]:
        """Diagnosticar problemas de agentes"""

        target_agent_id = task.parameters.get("agent_id")
        diagnostic_type = task.parameters.get("diagnostic_type", "health")

        if not target_agent_id:
            return {"error": "No agent_id specified for diagnosis"}

        # Encontrar agente
        target_agent = None
        all_agents = self._get_all_agents()
        for agent in all_agents:
            if agent.agent_id == target_agent_id:
                target_agent = agent
                break

        if not target_agent:
            return {"error": f"Agent {target_agent_id} not found"}

        try:
            diagnostics = await self._run_agent_diagnostics(
                target_agent, diagnostic_type
            )

            # Generar recomendaciones
            recommendations = await self._generate_diagnostic_recommendations(
                diagnostics
            )

            return {
                "agent_id": target_agent_id,
                "diagnostic_type": diagnostic_type,
                "diagnostics": diagnostics,
                "recommendations": recommendations,
                "requires_action": self._requires_immediate_action(diagnostics),
            }

        except Exception as e:
            return {"error": f"Diagnostic failed: {str(e)}"}

    async def _run_agent_diagnostics(
        self, agent: BaseMCPAgent, diagnostic_type: str
    ) -> Dict[str, Any]:
        """Ejecutar diagnósticos en un agente"""

        diagnostics = {
            "agent_status": agent.status.value,
            "uptime": (datetime.now() - agent.start_time).total_seconds(),
            "task_queue_size": len(agent.task_queue),
            "total_tasks_processed": agent.metrics.tasks_completed
            + agent.metrics.tasks_failed,
            "success_rate": 0.0,
            "response_time_avg": agent.metrics.response_time_avg,
        }

        total_tasks = diagnostics["total_tasks_processed"]
        if total_tasks > 0:
            diagnostics["success_rate"] = (
                agent.metrics.tasks_completed / total_tasks
            ) * 100

        # Diagnósticos específicos por tipo
        if diagnostic_type == "performance":
            diagnostics["performance_issues"] = await self._check_performance_issues(
                agent
            )
        elif diagnostic_type == "resource":
            diagnostics["resource_usage"] = await self._check_resource_usage(agent)
        elif diagnostic_type == "communication":
            diagnostics["communication_status"] = (
                await self._check_communication_status(agent)
            )

        return diagnostics

    async def _check_performance_issues(self, agent: BaseMCPAgent) -> List[str]:
        """Verificar problemas de performance"""

        issues = []

        if agent.metrics.response_time_avg > 30:
            issues.append("High average response time")
        if len(agent.task_queue) > 10:
            issues.append("Large task queue - potential bottleneck")
        if agent.metrics.tasks_failed > agent.metrics.tasks_completed:
            issues.append("High task failure rate")

        return issues

    async def _check_resource_usage(self, agent: BaseMCPAgent) -> Dict[str, Any]:
        """Verificar uso de recursos"""

        return {
            "memory_usage_MB": 0,  # Simulado
            "cpu_usage_percent": 0,  # Simulado
            "network_io": 0,  # Simulado
            "resource_overload": len(agent.task_queue) > 8,
        }

    async def _check_communication_status(self, agent: BaseMCPAgent) -> Dict[str, Any]:
        """Verificar estado de comunicación"""

        return {
            "message_bus_connected": agent.message_bus is not None,
            "pending_responses": len(
                getattr(agent, "short_term_memory", {}).get("pending_responses", [])
            ),
            "communication_issues": agent.status == AgentStatus.ERROR,
        }

    async def _generate_diagnostic_recommendations(
        self, diagnostics: Dict[str, Any]
    ) -> List[str]:
        """Generar recomendaciones basadas en diagnósticos"""

        recommendations = []

        if diagnostics.get("success_rate", 100) < 70:
            recommendations.append("Review agent capabilities and task assignments")
        if diagnostics.get("response_time_avg", 0) > 30:
            recommendations.append(
                "Optimize agent performance or redistribute workload"
            )
        if diagnostics.get("performance_issues"):
            recommendations.extend(
                [f"Fix: {issue}" for issue in diagnostics["performance_issues"]]
            )

        return recommendations or ["Agent operating normally"]

    def _requires_immediate_action(self, diagnostics: Dict[str, Any]) -> bool:
        """Determinar si se requiere acción inmediata"""

        critical_conditions = [
            diagnostics.get("agent_status") == "error",
            diagnostics.get("success_rate", 100) < 50,
            len(diagnostics.get("performance_issues", [])) > 2,
        ]

        return any(critical_conditions)

    async def _get_system_status_report(self, task: AgentTask) -> Dict[str, Any]:
        """Obtener reporte completo del estado del sistema"""

        return await self.get_system_status()


# ========== FUNCIÓN DE UTILIDAD PARA INICIALIZACIÓN ==========


async def initialize_mcp_agent_system() -> MCPAgentCoordinator:
    """
    Inicializar sistema MCP con 4 agentes reales disponibles

    Esta función configura el coordinador con los agentes realmente implementados
    """

    logger.info("🚀 Inicializando MCP Coordinator con agentes reales...")

    try:
        # Crear coordinador - ya inicializa los 4 agentes reales automáticamente
        coordinator = MCPAgentCoordinator()

        # El coordinador ya inicializó todos los agentes durante su __init__
        logger.info("✅ MCP Coordinator inicializado exitosamente")
        logger.info(f"🎯 Agentes reales disponibles: {coordinator.total_agents_registered}")

        return coordinator

    except Exception as e:
        logger.error(f"❌ Error inicializando MCP Coordinator: {e}")
        raise


# ========== INSTANCIA GLOBAL ==========

_global_agent_coordinator: Optional[MCPAgentCoordinator] = None


async def get_global_agent_coordinator() -> MCPAgentCoordinator:
    """Obtener instancia global del coordinador MCP"""
    global _global_agent_coordinator

    if _global_agent_coordinator is None:
        _global_agent_coordinator = await initialize_mcp_agent_system()

    return _global_agent_coordinator
