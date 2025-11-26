#!/usr/bin/env python3
"""
Sistema Multi-Agente para Sheily AI
Implementa coordinación y comunicación entre agentes especializados
Basado en patrones de Google: Hierarchical, Collaborative, Peer-to-Peer
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import contextlib

logger = logging.getLogger(__name__)

# Simple tracing context manager replacement
@contextlib.contextmanager
def trace_agent_execution(agent_name, operation):
    """Simple tracing replacement"""
    yield type('Trace', (), {'add_event': lambda self, n, d: None})()

# =============================================================================
# MODELOS DE DATOS PARA MULTI-AGENTE
# =============================================================================


class AgentRole(Enum):
    """Roles posibles para agentes en el sistema"""

    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    EVALUATOR = "evaluator"
    MEDIATOR = "mediator"
    EXECUTOR = "executor"


class CommunicationProtocol(Enum):
    """Protocolos de comunicación entre agentes"""

    DIRECT = "direct"  # Comunicación directa síncrona
    MESSAGE_QUEUE = "queue"  # Cola de mensajes asíncrona
    EVENT_DRIVEN = "event"  # Sistema basado en eventos
    SHARED_MEMORY = "memory"  # Memoria compartida


class TaskStatus(Enum):
    """Estados posibles de una tarea"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentCapability:
    """Capacidad específica de un agente"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    confidence_score: float = 1.0
    execution_time_estimate: float = 1.0  # segundos


@dataclass
class AgentProfile:
    """Perfil completo de un agente"""

    agent_id: str
    name: str
    role: AgentRole
    capabilities: List[AgentCapability] = field(default_factory=list)
    specialization_domains: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    communication_protocols: List[CommunicationProtocol] = field(default_factory=list)
    is_active: bool = True
    last_seen: Optional[datetime] = None


@dataclass
class Task:
    """Tarea en el sistema multi-agente"""

    task_id: str
    description: str
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1  # 1-10, 10 es máxima prioridad
    dependencies: List[str] = field(default_factory=list)  # IDs de tareas requeridas
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    quality_score: Optional[float] = None


@dataclass
class AgentMessage:
    """Mensaje entre agentes"""

    message_id: str
    sender_id: str
    receiver_id: str
    message_type: (
        str  # "task_assignment", "status_update", "collaboration_request", etc.
    )
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # Para rastrear conversaciones


# =============================================================================
# AGENTE BASE PARA EL SISTEMA MULTI-AGENTE
# =============================================================================


class MultiAgentBase:
    """Clase base para agentes en el sistema multi-agente"""

    def __init__(self, agent_id: str, name: str, role: AgentRole):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.capabilities: List[AgentCapability] = []
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.is_active = True
        self.last_activity = datetime.now()
        self.performance_stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_response_time": 0.0,
            "success_rate": 1.0,
        }

    async def start(self):
        """Iniciar el agente"""
        self.is_active = True
        logger.info(f"Agente {self.name} ({self.agent_id}) iniciado")

    async def stop(self):
        """Detener el agente"""
        self.is_active = False
        logger.info(f"Agente {self.name} ({self.agent_id}) detenido")

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Procesar un mensaje recibido"""
        raise NotImplementedError("Subclasses must implement process_message")

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Ejecutar una tarea asignada"""
        raise NotImplementedError("Subclasses must implement execute_task")

    def get_profile(self) -> AgentProfile:
        """Obtener perfil del agente"""
        return AgentProfile(
            agent_id=self.agent_id,
            name=self.name,
            role=self.role,
            capabilities=self.capabilities,
            performance_metrics=self.performance_stats,
            is_active=self.is_active,
            last_seen=self.last_activity,
        )

    def update_performance_stats(self, task_duration: float, success: bool):
        """Actualizar estadísticas de rendimiento"""
        self.performance_stats["tasks_completed"] += 1 if success else 0
        self.performance_stats["tasks_failed"] += 0 if success else 1

        # Actualizar tiempo promedio de respuesta
        current_avg = self.performance_stats["average_response_time"]
        total_tasks = (
            self.performance_stats["tasks_completed"]
            + self.performance_stats["tasks_failed"]
        )

        if total_tasks > 0:
            self.performance_stats["average_response_time"] = (
                (current_avg * (total_tasks - 1)) + task_duration
            ) / total_tasks

        # Actualizar tasa de éxito
        if total_tasks > 0:
            self.performance_stats["success_rate"] = (
                self.performance_stats["tasks_completed"] / total_tasks
            )


# =============================================================================
# AGENTE COORDINADOR (HIERARCHICAL PATTERN)
# =============================================================================


class CoordinatorAgent(MultiAgentBase):
    """Agente coordinador que gestiona y asigna tareas a agentes especializados"""

    def __init__(self, agent_id: str, name: str):
        super().__init__(agent_id, name, AgentRole.COORDINATOR)
        self.registered_agents: Dict[str, AgentProfile] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.coordination_strategy = (
            "load_balanced"  # "load_balanced", "specialization", "performance"
        )

    async def start(self):
        await super().start()
        # Iniciar procesamiento de tareas
        asyncio.create_task(self._process_task_queue())

    async def register_agent(self, agent_profile: AgentProfile):
        """Registrar un agente en el sistema"""
        self.registered_agents[agent_profile.agent_id] = agent_profile
        logger.info(
            f"Agente registrado: {agent_profile.name} ({agent_profile.agent_id})"
        )

    async def unregister_agent(self, agent_id: str):
        """Desregistrar un agente"""
        if agent_id in self.registered_agents:
            del self.registered_agents[agent_id]
            logger.info(f"Agente desregistrado: {agent_id}")

    async def submit_task(self, task: Task) -> str:
        """Enviar una tarea para procesamiento"""
        await self.task_queue.put(task)
        self.active_tasks[task.task_id] = task
        logger.info(f"Tarea enviada: {task.task_id} - {task.description}")
        return task.task_id

    async def _process_task_queue(self):
        """Procesar cola de tareas"""
        while self.is_active:
            try:
                task = await self.task_queue.get()

                # Asignar agente apropiado
                assigned_agent = await self._assign_task_to_agent(task)

                if assigned_agent:
                    # Enviar tarea al agente asignado
                    await self._send_task_to_agent(task, assigned_agent)
                else:
                    # Marcar tarea como fallida si no hay agente disponible
                    task.status = TaskStatus.FAILED
                    task.error_message = "No suitable agent available"
                    logger.warning(f"No se pudo asignar tarea {task.task_id}")

                self.task_queue.task_done()

            except Exception as e:
                logger.error(f"Error procesando tarea: {e}")

    async def _assign_task_to_agent(self, task: Task) -> Optional[str]:
        """Asignar tarea al agente más apropiado"""
        if self.coordination_strategy == "load_balanced":
            return await self._assign_load_balanced(task)
        elif self.coordination_strategy == "specialization":
            return await self._assign_by_specialization(task)
        elif self.coordination_strategy == "performance":
            return await self._assign_by_performance(task)
        else:
            return await self._assign_load_balanced(task)

    async def _assign_load_balanced(self, task: Task) -> Optional[str]:
        """Asignación balanceada de carga"""
        available_agents = [
            agent_id
            for agent_id, profile in self.registered_agents.items()
            if profile.is_active
        ]

        if not available_agents:
            return None

        # Asignar al agente con menos tareas activas (simplificado)
        # En producción, esto debería consultar métricas reales
        return available_agents[0]  # Simplificado

    async def _assign_by_specialization(self, task: Task) -> Optional[str]:
        """Asignación por especialización"""
        task_keywords = task.description.lower().split()

        best_agent = None
        best_score = 0

        for agent_id, profile in self.registered_agents.items():
            if not profile.is_active:
                continue

            # Calcular coincidencia con dominios de especialización
            score = 0
            for keyword in task_keywords:
                for domain in profile.specialization_domains:
                    if keyword in domain.lower():
                        score += 1

            if score > best_score:
                best_score = score
                best_agent = agent_id

        return best_agent

    async def _assign_by_performance(self, task: Task) -> Optional[str]:
        """Asignación por rendimiento"""
        best_agent = None
        best_score = 0

        for agent_id, profile in self.registered_agents.items():
            if not profile.is_active:
                continue

            # Usar métricas de rendimiento para decidir
            success_rate = profile.performance_metrics.get("success_rate", 0.5)
            avg_response_time = profile.performance_metrics.get(
                "avg_response_time", 10.0
            )

            # Score combinado: éxito alto + tiempo de respuesta bajo
            score = success_rate * (1.0 / (1.0 + avg_response_time))

            if score > best_score:
                best_score = score
                best_agent = agent_id

        return best_agent

    async def _send_task_to_agent(self, task: Task, agent_id: str):
        """Enviar tarea a un agente específico"""
        message = AgentMessage(
            message_id=f"task_{task.task_id}_{int(time.time())}",
            sender_id=self.agent_id,
            receiver_id=agent_id,
            message_type="task_assignment",
            content={"task": task.__dict__},
            correlation_id=task.task_id,
        )

        # Aquí iría el envío real del mensaje al agente
        # Por ahora, simulamos el procesamiento
        logger.info(f"Tarea {task.task_id} asignada a agente {agent_id}")

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Procesar mensajes del coordinador"""
        if message.message_type == "task_completed":
            task_id = message.correlation_id
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = TaskStatus.COMPLETED
                task.output_data = message.content.get("result")
                task.completed_at = datetime.now()

                logger.info(f"Tarea completada: {task_id}")

        elif message.message_type == "task_failed":
            task_id = message.correlation_id
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = TaskStatus.FAILED
                task.error_message = message.content.get("error")
                task.completed_at = datetime.now()

                logger.warning(f"Tarea fallida: {task_id} - {task.error_message}")

        return None

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """El coordinador no ejecuta tareas directamente"""
        return {"error": "Coordinator agent does not execute tasks directly"}


# =============================================================================
# AGENTE ESPECIALIZADO
# =============================================================================


class SpecializedAgent(MultiAgentBase):
    """Agente especializado en una tarea específica"""

    def __init__(self, agent_id: str, name: str, specialization: str):
        super().__init__(agent_id, name, AgentRole.SPECIALIST)
        self.specialization = specialization
        self.coordinator_id: Optional[str] = None

    async def start(self):
        await super().start()
        # Registrar capacidades basadas en especialización
        await self._register_capabilities()

    async def _register_capabilities(self):
        """Registrar capacidades del agente especializado"""
        if self.specialization == "code_analysis":
            self.capabilities.append(
                AgentCapability(
                    name="analyze_code",
                    description="Analizar código fuente para calidad y bugs",
                    input_schema={
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"issues": {"type": "array"}},
                    },
                )
            )
        elif self.specialization == "data_processing":
            self.capabilities.append(
                AgentCapability(
                    name="process_data",
                    description="Procesar y analizar datos",
                    input_schema={
                        "type": "object",
                        "properties": {"data": {"type": "array"}},
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"insights": {"type": "object"}},
                    },
                )
            )
        # Añadir más especializaciones según sea necesario

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Procesar mensajes del agente especializado"""
        if message.message_type == "task_assignment":
            task_data = message.content.get("task", {})
            task = Task(**task_data)

            # Ejecutar tarea
            try:
                result = await self.execute_task(task)

                # Enviar resultado de vuelta al coordinador
                response_message = AgentMessage(
                    message_id=f"result_{task.task_id}_{int(time.time())}",
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type="task_completed",
                    content={"result": result},
                    correlation_id=task.task_id,
                )

                return response_message

            except Exception as e:
                # Enviar error de vuelta
                error_message = AgentMessage(
                    message_id=f"error_{task.task_id}_{int(time.time())}",
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type="task_failed",
                    content={"error": str(e)},
                    correlation_id=task.task_id,
                )

                return error_message

        return None

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Ejecutar tarea especializada"""
        start_time = time.time()

        try:
            with trace_agent_execution(
                self.name, f"execute_task_{task.task_id}"
            ) as trace:
                trace.add_event("task_started", {"task_description": task.description})

                # Simular procesamiento basado en especialización
                if self.specialization == "code_analysis":
                    result = await self._analyze_code(task.input_data)
                elif self.specialization == "data_processing":
                    result = await self._process_data(task.input_data)
                else:
                    result = {
                        "message": f"Task executed by {self.specialization} agent"
                    }

                trace.add_event("task_completed", {"result_keys": list(result.keys())})

                # Actualizar estadísticas
                duration = time.time() - start_time
                self.update_performance_stats(duration, True)

                return result

        except Exception as e:
            duration = time.time() - start_time
            self.update_performance_stats(duration, False)
            raise e

    async def _analyze_code(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analizar código (simulado)"""
        code = input_data.get("code", "")
        # Simulación de análisis
        issues = []
        if "TODO" in code:
            issues.append({"type": "info", "message": "TODO comment found"})
        if len(code) > 1000:
            issues.append({"type": "warning", "message": "Code is quite long"})

        return {"issues": issues, "complexity_score": len(code) / 100}

    async def _process_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Procesar datos (simulado)"""
        data = input_data.get("data", [])
        # Simulación de procesamiento
        insights = {
            "total_records": len(data),
            "average_value": sum(data) / len(data) if data else 0,
            "max_value": max(data) if data else 0,
            "min_value": min(data) if data else 0,
        }

        return {"insights": insights}


# =============================================================================
# SISTEMA MULTI-AGENTE PRINCIPAL
# =============================================================================


class MultiAgentSystem:
    """Sistema principal de agentes múltiples"""

    def __init__(self):
        self.coordinator: Optional[CoordinatorAgent] = None
        self.agents: Dict[str, MultiAgentBase] = {}
        self.message_bus: asyncio.Queue = asyncio.Queue()
        self.is_running = False

    async def initialize(self):
        """Inicializar el sistema multi-agente"""
        logger.info("Inicializando sistema multi-agente...")

        # Crear coordinador
        self.coordinator = CoordinatorAgent("coordinator_001", "Master Coordinator")
        await self.coordinator.start()

        # Registrar coordinador como agente
        self.agents[self.coordinator.agent_id] = self.coordinator

        # Iniciar procesamiento de mensajes
        self.is_running = True
        asyncio.create_task(self._process_message_bus())

        logger.info("Sistema multi-agente inicializado")

    async def shutdown(self):
        """Apagar el sistema multi-agente"""
        logger.info("Apagando sistema multi-agente...")

        self.is_running = False

        # Detener todos los agentes
        for agent in self.agents.values():
            await agent.stop()

        self.agents.clear()
        logger.info("Sistema multi-agente apagado")

    async def register_agent(self, agent: MultiAgentBase):
        """Registrar un agente en el sistema"""
        self.agents[agent.agent_id] = agent
        await agent.start()

        # Registrar en el coordinador
        if self.coordinator:
            await self.coordinator.register_agent(agent.get_profile())

        logger.info(f"Agente registrado: {agent.name} ({agent.agent_id})")

    async def submit_task(
        self, description: str, input_data: Dict[str, Any] = None, priority: int = 1
    ) -> str:
        """Enviar una tarea al sistema"""
        task = Task(
            task_id=f"task_{int(time.time())}_{len(self.coordinator.active_tasks) if self.coordinator else 0}",
            description=description,
            input_data=input_data or {},
            priority=priority,
        )

        if self.coordinator:
            return await self.coordinator.submit_task(task)
        else:
            raise RuntimeError("Coordinator not initialized")

    async def get_task_status(self, task_id: str) -> Optional[Task]:
        """Obtener estado de una tarea"""
        if self.coordinator and task_id in self.coordinator.active_tasks:
            return self.coordinator.active_tasks[task_id]
        return None

    async def send_message(self, message: AgentMessage):
        """Enviar mensaje entre agentes"""
        await self.message_bus.put(message)

    async def _process_message_bus(self):
        """Procesar bus de mensajes"""
        while self.is_running:
            try:
                message = await self.message_bus.get()

                # Encontrar agente receptor
                if message.receiver_id in self.agents:
                    receiver = self.agents[message.receiver_id]

                    # Procesar mensaje
                    response = await receiver.process_message(message)

                    # Enviar respuesta si existe
                    if response:
                        await self.send_message(response)

                self.message_bus.task_done()

            except Exception as e:
                logger.error(f"Error procesando mensaje: {e}")

    async def evaluate_system_performance(self) -> Dict[str, Any]:
        """Evaluar rendimiento del sistema multi-agente"""
        if not self.coordinator:
            return {"error": "Coordinator not initialized"}

        # Recopilar métricas de todos los agentes
        system_metrics = {
            "total_agents": len(self.agents),
            "active_agents": sum(
                1 for agent in self.agents.values() if agent.is_active
            ),
            "total_tasks": len(self.coordinator.active_tasks),
            "completed_tasks": sum(
                1
                for task in self.coordinator.active_tasks.values()
                if task.status == TaskStatus.COMPLETED
            ),
            "failed_tasks": sum(
                1
                for task in self.coordinator.active_tasks.values()
                if task.status == TaskStatus.FAILED
            ),
            "agent_performance": {},
        }

        # Métricas por agente
        for agent_id, agent in self.agents.items():
            system_metrics["agent_performance"][agent_id] = {
                "name": agent.name,
                "role": agent.role.value,
                "performance_stats": agent.performance_stats,
                "is_active": agent.is_active,
            }

        # Calcular métricas globales
        total_tasks = system_metrics["total_tasks"]
        if total_tasks > 0:
            system_metrics["task_success_rate"] = (
                system_metrics["completed_tasks"] / total_tasks
            )
        else:
            system_metrics["task_success_rate"] = 0.0

        return system_metrics


# =============================================================================
# FUNCIONES DE UTILIDAD Y EJEMPLOS
# =============================================================================

# Instancia global del sistema multi-agente
multi_agent_system = MultiAgentSystem()


async def initialize_multi_agent_system():
    """Inicializar sistema multi-agente con agentes de ejemplo"""
    await multi_agent_system.initialize()

    # Crear agentes especializados de ejemplo
    code_agent = SpecializedAgent("code_001", "Code Analyzer", "code_analysis")
    data_agent = SpecializedAgent("data_001", "Data Processor", "data_processing")

    # Registrar agentes
    await multi_agent_system.register_agent(code_agent)
    await multi_agent_system.register_agent(data_agent)

    logger.info("Sistema multi-agente inicializado con agentes de ejemplo")


async def demo_multi_agent_workflow():
    """Demostración de workflow multi-agente"""
    # Inicializar sistema
    await initialize_multi_agent_system()

    # Enviar tarea de análisis de código
    task_id = await multi_agent_system.submit_task(
        "Analyze the following Python code for potential issues",
        {
            "code": """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers) if numbers else 0

# TODO: Add error handling
result = calculate_average([1, 2, 3, 4, 5])
print(result)
"""
        },
    )

    # Esperar un poco para procesamiento
    await asyncio.sleep(2)

    # Verificar estado
    status = await multi_agent_system.get_task_status(task_id)
    if status:
        print(f"Task {task_id} status: {status.status.value}")
        if status.output_data:
            print(f"Result: {status.output_data}")

    # Evaluar rendimiento del sistema
    performance = await multi_agent_system.evaluate_system_performance()
    print(f"System performance: {performance}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Clases principales
    "MultiAgentBase",
    "CoordinatorAgent",
    "SpecializedAgent",
    "MultiAgentSystem",
    # Modelos de datos
    "AgentRole",
    "CommunicationProtocol",
    "TaskStatus",
    "AgentCapability",
    "AgentProfile",
    "Task",
    "AgentMessage",
    # Instancia global
    "multi_agent_system",
    # Funciones de utilidad
    "initialize_multi_agent_system",
    "demo_multi_agent_workflow",
]

# Información del módulo
__version__ = "1.0.0"
__author__ = "Sheily AI Team - Multi-Agent System"
