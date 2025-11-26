"""
Base Agent System for MCP Enterprise Master
=============================================

Sistema base para todos los agentes MCP especializados.
Define la interfaz común y funcionalidades básicas de agentes.

Author: MCP Enterprise Master
Version: 1.0.0
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# Configuración de logging para agentes
logger = logging.getLogger("MCP_Agent_System")


class AgentStatus(Enum):
    """Estados posibles de un agente MCP"""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class AgentCapability(Enum):
    """Capacidades disponibles de los agentes"""

    RESEARCH = "research"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    EXECUTION = "execution"
    COORDINATION = "coordination"
    SECURITY = "security"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    BUSINESS = "business"
    STRATEGIC = "strategic"


@dataclass
class AgentMessage:
    """Mensaje estructurado entre agentes MCP"""

    sender_id: str
    receiver_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 1
    requires_response: bool = False
    correlation_id: Optional[str] = None


@dataclass
class AgentTask:
    """Tarea asignada a un agente"""

    task_id: str
    task_type: str
    parameters: Dict[str, Any]
    assigned_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    priority: int = 3
    status: str = "pending"


@dataclass
class AgentMetrics:
    """Métricas de performance de agente"""

    tasks_completed: int = 0
    tasks_failed: int = 0
    response_time_avg: float = 0.0
    uptime_percentage: float = 100.0
    last_activity: datetime = field(default_factory=datetime.now)


class MessageBus:
    """Sistema de mensajería entre agentes MCP"""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[AgentMessage] = []
        self.max_history = 1000

    async def publish(self, message: AgentMessage) -> bool:
        """Publicar mensaje al bus"""
        try:
            # Guardar en historial
            self.message_history.append(message)
            if len(self.message_history) > self.max_history:
                self.message_history = self.message_history[-self.max_history :]

            # Notificar suscriptores
            if message.message_type in self.subscribers:
                tasks = []
                for callback in self.subscribers[message.message_type]:
                    tasks.append(callback(message))

                await asyncio.gather(*tasks, return_exceptions=True)

            logger.debug(
                f"Message published: {message.message_type} from {message.sender_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False

    def subscribe(self, message_type: str, callback: Callable) -> None:
        """Suscribirse a tipo de mensaje"""
        if message_type not in self.subscribers:
            self.subscribers[message_type] = []

        self.subscribers[message_type].append(callback)
        logger.info(f"Subscribed to message type: {message_type}")


class BaseMCPAgent(ABC):
    """
    Agente Base MCP (Model Context Protocol)

    Define la interfaz común para todos los agentes especializados MCP.
    Cada agente especializado debe heredar de esta clase y implementar
    los métodos abstractos.
    """

    def __init__(
        self, agent_id: str, agent_name: str, capabilities: List[AgentCapability]
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.capabilities = capabilities
        self.status = AgentStatus.INITIALIZING

        # Componentes de memoria y estado
        self.short_term_memory: Dict[str, Any] = {}
        self.long_term_memory: Dict[str, Any] = {}
        self.task_queue: List[AgentTask] = []
        self.current_task: Optional[AgentTask] = None

        # Métricas y monitoreo
        self.metrics = AgentMetrics()
        self.start_time = datetime.now()

        # Sistema de mensajería
        self.message_bus = None  # Se asignará por el coordinator

        # Callbacks y handlers
        self.message_handlers: Dict[str, Callable] = {}
        self.task_callbacks: Dict[str, Callable] = {}

    async def initialize(self) -> bool:
        """Inicializar el agente MCP"""
        try:
            await self._setup_capabilities()
            await self._initialize_memory()
            await self._register_message_handlers()

            self.status = AgentStatus.IDLE
            logger.info(
                f"Agent {self.agent_name} ({self.agent_id}) initialized successfully"
            )

            # Actualizar métricas
            self.metrics.last_activity = datetime.now()

            return True

        except Exception as e:
            logger.error(f"Failed to initialize agent {self.agent_name}: {e}")
            self.status = AgentStatus.ERROR
            return False

    async def shutdown(self) -> bool:
        """Apagar el agente de forma ordenada"""
        try:
            await self._cleanup_resources()
            self.status = AgentStatus.SHUTDOWN
            logger.info(f"Agent {self.agent_name} shutdown successfully")
            return True

        except Exception as e:
            logger.error(f"Error during agent shutdown: {e}")
            return False

    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Ejecutar una tarea asignada"""
        try:
            self.status = AgentStatus.PROCESSING
            self.current_task = task

            start_time = datetime.now()

            # Verificar si el agente puede ejecutar esta tarea
            if not self._can_execute_task(task):
                raise ValueError(
                    f"Agent {self.agent_name} cannot execute task type: {task.task_type}"
                )

            # Ejecutar la tarea en la implementación especializada
            result = await self._execute_task_implementation(task)

            # Actualizar métricas
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_execution_metrics(True, execution_time)

            # Marcar tarea como completada
            task.status = "completed"

            self.status = AgentStatus.IDLE
            self.current_task = None

            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "agent_id": self.agent_id,
            }

        except Exception as e:
            logger.error(f"Task execution failed for {self.agent_name}: {e}")

            # Actualizar métricas de error
            self._update_execution_metrics(False, 0)

            self.status = AgentStatus.ERROR
            task.status = "failed"

            return {"success": False, "error": str(e), "agent_id": self.agent_id}

    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Manejar mensaje recibido del bus"""
        try:
            if message.message_type in self.message_handlers:
                handler = self.message_handlers[message.message_type]
                response = await handler(message)

                # Actualizar métricas
                self.metrics.last_activity = datetime.now()

                if message.requires_response:
                    return AgentMessage(
                        sender_id=self.agent_id,
                        receiver_id=message.sender_id,
                        message_type=f"{message.message_type}_response",
                        content=response or {},
                        correlation_id=message.message_id,
                    )

            return None

        except Exception as e:
            logger.error(f"Error handling message in {self.agent_name}: {e}")
            return None

    def request_capabilities(self) -> List[AgentCapability]:
        """Devolver lista de capacidades del agente"""
        return self.capabilities

    async def get_status(self) -> Dict[str, Any]:
        """Obtener estado completo del agente"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "current_task": self.current_task.task_id if self.current_task else None,
            "task_queue_size": len(self.task_queue),
            "metrics": {
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "response_time_avg": self.metrics.response_time_avg,
                "uptime_seconds": uptime,
            },
            "last_activity": self.metrics.last_activity.isoformat(),
        }

    def set_message_bus(self, message_bus: MessageBus) -> None:
        """Asignar bus de mensajería al agente"""
        self.message_bus = message_bus

    # === MÉTODOS ABSTRACTOS - IMPLEMENTAR EN CLASES HIJAS ===

    @abstractmethod
    async def _setup_capabilities(self) -> None:
        """Configurar las capacidades específicas del agente"""
        pass

    @abstractmethod
    async def _execute_task_implementation(self, task: AgentTask) -> Dict[str, Any]:
        """Implementación específica de ejecución de tareas"""
        pass

    @abstractmethod
    async def _register_message_handlers(self) -> None:
        """Registrar handlers para tipos de mensajes específicos"""
        pass

    # === MÉTODOS PROTEGIDOS PARA USO INTERNO ===

    def _can_execute_task(self, task: AgentTask) -> bool:
        """Verificar si el agente puede ejecutar el tipo de tarea"""
        task_capability_map = {
            "research": AgentCapability.RESEARCH,
            "analysis": AgentCapability.ANALYSIS,
            "communication": AgentCapability.COMMUNICATION,
            "execution": AgentCapability.EXECUTION,
            "security": AgentCapability.SECURITY,
            "creative": AgentCapability.CREATIVE,
            "technical": AgentCapability.TECHNICAL,
            "business": AgentCapability.BUSINESS,
            "strategic": AgentCapability.STRATEGIC,
        }

        required_capability = task_capability_map.get(task.task_type)
        return required_capability in self.capabilities if required_capability else True

    def _update_execution_metrics(self, success: bool, execution_time: float) -> None:
        """Actualizar métricas de ejecución"""
        if success:
            self.metrics.tasks_completed += 1
        else:
            self.metrics.tasks_failed += 1

        # Actualizar tiempo promedio de respuesta
        if execution_time > 0:
            total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
            self.metrics.response_time_avg = (
                (self.metrics.response_time_avg * (total_tasks - 1)) + execution_time
            ) / total_tasks

        self.metrics.last_activity = datetime.now()

    async def _initialize_memory(self) -> None:
        """Inicializar sistemas de memoria del agente"""
        # Sistema de memoria a corto plazo (últimas 10 tareas)
        self.short_term_memory = {
            "recent_tasks": [],
            "active_contexts": {},
            "pending_responses": [],
        }

        # Sistema de memoria a largo plazo (conocimiento adquirido)
        self.long_term_memory = {
            "learned_patterns": {},
            "expertise_areas": {},
            "past_interactions": [],
            "capability_history": {},
        }

    async def _cleanup_resources(self) -> None:
        """Limpiar recursos del agente"""
        self.task_queue.clear()
        self.short_term_memory.clear()
        self.long_term_memory.clear()
        self.current_task = None

    # === MÉTODOS DE UTILIDAD PARA AGENTES ===

    async def send_message(
        self,
        receiver_id: str,
        message_type: str,
        content: Dict[str, Any],
        requires_response: bool = False,
    ) -> Optional[str]:
        """Enviar mensaje a otro agente"""
        if not self.message_bus:
            return None

        message = AgentMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            requires_response=requires_response,
        )

        success = await self.message_bus.publish(message)
        return message.message_id if success else None

    async def broadcast_message(
        self, message_type: str, content: Dict[str, Any]
    ) -> bool:
        """Enviar mensaje broadcast a todos los agentes"""
        return (await self.send_message("*", message_type, content)) is not None

    def add_task_to_queue(self, task: AgentTask) -> None:
        """Agregar tarea a la cola del agente"""
        self.task_queue.append(task)

    def get_next_task(self) -> Optional[AgentTask]:
        """Obtener siguiente tarea de la cola (por prioridad)"""
        if not self.task_queue:
            return None

        # Ordenar por prioridad (menor número = mayor prioridad)
        sorted_tasks = sorted(self.task_queue, key=lambda t: t.priority)
        next_task = sorted_tasks[0]

        # Remover de la cola
        self.task_queue.remove(next_task)

        return next_task

    def remember_interaction(
        self, agent_id: str, interaction_type: str, details: Dict[str, Any]
    ) -> None:
        """Recordar interacción con otro agente"""
        if "past_interactions" not in self.long_term_memory:
            self.long_term_memory["past_interactions"] = []

        interaction = {
            "agent_id": agent_id,
            "type": interaction_type,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }

        self.long_term_memory["past_interactions"].append(interaction)

        # Mantener solo últimas 50 interacciones
        if len(self.long_term_memory["past_interactions"]) > 50:
            self.long_term_memory["past_interactions"] = self.long_term_memory[
                "past_interactions"
            ][-50:]


# Instancia global del message bus
_global_message_bus = MessageBus()


def get_global_message_bus() -> MessageBus:
    """Obtener instancia global del message bus"""
    return _global_message_bus
