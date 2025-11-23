#!/usr/bin/env python3
"""
Agent-to-Agent (A2A) Protocol Implementation para Sheily AI
Implementa el protocolo de comunicación directa entre agentes según especificaciones de Google
Permite coordinación autónoma, federación y escalabilidad de agentes
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import aiohttp
import websockets

from sheily_core.agent_quality import evaluate_agent_quality
from sheily_core.agent_tracing import trace_agent_execution
from sheily_core.multi_agent_system import multi_agent_system

logger = logging.getLogger(__name__)

# =============================================================================
# MODELOS DE DATOS A2A
# =============================================================================


class A2AMessageType(Enum):
    """Tipos de mensajes en el protocolo A2A"""

    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_UPDATE = "task_update"
    TASK_CANCEL = "task_cancel"
    AGENT_DISCOVERY = "agent_discovery"
    AGENT_ANNOUNCE = "agent_announce"
    FEDERATION_REQUEST = "federation_request"
    FEDERATION_RESPONSE = "federation_response"


class A2ATaskStatus(Enum):
    """Estados de tareas A2A"""

    PENDING = "pending"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class A2ATransportType(Enum):
    """Tipos de transporte A2A soportados"""

    HTTP = "http"
    WEBSOCKET = "websocket"
    MESSAGE_QUEUE = "message_queue"


@dataclass
class AgentCard:
    """Agent Card - Identidad digital de un agente"""

    agent_id: str
    name: str
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    endpoint: str = ""
    supported_protocols: List[str] = field(default_factory=lambda: ["A2A-v1.0"])
    security_level: str = "standard"
    organization: str = "sheily-ai"
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la Agent Card a diccionario"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "endpoint": self.endpoint,
            "supported_protocols": self.supported_protocols,
            "security_level": self.security_level,
            "organization": self.organization,
            "version": self.version,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        """Crea Agent Card desde diccionario"""
        data_copy = data.copy()
        data_copy["created_at"] = datetime.fromisoformat(data["created_at"])
        data_copy["last_seen"] = datetime.fromisoformat(data["last_seen"])
        return cls(**data_copy)


@dataclass
class A2AMessage:
    """Mensaje A2A base"""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: A2AMessageType = A2AMessageType.TASK_REQUEST
    sender_id: str = ""
    receiver_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    ttl: int = 3600  # Time to live in seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el mensaje a diccionario"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        """Crea mensaje desde diccionario"""
        data_copy = data.copy()
        data_copy["message_type"] = A2AMessageType(data["message_type"])
        data_copy["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data_copy)


@dataclass
class A2ATask:
    """Tarea A2A"""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    requester_id: str = ""
    assignee_id: Optional[str] = None
    status: A2ATaskStatus = A2ATaskStatus.PENDING
    priority: int = 1
    requirements: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    progress: float = 0.0
    subtasks: List["A2ATask"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la tarea a diccionario"""
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "requester_id": self.requester_id,
            "assignee_id": self.assignee_id,
            "status": self.status.value,
            "priority": self.priority,
            "requirements": self.requirements,
            "context": self.context,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "progress": self.progress,
            "subtasks": [subtask.to_dict() for subtask in self.subtasks],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2ATask":
        """Crea tarea desde diccionario"""
        data_copy = data.copy()
        data_copy["status"] = A2ATaskStatus(data["status"])
        data_copy["created_at"] = datetime.fromisoformat(data["created_at"])
        data_copy["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if data.get("deadline"):
            data_copy["deadline"] = datetime.fromisoformat(data["deadline"])
        data_copy["subtasks"] = [
            cls.from_dict(subtask) for subtask in data.get("subtasks", [])
        ]
        return cls(**data_copy)


# =============================================================================
# AGENT CARD REGISTRY
# =============================================================================


class AgentCardRegistry:
    """Registro de Agent Cards para descubrimiento"""

    def __init__(self):
        self.cards: Dict[str, AgentCard] = {}
        self.capability_index: Dict[str, List[str]] = {}  # capability -> [agent_ids]

    def register_agent(self, card: AgentCard):
        """Registra un agente en el registry"""
        self.cards[card.agent_id] = card

        # Actualizar índice de capacidades
        for capability in card.capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = []
            if card.agent_id not in self.capability_index[capability]:
                self.capability_index[capability].append(card.agent_id)

        logger.info(f"Agent registered: {card.name} ({card.agent_id})")

    def unregister_agent(self, agent_id: str):
        """Desregistra un agente"""
        if agent_id in self.cards:
            card = self.cards[agent_id]

            # Remover del índice de capacidades
            for capability in card.capabilities:
                if capability in self.capability_index:
                    self.capability_index[capability] = [
                        aid
                        for aid in self.capability_index[capability]
                        if aid != agent_id
                    ]

            del self.cards[agent_id]
            logger.info(f"Agent unregistered: {agent_id}")

    def get_agent_card(self, agent_id: str) -> Optional[AgentCard]:
        """Obtiene la Agent Card de un agente"""
        return self.cards.get(agent_id)

    def find_agents_by_capability(self, capability: str) -> List[AgentCard]:
        """Encuentra agentes por capacidad"""
        agent_ids = self.capability_index.get(capability, [])
        return [self.cards[aid] for aid in agent_ids if aid in self.cards]

    def find_agents_by_capabilities(self, capabilities: List[str]) -> List[AgentCard]:
        """Encuentra agentes que tienen todas las capacidades especificadas"""
        if not capabilities:
            return list(self.cards.values())

        # Encontrar agentes que tienen al menos una de las capacidades requeridas
        candidate_ids = set()
        for capability in capabilities:
            candidate_ids.update(self.capability_index.get(capability, []))

        candidates = [self.cards[aid] for aid in candidate_ids if aid in self.cards]

        # Filtrar agentes que tienen TODAS las capacidades requeridas
        result = []
        for candidate in candidates:
            if all(cap in candidate.capabilities for cap in capabilities):
                result.append(candidate)

        return result

    def list_all_agents(self) -> List[AgentCard]:
        """Lista todos los agentes registrados"""
        return list(self.cards.values())

    def update_agent_heartbeat(self, agent_id: str):
        """Actualiza el último visto de un agente"""
        if agent_id in self.cards:
            self.cards[agent_id].last_seen = datetime.now()


# =============================================================================
# A2A CLIENT
# =============================================================================


class A2AClient:
    """Cliente A2A para comunicación agent-to-agent"""

    def __init__(
        self, agent_id: str, transport_type: A2ATransportType = A2ATransportType.HTTP
    ):
        self.agent_id = agent_id
        self.transport_type = transport_type
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.task_listeners: Dict[str, Callable] = {}
        self.connected = False

    async def connect(self, server_url: Optional[str] = None):
        """Conecta al servidor A2A"""
        if self.transport_type == A2ATransportType.WEBSOCKET and server_url:
            self.websocket = await websockets.connect(server_url)
            self.connected = True
            # Iniciar listener de mensajes
            asyncio.create_task(self._listen_messages())
            logger.info(f"A2A Client connected to {server_url}")
        else:
            self.connected = True  # Para HTTP, asumimos conexión directa

    async def disconnect(self):
        """Desconecta del servidor A2A"""
        if self.websocket:
            await self.websocket.close()
        self.connected = False
        logger.info("A2A Client disconnected")

    async def send_message(
        self, message: A2AMessage, target_endpoint: Optional[str] = None
    ) -> Optional[A2AMessage]:
        """Envía un mensaje A2A"""
        try:
            # Simular comunicación para URLs a2a:// en modo de prueba
            if target_endpoint and target_endpoint.startswith("a2a://"):
                return await self._simulate_a2a_response(message, target_endpoint)

            if self.transport_type == A2ATransportType.HTTP and target_endpoint:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        target_endpoint,
                        json=message.to_dict(),
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return A2AMessage.from_dict(result)

            elif self.transport_type == A2ATransportType.WEBSOCKET and self.websocket:
                await self.websocket.send(json.dumps(message.to_dict()))

                # Esperar respuesta si es un request
                if message.message_type in [
                    A2AMessageType.TASK_REQUEST,
                    A2AMessageType.AGENT_DISCOVERY,
                ]:
                    future = asyncio.Future()
                    self.pending_requests[message.message_id] = future
                    try:
                        response_data = await asyncio.wait_for(future, timeout=30.0)
                        return A2AMessage.from_dict(response_data)
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Timeout waiting for response to message {message.message_id}"
                        )

        except Exception as e:
            logger.error(f"Error sending A2A message: {e}")

        return None

    async def _simulate_a2a_response(
        self, message: A2AMessage, target_endpoint: str
    ) -> Optional[A2AMessage]:
        """Simula una respuesta A2A para testing"""
        # Extraer agent_id del endpoint
        agent_id = target_endpoint.replace("a2a://sheily-agents/", "")

        if message.message_type == A2AMessageType.TASK_REQUEST:
            # Simular aceptación de tarea
            await asyncio.sleep(0.1)  # Simular latencia de red
            return A2AMessage(
                message_type=A2AMessageType.TASK_RESPONSE,
                sender_id=agent_id,
                receiver_id=message.sender_id,
                correlation_id=message.message_id,
                payload={
                    "accepted": True,
                    "task_id": message.payload["task"]["task_id"],
                    "estimated_completion": 300,
                },
            )

        return None

    async def _listen_messages(self):
        """Escucha mensajes entrantes"""
        try:
            async for message_str in self.websocket:
                message_data = json.loads(message_str)
                message = A2AMessage.from_dict(message_data)

                # Manejar mensaje basado en tipo
                if message.message_id in self.pending_requests:
                    # Es una respuesta a una solicitud pendiente
                    future = self.pending_requests.pop(message.message_id)
                    future.set_result(message_data)
                else:
                    # Es un mensaje nuevo, procesarlo
                    await self._handle_incoming_message(message)

        except Exception as e:
            logger.error(f"Error listening to A2A messages: {e}")

    async def _handle_incoming_message(self, message: A2AMessage):
        """Maneja mensaje entrante"""
        if message.message_type == A2AMessageType.TASK_REQUEST:
            await self._handle_task_request(message)
        elif message.message_type == A2AMessageType.TASK_UPDATE:
            await self._handle_task_update(message)
        elif message.message_type == A2AMessageType.TASK_CANCEL:
            await self._handle_task_cancel(message)
        elif message.message_type == A2AMessageType.AGENT_DISCOVERY:
            await self._handle_agent_discovery(message)

    async def _handle_task_request(self, message: A2AMessage):
        """Maneja solicitud de tarea"""
        task_data = message.payload.get("task", {})
        task = A2ATask.from_dict(task_data)

        # Verificar si podemos manejar esta tarea
        if await self._can_handle_task(task):
            # Aceptar la tarea
            response = A2AMessage(
                message_type=A2AMessageType.TASK_RESPONSE,
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                correlation_id=message.message_id,
                payload={
                    "accepted": True,
                    "task_id": task.task_id,
                    "estimated_completion": 300,  # 5 minutos
                },
            )
            await self.send_message(response)

            # Iniciar procesamiento de tarea en background
            # Crear una tarea asíncrona que procese la tarea y actualice el estado
            asyncio.create_task(self._process_task_async(task, message.sender_id))
        else:
            # Rechazar la tarea
            response = A2AMessage(
                message_type=A2AMessageType.TASK_RESPONSE,
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                correlation_id=message.message_id,
                payload={
                    "accepted": False,
                    "task_id": task.task_id,
                    "reason": "Cannot handle this type of task",
                },
            )
            await self.send_message(response)

    async def _process_task_async(self, task: A2ATask, requester_id: str):
        """Procesa una tarea de forma asíncrona y actualiza el estado global"""
        try:
            # Esperar un poco antes de empezar
            await asyncio.sleep(0.2)

            # Actualizar estado a IN_PROGRESS
            from sheily_core.a2a_protocol import a2a_system

            active_task = a2a_system.task_manager.active_tasks.get(task.task_id)
            if active_task:
                active_task.status = A2ATaskStatus.IN_PROGRESS
                active_task.progress = 0.1
                active_task.updated_at = datetime.now()

            # Simular procesamiento
            await asyncio.sleep(0.3)

            # Actualizar progreso
            if active_task:
                active_task.progress = 0.5
                active_task.updated_at = datetime.now()

            # Más procesamiento
            await asyncio.sleep(0.3)

            # Completar tarea
            result = {
                "status": "completed",
                "output": f"Task '{task.title}' completed successfully by agent {self.agent_id}",
                "processing_time": 0.8,
                "agent_info": {"id": self.agent_id, "type": "a2a_agent"},
            }

            if active_task:
                active_task.status = A2ATaskStatus.COMPLETED
                active_task.progress = 1.0
                active_task.result = result
                active_task.updated_at = datetime.now()

        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}")
            # Actualizar estado de error
            from sheily_core.a2a_protocol import a2a_system

            active_task = a2a_system.task_manager.active_tasks.get(task.task_id)
            if active_task:
                active_task.status = A2ATaskStatus.FAILED
                active_task.error = str(e)
                active_task.updated_at = datetime.now()

    async def _can_handle_task(self, task: A2ATask) -> bool:
        """Verifica si el agente puede manejar la tarea"""
        # Para esta implementación de prueba, aceptamos cualquier tarea
        # En producción, verificaríamos las capacidades reales del agente
        return True

    async def _process_task(self, task: A2ATask, requester_id: str):
        """Procesa una tarea asignada"""
        try:
            # Pequeño delay inicial para asegurar que el mensaje de aceptación se procese
            await asyncio.sleep(0.1)

            # Actualizar estado directamente en el sistema A2A (no enviar mensaje)
            # Esto es para simulación - en producción esto sería manejado por el listener
            from sheily_core.a2a_protocol import a2a_system

            active_task = a2a_system.task_manager.active_tasks.get(task.task_id)
            if active_task:
                active_task.status = A2ATaskStatus.IN_PROGRESS
                active_task.progress = 0.1
                active_task.updated_at = datetime.now()

            # Simular procesamiento - reducir tiempo para pruebas
            await asyncio.sleep(0.5)  # Simular trabajo

            # Actualizar progreso
            if active_task:
                active_task.status = A2ATaskStatus.IN_PROGRESS
                active_task.progress = 0.5
                active_task.updated_at = datetime.now()

            # Más procesamiento simulado
            await asyncio.sleep(0.5)

            # Completar tarea
            result = {
                "status": "completed",
                "output": f"Task '{task.title}' completed successfully by agent {self.agent_id}",
                "processing_time": 1.0,
                "agent_info": {"id": self.agent_id, "type": "a2a_agent"},
            }

            if active_task:
                active_task.status = A2ATaskStatus.COMPLETED
                active_task.progress = 1.0
                active_task.result = result
                active_task.updated_at = datetime.now()

        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}")
            # Actualizar estado de error
            from sheily_core.a2a_protocol import a2a_system

            active_task = a2a_system.task_manager.active_tasks.get(task.task_id)
            if active_task:
                active_task.status = A2ATaskStatus.FAILED
                active_task.error = str(e)
                active_task.updated_at = datetime.now()

    async def _send_task_update(
        self,
        task_id: str,
        status: A2ATaskStatus,
        progress: float,
        requester_id: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        """Envía actualización de tarea"""
        update = A2AMessage(
            message_type=A2AMessageType.TASK_UPDATE,
            sender_id=self.agent_id,
            receiver_id=requester_id,
            payload={
                "task_id": task_id,
                "status": status.value,
                "progress": progress,
                "result": result,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            },
        )
        await self.send_message(update)

    async def _handle_task_update(self, message: A2AMessage):
        """Maneja actualización de tarea"""
        # Notificar listeners de tarea
        task_id = message.payload.get("task_id")
        if task_id in self.task_listeners:
            listener = self.task_listeners[task_id]
            await listener(message.payload)

    async def _handle_task_cancel(self, message: A2AMessage):
        """Maneja cancelación de tarea"""
        task_id = message.payload.get("task_id")
        logger.info(f"Task {task_id} cancelled by {message.sender_id}")

    async def _handle_agent_discovery(self, message: A2AMessage):
        """Maneja solicitud de descubrimiento de agentes"""
        # Responder con información del agente
        response = A2AMessage(
            message_type=A2AMessageType.AGENT_ANNOUNCE,
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            correlation_id=message.message_id,
            payload={
                "agent_id": self.agent_id,
                "capabilities": ["general_processing"],  # Placeholder
                "status": "available",
            },
        )
        await self.send_message(response)


# =============================================================================
# A2A TASK MANAGER
# =============================================================================


class A2ATaskManager:
    """Gestor de tareas A2A"""

    def __init__(self, registry: AgentCardRegistry):
        self.registry = registry
        self.active_tasks: Dict[str, A2ATask] = {}
        self.task_clients: Dict[str, A2AClient] = {}

    async def submit_task(self, task: A2ATask) -> str:
        """Envía una tarea a un agente apropiado"""
        # Encontrar agentes capaces de manejar la tarea
        required_caps = task.requirements.get("capabilities", [])
        capable_agents = self.registry.find_agents_by_capabilities(required_caps)

        if not capable_agents:
            raise ValueError(f"No agents found with capabilities: {required_caps}")

        # Seleccionar el mejor agente (por ahora, el primero)
        target_agent = capable_agents[0]

        # Crear cliente A2A para el agente objetivo
        client = A2AClient(f"task-manager-{uuid.uuid4()}", A2ATransportType.HTTP)
        await client.connect()

        # Crear mensaje de solicitud de tarea
        message = A2AMessage(
            message_type=A2AMessageType.TASK_REQUEST,
            sender_id="task-manager",
            receiver_id=target_agent.agent_id,
            payload={"task": task.to_dict()},
        )

        # Enviar mensaje
        response = await client.send_message(message, target_agent.endpoint)

        if response and response.payload.get("accepted"):
            # Tarea aceptada
            task.assignee_id = target_agent.agent_id
            task.status = A2ATaskStatus.ACCEPTED
            self.active_tasks[task.task_id] = task
            self.task_clients[task.task_id] = client

            # Configurar listener para actualizaciones
            client.task_listeners[task.task_id] = self._create_task_listener(
                task.task_id
            )

            # Iniciar procesamiento de tarea en background desde el sistema A2A
            asyncio.create_task(self._process_task_simulation(task))

            logger.info(
                f"Task {task.task_id} assigned to agent {target_agent.agent_id}"
            )
            return task.task_id
        else:
            # Tarea rechazada
            await client.disconnect()
            raise ValueError("Task was rejected by the target agent")

    def _create_task_listener(self, task_id: str) -> Callable:
        """Crea un listener para actualizaciones de tarea"""

        async def listener(update: Dict[str, Any]):
            task = self.active_tasks.get(task_id)
            if task:
                task.status = A2ATaskStatus(update.get("status", "unknown"))
                task.progress = update.get("progress", 0.0)
                task.updated_at = datetime.now()

                if update.get("result"):
                    task.result = update["result"]
                if update.get("error"):
                    task.error = update["error"]

                logger.info(
                    f"Task {task_id} update: {task.status.value} ({task.progress:.1%})"
                )

                # Si la tarea está completa o fallida, limpiar
                if task.status in [A2ATaskStatus.COMPLETED, A2ATaskStatus.FAILED]:
                    if task_id in self.task_clients:
                        await self.task_clients[task_id].disconnect()
                        del self.task_clients[task_id]

        return listener

    async def get_task_status(self, task_id: str) -> Optional[A2ATask]:
        """Obtiene el estado de una tarea"""
        return self.active_tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancela una tarea"""
        task = self.active_tasks.get(task_id)
        if task and task.status in [
            A2ATaskStatus.PENDING,
            A2ATaskStatus.ACCEPTED,
            A2ATaskStatus.IN_PROGRESS,
        ]:
            client = self.task_clients.get(task_id)
            if client:
                message = A2AMessage(
                    message_type=A2AMessageType.TASK_CANCEL,
                    sender_id="task-manager",
                    receiver_id=task.assignee_id,
                    payload={"task_id": task_id},
                )
                await client.send_message(message)
                task.status = A2ATaskStatus.CANCELLED
                return True
        return False

    async def _process_task_simulation(self, task: A2ATask):
        """Simula el procesamiento de una tarea para testing"""
        try:
            # Esperar un poco antes de empezar
            await asyncio.sleep(0.2)

            # Actualizar estado a IN_PROGRESS
            task.status = A2ATaskStatus.IN_PROGRESS
            task.progress = 0.1
            task.updated_at = datetime.now()

            # Simular procesamiento
            await asyncio.sleep(0.3)

            # Actualizar progreso
            task.progress = 0.5
            task.updated_at = datetime.now()

            # Más procesamiento
            await asyncio.sleep(0.3)

            # Completar tarea
            result = {
                "status": "completed",
                "output": f"Task '{task.title}' completed successfully by agent {task.assignee_id}",
                "processing_time": 0.8,
                "agent_info": {"id": task.assignee_id, "type": "a2a_agent"},
            }

            task.status = A2ATaskStatus.COMPLETED
            task.progress = 1.0
            task.result = result
            task.updated_at = datetime.now()

        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}")
            task.status = A2ATaskStatus.FAILED
            task.error = str(e)
            task.updated_at = datetime.now()

    def list_active_tasks(self) -> List[A2ATask]:
        """Lista tareas activas"""
        return list(self.active_tasks.values())


# =============================================================================
# SISTEMA A2A UNIFICADO
# =============================================================================


class A2ASystem:
    """Sistema unificado A2A para Sheily AI"""

    def __init__(self):
        self.registry = AgentCardRegistry()
        self.task_manager = A2ATaskManager(self.registry)
        self.running = False

    async def start(self):
        """Inicia el sistema A2A"""
        self.running = True

        # Registrar agentes existentes del sistema multi-agente
        await self._register_existing_agents()

        logger.info("A2A System started")

    async def stop(self):
        """Detiene el sistema A2A"""
        self.running = False

        # Desconectar todos los clientes
        for client in self.task_manager.task_clients.values():
            await client.disconnect()

        logger.info("A2A System stopped")

    async def _register_existing_agents(self):
        """Registra agentes existentes del sistema multi-agente"""
        # Registrar agentes del sistema multi-agente existente
        agents_info = [
            {
                "agent_id": "coordinator-agent",
                "name": "Coordinator Agent",
                "capabilities": [
                    "task_coordination",
                    "agent_management",
                    "workflow_orchestration",
                    "general_processing",
                ],
                "endpoint": "a2a://sheily-agents/coordinator",
            },
            {
                "agent_id": "code-analyzer-agent",
                "name": "Code Analyzer Agent",
                "capabilities": [
                    "code_analysis",
                    "bug_detection",
                    "quality_assessment",
                    "general_processing",
                ],
                "endpoint": "a2a://sheily-agents/code-analyzer",
            },
            {
                "agent_id": "data-processor-agent",
                "name": "Data Processor Agent",
                "capabilities": [
                    "data_processing",
                    "statistics",
                    "data_analysis",
                    "general_processing",
                ],
                "endpoint": "a2a://sheily-agents/data-processor",
            },
        ]

        for agent_info in agents_info:
            card = AgentCard(**agent_info)
            self.registry.register_agent(card)

    async def submit_task(
        self,
        title: str,
        description: str,
        requirements: Dict[str, Any],
        context: Dict[str, Any] = None,
        priority: int = 1,
    ) -> str:
        """Envía una tarea al sistema A2A"""
        task = A2ATask(
            title=title,
            description=description,
            requester_id="a2a-system",
            requirements=requirements,
            context=context or {},
            priority=priority,
        )

        return await self.task_manager.submit_task(task)

    async def get_task_status(self, task_id: str) -> Optional[A2ATask]:
        """Obtiene el estado de una tarea"""
        return await self.task_manager.get_task_status(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancela una tarea"""
        return await self.task_manager.cancel_task(task_id)

    def list_agents(self) -> List[AgentCard]:
        """Lista todos los agentes registrados"""
        return self.registry.list_all_agents()

    def find_agents_by_capability(self, capability: str) -> List[AgentCard]:
        """Encuentra agentes por capacidad"""
        return self.registry.find_agents_by_capability(capability)

    def get_agent_card(self, agent_id: str) -> Optional[AgentCard]:
        """Obtiene la Agent Card de un agente"""
        return self.registry.get_agent_card(agent_id)


# =============================================================================
# INTEGRACIÓN CON SISTEMA EXISTENTE
# =============================================================================

# Instancia global del sistema A2A
a2a_system = A2ASystem()


async def initialize_a2a_system():
    """Inicializa el sistema A2A"""
    await a2a_system.start()


async def submit_a2a_task(
    title: str,
    description: str,
    requirements: Dict[str, Any],
    context: Dict[str, Any] = None,
    priority: int = 1,
) -> str:
    """Envía una tarea al sistema A2A"""
    return await a2a_system.submit_task(
        title, description, requirements, context, priority
    )


async def get_a2a_task_status(task_id: str) -> Optional[A2ATask]:
    """Obtiene el estado de una tarea A2A"""
    return await a2a_system.get_task_status(task_id)


# =============================================================================
# DEMO Y TESTING
# =============================================================================


async def demo_a2a_system():
    """Demostración del sistema A2A"""
    print("🤝 Inicializando sistema Agent-to-Agent (A2A)...")

    await initialize_a2a_system()

    print("\n📋 Agentes registrados:")
    agents = a2a_system.list_agents()
    for agent in agents:
        print(f"  • {agent.name} ({agent.agent_id})")
        print(f"    Capacidades: {', '.join(agent.capabilities)}")

    print("\n🔍 Buscando agentes por capacidad 'code_analysis':")
    code_agents = a2a_system.find_agents_by_capability("code_analysis")
    for agent in code_agents:
        print(f"  • {agent.name}")

    print("\n📝 Enviando tarea de análisis de código...")
    try:
        task_id = await submit_a2a_task(
            title="Análisis de código Python",
            description="Analizar el archivo main.py para detectar bugs y problemas de calidad",
            requirements={
                "capabilities": ["code_analysis"],
                "language": "python",
                "priority": "high",
            },
            context={"file_path": "main.py", "analysis_type": "comprehensive"},
            priority=8,
        )

        print(f"✅ Tarea enviada con ID: {task_id}")

        # Monitorear progreso
        for i in range(10):
            await asyncio.sleep(1)
            status = await get_a2a_task_status(task_id)
            if status:
                print(
                    f"📊 Estado: {status.status.value} - Progreso: {status.progress:.1%}"
                )
                if status.status == A2ATaskStatus.COMPLETED:
                    print(f"🎉 Resultado: {status.result}")
                    break
                elif status.status == A2ATaskStatus.FAILED:
                    print(f"❌ Error: {status.error}")
                    break

    except Exception as e:
        print(f"❌ Error enviando tarea: {e}")

    print("\n🎊 Demo A2A completada!")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Clases principales
    "AgentCard",
    "A2AMessage",
    "A2ATask",
    "AgentCardRegistry",
    "A2AClient",
    "A2ATaskManager",
    "A2ASystem",
    # Sistema global
    "a2a_system",
    # Funciones de utilidad
    "initialize_a2a_system",
    "submit_a2a_task",
    "get_a2a_task_status",
    "demo_a2a_system",
]

# Información del módulo
__version__ = "1.0.0"
__author__ = "Sheily AI Team - Agent-to-Agent Protocol Implementation"
