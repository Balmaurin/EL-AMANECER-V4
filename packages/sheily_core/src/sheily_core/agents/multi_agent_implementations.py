#!/usr/bin/env python3
"""
Implementaciones concretas de MultiAgentBase - Agentes colaborativos reales
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from .multi_agent_system import AgentMessage, AgentRole, MultiAgentBase


class TaskExecutorAgent(MultiAgentBase):
    """Agente que ejecuta tareas específicas"""

    def __init__(self):
        super().__init__(
            agent_id="task_executor_001", name="Task Executor", role=AgentRole.WORKER
        )
        self.completed_tasks = []

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Procesar mensajes de otros agentes"""
        if message.message_type == "task_request":
            result = await self.execute_task_from_message(message.content)

            return AgentMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type="task_response",
                content=result,
                timestamp=datetime.now(),
            )
        return None

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar una tarea asignada"""
        task_type = task.get("type", "unknown")

        if task_type == "data_processing":
            return await self.process_data(task.get("data", {}))
        elif task_type == "file_operation":
            return await self.handle_file_operation(task.get("operation", {}))
        else:
            return {"status": "error", "message": f"Unknown task type: {task_type}"}

    async def execute_task_from_message(
        self, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ejecutar tarea desde mensaje"""
        start_time = datetime.now()

        try:
            result = await self.execute_task(content)

            # Registrar tarea completada
            self.completed_tasks.append(
                {
                    "task": content,
                    "result": result,
                    "completed_at": datetime.now().isoformat(),
                    "duration": (datetime.now() - start_time).total_seconds(),
                }
            )

            # Actualizar estadísticas
            self.update_performance_stats(
                (datetime.now() - start_time).total_seconds(),
                result.get("status") == "success",
            )

            return result

        except Exception as e:
            error_result = {"status": "error", "message": str(e)}
            self.update_performance_stats(
                (datetime.now() - start_time).total_seconds(), False
            )
            return error_result

    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Procesar datos específicos"""
        # Simulación de procesamiento de datos
        await asyncio.sleep(0.1)  # Simular trabajo

        processed_count = len(data.get("items", []))

        return {
            "status": "success",
            "processed_items": processed_count,
            "result": f"Processed {processed_count} items",
        }

    async def handle_file_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Manejar operaciones de archivo"""
        op_type = operation.get("type", "read")
        file_path = operation.get("path", "")

        try:
            if op_type == "read" and file_path:
                # Simular lectura de archivo
                return {
                    "status": "success",
                    "operation": "file_read",
                    "path": file_path,
                    "result": f"Read file: {file_path}",
                }
            else:
                return {"status": "error", "message": "Invalid file operation"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


class CommunicationAgent(MultiAgentBase):
    """Agente que maneja comunicación entre otros agentes"""

    def __init__(self):
        super().__init__(
            agent_id="communication_hub_001",
            name="Communication Hub",
            role=AgentRole.COORDINATOR,
        )
        self.message_history = []
        self.active_conversations = {}

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Procesar y enrutar mensajes"""
        # Guardar mensaje en historial
        self.message_history.append(
            {"message": message, "processed_at": datetime.now().isoformat()}
        )

        # Mantener solo últimos 100 mensajes
        if len(self.message_history) > 100:
            self.message_history = self.message_history[-100:]

        # Procesar según tipo de mensaje
        if message.message_type == "broadcast":
            return await self.handle_broadcast(message)
        elif message.message_type == "direct":
            return await self.handle_direct_message(message)
        else:
            return await self.handle_unknown_message(message)

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar tareas de comunicación"""
        task_type = task.get("type", "unknown")

        if task_type == "send_notification":
            return await self.send_notification(task.get("notification", {}))
        elif task_type == "broadcast_message":
            return await self.broadcast_message(task.get("message", {}))
        else:
            return {
                "status": "error",
                "message": f"Unknown communication task: {task_type}",
            }

    async def handle_broadcast(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Manejar mensajes de difusión"""
        # Simular envío a múltiples agentes
        return AgentMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type="broadcast_ack",
            content={"status": "broadcast_sent", "recipients": "all_agents"},
            timestamp=datetime.now(),
        )

    async def handle_direct_message(
        self, message: AgentMessage
    ) -> Optional[AgentMessage]:
        """Manejar mensajes directos"""
        return AgentMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type="direct_ack",
            content={"status": "message_delivered"},
            timestamp=datetime.now(),
        )

    async def handle_unknown_message(
        self, message: AgentMessage
    ) -> Optional[AgentMessage]:
        """Manejar mensajes desconocidos"""
        return AgentMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type="error",
            content={
                "status": "unknown_message_type",
                "original_type": message.message_type,
            },
            timestamp=datetime.now(),
        )

    async def send_notification(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """Enviar notificación"""
        return {
            "status": "success",
            "notification_sent": True,
            "recipients": notification.get("recipients", []),
            "message": notification.get("message", ""),
        }

    async def broadcast_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Difundir mensaje a todos los agentes"""
        return {
            "status": "success",
            "broadcast_sent": True,
            "message": message.get("content", ""),
            "timestamp": datetime.now().isoformat(),
        }

    def get_communication_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de comunicación"""
        return {
            "total_messages_processed": len(self.message_history),
            "active_conversations": len(self.active_conversations),
            "last_activity": (
                self.message_history[-1]["processed_at"]
                if self.message_history
                else None
            ),
            "performance_stats": self.performance_stats,
        }


# CONVERTIR MultiAgentBase en agentes funcionales
task_executor_agent = TaskExecutorAgent()
communication_agent = CommunicationAgent()
