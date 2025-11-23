"""
Servicio de Conversaciones - Gestión del historial de chat
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.settings import settings

logger = logging.getLogger(__name__)


class ConversationService:
    """
    Servicio para gestionar conversaciones y su persistencia
    """

    def __init__(self):
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Inicializar el servicio de conversaciones

        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            await self._load_conversations()
            self._initialized = True
            logger.info(
                f"Conversaciones cargadas: {len(self.conversations)} conversaciones"  # noqa: E501
            )
            return True
        except Exception as e:
            logger.error(f"Error inicializando servicio de conversaciones: {e}")
            return False

    async def _load_conversations(self):
        """Cargar conversaciones desde archivo"""
        conversations_file = Path(settings.database.conversations_file)

        if conversations_file.exists():
            try:
                with open(conversations_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.conversations = data.get("conversations", {})
                logger.info(f"Conversaciones cargadas desde {conversations_file}")
            except Exception as e:
                logger.error(f"Error cargando conversaciones: {e}")
                self.conversations = {}
        else:
            logger.info("Archivo de conversaciones no encontrado, iniciando vacío")
            self.conversations = {}

    async def _save_conversations(self):
        """Guardar conversaciones a archivo"""
        try:
            conversations_file = Path(settings.database.conversations_file)
            conversations_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "conversations": self.conversations,
                "last_updated": datetime.now().isoformat(),
                "total_conversations": len(self.conversations),
                "total_messages": sum(
                    len(msgs) for msgs in self.conversations.values()
                ),
            }

            with open(conversations_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error guardando conversaciones: {e}")

    async def add_message(
        self, conversation_id: str, role: str, content: str, **metadata
    ) -> Dict[str, Any]:
        """
        Añadir un mensaje a una conversación

        Args:
            conversation_id: ID de la conversación
            role: Rol del mensaje ('user' o 'assistant')
            content: Contenido del mensaje
            **metadata: Metadatos adicionales

        Returns:
            Dict con el mensaje creado
        """
        # Inicializar conversación si no existe
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        # Crear mensaje
        timestamp = asyncio.get_event_loop().time()
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            **metadata,
        }

        # Añadir a la conversación
        self.conversations[conversation_id].append(message)

        # Limitar historial para evitar sobrecarga de memoria
        max_messages = 100  # Configurable
        if len(self.conversations[conversation_id]) > max_messages:
            # Mantener los mensajes más recientes
            self.conversations[conversation_id] = self.conversations[conversation_id][
                -max_messages:
            ]

        # Guardar periódicamente (cada 10 mensajes o cuando se complete una conversación)  # noqa: E501
        if len(self.conversations[conversation_id]) % 10 == 0:
            await self._save_conversations()

        logger.debug(f"Mensaje añadido a conversación {conversation_id}: {role}")

        return message

    async def get_conversation_history(
        self, conversation_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtener el historial de una conversación

        Args:
            conversation_id: ID de la conversación
            limit: Número máximo de mensajes a devolver

        Returns:
            List de mensajes de la conversación
        """
        messages = self.conversations.get(conversation_id, [])

        if limit:
            messages = messages[-limit:]  # Últimos mensajes

        return messages

    async def list_conversations(
        self, limit: int = 50, include_message_count: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Listar todas las conversaciones disponibles

        Args:
            limit: Número máximo de conversaciones a devolver
            include_message_count: Incluir conteo de mensajes

        Returns:
            List de conversaciones con metadatos
        """
        conv_list = []

        for conv_id, messages in self.conversations.items():
            if messages:
                last_message = messages[-1]
                conv_data = {
                    "id": conv_id,
                    "last_message": last_message.get("content", "")[
                        :100
                    ],  # Primeros 100 caracteres
                    "timestamp": last_message.get("timestamp", 0),
                }

                if include_message_count:
                    conv_data["message_count"] = len(messages)

                conv_list.append(conv_data)

        # Ordenar por timestamp descendente (más recientes primero)
        conv_list.sort(key=lambda x: x["timestamp"], reverse=True)

        return conv_list[:limit]

    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Eliminar una conversación

        Args:
            conversation_id: ID de la conversación a eliminar

        Returns:
            bool: True si se eliminó exitosamente
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            await self._save_conversations()
            logger.info(f"Conversación eliminada: {conversation_id}")
            return True
        else:
            logger.warning(f"Conversación no encontrada: {conversation_id}")
            return False

    async def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de conversaciones

        Returns:
            Dict con estadísticas
        """
        total_conversations = len(self.conversations)
        total_messages = sum(len(msgs) for msgs in self.conversations.values())

        # Calcular estadísticas adicionales
        avg_messages_per_conv = (
            total_messages / total_conversations if total_conversations > 0 else 0
        )

        # Encontrar conversación más activa
        most_active_conv = None
        max_messages = 0
        for conv_id, messages in self.conversations.items():
            if len(messages) > max_messages:
                max_messages = len(messages)
                most_active_conv = conv_id

        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "avg_messages_per_conversation": round(avg_messages_per_conv, 2),
            "most_active_conversation": most_active_conv,
            "max_messages_in_conversation": max_messages,
        }

    async def search_conversations(
        self, query: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Buscar en las conversaciones

        Args:
            query: Término de búsqueda
            limit: Número máximo de resultados

        Returns:
            List de conversaciones que contienen el término
        """
        results = []
        query_lower = query.lower()

        for conv_id, messages in self.conversations.items():
            matching_messages = []

            for msg in messages:
                content = msg.get("content", "").lower()
                if query_lower in content:
                    matching_messages.append(
                        {
                            "content": msg.get("content", ""),
                            "role": msg.get("role", ""),
                            "timestamp": msg.get("timestamp", 0),
                        }
                    )

            if matching_messages:
                results.append(
                    {
                        "conversation_id": conv_id,
                        "matching_messages": matching_messages,
                        "total_matches": len(matching_messages),
                        "last_match_timestamp": matching_messages[-1]["timestamp"],
                    }
                )

        # Ordenar por timestamp del último match
        results.sort(key=lambda x: x["last_match_timestamp"], reverse=True)

        return results[:limit]

    async def cleanup_old_conversations(self, days: int = 30) -> int:
        """
        Limpiar conversaciones antiguas

        Args:
            days: Días de antigüedad para considerar como "antigua"

        Returns:
            int: Número de conversaciones eliminadas
        """
        cutoff_time = asyncio.get_event_loop().time() - (days * 24 * 60 * 60)
        conversations_to_delete = []

        for conv_id, messages in self.conversations.items():
            if messages:
                last_timestamp = messages[-1].get("timestamp", 0)
                if last_timestamp < cutoff_time:
                    conversations_to_delete.append(conv_id)

        # Eliminar conversaciones antiguas
        for conv_id in conversations_to_delete:
            del self.conversations[conv_id]

        if conversations_to_delete:
            await self._save_conversations()
            logger.info(
                f"Conversaciones antiguas eliminadas: {len(conversations_to_delete)}"  # noqa: E501
            )

        return len(conversations_to_delete)

    async def export_conversation(
        self, conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Exportar una conversación completa

        Args:
            conversation_id: ID de la conversación

        Returns:
            Dict con la conversación completa o None si no existe
        """
        if conversation_id not in self.conversations:
            return None

        messages = self.conversations[conversation_id]
        return {
            "conversation_id": conversation_id,
            "exported_at": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": messages,
        }
