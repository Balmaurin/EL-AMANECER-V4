"""
Servicio de Chat - Orquesta la interacción entre LLM y RAG
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, Optional

from apps.backend.src.core.llm.llm_interface import LLMInterface
from apps.backend.src.core.rag.service import RAGService

logger = logging.getLogger(__name__)


class ChatService:
    """
    Servicio que maneja la lógica de chat, combinando LLM y RAG
    Compatible con tanto ChromaDB como TF-IDF+SVD RAG services
    """

    def __init__(self, llm: LLMInterface, rag_service):
        self.llm = llm
        self.rag_service = rag_service

    async def initialize(self) -> bool:
        """
        Inicializar el servicio de chat

        Returns:
            bool: True si la inicialización fue exitosa
        """
        # Los servicios core ya se inicializan en main.py
        return True

    async def process_message(
        self, conversation_id: str, user_message: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Procesar un mensaje del usuario

        Args:
            conversation_id: ID de la conversación
            user_message: Mensaje del usuario
            **kwargs: Parámetros adicionales

        Returns:
            Dict con la respuesta y metadatos
        """
        try:
            # Recuperar contexto relevante con RealRAGService (TF-IDF+SVD)
            context = ""
            try:
                if self.rag_service and hasattr(self.rag_service, 'search'):
                    # RealRAGService usa search() que es async y retorna dict
                    search_results = await self.rag_service.search(
                        user_message,
                        top_k=kwargs.get("rag_top_k", 3)
                    )

                    # Extraer contexto de resultados - filtra por similitud
                    context_docs = []
                    if search_results and "results" in search_results:
                        for result in search_results["results"]:
                            similarity = result.get("similarity", 0)
                            # Solo incluir documentos relevantes (similarity > 0.1)
                            if similarity > 0.1:
                                doc_text = result.get("document", "")
                                if doc_text:
                                    context_docs.append(doc_text)

                    # Unir documentos relevantes con separadores
                    context = "\n\n".join(context_docs)

            except Exception as e:
                logger.warning(f"RAG search failed: {e}")
                context = ""

            # Generar respuesta con LLM polymorphic
            response_text = await self.llm.generate_response(
                message=user_message,
                context=context,
                system_prompt=kwargs.get("system_prompt"),
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
            )

            # Crear respuesta
            timestamp = asyncio.get_event_loop().time()

            response_data = {
                "response": response_text,
                "conversation_id": conversation_id,
                "context_used": len(context.strip()) > 0,
                "context_length": len(context),
                "timestamp": timestamp,
                "model_info": self.llm.get_model_info(),
                "rag_info": self.rag_service.get_collection_info(),
            }

            logger.info(
                f"Mensaje procesado para conversación {conversation_id}: {len(response_text)} caracteres"  # noqa: E501
            )

            return response_data

        except Exception as e:
            logger.error(f"Error procesando mensaje: {e}")
            return {
                "error": "Error interno del servidor",
                "conversation_id": conversation_id,
                "timestamp": asyncio.get_event_loop().time(),
            }

    async def stream_message(
        self, conversation_id: str, user_message: str, **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Procesar un mensaje con streaming (para futuras implementaciones)

        Args:
            conversation_id: ID de la conversación
            user_message: Mensaje del usuario
            **kwargs: Parámetros adicionales

        Yields:
            Dict con chunks de la respuesta
        """
        # Por ahora, devolver la respuesta completa
        # En el futuro, implementar streaming real con el LLM
        response = await self.process_message(conversation_id, user_message, **kwargs)
        yield response

    async def get_chat_info(self) -> Dict[str, Any]:
        """
        Obtener información del servicio de chat

        Returns:
            Dict con información del servicio
        """
        return {
            "llm_ready": await self.llm.is_ready(),
            "rag_ready": await self.rag_service.is_ready(),
            "llm_info": self.llm.get_model_info(),
            "rag_info": self.rag_service.get_collection_info(),
        }

    async def validate_message(self, message: str) -> Dict[str, Any]:
        """
        Validar un mensaje antes del procesamiento

        Args:
            message: Mensaje a validar

        Returns:
            Dict con resultado de validación
        """
        if not message or not message.strip():
            return {"valid": False, "error": "Mensaje vacío"}

        if len(message) > 10000:  # Límite arbitrario
            return {"valid": False, "error": "Mensaje demasiado largo"}

        return {"valid": True, "length": len(message)}

    async def generate_system_prompt(
        self, conversation_context: Optional[str] = None, **kwargs
    ) -> str:
        """
        Generar un prompt del sistema personalizado

        Args:
            conversation_context: Contexto de la conversación
            **kwargs: Parámetros adicionales

        Returns:
            str: Prompt del sistema
        """
        base_prompt = """Eres un asistente de IA inteligente y útil.
Responde de manera clara, precisa y contextual. Si tienes información adicional del contexto proporcionado,
utilízala para dar respuestas más informadas y útiles.

Instrucciones importantes:
- Sé amable y servicial
- Proporciona respuestas precisas y bien fundamentadas
- Si no sabes algo, admítelo honestamente
- Mantén un tono conversacional y natural
- Usa el contexto proporcionado cuando sea relevante"""

        if conversation_context:
            base_prompt += f"\n\nContexto de la conversación:\n{conversation_context}"

        # Añadir instrucciones específicas si se proporcionan
        custom_instructions = kwargs.get("custom_instructions")
        if custom_instructions:
            base_prompt += f"\n\nInstrucciones específicas:\n{custom_instructions}"

        return base_prompt
