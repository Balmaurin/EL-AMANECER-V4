"""
Implementación Local LLM - Generic GGUF via llama-cpp-python
Provider específico que implementa LLMInterface para modelos locales GGUF
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from ..llm_interface import LLMInterface
from ....core.config.settings import settings

logger = logging.getLogger(__name__)


class LocalLLM(LLMInterface):
    """
    Servicio para interactuar con modelos locales
    """

    def __init__(self):
        self.llm: Optional[Llama] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Inicializar el modelo local

        Returns:
            bool: True si la inicialización fue exitosa
        """
        if self._initialized:
            logger.info("Modelo local ya inicializado")
            return True

        try:
            # Verificar que llama_cpp está disponible
            if Llama is None:
                logger.error(
                    "llama-cpp-python NO está instalado. Instala: pip install llama-cpp-python"
                )
                return False

            # Verificar que el modelo existe
            model_path = Path(settings.llm.model_path)
            logger.info(f"Buscando modelo en: {model_path}")

            if not model_path.exists():
                logger.error(f"[ERROR] Modelo NO encontrado en: {model_path}")
                logger.error(f"Verifica que el archivo existe y la ruta es correcta")
                return False

            logger.info(
                f"[OK] Modelo encontrado: {model_path.name} ({model_path.stat().st_size / 1024 / 1024:.0f} MB)"
            )
            logger.info(
                "[LOADING] Cargando modelo LLAMA 3 en memoria (esto puede tardar 10-30 segundos)..."
            )

            # Inicializar modelo LLAMA 3
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=settings.llm.n_ctx,
                n_threads=settings.llm.n_threads,
                chat_format=settings.llm.chat_format,
                verbose=settings.llm.verbose,
            )

            self._initialized = True
            logger.info("[OK] Modelo LLAMA 3 cargado exitosamente en memoria")
            logger.info(
                f"Configuración: n_ctx={settings.llm.n_ctx}, n_threads={settings.llm.n_threads}, chat_format={settings.llm.chat_format}"
            )
            return True

        except Exception as e:
            logger.error(f"[ERROR] ERROR CRITICO inicializando LLAMA 3: {e}")
            logger.error(f"Tipo de error: {type(e).__name__}")
            import traceback

            logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            return False

    async def generate_response(
        self,
        message: str,
        context: str = "",
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generar respuesta usando LLAMA 3 REAL - NO MOCKS

        Args:
            message: Mensaje del usuario
            context: Contexto adicional (RAG)
            system_prompt: Prompt del sistema personalizado
            **kwargs: Parámetros adicionales para el modelo

        Returns:
            str: Respuesta generada

        Raises:
            RuntimeError: Si el modelo no está disponible
        """
        if not self.llm:
            if not await self.initialize():
                raise RuntimeError(
                    "Modelo LLAMA no disponible. Verifica: "
                    "1) Ruta del modelo correcta, "
                    "2) llama-cpp-python instalado, "
                    "3) Suficiente memoria RAM"
                )

        try:
            # Prompt del sistema por defecto
            if system_prompt is None:
                system_prompt = """Eres LLAMA 3, un asistente de IA inteligente y útil creado por Meta.  # noqa: E501
Responde de manera clara, precisa y contextual. Si tienes información adicional del contexto proporcionado,
utilízala para dar respuestas más informadas y útiles.

Instrucciones importantes:
- Sé amable y servicial
- Proporciona respuestas precisas y bien fundamentadas
- Si no sabes algo, admítelo honestamente
- Mantén un tono conversacional y natural
- Usa el contexto proporcionado cuando sea relevante"""

            # Añadir contexto si existe
            if context:
                system_prompt += f"\n\nContexto adicional:\n{context}"

            # Crear conversación para LLAMA 3
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ]

            # Parámetros de generación
            generation_params = {
                "max_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
                "stop": kwargs.get("stop", ["User:", "System:", "Assistant:"]),
            }

            logger.info(f"[LLM] Llamando a LLAMA LLM con mensaje: {message[:50]}...")

            # Generar respuesta REAL con el modelo
            response = self.llm.create_chat_completion(
                messages=messages, **generation_params
            )

            if response and "choices" in response and response["choices"]:
                generated_text = response["choices"][0]["message"]["content"].strip()
                logger.info(
                    f"[OK] Respuesta generada por LLAMA LLM: {generated_text[:100]}..."
                )
                return generated_text
            else:
                raise RuntimeError("El modelo no devolvió una respuesta válida")

        except Exception as e:
            logger.error(f"[ERROR] Error generando respuesta con LLAMA 3: {e}")
            raise RuntimeError(f"Error en generación LLM: {str(e)}")

    async def is_ready(self) -> bool:
        """
        Verificar si el modelo está listo para usar

        Returns:
            bool: True si el modelo está inicializado
        """
        return self._initialized and self.llm is not None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtener información del modelo

        Returns:
            Dict con información del modelo
        """
        return {
            "model_path": settings.llm.model_path,
            "n_ctx": settings.llm.n_ctx,
            "n_threads": settings.llm.n_threads,
            "chat_format": settings.llm.chat_format,
            "initialized": self._initialized,
            "ready": self.llm is not None,
        }
