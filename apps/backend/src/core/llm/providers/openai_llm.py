"""
OpenAI LLM Provider - Implementación para GPT-4 y otros modelos OpenAI
"""

import logging
from typing import Any, Dict, Optional

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..llm_interface import LLMInterface

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """
    Implementación de LLMInterface para modelos OpenAI
    Soporta GPT-4, GPT-3.5-turbo y otros modelos de OpenAI
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ):
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI package no instalado. Funciones limitadas disponibles.")
            self.client = None
            self.api_key = None
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if self.api_key:
                # Inicializar cliente OpenAI
                self.client = OpenAI(api_key=self.api_key)
            else:
                logger.warning("OpenAI API key no configurada. Configura OPENAI_API_KEY.")
                self.client = None

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs
        self._initialized = False

        logger.info(f"OpenAI LLM inicializado con modelo: {model}")

    async def initialize(self) -> bool:
        """Verificar que OpenAI esté disponible y configurado"""
        try:
            if not self.api_key:
                logger.error("OpenAI API key no configurada")
                return False

            # Verificación básica - intentar hacer una llamada simple
            # Solo si no ha sido verificado antes
            if not self._initialized:
                # Esta verificación se puede omitir si es muy costosa
                # Por ahora solo validamos que el cliente esté creado
                logger.info("Verificando credenciales OpenAI...")
                # TODO: En producción, hacer una verificación real barata

            self._initialized = True
            logger.info("OpenAI LLM inicializado exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error inicializando OpenAI: {e}")
            return False

    async def generate_response(
        self,
        message: str,
        context: str = "",
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generar respuesta usando OpenAI
        """
        try:
            # Usar parámetros del método o los de instancia
            temp = kwargs.get("temperature", temperature if temperature is not None else self.temperature)
            max_tok = kwargs.get("max_tokens", max_tokens if max_tokens is not None else self.max_tokens)

            # Preparar system prompt
            if system_prompt is None:
                system_prompt = """Eres un asistente de IA inteligente y útil creado por OpenAI.
Responde de manera clara, precisa y contextual. Si tienes información adicional del contexto proporcionado,
utilízala para dar respuestas más informadas y útiles.

Instrucciones importantes:
- Sé amable y servicial
- Proporciona respuestas precisas y bien fundamentadas
- Si no sabes algo, admítelo honestamente
- Mantén un tono conversacional y natural"""

            # Añadir contexto si existe
            if context:
                system_prompt += f"\n\nContexto adicional:\n{context}"

            # Crear mensajes
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]

            logger.info(f"[OpenAI] Generando respuesta con {self.model}...")

            # Llamar a OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tok,
                temperature=temp,
                **self.extra_params
            )

            # Extraer respuesta
            generated_text = response.choices[0].message.content.strip()
            logger.info(f"[OpenAI] Respuesta generada exitosamente: {generated_text[:100]}...")

            return generated_text

        except Exception as e:
            logger.error(f"[OpenAI] Error generando respuesta: {e}")
            raise RuntimeError(f"Error en generación OpenAI: {str(e)}")

    async def is_ready(self) -> bool:
        """Verificar si OpenAI está listo"""
        return self._initialized and self.client is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Información del modelo OpenAI"""
        return {
            "provider": "openai",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "initialized": self._initialized,
            "api_key_configured": bool(self.api_key),
        }
