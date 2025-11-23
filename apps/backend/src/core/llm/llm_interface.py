"""
Interfaz abstracta para todos los proveedores LLM - Arquitrectura Polymorphic
Permite cambiar entre diferentes LLMs (LLAMA, OpenAI, Google, Anthropic, etc.)
sin modificar el código que los usa
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

class LLMInterface(ABC):
    """
    Interfaz común para todos los proveedores de LLM
    Define el contrato que deben implementar todos los LLMs
    """

    @abstractmethod
    async def generate_response(
        self,
        message: str,
        context: str = "",
        system_prompt: str = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generar respuesta usando el LLM específico

        Args:
            message: Mensaje del usuario
            context: Contexto adicional (RAG)
            system_prompt: Prompt del sistema personalizado
            max_tokens: Número máximo de tokens en respuesta
            temperature: Nivel de creatividad (0.0-1.0)
            **kwargs: Parámetros adicionales específicos del provider

        Returns:
            str: Respuesta generada por el LLM

        Raises:
            RuntimeError: Si el modelo no está disponible
        """
        pass

    @abstractmethod
    async def is_ready(self) -> bool:
        """
        Verificar si el modelo está listo para usar

        Returns:
            bool: True si el modelo está inicializado y listo
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtener información del modelo

        Returns:
            Dict con información específica del modelo/provider
        """
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Inicializar el modelo/provider

        Returns:
            bool: True si la inicialización fue exitosa
        """
        pass
