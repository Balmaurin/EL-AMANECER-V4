"""
LLM Factory - Creación Polymorphic de LLMs
Permite crear instancias de diferentes proveedores LLM dinámicamente
"""

import logging
from typing import Any, Dict, Optional

from .llm_interface import LLMInterface

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory para crear instancias de diferentes proveedores LLM
    Soporta configuración vía environment variables o JSON config
    """

    @staticmethod
    def create_llm(config: Optional[Dict[str, Any]] = None) -> LLMInterface:
        """
        Crear instancia de LLM basado en configuración

        Args:
            config: Configuración del LLM (opcional)

        Returns:
            Instancia de LLMInterface implementada

        Raises:
            ValueError: Si provider no es válido o soportado
        """
        # Si no se pasa config, usar configuración por defecto (local LLAMA)
        if config is None:
            config = {"provider": "local"}

        # Determinar provider a usar
        provider = config.get("provider", "local").lower()

        logger.info(f"[LLM FACTORY] Creando LLM provider: {provider}")

        # Crear instancia según provider
        if provider == "local":
            # Import local solo cuando se necesita
            from .providers.local_llm import LocalLLM
            return LocalLLM()

        elif provider == "openai":
            # Import OpenAI solo cuando se necesita
            from .providers.openai_llm import OpenAILLM

            # Extraer configuración OpenAI
            openai_config = config.get("openai", {})
            api_key = openai_config.get("api_key", "")

            # Remover api_key del diccionario para evitar duplicación
            filtered_config = {k: v for k, v in openai_config.items() if k != "api_key"}

            return OpenAILLM(api_key=api_key, **filtered_config)

        elif provider == "google":
            # Import Google solo cuando se necesita
            from .providers.google_llm import GoogleLLM
            return GoogleLLM(**config.get("google", {}))

        elif provider == "anthropic":
            # Import Anthropic solo cuando se necesita
            from .providers.anthropic_llm import AnthropicLLM
            return AnthropicLLM(**config.get("anthropic", {}))

        else:
            raise ValueError(f"LLM Provider no soportado: {provider}")

    @staticmethod
    def get_available_providers() -> list[str]:
        """
        Obtener lista de proveedores disponibles

        Returns:
            Lista con nombres de providers soportados
        """
        return ["local", "openai", "google", "anthropic"]

    @staticmethod
    def load_config_from_env() -> Dict[str, Any]:
        """
        Cargar configuración LLM desde environment variables

        Returns:
            Configuración del LLM
        """
        import os

        config = {
            "provider": os.getenv("LLM_PROVIDER", "local"),
        }

        # Configuración específica por provider
        provider = config["provider"]

        if provider == "openai":
            config["openai"] = {
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "model": os.getenv("OPENAI_MODEL", "gpt-4"),
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "512")),
            }

        elif provider == "google":
            config["google"] = {
                "api_key": os.getenv("GOOGLE_API_KEY", ""),
                "model": os.getenv("GOOGLE_MODEL", "gemini-pro"),
                "temperature": float(os.getenv("GOOGLE_TEMPERATURE", "0.7")),
            }

        elif provider == "anthropic":
            config["anthropic"] = {
                "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
                "model": os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet"),
                "temperature": float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7")),
            }

        return config

    @staticmethod
    def load_config_from_file(filepath: str = "config/llm_config.json") -> Dict[str, Any]:
        """
        Cargar configuración LLM desde archivo JSON

        Args:
            filepath: Ruta al archivo de configuración

        Returns:
            Configuración del LLM
        """
        import json
        import os

        try:
            if not os.path.exists(filepath):
                logger.warning(f"Archivo de configuración no encontrado: {filepath}")
                # Retornar configuración por defecto
                return {"provider": "local"}

            with open(filepath, "r", encoding="utf-8") as f:
                config = json.load(f)

            logger.info(f"[LLM FACTORY] Configuración cargada desde: {filepath}")
            return config

        except Exception as e:
            logger.error(f"Error cargando configuración LLM: {e}")
            # Fallback a configuración local
            return {"provider": "local"}
