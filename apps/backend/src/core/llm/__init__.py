"""
Módulo LLM - Gestión del modelo de lenguaje LLAMA 3
"""

from .llm_interface import LLMInterface
from .llm_factory import LLMFactory
from .providers.local_llm import LocalLLM

__all__ = ["LLMInterface", "LLMFactory", "LocalLLM"]
