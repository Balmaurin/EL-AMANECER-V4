"""
Módulo de proveedores LLM - Implementaciones específicas de LLMInterface
"""

from .local_llm import LocalLLM

# OpenAI provider (opcional - solo si está disponible)
try:
    from .openai_llm import OpenAILLM
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAILLM = None

# Exports disponibles
__all__ = ["LocalLLM"]
if _OPENAI_AVAILABLE:
    __all__.append("OpenAILLM")
