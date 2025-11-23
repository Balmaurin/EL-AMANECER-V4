"""
Sistema Universal de Optimización Automática de Prompts
"""

from .universal_prompt_optimizer import (
    ChainOfThoughtTechnique,
    ExpertPromptingTechnique,
    FewShotTechnique,
    LlamaCppAdapter,
    LLMAdapter,
    LLMProvider,
    OpenAIAdapter,
    PromptEvaluator,
    PromptTechnique,
    RailsTechnique,
    UniversalAutoImprovingPromptSystem,
    ZeroShotTechnique,
)

__version__ = "1.0.0"
__author__ = "Sheily AI"

__all__ = [
    "UniversalAutoImprovingPromptSystem",
    "LLMAdapter",
    "OpenAIAdapter",
    "LlamaCppAdapter",
    "PromptTechnique",
    "PromptEvaluator",
    "ChainOfThoughtTechnique",
    "FewShotTechnique",
    "ExpertPromptingTechnique",
    "RailsTechnique",
    "ZeroShotTechnique",
    "LLMProvider",
]
