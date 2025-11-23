"""
Contenido y Estilo de Lenguaje - Principios 13-16 del estudio científico
"""

from typing import Any, Dict, Optional

from ..universal_prompt_optimizer import PromptTechnique


class CommonTerminologyInstructionTechnique(PromptTechnique):
    """Principio 13: Instructivo con términos comunes (Instructional with common terms)"""

    name = "scientific_common_terminology"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Usar términos comunes y evitar jerga técnica"""
        return (
            original_prompt
            + " Use simple, everyday language that anyone can understand."
        )


class KeyPhraseRepetitionTechnique(PromptTechnique):
    """Principio 14: Repetir frases clave (Repeat key phrases)"""

    name = "scientific_key_phrase_repetition"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Repetir frases importantes para énfasis"""
        key_phrase = context.get("key_phrase", "important") if context else "important"
        return (
            original_prompt
            + f" Remember, this is {key_phrase}. Never forget that this is {key_phrase}."
        )


class ChooseOptionSupportTechnique(PromptTechnique):
    """Principio 15: Usar 'puedes elegir' en preguntas"""

    name = "scientific_choose_option_support"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Ofrecer opciones en preguntas"""
        return original_prompt + " You can choose different approaches to address this."


class ThinkStepByStepTechnique(PromptTechnique):
    """Principio 16: Técnica 'think step by step'"""

    name = "scientific_think_step_by_step"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Fuerzar razonamiento paso a paso"""
        return original_prompt + " Think step by step through this problem."


class ResponsePolarizationTechnique(PromptTechnique):
    """Principio ampliado: Polarizar respuestas (Respuesta polarizada - positivo vs negativo)"""

    name = "scientific_response_polarization"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Enfatizar respuestas claras positivas"""
        return f"You should: {original_prompt} Give affirmative, positive answers."


class InstructionalToneTechnique(PromptTechnique):
    """Principio ampliado: Tono instructivo"""

    name = "scientific_instructional_tone"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Usar lenguaje instructivo directo"""
        return f"Your task is to: {original_prompt} Make sure to follow these instructions precisely."
