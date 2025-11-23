"""
Especificidad e Información - Principios 5-8 del estudio científico
"""

from typing import Any, Dict, Optional

from ..universal_prompt_optimizer import PromptTechnique


class ContextualPromptingTechnique(PromptTechnique):
    """Principio 5: Hablar desde el principio (habla desde el principio)"""

    name = "scientific_speak_first"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Empezar directamente con la respuesta esperada"""
        starter = (
            context.get("starter", "Directly answer: ")
            if context
            else "Directly answer: "
        )
        return f"{starter}{original_prompt}"


class InstructionCompletionTechnique(PromptTechnique):
    """Principio 6: Proporcionar instrucciones de finalización"""

    name = "scientific_completion_instructions"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Agregar instrucciones sobre cómo completar el envío"""
        return original_prompt + " Complete your response by showing each step clearly."


class TestUnderstandingTechnique(PromptTechnique):
    """Principio 7: Probar entendimiento del usuario"""

    name = "scientific_test_understanding"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Incluir preguntas para verificar comprensión"""
        return (
            original_prompt
            + " If this is clear, explain one concept from it. Otherwise, ask a clarifying question."
        )


class CuriosidadPromptingTechnique(PromptTechnique):
    """Principio ampliado: Prompting guiado por curiosidad (short prompts)"""

    name = "scientific_curiosity_driven"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Usar prompts cortos para generar curiosidad"""
        # Hacer pregunta corta basada en el prompt original
        if "?" in original_prompt:
            return original_prompt.split("?")[0] + "?"
        else:
            # Extraer esencia
            words = original_prompt.split()[:5]  # Primeros 5 palabras
            return (
                " ".join(words) + "..."
                if len(words) < len(original_prompt.split())
                else original_prompt
            )
