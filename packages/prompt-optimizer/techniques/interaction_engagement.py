"""
Interacción y Compromiso del Usuario - Principios 9-12 del estudio científico
"""

from typing import Any, Dict, Optional

from ..universal_prompt_optimizer import PromptTechnique


class ExplainWithEvidenceTechnique(PromptTechnique):
    """Principio 9: Explicar con evidencia (Explaining with evidence)"""

    name = "scientific_explain_evidence"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Fuerzar explicación con evidencia"""
        return (
            original_prompt
            + " Explain your answer with concrete evidence and examples."
        )


class ComprehensiveCoverageTechnique(PromptTechnique):
    """Principio 10: Explicar con cobertura comprehensiva"""

    name = "scientific_comprehensive_coverage"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Asegurar cobertura completa del tema"""
        return (
            original_prompt
            + " Ensure your explanation covers all aspects of the topic comprehensively."
        )


class TopDownMeditationTechnique(PromptTechnique):
    """Principio 11: Permitir cuestionar/meditar top-down"""

    name = "scientific_top_down_meditation"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Permitir reconsideración y cuestionamiento"""
        return (
            original_prompt
            + " Take a moment to question your initial assumptions before proceeding."
        )


class MacroGenerationTechnique(PromptTechnique):
    """Principio 12: Usar generadores de macros"""

    name = "scientific_macro_generators"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generar respuesta a nivel macro antes de detalles"""
        return f"At a high level: {original_prompt} Then provide specific details."
