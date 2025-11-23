"""
Reglas Científicas: Estructura y Claridad - Principios 1-4 del estudio científico
"""

from typing import Any, Dict, Optional

from ..universal_prompt_optimizer import PromptTechnique


class DelimiterTechnique(PromptTechnique):
    """Regla 1: Usa delimitadores claros para separar secciones"""

    name = "scientific_delimiter"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Aplicar delimitadores para mejorar claridad estructural"""
        return f"### Instrucción ###\n{original_prompt}\n### Respuesta ###"


class OutputPrimerTechnique(PromptTechnique):
    """Regla 2: Incluir primers de output para guiar el formato"""

    name = "scientific_output_primer"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Agregar inicio de respuesta esperada"""
        primer_format = context.get("format", "structured") if context else "structured"

        if primer_format == "json":
            primer = "\n\nRespuesta esperada:\n```json\n{\n"
        elif primer_format == "list":
            primer = "\n\nRespuesta esperada:\n1."
        else:
            primer = "\n\nRespuesta clara y estructurada:\n"

        return original_prompt + primer


class AudienceIntegrationTechnique(PromptTechnique):
    """Regla 3: Integrar audiencia prevista en el prompt"""

    name = "scientific_audience_integration"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Incorporar especificaciones de audiencia"""
        audience = context.get("audience", "general") if context else "general"

        audience_intro = {
            "expert": "Como experto en el campo:",
            "beginner": "Explicación para principiantes:",
            "technical": "Análisis técnico detallado:",
            "business": "Perspectiva empresarial:",
        }.get(audience, "Para el lector general:")

        return f"{audience_intro} {original_prompt}"


class AffirmativeDirectivesTechnique(PromptTechnique):
    """Regla 4: Usar directivas afirmativas en lugar de negativas"""

    name = "scientific_affirmative_directives"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Convertir instrucciones negativas a positivas"""
        prompt = original_prompt

        # Reemplazar patrones negativos comunes
        replacements = {
            "no hagas": "haz",
            "no uses": "usa",
            "no incluyas": "incluye",
            "evita": "incentiva",
            "no menciones": "menciona",
        }

        for negative, positive in replacements.items():
            prompt = prompt.replace(negative, positive)

        return prompt


class SectionStructuredTechnique(PromptTechnique):
    """Principio ampliado: Estructura con secciones claras"""

    name = "scientific_section_structure"

    def apply(
        self, original_prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Estructurar prompt con secciones usando #"""
        sections_data = context.get("sections", []) if context else []

        if not sections_data:
            return f"### Contexto ###\n{original_prompt}\n### Requisitos ###"

        # Si hay secciones específicas, usarlas
        structured = "### Instrucción ###\n"
        structured += original_prompt + "\n\n"

        for section_name in sections_data:
            structured += f"### {section_name} ###\n\n"

        return structured
