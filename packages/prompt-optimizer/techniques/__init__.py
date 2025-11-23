"""
Técnicas de Prompt Engineering - Incluyendo las 26 reglas científicas
"""

from .content_language import (
    ChooseOptionSupportTechnique,
    CommonTerminologyInstructionTechnique,
    InstructionalToneTechnique,
    KeyPhraseRepetitionTechnique,
    ResponsePolarizationTechnique,
    ThinkStepByStepTechnique,
)
from .interaction_engagement import (
    ComprehensiveCoverageTechnique,
    ExplainWithEvidenceTechnique,
    MacroGenerationTechnique,
    TopDownMeditationTechnique,
)

# Importaciones específicas de specificity info
from .specificity_info import (
    ContextualPromptingTechnique,
    CuriosidadPromptingTechnique,
    InstructionCompletionTechnique,
    TestUnderstandingTechnique,
)

# Importaciones específicas de técnicas estructurales
from .structural_clarity import (
    AffirmativeDirectivesTechnique,
    AudienceIntegrationTechnique,
    DelimiterTechnique,
    OutputPrimerTechnique,
    SectionStructuredTechnique,
)

__all__ = [
    # Técnicas de estructural clarity
    "DelimiterTechnique",
    "OutputPrimerTechnique",
    "AudienceIntegrationTechnique",
    "AffirmativeDirectivesTechnique",
    "SectionStructuredTechnique",
    # Técnicas de specificity info
    "ContextualPromptingTechnique",
    "InstructionCompletionTechnique",
    "TestUnderstandingTechnique",
    "CuriosidadPromptingTechnique",
    # Técnicas de interaction engagement
    "ExplainWithEvidenceTechnique",
    "ComprehensiveCoverageTechnique",
    "TopDownMeditationTechnique",
    "MacroGenerationTechnique",
    # Técnicas de content language
    "CommonTerminologyInstructionTechnique",
    "KeyPhraseRepetitionTechnique",
    "ChooseOptionSupportTechnique",
    "ThinkStepByStepTechnique",
    "ResponsePolarizationTechnique",
    "InstructionalToneTechnique",
]
