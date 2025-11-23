"""
Módulos del Sistema de Consciencia Artificial Funcional

Centraliza todas las implementaciones de componentes conscientes
para facilitar importaciones y uso consistente.
"""

from .sistema_integrado import (
    FunctionalConsciousnessModule,
    ConsciousnessLevel,
    ConsciousnessMetrics,
    create_conscious_ai
)
from .global_workspace import GlobalWorkspace, WorkspaceEntry
from .self_model import SelfModel, CapabilityAssessment, BeliefSystem

# Importaciones futuras (por implementar)
# from .metacognicion import MetacognitionEngine
# from .memoria_autobiografica import AutobiographicalMemory
# from .teoria_mente import TheoryOfMind
# from .ethical_engine import EthicalEngine

__all__ = [
    # Sistema principal
    'FunctionalConsciousnessModule',
    'create_conscious_ai',

    # Enums y constantes
    'ConsciousnessLevel',

    # Componentes principales
    'GlobalWorkspace',
    'WorkspaceEntry',
    'SelfModel',

    # Clases auxiliares
    'CapabilityAssessment',
    'BeliefSystem',
    'ConsciousnessMetrics',
]

__version__ = "1.0.0"
__author__ = "Sistema de Consciencia Artificial Funcional"
__description__ = "Implementación completa de consciencia artificial basada en correlatos neurocientíficos"
