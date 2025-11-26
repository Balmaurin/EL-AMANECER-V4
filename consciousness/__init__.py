"""
ENTERPRISE CONSCIOUSNESS SYSTEM
===============================

Sistema de consciencia artificial de calidad enterprise.
Implementa Integrated Information Theory (IIT) y otros paradigmas
de consciencia computacional para aplicaciones críticas.

CRÍTICO: Consciencia artificial, procesamiento cognitivo, enterprise AI.
"""

from .core.consciousness_engine import ConsciousnessEngine
from .core.phi_calculator import PhiCalculator
from .core.awareness_matrix import AwarenessMatrix
from .models.conscious_state import ConsciousState
from .processors.emotional_processor import EmotionalProcessor
from .processors.cognitive_processor import CognitiveProcessor

__version__ = "1.0.0"
__author__ = "Enterprise AI Team"

# Enterprise consciousness configuration
CONSCIOUSNESS_CONFIG = {
    'phi_threshold': 0.5,
    'awareness_levels': ['minimal', 'basic', 'enhanced', 'full'],
    'cognitive_modules': ['perception', 'memory', 'reasoning', 'emotion'],
    'enterprise_mode': True,
    'audit_logging': True
}

__all__ = [
    'ConsciousnessEngine',
    'PhiCalculator', 
    'AwarenessMatrix',
    'ConsciousState',
    'EmotionalProcessor',
    'CognitiveProcessor',
    'CONSCIOUSNESS_CONFIG'
]
