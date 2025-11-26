"""
Conciencia Module - Artificial Consciousness System
==================================================

Main package for the artificial consciousness system, integrating
multiple cognitive modules and neural architectures.
"""

__version__ = "1.0.0"

# Safe imports with error handling
_available_modules = {}

try:
    from .meta_cognition_system import MetaCognitionSystem
    _available_modules['MetaCognitionSystem'] = MetaCognitionSystem
except ImportError as e:
    print(f"⚠️ Consciousness module import error: MetaCognitionSystem - {e}")
    MetaCognitionSystem = None

try:
    # Note: meta_cognition_system_simple.py defines MetaCognitionSystem class
    from .meta_cognition_system_simple import MetaCognitionSystem as MetaCognitionSystemSimple
    _available_modules['MetaCognitionSystemSimple'] = MetaCognitionSystemSimple
except ImportError as e:
    print(f"⚠️ Consciousness module import error: MetaCognitionSystemSimple - {e}")
    MetaCognitionSystemSimple = None

try:
    # Note: new_neural_component.py defines AdaptiveNeuralLayer class
    from .new_neural_component import AdaptiveNeuralLayer as NewNeuralComponent
    _available_modules['NewNeuralComponent'] = NewNeuralComponent
except ImportError as e:
    print(f"⚠️ Consciousness module import error: NewNeuralComponent - {e}")
    NewNeuralComponent = None

try:
    from . import modulos
    _available_modules['modulos'] = modulos
except ImportError as e:
    print(f"⚠️ Consciousness module import error: modulos - {e}")

__all__ = [
    'MetaCognitionSystem',
    'MetaCognitionSystemSimple',
    'NewNeuralComponent',
    'modulos',
    '_available_modules'
]
