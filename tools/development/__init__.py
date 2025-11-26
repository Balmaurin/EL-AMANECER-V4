"""
Development Tools for Sheily AI
================================

Herramientas de desarrollo, auditoría y mantenimiento del proyecto.
"""

from .audit_project import SheilyAuditor
from .generate_init_files import InitFileGenerator

__all__ = [
    "SheilyAuditor",
    "InitFileGenerator",
]

__version__ = "1.0.0"
__author__ = "Sheily AI Research Team"
