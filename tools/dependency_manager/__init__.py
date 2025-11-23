"""
Sheily MCP Enterprise - Dependency Management System
Sistema avanzado de gestión de dependencias inspirado en Google/DeepMind

Características principales:
- AI-aware dependency resolution
- Enterprise-grade locking
- Multi-environment support
- Security scanning automatizado
- Conflict resolution inteligente
"""

__version__ = "1.0.0"
__author__ = "Sheily MCP Team"
__description__ = "Enterprise dependency management for AI systems"

from .cli_interface import main
from .dependency_analyzer import DependencyAnalyzer
from .environment_manager import EnvironmentManager
from .installation_orchestrator import InstallationOrchestrator
from .optimization_engine import OptimizationEngine
from .security_scanner import SecurityScanner
from .update_manager import UpdateManager
from .validation_engine import ValidationEngine
from .version_locker import VersionLocker

__all__ = [
    "main",
    "DependencyAnalyzer",
    "EnvironmentManager",
    "InstallationOrchestrator",
    "VersionLocker",
    "UpdateManager",
    "ValidationEngine",
    "SecurityScanner",
    "OptimizationEngine",
]
