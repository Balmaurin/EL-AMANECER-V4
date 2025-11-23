"""
Sheily AI - Sistema de Paths Portables
======================================

Este módulo proporciona un sistema de rutas completamente portable que funciona
en cualquier PC sin configuración manual.

CARACTERÍSTICAS:
- ✅ Rutas absolutas calculadas dinámicamente
- ✅ Funciona desde cualquier directorio
- ✅ Compatible con Windows, Linux y Mac
- ✅ Sin hardcodeo de rutas
- ✅ Single source of truth para todas las rutas

USO:
    from tools.common.paths import PROJECT_ROOT, DATA_DIR, MODELS_DIR

    # Todas las rutas son Path objects absolutos
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    model_path = MODELS_DIR / "gemma-2-9b-it-Q4_K_M.gguf"
"""

import os
import sys
from pathlib import Path

# ============================================================================
# CÁLCULO DINÁMICO DE LA RAÍZ DEL PROYECTO
# ============================================================================
# Este archivo está en: tools/common/paths.py
# La raíz del proyecto está 2 niveles arriba: ../../
# Esto funciona desde CUALQUIER directorio donde se ejecute


def _calculate_project_root() -> Path:
    """Calcula dinámicamente la raíz del proyecto."""
    try:
        # Ruta absoluta a este archivo
        current_file = Path(__file__).resolve()

        # Navegar hacia arriba: tools/common/paths.py -> tools/common -> tools -> raíz
        project_root = current_file.parent.parent.parent

        # Verificar que estamos en el directorio correcto
        if not (project_root / "docker-compose.yml").exists():
            raise FileNotFoundError(
                f"No se encuentra docker-compose.yml en {project_root}"
            )

        return project_root

    except Exception as e:
        print(f"❌ Error calculando PROJECT_ROOT: {e}")
        print(
            "   Asegúrate de estar ejecutando desde el directorio del proyecto Sheily AI"
        )
        return Path.cwd()  # Fallback al directorio actual


# Calcular la raíz del proyecto
PROJECT_ROOT = _calculate_project_root()

# ============================================================================
# DIRECTORIOS PRINCIPALES (ABSOLUTOS)
# ============================================================================

# Directorios core del proyecto
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCS_DIR = PROJECT_ROOT / "docs"

# ============================================================================
# SUBDIRECTORIOS DE DATOS (ABSOLUTOS)
# ============================================================================

# Base de datos y almacenamiento
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
UPLOADS_DIR = DATA_DIR / "uploads"
DATASETS_DIR = DATA_DIR / "datasets"
TRAINED_MODELS_DIR = DATA_DIR / "trained_models"
TEMP_DIR = DATA_DIR / "temp_training"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EVAL_DATA_DIR = DATA_DIR / "eval"

# Cache y registros
CACHE_DIR = PROJECT_ROOT / ".cache"
CATALOG_DIR = DATA_DIR / "_registry"

# ============================================================================
# ARCHIVOS ESPECÍFICOS (ABSOLUTOS)
# ============================================================================

# Configuración
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE_FILE = PROJECT_ROOT / ".env.example"
DOCKER_COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
DOCKER_COMPOSE_DEV_FILE = PROJECT_ROOT / "docker-compose.dev.yml"

# Modelos principales
GEMMA_MODEL_PATH = MODELS_DIR / "gemma-2-9b-it-Q4_K_M.gguf"
LLAMA_MODEL_PATH = MODELS_DIR / "llama-3.2-3b-instruct.gguf"

# Base de datos de conversaciones (fallback)
CONVERSATIONS_DB_PATH = DATA_DIR / "conversations.json"

# Logs principales
MAIN_LOG_FILE = LOGS_DIR / "sheily.log"
ERROR_LOG_FILE = LOGS_DIR / "error.log"

# ============================================================================
# FUNCIONES UTILITARIAS
# ============================================================================


def ensure_directories_exist():
    """
    Crea todos los directorios necesarios si no existen.
    Llamar al inicio de la aplicación.
    """
    directories = [
        DATA_DIR,
        MODELS_DIR,
        LOGS_DIR,
        CONFIG_DIR,
        CHROMA_DB_DIR,
        UPLOADS_DIR,
        DATASETS_DIR,
        TRAINED_MODELS_DIR,
        TEMP_DIR,
        EMBEDDINGS_DIR,
        EVAL_DATA_DIR,
        CACHE_DIR,
        CATALOG_DIR,
    ]

    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️  No se pudo crear directorio {directory}: {e}")


def get_relative_path(absolute_path: Path) -> str:
    """
    Convierte una ruta absoluta a relativa al proyecto.

    Args:
        absolute_path: Ruta absoluta

    Returns:
        str: Ruta relativa al PROJECT_ROOT
    """
    try:
        return str(absolute_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(absolute_path)


def validate_project_structure() -> bool:
    """
    Valida que la estructura del proyecto sea correcta.

    Returns:
        bool: True si la estructura es válida
    """
    required_files = [
        PROJECT_ROOT / "docker-compose.yml",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "setup.sh",
        PROJECT_ROOT / "dev.sh",
    ]

    missing_files: list[Path] = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(file_path)

    if missing_files:
        print("❌ Archivos requeridos faltantes:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False

    print("✅ Estructura del proyecto validada")
    return True


def get_system_info() -> dict[str, str | bool]:
    """
    Retorna información del sistema para debugging.

    Returns:
        dict: Información del sistema
    """
    return {
        "project_root": str(PROJECT_ROOT),
        "current_dir": str(Path.cwd()),
        "python_path": os.environ.get("PYTHONPATH", ""),
        "data_dir_exists": DATA_DIR.exists(),
        "models_dir_exists": MODELS_DIR.exists(),
        "logs_dir_exists": LOGS_DIR.exists(),
        "platform": os.name,
        "python_version": sys.version,
    }


# ============================================================================
# INICIALIZACIÓN AUTOMÁTICA
# ============================================================================

# Crear directorios al importar el módulo
ensure_directories_exist()

# ============================================================================
# VALIDACIÓN EN MODO DEBUG
# ============================================================================

if os.environ.get("DEBUG") == "true":
    print("🔧 DEBUG - Información de paths:")
    print(f"   PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"   DATA_DIR: {DATA_DIR}")
    print(f"   MODELS_DIR: {MODELS_DIR}")
    print(f"   Current dir: {Path.cwd()}")

    if not validate_project_structure():
        print("⚠️  Estructura del proyecto incompleta")

# ============================================================================
# EXPORTS PARA FACILITAR EL USO
# ============================================================================

__all__ = [
    # Directorios principales
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "LOGS_DIR",
    "CONFIG_DIR",
    "SCRIPTS_DIR",
    "DOCS_DIR",
    # Subdirectorios
    "CHROMA_DB_DIR",
    "UPLOADS_DIR",
    "DATASETS_DIR",
    "TRAINED_MODELS_DIR",
    "TEMP_DIR",
    "EMBEDDINGS_DIR",
    "EVAL_DATA_DIR",
    "CACHE_DIR",
    "CATALOG_DIR",
    # Archivos específicos
    "ENV_FILE",
    "ENV_EXAMPLE_FILE",
    "DOCKER_COMPOSE_FILE",
    "DOCKER_COMPOSE_DEV_FILE",
    "GEMMA_MODEL_PATH",
    "LLAMA_MODEL_PATH",
    "CONVERSATIONS_DB_PATH",
    "MAIN_LOG_FILE",
    "ERROR_LOG_FILE",
    # Funciones utilitarias
    "ensure_directories_exist",
    "get_relative_path",
    "validate_project_structure",
    "get_system_info",
]
