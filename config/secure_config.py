"""
Configuración Segura - Variables de Entorno Requeridas

IMPORTANTE: Este archivo ya NO contiene secretos hardcodeados.
Todos los valores sensibles deben configurarse vía variables de entorno.

Variables de Entorno Requeridas:
- JWT_SECRET_KEY: Clave secreta para JWT (cambiar en producción)
- CORS_ALLOWED_ORIGINS: Orígenes permitidos separados por comas
- DATABASE_URL: URL completa de base de datos (NO hardcodear)
- REDIS_URL: URL de Redis para rate limiting
- DEBUG: Solo 'true' en desarrollo local
- SECRET_KEY: Clave secreta general de Flask/FastAPI

Ejemplo para desarrollo (.env):
JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
DATABASE_URL=postgresql://user:pass@localhost/dbname
REDIS_URL=redis://localhost:6379/1
DEBUG=false
SECRET_KEY=another-super-secret-key-change-too
"""

import json
import os
from typing import List


# === UTILIDADES DE CONFIGURACIÓN SEGURA ===
def _get_env_var(name: str, default=None, required: bool = False):
    """Obtiene variable de entorno con validación."""
    value = os.getenv(name, default)
    if required and value is None:
        raise ValueError(f"Variable de entorno requerida '{name}' no está configurada")
    return value


def _get_bool_env(name: str, default: bool = False) -> bool:
    """Obtiene variable booleana de entorno."""
    value = os.getenv(name, str(default).lower()).lower()
    return value in ("true", "1", "yes", "on")


# === CONFIGURACIÓN DE ENTORNO (NO HARDCODEAR) ===
DEBUG = _get_bool_env("DEBUG", False)
TESTING = _get_bool_env("TESTING", False)

# ✅ SEGURIDAD: Validar que DEBUG nunca esté habilitado en producción
if DEBUG and _get_env_var("ENVIRONMENT") == "production":
    raise RuntimeError("DEBUG no puede estar habilitado en entorno de producción")

# === JWT CONFIGURATION ===
# ✅ SEGURIDAD: Clave JWT desde variable de entorno (NUNCA HARDCODEAR)
JWT_SECRET_KEY = _get_env_var("JWT_SECRET_KEY", required=True)
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(_get_env_var("JWT_EXPIRE_MINUTES", "30"))

# ✅ SEGURIDAD: Validar longitud mínima de clave secreta
if len(JWT_SECRET_KEY) < 32:
    raise ValueError("JWT_SECRET_KEY debe tener al menos 32 caracteres")

# === CORS CONFIGURATION ===
# ✅ SEGURIDAD: Orígenes CORS desde variable de entorno
cors_origins = _get_env_var(
    "CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
)
CORS_ALLOWED_ORIGINS: List[str] = [
    origin.strip() for origin in cors_origins.split(",") if origin.strip()
]

# ✅ SEGURIDAD: Nunca permitir wildcard (*) en producción
if "*" in CORS_ALLOWED_ORIGINS and _get_env_var("ENVIRONMENT") == "production":
    raise RuntimeError("CORS wildcard (*) no permitido en producción")

CORS_ALLOW_CREDENTIALS = not DEBUG  # Solo permitir credenciales en desarrollo

# === DATABASE CONFIGURATION ===
# ✅ SEGURIDAD: URL completa de base de datos desde variable de entorno
DATABASE_URL = _get_env_var("DATABASE_URL", required=True)
SQLALCHEMY_ECHO = DEBUG  # Solo mostrar SQL en desarrollo

# === SECURITY HEADERS ===
# ✅ SEGURIDAD: Headers reforzados
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_SSL_REDIRECT = not DEBUG  # Solo forzar HTTPS en producción
SECURE_HSTS_SECONDS = 31536000 if not DEBUG else 0  # 1 año en producción
SECURE_HSTS_PRELOAD = not DEBUG

# === RATE LIMITING ===
# ✅ SEGURIDAD: Rate limiting configurable y con Redis desde variable de entorno
RATELIMIT_DEFAULT = _get_env_var("RATELIMIT_DEFAULT", "100 per minute")
RATELIMIT_STORAGE_URL = _get_env_var("REDIS_URL", "redis://localhost:6379/1")

# === SECRET KEY GENERAL ===
# ✅ SEGURIDAD: Clave secreta general (para sesiones Flask/FastAPI)
SECRET_KEY = _get_env_var("SECRET_KEY", required=True)
if len(SECRET_KEY) < 32:
    raise ValueError("SECRET_KEY debe tener al menos 32 caracteres")

# === VALIDACIÓN FINAL ===
print("✅ Configuración segura cargada correctamente")
print(f"🔒 Modo DEBUG: {DEBUG}")
print(f"🌍 Entorno: {_get_env_var('ENVIRONMENT', 'development')}")
print(f"🔑 JWT configurado: {'✅' if JWT_SECRET_KEY else '❌'}")
print(f"🗄️ Base de datos configurada: {'✅' if DATABASE_URL else '❌'}")
print(
    f"⚡ Redis configurado: {'✅' if RATELIMIT_STORAGE_URL.startswith('redis://') else '❌'}"
)

# === VALIDACIÓN DE SEGURIDAD EN PRODUCCIÓN ===
if _get_env_var("ENVIRONMENT") == "production":
    security_checks = [
        ("DEBUG deshabilitado", not DEBUG),
        ("JWT secret configurado", bool(JWT_SECRET_KEY)),
        ("Base de datos configurada", bool(DATABASE_URL)),
        ("Cors no wildcard", "*" not in CORS_ALLOWED_ORIGINS),
        ("HTTPS forzada", SECURE_SSL_REDIRECT),
        ("Secret key >= 32 chars", len(SECRET_KEY) >= 32),
    ]

    failed_checks = [check for check, passed in security_checks if not passed]

    if failed_checks:
        raise RuntimeError(
            f"❌ Fallaron validaciones de seguridad en producción: {failed_checks}"
        )

    print("🛡️ Todas las validaciones de seguridad pasaron correctamente")
