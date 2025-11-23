"""
Dependencias comunes para los routers de la API
Incluye autenticación, servicios y validaciones compartidas
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..core.auth import auth_service
from ..models.user import User
from ..services.user_service import UserService

# Seguridad JWT
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """
    Obtener usuario actual desde token JWT

    Valida el token JWT y retorna el usuario correspondiente.
    Lanza HTTPException si el token es inválido o expirado.
    """
    try:
        # Extraer token
        token = credentials.credentials

        # Validar token y obtener payload
        payload = auth_service.verify_token(token)

        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido o expirado",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Obtener usuario desde base de datos (simulado por ahora)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token malformado",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # TODO: Obtener usuario real de la base de datos
        # Por ahora retornamos un usuario mock
        user = User(
            id=user_id,
            email=payload.get("email", "user@example.com"),
            name=payload.get("name"),
            avatar_url=None,
            tokens=1000,
            level=1,
            experience=0,
            subscription="free",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
        )

        return user

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Error de autenticación",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_user_service() -> UserService:
    """
    Obtener instancia del servicio de usuarios

    Retorna una instancia del UserService para operaciones de usuario.
    """
    return UserService()


async def get_admin_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Verificar que el usuario actual tenga permisos de administrador

    Lanza HTTPException si el usuario no es admin.
    """
    # TODO: Implementar verificación de roles
    # Por ahora, asumimos que todos los usuarios son válidos
    return current_user


def validate_api_key(api_key: Optional[str] = None) -> bool:
    """
    Validar API key para acceso programático

    Retorna True si la API key es válida.
    """
    if not api_key:
        return False

    # TODO: Implementar validación real de API keys
    valid_keys = ["sk-test-1234567890abcdef"]  # Keys de prueba
    return api_key in valid_keys
