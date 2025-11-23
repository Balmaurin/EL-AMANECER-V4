"""
Endpoints de autenticación para Sheily AI
Gestión de usuarios, login, registro y tokens
"""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ...core.auth import (
    auth_service,
    create_token_response,
    create_user_response,
    get_current_active_user,
)
from ...models.base import APIToken, User
from ...models.database import get_db_session

router = APIRouter()


@router.post("/register", response_model=Dict[str, Any])
async def register_user(
    email: str,
    username: str,
    password: str,
    full_name: str = None,
    db: Session = Depends(get_db_session),
) -> Dict[str, Any]:
    """Registrar nuevo usuario"""
    try:
        user = auth_service.create_user(
            db=db,
            email=email,
            username=username,
            password=password,
            full_name=full_name,
        )

        # Crear tokens automáticamente después del registro
        access_token = auth_service.create_access_token(data={"sub": str(user.id)})
        refresh_token = auth_service.create_refresh_token(data={"sub": str(user.id)})

        return create_token_response(access_token, refresh_token, user)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}",
        )


@router.post("/login", response_model=Dict[str, Any])
async def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db_session),
) -> Dict[str, Any]:
    """Iniciar sesión de usuario"""
    user = auth_service.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email/username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )

    # Crear tokens
    access_token = auth_service.create_access_token(data={"sub": str(user.id)})
    refresh_token = auth_service.create_refresh_token(data={"sub": str(user.id)})

    return create_token_response(access_token, refresh_token, user)


@router.post("/refresh", response_model=Dict[str, Any])
async def refresh_access_token(
    refresh_token: str, db: Session = Depends(get_db_session)
) -> Dict[str, Any]:
    """Refrescar token de acceso"""
    payload = auth_service.verify_token(refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    # Crear nuevo access token
    access_token = auth_service.create_access_token(data={"sub": str(user.id)})
    new_refresh_token = auth_service.create_refresh_token(data={"sub": str(user.id)})

    return create_token_response(access_token, new_refresh_token, user)


@router.post("/logout")
async def logout_user(
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, str]:
    """Cerrar sesión (cliente debe eliminar tokens)"""
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=Dict[str, Any])
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Obtener información del usuario actual"""
    return create_user_response(current_user)


@router.put("/me", response_model=Dict[str, Any])
async def update_current_user(
    full_name: str = None,
    preferences: Dict[str, Any] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session),
) -> Dict[str, Any]:
    """Actualizar información del usuario actual"""
    updates = {}
    if full_name is not None:
        updates["full_name"] = full_name
    if preferences is not None:
        updates["preferences"] = preferences

    if updates:
        updated_user = auth_service.update_user(db, current_user.id, **updates)
        return create_user_response(updated_user)

    return create_user_response(current_user)


@router.post("/change-password")
async def change_password(
    old_password: str,
    new_password: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session),
) -> Dict[str, str]:
    """Cambiar contraseña del usuario actual"""
    if len(new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long",
        )

    success = auth_service.change_password(
        db, current_user.id, old_password, new_password
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect old password",
        )

    return {"message": "Password changed successfully"}


@router.post("/api-tokens", response_model=Dict[str, Any])
async def create_api_token(
    name: str,
    permissions: Dict[str, Any] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session),
) -> Dict[str, Any]:
    """Crear token de API para el usuario actual"""
    api_token = auth_service.create_api_token(
        db=db, user_id=current_user.id, name=name, permissions=permissions
    )

    # Retornar información del token (el token plano solo se muestra una vez)
    return {
        "id": api_token.id,
        "name": api_token.name,
        "token": getattr(api_token, "_plain_token", "Token already used"),
        "permissions": api_token.permissions,
        "created_at": api_token.created_at,
        "expires_at": api_token.expires_at,
    }


@router.get("/api-tokens", response_model=list[Dict[str, Any]])
async def list_api_tokens(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session),
) -> list[Dict[str, Any]]:
    """Listar tokens de API del usuario actual"""
    tokens = db.query(APIToken).filter(APIToken.user_id == current_user.id).all()

    return [
        {
            "id": token.id,
            "name": token.name,
            "is_active": token.is_active,
            "permissions": token.permissions,
            "created_at": token.created_at,
            "last_used_at": token.last_used_at,
            "expires_at": token.expires_at,
        }
        for token in tokens
    ]


@router.delete("/api-tokens/{token_id}")
async def revoke_api_token(
    token_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db_session),
) -> Dict[str, str]:
    """Revocar token de API"""
    success = auth_service.revoke_api_token(db, token_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API token not found"
        )

    return {"message": "API token revoked successfully"}


@router.post("/verify-token")
async def verify_token(
    token: str, db: Session = Depends(get_db_session)
) -> Dict[str, Any]:
    """Verificar validez de token JWT"""
    payload = auth_service.verify_token(token)
    if not payload:
        return {"valid": False, "user": None}

    if payload.get("type") == "access":
        user_id = payload.get("sub")
        if user_id:
            user = db.query(User).filter(User.id == int(user_id)).first()
            if user and user.is_active:
                return {
                    "valid": True,
                    "user": create_user_response(user),
                    "token_type": "access",
                }

    return {"valid": False, "user": None}
