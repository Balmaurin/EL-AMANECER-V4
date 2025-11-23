"""
Router de Usuarios - Sheily AI Backend
Gestión de perfiles de usuario, tokens, niveles y estadísticas
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ...models.user import User
from ...services.user_service import UserService
from ..dependencies import get_current_user, get_user_service

router = APIRouter()


class UserProfileResponse(BaseModel):
    """Respuesta del perfil de usuario"""

    id: str
    email: str
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    tokens: int
    level: int
    experience: int
    subscription: str
    created_at: str
    updated_at: str


class TokenBalanceResponse(BaseModel):
    """Respuesta del balance de tokens"""

    current_tokens: int
    level: int
    experience: int
    next_level_experience: int
    daily_limit: int
    monthly_limit: int


class TokenTransaction(BaseModel):
    """Transacción de tokens"""

    id: str
    type: str  # earned, spent, purchased
    amount: int
    description: str
    timestamp: str
    balance_after: int


@router.get("/profile", response_model=UserProfileResponse)
async def get_user_profile(
    current_user: User = Depends(get_current_user),
) -> UserProfileResponse:
    """
    Obtener perfil del usuario actual

    Retorna información completa del perfil del usuario autenticado,
    incluyendo tokens, nivel, experiencia y configuración.

    **Requiere autenticación JWT**
    """
    return UserProfileResponse(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        avatar_url=current_user.avatar_url,
        tokens=current_user.tokens,
        level=current_user.level,
        experience=current_user.experience,
        subscription=current_user.subscription,
        created_at=current_user.created_at.isoformat(),
        updated_at=current_user.updated_at.isoformat(),
    )


@router.put("/profile")
async def update_user_profile(
    profile_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
):
    """
    Actualizar perfil del usuario

    Permite actualizar nombre, avatar y otras configuraciones del perfil.

    **Campos actualizables:**
    - name: Nombre del usuario
    - avatar_url: URL del avatar

    **Requiere autenticación JWT**
    """
    allowed_fields = {"name", "avatar_url"}

    # Filtrar solo campos permitidos
    update_data = {k: v for k, v in profile_data.items() if k in allowed_fields}

    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se proporcionaron campos válidos para actualizar",
        )

    try:
        updated_user = await user_service.update_profile(current_user.id, update_data)
        return {
            "message": "Perfil actualizado correctamente",
            "user": UserProfileResponse(
                id=updated_user.id,
                email=updated_user.email,
                name=updated_user.name,
                avatar_url=updated_user.avatar_url,
                tokens=updated_user.tokens,
                level=updated_user.level,
                experience=updated_user.experience,
                subscription=updated_user.subscription,
                created_at=updated_user.created_at.isoformat(),
                updated_at=updated_user.updated_at.isoformat(),
            ),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error actualizando perfil: {str(e)}",
        )


@router.get("/tokens", response_model=TokenBalanceResponse)
async def get_token_balance(
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> TokenBalanceResponse:
    """
    Obtener balance de tokens del usuario

    Retorna información detallada sobre los tokens del usuario,
    incluyendo nivel actual, experiencia y límites.

    **Requiere autenticación JWT**
    """
    try:
        balance_info = await user_service.get_token_balance(current_user.id)

        return TokenBalanceResponse(
            current_tokens=balance_info["current_tokens"],
            level=balance_info["level"],
            experience=balance_info["experience"],
            next_level_experience=balance_info["next_level_experience"],
            daily_limit=balance_info["daily_limit"],
            monthly_limit=balance_info["monthly_limit"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo balance de tokens: {str(e)}",
        )


@router.post("/purchase")
async def purchase_tokens(
    amount: int,
    payment_method: str = "stripe",
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
):
    """
    Comprar tokens adicionales

    Permite al usuario adquirir tokens adicionales mediante diferentes métodos de pago.  # noqa: E501

    **Parámetros:**
    - amount: Cantidad de tokens a comprar (mínimo 100)
    - payment_method: Método de pago (stripe, paypal, crypto)

    **Requiere autenticación JWT**
    """
    if amount < 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La cantidad mínima de tokens a comprar es 100",
        )

    try:
        # Calcular precio (ejemplo: 1 token = $0.01)
        price_cents = amount

        # Crear sesión de pago (simulado)
        payment_session = await user_service.create_payment_session(
            user_id=current_user.id,
            amount=amount,
            price_cents=price_cents,
            payment_method=payment_method,
        )

        return {
            "message": "Sesión de pago creada",
            "payment_session": payment_session,
            "amount": amount,
            "price_cents": price_cents,
            "payment_method": payment_method,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando compra: {str(e)}",
        )


@router.get("/transactions")
async def get_token_transactions(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
):
    """
    Obtener historial de transacciones de tokens

    Retorna el historial de transacciones de tokens del usuario,
    incluyendo compras, uso en conversaciones, rewards, etc.

    **Parámetros:**
    - limit: Número máximo de transacciones (máximo 100)
    - offset: Offset para paginación

    **Requiere autenticación JWT**
    """
    if limit > 100:
        limit = 100

    try:
        transactions = await user_service.get_token_transactions(
            user_id=current_user.id, limit=limit, offset=offset
        )

        return {
            "transactions": [
                TokenTransaction(
                    id=tx["id"],
                    type=tx["type"],
                    amount=tx["amount"],
                    description=tx["description"],
                    timestamp=tx["timestamp"],
                    balance_after=tx["balance_after"],
                ).dict()
                for tx in transactions
            ],
            "total": len(transactions),
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo transacciones: {str(e)}",
        )


@router.post("/avatar")
async def upload_avatar(
    avatar_file: bytes,
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
):
    """
    Subir avatar del usuario

    Permite subir una imagen de avatar para el perfil del usuario.
    La imagen debe ser JPG, PNG o GIF, máximo 5MB.

    **Requiere autenticación JWT**
    """
    # Validar tamaño del archivo (máximo 5MB)
    max_size = 5 * 1024 * 1024  # 5MB
    if len(avatar_file) > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo es demasiado grande (máximo 5MB)",
        )

    # Validar tipo de archivo (simplificado)
    if not avatar_file.startswith(b"\xff\xd8"):  # JPEG
        if not avatar_file.startswith(b"\x89PNG"):  # PNG
            if not avatar_file.startswith(b"GIF"):  # GIF
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Tipo de archivo no soportado. Use JPG, PNG o GIF",
                )

    try:
        avatar_url = await user_service.upload_avatar(
            user_id=current_user.id, avatar_data=avatar_file
        )

        return {
            "message": "Avatar subido correctamente",
            "avatar_url": avatar_url,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error subiendo avatar: {str(e)}",
        )
