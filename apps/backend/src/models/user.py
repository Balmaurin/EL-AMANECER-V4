"""
Modelo de Usuario - Sheily AI Backend
Define la estructura de datos del usuario y operaciones relacionadas
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class User(BaseModel):
    """
    Modelo de Usuario

    Representa un usuario del sistema Sheily AI con toda su información
    de perfil, tokens, niveles y configuración.
    """

    # Identificación
    id: str
    email: EmailStr

    # Información personal
    name: Optional[str] = None
    avatar_url: Optional[str] = None

    # Sistema de tokens y niveles
    tokens: int = 0
    level: int = 1
    experience: int = 0
    subscription: str = "free"  # free, premium, enterprise

    # Timestamps
    created_at: str  # ISO format
    updated_at: str  # ISO format

    class Config:
        """Configuración del modelo"""

        from_attributes = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    def add_experience(self, amount: int) -> None:
        """
        Añadir experiencia al usuario

        Args:
            amount: Cantidad de experiencia a añadir
        """
        self.experience += amount

        # Calcular nuevo nivel (cada 1000 exp = 1 nivel)
        new_level = (self.experience // 1000) + 1

        if new_level > self.level:
            self.level = new_level
            # TODO: Trigger level up event

    def spend_tokens(self, amount: int) -> bool:
        """
        Gastar tokens del usuario

        Args:
            amount: Cantidad de tokens a gastar

        Returns:
            True si la transacción fue exitosa, False si no hay suficientes tokens  # noqa: E501
        """
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    def add_tokens(self, amount: int) -> None:
        """
        Añadir tokens al usuario

        Args:
            amount: Cantidad de tokens a añadir
        """
        self.tokens += amount

    def can_afford(self, cost: int) -> bool:
        """
        Verificar si el usuario puede costear una operación

        Args:
            cost: Costo en tokens

        Returns:
            True si puede costear, False en caso contrario
        """
        return self.tokens >= cost

    def get_level_progress(self) -> dict:
        """
        Obtener progreso hacia el siguiente nivel

        Returns:
            Diccionario con información del progreso
        """
        current_level_exp = (self.level - 1) * 1000
        next_level_exp = self.level * 1000
        progress_exp = self.experience - current_level_exp
        required_exp = next_level_exp - current_level_exp

        return {
            "current_level": self.level,
            "experience": self.experience,
            "progress_experience": progress_exp,
            "required_experience": required_exp,
            "progress_percentage": (
                (progress_exp / required_exp) * 100 if required_exp > 0 else 100
            ),
        }

    def to_dict(self) -> dict:
        """
        Convertir usuario a diccionario

        Returns:
            Diccionario con todos los campos del usuario
        """
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "avatar_url": self.avatar_url,
            "tokens": self.tokens,
            "level": self.level,
            "experience": self.experience,
            "subscription": self.subscription,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class UserCreate(BaseModel):
    """Modelo para crear un nuevo usuario"""

    email: EmailStr
    name: Optional[str] = None
    password: str
    subscription: str = "free"


class UserUpdate(BaseModel):
    """Modelo para actualizar un usuario"""

    name: Optional[str] = None
    avatar_url: Optional[str] = None
    subscription: Optional[str] = None


class UserLogin(BaseModel):
    """Modelo para login de usuario"""

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Respuesta con tokens JWT"""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: User


class UserStats(BaseModel):
    """Estadísticas del usuario"""

    total_conversations: int = 0
    total_messages: int = 0
    total_tokens_spent: int = 0
    total_tokens_earned: int = 0
    average_response_time: float = 0.0
    favorite_model: Optional[str] = None
    last_activity: Optional[str] = None
