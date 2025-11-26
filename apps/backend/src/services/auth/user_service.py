"""
Servicio de Usuarios - Sheily AI Backend
Lógica de negocio para operaciones relacionadas con usuarios
"""

import asyncio
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from apps.backend.src.models.user import User



class UserService:
    """
    Servicio para operaciones relacionadas con usuarios

    Maneja toda la lógica de negocio para perfiles de usuario,
    tokens, transacciones, y gestión de avatares.
    """

    def __init__(self, db_path: str = "project_state.db"):
        """Inicializar el servicio de usuarios con persistencia real"""
        self.db_path = Path(db_path)
        self._init_database()
        self._users_cache = {}  # Cache en memoria para performance

    def _init_database(self):
        """Inicializar base de datos y tablas necesarias"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    avatar_url TEXT,
                    tokens INTEGER DEFAULT 1000,
                    level INTEGER DEFAULT 1,
                    experience INTEGER DEFAULT 0,
                    subscription TEXT DEFAULT 'free',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS token_transactions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    type TEXT NOT NULL,  -- 'earned', 'spent', 'purchased'
                    amount INTEGER NOT NULL,
                    description TEXT,
                    balance_after INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_stats (
                    user_id TEXT PRIMARY KEY,
                    total_conversations INTEGER DEFAULT 0,
                    total_messages INTEGER DEFAULT 0,
                    total_tokens_spent INTEGER DEFAULT 0,
                    total_tokens_earned INTEGER DEFAULT 0,
                    average_response_time REAL DEFAULT 0.0,
                    favorite_model TEXT,
                    last_activity TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """
            )

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Obtener usuario por ID desde base de datos real

        Args:
            user_id: ID del usuario

        Returns:
            Usuario si existe, None en caso contrario
        """
        # Verificar cache primero para performance
        if user_id in self._users_cache:
            return self._users_cache[user_id]

        # Buscar en base de datos real
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, email, name, avatar_url, tokens, level, experience,
                       subscription, created_at, updated_at
                FROM users WHERE id = ?
            """,
                (user_id,),
            )

            row = cursor.fetchone()

            if row:
                user = User(
                    id=row[0],
                    email=row[1],
                    name=row[2],
                    avatar_url=row[3],
                    tokens=row[4],
                    level=row[5],
                    experience=row[6],
                    subscription=row[7],
                    created_at=row[8],
                    updated_at=row[9],
                )
                # Cachear para performance futura
                self._users_cache[user_id] = user
                return user
            else:
                # Usuario no encontrado - no crear mock, retornar None
                return None

    async def create_user(self, user_data: Dict[str, Any]) -> User:
        """
        Crear nuevo usuario en base de datos real

        Args:
            user_data: Datos del usuario a crear

        Returns:
            Usuario creado

        Raises:
            ValueError: Si faltan campos requeridos
        """
        required_fields = ["id", "email", "name"]
        for field in required_fields:
            if field not in user_data:
                raise ValueError(f"Campo '{field}' es requerido para crear usuario")

        user_id = user_data["id"]
        email = user_data["email"]
        name = user_data["name"]

        # Verificar si ya existe
        existing_user = await self.get_user_by_id(user_id)
        if existing_user:
            raise ValueError(f"Usuario con ID {user_id} ya existe")

        # Crear usuario con valores por defecto
        now = datetime.utcnow().isoformat()
        user = User(
            id=user_id,
            email=email,
            name=name,
            avatar_url=user_data.get("avatar_url"),
            tokens=user_data.get("tokens", 1000),
            level=user_data.get("level", 1),
            experience=user_data.get("experience", 0),
            subscription=user_data.get("subscription", "free"),
            created_at=now,
            updated_at=now,
        )

        # Persistir en base de datos real
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO users (id, email, name, avatar_url, tokens, level, experience, subscription, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user.id,
                    user.email,
                    user.name,
                    user.avatar_url,
                    user.tokens,
                    user.level,
                    user.experience,
                    user.subscription,
                    user.created_at,
                    user.updated_at,
                ),
            )

        # Cachear
        self._users_cache[user_id] = user

        return user

    async def update_profile(self, user_id: str, updates: Dict[str, Any]) -> User:
        """
        Actualizar perfil del usuario en base de datos real

        Args:
            user_id: ID del usuario
            updates: Campos a actualizar

        Returns:
            Usuario actualizado

        Raises:
            ValueError: Si el usuario no existe
        """
        user = await self.get_user_by_id(user_id)
        if not user:
            raise ValueError(f"Usuario {user_id} no encontrado")

        # Actualizar timestamp
        updates["updated_at"] = datetime.utcnow().isoformat()

        # Construir query dinámica para actualización
        set_parts = []
        values = []
        for field, value in updates.items():
            if field in [
                "email",
                "name",
                "avatar_url",
                "tokens",
                "level",
                "experience",
                "subscription",
                "updated_at",
            ]:
                set_parts.append(f"{field} = ?")
                values.append(value)

        if not set_parts:
            return user  # Nada que actualizar

        values.append(user_id)  # Para WHERE clause

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"""
                UPDATE users SET {', '.join(set_parts)} WHERE id = ?
            """,
                values,
            )

        # Actualizar objeto en memoria
        for field, value in updates.items():
            if hasattr(user, field):
                setattr(user, field, value)

        # Actualizar cache
        self._users_cache[user_id] = user

        return user

    async def get_token_balance(self, user_id: str) -> Dict[str, Any]:
        """
        Obtener balance detallado de tokens del usuario

        Args:
            user_id: ID del usuario

        Returns:
            Diccionario con información del balance
        """
        user = await self.get_user_by_id(user_id)
        if not user:
            raise ValueError(f"Usuario {user_id} no encontrado")

        # Calcular límites basados en suscripción
        if user.subscription == "free":
            daily_limit = 100
            monthly_limit = 1000
        elif user.subscription == "premium":
            daily_limit = 1000
            monthly_limit = 10000
        else:  # enterprise
            daily_limit = 10000
            monthly_limit = 100000

        # Calcular experiencia para siguiente nivel
        next_level_exp = user.level * 1000

        return {
            "current_tokens": user.tokens,
            "level": user.level,
            "experience": user.experience,
            "next_level_experience": next_level_exp,
            "daily_limit": daily_limit,
            "monthly_limit": monthly_limit,
        }

    async def create_payment_session(
        self, user_id: str, amount: int, price_cents: int, payment_method: str
    ) -> Dict[str, Any]:
        """
        Crear sesión de pago para compra de tokens

        Args:
            user_id: ID del usuario
            amount: Cantidad de tokens
            price_cents: Precio en centavos
            payment_method: Método de pago

        Returns:
            Información de la sesión de pago
        """
        session_id = str(uuid.uuid4())

        # TODO: Integrar con proveedores de pago reales (Stripe, PayPal, etc.)
        # Por ahora retornamos una sesión mock
        payment_session = {
            "session_id": session_id,
            "amount": amount,
            "price_cents": price_cents,
            "payment_method": payment_method,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (
                datetime.utcnow().replace(hour=23, minute=59, second=59)
            ).isoformat(),
        }

        return payment_session

    async def get_token_transactions(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Obtener historial de transacciones de tokens desde base de datos real

        Args:
            user_id: ID del usuario
            limit: Número máximo de transacciones
            offset: Offset para paginación

        Returns:
            Lista de transacciones
        """
        # Verificar que el usuario existe
        user = await self.get_user_by_id(user_id)
        if not user:
            raise ValueError(f"Usuario {user_id} no encontrado")

        # Obtener transacciones desde base de datos real
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, type, amount, description, balance_after, timestamp
                FROM token_transactions
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """,
                (user_id, limit, offset),
            )

            transactions = []
            for row in cursor.fetchall():
                transactions.append(
                    {
                        "id": row[0],
                        "type": row[1],
                        "amount": row[2],
                        "description": row[3],
                        "balance_after": row[4],
                        "timestamp": row[5],
                    }
                )

            return transactions

    async def upload_avatar(self, user_id: str, avatar_data: bytes) -> str:
        """
        Subir avatar del usuario

        Args:
            user_id: ID del usuario
            avatar_data: Datos binarios del avatar

        Returns:
            URL del avatar subido

        Raises:
            ValueError: Si el usuario no existe
        """
        user = await self.get_user_by_id(user_id)
        if not user:
            raise ValueError(f"Usuario {user_id} no encontrado")

        # TODO: Implementar subida real a cloud storage (AWS S3, Cloudinary, etc.)  # noqa: E501
        # Por ahora generamos una URL mock
        avatar_filename = f"avatar_{user_id}_{int(asyncio.get_event_loop().time())}.jpg"
        avatar_url = f"https://storage.sheily.ai/avatars/{avatar_filename}"

        # Actualizar usuario con nueva URL
        await self.update_profile(user_id, {"avatar_url": avatar_url})

        return avatar_url

    async def add_experience(self, user_id: str, amount: int) -> Dict[str, Any]:
        """
        Añadir experiencia al usuario

        Args:
            user_id: ID del usuario
            amount: Cantidad de experiencia a añadir

        Returns:
            Información del progreso actualizado
        """
        user = await self.get_user_by_id(user_id)
        if not user:
            raise ValueError(f"Usuario {user_id} no encontrado")

        old_level = user.level
        user.add_experience(amount)

        # Verificar si subió de nivel
        level_up = user.level > old_level

        # Actualizar usuario
        await self.update_profile(
            user_id, {"level": user.level, "experience": user.experience}
        )

        return {
            "experience_gained": amount,
            "new_experience": user.experience,
            "level_up": level_up,
            "new_level": user.level if level_up else old_level,
            "progress": user.get_level_progress(),
        }

    async def spend_tokens(
        self, user_id: str, amount: int, description: str = ""
    ) -> Dict[str, Any]:
        """
        Gastar tokens del usuario con persistencia real

        Args:
            user_id: ID del usuario
            amount: Cantidad de tokens a gastar
            description: Descripción de la transacción

        Returns:
            Información de la transacción

        Raises:
            ValueError: Si no hay suficientes tokens
        """
        user = await self.get_user_by_id(user_id)
        if not user:
            raise ValueError(f"Usuario {user_id} no encontrado")

        if not user.spend_tokens(amount):
            raise ValueError(f"Usuario {user_id} no tiene suficientes tokens")

        # Actualizar usuario en BD
        await self.update_profile(user_id, {"tokens": user.tokens})

        # Crear y persistir transacción real en BD
        transaction_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO token_transactions (id, user_id, type, amount, description, balance_after, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    transaction_id,
                    user_id,
                    "spent",
                    amount,
                    description or f"Uso de {amount} tokens",
                    user.tokens,
                    timestamp,
                ),
            )

        transaction = {
            "id": transaction_id,
            "type": "spent",
            "amount": amount,
            "description": description or f"Uso de {amount} tokens",
            "timestamp": timestamp,
            "balance_after": user.tokens,
        }

        return {
            "success": True,
            "transaction": transaction,
            "remaining_tokens": user.tokens,
        }

    async def add_tokens(
        self, user_id: str, amount: int, description: str = ""
    ) -> Dict[str, Any]:
        """
        Añadir tokens al usuario con persistencia real

        Args:
            user_id: ID del usuario
            amount: Cantidad de tokens a añadir
            description: Descripción de la transacción

        Returns:
            Información de la transacción
        """
        user = await self.get_user_by_id(user_id)
        if not user:
            raise ValueError(f"Usuario {user_id} no encontrado")

        user.add_tokens(amount)

        # Actualizar usuario en BD
        await self.update_profile(user_id, {"tokens": user.tokens})

        # Crear y persistir transacción real en BD
        transaction_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO token_transactions (id, user_id, type, amount, description, balance_after, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    transaction_id,
                    user_id,
                    "earned",
                    amount,
                    description or f"Recompensa de {amount} tokens",
                    user.tokens,
                    timestamp,
                ),
            )

        transaction = {
            "id": transaction_id,
            "type": "earned",
            "amount": amount,
            "description": description or f"Recompensa de {amount} tokens",
            "timestamp": timestamp,
            "balance_after": user.tokens,
        }

        return {
            "success": True,
            "transaction": transaction,
            "new_balance": user.tokens,
        }

    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Obtener estadísticas completas del usuario

        Args:
            user_id: ID del usuario

        Returns:
            Estadísticas del usuario
        """
        user = await self.get_user_by_id(user_id)
        if not user:
            raise ValueError(f"Usuario {user_id} no encontrado")

        # TODO: Calcular estadísticas reales desde la base de datos
        # Por ahora retornamos estadísticas mock
        stats = {
            "total_conversations": 25,
            "total_messages": 150,
            "total_tokens_spent": 500,
            "total_tokens_earned": 1500,
            "average_response_time": 1.2,
            "favorite_model": "default-model",
            "last_activity": datetime.utcnow().isoformat(),
            "level_progress": user.get_level_progress(),
        }

        return stats
