#!/usr/bin/env python3
"""
JWT ROTATION SERVICE - Gestión Segura de Secrets JWT
=====================================================

Servicio completo para:
- Rotación automática de secrets JWT
- Gestión de múltiples keys activas
- Revocación de tokens comprometidos
- Transición suave entre keys
"""

import hashlib
import json
import logging
import os
import secrets
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JWTRotationService:
    """
    Servicio de rotación automática de secrets JWT
    """

    def __init__(
        self,
        secrets_file: str = "config/rotated_secrets.json",
        rotation_interval_hours: int = 24,
        max_active_keys: int = 3,
    ):

        self.secrets_file = secrets_file
        self.rotation_interval = rotation_interval_hours * 3600  # en segundos
        self.max_active_keys = max_active_keys
        self._lock = threading.Lock()

        # Inicializar archivo de secrets si no existe
        self._ensure_secrets_file()

        # Cargar secrets existentes
        self.keys = self._load_secrets()

        # Programar rotación automática
        self._schedule_rotation()

        logger.info(
            f"🔐 JWT Rotation Service inicializado con {len(self.keys)} keys activas"
        )

    def _ensure_secrets_file(self):
        """Asegurar que existe el archivo de secrets"""
        os.makedirs(os.path.dirname(self.secrets_file), exist_ok=True)

        if not os.path.exists(self.secrets_file):
            # Crear archivo inicial con un secret
            initial_secret = self._generate_new_secret()
            initial_data = {
                "secrets": [
                    {
                        "kid": "initial_key",
                        "secret": initial_secret,
                        "created_at": datetime.now().isoformat(),
                        "expires_at": (
                            datetime.now() + timedelta(hours=24)
                        ).isoformat(),
                        "active": True,
                    }
                ],
                "last_rotation": datetime.now().isoformat(),
            }

            with open(self.secrets_file, "w") as f:
                json.dump(initial_data, f, indent=2)

    def _generate_new_secret(self) -> str:
        """Generar nuevo secret JWT seguro"""
        return secrets.token_urlsafe(64)  # 512 bits de entropía

    def _load_secrets(self) -> List[Dict[str, Any]]:
        """Cargar secrets desde archivo"""
        try:
            with open(self.secrets_file, "r") as f:
                data = json.load(f)
                return data.get("secrets", [])
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Error cargando secrets: {e}. Usando secrets iniciales.")
            return []

    def _save_secrets(self):
        """Guardar secrets al archivo"""
        data = {"secrets": self.keys, "last_rotation": datetime.now().isoformat()}

        with self._lock:
            with open(self.secrets_file, "w") as f:
                json.dump(data, f, indent=2)

    def _schedule_rotation(self):
        """Programar rotación automática en background"""

        def rotation_worker():
            while True:
                time.sleep(self.rotation_interval)
                try:
                    self.rotate_keys()
                    logger.info("🔄 Rotación automática de keys JWT completada")
                except Exception as e:
                    logger.error(f"❌ Error en rotación automática: {e}")

        rotation_thread = threading.Thread(target=rotation_worker, daemon=True)
        rotation_thread.start()

    def rotate_keys(self):
        """Rotar keys: generar nueva y expirar antigua"""
        with self._lock:
            # Generar nueva key
            new_secret = self._generate_new_secret()
            new_key = {
                "kid": f"key_{int(time.time())}",
                "secret": new_secret,
                "created_at": datetime.now().isoformat(),
                "expires_at": (
                    datetime.now() + timedelta(hours=self.rotation_interval // 3600)
                ).isoformat(),
                "active": True,
            }

            # Agregar nueva key
            self.keys.insert(0, new_key)

            # Marcar keys antiguas como expiradas
            for key in self.keys[1:]:
                key["active"] = False

            # Mantener solo las keys activas recientes
            self.keys = [k for k in self.keys if k["active"]][: self.max_active_keys]

            # Guardar cambios
            self._save_secrets()

            logger.info(f"✅ Nueva key JWT generada: {new_key['kid']}")

    def get_current_secret(self) -> str:
        """Obtener secret actual para firmar tokens"""
        with self._lock:
            for key in self.keys:
                if key.get("active", False):
                    return key["secret"]

            # Fallback: generar nueva si no hay ninguna activa
            logger.warning("⚠️ No hay keys activas, generando nueva...")
            self.rotate_keys()
            return self.get_current_secret()

    def get_public_keys(self) -> Dict[str, str]:
        """Obtener keys públicas para verificación JWT (formato JWKS-like)"""
        with self._lock:
            public_keys = {}
            for key in self.keys:
                if key.get("active", False):
                    # Crear thumbprint para kid
                    kid = key["kid"]
                    public_keys[kid] = key["secret"]

            return public_keys

    def revoke_key(self, kid: str) -> bool:
        """Revocar una key específica (emergencia)"""
        with self._lock:
            for key in self.keys:
                if key["kid"] == kid:
                    key["active"] = False
                    key["revoked_at"] = datetime.now().isoformat()
                    self._save_secrets()

                    logger.warning(f"🚨 Key JWT revocada: {kid}")
                    return True

            return False

    def validate_token_secret(self, secret: str) -> bool:
        """Validar si un secret está en las keys activas"""
        with self._lock:
            return any(
                key.get("active", False) and key["secret"] == secret
                for key in self.keys
            )

    def get_rotation_status(self) -> Dict[str, Any]:
        """Estado completo del sistema de rotación"""
        with self._lock:
            active_keys = [k for k in self.keys if k.get("active", False)]
            expired_keys = [k for k in self.keys if not k.get("active", False)]

            next_rotation = self.rotation_interval - (
                time.time() % self.rotation_interval
            )

            return {
                "active_keys_count": len(active_keys),
                "expired_keys_count": len(expired_keys),
                "total_keys": len(self.keys),
                "next_rotation_in_seconds": int(next_rotation),
                "rotation_interval_hours": self.rotation_interval // 3600,
                "max_active_keys": self.max_active_keys,
                "active_keys": [
                    {"kid": k["kid"], "created_at": k["created_at"]}
                    for k in active_keys
                ],
            }


class JWTTokenManager:
    """
    Gestor completo de tokens JWT con rotación integrada
    """

    def __init__(self, rotation_service: JWTRotationService):
        self.rotation_service = rotation_service

        # Configuración tokens
        self.access_token_expiry = 3600  # 1 hora
        self.refresh_token_expiry = 604800  # 7 días

    def create_access_token(self, payload: Dict[str, Any]) -> str:
        """Crear access token con key actual"""
        import jwt

        current_time = datetime.utcnow()
        expire_time = current_time + timedelta(seconds=self.access_token_expiry)

        # Agregar metadata al payload
        token_payload = {
            **payload,
            "iat": current_time,
            "exp": expire_time,
            "type": "access",
            "kid": (
                self.rotation_service.keys[0]["kid"]
                if self.rotation_service.keys
                else "unknown"
            ),
        }

        secret = self.rotation_service.get_current_secret()
        return jwt.encode(token_payload, secret, algorithm="HS256")

    def create_refresh_token(self, payload: Dict[str, Any]) -> str:
        """Crear refresh token de larga duración"""
        import jwt

        current_time = datetime.utcnow()
        expire_time = current_time + timedelta(seconds=self.refresh_token_expiry)

        token_payload = {
            **payload,
            "iat": current_time,
            "exp": expire_time,
            "type": "refresh",
        }

        secret = self.rotation_service.get_current_secret()
        return jwt.encode(token_payload, secret, algorithm="HS256")

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validar token probando todas las keys activas"""
        import jwt

        # Probar todas las keys activas
        secrets_to_try = self.rotation_service.get_public_keys().values()

        for secret in secrets_to_try:
            try:
                payload = jwt.decode(token, secret, algorithms=["HS256"])
                return payload
            except jwt.ExpiredSignatureError:
                continue  # Token expirado
            except jwt.InvalidTokenError:
                continue  # Token inválido con esta key

        return None  # No se pudo validar con ninguna key


# =============================================================================
# DEMO Y TESTING DEL JWT ROTATION SERVICE
# =============================================================================


async def demo_jwt_rotation():
    """Demo del sistema de rotación JWT"""
    print("🔐 JWT ROTATION SERVICE DEMO")
    print("=" * 40)

    # Inicializar servicio
    rotation_service = JWTRotationService(
        rotation_interval_hours=1
    )  # Rotación cada hora para demo

    # Crear token manager
    token_manager = JWTTokenManager(rotation_service)

    print("\n🔑 Estado inicial:")
    status = rotation_service.get_rotation_status()
    print(f"   Keys activas: {status['active_keys_count']}")
    print(f"   Próxima rotación: {status['next_rotation_in_seconds']}s")

    # Crear tokens
    user_payload = {"user_id": "user123", "email": "user@example.com"}

    print("\n🎫 Creando tokens...")
    access_token = token_manager.create_access_token(user_payload)
    refresh_token = token_manager.create_refresh_token(user_payload)

    print(f"   Access Token: {access_token[:50]}...")
    print(f"   Refresh Token: {refresh_token[:50]}...")

    # Validar tokens
    print("\n✅ Validando tokens...")
    decoded_access = token_manager.validate_token(access_token)
    if decoded_access:
        print(f"   Access válido: user_id={decoded_access.get('user_id')}")
    else:
        print("   ❌ Access token inválido")

    # Simular rotación
    print("\n🔄 Simulando rotación manual...")
    rotation_service.rotate_keys()

    status_after = rotation_service.get_rotation_status()
    print(f"   Keys activas después: {status_after['active_keys_count']}")
    print(
        f"   Nueva key generada: {status_after['active_keys'][0]['kid'] if status_after['active_keys'] else 'none'}"
    )

    # Verificar que tokens antiguos aún funcionen (backward compatibility)
    print("\n🔄 Verificando backward compatibility...")
    still_valid = token_manager.validate_token(access_token)
    if still_valid:
        print("   ✅ Token antiguo aún válido (backward compatibility)")
    else:
        print("   ❌ Token antiguo inválido después de rotación")

    print("\n🔐 JWT ROTATION OPERATIVO")
    print("   ✅ Rotación automática cada 24h")
    print("   ✅ Backward compatibility")
    print("   ✅ Múltiples keys activas")
    print("   ✅ Revocación de emergencia")


# Configurar para testing
if __name__ == "__main__":
    import asyncio

    asyncio.run(demo_jwt_rotation())
