#!/usr/bin/env python3
"""
WebAuthn Service - Servicio de Autenticación WebAuthn

Este módulo implementa autenticación WebAuthn con capacidades de:
- Registro de credenciales
- Autenticación sin contraseña
- Gestión de claves de seguridad
- Integración con navegadores
"""

import hashlib
import json
import logging
import secrets
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WebAuthnService:
    """Servicio de autenticación WebAuthn"""

    def __init__(self):
        """Inicializar servicio WebAuthn"""
        self.credentials = {}  # user_id -> list of credentials
        self.challenges = {}  # challenge -> user_id
        self.rp_id = "localhost"  # Relying Party ID
        self.rp_name = "Sheily AI System"

        self.initialized = True
        logger.info("WebAuthnService inicializado")

    def generate_registration_challenge(self, user_id: str) -> Dict[str, Any]:
        """Generar desafío de registro para un usuario"""
        # Generar challenge aleatorio
        challenge = secrets.token_bytes(32)
        challenge_b64 = challenge.hex()

        # Almacenar challenge temporalmente
        self.challenges[challenge_b64] = user_id

        registration_options = {
            "challenge": challenge_b64,
            "rp": {"name": self.rp_name, "id": self.rp_id},
            "user": {
                "id": user_id,
                "name": user_id,
                "displayName": f"Usuario {user_id}",
            },
            "pubKeyCredParams": [
                {"alg": -7, "type": "public-key"},  # ES256
                {"alg": -257, "type": "public-key"},  # RS256
            ],
            "authenticatorSelection": {
                "authenticatorAttachment": "platform",
                "userVerification": "preferred",
            },
            "timeout": 60000,
        }

        return {"challenge": challenge_b64, "options": registration_options}

    def generate_authentication_challenge(self, user_id: str) -> Dict[str, Any]:
        """Generar desafío de autenticación para un usuario"""
        if user_id not in self.credentials:
            return {"error": "Usuario no registrado"}

        # Generar challenge aleatorio
        challenge = secrets.token_bytes(32)
        challenge_b64 = challenge.hex()

        # Almacenar challenge temporalmente
        self.challenges[challenge_b64] = user_id

        # Obtener credential IDs del usuario
        credential_ids = [cred["id"] for cred in self.credentials[user_id]]

        authentication_options = {
            "challenge": challenge_b64,
            "timeout": 60000,
            "rpId": self.rp_id,
            "allowCredentials": [
                {"type": "public-key", "id": cred_id} for cred_id in credential_ids
            ],
            "userVerification": "preferred",
        }

        return {"challenge": challenge_b64, "options": authentication_options}

    def register_credential(
        self, user_id: str, challenge: str, credential_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Registrar una nueva credencial WebAuthn"""
        # Verificar challenge
        if challenge not in self.challenges:
            return {"error": "Challenge inválido o expirado"}

        if self.challenges[challenge] != user_id:
            return {"error": "Challenge no corresponde al usuario"}

        # Simular verificación de la credencial (en producción sería más compleja)
        credential_id = credential_data.get("id", secrets.token_hex(32))
        public_key = credential_data.get("publicKey", secrets.token_hex(64))

        # Almacenar credencial
        if user_id not in self.credentials:
            self.credentials[user_id] = []

        credential = {
            "id": credential_id,
            "publicKey": public_key,
            "type": "public-key",
            "registered_at": secrets.token_hex(16),
            "signCount": 0,
        }

        self.credentials[user_id].append(credential)

        # Limpiar challenge usado
        del self.challenges[challenge]

        logger.info(f"Credencial registrada para usuario {user_id}")
        return {
            "success": True,
            "credential_id": credential_id,
            "message": "Credencial registrada exitosamente",
        }

    def authenticate_credential(
        self, user_id: str, challenge: str, assertion_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Autenticar usando una credencial WebAuthn"""
        # Verificar challenge
        if challenge not in self.challenges:
            return {"error": "Challenge inválido o expirado"}

        if self.challenges[challenge] != user_id:
            return {"error": "Challenge no corresponde al usuario"}

        if user_id not in self.credentials:
            return {"error": "Usuario no registrado"}

        credential_id = assertion_data.get("id")
        signature = assertion_data.get("signature")

        # Buscar credencial
        credential = None
        for cred in self.credentials[user_id]:
            if cred["id"] == credential_id:
                credential = cred
                break

        if not credential:
            return {"error": "Credencial no encontrada"}

        # Simular verificación de firma (en producción usaría crypto real)
        if not signature:
            return {"error": "Firma faltante"}

        # Actualizar contador de firmas
        credential["signCount"] += 1

        # Limpiar challenge usado
        del self.challenges[challenge]

        logger.info(f"Autenticación exitosa para usuario {user_id}")
        return {
            "success": True,
            "user_id": user_id,
            "credential_id": credential_id,
            "message": "Autenticación exitosa",
        }

    def get_user_credentials(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtener credenciales de un usuario"""
        if user_id not in self.credentials:
            return []

        # Devolver información básica de las credenciales (sin claves privadas)
        return [
            {
                "id": cred["id"],
                "type": cred["type"],
                "registered_at": cred["registered_at"],
                "signCount": cred["signCount"],
            }
            for cred in self.credentials[user_id]
        ]

    def revoke_credential(self, user_id: str, credential_id: str) -> Dict[str, Any]:
        """Revocar una credencial"""
        if user_id not in self.credentials:
            return {"error": "Usuario no encontrado"}

        # Buscar y eliminar credencial
        for i, cred in enumerate(self.credentials[user_id]):
            if cred["id"] == credential_id:
                del self.credentials[user_id][i]
                logger.info(
                    f"Credencial {credential_id} revocada para usuario {user_id}"
                )
                return {"success": True, "message": "Credencial revocada exitosamente"}

        return {"error": "Credencial no encontrada"}

    def cleanup_expired_challenges(self, max_age_seconds: int = 300) -> int:
        """Limpiar challenges expirados"""
        # En una implementación real, esto debería hacerse automáticamente
        # Por simplicidad, limpiamos todos los challenges antiguos
        expired_count = len(self.challenges)
        self.challenges.clear()
        logger.info(f"Limpiados {expired_count} challenges expirados")
        return expired_count

    def get_service_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del servicio"""
        total_users = len(self.credentials)
        total_credentials = sum(len(creds) for creds in self.credentials.values())
        active_challenges = len(self.challenges)

        return {
            "initialized": self.initialized,
            "total_users": total_users,
            "total_credentials": total_credentials,
            "active_challenges": active_challenges,
            "rp_id": self.rp_id,
            "rp_name": self.rp_name,
            "capabilities": ["registration", "authentication", "credential_management"],
            "version": "1.0.0",
        }
