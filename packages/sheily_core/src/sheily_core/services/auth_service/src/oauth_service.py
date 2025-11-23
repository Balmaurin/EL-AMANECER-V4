#!/usr/bin/env python3
"""
OAuth Service - OAuth 2.0 Authentication Service

Este módulo implementa servicio OAuth 2.0 con capacidades de:
- Flujo de autorización
- Gestión de tokens
- Validación de clientes
"""

import logging
import secrets
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OAuthService:
    """Servicio OAuth 2.0"""

    def __init__(self):
        """Inicializar servicio OAuth"""
        self.clients = {}  # client_id -> client_secret
        self.tokens = {}  # access_token -> token_data
        self.auth_codes = {}  # auth_code -> code_data

        # Cliente por defecto para desarrollo
        self.clients["sheily_client"] = "sheily_secret"

        self.initialized = True
        logger.info("OAuthService inicializado")

    def register_client(
        self, client_id: str, client_secret: str, redirect_uri: str
    ) -> bool:
        """Registrar un nuevo cliente OAuth"""
        if client_id in self.clients:
            return False

        self.clients[client_id] = {
            "secret": client_secret,
            "redirect_uri": redirect_uri,
            "created_at": time.time(),
        }
        logger.info(f"Cliente OAuth registrado: {client_id}")
        return True

    def generate_auth_code(
        self, client_id: str, redirect_uri: str, scope: str = "read"
    ) -> Optional[str]:
        """Generar código de autorización"""
        if client_id not in self.clients:
            return None

        if self.clients[client_id]["redirect_uri"] != redirect_uri:
            return None

        auth_code = secrets.token_urlsafe(32)
        self.auth_codes[auth_code] = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "expires_at": time.time() + 600,  # 10 minutos
        }

        return auth_code

    def exchange_code_for_token(
        self, client_id: str, client_secret: str, auth_code: str
    ) -> Optional[Dict[str, Any]]:
        """Intercambiar código de autorización por token de acceso"""
        # Verificar cliente
        if (
            client_id not in self.clients
            or self.clients[client_id]["secret"] != client_secret
        ):
            return None

        # Verificar código
        if auth_code not in self.auth_codes:
            return None

        code_data = self.auth_codes[auth_code]

        if code_data["client_id"] != client_id:
            return None

        if time.time() > code_data["expires_at"]:
            del self.auth_codes[auth_code]
            return None

        # Generar token
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)

        token_data = {
            "client_id": client_id,
            "scope": code_data["scope"],
            "issued_at": time.time(),
            "expires_at": time.time() + 3600,  # 1 hora
            "refresh_token": refresh_token,
        }

        self.tokens[access_token] = token_data

        # Limpiar código usado
        del self.auth_codes[auth_code]

        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": refresh_token,
            "scope": code_data["scope"],
        }

    def validate_token(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Validar token de acceso"""
        if access_token not in self.tokens:
            return None

        token_data = self.tokens[access_token]

        if time.time() > token_data["expires_at"]:
            del self.tokens[access_token]
            return None

        return token_data

    def refresh_token(
        self, refresh_token: str, client_id: str, client_secret: str
    ) -> Optional[Dict[str, Any]]:
        """Refrescar token de acceso"""
        # Buscar token por refresh_token
        for token, data in self.tokens.items():
            if (
                data.get("refresh_token") == refresh_token
                and data["client_id"] == client_id
            ):
                # Verificar cliente
                if (
                    client_id not in self.clients
                    or self.clients[client_id]["secret"] != client_secret
                ):
                    return None

                # Generar nuevo token
                new_access_token = secrets.token_urlsafe(32)
                new_refresh_token = secrets.token_urlsafe(32)

                new_token_data = {
                    "client_id": client_id,
                    "scope": data["scope"],
                    "issued_at": time.time(),
                    "expires_at": time.time() + 3600,
                    "refresh_token": new_refresh_token,
                }

                # Reemplazar token
                del self.tokens[token]
                self.tokens[new_access_token] = new_token_data

                return {
                    "access_token": new_access_token,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "refresh_token": new_refresh_token,
                    "scope": data["scope"],
                }

        return None

    def revoke_token(self, access_token: str) -> bool:
        """Revocar token de acceso"""
        if access_token in self.tokens:
            del self.tokens[access_token]
            return True
        return False

    def get_service_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del servicio"""
        total_clients = len(self.clients)
        total_tokens = len(self.tokens)
        total_auth_codes = len(self.auth_codes)

        return {
            "initialized": self.initialized,
            "total_clients": total_clients,
            "total_tokens": total_tokens,
            "total_auth_codes": total_auth_codes,
            "capabilities": [
                "authorization_code_flow",
                "token_validation",
                "token_refresh",
            ],
            "version": "1.0.0",
        }
