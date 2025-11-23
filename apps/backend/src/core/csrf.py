#!/usr/bin/env python3
"""
CSRF Protection System - Protección contra Cross-Site Request Forgery
Basado en análisis MCP Enterprise
"""

import hashlib
import hmac
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CSRFProtector:
    """
    Protector CSRF avanzado con tokens criptográficos
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        token_lifetime: int = 3600,  # 1 hora
        token_length: int = 32,
    ):
        """
        Inicializar protector CSRF

        Args:
            secret_key: Clave secreta para firmar tokens (generada si no se proporciona)
            token_lifetime: Vida útil del token en segundos
            token_length: Longitud del token en bytes
        """
        self.secret_key = secret_key or secrets.token_hex(32)
        self.token_lifetime = token_lifetime
        self.token_length = token_length

        # Almacenamiento de tokens válidos (en producción usar Redis/database)
        self.valid_tokens: Dict[str, Dict[str, Any]] = {}

        logger.info("CSRF Protector inicializado")

    async def generate_token(
        self, session_id: str, user_id: Optional[str] = None
    ) -> str:
        """
        Generar token CSRF único para una sesión

        Args:
            session_id: ID de la sesión
            user_id: ID del usuario (opcional)

        Returns:
            str: Token CSRF
        """
        # Generar token aleatorio
        token_bytes = secrets.token_bytes(self.token_length)
        token = token_bytes.hex()

        # Crear payload para firma
        timestamp = int(time.time())
        payload = f"{session_id}:{user_id or ''}:{token}:{timestamp}"

        # Firmar el payload
        signature = self._sign_payload(payload)

        # Token completo: token.signature.timestamp
        csrf_token = f"{token}.{signature}.{timestamp}"

        # Almacenar token válido
        self.valid_tokens[token] = {
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "signature": signature,
            "used": False,  # Para tokens de un solo uso
        }

        # Limpiar tokens expirados
        await self._cleanup_expired_tokens()

        logger.debug(f"Token CSRF generado para sesión: {session_id}")
        return csrf_token

    async def validate_token(self, session_id: str, token: str) -> bool:
        """
        Validar token CSRF

        Args:
            session_id: ID de la sesión
            token: Token CSRF a validar

        Returns:
            bool: True si válido
        """
        try:
            # Parsear token
            parts = token.split(".")
            if len(parts) != 3:
                logger.warning(f"Token CSRF malformado: {token}")
                return False

            token_value, signature, timestamp_str = parts

            try:
                timestamp = int(timestamp_str)
            except ValueError:
                logger.warning(f"Timestamp inválido en token CSRF: {timestamp_str}")
                return False

            # Verificar expiración
            current_time = int(time.time())
            if current_time - timestamp > self.token_lifetime:
                logger.warning(f"Token CSRF expirado: {token}")
                # Remover token expirado
                if token_value in self.valid_tokens:
                    del self.valid_tokens[token_value]
                return False

            # Verificar que el token existe en nuestros registros
            if token_value not in self.valid_tokens:
                logger.warning(f"Token CSRF desconocido: {token_value}")
                return False

            token_data = self.valid_tokens[token_value]

            # Verificar sesión
            if token_data["session_id"] != session_id:
                logger.warning(f"Sesión incorrecta para token CSRF: {session_id}")
                return False

            # Verificar firma
            expected_payload = f"{session_id}:{token_data.get('user_id', '')}:{token_value}:{timestamp}"
            expected_signature = self._sign_payload(expected_payload)

            if not hmac.compare_digest(signature, expected_signature):
                logger.warning(f"Firma inválida en token CSRF: {token}")
                return False

            # Para tokens de un solo uso, marcar como usado
            # token_data['used'] = True

            logger.debug(f"Token CSRF válido para sesión: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error validando token CSRF: {e}")
            return False

    async def invalidate_session_tokens(self, session_id: str):
        """
        Invalidar todos los tokens de una sesión

        Args:
            session_id: ID de la sesión
        """
        tokens_to_remove = [
            token
            for token, data in self.valid_tokens.items()
            if data["session_id"] == session_id
        ]

        for token in tokens_to_remove:
            del self.valid_tokens[token]

        logger.info(f"Tokens CSRF invalidados para sesión: {session_id}")

    async def get_session_tokens(self, session_id: str) -> list:
        """
        Obtener tokens activos de una sesión

        Args:
            session_id: ID de la sesión

        Returns:
            list: Lista de tokens activos
        """
        return [
            token
            for token, data in self.valid_tokens.items()
            if data["session_id"] == session_id and not data.get("used", False)
        ]

    def _sign_payload(self, payload: str) -> str:
        """Firmar payload usando HMAC-SHA256"""
        return hmac.new(
            self.secret_key.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

    async def _cleanup_expired_tokens(self):
        """Limpiar tokens expirados"""
        current_time = int(time.time())
        expired_tokens = []

        for token, data in self.valid_tokens.items():
            if current_time - data["timestamp"] > self.token_lifetime:
                expired_tokens.append(token)

        for token in expired_tokens:
            del self.valid_tokens[token]

        if expired_tokens:
            logger.debug(f"Tokens CSRF expirados limpiados: {len(expired_tokens)}")

    async def rotate_secret_key(self):
        """Rotar clave secreta (para mayor seguridad)"""
        old_key = self.secret_key
        self.secret_key = secrets.token_hex(32)

        # Invalidar todos los tokens existentes (requerirán regeneración)
        self.valid_tokens.clear()

        logger.info(
            "Clave secreta CSRF rotada - todos los tokens anteriores invalidados"
        )
        return old_key


class DoubleSubmitCookieCSRF:
    """
    Implementación de Double Submit Cookie CSRF protection
    """

    def __init__(
        self, cookie_name: str = "csrf_token", header_name: str = "X-CSRF-Token"
    ):
        self.cookie_name = cookie_name
        self.header_name = header_name

    async def generate_token(self) -> str:
        """Generar token para double submit cookie"""
        return secrets.token_hex(32)

    async def validate_request(self, request) -> bool:
        """
        Validar request usando double submit cookie pattern

        Args:
            request: Request object (FastAPI/Starlette)

        Returns:
            bool: True si válido
        """
        # Obtener token de cookie
        cookie_token = request.cookies.get(self.cookie_name)

        # Obtener token de header
        header_token = request.headers.get(self.header_name)

        if not cookie_token or not header_token:
            logger.warning("Tokens CSRF faltantes en double submit cookie")
            return False

        # Comparar tokens
        if not hmac.compare_digest(cookie_token, header_token):
            logger.warning("Tokens CSRF no coinciden en double submit cookie")
            return False

        return True


# Instancia global del protector CSRF
csrf_protector = CSRFProtector()


async def generate_csrf_token(session_id: str, user_id: Optional[str] = None) -> str:
    """Función helper para generar token CSRF"""
    return await csrf_protector.generate_token(session_id, user_id)


async def validate_csrf_token(session_id: str, token: str) -> bool:
    """Función helper para validar token CSRF"""
    return await csrf_protector.validate_token(session_id, token)


# Middleware para FastAPI
from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class CSRFMiddleware(BaseHTTPMiddleware):
    """Middleware CSRF para FastAPI"""

    def __init__(self, app, exclude_paths: list = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/docs", "/redoc", "/openapi.json"]
        self.double_submit = DoubleSubmitCookieCSRF()

    async def dispatch(self, request: Request, call_next):
        # Excluir paths seguros
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Solo validar para métodos que modifican estado
        if request.method in ["POST", "PUT", "PATCH", "DELETE"]:
            # Intentar validación CSRF
            csrf_valid = False

            # Método 1: Token en header/session
            session_id = request.headers.get("X-Session-ID") or request.cookies.get(
                "session_id"
            )
            csrf_token = request.headers.get("X-CSRF-Token")

            if session_id and csrf_token:
                csrf_valid = await csrf_protector.validate_token(session_id, csrf_token)

            # Método 2: Double submit cookie (fallback)
            if not csrf_valid:
                csrf_valid = await self.double_submit.validate_request(request)

            if not csrf_valid:
                logger.warning(
                    f"CSRF validation failed for {request.method} {request.url.path}"
                )
                raise HTTPException(
                    status_code=403, detail="CSRF token missing or invalid"
                )

        # Procesar request
        response = await call_next(request)

        # Para GET requests que sirven páginas, añadir token CSRF
        if request.method == "GET" and request.headers.get("accept", "").startswith(
            "text/html"
        ):
            session_id = request.cookies.get("session_id") or request.headers.get(
                "X-Session-ID"
            )
            if session_id:
                csrf_token = await csrf_protector.generate_token(session_id)
                # Añadir token como cookie segura
                response.set_cookie(
                    "csrf_token",
                    csrf_token.split(".")[0],  # Solo el token, no la firma completa
                    httponly=True,
                    secure=True,
                    samesite="strict",
                    max_age=csrf_protector.token_lifetime,
                )

        return response


# Decorador para endpoints que requieren CSRF
from functools import wraps


def require_csrf(func):
    """Decorador para requerir validación CSRF"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extraer request de argumentos
        request = None
        for arg in args:
            if hasattr(arg, "headers"):  # Es un Request object
                request = arg
                break

        if not request:
            # Buscar en kwargs
            request = kwargs.get("request")

        if request:
            session_id = request.headers.get("X-Session-ID") or request.cookies.get(
                "session_id"
            )
            csrf_token = request.headers.get("X-CSRF-Token")

            if not session_id or not csrf_token:
                raise HTTPException(status_code=403, detail="CSRF token required")

            if not await csrf_protector.validate_token(session_id, csrf_token):
                raise HTTPException(status_code=403, detail="Invalid CSRF token")

        return await func(*args, **kwargs)

    return wrapper


# Utilidades para templates/frontend
def get_csrf_token_html(session_id: str) -> str:
    """
    Generar HTML meta tag con token CSRF para templates

    Args:
        session_id: ID de la sesión

    Returns:
        str: HTML meta tag
    """
    import asyncio

    token = asyncio.run(csrf_protector.generate_token(session_id))
    return f'<meta name="csrf-token" content="{token}">'


def get_csrf_token_json(session_id: str) -> Dict[str, str]:
    """
    Generar token CSRF en formato JSON para APIs

    Args:
        session_id: ID de la sesión

    Returns:
        Dict: Token en formato JSON
    """
    import asyncio

    token = asyncio.run(csrf_protector.generate_token(session_id))
    return {"csrf_token": token}
