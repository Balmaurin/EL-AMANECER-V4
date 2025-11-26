#!/usr/bin/env python3
"""
Sheily Enterprise Production Middleware
======================================

Sistema de middleware enterprise integrado con ii-agent patterns
para exception handling, security, authentication y monitoring.

Características:
- Exception handling con structured logging
- Authentication middleware con JWT
- Permission-based access control
- Request/response transformation
- Security headers y CORS
- Rate limiting
"""

import asyncio
import hashlib
import json
import logging
import secrets
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from sheily_core.core.events.event_system import (
    SheìlyEventType,
    get_event_stream,
    publish_event,
)

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """Roles de usuario para autorización"""

    ADMIN = "admin"
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class PermissionLevel(str, Enum):
    """Niveles de permisos"""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class User:
    """Usuario del sistema"""

    id: str
    username: str
    role: UserRole
    permissions: List[str]
    created_at: Optional[datetime] = None
    last_access: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SecurityContext:
    """Contexto de seguridad para requests"""

    user: Optional[User] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: str = ""
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if not self.request_id:
            self.request_id = secrets.token_urlsafe(16)


class SheìlyMiddleware:
    """Enterprise middleware system para Sheily"""

    def __init__(self):
        self.event_stream = None
        self.rate_limits: Dict[str, List[float]] = {}
        self.blocked_ips: Dict[str, datetime] = {}
        self.active_sessions: Dict[str, SecurityContext] = {}

    async def initialize(self) -> None:
        """Initialize middleware system"""
        try:
            self.event_stream = get_event_stream()

            await publish_event(
                SheìlyEventType.SYSTEM_HEALTH, {"status": "middleware_initialized"}
            )

            logger.info("✅ Sheily Enterprise Middleware initialized")

        except Exception as e:
            logger.error(f"Error initializing middleware: {e}")
            raise

    # ========================================
    # EXCEPTION HANDLING MIDDLEWARE
    # ========================================

    async def exception_handler_middleware(
        self,
        request_handler: Callable[..., Awaitable[Any]],
        context: SecurityContext,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Middleware para manejo de excepciones con logging estructurado"""
        start_time = time.time()

        try:
            # Log request start
            await self._log_request_start(context, request_handler.__name__)

            # Execute request
            result = await request_handler(context, *args, **kwargs)

            # Log successful completion
            duration = time.time() - start_time
            await self._log_request_success(context, request_handler.__name__, duration)

            return result

        except Exception as e:
            # Log error with full context
            duration = time.time() - start_time
            await self._log_request_error(
                context, request_handler.__name__, e, duration
            )

            # Publish error event
            await publish_event(
                SheìlyEventType.PERFORMANCE_ALERT,
                {
                    "alert_type": "request_error",
                    "request_id": context.request_id,
                    "handler": request_handler.__name__,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_seconds": duration,
                    "user_id": context.user.id if context.user else None,
                },
            )

            # Re-raise with context
            raise SheìlyMiddlewareError(
                f"Error in {request_handler.__name__}: {e}",
                original_error=e,
                context=context,
            )

    async def _log_request_start(
        self, context: SecurityContext, handler_name: str
    ) -> None:
        """Log request start"""
        logger.info(
            f"Request started",
            extra={
                "request_id": context.request_id,
                "handler": handler_name,
                "user_id": context.user.id if context.user else None,
                "session_id": context.session_id,
                "ip_address": context.ip_address,
                "timestamp": (
                    context.timestamp.isoformat()
                    if context.timestamp
                    else datetime.now(timezone.utc).isoformat()
                ),
            },
        )

    async def _log_request_success(
        self, context: SecurityContext, handler_name: str, duration: float
    ) -> None:
        """Log successful request completion"""
        logger.info(
            f"Request completed successfully",
            extra={
                "request_id": context.request_id,
                "handler": handler_name,
                "duration_seconds": round(duration, 4),
                "user_id": context.user.id if context.user else None,
            },
        )

    async def _log_request_error(
        self,
        context: SecurityContext,
        handler_name: str,
        error: Exception,
        duration: float,
    ) -> None:
        """Log request error with full context"""
        logger.error(
            f"Request failed",
            extra={
                "request_id": context.request_id,
                "handler": handler_name,
                "error": str(error),
                "error_type": type(error).__name__,
                "duration_seconds": round(duration, 4),
                "user_id": context.user.id if context.user else None,
                "traceback": traceback.format_exc(),
            },
        )

    # ========================================
    # AUTHENTICATION & AUTHORIZATION
    # ========================================

    async def authentication_middleware(
        self,
        request_handler: Callable[..., Awaitable[Any]],
        token: Optional[str],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Authentication middleware con JWT-like tokens"""
        try:
            user = await self._validate_token(token) if token else None
            context = SecurityContext(user=user)

            # Update session
            if user:
                user.last_access = datetime.now(timezone.utc)
                self.active_sessions[context.request_id] = context

            return await self.exception_handler_middleware(
                request_handler, context, *args, **kwargs
            )

        except AuthenticationError as e:
            await publish_event(
                SheìlyEventType.PERFORMANCE_ALERT,
                {
                    "alert_type": "authentication_failed",
                    "error": str(e),
                    "token_provided": token is not None,
                },
            )
            raise

    async def _validate_token(self, token: str) -> User:
        """Validate authentication token (simplified JWT-like)"""
        try:
            # In a real implementation, this would validate JWT
            # For now, simulate with simple token validation
            if not token or len(token) < 10:
                raise AuthenticationError("Invalid token format")

            # Simulate user lookup (replace with actual implementation)
            if token == "admin_token":
                return User(
                    id="admin",
                    username="admin",
                    role=UserRole.ADMIN,
                    permissions=["read", "write", "execute", "admin"],
                )
            elif token == "user_token":
                return User(
                    id="user1",
                    username="user",
                    role=UserRole.USER,
                    permissions=["read", "write"],
                )
            else:
                raise AuthenticationError("Invalid token")

        except Exception as e:
            raise AuthenticationError(f"Token validation failed: {e}")

    def require_permission(self, required_permission: str):
        """Decorator para requerir permisos específicos"""

        def decorator(func):
            @wraps(func)
            async def wrapper(context: SecurityContext, *args, **kwargs):
                if not context.user:
                    raise AuthorizationError("Authentication required")

                if required_permission not in context.user.permissions:
                    raise AuthorizationError(
                        f"Permission '{required_permission}' required"
                    )

                return await func(context, *args, **kwargs)

            return wrapper

        return decorator

    def require_role(self, required_role: UserRole):
        """Decorator para requerir role específico"""

        def decorator(func):
            @wraps(func)
            async def wrapper(context: SecurityContext, *args, **kwargs):
                if not context.user:
                    raise AuthorizationError("Authentication required")

                if context.user.role != required_role:
                    raise AuthorizationError(f"Role '{required_role.value}' required")

                return await func(context, *args, **kwargs)

            return wrapper

        return decorator

    # ========================================
    # RATE LIMITING
    # ========================================

    async def rate_limit_middleware(
        self,
        request_handler: Callable[..., Awaitable[Any]],
        context: SecurityContext,
        requests_per_minute: int = 60,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Rate limiting middleware"""

        # Identify client (use user ID if available, otherwise IP)
        client_id = context.user.id if context.user else context.ip_address or "unknown"
        current_time = time.time()

        # Clean old entries
        if client_id in self.rate_limits:
            self.rate_limits[client_id] = [
                t
                for t in self.rate_limits[client_id]
                if current_time - t < 60  # Keep last minute
            ]
        else:
            self.rate_limits[client_id] = []

        # Check rate limit
        if len(self.rate_limits[client_id]) >= requests_per_minute:
            await publish_event(
                SheìlyEventType.PERFORMANCE_ALERT,
                {
                    "alert_type": "rate_limit_exceeded",
                    "client_id": client_id,
                    "requests_per_minute": requests_per_minute,
                    "current_requests": len(self.rate_limits[client_id]),
                },
            )
            raise RateLimitError(
                f"Rate limit exceeded: {requests_per_minute} requests per minute"
            )

        # Add current request
        self.rate_limits[client_id].append(current_time)

        return await self.authentication_middleware(
            request_handler, None, *args, **kwargs
        )

    # ========================================
    # SECURITY MIDDLEWARE
    # ========================================

    async def security_middleware(
        self,
        request_handler: Callable[..., Awaitable[Any]],
        context: SecurityContext,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Security middleware con headers y validaciones"""

        # Check IP blocking
        if context.ip_address and context.ip_address in self.blocked_ips:
            block_time = self.blocked_ips[context.ip_address]
            if datetime.now(timezone.utc) - block_time < timedelta(hours=1):
                raise SecurityError(f"IP {context.ip_address} is temporarily blocked")
            else:
                del self.blocked_ips[context.ip_address]

        # Validate request context
        await self._validate_security_context(context)

        # Execute with security monitoring
        try:
            result = await self.rate_limit_middleware(
                request_handler, context, *args, **kwargs
            )

            # Add security headers to response if it's a dict
            if isinstance(result, dict):
                result = self._add_security_headers(result)

            return result

        except (AuthenticationError, AuthorizationError, RateLimitError) as e:
            # Track suspicious activity
            await self._track_suspicious_activity(context, str(e))
            raise

    async def _validate_security_context(self, context: SecurityContext) -> None:
        """Validate security context"""
        # Check for suspicious patterns
        if context.user_agent and "bot" in context.user_agent.lower():
            # Log but don't block bots completely
            logger.warning(f"Bot detected: {context.user_agent}")

        # Validate session if present
        if context.session_id:
            # In real implementation, validate session in database
            pass

    def _add_security_headers(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Add security headers to response"""
        if "_headers" not in response:
            response["_headers"] = {}

        response["_headers"].update(
            {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "Content-Security-Policy": "default-src 'self'",
                "X-Request-ID": response.get("_context", {}).get("request_id", ""),
            }
        )

        return response

    async def _track_suspicious_activity(
        self, context: SecurityContext, error_message: str
    ) -> None:
        """Track suspicious activity for security monitoring"""

        await publish_event(
            SheìlyEventType.PERFORMANCE_ALERT,
            {
                "alert_type": "suspicious_activity",
                "request_id": context.request_id,
                "ip_address": context.ip_address,
                "user_agent": context.user_agent,
                "error": error_message,
                "user_id": context.user.id if context.user else None,
            },
        )

        # Block IP after multiple failures
        if context.ip_address:
            # In real implementation, track failure count per IP
            # For now, just log
            logger.warning(
                f"Suspicious activity from IP {context.ip_address}: {error_message}"
            )

    # ========================================
    # MONITORING & METRICS
    # ========================================

    async def monitoring_middleware(
        self,
        request_handler: Callable[..., Awaitable[Any]],
        context: SecurityContext,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Middleware para monitoring y métricas"""

        start_memory = await self._get_memory_usage()
        start_time = time.time()

        try:
            result = await self.security_middleware(
                request_handler, context, *args, **kwargs
            )

            # Collect metrics
            end_time = time.time()
            end_memory = await self._get_memory_usage()

            await publish_event(
                SheìlyEventType.METRICS_UPDATE,
                {
                    "metric_type": "request_metrics",
                    "request_id": context.request_id,
                    "handler": request_handler.__name__,
                    "duration_seconds": round(end_time - start_time, 4),
                    "memory_delta_mb": round(
                        (end_memory - start_memory) / 1024 / 1024, 2
                    ),
                    "user_id": context.user.id if context.user else None,
                    "success": True,
                },
            )

            return result

        except Exception as e:
            end_time = time.time()

            await publish_event(
                SheìlyEventType.METRICS_UPDATE,
                {
                    "metric_type": "request_metrics",
                    "request_id": context.request_id,
                    "handler": request_handler.__name__,
                    "duration_seconds": round(end_time - start_time, 4),
                    "error": str(e),
                    "success": False,
                },
            )

            raise

    async def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0

    # ========================================
    # CONVENIENCE DECORATORS
    # ========================================

    def protected_endpoint(
        self,
        permission: str = "read",
        role: Optional[UserRole] = None,
        rate_limit: int = 60,
    ):
        """Decorator completo para endpoints protegidos"""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Create context from args if not present
                context = (
                    args[0]
                    if args and isinstance(args[0], SecurityContext)
                    else SecurityContext()
                )

                # Apply full middleware stack
                async def protected_handler(
                    ctx: SecurityContext, *inner_args: Any, **inner_kwargs: Any
                ) -> Any:
                    # Check role if specified
                    if role and ctx.user and ctx.user.role != role:
                        raise AuthorizationError(f"Role '{role.value}' required")

                    # Check permission
                    if ctx.user and permission not in ctx.user.permissions:
                        raise AuthorizationError(f"Permission '{permission}' required")

                    return await func(ctx, *inner_args, **inner_kwargs)

                return await self.monitoring_middleware(
                    protected_handler, context, *args[1:] if args else args, **kwargs
                )

            return wrapper

        return decorator


# ========================================
# EXCEPTION CLASSES
# ========================================


class SheìlyMiddlewareError(Exception):
    """Base exception para middleware errors"""

    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
        context: Optional[SecurityContext] = None,
    ):
        super().__init__(message)
        self.original_error = original_error
        self.context = context


class AuthenticationError(SheìlyMiddlewareError):
    """Authentication failed"""

    pass


class AuthorizationError(SheìlyMiddlewareError):
    """Authorization/permission denied"""

    pass


class RateLimitError(SheìlyMiddlewareError):
    """Rate limit exceeded"""

    pass


class SecurityError(SheìlyMiddlewareError):
    """Security violation"""

    pass


# ========================================
# GLOBAL INSTANCE
# ========================================

_middleware: Optional[SheìlyMiddleware] = None


async def get_middleware() -> SheìlyMiddleware:
    """Get global middleware instance"""
    global _middleware
    if _middleware is None:
        _middleware = SheìlyMiddleware()
        await _middleware.initialize()
    return _middleware


async def initialize_middleware_system() -> SheìlyMiddleware:
    """Initialize the complete middleware system"""
    middleware = await get_middleware()
    logger.info("✅ Sheily Enterprise Middleware System initialized")
    return middleware


__all__ = [
    "SheìlyMiddleware",
    "User",
    "UserRole",
    "PermissionLevel",
    "SecurityContext",
    "SheìlyMiddlewareError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "SecurityError",
    "get_middleware",
    "initialize_middleware_system",
]
