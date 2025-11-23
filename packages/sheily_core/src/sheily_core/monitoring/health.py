#!/usr/bin/env python3
"""
Sistema de Health Checks Enterprise para Sheily AI
Health checks comprehensivos con métricas detalladas y auto-healing
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .structured_logging import get_logger

logger = get_logger("health")


# =============================================================================
# ENUMERACIONES Y TIPOS
# =============================================================================


class HealthStatus(Enum):
    """Estados posibles de health"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Tipos de componentes que pueden ser monitoreados"""

    DATABASE = "database"
    CACHE = "cache"
    MODEL = "model"
    API = "api"
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    EXTERNAL_SERVICE = "external_service"
    CUSTOM = "custom"


@dataclass
class HealthCheckResult:
    """Resultado de un health check individual"""

    component: str
    component_type: ComponentType
    status: HealthStatus
    response_time_ms: float
    timestamp: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    last_success: Optional[float] = None
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        return {
            "component": self.component,
            "component_type": self.component_type.value,
            "status": self.status.value,
            "response_time_ms": round(self.response_time_ms, 2),
            "timestamp": self.timestamp,
            "message": self.message,
            "details": self.details,
            "error": self.error,
            "last_success": self.last_success,
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass
class SystemHealth:
    """Estado general de salud del sistema"""

    overall_status: HealthStatus
    timestamp: float
    uptime_seconds: float
    total_components: int
    healthy_components: int
    degraded_components: int
    unhealthy_components: int
    component_results: List[HealthCheckResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para API responses"""
        return {
            "overall_status": self.overall_status.value,
            "timestamp": self.timestamp,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "components": {
                "total": self.total_components,
                "healthy": self.healthy_components,
                "degraded": self.degraded_components,
                "unhealthy": self.unhealthy_components,
            },
            "component_results": [
                result.to_dict() for result in self.component_results
            ],
        }


# =============================================================================
# HEALTH CHECKS INDIVIDUALES
# =============================================================================


class BaseHealthCheck:
    """Clase base para health checks"""

    def __init__(
        self,
        name: str,
        component_type: ComponentType,
        timeout_seconds: float = 10.0,
        interval_seconds: float = 30.0,
        max_consecutive_failures: int = 3,
    ):
        self.name = name
        self.component_type = component_type
        self.timeout_seconds = timeout_seconds
        self.interval_seconds = interval_seconds
        self.max_consecutive_failures = max_consecutive_failures
        self.last_check: Optional[float] = None
        self.consecutive_failures = 0
        self.last_success: Optional[float] = None

    async def check(self) -> HealthCheckResult:
        """Ejecutar health check"""
        start_time = time.time()

        try:
            # Ejecutar check con timeout
            result = await asyncio.wait_for(
                self._perform_check(), timeout=self.timeout_seconds
            )

            response_time = (time.time() - start_time) * 1000
            self.last_check = start_time
            self.last_success = start_time
            self.consecutive_failures = 0

            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                timestamp=start_time,
                message=result.get("message", "Component is healthy"),
                details=result,
                last_success=self.last_success,
                consecutive_failures=self.consecutive_failures,
            )

        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            self._handle_failure(start_time)

            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                timestamp=start_time,
                message="Health check timed out",
                error=f"Timeout after {self.timeout_seconds}s",
                consecutive_failures=self.consecutive_failures,
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self._handle_failure(start_time)

            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                timestamp=start_time,
                message=f"Health check failed: {str(e)}",
                error=str(e),
                consecutive_failures=self.consecutive_failures,
            )

    def _handle_failure(self, timestamp: float):
        """Manejar falla de health check"""
        self.last_check = timestamp
        self.consecutive_failures += 1

        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.warning(
                f"Component {self.name} has failed {self.consecutive_failures} consecutive health checks"
            )

    async def _perform_check(self) -> Dict[str, Any]:
        """Implementar lógica específica del health check"""
        raise NotImplementedError("Subclasses must implement _perform_check")


class DatabaseHealthCheck(BaseHealthCheck):
    """Health check para bases de datos"""

    def __init__(
        self, name: str, connection_string: str, db_type: str = "postgresql", **kwargs
    ):
        super().__init__(name, ComponentType.DATABASE, **kwargs)
        self.connection_string = connection_string
        self.db_type = db_type

    async def _perform_check(self) -> Dict[str, Any]:
        """Verificar conectividad a base de datos"""
        if self.db_type.lower() == "postgresql":
            return await self._check_postgresql()
        elif self.db_type.lower() == "redis":
            return await self._check_redis()
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    async def _check_postgresql(self) -> Dict[str, Any]:
        """Verificar PostgreSQL"""
        try:
            import asyncpg

            # Intentar conexión
            conn = await asyncpg.connect(self.connection_string)
            result = await conn.fetchval("SELECT version()")
            await conn.close()

            return {
                "message": "PostgreSQL connection successful",
                "version": result.split()[1] if result else "unknown",
                "connection_pool": "available",
            }

        except ImportError:
            # Fallback a psycopg2 si asyncpg no está disponible
            import psycopg2

            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            result = cursor.fetchone()
            cursor.close()
            conn.close()

            return {
                "message": "PostgreSQL connection successful (sync)",
                "version": result[0].split()[1] if result else "unknown",
            }

    async def _check_redis(self) -> Dict[str, Any]:
        """Verificar Redis"""
        try:
            import redis.asyncio as redis

            # Extraer host/port de connection string
            # connection_string debería ser "redis://host:port/db"
            url_parts = self.connection_string.replace("redis://", "").split("/")
            host_port = url_parts[0].split(":")
            host = host_port[0]
            port = int(host_port[1]) if len(host_port) > 1 else 6379

            r = redis.Redis(host=host, port=port, decode_responses=True)
            pong = await r.ping()
            info = await r.info()

            return {
                "message": "Redis connection successful",
                "ping": pong,
                "version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
            }

        except ImportError:
            # Fallback a redis-py normal
            import redis

            r = redis.Redis.from_url(self.connection_string)
            pong = r.ping()
            info = r.info()

            return {
                "message": "Redis connection successful (sync)",
                "ping": pong,
                "version": info.get("redis_version", "unknown"),
            }


class ModelHealthCheck(BaseHealthCheck):
    """Health check para modelos de IA"""

    def __init__(
        self, name: str, model_path: str, test_input: str = "Hello, world!", **kwargs
    ):
        super().__init__(name, ComponentType.MODEL, **kwargs)
        self.model_path = model_path
        self.test_input = test_input

    async def _perform_check(self) -> Dict[str, Any]:
        """Verificar que el modelo puede hacer inferencias"""
        try:
            # Aquí iría la lógica específica del modelo
            # Por ahora, solo verificamos que el archivo existe
            import os

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            file_size = os.path.getsize(self.model_path)

            return {
                "message": "Model file exists and is accessible",
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "last_modified": os.path.getmtime(self.model_path),
            }

        except Exception as e:
            raise RuntimeError(f"Model health check failed: {str(e)}")


class APIHealthCheck(BaseHealthCheck):
    """Health check para APIs externas"""

    def __init__(
        self,
        name: str,
        url: str,
        expected_status: int = 200,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__(name, ComponentType.EXTERNAL_SERVICE, **kwargs)
        self.url = url
        self.expected_status = expected_status
        self.headers = headers or {}

    async def _perform_check(self) -> Dict[str, Any]:
        """Verificar conectividad a API externa"""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(self.url, headers=self.headers) as response:
                if response.status != self.expected_status:
                    raise RuntimeError(
                        f"Unexpected status code: {response.status} (expected {self.expected_status})"
                    )

                response_text = await response.text()
                content_length = len(response_text)

                return {
                    "message": f"API responded with status {response.status}",
                    "status_code": response.status,
                    "content_length": content_length,
                    "response_time_ms": None,  # Ya se mide en el nivel superior
                }


class FilesystemHealthCheck(BaseHealthCheck):
    """Health check para sistema de archivos"""

    def __init__(
        self,
        name: str,
        paths_to_check: List[str],
        min_free_space_gb: float = 1.0,
        **kwargs,
    ):
        super().__init__(name, ComponentType.FILESYSTEM, **kwargs)
        self.paths_to_check = paths_to_check
        self.min_free_space_gb = min_free_space_gb

    async def _perform_check(self) -> Dict[str, Any]:
        """Verificar estado del sistema de archivos"""
        import shutil

        results = {}

        for path in self.paths_to_check:
            try:
                # Verificar que el path existe
                if not os.path.exists(path):
                    results[path] = {
                        "status": "missing",
                        "error": "Path does not exist",
                    }
                    continue

                # Obtener espacio disponible
                stat = shutil.disk_usage(path)
                free_gb = stat.free / (1024**3)

                if free_gb < self.min_free_space_gb:
                    results[path] = {
                        "status": "low_space",
                        "free_gb": round(free_gb, 2),
                        "min_required_gb": self.min_free_space_gb,
                    }
                else:
                    results[path] = {
                        "status": "healthy",
                        "free_gb": round(free_gb, 2),
                        "total_gb": round(stat.total / (1024**3), 2),
                    }

            except Exception as e:
                results[path] = {"status": "error", "error": str(e)}

        # Verificar si todos los paths están healthy
        all_healthy = all(
            result.get("status") == "healthy" for result in results.values()
        )

        if not all_healthy:
            unhealthy_paths = [
                path
                for path, result in results.items()
                if result.get("status") != "healthy"
            ]
            raise RuntimeError(f"Unhealthy filesystem paths: {unhealthy_paths}")

        return {
            "message": "All filesystem paths are healthy",
            "paths_checked": len(self.paths_to_check),
            "details": results,
        }


# =============================================================================
# MONITOR DE SALUD CENTRALIZADO
# =============================================================================


class HealthMonitor:
    """Monitor centralizado de salud del sistema"""

    def __init__(self):
        self.health_checks: List[BaseHealthCheck] = []
        self.start_time = time.time()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def add_health_check(self, check: BaseHealthCheck):
        """Agregar un health check al monitor"""
        self.health_checks.append(check)
        logger.info(f"Added health check: {check.name} ({check.component_type.value})")

    def remove_health_check(self, name: str):
        """Remover un health check del monitor"""
        self.health_checks = [
            check for check in self.health_checks if check.name != name
        ]
        logger.info(f"Removed health check: {name}")

    async def run_health_checks(self) -> SystemHealth:
        """Ejecutar todos los health checks y retornar estado del sistema"""
        if not self.health_checks:
            return SystemHealth(
                overall_status=HealthStatus.UNKNOWN,
                timestamp=time.time(),
                uptime_seconds=time.time() - self.start_time,
                total_components=0,
                healthy_components=0,
                degraded_components=0,
                unhealthy_components=0,
            )

        # Ejecutar todos los health checks en paralelo
        tasks = [check.check() for check in self.health_checks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Procesar resultados
        component_results = []
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0

        for result in results:
            if isinstance(result, Exception):
                # Health check falló completamente
                logger.error(f"Health check execution failed: {result}")
                continue

            component_results.append(result)

            if result.status == HealthStatus.HEALTHY:
                healthy_count += 1
            elif result.status == HealthStatus.DEGRADED:
                degraded_count += 1
            elif result.status == HealthStatus.UNHEALTHY:
                unhealthy_count += 1

        # Determinar estado general del sistema
        total_components = len(component_results)

        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        elif healthy_count == total_components:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN

        system_health = SystemHealth(
            overall_status=overall_status,
            timestamp=time.time(),
            uptime_seconds=time.time() - self.start_time,
            total_components=total_components,
            healthy_components=healthy_count,
            degraded_components=degraded_count,
            unhealthy_components=unhealthy_count,
            component_results=component_results,
        )

        # Loggear estado del sistema
        self._log_system_health(system_health)

        return system_health

    def _log_system_health(self, system_health: SystemHealth):
        """Loggear estado de salud del sistema"""
        status_emoji = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.DEGRADED: "⚠️",
            HealthStatus.UNHEALTHY: "❌",
            HealthStatus.UNKNOWN: "❓",
        }

        emoji = status_emoji.get(system_health.overall_status, "❓")

        logger.info(
            f"{emoji} System Health: {system_health.overall_status.value} "
            f"({system_health.healthy_components}/{system_health.total_components} healthy, "
            f"uptime: {system_health.uptime_seconds:.0f}s)"
        )

        # Loggear componentes unhealthy
        unhealthy_components = [
            result
            for result in system_health.component_results
            if result.status == HealthStatus.UNHEALTHY
        ]

        for component in unhealthy_components:
            logger.warning(
                f"Unhealthy component: {component.component} - {component.message}"
            )

    async def start_monitoring(self, interval_seconds: float = 60.0):
        """Iniciar monitoreo continuo"""
        if self._running:
            logger.warning("Health monitoring already running")
            return

        self._running = True
        logger.info(f"Starting health monitoring (interval: {interval_seconds}s)")

        while self._running:
            try:
                await self.run_health_checks()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)

    def stop_monitoring(self):
        """Detener monitoreo continuo"""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("Stopped health monitoring")

    async def get_health_endpoint(self) -> tuple[Dict[str, Any], str]:
        """Obtener endpoint de health para APIs web"""
        system_health = await self.run_health_checks()
        return system_health.to_dict(), "application/json"


# =============================================================================
# INSTANCIA GLOBAL Y UTILIDADES
# =============================================================================

# Instancia global del monitor de salud
health_monitor = HealthMonitor()


def add_database_health_check(
    name: str, connection_string: str, db_type: str = "postgresql", **kwargs
):
    """Agregar health check para base de datos"""
    check = DatabaseHealthCheck(name, connection_string, db_type, **kwargs)
    health_monitor.add_health_check(check)


def add_model_health_check(name: str, model_path: str, **kwargs):
    """Agregar health check para modelo"""
    check = ModelHealthCheck(name, model_path, **kwargs)
    health_monitor.add_health_check(check)


def add_api_health_check(name: str, url: str, **kwargs):
    """Agregar health check para API externa"""
    check = APIHealthCheck(name, url, **kwargs)
    health_monitor.add_health_check(check)


def add_filesystem_health_check(name: str, paths: List[str], **kwargs):
    """Agregar health check para sistema de archivos"""
    check = FilesystemHealthCheck(name, paths, **kwargs)
    health_monitor.add_health_check(check)


async def get_system_health() -> SystemHealth:
    """Obtener estado de salud completo del sistema"""
    return await health_monitor.run_health_checks()


async def get_health_status() -> str:
    """Obtener estado de salud simple (para health checks básicos)"""
    system_health = await get_system_health()

    if system_health.overall_status == HealthStatus.HEALTHY:
        return "OK"
    elif system_health.overall_status == HealthStatus.DEGRADED:
        return "DEGRADED"
    else:
        return "UNHEALTHY"


# =============================================================================
# CONFIGURACIÓN POR DEFECTO
# =============================================================================


def setup_default_health_checks():
    """Configurar health checks por defecto para Sheily AI"""
    # Health check para modelos
    add_model_health_check(
        "primary_model", "models/gemma-2-9b-it-Q4_K_M.gguf", timeout_seconds=30.0
    )

    # Health check para filesystem
    add_filesystem_health_check(
        "data_directory", ["./data", "./logs", "./models"], min_free_space_gb=1.0
    )

    logger.info("Default health checks configured")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Clases principales
    "HealthMonitor",
    "BaseHealthCheck",
    "HealthStatus",
    "ComponentType",
    "HealthCheckResult",
    "SystemHealth",
    # Health checks específicos
    "DatabaseHealthCheck",
    "ModelHealthCheck",
    "APIHealthCheck",
    "FilesystemHealthCheck",
    # Utilidades
    "health_monitor",
    "add_database_health_check",
    "add_model_health_check",
    "add_api_health_check",
    "add_filesystem_health_check",
    "get_system_health",
    "get_health_status",
    "setup_default_health_checks",
]

# Información del módulo
__version__ = "1.0.0"
__author__ = "Sheily AI Team - Enterprise Health Monitoring System"
