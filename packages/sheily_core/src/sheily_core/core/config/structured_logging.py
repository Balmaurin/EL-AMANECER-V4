#!/usr/bin/env python3
"""
Sistema de Logging Estructurado para Sheily AI
Logging enterprise-grade con JSON, correlación y niveles múltiples
"""

import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger

# =============================================================================
# CONFIGURACIÓN BASE
# =============================================================================


class StructuredFormatter(jsonlogger.JsonFormatter):
    """Formateador JSON personalizado para logging estructurado"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()

    def format(self, record: logging.LogRecord) -> str:
        """Formatear log record como JSON estructurado"""

        # Campos base
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
            "process_name": record.processName,
        }

        # Agregar campos extra si existen
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "session_id"):
            log_entry["session_id"] = record.session_id
        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = record.correlation_id
        if hasattr(record, "component"):
            log_entry["component"] = record.component
        if hasattr(record, "operation"):
            log_entry["operation"] = record.operation
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms
        if hasattr(record, "status_code"):
            log_entry["status_code"] = record.status_code
        if hasattr(record, "user_agent"):
            log_entry["user_agent"] = record.user_agent
        if hasattr(record, "ip_address"):
            log_entry["ip_address"] = record.ip_address
        if hasattr(record, "endpoint"):
            log_entry["endpoint"] = record.endpoint
        if hasattr(record, "method"):
            log_entry["method"] = record.method

        # Agregar exception info si existe
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Agregar stack trace si existe
        if hasattr(record, "stack_trace"):
            log_entry["stack_trace"] = record.stack_trace

        # Agregar extra fields del record
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Formateador para consola con colores"""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Formatear con colores para consola"""
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])

        # Formato básico con color
        formatted = f"{color}[{record.levelname}] {record.name}: {record.getMessage()}{self.COLORS['RESET']}"

        # Agregar campos importantes
        extra_info = []
        if hasattr(record, "request_id"):
            extra_info.append(f"req={record.request_id}")
        if hasattr(record, "user_id"):
            extra_info.append(f"user={record.user_id}")
        if hasattr(record, "correlation_id"):
            extra_info.append(f"corr={record.correlation_id}")
        if hasattr(record, "component"):
            extra_info.append(f"comp={record.component}")

        if extra_info:
            formatted += f" ({', '.join(extra_info)})"

        # Agregar exception si existe
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


# =============================================================================
# GESTOR DE LOGGING CENTRALIZADO
# =============================================================================


class StructuredLogger:
    """Gestor centralizado de logging estructurado"""

    def __init__(self, name: str = "sheily_ai"):
        self.name = name
        self.logger = logging.getLogger(name)
        self._configured = False
        self._lock = threading.Lock()

    def configure(
        self,
        level: str = "INFO",
        log_file: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = True,
    ):
        """Configurar logging con múltiples handlers"""

        with self._lock:
            if self._configured:
                return

            # Limpiar handlers existentes
            self.logger.handlers.clear()

            # Convertir nivel de string a int
            numeric_level = getattr(logging, level.upper(), logging.INFO)
            self.logger.setLevel(numeric_level)

            # Formatter basado en configuración
            if enable_json:
                json_formatter = StructuredFormatter()
                console_formatter = (
                    json_formatter  # Usar JSON también en consola para consistencia
                )
            else:
                console_formatter = ConsoleFormatter()

            # Handler de consola
            if enable_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(numeric_level)
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)

            # Handler de archivo
            if enable_file and log_file:
                from logging.handlers import RotatingFileHandler

                # Crear directorio si no existe
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)

                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_file_size,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                file_handler.setLevel(numeric_level)

                # Archivo siempre en JSON para parsing
                json_formatter = StructuredFormatter()
                file_handler.setFormatter(json_formatter)

                self.logger.addHandler(file_handler)

            # Evitar propagación a root logger
            self.logger.propagate = False

            self._configured = True

    def get_logger(self, component: str = "") -> logging.LoggerAdapter:
        """Obtener logger adaptado para un componente específico"""
        if component:
            return logging.LoggerAdapter(self.logger, {"component": component})
        return logging.LoggerAdapter(self.logger, {})

    def set_correlation_id(self, correlation_id: str):
        """Establecer ID de correlación para el contexto actual"""
        self._correlation_id = correlation_id

    def get_correlation_id(self) -> Optional[str]:
        """Obtener ID de correlación actual"""
        return getattr(self, "_correlation_id", None)


# Instancia global del logger estructurado
structured_logger = StructuredLogger()


# =============================================================================
# FUNCIONES DE UTILIDAD PARA LOGGING
# =============================================================================


def get_logger(component: str = "") -> logging.LoggerAdapter:
    """Obtener logger para un componente específico"""
    return structured_logger.get_logger(component)


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = True,
):
    """Configurar logging global"""
    structured_logger.configure(
        level=level,
        log_file=log_file,
        enable_console=enable_console,
        enable_file=enable_file,
        enable_json=enable_json,
    )


def log_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration_ms: float,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
):
    """Loggear información de request HTTP"""
    logger = get_logger("http")

    extra = {
        "method": method,
        "endpoint": endpoint,
        "status_code": status_code,
        "duration_ms": duration_ms,
        "operation": "http_request",
    }

    if user_id:
        extra["user_id"] = user_id
    if request_id:
        extra["request_id"] = request_id
    if ip_address:
        extra["ip_address"] = ip_address
    if user_agent:
        extra["user_agent"] = user_agent

    if status_code >= 400:
        logger.error(f"HTTP {method} {endpoint} -> {status_code}", extra=extra)
    elif status_code >= 300:
        logger.warning(f"HTTP {method} {endpoint} -> {status_code}", extra=extra)
    else:
        logger.info(f"HTTP {method} {endpoint} -> {status_code}", extra=extra)


def log_model_inference(
    model_name: str,
    operation: str,
    duration_ms: float,
    tokens_processed: Optional[int] = None,
    success: bool = True,
):
    """Loggear inferencia de modelo"""
    logger = get_logger("ai")

    extra = {
        "model_name": model_name,
        "operation": operation,
        "duration_ms": duration_ms,
        "component": "model_inference",
    }

    if tokens_processed:
        extra["tokens_processed"] = tokens_processed

    if success:
        logger.info(f"Model inference: {model_name} ({operation})", extra=extra)
    else:
        logger.error(f"Model inference failed: {model_name} ({operation})", extra=extra)


def log_database_operation(
    operation: str,
    table: str,
    duration_ms: float,
    success: bool = True,
    error_message: Optional[str] = None,
):
    """Loggear operación de base de datos"""
    logger = get_logger("database")

    extra = {
        "operation": operation,
        "table": table,
        "duration_ms": duration_ms,
        "component": "database",
    }

    if success:
        logger.info(f"DB {operation} on {table}", extra=extra)
    else:
        extra["error_message"] = error_message
        logger.error(f"DB {operation} failed on {table}: {error_message}", extra=extra)


def log_security_event(
    event_type: str,
    severity: str,
    message: str,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None,
):
    """Loggear evento de seguridad"""
    logger = get_logger("security")

    extra = {"event_type": event_type, "severity": severity, "component": "security"}

    if user_id:
        extra["user_id"] = user_id
    if ip_address:
        extra["ip_address"] = ip_address
    if extra_data:
        extra.update(extra_data)

    if severity.lower() in ["high", "critical"]:
        logger.error(f"Security event: {event_type} - {message}", extra=extra)
    elif severity.lower() == "medium":
        logger.warning(f"Security event: {event_type} - {message}", extra=extra)
    else:
        logger.info(f"Security event: {event_type} - {message}", extra=extra)


def log_performance_metric(
    metric_name: str, value: float, unit: str = "ms", component: str = "performance"
):
    """Loggear métrica de performance"""
    logger = get_logger(component)

    extra = {
        "metric_name": metric_name,
        "value": value,
        "unit": unit,
        "component": component,
        "operation": "performance_metric",
    }

    logger.info(f"Performance: {metric_name} = {value}{unit}", extra=extra)


# =============================================================================
# CONTEXT MANAGERS PARA LOGGING AUTOMÁTICO
# =============================================================================


class RequestLogger:
    """Context manager para logging automático de requests"""

    def __init__(
        self,
        method: str,
        endpoint: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        self.method = method
        self.endpoint = endpoint
        self.user_id = user_id
        self.request_id = request_id
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.start_time = None
        self.status_code = 200

    def __enter__(self):
        self.start_time = time.time()
        logger = get_logger("http")

        extra = {
            "method": self.method,
            "endpoint": self.endpoint,
            "operation": "request_start",
        }
        if self.request_id:
            extra["request_id"] = self.request_id
        if self.user_id:
            extra["user_id"] = self.user_id

        logger.info(f"Started {self.method} {self.endpoint}", extra=extra)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000

            if exc_type:
                self.status_code = 500

            log_request(
                method=self.method,
                endpoint=self.endpoint,
                status_code=self.status_code,
                duration_ms=duration_ms,
                user_id=self.user_id,
                request_id=self.request_id,
                ip_address=self.ip_address,
                user_agent=self.user_agent,
            )


class DatabaseOperationLogger:
    """Context manager para logging automático de operaciones de BD"""

    def __init__(self, operation: str, table: str):
        self.operation = operation
        self.table = table
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            success = exc_type is None
            error_message = str(exc_val) if exc_val else None

            log_database_operation(
                operation=self.operation,
                table=self.table,
                duration_ms=duration_ms,
                success=success,
                error_message=error_message,
            )


# =============================================================================
# CONFIGURACIÓN POR DEFECTO
# =============================================================================


def setup_default_logging(
    level: str = "INFO", log_file: str = "logs/sheily_ai.log", enable_json: bool = True
):
    """Configurar logging con configuración por defecto"""
    configure_logging(
        level=level,
        log_file=log_file,
        enable_console=True,
        enable_file=True,
        enable_json=enable_json,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Clases principales
    "StructuredLogger",
    "RequestLogger",
    "DatabaseOperationLogger",
    # Funciones principales
    "get_logger",
    "configure_logging",
    "setup_default_logging",
    # Funciones especializadas
    "log_request",
    "log_model_inference",
    "log_database_operation",
    "log_security_event",
    "log_performance_metric",
    # Instancia global
    "structured_logger",
]

# Información del módulo
__version__ = "1.0.0"
__author__ = "Sheily AI Team - Enterprise Logging System"
