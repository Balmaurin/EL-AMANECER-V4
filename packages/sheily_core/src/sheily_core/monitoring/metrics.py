#!/usr/bin/env python3
"""
Sistema de Métricas y Monitoreo para Sheily AI
Métricas Prometheus enterprise-grade con observabilidad completa
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)

# Configurar logging
logger = logging.getLogger(__name__)

# =============================================================================
# Registry y Configuración Base
# =============================================================================

# Registry personalizado para aislamiento
registry = CollectorRegistry()

# =============================================================================
# MÉTRICAS DE APLICACIÓN
# =============================================================================

# Contadores de requests
REQUEST_COUNT = Counter(
    "sheily_http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"],
    registry=registry,
)

# Latencia de requests
REQUEST_LATENCY = Histogram(
    "sheily_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    registry=registry,
)

# Tamaño de responses
RESPONSE_SIZE = Histogram(
    "sheily_http_response_size_bytes",
    "HTTP response size in bytes",
    ["method", "endpoint"],
    buckets=(100, 1000, 10000, 100000, 1000000),
    registry=registry,
)

# =============================================================================
# MÉTRICAS DE AI/ML
# =============================================================================

# Contador de inferencias de modelo
MODEL_INFERENCE_COUNT = Counter(
    "sheily_model_inference_total",
    "Total number of model inferences",
    ["model_name", "model_type"],
    registry=registry,
)

# Latencia de inferencias
MODEL_INFERENCE_LATENCY = Histogram(
    "sheily_model_inference_duration_seconds",
    "Model inference latency in seconds",
    ["model_name", "model_type"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    registry=registry,
)

# Tokens procesados
TOKENS_PROCESSED = Counter(
    "sheily_tokens_processed_total",
    "Total number of tokens processed",
    ["model_name", "operation"],
    registry=registry,
)

# =============================================================================
# MÉTRICAS DE RAG
# =============================================================================

# Búsquedas en vector DB
VECTOR_SEARCH_COUNT = Counter(
    "sheily_vector_search_total",
    "Total number of vector searches",
    ["search_type", "index_type"],
    registry=registry,
)

# Latencia de búsquedas vectoriales
VECTOR_SEARCH_LATENCY = Histogram(
    "sheily_vector_search_duration_seconds",
    "Vector search latency in seconds",
    ["search_type", "index_type"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0),
    registry=registry,
)

# Resultados de búsqueda
SEARCH_RESULTS_COUNT = Histogram(
    "sheily_search_results_count",
    "Number of search results returned",
    ["search_type"],
    buckets=(0, 1, 5, 10, 25, 50, 100),
    registry=registry,
)

# =============================================================================
# MÉTRICAS DE SISTEMA
# =============================================================================

# Uso de memoria
MEMORY_USAGE = Gauge(
    "sheily_memory_usage_bytes",
    "Current memory usage in bytes",
    ["type"],  # heap, rss, vms
    registry=registry,
)

# Uso de CPU
CPU_USAGE = Gauge(
    "sheily_cpu_usage_percent", "Current CPU usage percentage", registry=registry
)

# Conexiones activas
ACTIVE_CONNECTIONS = Gauge(
    "sheily_active_connections",
    "Number of active connections",
    ["type"],  # websocket, database, cache
    registry=registry,
)

# Tamaño de colas
QUEUE_SIZE = Gauge(
    "sheily_queue_size", "Current queue size", ["queue_name"], registry=registry
)

# =============================================================================
# MÉTRICAS DE BASE DE DATOS
# =============================================================================

# Conexiones a BD
DB_CONNECTIONS_ACTIVE = Gauge(
    "sheily_db_connections_active",
    "Number of active database connections",
    ["db_type"],  # postgres, redis
    registry=registry,
)

# Queries ejecutadas
DB_QUERIES_TOTAL = Counter(
    "sheily_db_queries_total",
    "Total number of database queries",
    ["db_type", "operation"],  # select, insert, update, delete
    registry=registry,
)

# Latencia de queries
DB_QUERY_LATENCY = Histogram(
    "sheily_db_query_duration_seconds",
    "Database query latency in seconds",
    ["db_type", "operation"],
    buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0),
    registry=registry,
)

# =============================================================================
# MÉTRICAS DE ERRORES
# =============================================================================

# Errores por tipo
ERROR_COUNT = Counter(
    "sheily_errors_total",
    "Total number of errors",
    ["error_type", "component", "severity"],
    registry=registry,
)

# Errores de validación
VALIDATION_ERRORS = Counter(
    "sheily_validation_errors_total",
    "Total number of validation errors",
    ["field", "validator"],
    registry=registry,
)

# =============================================================================
# MÉTRICAS DE NEGOCIO
# =============================================================================

# Usuarios activos
ACTIVE_USERS = Gauge(
    "sheily_active_users",
    "Number of active users",
    ["time_window"],  # 1m, 5m, 15m, 1h
    registry=registry,
)

# Conversaciones activas
ACTIVE_CONVERSATIONS = Gauge(
    "sheily_active_conversations", "Number of active conversations", registry=registry
)

# Mensajes procesados
MESSAGES_PROCESSED = Counter(
    "sheily_messages_processed_total",
    "Total number of messages processed",
    ["message_type", "channel"],  # chat, api, websocket
    registry=registry,
)

# =============================================================================
# CLASES Y UTILIDADES
# =============================================================================


class MetricsCollector:
    """Colector centralizado de métricas"""

    def __init__(self):
        self._start_time = time.time()
        self._request_count = 0

    def increment_request_count(self, method: str, endpoint: str, status_code: int):
        """Incrementar contador de requests"""
        REQUEST_COUNT.labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()

    def observe_request_latency(self, method: str, endpoint: str, duration: float):
        """Observar latencia de request"""
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

    def observe_response_size(self, method: str, endpoint: str, size: int):
        """Observar tamaño de response"""
        RESPONSE_SIZE.labels(method=method, endpoint=endpoint).observe(size)

    def increment_model_inference(self, model_name: str, model_type: str):
        """Incrementar contador de inferencias de modelo"""
        MODEL_INFERENCE_COUNT.labels(model_name=model_name, model_type=model_type).inc()

    def observe_model_latency(self, model_name: str, model_type: str, duration: float):
        """Observar latencia de modelo"""
        MODEL_INFERENCE_LATENCY.labels(
            model_name=model_name, model_type=model_type
        ).observe(duration)

    def increment_tokens_processed(self, model_name: str, operation: str, count: int):
        """Incrementar contador de tokens procesados"""
        TOKENS_PROCESSED.labels(model_name=model_name, operation=operation).inc(count)

    def increment_vector_search(self, search_type: str, index_type: str):
        """Incrementar contador de búsquedas vectoriales"""
        VECTOR_SEARCH_COUNT.labels(search_type=search_type, index_type=index_type).inc()

    def observe_vector_search_latency(
        self, search_type: str, index_type: str, duration: float
    ):
        """Observar latencia de búsqueda vectorial"""
        VECTOR_SEARCH_LATENCY.labels(
            search_type=search_type, index_type=index_type
        ).observe(duration)

    def observe_search_results_count(self, search_type: str, count: int):
        """Observar número de resultados de búsqueda"""
        SEARCH_RESULTS_COUNT.labels(search_type=search_type).observe(count)

    def set_memory_usage(self, memory_type: str, bytes_used: int):
        """Establecer uso de memoria"""
        MEMORY_USAGE.labels(type=memory_type).set(bytes_used)

    def set_cpu_usage(self, percentage: float):
        """Establecer uso de CPU"""
        CPU_USAGE.set(percentage)

    def set_active_connections(self, conn_type: str, count: int):
        """Establecer conexiones activas"""
        ACTIVE_CONNECTIONS.labels(type=conn_type).set(count)

    def set_queue_size(self, queue_name: str, size: int):
        """Establecer tamaño de cola"""
        QUEUE_SIZE.labels(queue_name=queue_name).set(size)

    def set_db_connections_active(self, db_type: str, count: int):
        """Establecer conexiones activas a BD"""
        DB_CONNECTIONS_ACTIVE.labels(db_type=db_type).set(count)

    def increment_db_queries(self, db_type: str, operation: str):
        """Incrementar contador de queries"""
        DB_QUERIES_TOTAL.labels(db_type=db_type, operation=operation).inc()

    def observe_db_query_latency(self, db_type: str, operation: str, duration: float):
        """Observar latencia de query"""
        DB_QUERY_LATENCY.labels(db_type=db_type, operation=operation).observe(duration)

    def increment_error(self, error_type: str, component: str, severity: str):
        """Incrementar contador de errores"""
        ERROR_COUNT.labels(
            error_type=error_type, component=component, severity=severity
        ).inc()

    def increment_validation_error(self, field: str, validator: str):
        """Incrementar contador de errores de validación"""
        VALIDATION_ERRORS.labels(field=field, validator=validator).inc()

    def set_active_users(self, time_window: str, count: int):
        """Establecer usuarios activos"""
        ACTIVE_USERS.labels(time_window=time_window).set(count)

    def set_active_conversations(self, count: int):
        """Establecer conversaciones activas"""
        ACTIVE_CONVERSATIONS.set(count)

    def increment_messages_processed(self, message_type: str, channel: str):
        """Incrementar contador de mensajes procesados"""
        MESSAGES_PROCESSED.labels(message_type=message_type, channel=channel).inc()

    def get_metrics(self) -> str:
        """Obtener métricas en formato Prometheus"""
        return generate_latest(registry).decode("utf-8")

    def get_uptime(self) -> float:
        """Obtener tiempo de actividad en segundos"""
        return time.time() - self._start_time


# Instancia global del collector
metrics_collector = MetricsCollector()


# =============================================================================
# DECORADORES PARA INSTRUMENTACIÓN AUTOMÁTICA
# =============================================================================


def track_request(method: str, endpoint: str):
    """Decorador para trackear requests HTTP"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                status_code = getattr(result, "status_code", 200)
                response_size = len(str(result)) if result else 0

                # Métricas de éxito
                metrics_collector.increment_request_count(method, endpoint, status_code)
                metrics_collector.observe_response_size(method, endpoint, response_size)

                return result

            except Exception as e:
                # Métricas de error
                metrics_collector.increment_request_count(method, endpoint, 500)
                metrics_collector.increment_error("exception", "http", "high")
                raise
            finally:
                # Latencia siempre
                duration = time.time() - start_time
                metrics_collector.observe_request_latency(method, endpoint, duration)

        return wrapper

    return decorator


def track_model_inference(model_name: str, model_type: str):
    """Decorador para trackear inferencias de modelo"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                # Métricas de inferencia
                metrics_collector.increment_model_inference(model_name, model_type)

                return result

            finally:
                # Latencia siempre
                duration = time.time() - start_time
                metrics_collector.observe_model_latency(
                    model_name, model_type, duration
                )

        return wrapper

    return decorator


@contextmanager
def track_db_operation(db_type: str, operation: str):
    """Context manager para trackear operaciones de BD"""
    start_time = time.time()

    try:
        yield
        metrics_collector.increment_db_queries(db_type, operation)
    finally:
        duration = time.time() - start_time
        metrics_collector.observe_db_query_latency(db_type, operation, duration)


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================


def get_metrics_endpoint() -> tuple[str, str]:
    """Obtener endpoint de métricas para integración con web frameworks"""
    return metrics_collector.get_metrics(), CONTENT_TYPE_LATEST


def update_system_metrics():
    """Actualizar métricas del sistema (llamar periódicamente)"""
    import psutil

    try:
        # Memoria
        memory = psutil.virtual_memory()
        metrics_collector.set_memory_usage("rss", memory.rss)
        metrics_collector.set_memory_usage("available", memory.available)
        metrics_collector.set_memory_usage("percent", memory.percent)

        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics_collector.set_cpu_usage(cpu_percent)

    except ImportError:
        logger.warning("psutil not available, skipping system metrics")
    except Exception as e:
        logger.error(f"Error updating system metrics: {e}")


def log_metrics_summary():
    """Loggear resumen de métricas (para debugging)"""
    try:
        uptime = metrics_collector.get_uptime()
        logger.info(f"📊 Metrics Summary - Uptime: {uptime:.2f}s")

        # Aquí podríamos agregar más resúmenes específicos
        # Por ahora solo uptime para evitar complejidad

    except Exception as e:
        logger.error(f"Error logging metrics summary: {e}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Clases principales
    "MetricsCollector",
    "metrics_collector",
    # Métricas individuales
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "RESPONSE_SIZE",
    "MODEL_INFERENCE_COUNT",
    "MODEL_INFERENCE_LATENCY",
    "TOKENS_PROCESSED",
    "VECTOR_SEARCH_COUNT",
    "VECTOR_SEARCH_LATENCY",
    "SEARCH_RESULTS_COUNT",
    "MEMORY_USAGE",
    "CPU_USAGE",
    "ACTIVE_CONNECTIONS",
    "QUEUE_SIZE",
    "DB_CONNECTIONS_ACTIVE",
    "DB_QUERIES_TOTAL",
    "DB_QUERY_LATENCY",
    "ERROR_COUNT",
    "VALIDATION_ERRORS",
    "ACTIVE_USERS",
    "ACTIVE_CONVERSATIONS",
    "MESSAGES_PROCESSED",
    # Decoradores
    "track_request",
    "track_model_inference",
    "track_db_operation",
    # Utilidades
    "get_metrics_endpoint",
    "update_system_metrics",
    "log_metrics_summary",
    "registry",
]

# Información del módulo
__version__ = "1.0.0"
__author__ = "Sheily AI Team - Enterprise Metrics System"
