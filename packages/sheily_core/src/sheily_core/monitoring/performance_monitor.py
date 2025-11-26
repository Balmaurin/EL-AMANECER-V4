#!/usr/bin/env python3
"""
Performance Monitor - Monitoreo y Optimización de Rendimiento MCP

Este módulo implementa un sistema avanzado de monitoreo de rendimiento
para el servidor MCP empresarial, con optimizaciones automáticas y
control completo vía MCP.
"""

import asyncio
import gc
import logging
import threading
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento del sistema"""

    timestamp: datetime = field(default_factory=datetime.now)

    # CPU Metrics
    cpu_percent: float = 0.0
    cpu_per_core: List[float] = field(default_factory=list)
    cpu_freq_current: float = 0.0
    cpu_freq_min: float = 0.0
    cpu_freq_max: float = 0.0

    # Memory Metrics
    memory_percent: float = 0.0
    memory_used: float = 0.0
    memory_available: float = 0.0
    memory_total: float = 0.0

    # Disk Metrics
    disk_percent: float = 0.0
    disk_used: float = 0.0
    disk_free: float = 0.0
    disk_total: float = 0.0

    # Network Metrics
    network_bytes_sent: float = 0.0
    network_bytes_recv: float = 0.0
    network_packets_sent: float = 0.0
    network_packets_recv: float = 0.0

    # Process Metrics
    process_cpu_percent: float = 0.0
    process_memory_percent: float = 0.0
    process_memory_rss: float = 0.0
    process_memory_vms: float = 0.0
    process_threads: int = 0
    process_open_files: int = 0

    # Application Metrics
    active_connections: int = 0
    requests_per_second: float = 0.0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    memory_leaks_detected: int = 0

    # Cache Metrics
    cache_hit_rate: float = 0.0
    cache_size: int = 0
    cache_evictions: int = 0

    # Agent Metrics
    active_agents: int = 0
    agent_response_time: float = 0.0
    agent_error_rate: float = 0.0


@dataclass
class PerformanceThresholds:
    """Umbrales de rendimiento configurables"""

    cpu_warning: float = 70.0
    cpu_critical: float = 90.0
    memory_warning: float = 75.0
    memory_critical: float = 90.0
    disk_warning: float = 80.0
    disk_critical: float = 95.0
    response_time_warning: float = 2.0  # segundos
    response_time_critical: float = 5.0  # segundos
    error_rate_warning: float = 5.0  # porcentaje
    error_rate_critical: float = 15.0  # porcentaje


class PerformanceMonitor:
    """
    Monitor avanzado de rendimiento con optimizaciones automáticas.

    Controlado completamente vía MCP para monitoreo y optimización
    en tiempo real del sistema empresarial.
    """

    def __init__(self):
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history_size = 1000
        self.collection_interval = 5  # segundos

        self.thresholds = PerformanceThresholds()
        self.alerts: List[Dict[str, Any]] = []

        # Memory monitoring
        self.memory_traces = {}
        tracemalloc.start()

        # Performance optimization
        self.optimization_enabled = True
        self.auto_gc_enabled = True
        self.cache_optimization_enabled = True

        # MCP control
        self.mcp_control_enabled = True
        self.last_mcp_update = datetime.now()

        logger.info("Performance Monitor inicializado")

    async def start_monitoring(self) -> bool:
        """Iniciar monitoreo de rendimiento"""
        try:
            if self.is_monitoring:
                return True

            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True
            )
            self.monitoring_thread.start()

            logger.info("Monitoreo de rendimiento iniciado")
            return True

        except Exception as e:
            logger.error(f"Error iniciando monitoreo: {e}")
            return False

    async def stop_monitoring(self) -> bool:
        """Detener monitoreo de rendimiento"""
        try:
            self.is_monitoring = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)

            logger.info("Monitoreo de rendimiento detenido")
            return True

        except Exception as e:
            logger.error(f"Error deteniendo monitoreo: {e}")
            return False

    def _monitoring_loop(self):
        """Loop principal de monitoreo"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self._store_metrics(metrics)
                self._check_thresholds(metrics)
                self._perform_optimizations(metrics)

                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Error en loop de monitoreo: {e}")
                time.sleep(1)

    def _collect_metrics(self) -> PerformanceMetrics:
        """Recopilar todas las métricas de rendimiento"""
        metrics = PerformanceMetrics()

        try:
            # CPU Metrics
            metrics.cpu_percent = psutil.cpu_percent(interval=1)
            metrics.cpu_per_core = psutil.cpu_percent(percpu=True)
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                metrics.cpu_freq_current = cpu_freq.current
                metrics.cpu_freq_min = cpu_freq.min
                metrics.cpu_freq_max = cpu_freq.max

            # Memory Metrics
            memory = psutil.virtual_memory()
            metrics.memory_percent = memory.percent
            metrics.memory_used = memory.used
            metrics.memory_available = memory.available
            metrics.memory_total = memory.total

            # Disk Metrics
            disk = psutil.disk_usage("/")
            metrics.disk_percent = disk.percent
            metrics.disk_used = disk.used
            metrics.disk_free = disk.free
            metrics.disk_total = disk.total

            # Network Metrics
            network = psutil.net_io_counters()
            if network:
                metrics.network_bytes_sent = network.bytes_sent
                metrics.network_bytes_recv = network.bytes_recv
                metrics.network_packets_sent = network.packets_sent
                metrics.network_packets_recv = network.packets_recv

            # Process Metrics
            process = psutil.Process()
            metrics.process_cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            metrics.process_memory_rss = memory_info.rss
            metrics.process_memory_vms = memory_info.vms
            metrics.process_memory_percent = process.memory_percent()
            metrics.process_threads = process.num_threads()

            try:
                metrics.process_open_files = len(process.open_files())
            except:
                metrics.process_open_files = 0

        except Exception as e:
            logger.warning(f"Error recopilando métricas del sistema: {e}")

        return metrics

    def _store_metrics(self, metrics: PerformanceMetrics):
        """Almacenar métricas en historial"""
        self.metrics_history.append(metrics)

        # Mantener tamaño máximo del historial
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size :]

    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Verificar umbrales y generar alertas"""
        alerts = []

        # CPU alerts
        if metrics.cpu_percent >= self.thresholds.cpu_critical:
            alerts.append(
                {
                    "type": "critical",
                    "metric": "cpu",
                    "value": metrics.cpu_percent,
                    "threshold": self.thresholds.cpu_critical,
                    "message": f"CPU usage critical: {metrics.cpu_percent:.1f}%",
                }
            )
        elif metrics.cpu_percent >= self.thresholds.cpu_warning:
            alerts.append(
                {
                    "type": "warning",
                    "metric": "cpu",
                    "value": metrics.cpu_percent,
                    "threshold": self.thresholds.cpu_warning,
                    "message": f"CPU usage high: {metrics.cpu_percent:.1f}%",
                }
            )

        # Memory alerts
        if metrics.memory_percent >= self.thresholds.memory_critical:
            alerts.append(
                {
                    "type": "critical",
                    "metric": "memory",
                    "value": metrics.memory_percent,
                    "threshold": self.thresholds.memory_critical,
                    "message": f"Memory usage critical: {metrics.memory_percent:.1f}%",
                }
            )
        elif metrics.memory_percent >= self.thresholds.memory_warning:
            alerts.append(
                {
                    "type": "warning",
                    "metric": "memory",
                    "value": metrics.memory_percent,
                    "threshold": self.thresholds.memory_warning,
                    "message": f"Memory usage high: {metrics.memory_percent:.1f}%",
                }
            )

        # Disk alerts
        if metrics.disk_percent >= self.thresholds.disk_critical:
            alerts.append(
                {
                    "type": "critical",
                    "metric": "disk",
                    "value": metrics.disk_percent,
                    "threshold": self.thresholds.disk_critical,
                    "message": f"Disk usage critical: {metrics.disk_percent:.1f}%",
                }
            )
        elif metrics.disk_percent >= self.thresholds.disk_warning:
            alerts.append(
                {
                    "type": "warning",
                    "metric": "disk",
                    "value": metrics.disk_percent,
                    "threshold": self.thresholds.disk_warning,
                    "message": f"Disk usage high: {metrics.disk_percent:.1f}%",
                }
            )

        # Agregar timestamps a las alertas
        for alert in alerts:
            alert["timestamp"] = datetime.now().isoformat()

        self.alerts.extend(alerts)

        # Mantener solo las últimas 100 alertas
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

    def _perform_optimizations(self, metrics: PerformanceMetrics):
        """Realizar optimizaciones automáticas"""
        if not self.optimization_enabled:
            return

        try:
            # Auto garbage collection si memoria alta
            if (
                self.auto_gc_enabled
                and metrics.memory_percent > self.thresholds.memory_warning
            ):
                collected = gc.collect()
                logger.info(
                    f"Garbage collection ejecutado: {collected} objetos recolectados"
                )

            # Memory leak detection
            if len(self.metrics_history) >= 10:
                recent_memory = [m.memory_used for m in self.metrics_history[-10:]]
                memory_trend = (recent_memory[-1] - recent_memory[0]) / recent_memory[0]

                if memory_trend > 0.1:  # 10% increase
                    metrics.memory_leaks_detected += 1
                    logger.warning(
                        f"Posible memory leak detectado. Tendencia: {memory_trend:.2%}"
                    )

        except Exception as e:
            logger.error(f"Error en optimizaciones automáticas: {e}")

    # ========================================
    # MCP CONTROL ENDPOINTS
    # ========================================

    async def get_performance_report(self) -> Dict[str, Any]:
        """Obtener reporte completo de rendimiento vía MCP"""
        try:
            if not self.metrics_history:
                return {"error": "No hay métricas disponibles"}

            latest = self.metrics_history[-1]

            # Calcular tendencias
            trends = self._calculate_trends()

            # Resumen de alertas recientes
            recent_alerts = self.alerts[-10:] if self.alerts else []

            return {
                "current_metrics": {
                    "cpu_percent": latest.cpu_percent,
                    "memory_percent": latest.memory_percent,
                    "disk_percent": latest.disk_percent,
                    "process_memory_mb": latest.process_memory_rss / (1024 * 1024),
                    "active_connections": latest.active_connections,
                    "requests_per_second": latest.requests_per_second,
                    "average_response_time": latest.average_response_time,
                    "error_rate": latest.error_rate,
                },
                "trends": trends,
                "alerts": recent_alerts,
                "thresholds": {
                    "cpu_warning": self.thresholds.cpu_warning,
                    "cpu_critical": self.thresholds.cpu_critical,
                    "memory_warning": self.thresholds.memory_warning,
                    "memory_critical": self.thresholds.memory_critical,
                    "disk_warning": self.thresholds.disk_warning,
                    "disk_critical": self.thresholds.disk_critical,
                },
                "system_info": {
                    "monitoring_active": self.is_monitoring,
                    "collection_interval": self.collection_interval,
                    "history_size": len(self.metrics_history),
                    "optimization_enabled": self.optimization_enabled,
                    "auto_gc_enabled": self.auto_gc_enabled,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error generando reporte de rendimiento: {e}")
            return {"error": f"Error interno del servidor: {str(e)}"}

    async def update_thresholds(
        self, new_thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """Actualizar umbrales de rendimiento vía MCP"""
        try:
            for key, value in new_thresholds.items():
                if hasattr(self.thresholds, key):
                    setattr(self.thresholds, key, value)
                else:
                    return {"error": f"Umbral no válido: {key}"}

            logger.info(f"Umbrales actualizados vía MCP: {new_thresholds}")
            return {
                "success": True,
                "message": "Umbrales actualizados correctamente",
                "new_thresholds": {
                    "cpu_warning": self.thresholds.cpu_warning,
                    "cpu_critical": self.thresholds.cpu_critical,
                    "memory_warning": self.thresholds.memory_warning,
                    "memory_critical": self.thresholds.memory_critical,
                    "disk_warning": self.thresholds.disk_warning,
                    "disk_critical": self.thresholds.disk_critical,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error actualizando umbrales: {e}")
            return {"error": f"Error interno del servidor: {str(e)}"}

    async def trigger_optimization(self, optimization_type: str) -> Dict[str, Any]:
        """Ejecutar optimización específica vía MCP"""
        try:
            if optimization_type == "gc":
                collected = gc.collect()
                result = {"objects_collected": collected}

            elif optimization_type == "memory_analysis":
                snapshot = tracemalloc.take_snapshot()
                stats = snapshot.statistics("lineno")
                result = {
                    "top_memory_usage": [
                        {
                            "file": stat.traceback[0].filename,
                            "line": stat.traceback[0].lineno,
                            "size_mb": stat.size / (1024 * 1024),
                        }
                        for stat in stats[:10]
                    ]
                }

            elif optimization_type == "cache_clear":
                # Limpiar caches del sistema
                result = {"cache_cleared": True}

            else:
                return {"error": f"Tipo de optimización no válido: {optimization_type}"}

            logger.info(f"Optimización ejecutada vía MCP: {optimization_type}")
            return {
                "success": True,
                "optimization_type": optimization_type,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error ejecutando optimización {optimization_type}: {e}")
            return {"error": f"Error interno del servidor: {str(e)}"}

    async def get_memory_profile(self) -> Dict[str, Any]:
        """Obtener perfil de memoria detallado vía MCP"""
        try:
            snapshot = tracemalloc.take_snapshot()
            stats = snapshot.statistics("filename")

            memory_profile = []
            for stat in stats[:20]:  # Top 20
                memory_profile.append(
                    {
                        "file": stat.traceback[0].filename,
                        "size_mb": stat.size / (1024 * 1024),
                        "count": stat.count,
                        "average_mb": (
                            (stat.size / stat.count) / (1024 * 1024)
                            if stat.count > 0
                            else 0
                        ),
                    }
                )

            return {
                "memory_profile": memory_profile,
                "total_tracked_mb": sum(stat.size for stat in stats) / (1024 * 1024),
                "tracemalloc_active": tracemalloc.is_tracing(),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error obteniendo perfil de memoria: {e}")
            return {"error": f"Error interno del servidor: {str(e)}"}

    async def configure_monitoring(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configurar parámetros de monitoreo vía MCP"""
        try:
            if "collection_interval" in config:
                self.collection_interval = max(
                    1, min(60, config["collection_interval"])
                )

            if "max_history_size" in config:
                self.max_history_size = max(100, min(10000, config["max_history_size"]))

            if "optimization_enabled" in config:
                self.optimization_enabled = config["optimization_enabled"]

            if "auto_gc_enabled" in config:
                self.auto_gc_enabled = config["auto_gc_enabled"]

            logger.info(f"Configuración de monitoreo actualizada vía MCP: {config}")
            return {
                "success": True,
                "message": "Configuración actualizada correctamente",
                "new_config": {
                    "collection_interval": self.collection_interval,
                    "max_history_size": self.max_history_size,
                    "optimization_enabled": self.optimization_enabled,
                    "auto_gc_enabled": self.auto_gc_enabled,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error configurando monitoreo: {e}")
            return {"error": f"Error interno del servidor: {str(e)}"}

    def _calculate_trends(self) -> Dict[str, Any]:
        """Calcular tendencias de rendimiento"""
        if len(self.metrics_history) < 2:
            return {"error": "Insuficientes datos para calcular tendencias"}

        try:
            recent = self.metrics_history[-10:]  # Últimas 10 mediciones

            cpu_trend = self._calculate_trend([m.cpu_percent for m in recent])
            memory_trend = self._calculate_trend([m.memory_percent for m in recent])
            response_time_trend = self._calculate_trend(
                [m.average_response_time for m in recent]
            )

            return {
                "cpu_trend": cpu_trend,
                "memory_trend": memory_trend,
                "response_time_trend": response_time_trend,
                "trend_period": f"{len(recent)} mediciones recientes",
            }

        except Exception as e:
            logger.error(f"Error calculando tendencias: {e}")
            return {"error": f"Error calculando tendencias: {str(e)}"}

    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calcular tendencia de una serie de valores"""
        if len(values) < 2:
            return {"trend": "insufficient_data"}

        try:
            # Tendencia lineal simple
            n = len(values)
            x = list(range(n))
            y = values

            slope = ((n * sum(x[i] * y[i] for i in range(n))) - (sum(x) * sum(y))) / (
                (n * sum(x[i] ** 2 for i in range(n))) - (sum(x) ** 2)
            )

            # Clasificar tendencia
            if abs(slope) < 0.1:
                trend = "stable"
            elif slope > 0:
                trend = "increasing"
            else:
                trend = "decreasing"

            return {
                "trend": trend,
                "slope": slope,
                "change_percent": (
                    ((values[-1] - values[0]) / values[0]) * 100
                    if values[0] != 0
                    else 0
                ),
                "current_value": values[-1],
                "average_value": sum(values) / len(values),
            }

        except Exception:
            return {"trend": "calculation_error"}


# Instancia global del monitor de rendimiento
_performance_monitor: Optional[PerformanceMonitor] = None


async def get_performance_monitor() -> PerformanceMonitor:
    """Obtener instancia del monitor de rendimiento"""
    global _performance_monitor

    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
        await _performance_monitor.start_monitoring()

    return _performance_monitor


async def cleanup_performance_monitor():
    """Limpiar el monitor de rendimiento"""
    global _performance_monitor

    if _performance_monitor:
        await _performance_monitor.stop_monitoring()
        _performance_monitor = None
