#!/usr/bin/env python3
"""
Sheily Enterprise Real-Time Metrics & Analytics System
=====================================================

Sistema de métricas y analytics en tiempo real con integración
a los patterns de ii-agent para monitoring enterprise.

Características:
- Métricas de rendimiento en tiempo real
- Analytics de agents y operaciones
- Billing y usage tracking
- Performance monitoring
- Resource usage analytics
- Alerting system
"""

import asyncio
import json
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from sheily_core.core.events.event_system import (
    EventSubscriber,
    SheìlyEventType,
    get_event_stream,
    publish_event,
)

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Tipos de métricas disponibles"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertLevel(str, Enum):
    """Niveles de alerta"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricDataPoint:
    """Punto de datos de métrica"""

    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Serie temporal de métricas"""

    name: str
    metric_type: MetricType
    data_points: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_updated: Optional[datetime] = None
    total_count: int = 0
    tags: Dict[str, str] = field(default_factory=dict)

    def add_data_point(
        self,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add new data point"""
        point = MetricDataPoint(
            timestamp=datetime.now(timezone.utc),
            value=value,
            tags=tags or {},
            metadata=metadata or {},
        )
        self.data_points.append(point)
        self.last_updated = point.timestamp
        self.total_count += 1

    def get_recent_values(self, minutes: int = 5) -> List[float]:
        """Get values from recent minutes"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        return [dp.value for dp in self.data_points if dp.timestamp >= cutoff_time]

    def get_current_value(self) -> Optional[float]:
        """Get most recent value"""
        return self.data_points[-1].value if self.data_points else None

    def get_average(self, minutes: int = 5) -> Optional[float]:
        """Get average value for recent period"""
        values = self.get_recent_values(minutes)
        return statistics.mean(values) if values else None

    def get_rate_per_minute(self) -> float:
        """Get rate per minute for recent period"""
        recent_points = [
            dp
            for dp in self.data_points
            if datetime.now(timezone.utc) - dp.timestamp <= timedelta(minutes=1)
        ]
        return len(recent_points)


@dataclass
class Alert:
    """Alerta del sistema"""

    id: str
    name: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: float
    current_value: float
    triggered_at: datetime
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsSubscriber(EventSubscriber):
    """Subscriber especializado para métricas del event system"""

    def __init__(self, analytics_system: "SheìlyAnalyticsSystem"):
        super().__init__(subscription_id=f"metrics_subscriber_{int(time.time())}")
        self.analytics = analytics_system

    async def handle_event(self, event_type: SheìlyEventType, data: Dict[str, Any]):
        """Handle events and convert to metrics"""
        try:
            if event_type == SheìlyEventType.METRICS_UPDATE:
                await self._handle_metrics_update(data)
            elif event_type == SheìlyEventType.PERFORMANCE_ALERT:
                await self._handle_performance_alert(data)
            elif event_type == SheìlyEventType.AGENT_STATUS:
                await self._handle_agent_status(data)
            elif event_type == SheìlyEventType.SYSTEM_HEALTH:
                await self._handle_system_health(data)

        except Exception as e:
            logger.error(f"Error handling event in MetricsSubscriber: {e}")

    async def _handle_metrics_update(self, data: Dict[str, Any]):
        """Handle metric update events"""
        metric_type = data.get("metric_type", "unknown")

        if metric_type == "request_metrics":
            # Track request performance
            duration = data.get("duration_seconds", 0)
            success = data.get("success", True)
            handler = data.get("handler", "unknown")

            await self.analytics.record_counter(
                "sheily.requests.total",
                1,
                {"handler": handler, "success": str(success)},
            )

            await self.analytics.record_timer(
                "sheily.requests.duration", duration, {"handler": handler}
            )

            if not success:
                await self.analytics.record_counter(
                    "sheily.requests.errors",
                    1,
                    {"handler": handler, "error": data.get("error", "unknown")},
                )

        elif metric_type == "agent_operation":
            # Track agent operations
            agent_id = data.get("agent_id", "unknown")
            operation = data.get("operation", "unknown")
            success = data.get("success", True)

            await self.analytics.record_counter(
                "sheily.agents.operations",
                1,
                {"agent_id": agent_id, "operation": operation, "success": str(success)},
            )

    async def _handle_performance_alert(self, data: Dict[str, Any]):
        """Handle performance alerts"""
        alert_type = data.get("alert_type", "unknown")

        await self.analytics.record_counter(
            "sheily.alerts.total", 1, {"alert_type": alert_type, "severity": "high"}
        )

        # Create formal alert if critical
        if alert_type in [
            "rate_limit_exceeded",
            "suspicious_activity",
            "memory_threshold",
        ]:
            await self.analytics.create_alert(
                name=f"Performance Alert: {alert_type}",
                level=AlertLevel.CRITICAL,
                message=data.get("message", f"Alert triggered: {alert_type}"),
                metric_name="sheily.alerts.total",
                threshold=1.0,
                current_value=1.0,
                tags={"alert_type": alert_type},
            )

    async def _handle_agent_status(self, data: Dict[str, Any]):
        """Handle agent status updates"""
        agent_id = data.get("agent_id", "unknown")
        status = data.get("status", "unknown")

        await self.analytics.record_gauge(
            "sheily.agents.status",
            1.0 if status == "active" else 0.0,
            {"agent_id": agent_id, "status": status},
        )

    async def _handle_system_health(self, data: Dict[str, Any]):
        """Handle system health updates"""
        await self.analytics.record_gauge(
            "sheily.system.health", 1.0, {"status": data.get("status", "unknown")}
        )


class SheìlyAnalyticsSystem:
    """Sistema principal de analytics y métricas para Sheily Enterprise"""

    def __init__(self):
        self.metrics: Dict[str, MetricSeries] = {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.event_stream = None
        self.metrics_subscriber = None

        # Billing and usage tracking
        self.usage_tracking: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.billing_metrics: Dict[str, float] = defaultdict(float)

        # Performance monitoring
        self.performance_baselines: Dict[str, float] = {}
        self.resource_usage = {
            "cpu_percent": deque(maxlen=100),
            "memory_mb": deque(maxlen=100),
            "disk_io": deque(maxlen=100),
            "network_io": deque(maxlen=100),
        }

        # Lock for thread safety
        self._lock = threading.RLock()

        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False

    async def initialize(self) -> None:
        """Initialize analytics system"""
        try:
            self.event_stream = get_event_stream()

            # Create and register metrics subscriber
            self.metrics_subscriber = MetricsSubscriber(self)
            await self.event_stream.add_subscriber(self.metrics_subscriber)

            # Setup default alert rules
            await self._setup_default_alert_rules()

            # Start background monitoring
            self._running = True
            self._monitoring_task = asyncio.create_task(self._background_monitoring())

            await publish_event(
                SheìlyEventType.SYSTEM_HEALTH,
                {"status": "analytics_system_initialized"},
            )

            logger.info("✅ Sheily Analytics System initialized")

        except Exception as e:
            logger.error(f"Error initializing analytics system: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown analytics system"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

    # ========================================
    # METRICS RECORDING
    # ========================================

    async def record_counter(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record counter metric"""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = MetricSeries(
                    name=name, metric_type=MetricType.COUNTER
                )

            self.metrics[name].add_data_point(value, tags, metadata)

        await self._check_alert_rules(name, value, tags or {})

    async def record_gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record gauge metric"""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = MetricSeries(
                    name=name, metric_type=MetricType.GAUGE
                )

            self.metrics[name].add_data_point(value, tags, metadata)

        await self._check_alert_rules(name, value, tags or {})

    async def record_timer(
        self,
        name: str,
        duration_seconds: float,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record timer metric"""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = MetricSeries(
                    name=name, metric_type=MetricType.TIMER
                )

            self.metrics[name].add_data_point(duration_seconds, tags, metadata)

        await self._check_alert_rules(name, duration_seconds, tags or {})

    async def record_histogram(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record histogram metric"""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = MetricSeries(
                    name=name, metric_type=MetricType.HISTOGRAM
                )

            self.metrics[name].add_data_point(value, tags, metadata)

        await self._check_alert_rules(name, value, tags or {})

    # ========================================
    # USAGE TRACKING & BILLING
    # ========================================

    async def track_usage(
        self,
        user_id: str,
        resource_type: str,
        amount: float,
        cost_per_unit: float = 0.0,
    ) -> None:
        """Track resource usage for billing"""
        with self._lock:
            self.usage_tracking[user_id][resource_type] += amount
            if cost_per_unit > 0:
                cost = amount * cost_per_unit
                self.billing_metrics[f"user_{user_id}_cost"] += cost

        await self.record_counter(
            "sheily.billing.usage",
            amount,
            {"user_id": user_id, "resource_type": resource_type},
        )

        if cost_per_unit > 0:
            await self.record_counter(
                "sheily.billing.cost",
                amount * cost_per_unit,
                {"user_id": user_id, "resource_type": resource_type},
            )

    async def get_usage_summary(
        self, user_id: str, period_hours: int = 24
    ) -> Dict[str, Any]:
        """Get usage summary for user"""
        with self._lock:
            user_usage = self.usage_tracking.get(user_id, {})
            user_cost = self.billing_metrics.get(f"user_{user_id}_cost", 0.0)

        return {
            "user_id": user_id,
            "period_hours": period_hours,
            "total_cost": round(user_cost, 4),
            "resource_usage": dict(user_usage),
            "summary_generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ========================================
    # PERFORMANCE MONITORING
    # ========================================

    async def record_system_resources(self) -> None:
        """Record current system resource usage"""
        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.record_gauge("sheily.system.cpu_percent", cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / 1024 / 1024
            await self.record_gauge("sheily.system.memory_mb", memory_mb)
            await self.record_gauge("sheily.system.memory_percent", memory.percent)

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                await self.record_gauge(
                    "sheily.system.disk_read_mb", disk_io.read_bytes / 1024 / 1024
                )
                await self.record_gauge(
                    "sheily.system.disk_write_mb", disk_io.write_bytes / 1024 / 1024
                )

            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                await self.record_gauge(
                    "sheily.system.network_sent_mb", net_io.bytes_sent / 1024 / 1024
                )
                await self.record_gauge(
                    "sheily.system.network_recv_mb", net_io.bytes_recv / 1024 / 1024
                )

            # Store in local deques for analysis
            with self._lock:
                self.resource_usage["cpu_percent"].append(cpu_percent)
                self.resource_usage["memory_mb"].append(memory_mb)
                if disk_io:
                    self.resource_usage["disk_io"].append(
                        disk_io.read_bytes + disk_io.write_bytes
                    )
                if net_io:
                    self.resource_usage["network_io"].append(
                        net_io.bytes_sent + net_io.bytes_recv
                    )

        except ImportError:
            # psutil not available, skip system monitoring
            pass
        except Exception as e:
            logger.error(f"Error recording system resources: {e}")

    async def detect_performance_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies using statistical analysis"""
        anomalies = []

        with self._lock:
            for resource_name, values in self.resource_usage.items():
                if len(values) < 10:  # Need enough data
                    continue

                try:
                    # Calculate statistics
                    recent_values = list(values)[-10:]  # Last 10 values
                    mean_val = statistics.mean(recent_values)
                    std_dev = (
                        statistics.stdev(recent_values) if len(recent_values) > 1 else 0
                    )
                    latest_value = recent_values[-1]

                    # Detect outliers (values > 2 standard deviations from mean)
                    if std_dev > 0 and abs(latest_value - mean_val) > 2 * std_dev:
                        anomalies.append(
                            {
                                "resource": resource_name,
                                "current_value": latest_value,
                                "mean_value": mean_val,
                                "standard_deviation": std_dev,
                                "deviation_factor": abs(latest_value - mean_val)
                                / std_dev,
                                "detected_at": datetime.now(timezone.utc).isoformat(),
                            }
                        )

                except Exception as e:
                    logger.error(f"Error analyzing {resource_name}: {e}")

        return anomalies

    # ========================================
    # ALERTING SYSTEM
    # ========================================

    async def create_alert(
        self,
        name: str,
        level: AlertLevel,
        message: str,
        metric_name: str,
        threshold: float,
        current_value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create new alert"""
        alert_id = f"alert_{int(time.time())}_{len(self.alerts)}"

        alert = Alert(
            id=alert_id,
            name=name,
            level=level,
            message=message,
            metric_name=metric_name,
            threshold=threshold,
            current_value=current_value,
            triggered_at=datetime.now(timezone.utc),
            tags=tags or {},
        )

        with self._lock:
            self.alerts[alert_id] = alert

        # Publish alert event
        await publish_event(
            SheìlyEventType.PERFORMANCE_ALERT,
            {
                "alert_type": "system_alert",
                "alert_id": alert_id,
                "name": name,
                "level": level.value,
                "message": message,
                "metric_name": metric_name,
                "threshold": threshold,
                "current_value": current_value,
                "tags": tags or {},
            },
        )

        logger.warning(f"Alert created: {name} - {message}")
        return alert_id

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].acknowledged = True
                self.alerts[alert_id].acknowledged_at = datetime.now(timezone.utc)
                self.alerts[alert_id].acknowledged_by = acknowledged_by
                return True
        return False

    async def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules"""
        self.alert_rules = {
            "sheily.system.cpu_percent": {
                "threshold": 80.0,
                "condition": "greater_than",
                "level": AlertLevel.WARNING,
                "message": "High CPU usage detected",
            },
            "sheily.system.memory_percent": {
                "threshold": 85.0,
                "condition": "greater_than",
                "level": AlertLevel.CRITICAL,
                "message": "High memory usage detected",
            },
            "sheily.requests.errors": {
                "threshold": 10.0,
                "condition": "rate_greater_than",  # Per minute
                "level": AlertLevel.CRITICAL,
                "message": "High error rate detected",
            },
            "sheily.requests.duration": {
                "threshold": 5.0,  # 5 seconds
                "condition": "average_greater_than",
                "level": AlertLevel.WARNING,
                "message": "Slow response times detected",
            },
        }

    async def _check_alert_rules(
        self, metric_name: str, value: float, tags: Dict[str, str]
    ) -> None:
        """Check if metric value triggers any alert rules"""
        if metric_name not in self.alert_rules:
            return

        rule = self.alert_rules[metric_name]
        threshold = rule["threshold"]
        condition = rule["condition"]

        triggered = False

        if condition == "greater_than" and value > threshold:
            triggered = True
        elif condition == "less_than" and value < threshold:
            triggered = True
        elif condition == "rate_greater_than":
            # Check rate per minute
            series = self.metrics.get(metric_name)
            if series:
                rate = series.get_rate_per_minute()
                if rate > threshold:
                    triggered = True
        elif condition == "average_greater_than":
            # Check average over last 5 minutes
            series = self.metrics.get(metric_name)
            if series:
                avg = series.get_average(5)
                if avg and avg > threshold:
                    triggered = True

        if triggered:
            await self.create_alert(
                name=f"Alert: {metric_name}",
                level=AlertLevel(rule["level"]),
                message=rule["message"],
                metric_name=metric_name,
                threshold=threshold,
                current_value=value,
                tags=tags,
            )

    # ========================================
    # ANALYTICS & REPORTING
    # ========================================

    async def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        summary = {
            "period_hours": hours,
            "summary_generated_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
            "alerts": {"total": len(self.alerts), "unacknowledged": 0},
            "system_health": "unknown",
        }

        with self._lock:
            # Analyze each metric
            for name, series in self.metrics.items():
                recent_values = series.get_recent_values(hours * 60)
                if recent_values:
                    summary["metrics"][name] = {
                        "type": series.metric_type.value,
                        "current_value": series.get_current_value(),
                        "avg_value": round(statistics.mean(recent_values), 4),
                        "min_value": round(min(recent_values), 4),
                        "max_value": round(max(recent_values), 4),
                        "total_points": len(recent_values),
                        "last_updated": (
                            series.last_updated.isoformat()
                            if series.last_updated
                            else None
                        ),
                    }

            # Count unacknowledged alerts
            summary["alerts"]["unacknowledged"] = len(
                [alert for alert in self.alerts.values() if not alert.acknowledged]
            )

        # Determine overall system health
        if summary["alerts"]["unacknowledged"] == 0:
            summary["system_health"] = "healthy"
        elif any(
            alert.level == AlertLevel.CRITICAL
            for alert in self.alerts.values()
            if not alert.acknowledged
        ):
            summary["system_health"] = "critical"
        else:
            summary["system_health"] = "warning"

        return summary

    async def export_metrics(self, format: str = "json", hours: int = 24) -> str:
        """Export metrics in specified format"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        if format.lower() == "json":
            export_data = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "period_hours": hours,
                "metrics": {},
            }

            with self._lock:
                for name, series in self.metrics.items():
                    export_data["metrics"][name] = {
                        "type": series.metric_type.value,
                        "total_count": series.total_count,
                        "data_points": [
                            {
                                "timestamp": dp.timestamp.isoformat(),
                                "value": dp.value,
                                "tags": dp.tags,
                                "metadata": dp.metadata,
                            }
                            for dp in series.data_points
                            if dp.timestamp >= cutoff_time
                        ],
                    }

            return json.dumps(export_data, indent=2)

        else:
            raise ValueError(f"Unsupported export format: {format}")

    # ========================================
    # BACKGROUND MONITORING
    # ========================================

    async def _background_monitoring(self) -> None:
        """Background monitoring task"""
        logger.info("Started background analytics monitoring")

        while self._running:
            try:
                # Record system resources
                await self.record_system_resources()

                # Detect performance anomalies
                anomalies = await self.detect_performance_anomalies()
                for anomaly in anomalies:
                    await publish_event(
                        SheìlyEventType.PERFORMANCE_ALERT,
                        {"alert_type": "performance_anomaly", "anomaly_data": anomaly},
                    )

                # Cleanup old data (keep last 7 days)
                await self._cleanup_old_data(days=7)

                # Wait before next monitoring cycle
                await asyncio.sleep(60)  # Monitor every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                await asyncio.sleep(60)

        logger.info("Background analytics monitoring stopped")

    async def _cleanup_old_data(self, days: int = 7) -> None:
        """Cleanup old metric data"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)

        with self._lock:
            # Cleanup old alerts
            old_alert_ids = [
                alert_id
                for alert_id, alert in self.alerts.items()
                if alert.triggered_at < cutoff_time and alert.acknowledged
            ]
            for alert_id in old_alert_ids:
                del self.alerts[alert_id]

            # Note: MetricSeries already has maxlen=1000 on deques, so they self-cleanup


# ========================================
# GLOBAL INSTANCE
# ========================================

_analytics_system: Optional[SheìlyAnalyticsSystem] = None


async def get_analytics_system() -> SheìlyAnalyticsSystem:
    """Get global analytics system instance"""
    global _analytics_system
    if _analytics_system is None:
        _analytics_system = SheìlyAnalyticsSystem()
        await _analytics_system.initialize()
    return _analytics_system


async def initialize_analytics_system() -> SheìlyAnalyticsSystem:
    """Initialize the complete analytics system"""
    analytics = await get_analytics_system()
    logger.info("✅ Sheily Enterprise Analytics System initialized")
    return analytics


# ========================================
# CONVENIENCE FUNCTIONS
# ========================================


async def track_request_metrics(
    handler_name: str,
    duration: float,
    success: bool,
    user_id: Optional[str] = None,
    error: Optional[str] = None,
):
    """Convenience function to track request metrics"""
    analytics = await get_analytics_system()

    await analytics.record_counter(
        "sheily.requests.total", 1, {"handler": handler_name, "success": str(success)}
    )

    await analytics.record_timer(
        "sheily.requests.duration", duration, {"handler": handler_name}
    )

    if not success and error:
        await analytics.record_counter(
            "sheily.requests.errors", 1, {"handler": handler_name, "error": error}
        )

    if user_id:
        # Track usage for billing
        await analytics.track_usage(user_id, "requests", 1, cost_per_unit=0.001)


async def track_agent_metrics(
    agent_id: str,
    operation: str,
    success: bool,
    duration: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Convenience function to track agent metrics"""
    analytics = await get_analytics_system()

    await analytics.record_counter(
        "sheily.agents.operations",
        1,
        {"agent_id": agent_id, "operation": operation, "success": str(success)},
    )

    if duration:
        await analytics.record_timer(
            "sheily.agents.operation_duration",
            duration,
            {"agent_id": agent_id, "operation": operation},
        )


__all__ = [
    "SheìlyAnalyticsSystem",
    "MetricsSubscriber",
    "MetricType",
    "AlertLevel",
    "MetricDataPoint",
    "MetricSeries",
    "Alert",
    "get_analytics_system",
    "initialize_analytics_system",
    "track_request_metrics",
    "track_agent_metrics",
]
