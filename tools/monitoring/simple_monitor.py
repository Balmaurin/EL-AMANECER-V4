#!/usr/bin/env python3
"""
Simple Monitoring System - Sheily AI
====================================

Sistema de monitoreo básico pero funcional para métricas críticas.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleMonitor:
    """Sistema de monitoreo simple pero efectivo"""

    def __init__(self, metrics_dir: str = "monitoring/metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_history: List[Dict[str, Any]] = []

    def start_monitoring(self, interval_seconds: int = 60):
        """Iniciar monitoreo continuo"""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval_seconds,), daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"✅ Monitoring started (interval: {interval_seconds}s)")

    def stop_monitoring(self):
        """Detener monitoreo"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 Monitoring stopped")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Obtener métricas actuales"""
        return {
            "timestamp": datetime.now().isoformat(),
            "system": self._get_system_metrics(),
            "application": self._get_application_metrics(),
            "alerts": self._check_alerts(),
        }

    def _monitoring_loop(self, interval: int):
        """Loop principal de monitoreo"""
        while self.is_monitoring:
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)

                # Guardar métricas cada 5 minutos
                if len(self.metrics_history) % 5 == 0:
                    self._save_metrics_snapshot()

                # Verificar alertas
                alerts = metrics.get("alerts", [])
                if alerts:
                    self._handle_alerts(alerts)

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del sistema"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "disk_usage_percent": psutil.disk_usage("/").percent,
            "disk_free_gb": psutil.disk_usage("/").free / (1024**3),
            "network_connections": len(psutil.net_connections()),
            "load_average": (
                psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
            ),
        }

    def _get_application_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de aplicación (simuladas por ahora)"""
        # En producción, estas vendrían de la aplicación real
        return {
            "active_connections": 0,  # Vendría de FastAPI
            "requests_per_second": 0,  # Vendría del middleware
            "error_rate_percent": 0,  # Vendría de logs
            "response_time_avg_ms": 0,  # Vendría de middleware
            "memory_usage_mb": psutil.Process().memory_info().rss / (1024**2),
        }

    def _check_alerts(self) -> List[Dict[str, Any]]:
        """Verificar condiciones de alerta"""
        alerts = []
        metrics = self._get_system_metrics()

        # Alertas de CPU alta
        if metrics["cpu_percent"] > 80:
            alerts.append(
                {
                    "level": "WARNING",
                    "type": "HIGH_CPU",
                    "message": f"CPU usage is {metrics['cpu_percent']:.1f}%",
                    "value": metrics["cpu_percent"],
                    "threshold": 80,
                }
            )

        # Alertas de memoria alta
        if metrics["memory_percent"] > 85:
            alerts.append(
                {
                    "level": "CRITICAL",
                    "type": "HIGH_MEMORY",
                    "message": f"Memory usage is {metrics['memory_percent']:.1f}%",
                    "value": metrics["memory_percent"],
                    "threshold": 85,
                }
            )

        # Alertas de disco bajo
        if metrics["disk_usage_percent"] > 90:
            alerts.append(
                {
                    "level": "WARNING",
                    "type": "LOW_DISK_SPACE",
                    "message": f"Disk usage is {metrics['disk_usage_percent']:.1f}%",
                    "value": metrics["disk_usage_percent"],
                    "threshold": 90,
                }
            )

        return alerts

    def _handle_alerts(self, alerts: List[Dict[str, Any]]):
        """Manejar alertas detectadas"""
        for alert in alerts:
            level = alert["level"]
            message = alert["message"]

            if level == "CRITICAL":
                logger.critical(f"🚨 CRITICAL ALERT: {message}")
            elif level == "WARNING":
                logger.warning(f"⚠️ WARNING ALERT: {message}")
            else:
                logger.info(f"ℹ️ INFO ALERT: {message}")

            # En producción, aquí enviarías emails, Slack, etc.

    def _save_metrics_snapshot(self):
        """Guardar snapshot de métricas"""
        if not self.metrics_history:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_snapshot_{timestamp}.json"
        filepath = self.metrics_dir / filename

        # Guardar últimas 10 métricas
        recent_metrics = (
            self.metrics_history[-10:]
            if len(self.metrics_history) > 10
            else self.metrics_history
        )

        with open(filepath, "w") as f:
            json.dump(
                {
                    "snapshot_time": datetime.now().isoformat(),
                    "metrics_count": len(recent_metrics),
                    "metrics": recent_metrics,
                },
                f,
                indent=2,
                default=str,
            )

        # Mantener solo últimos 10 archivos
        self._cleanup_old_snapshots()

    def _cleanup_old_snapshots(self, keep_last: int = 10):
        """Limpiar snapshots antiguos"""
        snapshots = sorted(self.metrics_dir.glob("metrics_snapshot_*.json"))
        if len(snapshots) > keep_last:
            for old_snapshot in snapshots[:-keep_last]:
                old_snapshot.unlink()

    def get_metrics_report(self) -> Dict[str, Any]:
        """Generar reporte completo de métricas"""
        current = self.get_current_metrics()

        # Estadísticas históricas
        if self.metrics_history:
            cpu_values = [
                m["system"]["cpu_percent"] for m in self.metrics_history[-60:]
            ]  # Última hora
            memory_values = [
                m["system"]["memory_percent"] for m in self.metrics_history[-60:]
            ]

            stats = {
                "cpu_avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "cpu_max": max(cpu_values) if cpu_values else 0,
                "memory_avg": (
                    sum(memory_values) / len(memory_values) if memory_values else 0
                ),
                "memory_max": max(memory_values) if memory_values else 0,
                "total_measurements": len(self.metrics_history),
            }
        else:
            stats = {"message": "No historical data available"}

        return {
            "current_metrics": current,
            "historical_stats": stats,
            "alerts_active": len(current.get("alerts", [])),
            "monitoring_status": "active" if self.is_monitoring else "inactive",
        }


# Instancia global
monitor = SimpleMonitor()


def start_monitoring(interval_seconds: int = 60):
    """Función de conveniencia para iniciar monitoreo"""
    monitor.start_monitoring(interval_seconds)


def stop_monitoring():
    """Función de conveniencia para detener monitoreo"""
    monitor.stop_monitoring()


def get_metrics():
    """Función de conveniencia para obtener métricas"""
    return monitor.get_metrics_report()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple Monitoring System - Sheily AI")
    parser.add_argument("--start", action="store_true", help="Start monitoring")
    parser.add_argument("--stop", action="store_true", help="Stop monitoring")
    parser.add_argument("--status", action="store_true", help="Show monitoring status")
    parser.add_argument(
        "--interval", type=int, default=60, help="Monitoring interval in seconds"
    )

    args = parser.parse_args()

    if args.start:
        print(f"🚀 Starting monitoring (interval: {args.interval}s)...")
        start_monitoring(args.interval)
        print("✅ Monitoring started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping monitoring...")
            stop_monitoring()
            print("✅ Monitoring stopped.")

    elif args.stop:
        print("🛑 Stopping monitoring...")
        stop_monitoring()
        print("✅ Monitoring stopped.")

    elif args.status:
        report = get_metrics()
        print("📊 MONITORING STATUS")
        print("=" * 30)
        print(f"Status: {report['monitoring_status']}")
        print(f"Active Alerts: {report['alerts_active']}")
        print(
            f"Total Measurements: {report['historical_stats'].get('total_measurements', 0)}"
        )

        current = report["current_metrics"]["system"]
        print("\n📈 CURRENT METRICS:")
        print(f"CPU: {current['cpu_percent']:.1f}%")
        print(f"Memory: {current['memory_percent']:.1f}%")
        print(f"Disk: {current['disk_usage_percent']:.1f}%")

    else:
        parser.print_help()
