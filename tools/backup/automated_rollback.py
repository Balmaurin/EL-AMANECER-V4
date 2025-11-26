# Automated Rollback System for Sheily AI
# =====================================

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp


@dataclass
class DeploymentStatus:
    """Status of a deployment"""

    name: str
    namespace: str
    version: str
    status: str
    health_status: str
    sync_status: str
    last_sync: datetime
    errors: List[str] = field(default_factory=list)


@dataclass
class RollbackTrigger:
    """Trigger conditions for rollback"""

    error_rate_threshold: float = 5.0  # 5% error rate
    response_time_threshold: float = 2000  # 2 seconds
    cpu_threshold: float = 90.0  # 90% CPU
    memory_threshold: float = 90.0  # 90% memory
    consecutive_failures: int = 3
    time_window_minutes: int = 5


class AutomatedRollbackSystem:
    """
    Sistema de rollback automático para Sheily AI
    Monitorea deployments y ejecuta rollbacks automáticos cuando es necesario
    """

    def __init__(self, argocd_url: str, argocd_token: str):
        self.argocd_url = argocd_url.rstrip("/")
        self.argocd_token = argocd_token
        self.session: Optional[aiohttp.ClientSession] = None

        # Configuración de rollback
        self.rollback_trigger = RollbackTrigger()
        self.monitoring_interval = 30  # segundos
        self.rollback_timeout = 300  # 5 minutos

        # Estado
        self.last_known_good_versions: Dict[str, str] = {}
        self.failure_counts: Dict[str, int] = {}

        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.argocd_token}",
                "Content-Type": "application/json",
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def start_monitoring(self):
        """Inicia el monitoreo continuo de deployments"""
        self.logger.info("Starting automated rollback monitoring...")

        while True:
            try:
                await self.monitor_deployments()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def monitor_deployments(self):
        """Monitorea todos los deployments críticos"""
        applications = await self.get_argocd_applications()

        for app in applications:
            if app["metadata"]["name"].startswith("sheily-ai"):
                await self.check_application_health(app)

    async def check_application_health(self, app: Dict[str, Any]):
        """Verifica la salud de una aplicación específica"""
        app_name = app["metadata"]["name"]
        app_status = app.get("status", {})

        # Obtener métricas de health
        health_metrics = await self.get_application_metrics(app_name)

        # Evaluar si necesita rollback
        needs_rollback = await self.evaluate_rollback_conditions(
            app_name, health_metrics
        )

        if needs_rollback:
            await self.execute_rollback(app_name)
        else:
            # Actualizar versión buena conocida
            if app_status.get("sync", {}).get("revision"):
                self.last_known_good_versions[app_name] = app_status["sync"]["revision"]

    async def evaluate_rollback_conditions(
        self, app_name: str, metrics: Dict[str, Any]
    ) -> bool:
        """Evalúa si se cumplen las condiciones para rollback"""

        # Verificar métricas de error
        error_rate = metrics.get("error_rate", 0)
        if error_rate > self.rollback_trigger.error_rate_threshold:
            self.logger.warning(
                f"High error rate detected for {app_name}: {error_rate}%"
            )
            self.failure_counts[app_name] = self.failure_counts.get(app_name, 0) + 1

        # Verificar tiempo de respuesta
        avg_response_time = metrics.get("avg_response_time", 0)
        if avg_response_time > self.rollback_trigger.response_time_threshold:
            self.logger.warning(
                f"High response time for {app_name}: {avg_response_time}ms"
            )
            self.failure_counts[app_name] = self.failure_counts.get(app_name, 0) + 1

        # Verificar recursos del sistema
        cpu_usage = metrics.get("cpu_usage", 0)
        memory_usage = metrics.get("memory_usage", 0)

        if cpu_usage > self.rollback_trigger.cpu_threshold:
            self.logger.warning(f"High CPU usage for {app_name}: {cpu_usage}%")

        if memory_usage > self.rollback_trigger.memory_threshold:
            self.logger.warning(f"High memory usage for {app_name}: {memory_usage}%")

        # Evaluar si debe hacer rollback
        consecutive_failures = self.failure_counts.get(app_name, 0)

        if consecutive_failures >= self.rollback_trigger.consecutive_failures:
            self.logger.error(
                f"Triggering rollback for {app_name} - {consecutive_failures} consecutive failures"
            )
            return True

        return False

    async def execute_rollback(self, app_name: str):
        """Ejecuta rollback automático"""
        self.logger.info(f"Executing automated rollback for {app_name}")

        # Obtener la última versión buena conocida
        good_version = self.last_known_good_versions.get(app_name)
        if not good_version:
            self.logger.error(f"No known good version for {app_name}")
            return

        try:
            # Ejecutar rollback via ArgoCD API
            rollback_payload = {
                "revision": good_version,
                "prune": True,
                "strategy": {"hook": {"force": True}},
            }

            url = f"{self.argocd_url}/api/v1/applications/{app_name}/rollback"
            async with self.session.post(url, json=rollback_payload) as response:
                if response.status == 200:
                    self.logger.info(
                        f"Rollback successful for {app_name} to version {good_version}"
                    )
                    # Reset failure count
                    self.failure_counts[app_name] = 0
                    # Notificar equipos
                    await self.notify_rollback(app_name, good_version)
                else:
                    error_text = await response.text()
                    self.logger.error(f"Rollback failed for {app_name}: {error_text}")

        except Exception as e:
            self.logger.error(f"Error executing rollback for {app_name}: {e}")

    async def notify_rollback(self, app_name: str, version: str):
        """Notifica sobre el rollback ejecutado"""
        # Aquí se integraría con Slack, Teams, PagerDuty, etc.
        message = f"""
🚨 AUTOMATED ROLLBACK EXECUTED 🚨

Application: {app_name}
Rolled back to: {version}
Timestamp: {datetime.now().isoformat()}
Reason: Multiple health check failures detected

System has automatically restored to last known good state.
Please investigate root cause before deploying new changes.
        """

        self.logger.critical(message)
        # En producción: enviar a Slack, email, PagerDuty, etc.

    async def get_argocd_applications(self) -> List[Dict[str, Any]]:
        """Obtiene lista de aplicaciones de ArgoCD"""
        try:
            url = f"{self.argocd_url}/api/v1/applications"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("items", [])
                else:
                    self.logger.error(f"Failed to get applications: {response.status}")
                    return []
        except Exception as e:
            self.logger.error(f"Error getting applications: {e}")
            return []

    async def get_application_metrics(self, app_name: str) -> Dict[str, Any]:
        """Obtiene métricas de health de la aplicación"""
        # En producción, esto se conectaría con Prometheus, DataDog, etc.
        # Por ahora, simular métricas básicas
        return {
            "error_rate": 2.5,  # 2.5%
            "avg_response_time": 150,  # 150ms
            "cpu_usage": 65.0,  # 65%
            "memory_usage": 70.0,  # 70%
            "active_connections": 150,
        }


# Función principal
async def main():
    """Función principal del sistema de rollback automático"""
    argocd_url = "https://argocd.sheily-ai.com"
    argocd_token = "your-argocd-token-here"  # En producción: desde secrets

    async with AutomatedRollbackSystem(argocd_url, argocd_token) as rollback_system:
        await rollback_system.start_monitoring()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
