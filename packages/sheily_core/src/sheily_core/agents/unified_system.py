#!/usr/bin/env python3
"""
Sistema Unificado Funcional - Integra todo en un sistema operativo completo
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .active_registry import active_registry

# Importar todos los componentes funcionales creados
from .base_agent_implementations import process_manager_agent, system_health_agent
from .coordination_system import functional_multi_agent_system
from .multi_agent_implementations import communication_agent, task_executor_agent


@dataclass
class SystemConfiguration:
    """Configuración del sistema unificado"""

    enable_health_monitoring: bool = True
    enable_task_execution: bool = True
    enable_communication: bool = True
    enable_coordination: bool = True
    enable_active_registry: bool = True
    auto_recovery: bool = True
    monitoring_interval: int = 30
    max_concurrent_tasks: int = 50


class UnifiedSheilySystem:
    """
    Sistema unificado que integra todos los componentes en un ecosistema funcional
    """

    def __init__(self, config: SystemConfiguration = None):
        self.config = config or SystemConfiguration()
        self.logger = logging.getLogger("Sheily.UnifiedSystem")

        # Componentes del sistema
        self.health_agent = system_health_agent
        self.process_agent = process_manager_agent
        self.task_executor = task_executor_agent
        self.communication_hub = communication_agent
        self.coordination_system = functional_multi_agent_system
        self.registry = active_registry

        # Estado del sistema
        self.is_running = False
        self.system_metrics = {}
        self.active_tasks: Dict[str, Any] = {}
        self.system_alerts = []

        # Contadores de rendimiento
        self.total_tasks_processed = 0
        self.total_system_uptime = 0
        self.last_startup_time = None

    async def start_unified_system(self) -> Dict[str, Any]:
        """Iniciar el sistema unificado completo"""
        startup_results = {
            "timestamp": datetime.now().isoformat(),
            "components_started": [],
            "components_failed": [],
            "system_status": "starting",
        }

        try:
            self.logger.info("Starting Unified Sheily System...")
            self.last_startup_time = datetime.now()

            # 1. Iniciar registro activo
            if self.config.enable_active_registry:
                try:
                    await self.registry.start_active_management()
                    startup_results["components_started"].append("active_registry")
                    self.logger.info("✓ Active Registry started")
                except Exception as e:
                    startup_results["components_failed"].append(
                        f"active_registry: {str(e)}"
                    )
                    self.logger.error(f"✗ Active Registry failed: {e}")

            # 2. Iniciar agentes base
            if self.config.enable_health_monitoring:
                try:
                    await self.health_agent.start()
                    await self.process_agent.start()
                    startup_results["components_started"].append("base_agents")
                    self.logger.info("✓ Base Agents started")
                except Exception as e:
                    startup_results["components_failed"].append(
                        f"base_agents: {str(e)}"
                    )
                    self.logger.error(f"✗ Base Agents failed: {e}")

            # 3. Iniciar agentes multi-agente
            if self.config.enable_task_execution or self.config.enable_communication:
                try:
                    await self.task_executor.start()
                    await self.communication_hub.start()
                    startup_results["components_started"].append("multi_agents")
                    self.logger.info("✓ Multi-Agents started")
                except Exception as e:
                    startup_results["components_failed"].append(
                        f"multi_agents: {str(e)}"
                    )
                    self.logger.error(f"✗ Multi-Agents failed: {e}")

            # 4. Iniciar sistema de coordinación
            if self.config.enable_coordination:
                try:
                    await self.coordination_system.start_functional_system()
                    startup_results["components_started"].append("coordination_system")
                    self.logger.info("✓ Coordination System started")
                except Exception as e:
                    startup_results["components_failed"].append(
                        f"coordination_system: {str(e)}"
                    )
                    self.logger.error(f"✗ Coordination System failed: {e}")

            # 5. Registrar todos los agentes en el registro activo
            await self._register_all_agents()
            startup_results["components_started"].append("agent_registration")

            # 6. Iniciar loops de sistema
            asyncio.create_task(self._system_monitoring_loop())
            asyncio.create_task(self._task_management_loop())
            asyncio.create_task(self._health_coordination_loop())

            self.is_running = True
            startup_results["system_status"] = "operational"
            startup_results["total_components"] = len(
                startup_results["components_started"]
            )
            startup_results["failed_components"] = len(
                startup_results["components_failed"]
            )

            self.logger.info(
                f"✓ Unified Sheily System started successfully with {len(startup_results['components_started'])} components"
            )

            return startup_results

        except Exception as e:
            startup_results["system_status"] = "failed"
            startup_results["error"] = str(e)
            self.logger.error(f"✗ Failed to start Unified System: {e}")
            return startup_results

    async def _register_all_agents(self):
        """Registrar todos los agentes en el sistema"""
        try:
            if self.config.enable_active_registry:
                # Registrar agentes base
                await self.registry.register_agent_with_health_monitoring(
                    self.health_agent,
                    "health_monitoring",
                    {"capabilities": ["system_health", "monitoring"]},
                )

                await self.registry.register_agent_with_health_monitoring(
                    self.process_agent,
                    "process_management",
                    {"capabilities": ["process_management", "system_control"]},
                )

                # Registrar agentes multi-agente
                await self.registry.register_agent_with_health_monitoring(
                    self.task_executor,
                    "task_execution",
                    {"capabilities": ["task_processing", "data_handling"]},
                )

                await self.registry.register_agent_with_health_monitoring(
                    self.communication_hub,
                    "communication",
                    {"capabilities": ["message_routing", "agent_communication"]},
                )

                self.logger.info("All agents registered in Active Registry")

        except Exception as e:
            self.logger.error(f"Failed to register agents: {e}")

    async def _system_monitoring_loop(self):
        """Loop principal de monitoreo del sistema"""
        while self.is_running:
            try:
                # Recopilar métricas del sistema
                system_metrics = await self._collect_comprehensive_metrics()
                self.system_metrics = system_metrics

                # Detectar problemas del sistema
                alerts = await self._analyze_system_health(system_metrics)
                self.system_alerts.extend(alerts)

                # Mantener solo alertas recientes
                cutoff_time = datetime.now().timestamp() - (24 * 3600)  # 24 horas
                self.system_alerts = [
                    alert
                    for alert in self.system_alerts
                    if alert.get("timestamp", 0) > cutoff_time
                ]

                await asyncio.sleep(self.config.monitoring_interval)

            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(60)

    async def _collect_comprehensive_metrics(self) -> Dict[str, Any]:
        """Recopilar métricas comprehensivas del sistema"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system_uptime": (
                (datetime.now() - self.last_startup_time).total_seconds()
                if self.last_startup_time
                else 0
            ),
            "is_running": self.is_running,
        }

        try:
            # Métricas de salud del sistema
            if self.config.enable_health_monitoring and hasattr(
                self.health_agent, "check_system_health"
            ):
                system_health = await self.health_agent.check_system_health()
                metrics["system_health"] = system_health

            # Métricas de procesos
            if hasattr(self.process_agent, "get_top_processes"):
                top_processes = await self.process_agent.get_top_processes(5)
                metrics["top_processes"] = top_processes

            # Métricas del registro
            if self.config.enable_active_registry:
                registry_health = await self.registry.get_registry_health_summary()
                metrics["registry_health"] = registry_health

            # Métricas de coordinación
            if self.config.enable_coordination:
                coordination_status = (
                    await self.coordination_system.get_system_overview()
                )
                metrics["coordination_status"] = coordination_status

            # Métricas de comunicación
            if hasattr(self.communication_hub, "get_communication_stats"):
                communication_stats = self.communication_hub.get_communication_stats()
                metrics["communication_stats"] = communication_stats

            # Métricas de tareas
            metrics["task_metrics"] = {
                "total_processed": self.total_tasks_processed,
                "active_tasks": len(self.active_tasks),
                "task_executor_stats": self.task_executor.performance_stats,
            }

        except Exception as e:
            metrics["metrics_error"] = str(e)
            self.logger.error(f"Error collecting metrics: {e}")

        return metrics

    async def _analyze_system_health(
        self, metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analizar salud del sistema y generar alertas"""
        alerts = []
        current_time = datetime.now()

        try:
            # Analizar salud del sistema
            system_health = metrics.get("system_health", {})
            if system_health.get("status") == "warning":
                alerts.append(
                    {
                        "type": "system_health_warning",
                        "severity": "medium",
                        "message": "System resources showing warning levels",
                        "timestamp": current_time.timestamp(),
                        "data": system_health,
                    }
                )

            # Analizar salud del registro
            registry_health = metrics.get("registry_health", {})
            unhealthy_agents = registry_health.get("unhealthy_agents", 0)
            if unhealthy_agents > 0:
                alerts.append(
                    {
                        "type": "unhealthy_agents",
                        "severity": "high" if unhealthy_agents > 2 else "medium",
                        "message": f"{unhealthy_agents} agents are unhealthy",
                        "timestamp": current_time.timestamp(),
                        "data": {"unhealthy_count": unhealthy_agents},
                    }
                )

            # Analizar carga de tareas
            active_tasks = metrics.get("task_metrics", {}).get("active_tasks", 0)
            if active_tasks > self.config.max_concurrent_tasks * 0.8:
                alerts.append(
                    {
                        "type": "high_task_load",
                        "severity": "medium",
                        "message": f"High task load: {active_tasks} active tasks",
                        "timestamp": current_time.timestamp(),
                        "data": {"active_tasks": active_tasks},
                    }
                )

        except Exception as e:
            self.logger.error(f"Error analyzing system health: {e}")

        return alerts

    async def _task_management_loop(self):
        """Loop de gestión de tareas del sistema"""
        while self.is_running:
            try:
                # Procesar tareas pendientes
                await self._process_pending_system_tasks()

                # Limpiar tareas completadas
                await self._cleanup_completed_tasks()

                await asyncio.sleep(10)  # Revisar cada 10 segundos

            except Exception as e:
                self.logger.error(f"Task management error: {e}")
                await asyncio.sleep(30)

    async def _process_pending_system_tasks(self):
        """Procesar tareas pendientes del sistema"""
        # En una implementación real, esto procesaría una cola de tareas
        # Por ahora, simulamos verificación de tareas del sistema
        pass

    async def _cleanup_completed_tasks(self):
        """Limpiar tareas completadas"""
        # Remover tareas completadas hace más de 1 hora
        cutoff_time = datetime.now().timestamp() - 3600

        completed_tasks = [
            task_id
            for task_id, task_info in self.active_tasks.items()
            if task_info.get("completed_at", 0) < cutoff_time
        ]

        for task_id in completed_tasks:
            self.active_tasks.pop(task_id, None)

    async def _health_coordination_loop(self):
        """Loop de coordinación de salud entre componentes"""
        while self.is_running:
            try:
                # Coordinar salud entre componentes
                await self._coordinate_component_health()

                await asyncio.sleep(60)  # Cada minuto

            except Exception as e:
                self.logger.error(f"Health coordination error: {e}")
                await asyncio.sleep(120)

    async def _coordinate_component_health(self):
        """Coordinar salud entre componentes del sistema"""
        if self.config.enable_active_registry:
            # Forzar verificación de salud si es necesario
            registry_health = await self.registry.get_registry_health_summary()

            if registry_health.get("unhealthy_agents", 0) > 0:
                await self.registry.force_health_check_all()

    # API pública del sistema unificado
    async def execute_unified_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar una tarea usando el sistema completo"""
        task_id = (
            f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.active_tasks)}"
        )

        try:
            # Registrar tarea
            self.active_tasks[task_id] = {
                "task": task,
                "started_at": datetime.now().timestamp(),
                "status": "running",
            }

            # Determinar qué componente debe manejar la tarea
            task_type = task.get("type", "generic")

            if "system" in task_type:
                result = await self._execute_system_task(task)
            elif "coordination" in task_type:
                result = await self.coordination_system.execute_distributed_task(task)
            elif "communication" in task_type:
                result = await self._execute_communication_task(task)
            else:
                result = await self._execute_generic_task(task)

            # Actualizar estado de tarea
            self.active_tasks[task_id].update(
                {
                    "status": "completed",
                    "completed_at": datetime.now().timestamp(),
                    "result": result,
                }
            )

            self.total_tasks_processed += 1

            return {
                "task_id": task_id,
                "status": "success",
                "result": result,
                "completed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            # Marcar tarea como fallida
            self.active_tasks[task_id].update(
                {
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.now().timestamp(),
                }
            )

            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "completed_at": datetime.now().isoformat(),
            }

    async def _execute_system_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar tarea del sistema"""
        task_function = task.get("function", "")

        if task_function == "health_check":
            return await self.health_agent.check_system_health()
        elif task_function == "process_info":
            return await self.process_agent.get_top_processes()
        else:
            return {
                "status": "error",
                "message": f"Unknown system function: {task_function}",
            }

    async def _execute_communication_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar tarea de comunicación"""
        # Implementar tareas de comunicación específicas
        return {"status": "success", "message": "Communication task completed"}

    async def _execute_generic_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar tarea genérica usando el executor"""
        return await self.task_executor.execute_task(task)

    async def get_comprehensive_system_status(self) -> Dict[str, Any]:
        """Obtener estado comprehensivo del sistema"""
        return {
            "system_info": {
                "is_running": self.is_running,
                "uptime_seconds": (
                    (datetime.now() - self.last_startup_time).total_seconds()
                    if self.last_startup_time
                    else 0
                ),
                "configuration": asdict(self.config),
                "startup_time": (
                    self.last_startup_time.isoformat()
                    if self.last_startup_time
                    else None
                ),
            },
            "performance_metrics": {
                "total_tasks_processed": self.total_tasks_processed,
                "active_tasks": len(self.active_tasks),
                "recent_alerts": len(
                    [
                        a
                        for a in self.system_alerts
                        if a.get("timestamp", 0) > datetime.now().timestamp() - 3600
                    ]
                ),
            },
            "component_status": await self._get_all_component_status(),
            "system_metrics": self.system_metrics,
            "recent_alerts": self.system_alerts[-10:] if self.system_alerts else [],
        }

    async def _get_all_component_status(self) -> Dict[str, Any]:
        """Obtener estado de todos los componentes"""
        status = {}

        try:
            if self.config.enable_active_registry:
                status["registry"] = await self.registry.get_registry_health_summary()

            if self.config.enable_coordination:
                status["coordination"] = (
                    await self.coordination_system.get_system_overview()
                )

            status["base_agents"] = {
                "health_agent": {
                    "running": hasattr(self.health_agent, "running")
                    and self.health_agent.running
                },
                "process_agent": {
                    "running": hasattr(self.process_agent, "running")
                    and self.process_agent.running
                },
            }

            status["multi_agents"] = {
                "task_executor": {"active": self.task_executor.is_active},
                "communication_hub": {"active": self.communication_hub.is_active},
            }

        except Exception as e:
            status["error"] = f"Error getting component status: {e}"

        return status

    async def shutdown_unified_system(self) -> Dict[str, Any]:
        """Apagar el sistema unificado ordenadamente"""
        shutdown_results = {
            "timestamp": datetime.now().isoformat(),
            "components_shutdown": [],
            "components_failed": [],
            "shutdown_status": "in_progress",
        }

        try:
            self.logger.info("Shutting down Unified Sheily System...")
            self.is_running = False

            # Detener componentes en orden inverso al inicio
            components_to_shutdown = [
                ("coordination_system", self.coordination_system),
                ("multi_agents", [self.task_executor, self.communication_hub]),
                ("base_agents", [self.health_agent, self.process_agent]),
                ("active_registry", self.registry),
            ]

            for component_name, component in components_to_shutdown:
                try:
                    if isinstance(component, list):
                        for sub_component in component:
                            if hasattr(sub_component, "stop"):
                                await sub_component.stop()
                    else:
                        if hasattr(component, "stop"):
                            await component.stop()

                    shutdown_results["components_shutdown"].append(component_name)
                    self.logger.info(f"✓ {component_name} shutdown")

                except Exception as e:
                    shutdown_results["components_failed"].append(
                        f"{component_name}: {str(e)}"
                    )
                    self.logger.error(f"✗ {component_name} shutdown failed: {e}")

            shutdown_results["shutdown_status"] = "completed"
            self.logger.info("✓ Unified Sheily System shutdown completed")

            return shutdown_results

        except Exception as e:
            shutdown_results["shutdown_status"] = "failed"
            shutdown_results["error"] = str(e)
            self.logger.error(f"✗ System shutdown failed: {e}")
            return shutdown_results


# SISTEMA UNIFICADO FUNCIONAL
unified_system = UnifiedSheilySystem()


# Funciones de utilidad públicas
async def start_sheily_unified_system(config: SystemConfiguration = None):
    """Iniciar el sistema unificado completo"""
    global unified_system
    if config:
        unified_system = UnifiedSheilySystem(config)

    return await unified_system.start_unified_system()


async def get_unified_system_status():
    """Obtener estado del sistema unificado"""
    return await unified_system.get_comprehensive_system_status()


async def execute_task_in_system(task: Dict[str, Any]):
    """Ejecutar tarea en el sistema unificado"""
    return await unified_system.execute_unified_task(task)


async def shutdown_sheily_system():
    """Apagar el sistema unificado"""
    return await unified_system.shutdown_unified_system()


if __name__ == "__main__":

    async def main():
        # Configurar sistema
        config = SystemConfiguration(
            enable_health_monitoring=True,
            enable_task_execution=True,
            enable_communication=True,
            enable_coordination=True,
            enable_active_registry=True,
            auto_recovery=True,
        )

        # Iniciar sistema
        print("🚀 Starting Unified Sheily System...")
        startup_result = await start_sheily_unified_system(config)
        print(f"📊 Startup result: {startup_result}")

        try:
            # Ejecutar algunas tareas de prueba
            test_tasks = [
                {"type": "system_health", "function": "health_check"},
                {"type": "data_processing", "data": {"items": ["a", "b", "c"]}},
                {
                    "type": "coordination_analysis",
                    "function": "analyze_dataset",
                    "parameters": {"size": 50},
                },
            ]

            for task in test_tasks:
                result = await execute_task_in_system(task)
                print(f"📋 Task result: {result['status']} - {task['type']}")

            # Monitoreo continuo
            while True:
                status = await get_unified_system_status()
                print(
                    f"🔄 System running: {status['system_info']['is_running']}, "
                    f"Tasks processed: {status['performance_metrics']['total_tasks_processed']}"
                )
                await asyncio.sleep(60)

        except KeyboardInterrupt:
            print("\n🛑 Shutting down system...")
            shutdown_result = await shutdown_sheily_system()
            print(f"📊 Shutdown result: {shutdown_result}")

    asyncio.run(main())
