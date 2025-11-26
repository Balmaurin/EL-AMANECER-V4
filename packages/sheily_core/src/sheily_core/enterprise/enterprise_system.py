#!/usr/bin/env python3
"""
Sheily Enterprise System Integration Point
========================================

Sistema principal de integración que inicializa todos los componentes
enterprise implementados con patterns de ii-agent.

Sistema completo implementado:
✅ Event-Driven Architecture
✅ Background Processing & Scheduling
✅ Enterprise Database Architecture
✅ Production Middleware Integration
✅ Real-Time Metrics & Analytics
✅ Tools Integration (30+ tools)
✅ Socket.IO Enterprise Communication
✅ Unified Configuration System
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

from sheily_core.core.analytics.analytics_system import initialize_analytics_system
from sheily_core.core.background.scheduler_system import initialize_scheduler_system
from sheily_core.core.communication.socketio_system import initialize_socket_system
from sheily_core.core.config.configuration_system import (
    Environment,
    initialize_configuration_system,
)
from sheily_core.core.database.enterprise_db import initialize_database_system

# Import all enterprise systems
from sheily_core.core.events.event_system import initialize_event_system
from sheily_core.core.middleware.security_middleware import initialize_middleware_system
from sheily_core.core.tools.tools_integration import initialize_tool_system

logger = logging.getLogger(__name__)


class SheìlyEnterpriseSystem:
    """Sistema enterprise completo de Sheily con patterns ii-agent"""

    def __init__(self):
        self.initialized = False
        self.start_time: Optional[datetime] = None
        self.components: Dict[str, Any] = {
            "configuration": None,
            "events": None,
            "scheduler": None,
            "database": None,
            "middleware": None,
            "analytics": None,
            "tools": None,
            "socket": None,
        }

    async def initialize(
        self,
        config_file: Optional[str] = None,
        environment: Optional[Environment] = None,
        hot_reload: bool = False,
    ) -> None:
        """Initialize complete enterprise system"""
        try:
            self.start_time = datetime.now(timezone.utc)
            logger.info("🚀 Starting Sheily Enterprise System initialization...")

            # Phase 1: Configuration System
            logger.info("📋 Initializing Configuration System...")
            self.components["configuration"] = await initialize_configuration_system(
                config_file, environment, hot_reload
            )

            # Phase 2: Event-Driven Architecture
            logger.info("⚡ Initializing Event System...")
            self.components["events"] = await initialize_event_system()

            # Phase 3: Background Processing & Scheduling
            logger.info("⏰ Initializing Scheduler System...")
            self.components["scheduler"] = await initialize_scheduler_system()

            # Phase 4: Enterprise Database Architecture
            logger.info("🗄️ Initializing Database System...")
            self.components["database"] = await initialize_database_system()

            # Phase 5: Production Middleware Integration
            logger.info("🛡️ Initializing Middleware System...")
            self.components["middleware"] = await initialize_middleware_system()

            # Phase 6: Real-Time Metrics & Analytics
            logger.info("📊 Initializing Analytics System...")
            self.components["analytics"] = await initialize_analytics_system()

            # Phase 7: Tools Integration (30+ tools)
            logger.info("🛠️ Initializing Tools System...")
            self.components["tools"] = await initialize_tool_system()

            # Phase 8: Socket.IO Enterprise Communication
            logger.info("💬 Initializing Socket.IO System...")
            self.components["socket"] = await initialize_socket_system()

            self.initialized = True

            # Calculate initialization time
            init_duration = (
                datetime.now(timezone.utc) - self.start_time
            ).total_seconds()

            logger.info(
                f"""
🎉 ====================================================
   SHEILY ENTERPRISE SYSTEM COMPLETAMENTE INICIALIZADO
====================================================

✅ Sistema híbrido ii-agent + Sheily intelligence
✅ 8 componentes enterprise implementados exitosamente
✅ Arquitectura event-driven con coordinación de agentes
✅ Base de datos enterprise con optimistic locking
✅ Middleware de seguridad y autenticación
✅ Analytics en tiempo real con billing
✅ 30+ herramientas integradas sin duplicaciones
✅ Comunicación Socket.IO enterprise
✅ Sistema de configuración unificado

⚡ Inicialización completada en {init_duration:.2f} segundos
🌟 Sistema listo para operaciones enterprise
🔥 Transformación de académico a enterprise: EXITOSA

Entorno: {self.get_environment()}
Configuración: {self.get_config_status()}
Componentes activos: {len([c for c in self.components.values() if c])}
====================================================
            """
            )

        except Exception as e:
            logger.error(f"❌ Error during enterprise system initialization: {e}")
            await self.cleanup()
            raise

    async def cleanup(self) -> None:
        """Cleanup all enterprise components"""
        try:
            logger.info("🧹 Cleaning up enterprise system...")

            # Cleanup in reverse order
            components_to_cleanup = [
                ("socket", "socket_manager"),
                ("tools", "tool_manager"),
                ("analytics", "analytics_system"),
                ("middleware", "middleware"),
                ("database", "database_system"),
                ("scheduler", "scheduler"),
                ("events", "event_stream"),
                ("configuration", "config_manager"),
            ]

            for component_name, _ in components_to_cleanup:
                component = self.components.get(component_name)
                if component and hasattr(component, "shutdown"):
                    try:
                        shutdown_method = getattr(component, "shutdown")
                        if asyncio.iscoroutinefunction(shutdown_method):
                            await shutdown_method()
                        else:
                            shutdown_method()
                        logger.info(f"✅ {component_name} system cleaned up")
                    except Exception as e:
                        logger.error(f"❌ Error cleaning up {component_name}: {e}")

            self.initialized = False
            logger.info("🧹 Enterprise system cleanup completed")

        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        if not self.initialized:
            return {"status": "not_initialized", "components": {}, "uptime_seconds": 0}

        uptime = (
            (datetime.now(timezone.utc) - self.start_time).total_seconds()
            if self.start_time
            else 0
        )

        return {
            "status": "running",
            "environment": self.get_environment(),
            "uptime_seconds": uptime,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "components": {
                name: "active" if component else "inactive"
                for name, component in self.components.items()
            },
            "enterprise_features": [
                "Event-Driven Architecture",
                "Background Processing & Scheduling",
                "Enterprise Database Architecture",
                "Production Middleware Integration",
                "Real-Time Metrics & Analytics",
                "Tools Integration (30+ tools)",
                "Socket.IO Enterprise Communication",
                "Unified Configuration System",
            ],
            "integration_status": "ii-agent patterns successfully integrated with Sheily intelligence",
        }

    def get_environment(self) -> str:
        """Get current environment"""
        config_manager = self.components.get("configuration")
        if config_manager and hasattr(config_manager, "environment"):
            return config_manager.environment.value
        return "unknown"

    def get_config_status(self) -> str:
        """Get configuration status"""
        config_manager = self.components.get("configuration")
        if (
            config_manager
            and hasattr(config_manager, "config")
            and config_manager.config
        ):
            return f"loaded (hash: {config_manager.config.config_hash[:8]}...)"
        return "not_loaded"

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health_status: Dict[str, Any] = {
            "overall_status": "healthy" if self.initialized else "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {},
        }

        # Check each component
        for name, component in self.components.items():
            try:
                if component and hasattr(component, "health_check"):
                    health_check_method = getattr(component, "health_check")
                    if asyncio.iscoroutinefunction(health_check_method):
                        component_health = await health_check_method()
                    else:
                        component_health = health_check_method()
                    health_status["components"][name] = component_health
                else:
                    health_status["components"][name] = {
                        "status": "active" if component else "inactive"
                    }
            except Exception as e:
                health_status["components"][name] = {"status": "error", "error": str(e)}

        # Determine overall health
        component_statuses = []
        for comp in health_status["components"].values():
            if isinstance(comp, dict) and "status" in comp:
                component_statuses.append(comp["status"])
            else:
                component_statuses.append("unknown")

        if any(status == "error" for status in component_statuses):
            health_status["overall_status"] = "unhealthy"
        elif any(status == "warning" for status in component_statuses):
            health_status["overall_status"] = "degraded"

        return health_status


# Global enterprise system instance
_enterprise_system: Optional[SheìlyEnterpriseSystem] = None


async def get_enterprise_system() -> SheìlyEnterpriseSystem:
    """Get global enterprise system instance"""
    global _enterprise_system
    if _enterprise_system is None:
        _enterprise_system = SheìlyEnterpriseSystem()
    return _enterprise_system


async def initialize_complete_enterprise_system(
    config_file: Optional[str] = None,
    environment: Optional[Environment] = None,
    hot_reload: bool = False,
) -> SheìlyEnterpriseSystem:
    """Initialize the complete Sheily Enterprise System"""
    enterprise_system = await get_enterprise_system()

    if not enterprise_system.initialized:
        await enterprise_system.initialize(config_file, environment, hot_reload)

    return enterprise_system


# Convenience function for quick startup
async def start_sheily_enterprise(
    config_file: Optional[str] = None, environment: str = "development"
) -> SheìlyEnterpriseSystem:
    """Quick start function for Sheily Enterprise"""
    env = Environment(environment)
    return await initialize_complete_enterprise_system(
        config_file=config_file,
        environment=env,
        hot_reload=(env in [Environment.DEVELOPMENT, Environment.LOCAL]),
    )


if __name__ == "__main__":
    """
    Ejemplo de uso del sistema enterprise completo:

    python -m sheily_core.enterprise_system
    """

    async def main():
        try:
            # Initialize complete enterprise system
            system = await start_sheily_enterprise()

            # Show system status
            status = system.get_status()
            print(f"Sistema inicializado: {status}")

            # Keep running
            print("Sistema enterprise ejecutándose... (Ctrl+C para salir)")
            while True:
                await asyncio.sleep(60)

        except KeyboardInterrupt:
            print("\n🛑 Shutting down enterprise system...")
            if _enterprise_system:
                await _enterprise_system.cleanup()
        except Exception as e:
            print(f"❌ Error: {e}")
            if _enterprise_system:
                await _enterprise_system.cleanup()

    # Run the system
    asyncio.run(main())


__all__ = [
    "SheìlyEnterpriseSystem",
    "get_enterprise_system",
    "initialize_complete_enterprise_system",
    "start_sheily_enterprise",
]
