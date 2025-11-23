#!/usr/bin/env python3
"""
MCP Plugin System - Arquitectura de Plugins para Agentes Dinámicos
==================================================================

Este módulo implementa el sistema de plugins MCP que permite expandir
dinámicamente los 47 agentes a miles de agentes especializados.
"""

import asyncio
import importlib
import inspect
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from .agents.base_agent import AgentConfig, BaseAgent

logger = logging.getLogger(__name__)


class MCPPluginSystem:
    """
    Sistema de Plugins MCP - Arquitectura Extensible para Agentes

    Permite cargar dinámicamente nuevos tipos de agentes sin modificar
    el código base del sistema MCP.
    """

    def __init__(self):
        """Inicializar sistema de plugins MCP"""
        self.plugin_registry = {}  # name -> plugin_class
        self.active_plugins = {}  # name -> plugin_instance
        self.capability_registry = {}  # capability -> plugin_name
        self.plugin_directories = []
        self.initialized = False

        # Métricas del sistema de plugins
        self.metrics = {
            "total_plugins_loaded": 0,
            "active_plugins": 0,
            "capabilities_registered": 0,
            "plugin_load_time": 0,
        }

        logger.info("🧩 MCP Plugin System inicializado")

    async def initialize_plugin_system(self) -> bool:
        """
        Inicializar el sistema de plugins MCP

        Carga todos los plugins disponibles y registra sus capacidades.
        """
        try:
            logger.info("🔌 Inicializando sistema de plugins MCP...")

            # Establecer directorios de plugins
            await self._setup_plugin_directories()

            # Cargar plugins del sistema base
            await self._load_core_plugins()

            # Escanear y cargar plugins adicionales
            await self._scan_and_load_plugins()

            # Registrar capacidades de todos los plugins
            await self._register_plugin_capabilities()

            # Verificar integridad del sistema de plugins
            await self._verify_plugin_integrity()

            self.initialized = True
            logger.info(
                f"✅ Sistema de plugins MCP inicializado - {len(self.plugin_registry)} plugins cargados"
            )

            return True

        except Exception as e:
            logger.error(f"❌ Error inicializando sistema de plugins MCP: {e}")
            return False

    async def _setup_plugin_directories(self):
        """Configurar directorios donde buscar plugins"""
        self.plugin_directories = [
            os.path.join(os.path.dirname(__file__), "agents"),  # Plugins base
            os.path.join(os.path.dirname(__file__), "plugins"),  # Plugins adicionales
            "plugins",  # Directorio de plugins del proyecto
            "custom_plugins",  # Plugins personalizados
        ]

        # Crear directorios si no existen
        for directory in self.plugin_directories[1:]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"📁 Directorio de plugins creado: {directory}")

    async def _load_core_plugins(self):
        """Cargar plugins del sistema base (47 agentes originales)"""
        logger.info("🔧 Cargando plugins del sistema base...")

        # Los 47 agentes originales se convierten en plugins base
        core_plugins = {
            # Control y Coordinación (4)
            "autonomous_system_controller": "AutonomousSystemController",
            "agent_coordinator": "AgentCoordinator",
            "api_manager": "APIManagerAgent",
            "function_manager": "FunctionManagerAgent",
            # Desarrollo y Optimización (8)
            "auto_improvement": "AutoImprovementAgent",
            "code_quality": "CodeQualityAgent",
            "core_optimization": "CoreOptimizationAgent",
            "cache_optimization": "CacheLRUManager",
            "memory_tuning": "MemoryTuningAgent",
            "resource_management": "ResourceManagementAgent",
            "performance_profiling": "PerformanceProfilingAgent",
            "testing": "TestingAgent",
            # Monitoreo y Análisis (6)
            "performance_monitor": "RealtimePerformanceMonitor",
            "metrics_collector": "MetricsCollectorAgent",
            "monitoring_manager": "MonitoringManagerAgent",
            "log_manager": "LogManagerAgent",
            "dependency_manager": "DependencyManagerAgent",
            "module_manager": "ModuleManagerAgent",
            # Seguridad (4)
            "security_hardening": "SecurityHardeningAgent",
            "security_scanner": "SecurityVulnerabilityScanner",
            "backup": "BackupAgent",
            "disaster_recovery": "DisasterRecoveryAgent",
            # Deployment (7)
            "git_manager": "GitManagerAgent",
            "docker_manager": "DockerManagerAgent",
            "kubernetes": "KubernetesAgent",
            "terraform_manager": "TerraformManagerAgent",
            "database_manager": "DatabaseManagerAgent",
            "ci_cd": "CI_CD_Agent",
            "deployment_manager": "DeploymentManagerAgent",
            # Documentación y Testing (5)
            "documentation": "DocumentationAgent",
            "documentation_manager": "DocumentationManagerAgent",
            "test_manager": "TestManagerAgent",
            "report_manager": "ReportManagerAgent",
            "script_manager": "ScriptManagerAgent",
            # Gestión Empresarial (6)
            "backend_manager": "BackendManagerAgent",
            "frontend_manager": "FrontendManagerAgent",
            "user_management": "UserManagementAgent",
            "data_manager": "DataManagerAgent",
            "model_manager": "ModelManagerAgent",
            "config_manager": "ConfigManagerAgent",
        }

        for plugin_name, class_name in core_plugins.items():
            try:
                # Intentar cargar desde sheily_core.agents
                module = importlib.import_module(
                    f"sheily_core.agents.{plugin_name}_agent"
                )
                plugin_class = getattr(module, class_name, None)

                if plugin_class and issubclass(plugin_class, BaseAgent):
                    self.plugin_registry[plugin_name] = plugin_class
                    self.metrics["total_plugins_loaded"] += 1
                    logger.info(f"✅ Plugin base cargado: {plugin_name}")
                else:
                    logger.warning(f"⚠️ Plugin base no válido: {plugin_name}")

            except ImportError:
                logger.debug(f"📦 Plugin base no encontrado: {plugin_name}")
            except Exception as e:
                logger.error(f"❌ Error cargando plugin base {plugin_name}: {e}")

    async def _scan_and_load_plugins(self):
        """Escanear y cargar plugins adicionales"""
        logger.info("🔍 Escaneando plugins adicionales...")

        for plugin_dir in self.plugin_directories:
            if not os.path.exists(plugin_dir):
                continue

            for filename in os.listdir(plugin_dir):
                if filename.endswith("_plugin.py") or filename.endswith("_agent.py"):
                    plugin_name = filename[:-3]  # remover .py

                    # Evitar duplicados con plugins base
                    if plugin_name in self.plugin_registry:
                        continue

                    try:
                        await self._load_plugin_from_file(
                            plugin_dir, filename, plugin_name
                        )
                    except Exception as e:
                        logger.error(f"❌ Error cargando plugin {plugin_name}: {e}")

    async def _load_plugin_from_file(
        self, plugin_dir: str, filename: str, plugin_name: str
    ):
        """Cargar un plugin desde un archivo"""
        file_path = os.path.join(plugin_dir, filename)

        try:
            # Importar el módulo
            spec = importlib.util.spec_from_file_location(plugin_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Buscar clases de agentes en el módulo
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseAgent)
                        and attr != BaseAgent
                    ):

                        self.plugin_registry[plugin_name] = attr
                        self.metrics["total_plugins_loaded"] += 1
                        logger.info(
                            f"✅ Plugin adicional cargado: {plugin_name} -> {attr.__name__}"
                        )
                        break

        except Exception as e:
            logger.error(f"❌ Error cargando plugin desde archivo {file_path}: {e}")

    async def _register_plugin_capabilities(self):
        """Registrar capacidades de todos los plugins cargados"""
        logger.info("📋 Registrando capacidades de plugins...")

        for plugin_name, plugin_class in self.plugin_registry.items():
            try:
                # Instanciar plugin temporalmente para obtener capacidades
                temp_config = AgentConfig(
                    name=f"temp_{plugin_name}",
                    enabled=False,  # No activar realmente
                    interval=30,
                    max_retries=1,
                    auto_restart=False,
                )

                temp_plugin = plugin_class(temp_config)

                # Obtener capacidades del plugin
                capabilities = getattr(temp_plugin, "capabilities", [])
                if not capabilities:
                    # Inferir capacidades desde métodos del plugin
                    capabilities = await self._infer_capabilities_from_plugin(
                        temp_plugin
                    )

                # Registrar capacidades
                for capability in capabilities:
                    if capability not in self.capability_registry:
                        self.capability_registry[capability] = []
                    self.capability_registry[capability].append(plugin_name)

                self.metrics["capabilities_registered"] += len(capabilities)
                logger.info(
                    f"📋 Plugin {plugin_name}: {len(capabilities)} capacidades registradas"
                )

            except Exception as e:
                logger.error(f"❌ Error registrando capacidades de {plugin_name}: {e}")

    async def _infer_capabilities_from_plugin(self, plugin: BaseAgent) -> List[str]:
        """Inferir capacidades desde los métodos del plugin"""
        capabilities = []

        # Mapear métodos a capacidades
        method_capability_map = {
            "optimize_code": "code_optimization",
            "scan_security": "security_scanning",
            "deploy_container": "docker_management",
            "manage_git": "git_management",
            "generate_docs": "documentation_generation",
            "run_tests": "testing_automation",
            "monitor_performance": "performance_monitoring",
            "backup_data": "backup_management",
            "optimize_cache": "cache_management",
            "manage_resources": "resource_management",
        }

        for method_name in dir(plugin):
            if (
                method_name.startswith("execute_")
                or method_name in method_capability_map
            ):
                capability = method_capability_map.get(
                    method_name, method_name.replace("execute_", "")
                )
                if capability not in capabilities:
                    capabilities.append(capability)

        return capabilities

    async def _verify_plugin_integrity(self):
        """Verificar integridad del sistema de plugins"""
        logger.info("🔍 Verificando integridad del sistema de plugins...")

        # Verificar que todos los plugins tienen capacidades
        plugins_without_capabilities = []
        for plugin_name in self.plugin_registry.keys():
            plugin_capabilities = [
                cap
                for cap, plugins in self.capability_registry.items()
                if plugin_name in plugins
            ]
            if not plugin_capabilities:
                plugins_without_capabilities.append(plugin_name)

        if plugins_without_capabilities:
            logger.warning(
                f"⚠️ Plugins sin capacidades registradas: {plugins_without_capabilities}"
            )

        # Verificar métricas
        logger.info(
            f"📊 Verificación completada: {len(self.plugin_registry)} plugins, {len(self.capability_registry)} tipos de capacidades"
        )

    # ============================================
    # API DEL SISTEMA DE PLUGINS
    # ============================================

    async def create_agent_from_plugin(
        self, plugin_name: str, agent_config: dict = None
    ) -> dict:
        """
        Crear un agente desde un plugin registrado

        Permite expandir dinámicamente de 47 a N agentes.
        """
        try:
            if plugin_name not in self.plugin_registry:
                return {"error": f"Plugin no encontrado: {plugin_name}"}

            # Verificar si ya existe
            if plugin_name in self.active_plugins:
                return {"error": f"Plugin ya activo: {plugin_name}"}

            # Crear configuración
            config = agent_config or {
                "name": plugin_name,
                "enabled": True,
                "interval": 30,
                "max_retries": 3,
                "auto_restart": True,
                "learning_enabled": True,
            }

            agent_config_obj = AgentConfig(**config)
            plugin_class = self.plugin_registry[plugin_name]

            # Instanciar y activar plugin
            plugin_instance = plugin_class(agent_config_obj)
            success = await plugin_instance.initialize()

            if not success:
                return {"error": f"Error inicializando plugin: {plugin_name}"}

            start_success = await plugin_instance.start()
            if not start_success:
                return {"error": f"Error iniciando plugin: {plugin_name}"}

            # Registrar como activo
            self.active_plugins[plugin_name] = plugin_instance
            self.metrics["active_plugins"] += 1

            logger.info(f"🎯 Agente creado desde plugin: {plugin_name}")

            return {
                "success": True,
                "plugin_name": plugin_name,
                "agent_name": plugin_name,
                "capabilities": self.capability_registry.get(plugin_name, []),
                "status": "active",
            }

        except Exception as e:
            logger.error(f"❌ Error creando agente desde plugin {plugin_name}: {e}")
            return {"error": str(e)}

    async def get_available_plugins(self) -> dict:
        """Obtener lista de plugins disponibles"""
        return {
            "available_plugins": list(self.plugin_registry.keys()),
            "total_plugins": len(self.plugin_registry),
            "active_plugins": list(self.active_plugins.keys()),
            "capabilities": self.capability_registry,
            "metrics": self.metrics,
        }

    async def get_plugins_by_capability(self, capability: str) -> List[str]:
        """Obtener plugins que ofrecen una capacidad específica"""
        return self.capability_registry.get(capability, [])

    async def load_custom_plugin(self, plugin_code: str, plugin_name: str) -> dict:
        """
        Cargar un plugin personalizado desde código

        Permite crear agentes completamente nuevos en runtime.
        """
        try:
            # Crear archivo temporal para el plugin
            plugin_dir = "custom_plugins"
            if not os.path.exists(plugin_dir):
                os.makedirs(plugin_dir)

            plugin_file = os.path.join(plugin_dir, f"{plugin_name}_plugin.py")

            # Escribir código del plugin
            with open(plugin_file, "w") as f:
                f.write(plugin_code)

            # Intentar cargar el plugin
            success = await self._load_plugin_from_file(
                plugin_dir, f"{plugin_name}_plugin.py", plugin_name
            )

            if success and plugin_name in self.plugin_registry:
                # Registrar capacidades
                await self._register_plugin_capabilities()

                return {
                    "success": True,
                    "plugin_name": plugin_name,
                    "message": f"Plugin personalizado cargado: {plugin_name}",
                }
            else:
                return {"error": f"Error cargando plugin personalizado: {plugin_name}"}

        except Exception as e:
            logger.error(f"❌ Error cargando plugin personalizado {plugin_name}: {e}")
            return {"error": str(e)}

    async def unload_plugin(self, plugin_name: str) -> dict:
        """Descargar un plugin"""
        try:
            if plugin_name not in self.active_plugins:
                return {"error": f"Plugin no activo: {plugin_name}"}

            # Detener plugin
            plugin = self.active_plugins[plugin_name]
            await plugin.stop()

            # Remover de registros
            del self.active_plugins[plugin_name]
            self.metrics["active_plugins"] -= 1

            # Remover del registro de plugins (opcional)
            if plugin_name in self.plugin_registry:
                del self.plugin_registry[plugin_name]

            logger.info(f"🗑️ Plugin descargado: {plugin_name}")

            return {
                "success": True,
                "plugin_name": plugin_name,
                "message": f"Plugin descargado exitosamente",
            }

        except Exception as e:
            logger.error(f"❌ Error descargando plugin {plugin_name}: {e}")
            return {"error": str(e)}

    async def get_plugin_system_status(self) -> dict:
        """Obtener estado completo del sistema de plugins"""
        return {
            "plugin_system": {
                "initialized": self.initialized,
                "total_plugins_registered": len(self.plugin_registry),
                "active_plugins": len(self.active_plugins),
                "capabilities_registered": len(self.capability_registry),
            },
            "metrics": self.metrics,
            "active_plugins": list(self.active_plugins.keys()),
            "available_plugins": list(self.plugin_registry.keys()),
            "capability_matrix": self.capability_registry,
            "timestamp": datetime.now().isoformat(),
        }

    async def cleanup_plugin_system(self):
        """Limpiar recursos del sistema de plugins"""
        try:
            # Detener todos los plugins activos
            for plugin_name, plugin in self.active_plugins.items():
                try:
                    await plugin.stop()
                except Exception as e:
                    logger.error(f"❌ Error deteniendo plugin {plugin_name}: {e}")

            self.active_plugins.clear()
            logger.info("🧹 Sistema de plugins MCP limpiado")

        except Exception as e:
            logger.error(f"❌ Error limpiando sistema de plugins: {e}")


# Instancia global del sistema de plugins
_plugin_system: Optional[MCPPluginSystem] = None


async def get_mcp_plugin_system() -> MCPPluginSystem:
    """Obtener instancia del sistema de plugins MCP"""
    global _plugin_system

    if _plugin_system is None:
        _plugin_system = MCPPluginSystem()
        await _plugin_system.initialize_plugin_system()

    return _plugin_system


async def cleanup_mcp_plugin_system():
    """Limpiar el sistema de plugins MCP"""
    global _plugin_system

    if _plugin_system:
        await _plugin_system.cleanup_plugin_system()
        _plugin_system = None
