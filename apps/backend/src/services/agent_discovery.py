#!/usr/bin/env python3
"""
Sistema de Discovery Dinámico para Agentes Reales del Proyecto
Detección automática de agentes, servicios y capacidades actuales
"""

import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class AgentDiscoveryService:
    """
    Servicio que escanea automáticamente los agentes reales del proyecto
    y mantiene un registro actualizado de capacidades y servicios disponibles
    """

    def __init__(self):
        self.discovered_agents = []
        self.discovered_services = []
        self.agent_capabilities = {}
        self.service_capabilities = {}
        self.last_scan = None

        # Rutas de scanning automáticas
        self.scan_paths = [
            Path(__file__).parent.parent.parent / "agents",  # /agents
            Path(__file__).parent.parent.parent / "sheily_core" / "agents",  # /sheily_core/agents
            Path(__file__).parent.parent.parent / "backend" / "core",  # /backend/core
            Path(__file__).parent.parent.parent / "backend" / "services",  # /backend/services
        ]

    def scan_all_agents(self) -> Dict[str, Any]:
        """
        Escaneo completo de todos los agentes disponibles en el proyecto

        Returns:
            Dict con información completa de agentes detectados
        """
        logger.info("🔍 Iniciando escaneo completo de agentes...")

        self.discovered_agents = []
        self.agent_capabilities = {}

        # Escanear todas las rutas configuradas
        for scan_path in self.scan_paths:
            if scan_path.exists():
                logger.info(f"📂 Escaneando: {scan_path}")
                self._scan_directory(scan_path)

        # Escanear imports directos disponibles
        self._scan_available_imports()

        # Crear resumen
        summary = {
            "total_agents": len(self.discovered_agents),
            "agents": self.discovered_agents,
            "capabilities": self.agent_capabilities,
            "scan_timestamp": self.last_scan,
            "scan_paths": [str(p) for p in self.scan_paths],
        }

        logger.info(f"✅ Escaneo completado: {summary['total_agents']} agentes encontrados")
        return summary

    def _scan_directory(self, directory: Path) -> None:
        """Escanea un directorio específico en busca de agentes"""
        try:
            for file_path in directory.rglob("*.py"):
                if file_path.name.startswith("__") or file_path.name == "setup.py":
                    continue

                try:
                    # Intentar importar el módulo
                    module_name = self._path_to_module_name(file_path)
                    if module_name:
                        spec = importlib.util.spec_from_file_location(module_name, file_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)

                            # Buscar clases que parezcan agentes
                            self._analyze_module(module, file_path, module_name)

                except Exception as e:
                    logger.debug(f"Error analizando {file_path}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Error escaneando directorio {directory}: {e}")

    def _path_to_module_name(self, file_path: Path) -> Optional[str]:
        """Convierte una ruta de archivo a un nombre de módulo importable"""
        try:
            # Encontrar la raíz del proyecto
            current = file_path.parent
            parts = []

            while current != current.parent:
                if (current / "__init__.py").exists():
                    parts.insert(0, current.name)
                elif current.name in ["agents", "sheily_core", "backend", "core", "services"]:
                    parts.insert(0, current.name)
                    break
                current = current.parent

            parts.append(file_path.stem)
            return ".".join(parts)

        except Exception:
            return None

    def _analyze_module(self, module, file_path: Path, module_name: str) -> None:
        """Analiza un módulo en busca de agentes"""
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and not name.startswith("_"):
                # Verificar si parece ser un agente
                if self._is_agent_class(obj, name):
                    agent_info = self._extract_agent_info(obj, name, module_name, file_path)
                    if agent_info:
                        self.discovered_agents.append(agent_info)
                        logger.info(f"🔍 Agente detectado: {agent_info['name']}")

    def _is_agent_class(self, cls, name: str) -> bool:
        """Determina si una clase parece ser un agente"""
        # Criterios para detectar agentes (más amplios)
        agent_indicators = [
            "agent" in name.lower(),
            "Agent" in name,
            "Trainer" in name,  # Para AdvancedAgentTrainer
            hasattr(cls, "execute") or hasattr(cls, "run") or hasattr(cls, "process"),
            hasattr(cls, "train") or hasattr(cls, "optimize") or hasattr(cls, "improve"),
            hasattr(cls, "__doc__") and cls.__doc__ and (
                "agent" in cls.__doc__.lower() or
                "entrenamiento" in cls.__doc__.lower() or
                "ai" in cls.__doc__.lower() or
                "inteligente" in cls.__doc__.lower() or
                "autonomous" in cls.__doc__.lower() or
                "sistema" in cls.__doc__.lower()
            )
        ]

        # Al menos 2 indicadores positivos para considerar que es agente
        return sum(agent_indicators) >= 2

    def _extract_agent_info(self, cls, name: str, module_name: str, file_path: Path) -> Optional[Dict]:
        """Extrae información detallada de un agente"""
        try:
            info = {
                "id": f"{module_name}.{name}",
                "name": name,
                "type": name.replace("Agent", "").replace("Controller", "").replace("Manager", ""),
                "module": module_name,
                "file_path": str(file_path),
                "description": cls.__doc__.strip() if cls.__doc__ else f"Agent {name}",
                "capabilities": [],
                "methods": [],
                "status": "detected",  # Cambiará basado en verificación posterior
                "last_seen": self.last_scan,
            }

            # Extraer métodos disponibles
            methods = [m for m in dir(cls) if not m.startswith("_") and callable(getattr(cls, m, None))]
            info["methods"] = methods[:10]  # Limitar a 10 métodos por agente

            # Determinar capabilities basado en métodos
            info["capabilities"] = self._infer_capabilities(methods, name)

            return info

        except Exception as e:
            logger.debug(f"Error extrayendo info de {name}: {e}")
            return None

    def _infer_capabilities(self, methods: List[str], class_name: str) -> List[str]:
        """Infiero capacidades basado en métodos disponibles"""
        capabilities = []

        # Mapas de capacidades por nombre de método
        capability_maps = {
            "learning": ["train", "learn", "optimize", "improve"],
            "monitoring": ["monitor", "watch", "check", "health", "status"],
            "execution": ["execute", "run", "process", "handle"],
            "communication": ["chat", "message", "communicate", "send"],
            "analysis": ["analyze", "evaluate", "assess", "review"],
            "coordination": ["coordinate", "manage", "orchestrate"],
            "caching": ["cache", "store", "retrieve", "memory"],
            "security": ["secure", "validate", "protect", "auth"],
            "backup": ["backup", "restore", "save", "recover"],
            "optimization": ["optimize", "tune", "performance", "speed"],
        }

        # Determinar capabilities
        found_capabilities = set()
        lowercase_methods = [m.lower() for m in methods]

        for cap, keywords in capability_maps.items():
            if any(keyword in " ".join(lowercase_methods) for keyword in keywords):
                found_capabilities.add(cap)

        # Inferir de nombre de clase
        class_lower = class_name.lower()
        if "training" in class_lower:
            found_capabilities.add("learning")
        if "monitor" in class_lower:
            found_capabilities.add("monitoring")
        if "coord" in class_lower or "controller" in class_lower:
            found_capabilities.add("coordination")

        return list(found_capabilities)

    def _scan_available_imports(self) -> None:
        """Escanea imports disponibles que podrían ser agentes"""
        known_agent_modules = [
            "agents.advanced_training_system",
            "agents.constitutional_evaluator",
            "agents.reflexion_agent",
            "agents.toolformer_agent",
            "sheily_core.agents.autonomous_system_controller",
            "sheily_core.agents.specialized.advanced_quantitative_agent",
            "sheily_core.agents.specialized.finance_agent",
        ]

        for module_name in known_agent_modules:
            try:
                module = importlib.import_module(module_name)
                self._analyze_module(module, Path(module.__file__), module_name)
            except (ImportError, AttributeError):
                continue

    def get_agent_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de agentes detectados"""
        if not self.discovered_agents:
            self.scan_all_agents()

        capability_counts = {}
        type_counts = {}
        status_counts = {}

        for agent in self.discovered_agents:
            # Contar por capabilities
            for cap in agent.get("capabilities", []):
                capability_counts[cap] = capability_counts.get(cap, 0) + 1

            # Contar por tipo
            agent_type = agent.get("type", "unknown")
            type_counts[agent_type] = type_counts.get(agent_type, 0) + 1

            # Contar por status
            status = agent.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_agents": len(self.discovered_agents),
            "capabilities": capability_counts,
            "types": type_counts,
            "statuses": status_counts,
            "agents": self.discovered_agents,
        }

    def discover_services(self) -> Dict[str, Any]:
        """Descubre servicios además de agentes"""
        self.discovered_services = []

        # Escanear servicios en backend/services
        services_path = Path(__file__).parent.parent.parent / "backend" / "services"
        if services_path.exists():
            for file_path in services_path.glob("*.py"):
                if not file_path.name.startswith("__"):
                    try:
                        module_name = f"backend.services.{file_path.stem}"
                        module = importlib.import_module(module_name)

                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) and not name.startswith("_"):
                                if "Service" in name and hasattr(obj, "__doc__"):
                                    service_info = {
                                        "id": f"{module_name}.{name}",
                                        "name": name,
                                        "type": "service",
                                        "module": module_name,
                                        "description": obj.__doc__.strip() if obj.__doc__ else f"Service {name}",
                                        "file_path": str(file_path),
                                    }
                                    self.discovered_services.append(service_info)

                    except Exception as e:
                        logger.debug(f"Error escaneando servicio {file_path}: {e}")

        return {
            "total_services": len(self.discovered_services),
            "services": self.discovered_services,
        }

    def get_system_overview(self) -> Dict[str, Any]:
        """Obtiene una visión general completa del sistema real"""
        agent_stats = self.get_agent_stats()
        service_stats = self.discover_services()

        return {
            "timestamp": self.last_scan,
            "agents": agent_stats,
            "services": service_stats,
            "total_components": agent_stats["total_agents"] + service_stats["total_services"],
            "system_health": "operational" if agent_stats["total_agents"] > 0 else "limited",
            "capabilities_available": list(agent_stats["capabilities"].keys()),
        }


# Instancia global para uso en toda la aplicación
agent_discovery = AgentDiscoveryService()


def get_real_agent_count() -> int:
    """Función de conveniencia para obtener conteo real de agentes"""
    return len(agent_discovery.get_agent_stats()["agents"])


def get_real_agent_list() -> List[Dict]:
    """Función de conveniencia para obtener lista de agentes reales"""
    return agent_discovery.get_agent_stats()["agents"]


if __name__ == "__main__":
    print("🔍 Agent Discovery Service Demo")
    print("=" * 50)

    # Escanear agentes
    summary = agent_discovery.scan_all_agents()
    print(f"📊 Agentes encontrados: {summary['total_agents']}")

    # Mostrar agentes
    for agent in summary['agents'][:10]:  # Primeros 10
        print(f"🤖 {agent['name']}: {agent['description'][:60]}...")
        print(f"   📁 {agent['module']}")
        print(f"   ⚡ Capacidades: {', '.join(agent['capabilities'])}")
        print()

    # Estadísticas
    stats = agent_discovery.get_agent_stats()
    print("📈 Estadísticas:")
    print(f"   Total agentes: {stats['total_agents']}")
    print(f"   Tipos: {stats['types']}")
    print(f"   Capacidades principales: {stats['capabilities']}")

    print("\n🎯 System Overview:")
    overview = agent_discovery.get_system_overview()
    print(f"   Componentes totales: {overview['total_components']}")
    print(f"   Estado: {overview['system_health']}")
    print(f"   Capacidades: {len(overview['capabilities_available'])} disponibles")
