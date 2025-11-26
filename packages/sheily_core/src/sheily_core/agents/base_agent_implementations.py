#!/usr/bin/env python3
"""
Implementaciones concretas de BaseAgent - Agentes funcionales reales
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import psutil

from .base_agent import AgentConfig, BaseAgent


class SystemHealthAgent(BaseAgent):
    """Agente que monitorea la salud del sistema"""

    def __init__(self):
        config = AgentConfig(
            name="system_health", agent_type="monitoring", log_level="INFO"
        )
        super().__init__(config)

    async def _initialize_agent(self):
        """Implementación requerida por BaseAgent"""
        self.logger.info("System Health Agent initialized")

    async def check_system_health(self):
        """Función específica: verificar salud del sistema"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": (disk.used / disk.total) * 100,
            "status": (
                "healthy" if cpu_percent < 80 and memory.percent < 85 else "warning"
            ),
        }


class ProcessManagerAgent(BaseAgent):
    """Agente que gestiona procesos del sistema"""

    def __init__(self):
        config = AgentConfig(
            name="process_manager", agent_type="management", log_level="INFO"
        )
        super().__init__(config)

    async def _initialize_agent(self):
        """Implementación requerida por BaseAgent"""
        self.logger.info("Process Manager Agent initialized")

    async def get_top_processes(self, limit=10):
        """Función específica: obtener procesos principales"""
        processes = []
        for proc in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_percent"]
        ):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Ordenar por CPU
        processes.sort(key=lambda x: x["cpu_percent"] or 0, reverse=True)
        return processes[:limit]


# CONVERTIR BaseAgent en agentes funcionales
system_health_agent = SystemHealthAgent()
process_manager_agent = ProcessManagerAgent()
