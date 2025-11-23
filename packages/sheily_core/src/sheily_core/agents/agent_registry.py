#!/usr/bin/env python3
"""
Agent Registry - Sistema base de registro de agentes
"""

import logging
import time
from typing import Any, Dict, List, Optional

class AgentRegistry:
    """
    Registro base para agentes del sistema.
    Mantiene un catálogo de los agentes disponibles y sus metadatos.
    """

    def __init__(self):
        self._agents: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger("Sheily.Registry")
        self._is_running = False

    async def start_registry(self):
        """Iniciar el servicio de registro"""
        self._is_running = True
        self.logger.info("Agent Registry service started")

    async def stop_registry(self):
        """Detener el servicio de registro"""
        self._is_running = False
        self.logger.info("Agent Registry service stopped")

    async def register_agent(
        self, agent: Any, agent_type: str = "generic", metadata: Dict[str, Any] = None
    ) -> str:
        """
        Registrar un nuevo agente en el sistema.
        
        Args:
            agent: Instancia del agente o referencia
            agent_type: Tipo de agente (coordinator, specialist, etc)
            metadata: Metadatos adicionales
            
        Returns:
            str: ID del agente registrado
        """
        # Generar ID si el agente no tiene uno
        agent_id = getattr(agent, "agent_id", f"agent_{int(time.time())}_{len(self._agents)}")
        
        self._agents[agent_id] = {
            "instance": agent,
            "type": agent_type,
            "metadata": metadata or {},
            "registered_at": time.time(),
            "status": "active"
        }
        
        self.logger.info(f"Registered agent: {agent_id} (Type: {agent_type})")
        return agent_id

    async def unregister_agent(self, agent_id: str) -> bool:
        """Desregistrar un agente"""
        if agent_id in self._agents:
            del self._agents[agent_id]
            self.logger.info(f"Unregistered agent: {agent_id}")
            return True
        return False

    def get_agent(self, agent_id: str) -> Optional[Any]:
        """Obtener instancia de un agente por ID"""
        agent_data = self._agents.get(agent_id)
        return agent_data["instance"] if agent_data else None

    def get_all_agents(self) -> List[Dict[str, Any]]:
        """Obtener lista de todos los agentes registrados"""
        return [
            {"id": aid, **data} 
            for aid, data in self._agents.items()
        ]
