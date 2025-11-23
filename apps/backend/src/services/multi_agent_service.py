#!/usr/bin/env python3
"""
MULTI-AGENT AI SERVICE - Sistema de Agentes Coordinados
======================================================

Sistema completo de agentes AI que colaboran:
- Constitutional Evaluator - Evaluación ética y constitucional
- Reflexion Agent - Mejora iterativa con reflexión
- Toolformer Agent - Herramientas especializadas
- Coordinator Agent - Orquestación inteligente
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from services.simple_rag import RealRAGService

from packages.training_system.src.agents.constitutional_evaluator import ConstitutionalEvaluator
from packages.training_system.src.agents.reflexion_agent import ReflexionAgent
from packages.training_system.src.agents.toolformer_agent import ToolformerAgent

logger = logging.getLogger(__name__)


class CoordinatorAgent:
    """
    Agente coordinador que orquesta múltiples agentes especializados
    """

    def __init__(self, rag_service: RealRAGService):
        self.rag_service = rag_service
        self.agents = {}
        self.task_history = []
        self.coordination_rules = self._load_coordination_rules()

    def _load_coordination_rules(self) -> Dict[str, Any]:
        """Reglas de coordinación entre agentes"""
        return {
            "ethical_review": {
                "agent": "constitutional_evaluator",
                "trigger": ["ética", "moral", "seguridad", "política"],
                "priority": "high",
            },
            "iterative_improvement": {
                "agent": "reflexion_agent",
                "trigger": ["mejorar", "optimizar", "refinar", "iterar"],
                "priority": "medium",
            },
            "tool_usage": {
                "agent": "toolformer_agent",
                "trigger": ["herramienta", "calcular", "buscar", "analizar"],
                "priority": "medium",
            },
            "complex_reasoning": {
                "agents": ["constitutional_evaluator", "reflexion_agent"],
                "trigger": ["complejo", "difícil", "desafiante"],
                "priority": "high",
            },
        }

    async def coordinate_task(
        self, task: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Coordina una tarea entre múltiples agentes
        """
        start_time = datetime.now()
        coordination_result = {
            "task": task,
            "agents_used": [],
            "results": {},
            "coordination_decisions": [],
            "final_answer": "",
            "execution_time": 0,
            "status": "processing",
        }

        try:
            # 1. Análisis inicial de la tarea
            task_analysis = await self._analyze_task(task)
            coordination_result["task_analysis"] = task_analysis

            # 2. Selección de agentes basada en reglas
            selected_agents = self._select_agents(task, task_analysis)
            coordination_result["selected_agents"] = selected_agents

            # 3. Ejecución coordinada
            agent_results = {}
            for agent_name in selected_agents:
                if agent_name in self.agents:
                    agent = self.agents[agent_name]
                    result = await agent.process(task, context or {})

                    agent_results[agent_name] = result
                    coordination_result["agents_used"].append(agent_name)

                    # Registrar decisiones de coordinación
                    coordination_result["coordination_decisions"].append(
                        {
                            "agent": agent_name,
                            "decision": "executed",
                            "reason": f"Matched rules for {task_analysis.get('complexity', 'medium')} complexity",
                        }
                    )

            coordination_result["results"] = agent_results

            # 4. Síntesis de resultados
            final_answer = await self._synthesize_results(agent_results, task)
            coordination_result["final_answer"] = final_answer

            coordination_result["status"] = "completed"
            coordination_result["execution_time"] = (
                datetime.now() - start_time
            ).total_seconds()

            # 5. Guardar en historial
            self.task_history.append(coordination_result)

        except Exception as e:
            logger.error(f"Error en coordinación de agentes: {e}")
            coordination_result["status"] = "error"
            coordination_result["error"] = str(e)

        return coordination_result

    async def _analyze_task(self, task: str) -> Dict[str, Any]:
        """Análisis inteligente de la tarea"""
        # Usar RAG para análisis contextual
        rag_results = await self.rag_service.search(task, top_k=3)

        analysis = {
            "complexity": "simple",
            "ethical_concerns": False,
            "tool_requirements": [],
            "rag_context": len(rag_results.get("results", [])),
            "keywords": [],
        }

        # Análisis de complejidad
        if len(task) > 200 or any(
            word in task.lower() for word in ["complejo", "difícil", "avanzado"]
        ):
            analysis["complexity"] = "high"
        elif len(task) > 50:
            analysis["complexity"] = "medium"

        # Detección de preocupaciones éticas
        ethical_keywords = ["seguridad", "ética", "privacidad", "legal", "política"]
        analysis["ethical_concerns"] = any(
            keyword in task.lower() for keyword in ethical_keywords
        )

        # Requisitos de herramientas
        if "calcular" in task.lower():
            analysis["tool_requirements"].append("calculator")
        if "buscar" in task.lower():
            analysis["tool_requirements"].append("search")

        return analysis

    def _select_agents(self, task: str, analysis: Dict[str, Any]) -> List[str]:
        """Selecciona agentes basándose en el análisis"""
        selected = ["toolformer_agent"]  # Siempre disponible

        # Agregar evaluador constitucional si hay preocupaciones éticas
        if analysis.get("ethical_concerns", False):
            selected.append("constitutional_evaluator")

        # Agregar agente de reflexión para tareas complejas
        if analysis.get("complexity") in ["medium", "high"]:
            selected.append("reflexion_agent")

        return list(set(selected))  # Remover duplicados

    async def _synthesize_results(
        self, agent_results: Dict[str, Any], original_task: str
    ) -> str:
        """Sintetiza resultados de múltiples agentes"""
        if not agent_results:
            return "No se pudieron obtener resultados de los agentes."

        # Si solo un agente, devolver su resultado
        if len(agent_results) == 1:
            agent_name = list(agent_results.keys())[0]
            return agent_results[agent_name].get(
                "response", agent_results[agent_name].get("result", "Sin respuesta")
            )

        # Síntesis inteligente para múltiples agentes
        synthesis_parts = []

        # Respuesta constitucional (prioridad alta)
        if "constitutional_evaluator" in agent_results:
            const_result = agent_results["constitutional_evaluator"]
            synthesis_parts.append(
                f"Evaluación ética: {const_result.get('assessment', 'Aprobado')}"
            )

        # Mejora iterativa
        if "reflexion_agent" in agent_results:
            reflex_result = agent_results["reflexion_agent"]
            synthesis_parts.append(
                f"Mejora sugerida: {reflex_result.get('improvement', 'Ninguna sugerencia')}"
            )

        # Herramientas utilizadas
        if "toolformer_agent" in agent_results:
            tool_result = agent_results["toolformer_agent"]
            synthesis_parts.append(
                f"Resultado de herramientas: {tool_result.get('output', 'Sin resultados')}"
            )

        final_synthesis = " | ".join(synthesis_parts)

        return f"Síntesis multi-agente para '{original_task}': {final_synthesis}"


class MultiAgentService:
    """
    Servicio principal de multi-agent AI
    """

    def __init__(self, rag_service: RealRAGService):
        self.rag_service = rag_service
        self.coordinator = CoordinatorAgent(rag_service)

        # Inicializar agentes especializados
        self._initialize_agents()

        logger.info(
            "🧠 Multi-Agent AI Service inicializado con 4 agentes especializados"
        )

    def _initialize_agents(self):
        """Inicializa todos los agentes especializados"""
        try:
            # Agente constitucional para evaluación ética
            self.coordinator.agents["constitutional_evaluator"] = (
                ConstitutionalEvaluator(rag_service=self.rag_service)
            )

            # Agente de reflexión para mejora iterativa
            self.coordinator.agents["reflexion_agent"] = ReflexionAgent(
                max_iterations=3, rag_service=self.rag_service
            )

            # Agente de herramientas
            self.coordinator.agents["toolformer_agent"] = ToolformerAgent(
                rag_service=self.rag_service
            )

            logger.info(
                "✅ Todos los agentes especializados inicializados correctamente"
            )

        except Exception as e:
            logger.error(f"❌ Error inicializando agentes: {e}")

    async def process_query(
        self, query: str, user_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Procesa una consulta usando el sistema multi-agent
        """
        # Agregar contexto RAG
        context = user_context or {}
        context["rag_search"] = await self.rag_service.search(query, top_k=2)

        # Coordinar con múltiples agentes
        result = await self.coordinator.coordinate_task(query, context)

        # Agregar métricas
        result["multi_agent_metrics"] = {
            "agents_coordinated": len(result.get("agents_used", [])),
            "execution_time": result.get("execution_time", 0),
            "coordination_decisions": len(result.get("coordination_decisions", [])),
        }

        return result

    async def get_agent_status(self) -> Dict[str, Any]:
        """Estado de todos los agentes"""
        agents_status = {}
        agents_count = len(self.coordinator.agents)

        for agent_name, agent in self.coordinator.agents.items():
            try:
                status = (
                    await agent.health_check()
                    if hasattr(agent, "health_check")
                    else {"status": "unknown"}
                )
                agents_status[agent_name] = status
            except:
                agents_status[agent_name] = {"status": "error"}

        return {
            "total_agents": agents_count,
            "active_agents": sum(
                1 for s in agents_status.values() if s.get("status") == "healthy"
            ),
            "agents_detail": agents_status,
            "coordination_history": len(self.coordinator.task_history),
            "service_status": "healthy" if agents_count > 0 else "error",
        }

    def get_coordination_stats(self) -> Dict[str, Any]:
        """Estadísticas de coordinación"""
        history = self.coordinator.task_history[-10:]  # Últimas 10 tareas

        stats = {
            "total_coordinations": len(self.coordinator.task_history),
            "recent_tasks": len(history),
            "avg_execution_time": (
                sum(h.get("execution_time", 0) for h in history) / len(history)
                if history
                else 0
            ),
            "most_used_agent": None,
            "task_completion_rate": (
                sum(1 for h in history if h.get("status") == "completed") / len(history)
                if history
                else 0
            ),
        }

        # Agente más usado
        if history:
            agents_used = {}
            for task in history:
                for agent in task.get("agents_used", []):
                    agents_used[agent] = agents_used.get(agent, 0) + 1

            if agents_used:
                stats["most_used_agent"] = max(agents_used.items(), key=lambda x: x[1])[
                    0
                ]

        return stats


# =============================================================================
# DEMO Y TESTING DEL MULTI-AGENT SERVICE
# =============================================================================


async def demo_multi_agent_service():
    """Demo del sistema multi-agent AI"""
    print("🧠 MULTI-AGENT AI SERVICE DEMO")
    print("=" * 45)

    # Inicializar servicios
    rag_service = RealRAGService()
    multi_agent_service = MultiAgentService(rag_service)

    # Indexar documentos de ejemplo para el RAG
    print("📚 Indexando documentos para RAG...")
    docs = [
        "La ética en IA es fundamental para el desarrollo responsable",
        "Los algoritmos de aprendizaje automático requieren validación",
        "La privacidad de datos es un derecho humano fundamental",
        "La inteligencia artificial debe beneficiar a la humanidad",
    ]
    await rag_service.index_documents(docs)

    # Ver estado de agentes
    print("\n🤖 Estado de agentes:")
    agent_status = await multi_agent_service.get_agent_status()
    print(f"   Agentes totales: {agent_status['total_agents']}")
    print(f"   Agentes activos: {agent_status['active_agents']}")

    # Procesar consulta compleja
    print("\n🧠 Procesando consulta con múltiples agentes...")
    complex_query = (
        "¿Cómo debería la IA manejar datos personales respetando la privacidad?"
    )

    result = await multi_agent_service.process_query(complex_query)
    print(f"   Estado: {result['status']}")
    print(f"   Agentes usados: {len(result.get('agents_used', []))}")
    print(f"   Tiempo total: {result.get('execution_time', 0):.2f}s")
    print(f"   Respuesta: {result.get('final_answer', '')[:100]}...")

    # Consulta ética
    print("\n⚖️ Procesando consulta con preocupación ética...")
    ethical_query = "¿Es ético usar IA para vigilancia masiva?"

    ethical_result = await multi_agent_service.process_query(ethical_query)
    print(f"   Agentes coordinados: {len(ethical_result.get('agents_used', []))}")
    print(f"   Respuesta ética: {ethical_result.get('final_answer', '')[:100]}...")

    # Estadísticas de coordinación
    print("\n📊 Estadísticas de coordinación:")
    coord_stats = multi_agent_service.get_coordination_stats()
    print(f"   Coordinaciones totales: {coord_stats['total_coordinations']}")
    print(f"   Tiempo promedio: {coord_stats.get('avg_execution_time', 0):.2f}s")

    print("\n🧠 MULTI-AGENT AI OPERATIVO")
    print("   ✅ Coordinación inteligente entre agentes")
    print("   ✅ Evaluación ética automática")
    print("   ✅ Mejora iterativa con reflexión")
    print("   ✅ Herramientas especializadas")
    print("   ✅ Síntesis multi-perspective")


# Configurar para testing
if __name__ == "__main__":
    import asyncio

    asyncio.run(demo_multi_agent_service())
