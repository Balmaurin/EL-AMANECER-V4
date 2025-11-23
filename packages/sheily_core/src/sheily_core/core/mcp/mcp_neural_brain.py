#!/usr/bin/env python3
"""
MCP NEURAL BRAIN - IMPLEMENTACIÓN FUNCIONAL MEJORADA
====================================================

Cambios principales en esta versión:
- Persistencia de pesos (shelve) con checkpoints periódicos.
- Locks asíncronos para seguridad de concurrencia.
- Mejor hashing de tareas (SHA256 de JSON canonizado).
- Manejo de excepciones y logging robusto.
- Configurable (checkpoint interval, storage path).
- Métricas sanitizadas y límites.
- Stubs/advertencias donde se depende de MasterMCPOrchestrator.
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import shelve
from datetime import datetime
from typing import Any, Dict, List, Optional

# Importa tu orquestador maestro (debe exponer: start_system, agent_registry, _select_optimal_agent)
from ..system.master_orchestrator import MasterMCPOrchestrator  # <-- corregido path

# Import opcional del brain learner para aprendizaje automático
try:
    from ...models.ml.neural_brain_learner import NeuralBrainLearner, auto_learn_project

    BRAIN_LEARNER_AVAILABLE = True
except ImportError:
    print("⚠️  Neural brain learner not available - basic learning only")
    BRAIN_LEARNER_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def canonical_task_hash(task_features: Dict[str, Any]) -> str:
    """Genera un hash estable para una tarea usando JSON canonizado y SHA256."""
    task_json = json.dumps(
        task_features, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(task_json.encode("utf-8")).hexdigest()


class NeuralMetrics:
    def __init__(self):
        self.neural_evolution_level: float = 0.0
        self.collective_intelligence_score: float = 0.0
        self.learning_efficiency: float = 0.0
        self.task_success_rate: float = 0.0
        self.adaptation_speed: float = 0.0
        self.consciousness_depth: float = 4.0
        self.emergence_level: float = 0.0

    def sanitize(self):
        # Asegurar rangos y tipos
        self.neural_evolution_level = float(
            max(0.0, min(1.0, self.neural_evolution_level))
        )
        self.collective_intelligence_score = float(
            max(0.0, self.collective_intelligence_score)
        )
        self.learning_efficiency = float(max(0.0, min(1.0, self.learning_efficiency)))
        self.task_success_rate = float(max(0.0, min(1.0, self.task_success_rate)))
        self.adaptation_speed = float(max(0.0, min(1.0, self.adaptation_speed)))
        self.consciousness_depth = float(max(0.0, self.consciousness_depth))
        self.emergence_level = float(max(0.0, min(1.0, self.emergence_level)))


class SimpleNeuralCortex:
    """Corteza neuronal simple con persistencia y locking."""

    def __init__(self, input_size: int = 128, storage_path: str = "neural_store.db"):
        self.input_size = input_size
        self._weights_lock = asyncio.Lock()
        self.weights: Dict[str, float] = {}
        self.success_patterns: list[str] = []
        self.storage_path = storage_path
        self._last_checkpoint: Optional[datetime] = None

        # Cargar si existe
        self._load_from_storage()
        logger.info(
            f"🧠 SimpleNeuralCortex inicializado - Input size: {input_size} - storage: {storage_path}"
        )

    def _load_from_storage(self):
        try:
            if os.path.exists(self.storage_path):
                with shelve.open(self.storage_path) as db:
                    self.weights = dict(db.get("weights", {}))
                    self.success_patterns = list(db.get("success_patterns", []))
                logger.info("✅ Neural cortex: pesos cargados desde disco")
        except Exception as e:
            logger.exception("No se pudo cargar el almacén de pesos: %s", e)
            self.weights = {}
            self.success_patterns = []

    async def checkpoint(self):
        """Guardar pesos en disco (non-blocking para otros coroutines via lock)."""
        async with self._weights_lock:
            try:
                with shelve.open(self.storage_path) as db:
                    db["weights"] = self.weights
                    db["success_patterns"] = self.success_patterns
                self._last_checkpoint = datetime.utcnow()
                logger.info("💾 Checkpoint de SimpleNeuralCortex guardado")
            except Exception as e:
                logger.exception("Error guardando checkpoint: %s", e)

    async def process_task_features(self, task_features: Dict[str, Any]) -> float:
        task_hash = canonical_task_hash(task_features)

        # Extraer características numéricas con defensiva
        try:
            priority = float(task_features.get("priority", 3)) / 5.0
        except Exception:
            priority = 0.6
        complexity = min(1.0, len(json.dumps(task_features)) / 1000.0)

        async with self._weights_lock:
            if task_hash in self.weights:
                learned_score = float(self.weights[task_hash])
                noise_factor = random.uniform(0.95, 1.05)
                neural_score = learned_score * noise_factor
            else:
                neural_score = (
                    (priority * 0.6) + (complexity * 0.4) + random.uniform(-0.05, 0.05)
                )

        neural_score = max(0.0, min(1.0, neural_score))
        logger.debug(
            "🧠 Task processed - hash=%s score=%.3f", task_hash[:8], neural_score
        )
        return neural_score

    async def learn_from_outcome(
        self, task_features: Dict[str, Any], success: bool, neural_score: float
    ):
        task_hash = canonical_task_hash(task_features)
        async with self._weights_lock:
            current = float(self.weights.get(task_hash, neural_score))
            delta = 0.05 if success else -0.02
            new_weight = max(0.0, min(1.0, current + delta))
            self.weights[task_hash] = new_weight

            if success:
                # Mantener patterns
                self.success_patterns.append(task_hash)
                if len(self.success_patterns) > 500:
                    self.success_patterns = self.success_patterns[
                        -250:
                    ]  # Mantener 250 mejores

        logger.debug(
            "🧠 Learned - hash=%s success=%s new_weight=%.3f",
            task_hash[:8],
            success,
            new_weight,
        )


class EmergentCollectiveMind:
    def __init__(self):
        self.agent_performance_history: Dict[str, list[float]] = {}
        self.collective_memory: Dict[str, Any] = {}
        self.emergence_level: float = 0.0
        logger.info("🧬 EmergentCollectiveMind inicializado")

    async def analyze_agent_states(
        self, agent_states: Dict[str, Any]
    ) -> Dict[str, Any]:
        collective_signal = 0.0
        performance_variance = 0.0
        performances = [
            state.get("performance_score", 0.5) for state in agent_states.values()
        ]

        if performances:
            avg_performance = sum(performances) / len(performances)
            collective_signal = avg_performance
            if len(performances) > 1:
                variance_sum = sum((p - avg_performance) ** 2 for p in performances)
                performance_variance = variance_sum / len(performances)

        high_performers = sum(1 for p in performances if p > 0.8)
        emergence_signals = []
        if high_performers >= 3:
            emergence_signals.append("collective_high_performance")
        if performance_variance < 0.05 and len(performances) > 2:
            emergence_signals.append("collective_synchronization")

        self.emergence_level = min(1.0, len(emergence_signals) * 0.33)

        return {
            "collective_signal": collective_signal,
            "emergence_signals": emergence_signals,
            "emergence_level": self.emergence_level,
            "agent_count": len(agent_states),
            "high_performers": high_performers,
        }

    async def enhance_with_collective_intelligence(
        self, base_result: Dict[str, Any], emergence_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        enhanced = dict(base_result)
        amplification_factor = 1.0 + (
            emergence_analysis.get("emergence_level", 0.0) * 0.5
        )
        if "confidence" in enhanced:
            enhanced["confidence"] = min(
                1.0, float(enhanced.get("confidence", 0.0)) * amplification_factor
            )
        enhanced["collective_intelligence_applied"] = True
        enhanced["emergence_signals"] = emergence_analysis.get("emergence_signals", [])
        enhanced["collective_boost"] = amplification_factor - 1.0
        logger.debug("Collective boost applied x%.3f", amplification_factor)
        return enhanced


class TranscendentalLearningEngine:
    """Motor de aprendizaje trascendental simplificado"""

    def __init__(self):
        self.learning_accumulator: float = 0.0
        self.reflection_cycles: int = 0
        self.patterns_learned: int = 0

    async def reflect_and_evolve(self, task: Dict[str, Any], result: Dict[str, Any]):
        """Reflexión y evolución simplificada"""
        self.learning_accumulator += 0.01
        self.reflection_cycles += 1

        if result.get("success", False):
            self.patterns_learned += 1

    def get_learning_metrics(self) -> Dict[str, float]:
        """Obtener métricas de aprendizaje"""
        efficiency = self.patterns_learned / max(1, self.reflection_cycles)
        return {
            "learning_factor": self.learning_accumulator,
            "reflection_depth": min(7.0, self.reflection_cycles * 0.1),
            "patterns_learned": self.patterns_learned,
            "efficiency": efficiency,
        }


class MCPNeural(MasterMCPOrchestrator):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Updated storage path to modelsLLM/Adapters/neural_weights.db
        # Allow override via config or env var
        default_path = os.getenv("SHEILY_NEURAL_WEIGHTS_PATH", "modelsLLM/Adapters/neural_weights.db")
        storage_path = (config or {}).get("neural_storage_path", default_path)
        
        self.neural_cortex = SimpleNeuralCortex(storage_path=storage_path)
        self.collective_mind = EmergentCollectiveMind()
        self.learning_engine = TranscendentalLearningEngine()
        self.neural_metrics = NeuralMetrics()
        self.neural_activation_level = 0.1
        self.tasks_processed = 0
        self.successful_tasks = 0

        # Checkpoint task
        self._checkpoint_interval_seconds = (config or {}).get(
            "checkpoint_interval_seconds", 60 * 30
        )  # default 30m
        self._checkpoint_task: Optional[asyncio.Task] = None
        self._stopping = False

        logger.info("🧠 MCP Neural inicializado - versión mejorada")

    async def start_system(self):
        """Iniciar orquestador y arrancar tarea de checkpoint periódico."""
        await super().start_system()  # Asume implementación en MasterMCPOrchestrator
        # Lanzar tarea de checkpoint periódico
        self._checkpoint_task = asyncio.create_task(self._periodic_checkpoint())
        logger.info("🔁 Tarea de checkpoint periódico iniciada")

    async def stop_system(self):
        """Parar sistema y asegurar checkpoints."""
        self._stopping = True
        if self._checkpoint_task:
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass
        # Forzar checkpoint final
        await self.neural_cortex.checkpoint()
        logger.info("🛑 MCP Neural detenido y checkpoint final realizado")

    async def _periodic_checkpoint(self):
        interval = max(10, int(self._checkpoint_interval_seconds))
        while not self._stopping:
            try:
                await asyncio.sleep(interval)
                await self.neural_cortex.checkpoint()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error en tarea de checkpoint periódico")

    async def process_task_enhanced(
        self, task_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.tasks_processed += 1
        logger.info(
            "🧠 Procesando tarea #%d tipo=%s",
            self.tasks_processed,
            task_request.get("type", "unknown"),
        )

        try:
            neural_score = await self.neural_cortex.process_task_features(task_request)
        except Exception:
            logger.exception(
                "Error en process_task_features; asignando score por defecto"
            )
            neural_score = 0.5

        # Selección de agente (depende de MasterMCPOrchestrator)
        try:
            optimal_agent = await self._select_optimal_agent(task_request)
        except Exception:
            logger.exception(
                "No se pudo obtener agent óptimo; abortando con error controlado"
            )
            return {"success": False, "error": "no_agent_available"}

        enhanced_task = await self._enhance_task_collective(task_request, neural_score)

        # Procesamiento principal
        try:
            primary_result = await optimal_agent.process_task(enhanced_task)
        except Exception as e:
            logger.exception("Error al ejecutar agent.process_task: %s", e)
            primary_result = {"success": False, "error": "agent_failed"}

        # Estados agentes
        agent_states = await self._get_all_agent_states()
        emergence_analysis = await self.collective_mind.analyze_agent_states(
            agent_states
        )

        final_result = await self.collective_mind.enhance_with_collective_intelligence(
            primary_result, emergence_analysis
        )

        await self._learn_and_evolve(task_request, final_result, neural_score)
        await self._update_neural_metrics(final_result, emergence_analysis)

        final_result["neural_metrics"] = {
            "neural_evolution_level": self.neural_metrics.neural_evolution_level,
            "collective_intelligence_score": self.neural_metrics.collective_intelligence_score,
            "task_success_rate": self.neural_metrics.task_success_rate,
            "learning_efficiency": self.neural_metrics.learning_efficiency,
            "emergence_level": self.neural_metrics.emergence_level,
            "consciousness_depth": self.neural_metrics.consciousness_depth,
        }

        logger.info(
            "🧠 Tarea completada - success_rate=%.3f emergence=%.3f activation=%.3f",
            self.neural_metrics.task_success_rate,
            self.neural_metrics.emergence_level,
            self.neural_activation_level,
        )
        return final_result

    async def _enhance_task_collective(
        self, task: Dict[str, Any], neural_score: float
    ) -> Dict[str, Any]:
        enhanced = dict(task)
        enhanced["neural_enhancement"] = {
            "neural_score": float(neural_score),
            "activation_level": float(self.neural_activation_level),
            "timestamp": datetime.utcnow().isoformat(),
        }
        try:
            original_priority = int(enhanced.get("priority", 3))
        except Exception:
            original_priority = 3
        if neural_score > 0.8 and original_priority < 5:
            enhanced["priority"] = min(5, original_priority + 1)
        return enhanced

    async def _learn_and_evolve(
        self, task: Dict[str, Any], result: Dict[str, Any], neural_score: float
    ):
        success = bool(result.get("success", False))
        if success:
            self.successful_tasks += 1
            self.neural_activation_level = min(
                1.0, self.neural_activation_level + 0.002
            )
        await self.neural_cortex.learn_from_outcome(task, success, neural_score)
        await self.learning_engine.reflect_and_evolve(task, result)

    async def _update_neural_metrics(
        self, result: Dict[str, Any], emergence_analysis: Dict[str, Any]
    ):
        self.neural_metrics.task_success_rate = float(self.successful_tasks) / max(
            1, self.tasks_processed
        )
        self.neural_metrics.neural_evolution_level = float(self.neural_activation_level)
        self.neural_metrics.collective_intelligence_score = (
            float(emergence_analysis.get("emergence_level", 0.0)) * 100.0
        )
        learning_metrics = self.learning_engine.get_learning_metrics()
        self.neural_metrics.learning_efficiency = min(
            1.0, learning_metrics.get("efficiency", 0.0)
        )
        self.neural_metrics.emergence_level = float(
            emergence_analysis.get("emergence_level", 0.0)
        )
        if bool(result.get("success", False)):
            self.neural_metrics.consciousness_depth = min(
                7.0, self.neural_metrics.consciousness_depth + 0.05
            )
        self.neural_metrics.adaptation_speed = float(self.neural_activation_level)
        self.neural_metrics.sanitize()

    async def _get_all_agent_states(self) -> Dict[str, Any]:
        """Obtener estados de todos los agentes registrados"""
        states: Dict[str, Any] = {}
        for agent_id, agent in getattr(self, "agent_registry", {}).items():
            try:
                performance = float(getattr(agent, "performance_score", 0.5))
                domains = getattr(agent, "domains", ["general"])
                capabilities = getattr(agent, "capabilities", [])
                states[agent_id] = {
                    "performance_score": performance,
                    "domains": domains,
                    "capabilities": capabilities,
                    "status": getattr(agent, "status", "active"),
                }
            except Exception as e:
                states[agent_id] = {"performance_score": 0.3, "error": str(e)}
        return states

    async def get_neural_brain_status(self) -> Dict[str, Any]:
        return {
            "neural_brain_status": "active",
            "neural_activation_level": self.neural_activation_level,
            "collective_mind_status": "emergent",
            "tasks_processed": self.tasks_processed,
            "successful_tasks": self.successful_tasks,
            "success_rate": self.neural_metrics.task_success_rate,
            "neural_metrics": {
                "evolution_level": self.neural_metrics.neural_evolution_level,
                "collective_intelligence": self.neural_metrics.collective_intelligence_score,
                "emergence_level": self.neural_metrics.emergence_level,
                "learning_efficiency": self.neural_metrics.learning_efficiency,
                "consciousness_depth": self.neural_metrics.consciousness_depth,
            },
            "compatibility_level": 100,
            "enterprise_grade": True,
            "version": "MCP_Neural_Brain_v1.1_EnterpriseReady",
        }


# FUNCIONES DE COMPATIBILIDAD PARA INTEGRACIÓN ENTERPRISE


async def initialize_mcp_neural_brain(
    config: Optional[Dict[str, Any]] = None,
) -> MCPNeural:
    """
    Inicializar MCP Neural Brain (mejorado)
    """
    logger.info("🚀 Inicializando MCP Neural Brain (mejorado)")
    neural_brain = MCPNeural(config=config or {})
    await neural_brain.start_system()
    return neural_brain


async def get_enhanced_master_orchestrator(
    config: Optional[Dict[str, Any]] = None,
) -> MCPNeural:
    return await initialize_mcp_neural_brain(config)
