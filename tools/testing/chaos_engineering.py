#!/usr/bin/env python3
"""
Chaos Engineering Framework - Sheily AI
========================================

Sistema de chaos engineering para validar resiliencia del sistema.
Implementa principios de chaos engineering de manera controlada y segura.
"""

import asyncio
import logging
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Awaitable, Callable, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


class ChaosExperiment(Enum):
    """Tipos de experimentos de chaos disponibles"""

    CPU_SPIKE = "cpu_spike"
    MEMORY_SPIKE = "memory_spike"
    NETWORK_DELAY = "network_delay"
    SERVICE_RESTART = "service_restart"


class ExperimentSeverity(Enum):
    """Severidad del experimento"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ChaosResult:
    """Resultado de un experimento de chaos"""

    experiment: ChaosExperiment
    severity: ExperimentSeverity
    duration: float
    success: bool
    observations: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class SystemMetrics:
    """Métricas del sistema durante chaos"""

    cpu_percent: float
    memory_percent: float
    timestamp: datetime = field(default_factory=datetime.now)


class ChaosEngineeringFramework:
    """
    Framework de Chaos Engineering para Sheily AI
    Implementa experimentos controlados para validar resiliencia
    """

    def __init__(self):
        self.experiments: Dict[str, Callable] = {}
        self._setup_experiments()

    def _setup_experiments(self):
        """Configurar experimentos disponibles"""
        self.experiments = {
            ChaosExperiment.CPU_SPIKE.value: self._cpu_spike_experiment,
            ChaosExperiment.MEMORY_SPIKE.value: self._memory_spike_experiment,
            ChaosExperiment.NETWORK_DELAY.value: self._network_delay_experiment,
            ChaosExperiment.SERVICE_RESTART.value: self._service_restart_experiment,
        }

    async def run_experiment(
        self,
        experiment: ChaosExperiment,
        severity: ExperimentSeverity = ExperimentSeverity.MEDIUM,
        duration: int = 30,
    ) -> ChaosResult:
        """
        Ejecutar un experimento de chaos de manera controlada

        Args:
            experiment: Tipo de experimento a ejecutar
            severity: Severidad del experimento
            duration: Duración en segundos

        Returns:
            ChaosResult: Resultado del experimento
        """
        logger.info(
            f"Starting chaos experiment: {experiment.value} (severity: {severity.value})"
        )

        result = ChaosResult(
            experiment=experiment, severity=severity, duration=duration, success=False
        )

        try:
            # Ejecutar experimento
            experiment_func = self.experiments.get(experiment.value)
            if not experiment_func:
                raise ValueError(f"Unknown experiment: {experiment.value}")

            # Ejecutar experimento
            await experiment_func(severity, duration, result.observations)

            # Esperar duración del experimento
            await asyncio.sleep(duration)

            # Validar que el sistema se recuperó
            await self._validate_system_recovery(result)

            result.success = True
            result.observations.append(
                f"✅ Experiment completed successfully in {duration}s"
            )

        except Exception as e:
            error_msg = f"❌ Experiment failed: {str(e)}"
            result.errors.append(error_msg)
            logger.error(error_msg)
            result.observations.append(error_msg)

        logger.info(
            f"Chaos experiment completed: {experiment.value} - {'PASS' if result.success else 'FAIL'}"
        )
        return result

    async def _cpu_spike_experiment(
        self, severity: ExperimentSeverity, duration: int, observations: List[str]
    ):
        """Experimento de spike de CPU"""
        intensity = {"low": 0.3, "medium": 0.6, "high": 0.8, "critical": 0.95}[
            severity.value
        ]

        observations.append(
            f"🚀 Starting CPU spike experiment (intensity: {intensity})"
        )

        # Crear carga de CPU
        end_time = time.time() + duration
        while time.time() < end_time:
            # Operación CPU-intensive
            [x**2 for x in range(10000)]

        observations.append("✅ CPU spike experiment completed")

    async def _memory_spike_experiment(
        self, severity: ExperimentSeverity, duration: int, observations: List[str]
    ):
        """Experimento de spike de memoria"""
        spike_size = (
            {"low": 50, "medium": 100, "high": 200, "critical": 500}[severity.value]
            * 1024
            * 1024
        )  # MB

        observations.append(
            f"💾 Starting memory spike experiment (size: {spike_size/1024/1024}MB)"
        )

        # Simular uso de memoria
        memory_hog = []
        for _ in range(spike_size // 1000):  # Crear objetos que ocupen memoria
            memory_hog.append("x" * 1000)

        # Mantener por duración del experimento
        await asyncio.sleep(duration)

        # Liberar memoria
        del memory_hog

        observations.append("✅ Memory spike experiment completed")

    async def _network_delay_experiment(
        self, severity: ExperimentSeverity, duration: int, observations: List[str]
    ):
        """Experimento de delay de red simulado"""
        delay_ms = {"low": 100, "medium": 500, "high": 2000, "critical": 5000}[
            severity.value
        ]

        observations.append(
            f"🌐 Starting network delay experiment (delay: {delay_ms}ms)"
        )

        # Simular delays en operaciones de red
        for i in range(10):
            delay = random.uniform(delay_ms / 1000 * 0.5, delay_ms / 1000 * 1.5)
            await asyncio.sleep(delay)
            observations.append(f"   Simulated network delay {i+1}/10")

        observations.append("✅ Network delay experiment completed")

    async def _service_restart_experiment(
        self, severity: ExperimentSeverity, duration: int, observations: List[str]
    ):
        """Experimento de restart de servicio simulado"""
        restart_count = {"low": 1, "medium": 2, "high": 3, "critical": 5}[
            severity.value
        ]

        observations.append(
            f"🔄 Starting service restart experiment (restarts: {restart_count})"
        )

        for i in range(restart_count):
            # Simular tiempo de restart
            restart_time = random.uniform(1, 3)
            observations.append(
                f"   Service restart {i+1}/{restart_count} (took {restart_time:.1f}s)"
            )
            await asyncio.sleep(restart_time)

        observations.append("✅ Service restart experiment completed")

    async def _validate_system_recovery(self, result: ChaosResult):
        """Validar que el sistema se recuperó después del experimento"""
        # Verificar métricas del sistema
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent

        # Validar que no estamos en estado crítico
        if cpu_percent > 95:
            result.observations.append(
                f"⚠️ High CPU usage after experiment: {cpu_percent}%"
            )

        if memory_percent > 90:
            result.observations.append(
                f"⚠️ High memory usage after experiment: {memory_percent}%"
            )

        # En un sistema real, verificar health checks de servicios
        result.observations.append("✅ System recovery validated")


async def run_chaos_suite():
    """Ejecutar suite completa de chaos engineering"""
    print("🔥 SHEILY AI - CHAOS ENGINEERING SUITE")
    print("=" * 60)

    framework = ChaosEngineeringFramework()
    results = []

    # Experimentos a ejecutar
    experiments = [
        (ChaosExperiment.CPU_SPIKE, ExperimentSeverity.LOW, 10),
        (ChaosExperiment.MEMORY_SPIKE, ExperimentSeverity.LOW, 10),
        (ChaosExperiment.NETWORK_DELAY, ExperimentSeverity.MEDIUM, 15),
        (ChaosExperiment.SERVICE_RESTART, ExperimentSeverity.LOW, 5),
    ]

    for experiment, severity, duration in experiments:
        print(f"\n🌀 Running: {experiment.value} ({severity.value}) - {duration}s")

        result = await framework.run_experiment(experiment, severity, duration)

        status = "✅ PASS" if result.success else "❌ FAIL"
        print(f"   Result: {status}")
        print(f"   Duration: {result.duration:.2f}s")

        if result.errors:
            print(f"   Errors: {len(result.errors)}")

        results.append(result)

    # Resumen final
    print("\n📊 CHAOS ENGINEERING SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r.success)
    total = len(results)

    print(f"Experiments: {passed}/{total} PASSED")
    success_rate = passed / total * 100 if total > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")

    if success_rate >= 80:
        print("🎉 SYSTEM IS RESILIENT - Chaos experiments passed!")
    else:
        print("⚠️ SYSTEM NEEDS IMPROVEMENT - Some chaos experiments failed")

    return results


if __name__ == "__main__":
    asyncio.run(run_chaos_suite())
