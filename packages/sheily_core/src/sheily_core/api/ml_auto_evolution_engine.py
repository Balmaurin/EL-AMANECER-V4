#!/usr/bin/env python3
"""
SISTEMA DE EVOLUCIÓN AUTOMÁTICA ML
==================================

Sistema de evolución automática de modelos de machine learning
con algoritmos genéticos y optimización automática.
"""

import asyncio
import hashlib
import json
import logging
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class MLModelGenome:
    """Genoma de un modelo ML para evolución genética"""

    model_id: str
    architecture: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class EvolutionMetrics:
    """Métricas de evolución ML"""

    generation: int
    best_fitness: float
    average_fitness: float
    population_size: int
    convergence_rate: float
    mutation_rate: float
    crossover_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


class MLAutoEvolutionEngine:
    """Motor de evolución automática ML"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()

        # Población de modelos
        self.population: Dict[str, MLModelGenome] = {}
        self.generation = 0
        self.best_model: Optional[MLModelGenome] = None

        # Historial de evolución
        self.evolution_history: List[EvolutionMetrics] = []
        self.fitness_evaluations: Dict[str, float] = {}

        # Motores de evolución
        self.genetic_engine = GeneticEvolutionEngine()
        self.neural_architecture_search = NeuralArchitectureSearch()
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        self.model_validator = ModelValidator()

        # Configuración de evolución
        self.population_size = self.config["population_size"]
        self.mutation_rate = self.config["mutation_rate"]
        self.crossover_rate = self.config["crossover_rate"]
        self.elitism_rate = self.config["elitism_rate"]
        self.tournament_size = self.config["tournament_size"]

        # Estado del sistema
        self.is_evolving = False
        self.evolution_thread: Optional[threading.Thread] = None

        logger.info("🧬 ML Auto-Evolution Engine initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Configuración por defecto"""
        return {
            "population_size": 50,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "elitism_rate": 0.1,
            "tournament_size": 5,
            "max_generations": 100,
            "fitness_threshold": 0.95,
            "evolution_interval": 300,  # 5 minutos
            "convergence_check_generations": 10,
            "architecture_search_space": {
                "layers": [1, 2, 3, 4, 5],
                "neurons_per_layer": [32, 64, 128, 256, 512],
                "activation_functions": ["relu", "tanh", "sigmoid", "leaky_relu"],
                "dropout_rates": [0.0, 0.1, 0.2, 0.3, 0.5],
            },
            "hyperparameter_space": {
                "learning_rate": [1e-5, 1e-4, 1e-3, 1e-2],
                "batch_size": [16, 32, 64, 128],
                "optimizer": ["adam", "sgd", "rmsprop", "adamw"],
                "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
            },
        }

    async def evolve_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evoluciona automáticamente un modelo ML"""
        try:
            # Crear genoma inicial desde datos del modelo
            initial_genome = await self._create_initial_genome(model_data)

            # Añadir a población
            self.population[initial_genome.model_id] = initial_genome

            # Iniciar evolución si no está corriendo
            if not self.is_evolving:
                await self._start_evolution_process()

            # Esperar resultados de evolución
            evolved_model = await self._wait_for_evolution_results(
                initial_genome.model_id
            )

            return {
                "status": "evolution_completed",
                "evolved_model": evolved_model,
                "generations_evolved": self.generation,
                "fitness_improvement": evolved_model.fitness_score
                - initial_genome.fitness_score,
                "evolution_metrics": self._get_current_metrics(),
            }

        except Exception as e:
            logger.error(f"ML evolution failed: {e}")
            return {
                "status": "evolution_failed",
                "error": str(e),
                "fallback_model": model_data,
            }

    async def _create_initial_genome(self, model_data: Dict[str, Any]) -> MLModelGenome:
        """Crear genoma inicial desde datos del modelo"""
        model_id = f"model_{hashlib.md5(str(model_data).encode()).hexdigest()[:8]}_{int(time.time())}"

        # Extraer arquitectura del modelo
        architecture = self._extract_model_architecture(model_data)

        # Generar hiperparámetros iniciales
        hyperparameters = self._generate_initial_hyperparameters()

        genome = MLModelGenome(
            model_id=model_id,
            architecture=architecture,
            hyperparameters=hyperparameters,
            generation=self.generation,
        )

        # Evaluar fitness inicial
        genome.fitness_score = await self._evaluate_fitness(genome)

        return genome

    def _extract_model_architecture(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extraer arquitectura del modelo"""
        # Implementación simplificada - en producción sería más sofisticada
        if "layers" in model_data:
            return model_data["layers"]
        else:
            # Arquitectura por defecto
            return {
                "input_dim": model_data.get("input_dim", 784),
                "output_dim": model_data.get("output_dim", 10),
                "hidden_layers": [
                    {"neurons": 128, "activation": "relu", "dropout": 0.2},
                    {"neurons": 64, "activation": "relu", "dropout": 0.1},
                ],
            }

    def _generate_initial_hyperparameters(self) -> Dict[str, Any]:
        """Generar hiperparámetros iniciales"""
        return {
            "learning_rate": random.choice(
                self.config["hyperparameter_space"]["learning_rate"]
            ),
            "batch_size": random.choice(
                self.config["hyperparameter_space"]["batch_size"]
            ),
            "optimizer": random.choice(
                self.config["hyperparameter_space"]["optimizer"]
            ),
            "weight_decay": random.choice(
                self.config["hyperparameter_space"]["weight_decay"]
            ),
            "epochs": 10,
        }

    async def _evaluate_fitness(self, genome: MLModelGenome) -> float:
        """Evaluar fitness de un genoma"""
        try:
            # Simulación de evaluación - en producción usaría datos reales
            # Calcular complejidad de arquitectura
            architecture_complexity = (
                len(genome.architecture.get("hidden_layers", [])) * 0.1
            )

            # Calcular eficiencia de hiperparámetros
            lr_score = (
                1.0
                - abs(genome.hyperparameters.get("learning_rate", 0.001) - 0.001) / 0.01
            )
            batch_score = genome.hyperparameters.get("batch_size", 32) / 128.0

            # Fitness base + ruido
            base_fitness = (architecture_complexity + lr_score + batch_score) / 3.0
            fitness = base_fitness + random.uniform(-0.1, 0.1)

            # Limitar entre 0 y 1
            fitness = max(0.0, min(1.0, fitness))

            self.fitness_evaluations[genome.model_id] = fitness
            return fitness

        except Exception as e:
            logger.error(f"Fitness evaluation failed: {e}")
            return 0.0

    async def _start_evolution_process(self):
        """Iniciar proceso de evolución"""
        if self.is_evolving:
            return

        self.is_evolving = True
        self.evolution_thread = threading.Thread(
            target=self._evolution_loop, daemon=True
        )
        self.evolution_thread.start()

        logger.info("🧬 ML evolution process started")

    def _evolution_loop(self):
        """Bucle principal de evolución"""
        while self.is_evolving:
            try:
                asyncio.run(self._perform_evolution_step())
                time.sleep(self.config["evolution_interval"])
            except Exception as e:
                logger.error(f"Evolution step failed: {e}")
                time.sleep(10)

    async def _perform_evolution_step(self):
        """Realizar un paso de evolución"""
        if len(self.population) < self.config["population_size"]:
            # Esperar más modelos
            return

        self.generation += 1

        # Evaluar población actual
        await self._evaluate_population()

        # Seleccionar mejores individuos (elitismo)
        elite_size = int(self.config["elitism_rate"] * len(self.population))
        elite_individuals = await self._select_elite(elite_size)

        # Crear nueva generación
        new_population = elite_individuals.copy()

        # Generar descendientes
        while len(new_population) < self.config["population_size"]:
            # Selección por torneo
            parent1 = await self._tournament_selection()
            parent2 = await self._tournament_selection()

            # Cruce
            if random.random() < self.config["crossover_rate"]:
                offspring1, offspring2 = await self.genetic_engine.crossover(
                    parent1, parent2
                )
            else:
                offspring1, offspring2 = parent1, parent2

            # Mutación
            if random.random() < self.config["mutation_rate"]:
                offspring1 = await self.genetic_engine.mutate(offspring1)
            if random.random() < self.config["mutation_rate"]:
                offspring2 = await self.genetic_engine.mutate(offspring2)

            # Añadir a nueva población
            new_population.extend([offspring1, offspring2])

        # Limitar población
        new_population = new_population[: self.config["population_size"]]

        # Actualizar población
        self.population = {genome.model_id: genome for genome in new_population}

        # Actualizar mejor modelo
        current_best = max(self.population.values(), key=lambda x: x.fitness_score)
        if (
            self.best_model is None
            or current_best.fitness_score > self.best_model.fitness_score
        ):
            self.best_model = current_best

        # Registrar métricas
        await self._record_evolution_metrics()

        logger.info(
            f"🧬 Generation {self.generation} completed. Best fitness: {current_best.fitness_score:.3f}"
        )

    async def _evaluate_population(self):
        """Evaluar toda la población"""
        evaluation_tasks = []
        for genome in self.population.values():
            if genome.model_id not in self.fitness_evaluations:
                task = self._evaluate_fitness(genome)
                evaluation_tasks.append(task)

        if evaluation_tasks:
            fitness_scores = await asyncio.gather(*evaluation_tasks)
            # Las evaluaciones ya actualizan fitness_evaluations

    async def _select_elite(self, elite_size: int) -> List[MLModelGenome]:
        """Seleccionar individuos élite"""
        sorted_population = sorted(
            self.population.values(), key=lambda x: x.fitness_score, reverse=True
        )
        return sorted_population[:elite_size]

    async def _tournament_selection(self) -> MLModelGenome:
        """Selección por torneo"""
        tournament = random.sample(
            list(self.population.values()),
            min(self.config["tournament_size"], len(self.population)),
        )
        return max(tournament, key=lambda x: x.fitness_score)

    async def _wait_for_evolution_results(
        self, model_id: str, timeout: int = 300
    ) -> MLModelGenome:
        """Esperar resultados de evolución para un modelo específico"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if model_id in self.population:
                evolved_model = self.population[model_id]
                if evolved_model.generation > 0:  # Ha sido evolucionado
                    return evolved_model
            await asyncio.sleep(1)

        # Timeout - devolver modelo original o mejor disponible
        return self.population.get(model_id, self.best_model)

    async def _record_evolution_metrics(self):
        """Registrar métricas de evolución"""
        fitness_scores = [genome.fitness_score for genome in self.population.values()]

        metrics = EvolutionMetrics(
            generation=self.generation,
            best_fitness=max(fitness_scores),
            average_fitness=sum(fitness_scores) / len(fitness_scores),
            population_size=len(self.population),
            convergence_rate=self._calculate_convergence_rate(),
            mutation_rate=self.config["mutation_rate"],
            crossover_rate=self.config["crossover_rate"],
        )

        self.evolution_history.append(metrics)

    def _calculate_convergence_rate(self) -> float:
        """Calcular tasa de convergencia"""
        if len(self.evolution_history) < 2:
            return 0.0

        recent_fitness = [m.best_fitness for m in self.evolution_history[-5:]]
        if len(recent_fitness) < 2:
            return 0.0

        # Tasa de mejora en las últimas generaciones
        improvements = []
        for i in range(1, len(recent_fitness)):
            improvement = recent_fitness[i] - recent_fitness[i - 1]
            improvements.append(improvement)

        return sum(improvements) / len(improvements) if improvements else 0.0

    def _get_current_metrics(self) -> Dict[str, Any]:
        """Obtener métricas actuales"""
        if not self.evolution_history:
            return {}

        latest = self.evolution_history[-1]
        return {
            "generation": latest.generation,
            "best_fitness": latest.best_fitness,
            "average_fitness": latest.average_fitness,
            "population_size": latest.population_size,
            "convergence_rate": latest.convergence_rate,
            "total_evaluations": len(self.fitness_evaluations),
        }

    async def get_evolution_status(self) -> Dict[str, Any]:
        """Obtener estado de evolución"""
        return {
            "is_evolving": self.is_evolving,
            "current_generation": self.generation,
            "population_size": len(self.population),
            "best_model_fitness": (
                self.best_model.fitness_score if self.best_model else 0.0
            ),
            "evolution_metrics": self._get_current_metrics(),
            "evolution_history_length": len(self.evolution_history),
        }

    def stop_evolution(self):
        """Detener evolución"""
        self.is_evolving = False
        logger.info("🧬 ML evolution stopped")


class GeneticEvolutionEngine:
    """Motor de evolución genética"""

    async def crossover(
        self, parent1: MLModelGenome, parent2: MLModelGenome
    ) -> Tuple[MLModelGenome, MLModelGenome]:
        """Cruce entre dos genomas"""
        # Cruce de arquitectura
        architecture1 = self._crossover_architecture(
            parent1.architecture, parent2.architecture
        )
        architecture2 = self._crossover_architecture(
            parent2.architecture, parent1.architecture
        )

        # Cruce de hiperparámetros
        hyperparams1 = self._crossover_hyperparameters(
            parent1.hyperparameters, parent2.hyperparameters
        )
        hyperparams2 = self._crossover_hyperparameters(
            parent2.hyperparameters, parent1.hyperparameters
        )

        # Crear descendientes
        offspring1 = MLModelGenome(
            model_id=f"crossover_{parent1.model_id[:4]}_{parent2.model_id[:4]}_{int(time.time())}",
            architecture=architecture1,
            hyperparameters=hyperparams1,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.model_id, parent2.model_id],
        )

        offspring2 = MLModelGenome(
            model_id=f"crossover_{parent2.model_id[:4]}_{parent1.model_id[:4]}_{int(time.time()) + 1}",
            architecture=architecture2,
            hyperparameters=hyperparams2,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent2.model_id, parent1.model_id],
        )

        return offspring1, offspring2

    def _crossover_architecture(
        self, arch1: Dict[str, Any], arch2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Cruce de arquitecturas"""
        # Implementación simplificada
        if random.random() < 0.5:
            return arch1.copy()
        else:
            return arch2.copy()

    def _crossover_hyperparameters(
        self, hp1: Dict[str, Any], hp2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Cruce de hiperparámetros"""
        crossover_hp = {}
        for key in set(hp1.keys()) | set(hp2.keys()):
            if random.random() < 0.5:
                crossover_hp[key] = hp1.get(key, hp2.get(key))
            else:
                crossover_hp[key] = hp2.get(key, hp1.get(key))
        return crossover_hp

    async def mutate(self, genome: MLModelGenome) -> MLModelGenome:
        """Mutar un genoma"""
        mutated_genome = MLModelGenome(
            model_id=f"mutated_{genome.model_id[:6]}_{int(time.time())}",
            architecture=genome.architecture.copy(),
            hyperparameters=genome.hyperparameters.copy(),
            generation=genome.generation,
            parent_ids=[genome.model_id],
            mutation_history=genome.mutation_history
            + [f"generation_{genome.generation}"],
        )

        # Mutar arquitectura
        mutated_genome.architecture = self._mutate_architecture(
            mutated_genome.architecture
        )

        # Mutar hiperparámetros
        mutated_genome.hyperparameters = self._mutate_hyperparameters(
            mutated_genome.hyperparameters
        )

        return mutated_genome

    def _mutate_architecture(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Mutar arquitectura"""
        # Implementación simplificada - cambiar número de neuronas aleatoriamente
        if "hidden_layers" in architecture:
            for layer in architecture["hidden_layers"]:
                if random.random() < 0.3:  # 30% probabilidad de mutación
                    layer["neurons"] = random.choice([32, 64, 128, 256, 512])
        return architecture

    def _mutate_hyperparameters(
        self, hyperparameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mutar hiperparámetros"""
        mutated_hp = hyperparameters.copy()

        # Mutar learning rate
        if random.random() < 0.2:
            mutated_hp["learning_rate"] = random.choice([1e-5, 1e-4, 1e-3, 1e-2])

        # Mutar batch size
        if random.random() < 0.2:
            mutated_hp["batch_size"] = random.choice([16, 32, 64, 128])

        return mutated_hp


class NeuralArchitectureSearch:
    """Búsqueda de arquitectura neuronal"""

    async def search_architecture(
        self, input_dim: int, output_dim: int
    ) -> Dict[str, Any]:
        """Buscar arquitectura óptima"""
        # Implementación simplificada
        return {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_layers": [
                {
                    "neurons": random.choice([64, 128, 256]),
                    "activation": "relu",
                    "dropout": 0.2,
                },
                {
                    "neurons": random.choice([32, 64, 128]),
                    "activation": "relu",
                    "dropout": 0.1,
                },
            ],
        }


class HyperparameterOptimizer:
    """Optimizador de hiperparámetros"""

    async def optimize_hyperparameters(
        self, model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimizar hiperparámetros"""
        # Implementación simplificada
        return {
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "adam",
            "weight_decay": 1e-4,
            "epochs": 10,
        }


class ModelValidator:
    """Validador de modelos"""

    async def validate_model(self, model_config: Dict[str, Any]) -> bool:
        """Validar configuración del modelo"""
        # Validaciones básicas
        if "input_dim" not in model_config or "output_dim" not in model_config:
            return False
        if (
            model_config.get("input_dim", 0) <= 0
            or model_config.get("output_dim", 0) <= 0
        ):
            return False
        return True


# Instancia global
ml_auto_evolution_engine = MLAutoEvolutionEngine()


async def evolve_model(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Función pública para evolución ML"""
    return await ml_auto_evolution_engine.evolve_model(model_data)


async def get_evolution_status() -> Dict[str, Any]:
    """Obtener estado de evolución ML"""
    return await ml_auto_evolution_engine.get_evolution_status()


def stop_ml_evolution():
    """Detener evolución ML"""
    ml_auto_evolution_engine.stop_evolution()
