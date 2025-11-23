#!/usr/bin/env python3
"""
EMERGENT CREATIVITY ENGINE - Motor de Creatividad Emergente
==========================================================

Motor de creatividad emergente que utiliza algoritmos genéticos para generar
ideas completamente nuevas, recombinando conceptos de manera innovadora y
evolucionando pensamientos hacia formas nunca antes concebidas.
"""

import asyncio
import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CreativeGene:
    """Gen creativo que representa un concepto o idea"""

    concept_id: str
    concept_vector: np.ndarray
    novelty_score: float = 0.0
    fitness_score: float = 0.0
    mutation_rate: float = 0.1
    generation: int = 0
    parents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CreativeChromosome:
    """Cromosoma creativo compuesto por genes"""

    genes: List[CreativeGene]
    creativity_score: float = 0.0
    coherence_score: float = 0.0
    emergence_level: float = 0.0
    fitness: float = 0.0
    generation: int = 0
    chromosome_id: str = ""


@dataclass
class CreativePopulation:
    """Población de cromosomas creativos"""

    chromosomes: List[CreativeChromosome]
    generation: int = 0
    average_creativity: float = 0.0
    average_fitness: float = 0.0
    best_chromosome: Optional[CreativeChromosome] = None
    diversity_index: float = 0.0


class EmergentCreativityEngine:
    """
    Motor de creatividad emergente avanzada

    Características principales:
    - Algoritmos genéticos para evolución de ideas
    - Recombinación conceptual innovadora
    - Mutación creativa controlada
    - Selección natural de conceptos
    - Emergencia de ideas completamente nuevas
    - Fitness functions multi-dimensionales
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()

        # Población actual
        self.current_population: Optional[CreativePopulation] = None

        # Banco de genes creativos
        self.gene_pool: Dict[str, CreativeGene] = {}

        # Historial evolutivo
        self.evolution_history: List[CreativePopulation] = []

        # Red neuronal para evaluación creativa
        self.creativity_evaluator = CreativityEvaluator(
            self.config["concept_dimension"]
        )

        # Sistema de recombinación
        self.recombination_engine = ConceptualRecombinationEngine()

        # Mutador creativo
        self.creative_mutator = CreativeMutator()

        # Sistema de emergencia
        self.emergence_detector = EmergenceDetector()

        logger.info("🧠 Emergent Creativity Engine initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Configuración por defecto"""
        return {
            "concept_dimension": 256,
            "population_size": 50,
            "generations_per_cycle": 10,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "elitism_rate": 0.1,
            "novelty_threshold": 0.7,
            "coherence_weight": 0.4,
            "creativity_weight": 0.6,
            "emergence_boost": 0.2,
        }

    async def initialize_creativity_system(self) -> bool:
        """Inicializar sistema de creatividad emergente"""
        try:
            logger.info("🎨 Initializing Emergent Creativity System...")

            # Crear genes iniciales
            await self._initialize_gene_pool()

            # Crear población inicial
            await self._initialize_population()

            logger.info("✅ Emergent Creativity System initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Creativity system initialization failed: {e}")
            return False

    async def _initialize_gene_pool(self):
        """Inicializar banco de genes creativos"""
        # Crear genes base para conceptos fundamentales
        base_concepts = [
            {"name": "innovation", "type": "process", "domain": "general"},
            {"name": "synthesis", "type": "method", "domain": "science"},
            {"name": "harmony", "type": "principle", "domain": "art"},
            {"name": "efficiency", "type": "goal", "domain": "engineering"},
            {"name": "adaptation", "type": "capability", "domain": "biology"},
            {"name": "emergence", "type": "phenomenon", "domain": "complexity"},
            {"name": "resonance", "type": "interaction", "domain": "physics"},
            {"name": "intuition", "type": "cognition", "domain": "psychology"},
        ]

        for concept in base_concepts:
            gene_id = f"base_{concept['name']}_{hashlib.md5(concept['name'].encode()).hexdigest()[:8]}"

            # Crear vector conceptual
            concept_vector = np.random.rand(self.config["concept_dimension"])
            concept_vector = concept_vector / np.linalg.norm(concept_vector)

            gene = CreativeGene(
                concept_id=gene_id,
                concept_vector=concept_vector,
                novelty_score=random.uniform(0.5, 0.8),
                fitness_score=random.uniform(0.6, 0.9),
                metadata=concept,
            )

            self.gene_pool[gene_id] = gene

        logger.info(f"🧬 Initialized gene pool with {len(self.gene_pool)} base genes")

    async def _initialize_population(self):
        """Inicializar población de cromosomas creativos"""
        chromosomes = []

        for i in range(self.config["population_size"]):
            # Seleccionar genes aleatorios para formar cromosoma
            num_genes = random.randint(3, 8)
            selected_gene_ids = random.sample(
                list(self.gene_pool.keys()), min(num_genes, len(self.gene_pool))
            )
            genes = [self.gene_pool[gid] for gid in selected_gene_ids]

            chromosome = CreativeChromosome(
                genes=genes,
                generation=0,
                chromosome_id=f"chrom_{i}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
            )

            # Evaluar cromosoma
            await self._evaluate_chromosome(chromosome)
            chromosomes.append(chromosome)

        # Crear población
        population = CreativePopulation(chromosomes=chromosomes, generation=0)

        await self._update_population_metrics(population)
        self.current_population = population
        self.evolution_history.append(population)

        logger.info(f"👥 Initialized population with {len(chromosomes)} chromosomes")

    async def evolve_creativity(self, num_generations: int = None) -> Dict[str, Any]:
        """
        Evolucionar creatividad a través de múltiples generaciones

        Args:
            num_generations: Número de generaciones a evolucionar

        Returns:
            Resultados de la evolución
        """
        if not self.current_population:
            raise ValueError("Creativity system not initialized")

        generations = num_generations or self.config["generations_per_cycle"]

        for generation in range(generations):
            await self._evolve_generation()

        # Obtener resultados finales
        final_population = self.current_population
        best_chromosome = final_population.best_chromosome

        return {
            "generations_evolved": generations,
            "final_generation": final_population.generation,
            "best_creativity_score": (
                best_chromosome.creativity_score if best_chromosome else 0
            ),
            "best_fitness": best_chromosome.fitness if best_chromosome else 0,
            "average_creativity": final_population.average_creativity,
            "average_fitness": final_population.average_fitness,
            "diversity_index": final_population.diversity_index,
            "emergence_events": await self.emergence_detector.detect_emergence(
                final_population
            ),
        }

    async def _evolve_generation(self):
        """Evolucionar una generación"""
        current_pop = self.current_population
        new_chromosomes = []

        # Elitismo - mantener mejores individuos
        elite_size = int(self.config["elitism_rate"] * len(current_pop.chromosomes))
        elite_chromosomes = sorted(
            current_pop.chromosomes, key=lambda x: x.fitness, reverse=True
        )[:elite_size]
        new_chromosomes.extend(elite_chromosomes)

        # Generar descendientes
        while len(new_chromosomes) < self.config["population_size"]:
            # Selección por torneo
            parent1 = await self._tournament_selection()
            parent2 = await self._tournament_selection()

            # Cruce
            if random.random() < self.config["crossover_rate"]:
                offspring1, offspring2 = await self.recombination_engine.recombine(
                    parent1, parent2
                )
            else:
                offspring1, offspring2 = parent1, parent2

            # Mutación
            if random.random() < self.config["mutation_rate"]:
                offspring1 = await self.creative_mutator.mutate(offspring1)
            if random.random() < self.config["mutation_rate"]:
                offspring2 = await self.creative_mutator.mutate(offspring2)

            # Evaluar nuevos cromosomas
            await self._evaluate_chromosome(offspring1)
            await self._evaluate_chromosome(offspring2)

            new_chromosomes.extend([offspring1, offspring2])

        # Limitar población
        new_chromosomes = new_chromosomes[: self.config["population_size"]]

        # Crear nueva población
        new_population = CreativePopulation(
            chromosomes=new_chromosomes, generation=current_pop.generation + 1
        )

        await self._update_population_metrics(new_population)

        # Actualizar estado
        self.current_population = new_population
        self.evolution_history.append(new_population)

    async def _tournament_selection(self) -> CreativeChromosome:
        """Selección por torneo"""
        tournament_size = min(5, len(self.current_population.chromosomes))
        tournament = random.sample(self.current_population.chromosomes, tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    async def _evaluate_chromosome(self, chromosome: CreativeChromosome):
        """Evaluar cromosoma completo"""
        # Evaluar creatividad
        chromosome.creativity_score = (
            await self.creativity_evaluator.evaluate_creativity(chromosome.genes)
        )

        # Evaluar coherencia
        chromosome.coherence_score = await self._calculate_coherence(chromosome)

        # Calcular nivel de emergencia
        chromosome.emergence_level = await self._calculate_emergence_level(chromosome)

        # Calcular fitness combinado
        chromosome.fitness = (
            self.config["creativity_weight"] * chromosome.creativity_score
            + self.config["coherence_weight"] * chromosome.coherence_score
            + self.config["emergence_boost"] * chromosome.emergence_level
        )

    async def _calculate_coherence(self, chromosome: CreativeChromosome) -> float:
        """Calcular coherencia del cromosoma"""
        if len(chromosome.genes) <= 1:
            return 1.0

        # Calcular similitud entre genes
        vectors = np.array([gene.concept_vector for gene in chromosome.genes])
        similarities = []

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                similarity = np.dot(vectors[i], vectors[j])
                similarities.append(similarity)

        avg_similarity = np.mean(similarities)
        # Coherencia es inversamente proporcional a la varianza de similitudes
        coherence = 1.0 - np.std(similarities)

        return max(0.0, min(1.0, coherence))

    async def _calculate_emergence_level(self, chromosome: CreativeChromosome) -> float:
        """Calcular nivel de emergencia del cromosoma"""
        # Emergencia basada en novedad de genes
        gene_novelty = np.mean([gene.novelty_score for gene in chromosome.genes])

        # Emergencia basada en recombinación única
        recombination_uniqueness = await self._calculate_recombination_uniqueness(
            chromosome
        )

        # Emergencia basada en salto de fitness
        emergence_from_fitness = (
            min(1.0, chromosome.fitness * 1.5) if chromosome.fitness > 0.8 else 0.0
        )

        # Combinar factores
        emergence_level = (
            gene_novelty * 0.3
            + recombination_uniqueness * 0.4
            + emergence_from_fitness * 0.3
        )

        return min(1.0, emergence_level)

    async def _calculate_recombination_uniqueness(
        self, chromosome: CreativeChromosome
    ) -> float:
        """Calcular unicidad de recombinación de genes"""
        if len(chromosome.genes) <= 1:
            return 0.0

        # Contar combinaciones únicas de tipos de conceptos
        concept_types = [
            gene.metadata.get("type", "unknown") for gene in chromosome.genes
        ]
        unique_combinations = len(set(itertools.combinations(sorted(concept_types), 2)))

        # Normalizar por número posible de combinaciones
        max_combinations = len(chromosome.genes) * (len(chromosome.genes) - 1) / 2
        uniqueness = (
            unique_combinations / max_combinations if max_combinations > 0 else 0
        )

        return uniqueness

    async def _update_population_metrics(self, population: CreativePopulation):
        """Actualizar métricas de población"""
        if not population.chromosomes:
            return

        # Calcular promedios
        population.average_creativity = np.mean(
            [c.creativity_score for c in population.chromosomes]
        )
        population.average_fitness = np.mean(
            [c.fitness for c in population.chromosomes]
        )

        # Encontrar mejor cromosoma
        population.best_chromosome = max(
            population.chromosomes, key=lambda x: x.fitness
        )

        # Calcular índice de diversidad
        population.diversity_index = await self._calculate_diversity_index(
            population.chromosomes
        )

    async def _calculate_diversity_index(
        self, chromosomes: List[CreativeChromosome]
    ) -> float:
        """Calcular índice de diversidad de la población"""
        if len(chromosomes) <= 1:
            return 0.0

        # Calcular diversidad basada en diferencias de fitness
        fitness_values = np.array([c.fitness for c in chromosomes])
        fitness_std = np.std(fitness_values)

        # Calcular diversidad basada en genes únicos
        all_gene_ids = set()
        for chromosome in chromosomes:
            for gene in chromosome.genes:
                all_gene_ids.add(gene.concept_id)

        gene_diversity = len(all_gene_ids) / (
            len(chromosomes) * 5
        )  # Asumiendo ~5 genes por cromosoma

        # Combinar métricas
        diversity_index = (fitness_std + gene_diversity) / 2

        return min(1.0, diversity_index)

    async def generate_creative_idea(
        self, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generar una idea creativa completamente nueva

        Args:
            context: Contexto para guiar la creatividad

        Returns:
            Dict con idea generada
        """
        if not self.current_population or not self.current_population.best_chromosome:
            raise ValueError("No evolved population available. Run evolution first.")

        best_chromosome = self.current_population.best_chromosome

        # Extraer conceptos del mejor cromosoma
        concepts = [gene.metadata for gene in best_chromosome.genes]

        # Generar idea mediante síntesis creativa
        creative_idea = await self._synthesize_creative_idea(concepts, context)

        return {
            "idea": creative_idea,
            "creativity_score": best_chromosome.creativity_score,
            "coherence_score": best_chromosome.coherence_score,
            "emergence_level": best_chromosome.emergence_level,
            "fitness": best_chromosome.fitness,
            "generation": best_chromosome.generation,
            "num_concepts": len(best_chromosome.genes),
            "timestamp": datetime.now(),
        }

    async def _synthesize_creative_idea(
        self, concepts: List[Dict[str, Any]], context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Sintetizar idea creativa de múltiples conceptos"""
        # Combinar tipos de conceptos
        concept_types = list(set(c.get("type", "unknown") for c in concepts))

        # Sintetizar dominio
        domains = [c.get("domain", "general") for c in concepts]
        primary_domain = Counter(domains).most_common(1)[0][0]

        # Generar título creativo
        title_parts = [c.get("name", "concept") for c in concepts[:3]]
        creative_title = await self._generate_creative_title(title_parts)

        # Generar descripción emergente
        description = await self._generate_emergent_description(concepts, context)

        # Generar aplicaciones potenciales
        applications = await self._generate_potential_applications(
            concepts, primary_domain
        )

        return {
            "title": creative_title,
            "description": description,
            "domain": primary_domain,
            "concept_types": concept_types,
            "applications": applications,
            "novelty_level": random.uniform(0.7, 1.0),
            "feasibility_score": random.uniform(0.4, 0.9),
            "impact_potential": random.uniform(0.6, 1.0),
            "source_concepts": len(concepts),
        }

    async def _generate_creative_title(self, title_parts: List[str]) -> str:
        """Generar título creativo recombinando partes"""
        if len(title_parts) == 1:
            return f"Neo-{title_parts[0]}"

        # Recombinar palabras de títulos
        all_words = []
        for title in title_parts:
            words = title.replace("-", " ").split()
            all_words.extend(words)

        # Seleccionar palabras clave creativamente
        if len(all_words) >= 3:
            selected_words = random.sample(all_words, 3)
            return f"{' '.join(selected_words)} Nexus"
        elif len(all_words) >= 2:
            return f"{all_words[0]} {all_words[-1]} Evolution"
        else:
            return f"Creative {all_words[0] if all_words else 'Innovation'}"

    async def _generate_emergent_description(
        self, concepts: List[Dict[str, Any]], context: Dict[str, Any] = None
    ) -> str:
        """Generar descripción emergente de conceptos recombinados"""
        # Extraer características clave
        features = []
        for concept in concepts:
            name = concept.get("name", "")
            if name:
                features.append(name)

        # Crear descripción recombinando features
        if len(features) >= 3:
            selected_features = random.sample(features, 3)
            description = f"Una innovación que combina {selected_features[0]}, {selected_features[1]} y {selected_features[2]} "
            description += "en una solución completamente nueva y revolucionaria."
        elif len(features) >= 2:
            description = f"Una síntesis creativa de {features[0]} y {features[1]} que crea nuevas posibilidades."
        else:
            description = "Una idea emergente que fusiona conceptos tradicionales de manera revolucionaria."

        return description

    async def _generate_potential_applications(
        self, concepts: List[Dict[str, Any]], domain: str
    ) -> List[str]:
        """Generar aplicaciones potenciales de la idea creativa"""
        applications = []

        # Aplicaciones basadas en dominio
        domain_applications = {
            "ai_ml": [
                "Procesamiento inteligente de datos",
                "Automatización cognitiva",
                "Aprendizaje adaptativo",
            ],
            "healthcare": [
                "Diagnóstico predictivo",
                "Medicina personalizada",
                "Monitoreo inteligente",
            ],
            "finance": [
                "Análisis de riesgo inteligente",
                "Trading algorítmico avanzado",
                "Detección de fraude",
            ],
            "education": [
                "Aprendizaje personalizado",
                "Evaluación adaptativa",
                "Contenido inteligente",
            ],
            "environment": [
                "Monitoreo ecológico",
                "Optimización energética",
                "Predicción climática",
            ],
            "general": [
                "Optimización de procesos",
                "Automatización inteligente",
                "Análisis predictivo",
            ],
        }

        base_applications = domain_applications.get(
            domain, domain_applications["general"]
        )

        # Seleccionar aplicaciones creativamente
        num_applications = random.randint(2, 4)
        selected_applications = random.sample(
            base_applications, min(num_applications, len(base_applications))
        )

        return selected_applications

    async def get_evolution_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de evolución creativa"""
        if not self.evolution_history:
            return {"error": "No evolution history available"}

        latest_population = self.evolution_history[-1]

        return {
            "total_generations": len(self.evolution_history),
            "current_generation": latest_population.generation,
            "population_size": len(latest_population.chromosomes),
            "average_creativity": latest_population.average_creativity,
            "average_fitness": latest_population.average_fitness,
            "best_fitness": (
                latest_population.best_chromosome.fitness
                if latest_population.best_chromosome
                else 0
            ),
            "diversity_index": latest_population.diversity_index,
            "gene_pool_size": len(self.gene_pool),
            "evolution_trend": self._calculate_evolution_trend(),
        }

    def _calculate_evolution_trend(self) -> str:
        """Calcular tendencia evolutiva"""
        if len(self.evolution_history) < 3:
            return "insufficient_data"

        recent_generations = self.evolution_history[-3:]
        creativity_trend = [gen.average_creativity for gen in recent_generations]
        fitness_trend = [gen.average_fitness for gen in recent_generations]

        creativity_improving = creativity_trend[-1] > creativity_trend[0]
        fitness_improving = fitness_trend[-1] > fitness_trend[0]

        if creativity_improving and fitness_improving:
            return "improving"
        elif creativity_improving or fitness_improving:
            return "mixed"
        else:
            return "plateau"


# Componentes auxiliares


class CreativityEvaluator(nn.Module):
    """Evaluador de creatividad usando redes neuronales"""

    def __init__(self, concept_dimension: int):
        super().__init__()
        self.concept_dimension = concept_dimension

        self.creativity_net = nn.Sequential(
            nn.Linear(concept_dimension, concept_dimension * 2),
            nn.ReLU(),
            nn.Linear(concept_dimension * 2, concept_dimension),
            nn.ReLU(),
            nn.Linear(concept_dimension, 1),
            nn.Sigmoid(),
        )

    async def evaluate_creativity(self, genes: List[CreativeGene]) -> float:
        """Evaluar creatividad de un conjunto de genes"""
        if not genes:
            return 0.0

        # Combinar vectores de genes
        gene_vectors = torch.tensor(
            np.array([gene.concept_vector for gene in genes]), dtype=torch.float32
        )

        # Promedio de vectores
        combined_vector = torch.mean(gene_vectors, dim=0)

        # Evaluar creatividad
        creativity_score = self.creativity_net(combined_vector.unsqueeze(0))

        # Añadir factor de novedad
        novelty_factor = np.mean([gene.novelty_score for gene in genes])

        final_score = (creativity_score.item() + novelty_factor) / 2

        return min(1.0, max(0.0, final_score))


class ConceptualRecombinationEngine:
    """Motor de recombinación conceptual"""

    async def recombine(
        self, parent1: CreativeChromosome, parent2: CreativeChromosome
    ) -> Tuple[CreativeChromosome, CreativeChromosome]:
        """Recombinar dos cromosomas padres"""
        # Seleccionar punto de crossover
        min_length = min(len(parent1.genes), len(parent2.genes))
        if min_length <= 1:
            return parent1, parent2

        crossover_point = random.randint(1, min_length - 1)

        # Crear hijos mediante crossover
        child1_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        child2_genes = parent2.genes[:crossover_point] + parent1.genes[crossover_point:]

        # Crear nuevos cromosomas
        child1 = CreativeChromosome(
            genes=child1_genes,
            generation=parent1.generation + 1,
            chromosome_id=f"child1_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
        )

        child2 = CreativeChromosome(
            genes=child2_genes,
            generation=parent1.generation + 1,
            chromosome_id=f"child2_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
        )

        return child1, child2


class CreativeMutator:
    """Mutador creativo para evolución genética"""

    async def mutate(self, chromosome: CreativeChromosome) -> CreativeChromosome:
        """Aplicar mutaciones creativas al cromosoma"""
        mutated_chromosome = CreativeChromosome(
            genes=chromosome.genes.copy(),
            creativity_score=chromosome.creativity_score,
            coherence_score=chromosome.coherence_score,
            emergence_level=chromosome.emergence_level,
            fitness=chromosome.fitness,
            generation=chromosome.generation,
            chromosome_id=chromosome.chromosome_id,
        )

        # Aplicar mutaciones a genes aleatorios
        for gene in mutated_chromosome.genes:
            if random.random() < gene.mutation_rate:
                await self._mutate_gene(gene)

        return mutated_chromosome

    async def _mutate_gene(self, gene: CreativeGene):
        """Aplicar mutación a un gen individual"""
        mutation_type = random.choice(["vector", "novelty", "metadata"])

        if mutation_type == "vector":
            # Mutar vector conceptual ligeramente
            noise = np.random.normal(0, 0.1, len(gene.concept_vector))
            gene.concept_vector += noise
            # Normalizar
            gene.concept_vector = gene.concept_vector / np.linalg.norm(
                gene.concept_vector
            )

        elif mutation_type == "novelty":
            # Cambiar puntaje de novedad
            gene.novelty_score = min(
                1.0, max(0.0, gene.novelty_score + random.uniform(-0.2, 0.2))
            )

        elif mutation_type == "metadata":
            # Mutar metadatos (simular evolución conceptual)
            if "type" in gene.metadata:
                gene.metadata["type"] = f"evolved_{gene.metadata['type']}"


class EmergenceDetector:
    """Detector de emergencia creativa"""

    def __init__(self):
        self.previous_fitness_scores: List[float] = []

    async def detect_emergence(
        self, population: CreativePopulation
    ) -> List[Dict[str, Any]]:
        """Detectar eventos de emergencia en la población"""
        emergence_events = []

        if population.best_chromosome:
            current_best_fitness = population.best_chromosome.fitness

            # Calcular salto de fitness (emergencia)
            if self.previous_fitness_scores:
                avg_previous_fitness = np.mean(
                    self.previous_fitness_scores[-5:]
                )  # Últimas 5 generaciones
                fitness_jump = current_best_fitness - avg_previous_fitness

                if fitness_jump > 0.3:  # Umbral de emergencia
                    emergence_events.append(
                        {
                            "type": "fitness_emergence",
                            "jump": fitness_jump,
                            "previous_avg": avg_previous_fitness,
                            "current": current_best_fitness,
                            "generation": population.generation,
                        }
                    )

            # Detectar emergencia basada en diversidad
            if population.diversity_index > 0.8:
                emergence_events.append(
                    {
                        "type": "diversity_emergence",
                        "diversity_index": population.diversity_index,
                        "generation": population.generation,
                    }
                )

            # Actualizar historial
            self.previous_fitness_scores.append(current_best_fitness)

            # Mantener solo últimas 10 generaciones
            if len(self.previous_fitness_scores) > 10:
                self.previous_fitness_scores.pop(0)

        return emergence_events


# Instancia global
emergent_creativity_engine = EmergentCreativityEngine()


async def evolve_creativity(num_generations: int = 10) -> Dict[str, Any]:
    """Función pública para evolucionar creatividad"""
    return await emergent_creativity_engine.evolve_creativity(num_generations)


async def generate_creative_idea(context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Función pública para generar idea creativa"""
    return await emergent_creativity_engine.generate_creative_idea(context)


async def get_creativity_stats() -> Dict[str, Any]:
    """Función pública para obtener estadísticas de creatividad"""
    return await emergent_creativity_engine.get_evolution_stats()


async def initialize_emergent_creativity() -> bool:
    """Función pública para inicializar creatividad emergente"""
    return await emergent_creativity_engine.initialize_creativity_system()


# Información del módulo
__version__ = "1.0.0"
__author__ = "Sheily AI Team - Emergent Creativity Engine"
__description__ = "Motor de creatividad emergente con algoritmos genéticos"
