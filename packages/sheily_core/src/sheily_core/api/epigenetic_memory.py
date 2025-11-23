#!/usr/bin/env python3
"""
EPIGENETIC MEMORY - Sistema de Memoria Epigenética
==================================================

Sistema de memoria epigenética que permite herencia de conocimientos y patrones
de aprendizaje a través de "generaciones" de experiencias, similar a la epigenética
biológica donde las experiencias modifican la expresión genética sin alterar
el código base.
"""

import asyncio
import hashlib
import json
import logging
import math
import pickle
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class EpigeneticMark:
    """Marca epigenética que modifica expresión de conocimientos"""

    mark_id: str
    target_knowledge: str
    modification_type: str  # 'methylation', 'acetylation', 'phosphorylation'
    intensity: float
    duration: timedelta
    created_at: datetime
    source_experience: str
    inheritance_probability: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGene:
    """Gen de conocimiento con expresión epigenética"""

    gene_id: str
    knowledge_vector: np.ndarray
    base_expression: float
    epigenetic_marks: List[EpigeneticMark] = field(default_factory=list)
    expression_level: float = 0.0
    stability: float = 1.0
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpigeneticInheritance:
    """Herencia epigenética entre generaciones"""

    parent_generation: int
    child_generation: int
    inherited_marks: List[EpigeneticMark]
    transmission_efficiency: float
    adaptation_benefits: float
    timestamp: datetime


@dataclass
class MemoryGeneration:
    """Generación de memoria con conocimientos y marcas epigenéticas"""

    generation_id: int
    knowledge_genes: Dict[str, KnowledgeGene]
    epigenetic_marks: List[EpigeneticMark]
    creation_timestamp: datetime
    parent_generation_id: Optional[int] = None
    fitness_score: float = 0.0
    adaptation_level: float = 0.0


class EpigeneticMemorySystem:
    """
    Sistema de memoria epigenética avanzada

    Características principales:
    - Herencia epigenética de conocimientos
    - Marcas epigenéticas que modifican expresión
    - Adaptación transgeneracional
    - Estabilidad y evolución de memoria
    - Transmisión selectiva de experiencias
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()

        # Generaciones de memoria
        self.memory_generations: Dict[int, MemoryGeneration] = {}
        self.current_generation_id = 0

        # Banco de genes de conocimiento
        self.knowledge_gene_pool: Dict[str, KnowledgeGene] = {}

        # Historial de herencia
        self.inheritance_history: List[EpigeneticInheritance] = []

        # Sistema de expresión epigenética
        self.expression_regulator = EpigeneticExpressionRegulator(
            self.config["knowledge_dimension"]
        )

        # Motor de adaptación
        self.adaptation_engine = TransgenerationalAdaptationEngine()

        logger.info("🧬 Epigenetic Memory System initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Configuración por defecto"""
        return {
            "knowledge_dimension": 512,
            "max_generations": 100,
            "mark_transmission_rate": 0.7,
            "expression_decay_rate": 0.05,
            "adaptation_threshold": 0.6,
            "inheritance_stability": 0.8,
            "epigenetic_memory_size": 1000,
        }

    async def initialize_epigenetic_memory(self) -> bool:
        """Inicializar sistema de memoria epigenética"""
        try:
            logger.info("🧬 Initializing Epigenetic Memory System...")

            # Crear generación inicial
            initial_generation = MemoryGeneration(
                generation_id=0,
                knowledge_genes={},
                epigenetic_marks=[],
                creation_timestamp=datetime.now(),
            )

            self.memory_generations[0] = initial_generation
            self.current_generation_id = 0

            logger.info("✅ Epigenetic Memory System initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Epigenetic memory initialization failed: {e}")
            return False

    async def store_experience(self, experience_data: Dict[str, Any]) -> str:
        """
        Almacenar experiencia con marcas epigenéticas

        Args:
            experience_data: Datos de la experiencia

        Returns:
            ID del gen de conocimiento creado
        """
        # Crear vector de conocimiento
        knowledge_content = experience_data.get("content", "")
        knowledge_vector = await self._encode_knowledge(knowledge_content)

        # Crear gen de conocimiento
        gene_id = f"gene_{hashlib.md5(str(experience_data).encode()).hexdigest()[:8]}_{self.current_generation_id}"

        gene = KnowledgeGene(
            gene_id=gene_id,
            knowledge_vector=knowledge_vector,
            base_expression=experience_data.get("importance", 0.5),
            generation=self.current_generation_id,
            metadata={
                "experience_type": experience_data.get("type", "general"),
                "emotional_impact": experience_data.get("emotional_impact", 0.0),
                "learning_context": experience_data.get("context", {}),
                "timestamp": datetime.now(),
            },
        )

        # Aplicar marcas epigenéticas basadas en la experiencia
        epigenetic_marks = await self._generate_epigenetic_marks(experience_data)
        gene.epigenetic_marks.extend(epigenetic_marks)

        # Calcular expresión inicial
        gene.expression_level = await self.expression_regulator.calculate_expression(
            gene.base_expression, gene.epigenetic_marks
        )

        # Almacenar gen
        self.knowledge_gene_pool[gene_id] = gene

        # Añadir a generación actual
        if self.current_generation_id in self.memory_generations:
            self.memory_generations[self.current_generation_id].knowledge_genes[
                gene_id
            ] = gene

        logger.info(f"🧬 Stored experience as knowledge gene: {gene_id}")

        return gene_id

    async def _encode_knowledge(self, knowledge_content: str) -> np.ndarray:
        """Codificar contenido de conocimiento en vector"""
        # Simulación simplificada - en producción usaría embeddings avanzados
        content_hash = hashlib.md5(knowledge_content.encode()).hexdigest()
        vector = np.array(
            [
                int(content_hash[i : i + 2], 16) / 255.0
                for i in range(0, min(64, len(content_hash)), 2)
            ]
        )

        # Rellenar a dimensión requerida
        while len(vector) < self.config["knowledge_dimension"]:
            vector = np.concatenate([vector, vector * 0.9])

        return vector[: self.config["knowledge_dimension"]]

    async def _generate_epigenetic_marks(
        self, experience_data: Dict[str, Any]
    ) -> List[EpigeneticMark]:
        """Generar marcas epigenéticas basadas en experiencia"""
        marks = []

        # Marca por intensidad emocional
        emotional_impact = experience_data.get("emotional_impact", 0.0)
        if emotional_impact > 0.3:
            mark = EpigeneticMark(
                mark_id=f"mark_emotion_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
                target_knowledge=experience_data.get("content", ""),
                modification_type=(
                    "acetylation" if emotional_impact > 0 else "methylation"
                ),
                intensity=emotional_impact,
                duration=timedelta(days=30),  # Efecto duradero
                created_at=datetime.now(),
                source_experience=experience_data.get("type", "unknown"),
                inheritance_probability=min(0.8, emotional_impact * 1.5),
            )
            marks.append(mark)

        # Marca por importancia de aprendizaje
        importance = experience_data.get("importance", 0.5)
        if importance > 0.7:
            mark = EpigeneticMark(
                mark_id=f"mark_importance_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
                target_knowledge=experience_data.get("content", ""),
                modification_type="phosphorylation",
                intensity=importance,
                duration=timedelta(days=90),  # Efecto muy duradero
                created_at=datetime.now(),
                source_experience=experience_data.get("type", "unknown"),
                inheritance_probability=min(0.9, importance),
            )
            marks.append(mark)

        # Marca por contexto de aprendizaje
        context = experience_data.get("context", {})
        if context.get("stressful", False):
            mark = EpigeneticMark(
                mark_id=f"mark_stress_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
                target_knowledge=experience_data.get("content", ""),
                modification_type="methylation",
                intensity=0.6,
                duration=timedelta(days=60),
                created_at=datetime.now(),
                source_experience="stressful_learning",
                inheritance_probability=0.5,
            )
            marks.append(mark)

        return marks

    async def retrieve_knowledge(
        self, query: str, context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Recuperar conocimientos con expresión epigenética

        Args:
            query: Consulta de búsqueda
            context: Contexto adicional

        Returns:
            Lista de conocimientos relevantes con expresión actual
        """
        # Buscar genes relevantes
        relevant_genes = await self._find_relevant_genes(query)

        # Calcular expresión actual para cada gen
        expressed_knowledge = []
        for gene in relevant_genes:
            current_expression = await self.expression_regulator.calculate_expression(
                gene.base_expression, gene.epigenetic_marks
            )

            # Actualizar expresión del gen
            gene.expression_level = current_expression

            if current_expression > self.config["expression_decay_rate"]:
                expressed_knowledge.append(
                    {
                        "gene_id": gene.gene_id,
                        "knowledge_vector": gene.knowledge_vector,
                        "expression_level": current_expression,
                        "stability": gene.stability,
                        "metadata": gene.metadata,
                        "epigenetic_marks": len(gene.epigenetic_marks),
                        "generation": gene.generation,
                    }
                )

        # Ordenar por expresión
        expressed_knowledge.sort(key=lambda x: x["expression_level"], reverse=True)

        return expressed_knowledge[:10]  # Top 10

    async def _find_relevant_genes(self, query: str) -> List[KnowledgeGene]:
        """Encontrar genes relevantes para la consulta"""
        query_vector = await self._encode_knowledge(query)
        relevant_genes = []

        for gene in self.knowledge_gene_pool.values():
            # Calcular similitud coseno
            similarity = np.dot(query_vector, gene.knowledge_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(gene.knowledge_vector)
            )

            if similarity > 0.3:  # Umbral de relevancia
                relevant_genes.append((gene, similarity))

        # Ordenar por similitud
        relevant_genes.sort(key=lambda x: x[1], reverse=True)

        return [gene for gene, _ in relevant_genes[:20]]  # Top 20 candidatos

    async def create_new_generation(self, adaptation_factors: Dict[str, Any]) -> int:
        """
        Crear nueva generación de memoria con herencia epigenética

        Args:
            adaptation_factors: Factores de adaptación para la nueva generación

        Returns:
            ID de la nueva generación
        """
        parent_generation_id = self.current_generation_id
        new_generation_id = parent_generation_id + 1

        # Obtener generación padre
        parent_generation = self.memory_generations.get(parent_generation_id)
        if not parent_generation:
            raise ValueError(f"Parent generation {parent_generation_id} not found")

        # Heredar genes con modificaciones epigenéticas
        inherited_genes = await self._inherit_knowledge_genes(
            parent_generation, adaptation_factors
        )

        # Crear nuevas marcas epigenéticas basadas en adaptación
        new_marks = await self._generate_adaptation_marks(adaptation_factors)

        # Crear nueva generación
        new_generation = MemoryGeneration(
            generation_id=new_generation_id,
            knowledge_genes=inherited_genes,
            epigenetic_marks=new_marks,
            creation_timestamp=datetime.now(),
            parent_generation_id=parent_generation_id,
        )

        # Evaluar fitness de la nueva generación
        new_generation.fitness_score = await self._evaluate_generation_fitness(
            new_generation, adaptation_factors
        )
        new_generation.adaptation_level = await self._calculate_adaptation_level(
            new_generation, adaptation_factors
        )

        # Almacenar nueva generación
        self.memory_generations[new_generation_id] = new_generation
        self.current_generation_id = new_generation_id

        # Registrar herencia
        inheritance = EpigeneticInheritance(
            parent_generation=parent_generation_id,
            child_generation=new_generation_id,
            inherited_marks=new_marks,
            transmission_efficiency=self.config["mark_transmission_rate"],
            adaptation_benefits=new_generation.adaptation_level,
            timestamp=datetime.now(),
        )

        self.inheritance_history.append(inheritance)

        logger.info(f"🧬 Created new memory generation: {new_generation_id}")

        return new_generation_id

    async def _inherit_knowledge_genes(
        self, parent_generation: MemoryGeneration, adaptation_factors: Dict[str, Any]
    ) -> Dict[str, KnowledgeGene]:
        """Heredar genes de conocimiento con modificaciones epigenéticas"""
        inherited_genes = {}

        for gene_id, parent_gene in parent_generation.knowledge_genes.items():
            # Decidir si heredar este gen
            inheritance_probability = await self._calculate_inheritance_probability(
                parent_gene, adaptation_factors
            )

            if random.random() < inheritance_probability:
                # Crear gen heredado
                inherited_gene = KnowledgeGene(
                    gene_id=f"{gene_id}_gen{parent_gene.generation + 1}",
                    knowledge_vector=parent_gene.knowledge_vector.copy(),
                    base_expression=parent_gene.base_expression,
                    epigenetic_marks=[],  # Se añadirán después
                    stability=parent_gene.stability,
                    generation=parent_gene.generation + 1,
                    metadata=parent_gene.metadata.copy(),
                )

                # Heredar marcas epigenéticas selectivamente
                inherited_marks = await self._inherit_epigenetic_marks(
                    parent_gene.epigenetic_marks, adaptation_factors
                )
                inherited_gene.epigenetic_marks = inherited_marks

                # Calcular nueva expresión
                inherited_gene.expression_level = (
                    await self.expression_regulator.calculate_expression(
                        inherited_gene.base_expression, inherited_gene.epigenetic_marks
                    )
                )

                inherited_genes[inherited_gene.gene_id] = inherited_gene

        return inherited_genes

    async def _calculate_inheritance_probability(
        self, gene: KnowledgeGene, adaptation_factors: Dict[str, Any]
    ) -> float:
        """Calcular probabilidad de herencia de un gen"""
        base_probability = 0.5

        # Modificar por estabilidad
        stability_factor = gene.stability

        # Modificar por expresión actual
        expression_factor = gene.expression_level

        # Modificar por factores de adaptación
        adaptation_relevance = adaptation_factors.get("relevance_threshold", 0.5)

        # Probabilidad final
        inheritance_prob = base_probability * stability_factor * expression_factor
        inheritance_prob = max(0.1, min(0.9, inheritance_prob))

        return inheritance_prob

    async def _inherit_epigenetic_marks(
        self, parent_marks: List[EpigeneticMark], adaptation_factors: Dict[str, Any]
    ) -> List[EpigeneticMark]:
        """Heredar marcas epigenéticas selectivamente"""
        inherited_marks = []

        for mark in parent_marks:
            # Verificar si la marca aún es válida (no expirada)
            if datetime.now() - mark.created_at < mark.duration:
                # Probabilidad de herencia
                if (
                    random.random()
                    < mark.inheritance_probability
                    * self.config["mark_transmission_rate"]
                ):
                    # Crear marca heredada con posible modificación
                    inherited_mark = EpigeneticMark(
                        mark_id=f"{mark.mark_id}_inherited_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:4]}",
                        target_knowledge=mark.target_knowledge,
                        modification_type=mark.modification_type,
                        intensity=mark.intensity
                        * random.uniform(0.8, 1.2),  # Ligera variación
                        duration=mark.duration,
                        created_at=datetime.now(),
                        source_experience=f"inherited_from_{mark.source_experience}",
                        inheritance_probability=mark.inheritance_probability
                        * 0.9,  # Se reduce con generaciones
                    )
                    inherited_marks.append(inherited_mark)

        return inherited_marks

    async def _generate_adaptation_marks(
        self, adaptation_factors: Dict[str, Any]
    ) -> List[EpigeneticMark]:
        """Generar nuevas marcas epigenéticas basadas en adaptación"""
        marks = []

        # Marca por adaptación al entorno
        if adaptation_factors.get("environmental_adaptation", 0) > 0.5:
            mark = EpigeneticMark(
                mark_id=f"adaptation_env_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
                target_knowledge="environmental_adaptation",
                modification_type="acetylation",
                intensity=adaptation_factors["environmental_adaptation"],
                duration=timedelta(days=45),
                created_at=datetime.now(),
                source_experience="environmental_adaptation",
                inheritance_probability=0.7,
            )
            marks.append(mark)

        # Marca por aprendizaje social
        if adaptation_factors.get("social_learning", 0) > 0.5:
            mark = EpigeneticMark(
                mark_id=f"adaptation_social_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
                target_knowledge="social_learning",
                modification_type="phosphorylation",
                intensity=adaptation_factors["social_learning"],
                duration=timedelta(days=60),
                created_at=datetime.now(),
                source_experience="social_learning",
                inheritance_probability=0.8,
            )
            marks.append(mark)

        return marks

    async def _evaluate_generation_fitness(
        self, generation: MemoryGeneration, adaptation_factors: Dict[str, Any]
    ) -> float:
        """Evaluar fitness de una generación"""
        if not generation.knowledge_genes:
            return 0.0

        # Fitness basado en expresión promedio
        avg_expression = np.mean(
            [gene.expression_level for gene in generation.knowledge_genes.values()]
        )

        # Fitness basado en estabilidad
        avg_stability = np.mean(
            [gene.stability for gene in generation.knowledge_genes.values()]
        )

        # Fitness basado en adaptación
        adaptation_score = generation.adaptation_level

        # Fitness combinado
        fitness = avg_expression * 0.4 + avg_stability * 0.3 + adaptation_score * 0.3

        return min(1.0, fitness)

    async def _calculate_adaptation_level(
        self, generation: MemoryGeneration, adaptation_factors: Dict[str, Any]
    ) -> float:
        """Calcular nivel de adaptación de la generación"""
        # Adaptación basada en marcas epigenéticas de adaptación
        adaptation_marks = [
            mark for mark in generation.epigenetic_marks if "adaptation" in mark.mark_id
        ]

        if not adaptation_marks:
            return 0.0

        # Promedio de intensidad de marcas de adaptación
        avg_adaptation_intensity = np.mean(
            [mark.intensity for mark in adaptation_marks]
        )

        # Factor de diversidad de adaptación
        adaptation_types = set(mark.modification_type for mark in adaptation_marks)
        diversity_factor = len(adaptation_types) / 3.0  # Máximo 3 tipos

        adaptation_level = (avg_adaptation_intensity + diversity_factor) / 2.0

        return min(1.0, adaptation_level)

    async def get_epigenetic_memory_status(self) -> Dict[str, Any]:
        """Obtener estado del sistema de memoria epigenética"""
        return {
            "current_generation": self.current_generation_id,
            "total_generations": len(self.memory_generations),
            "total_knowledge_genes": len(self.knowledge_gene_pool),
            "total_epigenetic_marks": sum(
                len(gen.epigenetic_marks) for gen in self.memory_generations.values()
            ),
            "inheritance_events": len(self.inheritance_history),
            "average_generation_fitness": (
                np.mean([gen.fitness_score for gen in self.memory_generations.values()])
                if self.memory_generations
                else 0.0
            ),
            "average_adaptation_level": (
                np.mean(
                    [gen.adaptation_level for gen in self.memory_generations.values()]
                )
                if self.memory_generations
                else 0.0
            ),
            "config": self.config,
        }

    async def decay_epigenetic_marks(self):
        """Decaer marcas epigenéticas expiradas"""
        current_time = datetime.now()

        for generation in self.memory_generations.values():
            # Decaer marcas en genes
            for gene in generation.knowledge_genes.values():
                active_marks = []
                for mark in gene.epigenetic_marks:
                    if current_time - mark.created_at < mark.duration:
                        active_marks.append(mark)
                    else:
                        # Marcar como expirada
                        mark.intensity *= 0.1  # Reducir drasticamente

                gene.epigenetic_marks = active_marks

                # Recalcular expresión
                gene.expression_level = (
                    await self.expression_regulator.calculate_expression(
                        gene.base_expression, gene.epigenetic_marks
                    )
                )

            # Decaer marcas en generación
            generation.epigenetic_marks = [
                mark
                for mark in generation.epigenetic_marks
                if current_time - mark.created_at < mark.duration
            ]


# Componentes auxiliares


class EpigeneticExpressionRegulator:
    """Regulador de expresión epigenética"""

    def __init__(self, knowledge_dimension: int):
        self.knowledge_dimension = knowledge_dimension

    async def calculate_expression(
        self, base_expression: float, epigenetic_marks: List[EpigeneticMark]
    ) -> float:
        """Calcular nivel de expresión considerando marcas epigenéticas"""
        expression = base_expression

        for mark in epigenetic_marks:
            if mark.modification_type == "acetylation":
                # Aumenta expresión
                expression += mark.intensity * 0.3
            elif mark.modification_type == "methylation":
                # Reduce expresión
                expression -= mark.intensity * 0.2
            elif mark.modification_type == "phosphorylation":
                # Modula expresión
                expression *= 1 + mark.intensity * 0.1

        # Limitar entre 0 y 1
        return max(0.0, min(1.0, expression))


class TransgenerationalAdaptationEngine:
    """Motor de adaptación transgeneracional"""

    async def evaluate_adaptation(
        self, generation: MemoryGeneration, environmental_factors: Dict[str, Any]
    ) -> float:
        """Evaluar adaptación de una generación a factores ambientales"""
        # Implementación simplificada
        adaptation_score = 0.5

        # Adaptación basada en diversidad de genes
        gene_diversity = len(generation.knowledge_genes) / 100.0  # Normalizar
        adaptation_score += gene_diversity * 0.2

        # Adaptación basada en marcas epigenéticas
        epigenetic_diversity = len(generation.epigenetic_marks) / 50.0  # Normalizar
        adaptation_score += epigenetic_diversity * 0.3

        return min(1.0, adaptation_score)


# Instancia global
epigenetic_memory_system = EpigeneticMemorySystem()


async def store_epigenetic_experience(experience_data: Dict[str, Any]) -> str:
    """Función pública para almacenar experiencia epigenética"""
    return await epigenetic_memory_system.store_experience(experience_data)


async def retrieve_epigenetic_knowledge(
    query: str, context: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """Función pública para recuperar conocimiento epigenético"""
    return await epigenetic_memory_system.retrieve_knowledge(query, context)


async def create_epigenetic_generation(adaptation_factors: Dict[str, Any]) -> int:
    """Función pública para crear nueva generación epigenética"""
    return await epigenetic_memory_system.create_new_generation(adaptation_factors)


async def get_epigenetic_memory_status() -> Dict[str, Any]:
    """Función pública para estado de memoria epigenética"""
    return await epigenetic_memory_system.get_epigenetic_memory_status()


async def initialize_epigenetic_memory() -> bool:
    """Función pública para inicializar memoria epigenética"""
    return await epigenetic_memory_system.initialize_epigenetic_memory()


# Información del módulo
__version__ = "1.0.0"
__author__ = "Sheily AI Team - Epigenetic Memory System"
__description__ = "Sistema de memoria epigenética con herencia transgeneracional"
