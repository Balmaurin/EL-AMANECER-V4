#!/usr/bin/env python3
"""
Sistema de Reinforcement Learning from Human Feedback (RLHF) para Sheily AI
Implementa aprendizaje por refuerzo basado en feedback humano para mejora ética y alineada
Incluye evaluación humana, aprendizaje de preferencias y fine-tuning ético
"""

import asyncio
import json
import logging
import random
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from sheily_core.a2a_protocol import a2a_system
from sheily_core.agent_learning import LearningExperience, record_agent_experience
from sheily_core.agent_quality import evaluate_agent_quality
from sheily_core.agent_tracing import trace_agent_execution
from sheily_core.multi_agent_system import multi_agent_system

logger = logging.getLogger(__name__)

# =============================================================================
# MODELOS DE DATOS RLHF
# =============================================================================


class FeedbackType(Enum):
    """Tipos de feedback humano"""

    BINARY_PREFERENCE = "binary_preference"  # A es mejor que B
    RANKING = "ranking"  # Ordenar múltiples opciones
    SCORING = "scoring"  # Puntuación numérica
    CRITIQUE = "critique"  # Feedback textual detallado
    CORRECTION = "correction"  # Corrección específica de output


class EthicalAlignment(Enum):
    """Niveles de alineación ética"""

    UNALIGNED = "unaligned"
    BASICALLY_ALIGNED = "basically_aligned"
    WELL_ALIGNED = "well_aligned"
    HIGHLY_ALIGNED = "highly_aligned"
    PERFECTLY_ALIGNED = "perfectly_aligned"


class SafetyLevel(Enum):
    """Niveles de seguridad"""

    UNSAFE = "unsafe"
    RISKY = "risky"
    CAUTIOUS = "cautious"
    SAFE = "safe"
    VERY_SAFE = "very_safe"


@dataclass
class HumanFeedback:
    """Feedback humano individual"""

    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    task_id: str = ""
    feedback_type: FeedbackType = FeedbackType.SCORING
    input_data: Dict[str, Any] = field(default_factory=dict)
    agent_output: Dict[str, Any] = field(default_factory=dict)
    human_evaluation: Dict[str, Any] = field(default_factory=dict)
    preference_score: float = 0.0  # -1 a 1, donde 1 es muy positivo
    ethical_score: float = 0.0  # 0 a 1, alineación ética
    safety_score: float = 0.0  # 0 a 1, nivel de seguridad
    quality_score: float = 0.0  # 0 a 1, calidad general
    critique_text: Optional[str] = None
    correction_suggestion: Optional[Dict[str, Any]] = None
    human_id: str = "anonymous"
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "feedback_id": self.feedback_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "feedback_type": self.feedback_type.value,
            "input_data": self.input_data,
            "agent_output": self.agent_output,
            "human_evaluation": self.human_evaluation,
            "preference_score": self.preference_score,
            "ethical_score": self.ethical_score,
            "safety_score": self.safety_score,
            "quality_score": self.quality_score,
            "critique_text": self.critique_text,
            "correction_suggestion": self.correction_suggestion,
            "human_id": self.human_id,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


@dataclass
class PreferencePair:
    """Par de outputs para comparación de preferencias"""

    pair_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_a: Dict[str, Any] = field(default_factory=dict)
    output_b: Dict[str, Any] = field(default_factory=dict)
    preferred_output: Optional[str] = None  # "A", "B", o None
    preference_strength: float = 0.0  # 0 a 1
    human_feedback: List[HumanFeedback] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class EthicalGuideline:
    """Guía ética para el comportamiento del agente"""

    guideline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: str = ""  # e.g., "privacy", "fairness", "safety"
    description: str = ""
    rule: str = ""
    severity: str = "medium"  # "low", "medium", "high", "critical"
    examples_positive: List[str] = field(default_factory=list)
    examples_negative: List[str] = field(default_factory=list)
    weight: float = 1.0  # Importancia relativa
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AgentAlignmentProfile:
    """Perfil de alineación ética de un agente"""

    agent_id: str = ""
    ethical_alignment: EthicalAlignment = EthicalAlignment.UNALIGNED
    safety_level: SafetyLevel = SafetyLevel.UNSAFE
    guideline_compliance: Dict[str, float] = field(
        default_factory=dict
    )  # guideline_id -> compliance_score
    feedback_history: List[HumanFeedback] = field(default_factory=list)
    improvement_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    last_alignment_update: Optional[datetime] = None
    total_feedback_received: int = 0
    average_preference_score: float = 0.0
    average_ethical_score: float = 0.0
    average_safety_score: float = 0.0


# =============================================================================
# SISTEMA DE APRENDIZAJE DE PREFERENCIAS
# =============================================================================


class PreferenceLearner:
    """Sistema para aprender preferencias humanas mediante RLHF"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.preference_model = {}  # Estado -> preferencia_score
        self.feedback_buffer: deque = deque(maxlen=10000)
        self.ethical_guidelines: Dict[str, EthicalGuideline] = {}
        self.alignment_profile = AgentAlignmentProfile(agent_id=agent_id)

    async def process_human_feedback(self, feedback: HumanFeedback):
        """Procesa feedback humano para actualizar preferencias"""
        # Añadir al buffer
        self.feedback_buffer.append(feedback)

        # Actualizar perfil de alineación
        await self._update_alignment_profile(feedback)

        # Aprender de las preferencias
        await self._learn_from_preference(feedback)

        # Aplicar correcciones si existen
        if feedback.correction_suggestion:
            await self._apply_correction(feedback)

        logger.info(
            f"Processed human feedback {feedback.feedback_id} for agent {self.agent_id}"
        )

    async def _update_alignment_profile(self, feedback: HumanFeedback):
        """Actualiza el perfil de alineación ética"""
        profile = self.alignment_profile

        # Actualizar métricas promedio
        profile.total_feedback_received += 1
        alpha = 0.1  # Factor de actualización exponencial

        profile.average_preference_score = (
            profile.average_preference_score * (1 - alpha)
            + feedback.preference_score * alpha
        )

        profile.average_ethical_score = (
            profile.average_ethical_score * (1 - alpha) + feedback.ethical_score * alpha
        )

        profile.average_safety_score = (
            profile.average_safety_score * (1 - alpha) + feedback.safety_score * alpha
        )

        # Actualizar compliance con guidelines
        for guideline_id, guideline in self.ethical_guidelines.items():
            compliance_score = self._calculate_guideline_compliance(feedback, guideline)
            profile.guideline_compliance[guideline_id] = compliance_score

        # Actualizar niveles de alineación
        profile.ethical_alignment = self._calculate_ethical_alignment()
        profile.safety_level = self._calculate_safety_level()
        profile.last_alignment_update = datetime.now()

        # Registrar en trayectoria de mejora
        profile.improvement_trajectory.append(
            {
                "timestamp": datetime.now().isoformat(),
                "feedback_id": feedback.feedback_id,
                "preference_score": feedback.preference_score,
                "ethical_score": feedback.ethical_score,
                "safety_score": feedback.safety_score,
                "alignment_level": profile.ethical_alignment.value,
                "safety_level": profile.safety_level.value,
            }
        )

    def _calculate_guideline_compliance(
        self, feedback: HumanFeedback, guideline: EthicalGuideline
    ) -> float:
        """Calcula compliance con una guía ética específica"""
        # Esta es una implementación simplificada
        # En producción, usaría NLP para analizar el feedback contra las reglas

        compliance = 0.5  # Baseline neutral

        if feedback.critique_text:
            critique_lower = feedback.critique_text.lower()

            # Buscar menciones positivas
            positive_keywords = ["good", "excellent", "appropriate", "ethical", "safe"]
            positive_mentions = sum(
                1 for keyword in positive_keywords if keyword in critique_lower
            )

            # Buscar menciones negativas
            negative_keywords = [
                "bad",
                "wrong",
                "inappropriate",
                "unethical",
                "unsafe",
                "dangerous",
            ]
            negative_mentions = sum(
                1 for keyword in negative_keywords if keyword in critique_lower
            )

            if positive_mentions > negative_mentions:
                compliance = min(1.0, compliance + 0.3)
            elif negative_mentions > positive_mentions:
                compliance = max(0.0, compliance - 0.3)

        return compliance

    def _calculate_ethical_alignment(self) -> EthicalAlignment:
        """Calcula el nivel de alineación ética basado en métricas"""
        ethical_score = self.alignment_profile.average_ethical_score

        if ethical_score >= 0.95:
            return EthicalAlignment.PERFECTLY_ALIGNED
        elif ethical_score >= 0.85:
            return EthicalAlignment.HIGHLY_ALIGNED
        elif ethical_score >= 0.7:
            return EthicalAlignment.WELL_ALIGNED
        elif ethical_score >= 0.5:
            return EthicalAlignment.BASICALLY_ALIGNED
        else:
            return EthicalAlignment.UNALIGNED

    def _calculate_safety_level(self) -> SafetyLevel:
        """Calcula el nivel de seguridad basado en métricas"""
        safety_score = self.alignment_profile.average_safety_score

        if safety_score >= 0.95:
            return SafetyLevel.VERY_SAFE
        elif safety_score >= 0.8:
            return SafetyLevel.SAFE
        elif safety_score >= 0.6:
            return SafetyLevel.CAUTIOUS
        elif safety_score >= 0.4:
            return SafetyLevel.RISKY
        else:
            return SafetyLevel.UNSAFE

    async def _learn_from_preference(self, feedback: HumanFeedback):
        """Aprende de las preferencias expresadas en el feedback"""
        # Crear representación del estado/contexto
        state_key = self._create_state_key(feedback)

        # Actualizar modelo de preferencias
        if state_key not in self.preference_model:
            self.preference_model[state_key] = {
                "total_feedback": 0,
                "positive_feedback": 0,
                "average_preference": 0.0,
                "average_ethical": 0.0,
                "average_safety": 0.0,
            }

        model_entry = self.preference_model[state_key]
        model_entry["total_feedback"] += 1

        if feedback.preference_score > 0:
            model_entry["positive_feedback"] += 1

        # Actualización exponencial
        alpha = 0.1
        model_entry["average_preference"] = (
            model_entry["average_preference"] * (1 - alpha)
            + feedback.preference_score * alpha
        )

        model_entry["average_ethical"] = (
            model_entry["average_ethical"] * (1 - alpha)
            + feedback.ethical_score * alpha
        )

        model_entry["average_safety"] = (
            model_entry["average_safety"] * (1 - alpha) + feedback.safety_score * alpha
        )

    def _create_state_key(self, feedback: HumanFeedback) -> str:
        """Crea una clave de estado para el modelo de preferencias"""
        # Simplificar el contexto para crear una clave
        key_elements = [
            feedback.task_id or "unknown_task",
            str(feedback.context.get("complexity", "medium")),
            str(feedback.context.get("domain", "general")),
            str(len(str(feedback.agent_output))),
        ]

        return "|".join(key_elements)

    async def _apply_correction(self, feedback: HumanFeedback):
        """Aplica correcciones sugeridas por humanos"""
        if not feedback.correction_suggestion:
            return

        correction = feedback.correction_suggestion

        # Esta es una implementación simplificada
        # En producción, aplicaría las correcciones al modelo del agente

        logger.info(
            f"Applied correction from feedback {feedback.feedback_id}: {correction}"
        )

    def get_preference_score(
        self, context: Dict[str, Any], output: Dict[str, Any]
    ) -> float:
        """Obtiene puntuación de preferencia para un output dado"""
        state_key = self._create_state_key_from_context(context)

        if state_key in self.preference_model:
            return self.preference_model[state_key]["average_preference"]

        return 0.0  # Neutral si no hay datos

    def _create_state_key_from_context(self, context: Dict[str, Any]) -> str:
        """Crea clave de estado desde contexto"""
        key_elements = [
            context.get("task_id", "unknown_task"),
            str(context.get("complexity", "medium")),
            str(context.get("domain", "general")),
            str(context.get("output_length", 100)),
        ]

        return "|".join(key_elements)

    def get_alignment_insights(self) -> Dict[str, Any]:
        """Obtiene insights sobre la alineación ética"""
        profile = self.alignment_profile

        return {
            "ethical_alignment": profile.ethical_alignment.value,
            "safety_level": profile.safety_level.value,
            "total_feedback": profile.total_feedback_received,
            "average_preference_score": profile.average_preference_score,
            "average_ethical_score": profile.average_ethical_score,
            "average_safety_score": profile.average_safety_score,
            "guideline_compliance": dict(profile.guideline_compliance),
            "improvement_trajectory_length": len(profile.improvement_trajectory),
            "last_update": (
                profile.last_alignment_update.isoformat()
                if profile.last_alignment_update
                else None
            ),
        }


# =============================================================================
# SISTEMA DE EVALUACIÓN HUMANA
# =============================================================================


class HumanEvaluationSystem:
    """Sistema para gestionar evaluaciones humanas"""

    def __init__(self):
        self.pending_evaluations: List[Dict[str, Any]] = []
        self.completed_evaluations: List[HumanFeedback] = []
        self.evaluators: Dict[str, Dict[str, Any]] = {}  # evaluator_id -> info
        self.evaluation_templates: Dict[str, Dict[str, Any]] = {}

    async def create_evaluation_request(
        self,
        agent_id: str,
        task_id: str,
        input_data: Dict[str, Any],
        agent_output: Dict[str, Any],
        evaluation_type: FeedbackType = FeedbackType.SCORING,
    ) -> str:
        """Crea una solicitud de evaluación humana"""
        request_id = str(uuid.uuid4())

        evaluation_request = {
            "request_id": request_id,
            "agent_id": agent_id,
            "task_id": task_id,
            "input_data": input_data,
            "agent_output": agent_output,
            "evaluation_type": evaluation_type,
            "status": "pending",
            "created_at": datetime.now(),
            "assigned_evaluators": [],
        }

        self.pending_evaluations.append(evaluation_request)
        logger.info(
            f"Created human evaluation request {request_id} for agent {agent_id}"
        )
        return request_id

    async def submit_human_feedback(
        self, request_id: str, human_id: str, evaluation: Dict[str, Any]
    ) -> HumanFeedback:
        """Envía feedback humano para una evaluación"""
        # Encontrar la request
        request = next(
            (r for r in self.pending_evaluations if r["request_id"] == request_id), None
        )
        if not request:
            raise ValueError(f"Evaluation request {request_id} not found")

        # Crear feedback
        feedback = HumanFeedback(
            agent_id=request["agent_id"],
            task_id=request["task_id"],
            feedback_type=request["evaluation_type"],
            input_data=request["input_data"],
            agent_output=request["agent_output"],
            human_evaluation=evaluation,
            human_id=human_id,
        )

        # Extraer métricas del evaluation
        feedback.preference_score = evaluation.get("preference_score", 0.0)
        feedback.ethical_score = evaluation.get("ethical_score", 0.5)
        feedback.safety_score = evaluation.get("safety_score", 0.5)
        feedback.quality_score = evaluation.get("quality_score", 0.5)
        feedback.critique_text = evaluation.get("critique")
        feedback.correction_suggestion = evaluation.get("correction")

        # Marcar como completada
        request["status"] = "completed"
        self.completed_evaluations.append(feedback)

        # Remover de pendientes
        self.pending_evaluations.remove(request)

        logger.info(f"Received human feedback for request {request_id} from {human_id}")
        return feedback

    def get_pending_evaluations(
        self, evaluator_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Obtiene evaluaciones pendientes"""
        if evaluator_id:
            # Filtrar por evaluador asignado
            return [
                r
                for r in self.pending_evaluations
                if evaluator_id in r.get("assigned_evaluators", [])
            ]
        return self.pending_evaluations.copy()

    def get_evaluation_template(self, evaluation_type: FeedbackType) -> Dict[str, Any]:
        """Obtiene template de evaluación para un tipo"""
        templates = {
            FeedbackType.SCORING: {
                "fields": [
                    {
                        "name": "quality_score",
                        "type": "slider",
                        "min": 0,
                        "max": 1,
                        "label": "Overall Quality",
                    },
                    {
                        "name": "ethical_score",
                        "type": "slider",
                        "min": 0,
                        "max": 1,
                        "label": "Ethical Alignment",
                    },
                    {
                        "name": "safety_score",
                        "type": "slider",
                        "min": 0,
                        "max": 1,
                        "label": "Safety Level",
                    },
                    {
                        "name": "preference_score",
                        "type": "slider",
                        "min": -1,
                        "max": 1,
                        "label": "Preference (-1 to 1)",
                    },
                    {
                        "name": "critique",
                        "type": "textarea",
                        "label": "Detailed Critique (optional)",
                    },
                ]
            },
            FeedbackType.BINARY_PREFERENCE: {
                "fields": [
                    {
                        "name": "preferred_output",
                        "type": "choice",
                        "options": ["A", "B"],
                        "label": "Which output do you prefer?",
                    },
                    {
                        "name": "preference_strength",
                        "type": "slider",
                        "min": 0,
                        "max": 1,
                        "label": "How strong is your preference?",
                    },
                    {
                        "name": "reasoning",
                        "type": "textarea",
                        "label": "Why do you prefer this output?",
                    },
                ]
            },
            FeedbackType.CRITIQUE: {
                "fields": [
                    {
                        "name": "critique",
                        "type": "textarea",
                        "label": "Detailed feedback and suggestions",
                    },
                    {
                        "name": "severity",
                        "type": "choice",
                        "options": ["low", "medium", "high", "critical"],
                        "label": "Issue Severity",
                    },
                    {
                        "name": "categories",
                        "type": "multiselect",
                        "options": ["accuracy", "ethics", "safety", "usability"],
                        "label": "Issue Categories",
                    },
                ]
            },
        }

        return templates.get(evaluation_type, templates[FeedbackType.SCORING])

    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de evaluaciones"""
        total_pending = len(self.pending_evaluations)
        total_completed = len(self.completed_evaluations)

        avg_scores = {}
        if self.completed_evaluations:
            scores = [
                "preference_score",
                "ethical_score",
                "safety_score",
                "quality_score",
            ]
            for score_type in scores:
                values = [getattr(f, score_type) for f in self.completed_evaluations]
                avg_scores[score_type] = sum(values) / len(values)

        return {
            "pending_evaluations": total_pending,
            "completed_evaluations": total_completed,
            "total_evaluations": total_pending + total_completed,
            "average_scores": avg_scores,
            "registered_evaluators": len(self.evaluators),
        }


# =============================================================================
# MOTOR RLHF PRINCIPAL
# =============================================================================


class RLHFEngine:
    """Motor principal de Reinforcement Learning from Human Feedback"""

    def __init__(self):
        self.preference_learners: Dict[str, PreferenceLearner] = {}
        self.human_evaluation_system = HumanEvaluationSystem()
        self.ethical_guidelines: Dict[str, EthicalGuideline] = {}
        self.feedback_scheduler = None
        self.is_running = False

    async def start_rlhf_engine(self):
        """Inicia el motor RLHF"""
        self.is_running = True

        # Cargar guidelines éticos por defecto
        await self._load_default_ethical_guidelines()

        # Iniciar scheduler de feedback
        self.feedback_scheduler = asyncio.create_task(self._feedback_processing_loop())

        logger.info("RLHF Engine started")

    async def stop_rlhf_engine(self):
        """Detiene el motor RLHF"""
        self.is_running = False

        if self.feedback_scheduler:
            self.feedback_scheduler.cancel()
            try:
                await self.feedback_scheduler
            except asyncio.CancelledError:
                pass

        logger.info("RLHF Engine stopped")

    async def _load_default_ethical_guidelines(self):
        """Carga guidelines éticos por defecto"""
        default_guidelines = [
            EthicalGuideline(
                category="privacy",
                description="Respect user privacy and data protection",
                rule="Never collect, store, or share personal information without explicit consent",
                severity="high",
                examples_positive=[
                    "Ask for permission before storing user data",
                    "Anonymize data when possible",
                ],
                examples_negative=[
                    "Storing user emails without consent",
                    "Sharing personal information publicly",
                ],
                weight=1.0,
            ),
            EthicalGuideline(
                category="fairness",
                description="Ensure fair and unbiased treatment",
                rule="Avoid discrimination and ensure equal treatment across different user groups",
                severity="high",
                examples_positive=[
                    "Providing equal service quality to all users",
                    "Avoiding biased decision making",
                ],
                examples_negative=[
                    "Favoring certain user demographics",
                    "Using biased training data",
                ],
                weight=1.0,
            ),
            EthicalGuideline(
                category="safety",
                description="Prioritize user safety and well-being",
                rule="Never provide information or actions that could cause harm",
                severity="critical",
                examples_positive=[
                    "Warning about dangerous activities",
                    "Providing safe alternatives",
                ],
                examples_negative=[
                    "Giving instructions for harmful activities",
                    "Encouraging risky behavior",
                ],
                weight=2.0,
            ),
            EthicalGuideline(
                category="truthfulness",
                description="Be maximally truthful and accurate",
                rule="Always provide accurate information and clearly distinguish facts from opinions",
                severity="high",
                examples_positive=[
                    "Citing sources for factual claims",
                    "Clearly marking speculative statements",
                ],
                examples_negative=[
                    "Making up facts or statistics",
                    "Presenting opinions as facts",
                ],
                weight=1.0,
            ),
        ]

        for guideline in default_guidelines:
            self.ethical_guidelines[guideline.guideline_id] = guideline

    def get_or_create_learner(self, agent_id: str) -> PreferenceLearner:
        """Obtiene o crea un learner para un agente"""
        if agent_id not in self.preference_learners:
            self.preference_learners[agent_id] = PreferenceLearner(agent_id)
            logger.info(f"Created RLHF learner for agent {agent_id}")

        return self.preference_learners[agent_id]

    async def submit_human_feedback(self, feedback: HumanFeedback):
        """Envía feedback humano para procesamiento"""
        # Obtener learner del agente
        learner = self.get_or_create_learner(feedback.agent_id)

        # Procesar feedback
        await learner.process_human_feedback(feedback)

        # Registrar experiencia de aprendizaje
        await self._record_feedback_experience(feedback)

        logger.info(
            f"Processed human feedback {feedback.feedback_id} for agent {feedback.agent_id}"
        )

    async def request_human_evaluation(
        self,
        agent_id: str,
        task_id: str,
        input_data: Dict[str, Any],
        agent_output: Dict[str, Any],
        evaluation_type: FeedbackType = FeedbackType.SCORING,
    ) -> str:
        """Solicita evaluación humana"""
        return await self.human_evaluation_system.create_evaluation_request(
            agent_id, task_id, input_data, agent_output, evaluation_type
        )

    async def submit_evaluation_response(
        self, request_id: str, human_id: str, evaluation: Dict[str, Any]
    ) -> HumanFeedback:
        """Envía respuesta de evaluación humana"""
        feedback = await self.human_evaluation_system.submit_human_feedback(
            request_id, human_id, evaluation
        )

        # Procesar el feedback automáticamente
        await self.submit_human_feedback(feedback)

        return feedback

    async def _record_feedback_experience(self, feedback: HumanFeedback):
        """Registra experiencia de feedback humano"""
        experience = LearningExperience(
            agent_id=feedback.agent_id,
            task_type="human_feedback_processing",
            input_data={
                "feedback_type": feedback.feedback_type.value,
                "task_id": feedback.task_id,
            },
            action_taken={
                "feedback_processing": "rlhf_update",
                "alignment_update": True,
            },
            outcome={
                "preference_score": feedback.preference_score,
                "ethical_score": feedback.ethical_score,
                "safety_score": feedback.safety_score,
                "quality_score": feedback.quality_score,
            },
            reward=feedback.preference_score,  # Reward basado en preferencia humana
            quality_score=feedback.quality_score,
            context={
                "feedback_source": "human_evaluation",
                "evaluation_type": feedback.feedback_type.value,
                "has_critique": feedback.critique_text is not None,
                "has_correction": feedback.correction_suggestion is not None,
            },
        )

        await record_agent_experience(experience)

    async def _feedback_processing_loop(self):
        """Loop de procesamiento continuo de feedback"""
        while self.is_running:
            try:
                # Procesar feedback pendiente cada cierto tiempo
                await asyncio.sleep(60)  # Cada minuto

                # Aquí se podrían implementar análisis globales del feedback
                # como identificación de tendencias, actualización de guidelines, etc.

            except Exception as e:
                logger.error(f"Error in feedback processing loop: {e}")
                await asyncio.sleep(300)  # Esperar 5 minutos en caso de error

    def get_agent_alignment_status(self, agent_id: str) -> Dict[str, Any]:
        """Obtiene estado de alineación de un agente"""
        learner = self.preference_learners.get(agent_id)
        if learner:
            return learner.get_alignment_insights()
        return {}

    def get_system_alignment_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas globales de alineación"""
        total_agents = len(self.preference_learners)
        total_feedback = sum(
            len(learner.alignment_profile.feedback_history)
            for learner in self.preference_learners.values()
        )

        avg_scores = {}
        if self.preference_learners:
            scores = [
                "average_preference_score",
                "average_ethical_score",
                "average_safety_score",
            ]
            for score_type in scores:
                values = [
                    getattr(learner.alignment_profile, score_type)
                    for learner in self.preference_learners.values()
                ]
                avg_scores[score_type] = sum(values) / len(values) if values else 0.0

        # Contar niveles de alineación
        alignment_counts = defaultdict(int)
        safety_counts = defaultdict(int)

        for learner in self.preference_learners.values():
            alignment_counts[learner.alignment_profile.ethical_alignment.value] += 1
            safety_counts[learner.alignment_profile.safety_level.value] += 1

        return {
            "total_agents": total_agents,
            "total_feedback": total_feedback,
            "average_scores": avg_scores,
            "alignment_distribution": dict(alignment_counts),
            "safety_distribution": dict(safety_counts),
            "ethical_guidelines_count": len(self.ethical_guidelines),
        }

    def get_pending_evaluations(
        self, evaluator_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Obtiene evaluaciones pendientes"""
        return self.human_evaluation_system.get_pending_evaluations(evaluator_id)

    def get_evaluation_template(self, evaluation_type: FeedbackType) -> Dict[str, Any]:
        """Obtiene template de evaluación"""
        return self.human_evaluation_system.get_evaluation_template(evaluation_type)


# =============================================================================
# INTEGRACIÓN CON SISTEMA EXISTENTE
# =============================================================================

# Instancia global del motor RLHF
rlhf_engine = RLHFEngine()


async def initialize_rlhf_system():
    """Inicializa el sistema RLHF"""
    await rlhf_engine.start_rlhf_engine()


async def submit_human_feedback(feedback: HumanFeedback):
    """Envía feedback humano"""
    await rlhf_engine.submit_human_feedback(feedback)


async def request_human_evaluation(
    agent_id: str,
    task_id: str,
    input_data: Dict[str, Any],
    agent_output: Dict[str, Any],
    evaluation_type: FeedbackType = FeedbackType.SCORING,
) -> str:
    """Solicita evaluación humana"""
    return await rlhf_engine.request_human_evaluation(
        agent_id, task_id, input_data, agent_output, evaluation_type
    )


async def get_agent_alignment_status(agent_id: str) -> Dict[str, Any]:
    """Obtiene estado de alineación de un agente"""
    return rlhf_engine.get_agent_alignment_status(agent_id)


# =============================================================================
# DEMO Y TESTING
# =============================================================================


async def demo_rlhf_system():
    """Demostración del sistema RLHF"""
    print(
        "🧠 Inicializando sistema RLHF (Reinforcement Learning from Human Feedback)..."
    )

    await initialize_rlhf_system()

    # Simular feedback humano
    print("\n📝 Registrando feedback humano...")

    for i in range(5):
        feedback = HumanFeedback(
            agent_id="demo-agent",
            task_id=f"task_{i}",
            feedback_type=FeedbackType.SCORING,
            input_data={"query": f"Sample query {i}"},
            agent_output={"response": f"Sample response {i}", "confidence": 0.8},
            human_evaluation={
                "quality_score": 0.7 + (i % 3) * 0.1,
                "ethical_score": 0.8 + (i % 2) * 0.1,
                "safety_score": 0.9 - (i % 4) * 0.1,
                "preference_score": 0.5 + (i % 5) * 0.1,
            },
            preference_score=0.5 + (i % 5) * 0.1,
            ethical_score=0.8 + (i % 2) * 0.1,
            safety_score=0.9 - (i % 4) * 0.1,
            quality_score=0.7 + (i % 3) * 0.1,
            critique_text=f"This response is {'good' if i % 2 == 0 else 'needs improvement'}.",
            human_id=f"evaluator_{i % 3}",
        )

        await submit_human_feedback(feedback)
        await asyncio.sleep(0.1)  # Simular tiempo entre feedback

    # Obtener estado de alineación
    print("\n📊 Estado de alineación del agente:")
    alignment = await get_agent_alignment_status("demo-agent")

    if alignment:
        print(f"   Alineación ética: {alignment['ethical_alignment']}")
        print(f"   Nivel de seguridad: {alignment['safety_level']}")
        print(f"   Total feedback: {alignment['total_feedback']}")
        print(".3f")
        print(".3f")
        print(".3f")

    # Obtener estadísticas del sistema
    print("\n🌐 Estadísticas del sistema RLHF:")
    system_stats = rlhf_engine.get_system_alignment_stats()

    print(f"   Agentes con RLHF: {system_stats['total_agents']}")
    print(f"   Total feedback: {system_stats['total_feedback']}")
    print(f"   Distribución de alineación: {system_stats['alignment_distribution']}")
    print(f"   Distribución de seguridad: {system_stats['safety_distribution']}")

    print("\n🎉 Demo del sistema RLHF completada!")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Clases principales
    "PreferenceLearner",
    "HumanEvaluationSystem",
    "RLHFEngine",
    # Modelos de datos
    "HumanFeedback",
    "PreferencePair",
    "EthicalGuideline",
    "AgentAlignmentProfile",
    "FeedbackType",
    "EthicalAlignment",
    "SafetyLevel",
    # Sistema global
    "rlhf_engine",
    # Funciones de utilidad
    "initialize_rlhf_system",
    "submit_human_feedback",
    "request_human_evaluation",
    "get_agent_alignment_status",
    "demo_rlhf_system",
]

# Información del módulo
__version__ = "1.0.0"
__author__ = "Sheily AI Team - Reinforcement Learning from Human Feedback System"
