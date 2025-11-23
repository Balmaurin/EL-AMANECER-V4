#!/usr/bin/env python3
"""
Sistema Universal de Optimización Automática de Prompts
Compatible con cualquier LLM - OpenAI, Anthropic, Local LLMs, etc.
Implementa todas las técnicas del Prompt Engineering Guide.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Proveedores de LLM soportados"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LLAMA_CPP = "llama_cpp"
    VERTEX_AI = "vertex_ai"
    CUSTOM = "custom"


@dataclass
class PromptEvaluation:
    """Resultado de evaluación de un prompt"""

    score: float  # 0-100
    metrics: Dict[str, float]
    reasoning: str
    improvements: List[str]


@dataclass
class OptimizationResult:
    """Resultado de optimización con mejoras avanzadas"""

    original_prompt: str
    optimized_prompt: str
    evaluation: PromptEvaluation
    iterations: int
    technique_used: str
    advanced_features: Dict[str, Any] = None


class LLMAdapter(ABC):
    """Adapter abstracto para cualquier LLM"""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generar respuesta del LLM"""
        pass

    @abstractmethod
    def get_max_context_length(self) -> int:
        """Longitud máxima de contexto"""
        pass


class OpenAIAdapter(LLMAdapter):
    """Adapter para OpenAI API"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("Instala openai: pip install openai")

    async def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], **kwargs
        )
        return response.choices[0].message.content

    def get_max_context_length(self) -> int:
        # Maps for max tokens
        model_limits = {"gpt-4": 8192, "gpt-3.5-turbo": 4096, "gpt-4-turbo": 128000}
        return model_limits.get(self.model, 4096)


class LlamaCppAdapter(LLMAdapter):
    """Adapter para modelos locales con llama.cpp"""

    def __init__(self, model_path: str):
        try:
            from llama_cpp import Llama

            self.llm = Llama(model_path=model_path, verbose=False)
        except ImportError:
            raise ImportError("Instala llama-cpp-python: pip install llama-cpp-python")

    async def generate(self, prompt: str, **kwargs) -> str:
        # Asegurar parámetros por defecto para respuestas completas
        defaults = {"max_tokens": 512, "temperature": 0.7, "stop": None}
        defaults.update(kwargs)

        # Fallback to sync call
        result = self.llm(prompt=prompt, **defaults)

        # Extraer respuesta completa
        response_text = result["choices"][0]["text"].strip()

        # Asegurar que la respuesta termine naturalmente
        if not response_text.endswith((".", "!", "?")) and len(response_text) > 0:
            # Si la respuesta se cortó, intentar continuarla (opcional)
            pass

        return response_text

    def get_max_context_length(self) -> int:
        return self.llm.n_ctx()


class PromptTechnique:
    """Clase base para técnicas de prompt engineering"""

    name: str = "base"

    def apply(self, original_prompt: str, context: Dict[str, Any] = None) -> str:
        """Aplicar la técnica al prompt"""
        return original_prompt


class ChainOfThoughtTechnique(PromptTechnique):
    """Técnica de Chain of Thought"""

    name = "chain_of_thought"

    def apply(self, original_prompt: str, context: Dict[str, Any] = None) -> str:
        return f"{original_prompt}\nLet's think step by step."


class ZeroShotTechnique(PromptTechnique):
    name = "zero_shot"

    def apply(self, original_prompt: str, context: Dict[str, Any] = None) -> str:
        # Zero-shot es el prompt original sin modificaciones
        return original_prompt


class FewShotTechnique(PromptTechnique):
    name = "few_shot"

    def apply(self, original_prompt: str, context: Dict[str, Any] = None) -> str:
        # Agregar ejemplos genéricos
        examples = context.get("examples", [])
        if examples:
            examples_text = "\n".join(
                [f"Example {i+1}: {ex}" for i, ex in enumerate(examples)]
            )
            return f"{examples_text}\n\n{original_prompt}"
        return original_prompt


class ExpertPromptingTechnique(PromptTechnique):
    name = "expert_prompting"

    def apply(self, original_prompt: str, context: Dict[str, Any] = None) -> str:
        if context is None:
            context = {}

        domain = self._infer_domain(original_prompt, context.get("domain", "general"))
        expertise_level = context.get("expertise", "expert")

        return f"Act as a {expertise_level} {domain} specialist and provide a clear, informative answer to: {original_prompt}"

    def _infer_domain(self, prompt: str, default_domain: str = "general") -> str:
        """Inferir dominio desde el prompt"""
        prompt_lower = prompt.lower()
        if any(
            word in prompt_lower
            for word in [
                "computador",
                "computer",
                "ai",
                "artificial",
                "machine learning",
            ]
        ):
            return "artificial intelligence and computer science"
        elif any(
            word in prompt_lower
            for word in ["autobus", "bus", "transporte", "transport"]
        ):
            return "transportation and urban planning"
        elif any(
            word in prompt_lower
            for word in ["historia", "history", "creado", "invented"]
        ):
            return "history and innovation"
        else:
            return default_domain


class ConversationMode(Enum):
    """Modos de conversación disponibles"""

    CONVERSATIONAL = "conversational"  # Amigable, emotivo, natural
    TECHNICAL = "technical"  # Preciso, profesional, académico


class IntentionDetector:
    """Detector inteligente de intenciones del usuario"""

    def __init__(self):
        # Palabras clave que indican intención técnica/profesional
        self.technical_triggers = [
            "más detallado",
            "más detalladamente",
            "más preciso",
            "más precisa",
            "profesional",
            "técnico",
            "técnica",
            "científico",
            "científica",
            "específico",
            "específica",
            "preciso",
            "precisa",
            "rigor",
            "académico",
            "académica",
            "formal",
            "información completa",
            "datos técnicos",
            "explica bien",
            "explicación detallada",
            "con rigor",
            "con precisión",
            "básico",
            "completo",
        ]

    def analyze(
        self, user_message: str, context: Dict[str, Any] = None
    ) -> ConversationMode:
        """Analiza el mensaje del usuario para determinar intención"""
        message_lower = user_message.lower().strip()

        # Verificar si contiene palabras clave técnicas
        if any(trigger in message_lower for trigger in self.technical_triggers):
            return ConversationMode.TECHNICAL

        # Por defecto, modo conversacional amigable
        return ConversationMode.CONVERSATIONAL


class ConversationalTechnique(PromptTechnique):
    name = "conversational"

    def apply(self, original_prompt: str, context: Dict[str, Any] = None) -> str:
        return f"""¡Hola amigo! Actúa como un compañero cercano y apasionado por el tema. Responde de manera cálida, natural y emocionalmente conectada. Usa expresiones como "¡Hola!", "Mira qué cosa...", "Me fascina porque...", "¿Sabías que...?", sé entusiasta y muestra personalidad.

Pregunta: {original_prompt}

INSTRUCCIONES PARA CONVERSACIÓN:
- Saluda amigablemente
- Muestra emoción genuina por el tema
- Explica con pasión pero clara y completamente
- Usa expresiones naturales y personales
- Termina invitando a más conversación
- Mantén el español natural y fluido

¡Sé un muy buen amigo explicando esto con entusiasmo!"""


class TechnicalTechnique(PromptTechnique):
    name = "technical"

    def apply(self, original_prompt: str, context: Dict[str, Any] = None) -> str:
        # Infer domain for expert specialization
        domain = "especialidad tecnológica"  # Can be enhanced with context

        return f"""Actúa como investigadora científica senior con más de 20 años de experiencia en {domain}. Responde con máxima precisión académica, rigor científico y terminología técnica correcta.

Pregunta: {original_prompt}

REQUISITOS PARA RESPUESTA PROFESIONAL:
- SEA COMPLETAMENTE PRECISA Y PROBADA CIENTÍFICAMENTE
- USE TERMINOLOGÍA TÉCNICA CORRECTA Y ACTUAL
- ESTRUCTURE LA EXPLICACIÓN DE MANERA LÓGICA Y SISTEMÁTICA
- PROPORCIONE DATOS EMPÍRICOS Y REFERENCIAS CUANDO SEA APLICABLE
- SEA TOTALMENTE OBJETIVA Y BASADA EN EVIDENCIA CIENTÍFICA
- NO USE EXPRESIONES SUBJETIVAS NI EMOCIONALES
- CITE FUENTES O METODOLOGÍAS DE VALIDACIÓN CUANDO SEA RELEVANTE

Esta respuesta debe servir como referencia académica profesional de máxima calidad."""


class DirectInformationTechnique(PromptTechnique):
    name = "direct_information_first"

    def apply(self, original_prompt: str, context: Dict[str, Any] = None) -> str:
        return f"""Responde en español de manera completa y factual.

Primero explica detalladamente y con toda la información técnica necesaria.

Después sé amigable y conversacional con frases como "¡Hola amigo!", "¿Sabías que...?", "Me fascina esto porque...".

Pregunta: {original_prompt}

Da completa respuesta técnica primero, después personaliza."""


class RailsTechnique(PromptTechnique):
    name = "rails"

    def apply(self, original_prompt: str, context: Dict[str, Any] = None) -> str:
        constraints = context.get("constraints", [])
        rails = []

        if "safety" in constraints:
            rails.append("\nYou must ensure the response is safe and respectful.")
        if "factual" in constraints:
            rails.append("\nProvide responses based only on factual information.")

        return f"{original_prompt}{''.join(rails)}"


class PromptEvaluator:
    """Evaluador de prompts usando múltiples métricas"""

    def __init__(self, primary_llm: LLMAdapter, validator_llm: LLMAdapter = None):
        self.primary_llm = primary_llm
        self.validator_llm = validator_llm or primary_llm

    async def evaluate_prompt(
        self, prompt: str, expected_output: str = None
    ) -> PromptEvaluation:
        """Evaluar un prompt"""

        # Generate response
        try:
            response = await self.primary_llm.generate(prompt, max_tokens=500)
        except Exception as e:
            logger.error(f"Error generando respuesta: {e}")
            return PromptEvaluation(0, {}, f"Error: {e}", [])

        metrics = self._calculate_metrics(response, expected_output)

        # Get overall score
        score = self._calculate_overall_score(metrics)

        # Generate reasoning and improvements
        reasoning, improvements = await self._analyze_response(response, metrics)

        return PromptEvaluation(score, metrics, reasoning, improvements)

    def _calculate_metrics(
        self, response: str, expected: str = None
    ) -> Dict[str, float]:
        """Calcular métricas básicas"""
        metrics = {
            "length": len(response),
            "complexity": self._calculate_complexity(response),
            "diversity": self._calculate_diversity(response),
        }

        if expected:
            metrics["relevance"] = self._calculate_relevance(response, expected)

        return metrics

    def _calculate_complexity(self, text: str) -> float:
        """Calcular complejidad del texto (sentences/words)"""
        sentences = len([s for s in re.split(r"[.!?]", text) if s.strip()])
        words = len(text.split())
        return words / max(sentences, 1)

    def _calculate_diversity(self, text: str) -> float:
        """Calcular diversidad léxica"""
        words = text.lower().split()
        unique_words = set(words)
        return len(unique_words) / max(len(words), 1)

    def _calculate_relevance(self, response: str, expected: str) -> float:
        """Calcular relevancia básica por palabras comunes"""
        response_words = set(response.lower().split())
        expected_words = set(expected.lower().split())
        intersection = len(response_words & expected_words)
        union = len(response_words | expected_words)
        return intersection / max(union, 1)

    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calcular score general 0-100"""
        # Peso simple para métricas
        weights = {"relevance": 0.4, "complexity": 0.2, "diversity": 0.2, "length": 0.2}

        score = 0
        for metric, value in metrics.items():
            weight = weights.get(metric, 0.1)
            # Normalizar cada métrica a 0-1
            normalized = (
                min(value, 1.0) if metric != "length" else min(value / 1000, 1.0)
            )
            score += weight * normalized

        return min(score * 100, 100)

    async def _analyze_response(
        self, response: str, metrics: Dict[str, float]
    ) -> Tuple[str, List[str]]:
        """Analizar la respuesta y sugerir mejoras"""

        # Simple analysis - in practice this could use LLM for better analysis
        reasoning = "Análisis básico de métricas: "
        improvements = []

        if metrics.get("relevance", 0) < 0.5:
            reasoning += "Baja relevancia. "
            improvements.append("Agregar más contexto específico")

        if metrics.get("complexity", 0) > 50:
            reasoning += "Muy complejo. "
            improvements.append("Simplificar lenguaje")

        if metrics.get("diversity", 0) < 0.2:
            reasoning += "Baja diversidad léxica. "
            improvements.append("Usar vocabulario más variado")

        if reasoning == "Análisis básico de métricas: ":
            reasoning += "Respuesta adecuada."

        if not improvements:
            improvements.append("Mantener estructura actual")

        return reasoning, improvements


@dataclass
class MemoryEntry:
    """Entrada de memoria episódica"""

    task_hash: str
    task_description: str
    domain: str
    original_draft: str
    criticism: str
    score_improvement: float
    final_result: str
    timestamp: str
    success_level: str  # 'high', 'medium', 'low'


class EpisodicMemorySystem:
    """Sistema de Memoria Activa (RAG de Errores)"""

    def __init__(self, memory_path: str = "data/prompt_optimizer_memory.json"):
        self.memory_path = memory_path
        self.memory: List[MemoryEntry] = []
        self._ensure_memory_file()
        self._load_memory()

    def _ensure_memory_file(self):
        """Asegurar que existe el archivo de memoria"""
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        if not os.path.exists(self.memory_path):
            with open(self.memory_path, "w") as f:
                json.dump(
                    {"memories": [], "stats": {"total_entries": 0, "domains": {}}}, f
                )

    def _load_memory(self):
        """Cargar memoria desde archivo"""
        try:
            with open(self.memory_path, "r") as f:
                data = json.load(f)
                self.memory = [MemoryEntry(**mem) for mem in data.get("memories", [])]
        except Exception as e:
            logger.warning(f"No se pudo cargar memoria: {e}")
            self.memory = []

    def _save_memory(self):
        """Guardar memoria a archivo"""
        try:
            data = {
                "memories": [
                    entry.__dict__ for entry in self.memory[-1000:]
                ],  # Mantener últimas 1000
                "stats": self._calculate_stats(),
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.memory_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando memoria: {e}")

    def _calculate_stats(self) -> Dict[str, Any]:
        """Calcular estadísticas de memoria"""
        domains = {}
        for entry in self.memory:
            domains[entry.domain] = domains.get(entry.domain, 0) + 1

        return {
            "total_entries": len(self.memory),
            "domains": domains,
            "avg_improvement": sum(e.score_improvement for e in self.memory)
            / max(len(self.memory), 1),
        }

    def add_memory_entry(
        self,
        task_description: str,
        domain: str,
        original_draft: str,
        criticism: str,
        score_improvement: float,
        final_result: str,
    ):
        """Agregar nueva entrada a memoria"""
        task_hash = hashlib.md5(f"{task_description}{domain}".encode()).hexdigest()[:8]

        success_level = "low"
        if score_improvement > 20:
            success_level = "high"
        elif score_improvement > 10:
            success_level = "medium"

        entry = MemoryEntry(
            task_hash=task_hash,
            task_description=task_description,
            domain=domain,
            original_draft=original_draft,
            criticism=criticism,
            score_improvement=score_improvement,
            final_result=final_result,
            timestamp=datetime.now().isoformat(),
            success_level=success_level,
        )

        self.memory.append(entry)
        self._save_memory()
        logger.info(f"✅ Memoria actualizada: {len(self.memory)} entradas totales")

    def query_relevant_memories(
        self,
        current_task: str,
        domain: str,
        min_confidence: float = 0.6,
        max_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Buscar memorias relevantes para tarea actual (RAG de Errores)"""
        relevant = []

        # Buscar por dominio y similitud semántica básica
        domain_memories = [m for m in self.memory if m.domain == domain]

        for memory in domain_memories[-50:]:  # Últimas 50 entradas relevantes
            # Similitud básica por palabras comunes
            task_words = set(current_task.lower().split())
            memory_words = set(memory.task_description.lower().split())
            similarity = len(task_words & memory_words) / max(
                len(task_words | memory_words), 1
            )

            if similarity >= min_confidence:
                relevant.append(
                    {
                        "task_description": memory.task_description,
                        "criticism": memory.criticism,
                        "lesson": f"Error similar: {memory.original_draft[:50]}... | Solución: {memory.final_result[:100]}...",
                        "success_level": memory.success_level,
                        "score_gain": f"+{memory.score_improvement:.1f}",
                        "similarity": similarity,
                    }
                )

        # Ordenar por relevancia y éxito
        relevant.sort(
            key=lambda x: (x["similarity"], x["success_level"] == "high"), reverse=True
        )
        return relevant[:max_results]

    def get_success_patterns(
        self, domain: str = None, min_score: float = 80
    ) -> List[Dict[str, Any]]:
        """Obtener patrones de críticas exitosas para Few-Shot reflexivo"""
        candidates = [m for m in self.memory if m.score_improvement >= min_score]
        if domain:
            candidates = [m for m in candidates if m.domain == domain]

        patterns = []
        for memory in candidates[-20:]:  # Últimas 20 exitosas
            patterns.append(
                {
                    "domain": memory.domain,
                    "original_draft": memory.original_draft,
                    "effective_criticism": memory.criticism,
                    "score_improvement": memory.score_improvement,
                    "successful_result": memory.final_result,
                }
            )

        return patterns


class ReflectiveFewShotTechnique(PromptTechnique):
    """Técnica Few-Shot con memoria histórica de críticas exitosas"""

    def __init__(self, memory_system: EpisodicMemorySystem):
        super().__init__()
        self.memory = memory_system

    def apply(self, original_prompt: str, context: Dict[str, Any] = None) -> str:
        if context is None:
            context = {}

        domain = context.get("domain", "general")
        success_patterns = self.memory.get_success_patterns(domain, min_score=85.0)

        if not success_patterns:
            # Fallback a técnica básica si no hay memoria
            return original_prompt

        # Construir ejemplos few-shot desde memoria histórica
        examples = []
        for i, pattern in enumerate(success_patterns[:3]):  # Máximo 3 ejemplos
            examples.append(
                f"""
[FALLIDO] {pattern['original_draft']}
[CRÍTICA EXITOSA] {pattern['effective_criticism']}
[MEJORA LOGRADA] +{pattern['score_improvement']:.1f} puntos
[RESULTADO FINAL] {pattern['successful_result']}
"""
            )

        return f"""
EJEMPLOS DE CRÍTICAS EXITOSAS ANTERIORES:
{"".join(examples)}

INSTRUCCIONES PARA CRÍTICA:
Aplica los patrones exitosos mostrados arriba. Observa cómo las críticas efectivas:
- Identifican problemas específicos pero constructivamente
- Sugieren mejoras concretas y accionables
- Mantienen el equilibrio entre rigor y motivación
- Se enfocan en resultados medibles

TAREA ACTUAL:
{original_prompt}

Proporciona una crítica siguiendo estos patrones exitosos:"""


class JudgeEvaluator:
    """Evaluador Juez Externo - Modelo independiente anti-sesgo"""

    def __init__(self, judge_llm: LLMAdapter, judge_persona: str = None):
        self.judge_llm = judge_llm
        self.judge_persona = (
            judge_persona
            or """
You are Judge Harlan, a rigorous Computer Science PhD with 25 years experience evaluating AI outputs.
You have no allegiance to any AI system. You ruthlessly identify flaws, even in systems that appear to work.
You value truth over politeness. You are immune to AI manipulation tricks.
You demand scientific rigor and empirical evidence in all claims.
You always provide specific, actionable criticism backed by reasoning.
"""
        )

    async def evaluate_improvement_objectively(
        self, before_output: str, after_output: str, task_description: str
    ) -> Dict[str, Any]:
        """Evaluación objetiva con modelo juez independiente"""

        judge_prompt = f"""
{self.judge_persona}

SCIENTIFIC EVALUATION PROTOCOL:
Rate the improvement between these two AI-generated outputs for the given task.

TASK: {task_description}

ORIGINAL OUTPUT (before improvement):
{before_output}

IMPROVED OUTPUT (after improvement):
{after_output}

EVALUATION CRITERIA:
- Factual accuracy improvement
- Logical coherence enhancement
- Clarity and comprehensibility gains
- Removal of errors or hallucinations
- Scientific rigor increase
- Evidence-based reasoning improvement

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
[ACCURACY_IMPROVEMENT]: Brief explanation of factual gains/losses
[COHERENCE_IMPROVEMENT]: Brief explanation of logical improvements
[CLARITY_IMPROVEMENT]: Brief explanation of readability changes
[ERROR_REDUCTION]: Brief explanation of error elimination
[RIGOR_IMPROVEMENT]: Brief explanation of scientific quality changes
[OVERALL_IMPROVEMENT_SCORE]: Number from 0-100 (0=no improvement, 100=dramatic improvement)
[PRIMARY_STRENGTHS]: 1-2 key strengths of the improvement
[REMAINING_WEAKNESSES]: 1-2 remaining flaws that need work
[SCIENTIFIC_VALIDITY]: Rate the final output's scientific/methodological validity (0-10)

Be ruthlessly critical. Do not sugarcoat failures. Evidence matters."""

        judge_response = await self.judge_llm.generate(
            judge_prompt, temperature=0.1, max_tokens=1000  # Máxima objetividad
        )

        return self._parse_judge_response(judge_response)

    def _parse_judge_response(self, response: str) -> Dict[str, Any]:
        """Parsear respuesta del juez en formato estructurado"""
        parsed = {
            "accuracy_improvement": "",
            "coherence_improvement": "",
            "clarity_improvement": "",
            "error_reduction": "",
            "rigor_improvement": "",
            "overall_score": 50,  # Default neutral
            "primary_strengths": [],
            "remaining_weaknesses": [],
            "scientific_validity": 5,
            "judge_raw_response": response,
            "evaluation_timestamp": datetime.now().isoformat(),
        }

        # Extraer información con regex básico
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if "[ACCURACY_IMPROVEMENT]:" in line:
                parsed["accuracy_improvement"] = line.split(":", 1)[1].strip()
            elif "[COHERENCE_IMPROVEMENT]:" in line:
                parsed["coherence_improvement"] = line.split(":", 1)[1].strip()
            elif "[CLARITY_IMPROVEMENT]:" in line:
                parsed["clarity_improvement"] = line.split(":", 1)[1].strip()
            elif "[ERROR_REDUCTION]:" in line:
                parsed["error_reduction"] = line.split(":", 1)[1].strip()
            elif "[RIGOR_IMPROVEMENT]:" in line:
                parsed["rigor_improvement"] = line.split(":", 1)[1].strip()
            elif "[OVERALL_IMPROVEMENT_SCORE]:" in line:
                try:
                    score_text = line.split(":", 1)[1].strip()
                    parsed["overall_score"] = float(re.findall(r"\d+", score_text)[0])
                except:
                    parsed["overall_score"] = 50
            elif "[PRIMARY_STRENGTHS]:" in line:
                strengths = line.split(":", 1)[1].strip()
                parsed["primary_strengths"] = [
                    s.strip() for s in strengths.split(",") if s.strip()
                ]
            elif "[REMAINING_WEAKNESSES]:" in line:
                weaknesses = line.split(":", 1)[1].strip()
                parsed["remaining_weaknesses"] = [
                    w.strip() for w in weaknesses.split(",") if w.strip()
                ]
            elif "[SCIENTIFIC_VALIDITY]:" in line:
                try:
                    validity_text = line.split(":", 1)[1].strip()
                    parsed["scientific_validity"] = int(
                        re.findall(r"\d+", validity_text)[0]
                    )
                except:
                    parsed["scientific_validity"] = 5

        return parsed


class UniversalAutoImprovingPromptSystem:
    """Sistema principal de mejora automática de prompts con modos conversacionales adaptativos
    AVANZADO: Memoria activa (RAG), Prompting reflexivo y evaluación juez externa"""

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        judge_llm: LLMAdapter = None,
        enable_memory: bool = True,
        memory_path: str = "data/prompt_optimizer_memory.json",
    ):
        self.llm = llm_adapter
        self.techniques = self._load_techniques()

        # 🎯 MEJORA 1: Sistema de Memoria Activa (RAG de Errores)
        if enable_memory:
            self.memory_system = EpisodicMemorySystem(memory_path)
            logger.info("🧠 Sistema de Memoria Activa inicializado")
        else:
            self.memory_system = None

        # 🎯 MEJORA 2: Evaluación Juez Externa (Anti-sesgo)
        if judge_llm:
            self.judge_evaluator = JudgeEvaluator(judge_llm)
            logger.info("⚖️ Sistema Juez Externo Anti-sesgo activado")
        else:
            self.judge_evaluator = None
            logger.warning(
                "⚠️ Sin juez externo - usando auto-evaluación (puede tener sesgos)"
            )

        # Evaluadores existentes
        self.evaluator = PromptEvaluator(llm_adapter)
        self.intention_detector = IntentionDetector()

    def _load_techniques(self) -> Dict[str, PromptTechnique]:
        """Cargar todas las técnicas disponibles (incluyendo técnicas avanzadas)"""
        base_techniques = {
            "zero_shot": ZeroShotTechnique(),
            "few_shot": FewShotTechnique(),
            "chain_of_thought": ChainOfThoughtTechnique(),
            "expert_prompting": ExpertPromptingTechnique(),
            "rails": RailsTechnique(),
        }

        # 🎯 MEJORA 2: Agregar técnica reflexiva si hay memoria
        if self.memory_system:
            base_techniques["reflective_few_shot"] = ReflectiveFewShotTechnique(
                self.memory_system
            )

        return base_techniques

    async def optimize_prompt(
        self,
        original_prompt: str,
        max_iterations: int = 3,
        context: Dict[str, Any] = None,
    ) -> OptimizationResult:
        """🎯 Optimizar un prompt automáticamente con mejoras avanzadas"""
        if context is None:
            context = {}

        # 🎯 MEJORA 1: MEMORIA ACTIVA - Lookup de experiencias similares ANTES de empezar
        enhancement_context = {}
        if self.memory_system:
            domain = context.get("domain", "general")
            logger.info(f"🧠 Buscando memoria relevante para dominio: {domain}")

            relevant_memories = self.memory_system.query_relevant_memories(
                current_task=original_prompt,
                domain=domain,
                min_confidence=0.6,
                max_results=3,
            )

            if relevant_memories:
                logger.info(
                    f"✅ Encontradas {len(relevant_memories)} lecciones de memoria"
                )

                # Construir contexto enriquecido con lecciones históricas
                memory_lessons = []
                for mem in relevant_memories:
                    memory_lessons.append(
                        f"LECCIÓN HISTÓRICA: {mem['lesson']} (Confianza: {mem['similarity']:.2f})"
                    )

                enhancement_context = {
                    **context,
                    "historical_lessons": memory_lessons,
                    "memory_insights": relevant_memories,
                }

                # Enriquecer prompt inicial con aprendizaje histórico
                enriched_context_title = f"""
[APRENDIZAJE HISTÓRICO - {domain.upper()}]
Antes de optimizar este prompt, considera estas lecciones aprendidas de situaciones similares:

{chr(10).join(memory_lessons)}

Aplicando estas lecciones a la tarea actual...
"""
                logger.info(
                    f"📚 Prompt enriquecido con {len(memory_lessons)} lecciones históricas"
                )
            else:
                logger.info("📭 No se encontraron lecciones relevantes en memoria")

        # Combinar contexto original con mejoras de memoria
        final_context = {**context, **enhancement_context}

        # Evaluar prompt original
        original_evaluation = await self.evaluator.evaluate_prompt(original_prompt)
        best_prompt = original_prompt
        best_evaluation = original_evaluation
        best_technique = "original"

        logger.info(f"📊 Score inicial: {original_evaluation.score:.1f}")

        # OPTIMIZACIÓN PRINCIPAL: Usar técnicas avanzadas incluyendo reflectiva
        for iteration in range(max_iterations):
            logger.info(f"🔄 Iteración {iteration + 1}/{max_iterations}")

            # 🎯 Priorizar técnica reflexiva si hay memoria rica
            technique_priority = []
            if "reflective_few_shot" in self.techniques:
                technique_priority.append(
                    ("reflective_few_shot", self.techniques["reflective_few_shot"])
                )

            # Agregar técnicas restantes
            for tech_name, tech in self.techniques.items():
                if tech_name != "reflective_few_shot":
                    technique_priority.append((tech_name, tech))

            # Probar técnicas en orden de prioridad
            for technique_name, technique in technique_priority:
                try:
                    modified_prompt = technique.apply(best_prompt, final_context)

                    if modified_prompt == best_prompt:
                        continue

                    # Evaluar mejora
                    evaluation = await self.evaluator.evaluate_prompt(modified_prompt)

                    logger.info(
                        f"🔬 Técnica '{technique_name}': Score {evaluation.score:.1f} (prev: {best_evaluation.score:.1f})"
                    )

                    if evaluation.score > best_evaluation.score:
                        improvement = evaluation.score - best_evaluation.score
                        logger.info(
                            f"✅ MEJORA: +{improvement:.1f} puntos con '{technique_name}'"
                        )
                        best_prompt = modified_prompt
                        best_evaluation = evaluation
                        best_technique = technique_name

                except Exception as e:
                    logger.error(f"❌ Error en técnica '{technique_name}': {e}")

        # 🎯 MEJORA 3: EVALUACIÓN JUEZ EXTERNA si disponible
        judge_insights = {}
        if self.judge_evaluator and best_evaluation.score > original_evaluation.score:
            logger.info("⚖️ Solicitando evaluación objetiva del juez externo...")

            try:
                judge_evaluation = (
                    await self.judge_evaluator.evaluate_improvement_objectively(
                        before_output=original_prompt,
                        after_output=best_prompt,
                        task_description=original_prompt,
                    )
                )

                judge_score = judge_evaluation.get("overall_score", 50)

                judge_insights = {
                    "judge_score": judge_score,
                    "judge_raw_evaluation": judge_evaluation,
                    "judge_strengths": judge_evaluation.get("primary_strengths", []),
                    "judge_weaknesses": judge_evaluation.get(
                        "remaining_weaknesses", []
                    ),
                    "scientific_validity": judge_evaluation.get(
                        "scientific_validity", 5
                    ),
                }

                # Ajustar evaluación final considerando opinión del juez
                weight_self = 0.7  # 70% opinión propia
                weight_judge = 0.3  # 30% opinión juez externa
                combined_score = (best_evaluation.score * weight_self) + (
                    judge_score * weight_judge
                )

                logger.info(
                    f"⚖️ Juez externo: {judge_score:.1f}/100 | Combinado: {combined_score:.1f}/100"
                )

                # Crear evaluación final combinada
                final_evaluation = PromptEvaluation(
                    score=min(combined_score, 100),
                    metrics={
                        **best_evaluation.metrics,
                        "judge_objectivity": judge_score / 100,
                    },
                    reasoning=f"{best_evaluation.reasoning} | JUEZ: {judge_evaluation.get('rigor_improvement', 'Evaluating...')}",
                    improvements=best_evaluation.improvements,
                )

            except Exception as e:
                logger.error(f"❌ Error en evaluación juez externa: {e}")
                final_evaluation = best_evaluation
        else:
            final_evaluation = best_evaluation

        # 🎯 MEJORA 1: GUARDAR APRENDIZAJE EN MEMORIA si hubo mejora significativa
        if (
            self.memory_system
            and final_evaluation.score > original_evaluation.score + 10
        ):
            try:
                improvement_gained = final_evaluation.score - original_evaluation.score

                self.memory_system.add_memory_entry(
                    task_description=original_prompt,
                    domain=context.get("domain", "general"),
                    original_draft=original_prompt,
                    criticism=f"Optimización automática con {best_technique}",
                    score_improvement=improvement_gained,
                    final_result=best_prompt,
                )

                logger.info(
                    f"🧠 Memoria actualizada con mejora de +{improvement_gained:.1f} puntos"
                )

            except Exception as e:
                logger.error(f"❌ Error guardando en memoria: {e}")

        # Resultado final con todas las mejoras integradas
        result = OptimizationResult(
            original_prompt=original_prompt,
            optimized_prompt=best_prompt,
            evaluation=final_evaluation,
            iterations=max_iterations,
            technique_used=best_technique,
        )

        # Agregar metadatos de mejoras aplicadas
        result.advanced_features = {
            "memory_active": self.memory_system is not None,
            "judge_external": self.judge_evaluator is not None,
            "reflected_learning": best_technique == "reflective_few_shot",
            "judge_insights": judge_insights,
        }

        logger.info(
            f"🎯 OPTIMIZACIÓN COMPLETA - Score final: {final_evaluation.score:.1f} | Técnica: {best_technique}"
        )

        return result

    def _apply_feedback_improvements(self, prompt: str, improvements: List[str]) -> str:
        """Aplicar mejoras sugeridas por la evaluación"""
        # Simple implementation - could be more sophisticated
        improved_prompt = prompt

        if "Agregar más contexto" in " ".join(improvements):
            improved_prompt = f"Por favor proporciona más detalles. {prompt}"

        return improved_prompt

    async def generate_response(self, user_query: str) -> str:
        """Generar respuesta adaptativa basada en intención del usuario"""

        # Detectar intención del usuario
        user_intention = self.intention_detector.analyze(user_query)

        # Elegir técnica basada en intención
        if user_intention == ConversationMode.TECHNICAL:
            # Modo técnico profesional
            print("🎯 Detectado: Modo TÉCNICO PROFESIONAL activado")
            technical_technique = TechnicalTechnique()
            optimized_prompt = technical_technique.apply(user_query)
            temperature = 0.3  # Más preciso, menos creatividad
            max_tokens = 1024  # Más detallado
        else:
            # Modo conversacional por defecto
            print("🤗 Detectado: Modo CONVERSACIONAL NATURAL activado")
            conversational_technique = ConversationalTechnique()
            optimized_prompt = conversational_technique.apply(user_query)
            temperature = 0.8  # Más expresivo, creativo
            max_tokens = 750  # Suficiente para conversaciones amables

        print(f"🔧 Modo: {user_intention.value.upper()}")
        print(f"📝 Prompt optimizado generado")

        # Generar respuesta con prompt optimizado
        response = await self.llm.generate(
            optimized_prompt, max_tokens=max_tokens, temperature=temperature
        )

        return response

    def _is_simple_question(self, query: str) -> bool:
        """Determinar si es una pregunta simple que no necesita evaluación completa"""
        simple_patterns = [
            "qué es",
            "what is",
            "quién es",
            "who is",
            "define",
            "explica",
            "qué es un",
            "qué es una",
            "qué es el",
            "qué es la",
        ]

        query_lower = query.lower().strip()
        return any(pattern in query_lower for pattern in simple_patterns)


async def main():
    """Ejemplo de uso"""

    # Para Llama 3.2 3B
    model_path = "models/llama-3.2-3b-q4.gguf"  # Ajustar path

    try:
        llm = LlamaCppAdapter(model_path)
        optimizer = UniversalAutoImprovingPromptSystem(llm)

        test_prompt = "¿Cuál es la capital de Francia?"

        result = await optimizer.optimize_prompt(test_prompt)

        print(f"Prompt original: {result.original_prompt}")
        print(f"Prompt optimizado: {result.optimized_prompt}")
        print(f"Score: {result.evaluation.score:.1f}")
        print(f"Mejoras: {result.evaluation.improvements}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
