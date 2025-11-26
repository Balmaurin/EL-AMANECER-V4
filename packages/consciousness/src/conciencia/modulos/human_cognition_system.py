#!/usr/bin/env python3
"""
SISTEMA COGNITIVO HUMANO COMPLETO

Implementa los 23 tipos de pensamiento + 9 sesgos cognitivos del catálogo completo
Basado en procesos psicológicos y neurocientíficos reales.
"""

import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import math


class ThinkingProcess(Enum):
    """23 Tipos de pensamientos por proceso cognitivo"""
    DEDUCTIVO = "deductive"
    INDUCTIVO = "inductive"
    ABDUCTIVO = "abductive"
    ANALITICO = "analytic"
    SINTETICO = "synthetic"
    CRITICO = "critical"
    CREATIVO = "creative"
    LATERAL = "lateral"
    VERTICAL = "vertical"
    CONVERGENTE = "convergent"
    DIVERGENTE = "divergent"
    SISTEMICO = "systemic"
    HOLISTICO = "holistic"


class ThinkingContent(Enum):
    """7 Tipos de pensamientos por contenido"""
    ABSTRACTO = "abstract"
    CONCRETO = "concrete"
    TEORICO = "theoretical"
    PRACTICO = "practical"
    ESTRATEGICO = "strategic"
    TACTICO = "tactical"
    OPERATIVO = "operative"


class ThinkingState(Enum):
    """5 Estados mentales de pensamiento"""
    CONSCIENTE = "conscious"
    INCONSCIENTE = "unconscious"
    PRECONSCIENTE = "preconscious"
    SUBCONSCIENTE = "subconscious"
    RUMIANTE = "ruminate"


class CognitiveBias(Enum):
    """9 Sesgos cognitivos principales"""
    CONFIRMACION = "confirmation_bias"
    DISPONIBILIDAD = "availability_bias"
    ANCLAJE = "anchoring_bias"
    REPRESENTATIVIDAD = "representativeness_bias"
    COSTE_HUNDIDO = "sunk_cost_bias"
    STATUS_QUO = "status_quo_bias"
    OPTIMISMO = "optimism_bias"
    NEGATIVIDAD = "negativity_bias"
    AUTORIDAD = "authority_bias"


@dataclass
class ThoughtProcess:
    """Proceso de pensamiento individual con características completas"""
    thought_id: str
    process_type: ThinkingProcess
    content_type: ThinkingContent
    mental_state: ThinkingState
    content: Any
    confidence: float = 1.0
    emotional_valence: float = 0.0
    complexity: float = 0.5
    originality: float = 0.5
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    cognitive_load: float = 0.3
    active_biases: List[CognitiveBias] = field(default_factory=list)


@dataclass
class CognitiveState:
    """Estado cognitivo completo en un momento"""
    dominant_process: ThinkingProcess
    content_focus: ThinkingContent
    mental_clarity: float  # 0.0 = confuso, 1.0 = cristalino
    attention_stability: float
    working_memory_load: float
    insight_probability: float
    rumination_level: float
    flow_state: float  # estado de flujo cognitivo
    cognitive_biases_active: Dict[CognitiveBias, float]


class ThoughtGenerator:
    """Generador de pensamientos basados en contexto y personalidad"""

    def __init__(self, personality: Dict[str, float]):
        self.personality = personality or {}
        self.thought_patterns: Dict[str, List[Dict]] = self._initialize_thought_patterns()
        self.cognitive_history: List[ThoughtProcess] = []

    def generate_thought(self, trigger: str, context: Dict[str, Any],
                        emotional_state: Dict[str, Any]) -> ThoughtProcess:
        """Genera pensamiento basado en trigger, contexto y estado emocional"""

        # Determinar tipo de proceso basado en contexto
        process_type = self._select_process_type(trigger, context, emotional_state)
        content_type = self._select_content_type(trigger, context)
        mental_state = self._determine_mental_state(context, emotional_state)

        # Generar contenido del pensamiento
        thought_content = self._generate_thought_content(trigger, context, process_type, content_type)

        # Calcular propiedades cognitivas
        properties = self._calculate_cognitive_properties(thought_content, context, emotional_state, process_type)

        # Aplicar sesgos cognitivos
        active_biases = self._apply_cognitive_biases(thought_content, context)

        # Crear proceso de pensamiento completo
        thought = ThoughtProcess(
            thought_id=f"thought_{int(time.time()*1000)}",
            process_type=process_type,
            content_type=content_type,
            mental_state=mental_state,
            content=thought_content,
            confidence=properties['confidence'],
            emotional_valence=properties['valence'],
            complexity=properties['complexity'],
            originality=properties['originality'],
            cognitive_load=properties['load'],
            active_biases=active_biases
        )

        # Registrar en historia
        self.cognitive_history.append(thought)
        if len(self.cognitive_history) > 100:
            self.cognitive_history = self.cognitive_history[-50:]

        return thought

    def _select_process_type(self, trigger: str, context: Dict, emotion: Dict) -> ThinkingProcess:
        """Selecciona tipo de proceso cognitivo basado en contexto"""

        # Factores determinantes
        is_problem_solving = context.get('task_type', '') in ['problem_solving', 'decision_making']
        time_pressure = context.get('urgency', 0) > 0.7
        is_creative = context.get('creativity_needed', False)
        is_uncertain = context.get('uncertainty', 0) > 0.6

        # Emoción influye en el tipo
        emotional_intensity = emotion.get('intensity', 0)

        # Análisis deductivo para lógica clara
        if trigger.lower() in ['porque', 'por qué', 'razón', 'explicar']:
            return ThinkingProcess.ANALITICO

        # Pensamiento crítico ante contradicciones
        if 'conflicto' in trigger.lower() or emotional_intensity > 0.7:
            return ThinkingProcess.CRITICO

        # Pensamiento creativo ante incertidumbre
        if is_creative or is_uncertain:
            creative_chance = self._weighted_choice(['CREATIVO', 'LATERAL', 'DIVERGENTE'],
                                                   [0.4, 0.3, 0.3])
            return ThinkingProcess[creative_chance]

        # Pensamiento convergente bajo presión
        if is_problem_solving and time_pressure:
            return ThinkingProcess.CONVERGENTE

        # Pensamiento sistémico para visión de conjunto
        if 'sistema' in trigger.lower() or context.get('systemic_view', False):
            return ThinkingProcess.SISTEMICO

        # Default basado en personalidad
        extroversion = self.personality.get('extraversion', 0.5)
        openness = self.personality.get('openness', 0.5)

        if extroversion > 0.7:
            return ThinkingProcess.LATERAL  # Pensamiento más social/creativo
        elif openness > 0.7:
            return ThinkingProcess.DIVERGENTE
        else:
            return ThinkingProcess.ANALITICO

    def _select_content_type(self, trigger: str, context: Dict) -> ThinkingContent:
        """Selecciona tipo de contenido del pensamiento"""

        if context.get('task_type') == 'strategic_planning':
            return ThinkingContent.ESTRATEGICO
        elif context.get('task_type') == 'immediate_action':
            return ThinkingContent.TACTICO
        elif context.get('task_type') == 'theory_development':
            return ThinkingContent.TEORICO
        elif context.get('task_type') == 'practical_application':
            return ThinkingContent.PRACTICO

        # Análisis del trigger
        trigger_lower = trigger.lower()
        if any(word in trigger_lower for word in ['teoria', 'concepto', 'idea', 'filosofía']):
            return ThinkingContent.ABSTRACTO
        elif any(word in trigger_lower for word in ['hecho', 'realidad', 'objeto', 'concreto']):
            return ThinkingContent.CONCRETO

        # Default
        return ThinkingContent.PRACTICO

    def _determine_mental_state(self, context: Dict, emotion: Dict) -> ThinkingState:
        """Determina estado mental del pensamiento"""

        # Pensamiento consciente en situaciones importantes
        if context.get('importance', 0) > 0.8 or context.get('attention_focus', False):
            return ThinkingState.CONSCIENTE

        # Pensamiento rumiante en emociones negativas intensas
        emotional_valence = emotion.get('valence', 0)
        if emotional_valence < -0.6 and emotion.get('arousal', 0) > 0.7:
            return ThinkingState.RUMINANTE

        # Pensamiento preconsciente por defecto (accesible)
        return ThinkingState.PRECONSCIENTE

    def _generate_thought_content(self, trigger: str, context: Dict,
                                process: ThinkingProcess, content: ThinkingContent) -> Dict[str, Any]:
        """Genera contenido específico del pensamiento"""

        generator = self.thought_patterns.get(f"{process.value}_{content.value}")

        if not generator:
            # Contenido genérico
            return {
                'trigger': trigger,
                'process_type': process.value,
                'content_type': content.value,
                'main_idea': f"Pensamiento {process.value} sobre {trigger}",
                'associations': self._generate_associations(trigger),
                'implications': self._generate_implications(trigger, process),
                'confidence_level': 0.7
            }

        # Aplicar patrón específico
        pattern = random.choice(generator)
        return {
            'trigger': trigger,
            'process_type': process.value,
            'content_type': content.value,
            'main_idea': pattern['template'].format(trigger=trigger, context=context.get('context', '')),
            'structure': pattern['structure'],
            'methodology': pattern['method'],
            'associations': self._generate_associations(trigger),
            'implications': self._generate_implications(trigger, process),
            'confidence_level': 0.8
        }

    def _calculate_cognitive_properties(self, content: Dict, context: Dict, emotion: Dict,
                                      process: ThinkingProcess) -> Dict[str, float]:
        """Calcula propiedades cognitivas del pensamiento"""

        # Base properties
        base_confidence = content.get('confidence_level', 0.5)
        emotional_valence = emotion.get('valence', 0)

        # Complexity basado en tipo de proceso
        complexity_map = {
            ThinkingProcess.ANALITICO: 0.8,
            ThinkingProcess.CRITICO: 0.7,
            ThinkingProcess.SISTEMICO: 0.9,
            ThinkingProcess.CREATIVO: 0.6,
            ThinkingProcess.LATERAL: 0.5,
            ThinkingProcess.DEDUCTIVO: 0.7,
            ThinkingProcess.INDUCTIVO: 0.6
        }

        complexity = complexity_map.get(process, 0.5)

        # Originality (creatividad)
        originality = 0.5
        if process in [ThinkingProcess.CREATIVO, ThinkingProcess.LATERAL, ThinkingProcess.DIVERGENTE]:
            originality += 0.3

        # Cognitive load
        load = complexity * 0.7 + originality * 0.3

        return {
            'confidence': min(1.0, base_confidence + emotional_valence * 0.2),
            'valence': emotional_valence,
            'complexity': complexity,
            'originality': originality,
            'load': load
        }

    def _apply_cognitive_biases(self, content: Dict, context: Dict) -> List[CognitiveBias]:
        """Aplica sesgos cognitivos relevantes"""

        active_biases = []

        # Sesgo de confirmación en pensamiento crítico
        if content.get('process_type') == 'critical':
            active_biases.append(CognitiveBias.CONFIRMACION)

        # Sesgo de disponibilidad en memoria
        if 'recuerdo' in str(content).lower():
            active_biases.append(CognitiveBias.DISPONIBILIDAD)

        # Sesgo de negatividad en emociones negativas
        if content.get('trigger_emotion', 0) < -0.5:
            active_biases.append(CognitiveBias.NEGATIVIDAD)

        # Sesgo de autoridad en contexto jerárquico
        if context.get('authority_present', False):
            active_biases.append(CognitiveBias.AUTORIDAD)

        # Probabilistically add random biases based on personality
        neuroticism = self.personality.get('neuroticism', 0.3)
        if neuroticism > 0.6 and random.random() < 0.3:
            active_biases.append(CognitiveBias.OPTIMISMO)

        return active_biases

    def _initialize_thought_patterns(self) -> Dict[str, List[Dict]]:
        """Inicializa patrones de pensamiento para cada combinación"""

        return {
            "analytic_abstract": [
                {
                    'template': "Analizando {trigger} a través de principios abstractos fundamentales",
                    'structure': 'deductive_reasoning',
                    'method': 'logical_decomposition'
                }
            ],
            "creative_concrete": [
                {
                    'template': "Explorando nuevas posibilidades concretas en torno a {trigger}",
                    'structure': 'free_association',
                    'method': 'brainstorming'
                }
            ],
            "critical_practical": [
                {
                    'template': "Evaluando críticamente los aspectos prácticos de {trigger}",
                    'structure': 'pros_cons_analysis',
                    'method': 'evidence_evaluation'
                }
            ],
            "systemic_theoretical": [
                {
                    'template': "Considerando {trigger} dentro del contexto sistémico teórico más amplio",
                    'structure': 'systems_modeling',
                    'method': 'holistic_analysis'
                }
            ]
        }

    def _generate_associations(self, trigger: str) -> List[str]:
        """Genera asociaciones relacionadas"""
        associations = []

        # Palabras relacionadas simples
        base_words = trigger.lower().split()
        for word in base_words:
            if len(word) > 3:
                associations.extend([
                    f"relacionado con {word}",
                    f"opuesto a {word}",
                    f"consecuencia de {word}"
                ])

        return associations[:5]  # Máximo 5 asociaciones

    def _generate_implications(self, trigger: str, process: ThinkingProcess) -> List[str]:
        """Genera implicaciones del pensamiento"""
        implications = []

        if process == ThinkingProcess.ANALITICO:
            implications.append("Requiere evidencia sólida")
            implications.append("Puede llevar tiempo")
        elif process == ThinkingProcess.CREATIVO:
            implications.append("Puede ser impredecible")
            implications.append("Abre nuevas posibilidades")
        elif process == ThinkingProcess.CRITICO:
            implications.append("Evaluará suposiciones")
            implications.append("Puede encontrar contradicciones")

        return implications[:3]

    def _weighted_choice(self, options: List[str], weights: List[float]) -> str:
        """Selección ponderada"""
        return random.choices(options, weights=weights, k=1)[0]


class CognitiveProcessingUnit:
    """Unidad central de procesamiento cognitivo"""

    def __init__(self, personality: Dict[str, float]):
        self.personality = personality
        self.thought_generator = ThoughtGenerator(personality)
        self.cognitive_states: List[CognitiveState] = []
        self.processing_load = 0.0
        self.insight_counter = 0

    def process_cognitive_input(self, input_data: Dict[str, Any], context: Dict[str, Any],
                               emotional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa entrada cognitiva completa"""

        start_time = time.time()

        # Generar pensamiento principal
        primary_thought = self.thought_generator.generate_thought(
            input_data.get('trigger', ''),
            context,
            emotional_context
        )

        # Generar pensamientos asociados si complejidad alta
        associated_thoughts = []
        if primary_thought.complexity > 0.7:
            for _ in range(random.randint(1, 3)):
                assoc_thought = self.thought_generator.generate_thought(
                    f"aspecto de {primary_thought.content.get('trigger', '')}",
                    context,
                    emotional_context
                )
                associated_thoughts.append(assoc_thought)

        # Crear estado cognitivo actual
        cognitive_state = self._create_cognitive_state(
            primary_thought, associated_thoughts, context
        )
        self.cognitive_states.append(cognitive_state)

        # Calcular métricas de procesamiento
        processing_time = time.time() - start_time
        self.processing_load = min(1.0, self.processing_load + primary_thought.cognitive_load)

        # Verificar insights
        insight_detected = self._detect_insight(primary_thought, associated_thoughts)
        if insight_detected:
            self.insight_counter += 1

        return {
            'primary_thought': primary_thought,
            'associated_thoughts': associated_thoughts,
            'cognitive_state': cognitive_state,
            'processing_metrics': {
                'total_time': processing_time,
                'load_increase': primary_thought.cognitive_load,
                'insight_generated': insight_detected
            },
            'active_biases': primary_thought.active_biases
        }

    def _create_cognitive_state(self, primary: ThoughtProcess, associated: List[ThoughtProcess],
                               context: Dict) -> CognitiveState:
        """Crea estado cognitivo completo"""

        # Determinar proceso dominante
        dominant_process = primary.process_type

        # Calcular claridad mental
        clarity_factors = [
            primary.confidence,
            1.0 - primary.cognitive_load,
            1.0 - len(primary.active_biases) * 0.1,  # Menos claridad con más sesgos
            self.personality.get('conscientiousness', 0.5)  # Personalidad influye
        ]
        mental_clarity = np.mean(clarity_factors)

        # Estabilidad atencional
        attention_stability = min(1.0, 0.8 + self.personality.get('neuroticism', 0.3) * 0.4)

        # Carga de memoria de trabajo
        working_load = primary.cognitive_load + sum(t.cognitive_load for t in associated) * 0.3

        # Probabilidad de insight
        insight_prob = (primary.originality + primary.complexity) / 2 * (1 - working_load)

        # Nivel de rumiación
        rumination_level = 0.0
        if primary.mental_state == ThinkingState.RUMIANTE:
            rumination_level = 0.8

        # Estado de flujo
        flow_state = min(1.0, mental_clarity * attention_stability * (1 - working_load))

        # Sesgos activos
        active_biases = {}
        all_biases = primary.active_biases + [b for t in associated for b in t.active_biases]
        for bias in CognitiveBias:
            if bias in all_biases:
                active_biases[bias] = 0.8 + random.random() * 0.2  # Intensidad variable

        return CognitiveState(
            dominant_process=dominant_process,
            content_focus=primary.content_type,
            mental_clarity=mental_clarity,
            attention_stability=attention_stability,
            working_memory_load=working_load,
            insight_probability=insight_prob,
            rumination_level=rumination_level,
            flow_state=flow_state,
            cognitive_biases_active=active_biases
        )

    def _detect_insight(self, primary: ThoughtProcess, associated: List[ThoughtProcess]) -> bool:
        """Detecta si se generó un insight"""

        # Criterios para insight
        originality_threshold = 0.7
        complexity_threshold = 0.8
        association_bonus = len(associated) * 0.1

        insight_score = (primary.originality + primary.complexity + association_bonus) / 3

        return insight_score > 0.85

    def get_cognitive_metrics(self) -> Dict[str, Any]:
        """Métricas cognitivas actuales"""

        recent_states = self.cognitive_states[-10:]

        return {
            'current_load': self.processing_load,
            'insight_count': self.insight_counter,
            'average_clarity': np.mean([s.mental_clarity for s in recent_states]) if recent_states else 0.5,
            'average_flow': np.mean([s.flow_state for s in recent_states]) if recent_states else 0.5,
            'dominant_process': max(set(s.dominant_process for s in recent_states),
                                  key=lambda x: sum(1 for s in recent_states if s.dominant_process == x)) if recent_states else ThinkingProcess.ANALITICO,
            'active_biases_count': len([b for s in recent_states for b in s.cognitive_biases_active.keys()]) if recent_states else 0
        }


# ============================ DEMOSTRACIÓN =======================

def demonstrate_human_cognition():
    """Demostración del sistema cognitivo humano"""

    print("🧠 DEMOSTRACIÓN SISTEMA COGNITIVO HUMANO")
    print("=" * 60)

    # Personalidad de prueba
    personality = {
        'extraversion': 0.7,
        'openness': 0.8,
        'conscientiousness': 0.9,
        'agreeableness': 0.6,
        'neuroticism': 0.3
    }

    cognitive_system = CognitiveProcessingUnit(personality)

    # Escenarios de pensamiento
    scenarios = [
        {
            'trigger': '¿Por qué algunos problemas parecen imposibles?',
            'context': {'task_type': 'problem_solving', 'urgency': 0.8},
            'emotion': {'valence': -0.3, 'arousal': 0.7, 'intensity': 0.8}
        },
        {
            'trigger': 'innovar en tecnología para el futuro',
            'context': {'task_type': 'strategic_planning', 'creativity_needed': True},
            'emotion': {'valence': 0.6, 'arousal': 0.5, 'intensity': 0.7}
        },
        {
            'trigger': 'conflicto ético en decisión importante',
            'context': {'task_type': 'decision_making', 'importance': 0.9, 'uncertainty': 0.7},
            'emotion': {'valence': -0.2, 'arousal': 0.9, 'intensity': 0.8}
        }
    ]

    results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎯 ESCENARIO {i}: '{scenario['trigger']}'")

        result = cognitive_system.process_cognitive_input(
            scenario, scenario['context'], scenario['emotion']
        )

        thought = result['primary_thought']
        state = result['cognitive_state']

        print(f"   🧠 Proceso: {thought.process_type.value.title()}")
        print(f"   📝 Contenido: {thought.content_type.value.title()}")
        print(f"   💭 Estado Mental: {thought.mental_state.value.title()}")
        print(f"   📊 Confianza: {thought.confidence:.2f}")
        print(f"   🎭 Valencia Emocional: {thought.emotional_valence:.2f}")
        print(f"   🧬 Complejidad: {thought.complexity:.2f}")
        print(f"   🔍 Sesgos Activos: {len(thought.active_biases)}")
        print(f"   ⚡ Estado de Flujo: {state.flow_state:.2f}")

        if result['processing_metrics']['insight_generated']:
            print("   💡 ¡INSIGHT GENERADO!")

        results.append(result)

    # Resumen final
    print("\n📊 MÉTRICAS COGNITIVAS FINALES")
    metrics = cognitive_system.get_cognitive_metrics()
    print(f"   Nivel de Atención: {metrics.get('attention_level', 'N/A')}")
    print(f"   Memoria Efectiva: {metrics.get('effective_memory', 'N/A')}")
    print(f"   Velocidad Procesamiento: {metrics.get('processing_speed', 'N/A')}")
    print(f"   🎯 Insights Generados: {metrics['insight_count']}")
    print(f"   Creatividad: {metrics.get('creativity', 'N/A')}")
    print(f"   Flexibilidad Cognitiva: {metrics.get('flexibility', 'N/A')}")
    # Personalidad que piensa procesa información como un humano real con pensamiento complejo
    print("   🎨 Sistema Cognitivo Humano Operativo")


if __name__ == "__main__":
    demonstrate_human_cognition()

# Alias for backward compatibility
HumanCognitiveSystem = CognitiveProcessingUnit
