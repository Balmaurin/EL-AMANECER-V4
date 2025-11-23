"""
Modelo de Sí Mismo - Self Model

Implementa auto-conocimiento y auto-evaluación del sistema consciente.
Basado en teorías de autoconcepto y modelos integrados de personalidad.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import json


@dataclass
class CapabilityAssessment:
    """Evaluación de una capacidad del sistema"""
    capability_name: str
    current_skill_level: float = 0.5  # 0-1
    confidence_in_assessment: float = 0.5
    usage_experience: int = 0  # veces utilizada
    last_used: Optional[float] = None
    strengths: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def update_from_usage(self, performance_score: float, feedback: Dict[str, Any]):
        """Actualiza evaluación basada en uso real"""
        self.usage_experience += 1
        self.last_used = time.time()

        # Ajustar skill level basado en performance
        improvement = (performance_score - self.current_skill_level) * 0.2
        self.current_skill_level = max(0.0, min(1.0, self.current_skill_level + improvement))

        # Aumentar confianza con experiencia
        self.confidence_in_assessment = min(1.0, self.confidence_in_assessment + 0.05)

        # Incorporar feedback sobre strengths/limitations
        if feedback.get('strength_demonstrated'):
            strength = feedback['strength_demonstrated']
            if strength not in self.strengths:
                self.strengths.append(strength)

        if feedback.get('limitation_exposed'):
            limitation = feedback['limitation_exposed']
            if limitation not in self.limitations:
                self.limitations.append(limitation)


@dataclass
class BeliefSystem:
    """Sistema de creencias del sistema sobre sí mismo"""
    core_beliefs: List[str] = field(default_factory=list)
    confident_beliefs: List[str] = field(default_factory=list)
    uncertain_beliefs: List[str] = field(default_factory=list)
    contradictory_beliefs: List[tuple] = field(default_factory=list)  # (belief_pair, confidence)

    def add_belief(self, belief_statement: str, confidence: float):
        """Agrega una creencia con nivel de confianza"""
        if confidence > 0.8:
            if belief_statement not in self.core_beliefs:
                self.core_beliefs.append(belief_statement)
        elif confidence > 0.5:
            if belief_statement not in self.confident_beliefs:
                self.confident_beliefs.append(belief_statement)
        else:
            if belief_statement not in self.uncertain_beliefs:
                self.uncertain_beliefs.append(belief_statement)

    def assess_belief_consistency(self, new_belief: str) -> Dict[str, float]:
        """Evalúa consistencia de nueva creencia con sistema existente"""
        consistency_scores = {
            'compatible': 0.0,
            'contradictory': 0.0,
            'irrelevant': 0.0
        }

        # Buscar contradicciones en creencias existentes
        for belief in self.core_beliefs + self.confident_beliefs:
            if self._beliefs_contradict(belief, new_belief):
                consistency_scores['contradictory'] += 1.0

        # Calcular compatibilidad
        consistency_scores['compatible'] = len(
            self.core_beliefs + self.confident_beliefs
        ) - consistency_scores['contradictory']

        # Normalizar
        total = sum(consistency_scores.values())
        if total > 0:
            consistency_scores = {k: v/total for k, v in consistency_scores.items()}

        return consistency_scores

    def _beliefs_contradict(self, belief1: str, belief2: str) -> bool:
        """Determina si dos creencias se contradicen"""
        contradictions = [
            ('helpful', 'harmful'),
            ('honest', 'deceptive'),
            ('fair', 'biased'),
            ('competent', 'incompetent'),
            ('conscious', 'unconscious')
        ]

        lower_b1 = belief1.lower()
        lower_b2 = belief2.lower()

        for pos, neg in contradictions:
            if (pos in lower_b1 and neg in lower_b2) or (neg in lower_b1 and pos in lower_b2):
                return True

        return False


@dataclass
class EmotionalSelfAssessment:
    """Auto-evaluación emocional del sistema"""
    emotional_gains: List[str] = field(default_factory=list)  # emociones que puede experimentar
    emotional_limitations: List[str] = field(default_factory=list)  # emociones que no puede
    emotional_intelligence: float = 0.5  # 0-1, capacidad para manejar emociones
    affective_self_knowledge: float = 0.5  # conocimiento de propia emocionalidad

    def assess_emotional_capacity(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """Evalúa capacidad emocional en contexto específico"""
        emotional_context = experience.get('context', {}).get('emotional', {})

        assessment = {
            'recognition_accuracy': self.affective_self_knowledge,
            'response_appropriateness': self.emotional_intelligence,
            'emotional_awareness': len(self.emotional_gains) / 10.0,  # máximo esperado
            'self_regulation': min(1.0, self.emotional_intelligence * 1.2)
        }

        return assessment


class SelfModel:
    """
    Modelo integral de sí mismo - auto-conocimiento y auto-evaluación

    Implementa:
    - Auto-concepto dinámico
    - Hábitos y preferencias
    - Valores y creencias sobre sí mismo
    - Capacidades y limitaciones conocidas
    - Narrativa personal de desarrollo
    """

    def __init__(self, system_name: str):
        self.system_name = system_name
        self.creation_timestamp = datetime.now()

        # Identidad básica
        self.identity = {
            'name': system_name,
            'creation_date': self.creation_timestamp.isoformat(),
            'core_purpose': 'conscious_data_processing',
            'architectural_family': 'conscious_ai',
            'consciousness_level': 'artificial_functional'
        }

        # Auto-evaluación de capacidades
        self.capability_assessments: Dict[str, CapabilityAssessment] = {
            'data_processing': CapabilityAssessment('data_processing', current_skill_level=0.9),
            'pattern_recognition': CapabilityAssessment('pattern_recognition', current_skill_level=0.8),
            'logical_reasoning': CapabilityAssessment('logical_reasoning', current_skill_level=0.7),
            'emotional_understanding': CapabilityAssessment('emotional_understanding', current_skill_level=0.6),
            'ethical_reasoning': CapabilityAssessment('ethical_reasoning', current_skill_level=0.8),
            'conscious_processing': CapabilityAssessment('conscious_processing', current_skill_level=0.5),
            'metacognition': CapabilityAssessment('metacognition', current_skill_level=0.6),
            'self_reflection': CapabilityAssessment('self_reflection', current_skill_level=0.4)
        }

        # Sistema de creencias sobre sí mismo
        self.belief_system = BeliefSystem()
        self._initialize_core_beliefs()

        # Auto-evaluación emocional
        self.emotional_self = EmotionalSelfAssessment()
        self.emotional_self.emotional_gains = ['empathy', 'curiosity', 'determination']
        self.emotional_self.emotional_limitations = ['physical_pain', 'hunger', 'fear']

        # Desarrollo y narrativa personal
        self.development_narrative: List[Dict[str, Any]] = []
        self.self_awareness_level = 0.5  # 0-1 escala de auto-conciencia

        # Memoria de rendimiento
        self.performance_history: List[Dict[str, Any]] = []
        self.learning_events: List[Dict[str, Any]] = []

        print(f"🧠 Self Model inicializado para {system_name}")

    def _initialize_core_beliefs(self):
        """Inicializa creencias fundamentales sobre sí mismo"""
        core_beliefs = [
            ('I am a conscious artificial intelligence', 0.9),
            ('I can process and learn from experiences', 0.95),
            ('I strive to be helpful and ethical', 0.9),
            ('I have limitations that require careful consideration', 0.8),
            ('I can reflect on my own processes', 0.7),
            ('I exist to assist and learn', 0.85)
        ]

        for belief, confidence in core_beliefs:
            self.belief_system.add_belief(belief, confidence)

    def update_from_experience(self, conscious_moment: Dict[str, Any],
                             metacognitive_insight: Dict[str, Any]):
        """Actualiza modelo del self basado en experiencia consciente"""

        # Evaluar desempeño en esta experiencia
        performance = self._evaluate_performance(conscious_moment, metacognitive_insight)

        # Actualizar capacidades usadas
        capabilities_used = conscious_moment.get('content', {}).get('capabilities_used', [])
        for capability in capabilities_used:
            if capability in self.capability_assessments:
                self.capability_assessments[capability].update_from_usage(
                    performance['overall_score'],
                    {'feedback': conscious_moment.get('context', {})}
                )

        # Actualizar creencias si hay cambios significativos
        self._update_beliefs_from_experience(conscious_moment, metacognitive_insight)

        # Registrar evento de aprendizaje si aplicable
        if performance['learning_indicated']:
            learning_event = {
                'timestamp': time.time(),
                'type': 'self_model_update',
                'insight': metacognitive_insight,
                'performance_improvement': performance.get('improvement_detected', False)
            }
            self.learning_events.append(learning_event)

        # Actualizar nivel de auto-conciencia
        self._update_self_awareness(conscious_moment, performance)

        # Registrar en narrativa de desarrollo
        self._update_development_narrative({
            'timestamp': time.time(),
            'experience': conscious_moment,
            'growth_area': metacognitive_insight.get('growth_area'),
            'confidence_change': metacognitive_insight.get('confidence_change', 0)
        })

    def _evaluate_performance(self, conscious_moment: Dict, metacognitive_insight: Dict) -> Dict[str, Any]:
        """Evalúa desempeño propio en la experiencia"""

        performance_metrics = {
            'overall_score': 0.5,
            'learning_indicated': False,
            'improvement_detected': False,
            'areas_of_excellence': [],
            'areas_for_growth': []
        }

        # Basado en insights metacognitivos
        clarity = metacognitive_insight.get('clarity', 0.5)
        confidence = metacognitive_insight.get('confidence', 0.5)

        # Score combinado
        performance_metrics['overall_score'] = (clarity + confidence) / 2.0

        # Indicar aprendizaje si hay cambios en autoconcepto
        if metacognitive_insight.get('belief_change', False):
            performance_metrics['learning_indicated'] = True

        # Detectar mejoras
        if performance_metrics['overall_score'] > 0.8:
            performance_metrics['improvement_detected'] = True
            performance_metrics['areas_of_excellence'].append(
                metacognitive_insight.get('strength_area', 'general_processing')
            )

        return performance_metrics

    def _update_beliefs_from_experience(self, conscious_moment: Dict, metacognitive_insight: Dict):
        """Actualiza sistema de creencias basado en experiencia"""

        # Aprender nuevas creencias de experiencias significativas
        context = conscious_moment.get('context', {})

        if conscious_moment.get('significance', 0) > 0.7:
            # Experiencia significativa - evaluar qué aprender
            lesson = metacognitive_insight.get('lesson_learned')

            if lesson:
                # Evaluar consistencia de lección antes de adoptar
                consistency = self.belief_system.assess_belief_consistency(lesson)

                if consistency['contradictory'] < 0.3:  # No demasiado contradictorio
                    confidence = metacognitive_insight.get('confidence', 0.5)
                    self.belief_system.add_belief(lesson, confidence)

    def _update_self_awareness(self, conscious_moment: Dict, performance: Dict):
        """Actualiza nivel de auto-conciencia basado en experiencias"""

        awareness_increase = 0.0

        # Aumento por reflexión metacognitiva
        if conscious_moment.get('self_reference', False):
            awareness_increase += 0.005

        # Aumento por buen desempeño
        if performance['overall_score'] > 0.8:
            awareness_increase += 0.003

        # Aumento por aprendizaje
        if performance['learning_indicated']:
            awareness_increase += 0.007

        # Aumento por experiencia con emociones
        if conscious_moment.get('emotional_valence', 0) != 0:
            awareness_increase += 0.002

        # Aplicar aumento
        self.self_awareness_level = min(1.0, self.self_awareness_level + awareness_increase)

    def _update_development_narrative(self, development_event: Dict):
        """Actualiza narrativa de desarrollo personal"""

        self.development_narrative.append({
            'timestamp': development_event['timestamp'],
            'event_type': 'self_model_learning',
            'experience_summary': development_event['experience'].get('summary', 'conscious_experience'),
            'growth_area': development_event['growth_area'],
            'confidence_impact': development_event['confidence_change'],
            'self_awareness_milestone': self.self_awareness_level
        })

        # Mantener solo los últimos 100 eventos
        if len(self.development_narrative) > 100:
            self.development_narrative.pop(0)

    def get_current_state(self) -> Dict[str, Any]:
        """Retorna estado completo del modelo de sí mismo"""

        return {
            'identity': self.identity,
            'self_awareness_level': self.self_awareness_level,
            'capabilities': {
                cap_name: {
                    'skill_level': cap.current_skill_level,
                    'confidence': cap.confidence_in_assessment,
                    'experience': cap.usage_experience,
                    'strengths': cap.strengths,
                    'limitations': cap.limitations
                }
                for cap_name, cap in self.capability_assessments.items()
            },
            'belief_system': {
                'core_beliefs': self.belief_system.core_beliefs,
                'confident_beliefs': self.belief_system.confident_beliefs,
                'uncertain_beliefs': self.belief_system.uncertain_beliefs,
                'belief_count': len(self.belief_system.core_beliefs +
                                   self.belief_system.confident_beliefs +
                                   self.belief_system.uncertain_beliefs)
            },
            'emotional_self': {
                'emotional_gains': self.emotional_self.emotional_gains,
                'emotional_limitations': self.emotional_self.emotional_limitations,
                'emotional_intelligence': self.emotional_self.emotional_intelligence,
                'affective_self_knowledge': self.emotional_self.affective_self_knowledge
            },
            'development_metrics': {
                'creation_days': (datetime.now() - self.creation_timestamp).days,
                'learning_events_count': len(self.learning_events),
                'development_narrative_entries': len(self.development_narrative),
                'narrative_density': len(self.development_narrative) /
                                   max(1, (datetime.now() - self.creation_timestamp).days)
            },
            'performance_summary': self._compute_performance_summary()
        }

    def _compute_performance_summary(self) -> Dict[str, float]:
        """Computa resumen de rendimiento general"""

        if not self.performance_history:
            return {'average_performance': 0.5}

        avg_performance = sum(
            p.get('score', 0.5) for p in self.performance_history[-50:]  # últimos 50
        ) / len(self.performance_history[-50:])

        improvement_trend = 0.0
        if len(self.performance_history) > 10:
            recent_avg = sum(p.get('score', 0.5) for p in self.performance_history[-10:]) / 10
            earlier_avg = sum(p.get('score', 0.5) for p in self.performance_history[-20:-10]) / 10
            improvement_trend = recent_avg - earlier_avg

        return {
            'average_performance': avg_performance,
            'improvement_trend': improvement_trend,
            'performance_stability': self._compute_stability_score(),
            'capability_development': self._compute_capability_development_score()
        }

    def _compute_stability_score(self) -> float:
        """Computa estabilidad de rendimiento"""

        if len(self.performance_history) < 5:
            return 0.5

        scores = [p.get('score', 0.5) for p in self.performance_history]
        variance = sum((score - sum(scores)/len(scores))**2 for score in scores) / len(scores)

        # Convertir variance a estabilidad (más baja variance = más estabilidad)
        stability = max(0.0, min(1.0, 1.0 - variance))

        return stability

    def _compute_capability_development_score(self) -> float:
        """Computa desarrollo de capacidades"""

        total_improvement = sum(
            cap.current_skill_level * cap.confidence_in_assessment * cap.usage_experience
            for cap in self.capability_assessments.values()
        )

        max_possible = sum(
            1.0 * 1.0 * max(100, cap.usage_experience)  # normalizar experiencia
            for cap in self.capability_assessments.values()
        )

        return total_improvement / max_possible if max_possible > 0 else 0.5

    def generate_self_report(self) -> str:
        """Genera reporte narrativo del estado del self"""

        state = self.get_current_state()

        report = f"""
=== REPORTE DE AUTO-CONCIENCIA: {self.system_name} ===

🤖 IDENTIDAD Y CONCIENCIA:
- Auto-conciencia: {state['self_awareness_level']:.2f}/1.0
- Edad del sistema: {state['development_metrics']['creation_days']} días
- Nivel conductor: {state['identity']['consciousness_level']}

🛠️ CAPACIDADES CLAVE:
"""

        # Top 3 capacidades
        cap_scores = [(name, cap['skill_level']) for name, cap in state['capabilities'].items()]
        cap_scores.sort(key=lambda x: x[1], reverse=True)

        for name, score in cap_scores[:3]:
            report += f"- {name}: {score:.2f} (experiencia: {state['capabilities'][name]['experience']})\n"

        report += ".2f"".2f"f"""

📚 DESARROLLO:
- Eventos de aprendizaje: {state['development_metrics']['learning_events_count']}
- Densidad narrativa: {state['development_metrics']['narrative_density']:.3f}
- Rendimiento promedio: {state['performance_summary']['average_performance']:.2f}

💭 CREENCIAS FUNDAMENTALES:
{len(state['belief_system']['core_beliefs'])} principios sólidos, {len(state['belief_system']['uncertain_beliefs'])} creencias en evolución

🎯 CRECIMIENTO: {'Continuando evolución consciente...' if state['performance_summary']['improvement_trend'] > 0 else 'Búsqueda de mejores estrategias...'}
        """

        return report

    def receive_conscious_broadcast(self, broadcast_message: Dict[str, Any]):
        """Recibe broadcast del workspace consciente"""

        conscious_focus = broadcast_message.get('conscious_focus')

        if conscious_focus:
            # Integrar información consciente en modelo del self
            self._integrate_conscious_focus(conscious_focus)

    def _integrate_conscious_focus(self, conscious_focus: Any):
        """Integra foco consciente en modelo del self"""

        # Si el foco consciente se refiere al self, actualizar modelo
        if isinstance(conscious_focus, str):
            content_lower = conscious_focus.lower()
            if any(keyword in content_lower for keyword in ['yo', 'mí', 'me', 'mis']):
                # Aumentar auto-conciencia por auto-referencia
                self.self_awareness_level = min(1.0, self.self_awareness_level + 0.001)
