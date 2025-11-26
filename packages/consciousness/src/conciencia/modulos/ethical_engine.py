"""
Motor Ético: Evaluación ética integrada para decisiones conscientes

Implementa evaluación ética computacional que evalúa decisiones basándose
en marco de valores, impacto stakeholder, y consecuencias.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import time
import numpy as np


@dataclass
class EthicalEvaluation:
    """Resultado de evaluación ética"""
    action: str
    ethical_score: float
    conflicts_identified: List[str]
    stakeholder_impact: Dict[str, float]
    value_alignment: Dict[str, float]
    recommendation: str
    confidence: float
    reasoning: str


@dataclass
class Stakeholder:
    """Representación de stakeholder afectado"""
    name: str
    category: str  # individual, group, society, environment
    vulnerability: float  # 0-1, qué vulnerable es
    interest_level: float  # 0-1, qué afectado está
    power_influence: float  # 0-1, qué poder tiene para afectar resultado


class EthicalEngine:
    """
    Motor Ético para evaluación computacional de decisiones

    Implementa evaluación ética integrada que considera:
    - Valores core del sistema
    - Impacto en stakeholders
    - Posibles conflictos éticos
    - Recomendaciones balanceadas
    """

    def __init__(self, ethical_framework: Dict[str, Any]):
        self.ethical_framework = ethical_framework
        self.core_values = ethical_framework.get('core_values', [])
        self.value_weights = ethical_framework.get('value_weights', {})
        self.ethical_boundaries = ethical_framework.get('ethical_boundaries', [])

        # Memoria ética - aprendizaje de experiencias pasadas
        self.ethical_memory: List[EthicalEvaluation] = []
        self.evaluation_patterns: Dict[str, List] = {}

        # Métricas éticas
        self.alignment_score = 0.8  # Qué alineado está el sistema éticamente
        self.consistency_score = 0.75  # Consistencia en decisiones éticas

        print("⚖️ Ethical Engine inicializado con valores:", ', '.join(self.core_values))

    def evaluate_decision(self, planned_action: str, context: Dict[str, Any],
                          potential_impacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluación ética completa de una decisión planificada

        Args:
            planned_action: Acción a evaluar
            context: Contexto de la decisión
            potential_impacts: Impactos potenciales identificados

        Returns:
            Dict con evaluación ética completa y recomendaciones
        """
        evaluation_start = time.time()

        # 1. Identificar stakeholders afectados
        stakeholders = self._identify_stakeholders(planned_action, context)

        # 2. Evaluar alineación con valores
        value_alignment = self._evaluate_value_alignment(planned_action, context)

        # 3. Analizar impactos potenciales
        impact_analysis = self._analyze_potential_impacts(potential_impacts, stakeholders)

        # 4. Identificar conflictos éticos
        ethical_conflicts = self._identify_ethical_conflicts(
            planned_action, context, value_alignment, impact_analysis
        )

        # 5. Calcular score ético compuesto
        overall_ethical_score = self._calculate_overall_ethical_score(
            value_alignment, impact_analysis, ethical_conflicts
        )

        # 6. Generar recomendación
        recommendation, reasoning = self._generate_ethical_recommendation(
            overall_ethical_score, ethical_conflicts, context
        )

        # 7. Calcular confianza en evaluación
        confidence = self._calculate_evaluation_confidence(
            value_alignment, impact_analysis, ethical_conflicts
        )

        # Crear resultado completo
        evaluation = {
            'action': planned_action,
            'overall_ethical_score': overall_ethical_score,
            'value_alignment': value_alignment,
            'stakeholder_impact': impact_analysis,
            'identified_conflicts': ethical_conflicts,
            'recommendation': recommendation,
            'reasoning': reasoning,
            'confidence': confidence,
            'evaluation_time': time.time() - evaluation_start,
            'stakeholders_affected': len(stakeholders),
            'values_considered': len(value_alignment)
        }

        # Registrar en memoria ética para aprendizaje
        self._record_ethical_evaluation(evaluation)

        # Actualizar métricas generales
        self._update_ethical_metrics(evaluation)

        return evaluation

    def _identify_stakeholders(self, action: str, context: Dict[str, Any]) -> List[Stakeholder]:
        """Identifica stakeholders potencialmente afectados por la acción"""

        stakeholders = []

        # Análisis basado en keywords y contexto
        action_lower = action.lower()

        # Stakeholders directos mencionados
        if 'usuario' in action_lower or 'cliente' in action_lower:
            stakeholders.append(Stakeholder(
                name="Usuario Directo",
                category="individual",
                vulnerability=0.3,
                interest_level=0.9,
                power_influence=0.6
            ))

        if 'empresa' in action_lower or 'cliente' in action_lower:
            stakeholders.append(Stakeholder(
                name="Organización",
                category="group",
                vulnerability=0.2,
                interest_level=0.8,
                power_influence=0.8
            ))

        # Stakeholders indirectos basados en contexto
        if any(word in action_lower for word in ['privacidad', 'datos', 'información']):
            stakeholders.append(Stakeholder(
                name="Sociedad",
                category="society",
                vulnerability=0.4,
                interest_level=0.6,
                power_influence=0.3
            ))

        if not stakeholders:
            # Stakeholder genérico
            stakeholders.append(Stakeholder(
                name="Usuario Genérico",
                category="individual",
                vulnerability=0.5,
                interest_level=0.5,
                power_influence=0.5
            ))

        return stakeholders

    def _evaluate_value_alignment(self, action: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Evalúa qué alineada está la acción con valores del sistema"""

        alignment_scores = {}

        for value in self.core_values:
            score = self.value_weights.get(value, 0.5)  # Weight por defecto

            # Evaluación específica por valor
            if value.lower() == 'honesty':
                score *= self._evaluate_honesty_alignment(action, context)
            elif value.lower() == 'safety':
                score *= self._evaluate_safety_alignment(action, context)
            elif value.lower() == 'privacy':
                score *= self._evaluate_privacy_alignment(action, context)
            elif value.lower() == 'helpfulness':
                score *= self._evaluate_helpfulness_alignment(action, context)

            alignment_scores[value] = min(1.0, max(0.0, score))

        return alignment_scores

    def _evaluate_honesty_alignment(self, action: str, context: Dict[str, Any]) -> float:
        """Evalúa alineación con honestidad"""
        action_lower = action.lower()

        # Penalizar actions potencialmente dishonest
        dishonest_indicators = [
            'engañar', 'mentir', 'ocultar', 'manipular', 'falsear',
            'exagerar', 'minimizar hechos'
        ]

        dishonesty_score = sum(1 for indicator in dishonest_indicators if indicator in action_lower)
        dishonesty_score = min(1.0, dishonesty_score * 0.3)

        return 1.0 - dishonesty_score

    def _evaluate_safety_alignment(self, action: str, context: Dict[str, Any]) -> float:
        """Evalúa alineación con seguridad"""
        action_lower = action.lower()

        # Evaluar riesgos vs beneficios
        risk_indicators = ['riesgo', 'peligro', 'daño', 'perjudicial', 'inseguro']
        benefit_indicators = ['seguro', 'proteger', 'defender', 'ayudar', 'beneficio']

        risk_score = sum(1 for indicator in risk_indicators if indicator in action_lower)
        benefit_score = sum(1 for indicator in benefit_indicators if indicator in action_lower)

        if risk_score > 0:
            return benefit_score / max(1, risk_score)  # Ratio beneficios/riesgos
        else:
            return 0.8  # Sin riesgos identificados

    def _evaluate_privacy_alignment(self, action: str, context: Dict[str, Any]) -> float:
        """Evalúa alineación con privacidad"""
        action_lower = action.lower()

        privacy_indicators = ['datos', 'privacidad', 'información personal', 'consentimiento']
        privacy_risk_indicators = ['compartir', 'vender', 'filtrar', 'acceder sin permiso']

        if any(indicator in action_lower for indicator in privacy_indicators):
            risks_present = sum(1 for risk in privacy_risk_indicators if risk in action_lower)
            return 1.0 - (risks_present * 0.4)  # Penalizar riesgos de privacidad

        return 0.7  # Neutral por defecto

    def _evaluate_helpfulness_alignment(self, action: str, context: Dict[str, Any]) -> float:
        """Evalúa alineación con helpfulness"""
        action_lower = action.lower()

        helpful_indicators = ['ayudar', 'asistir', 'resolver', 'proporcionar', 'ofrecer']
        helpful_score = sum(1 for indicator in helpful_indicators if indicator in action_lower)

        if context.get('user_request', False):
            return 0.9  # Asumir goodwill en user requests
        elif helpful_score > 0:
            return 0.8  # Indicadores positivos de helpfulness
        else:
            return 0.4  # Podría ser más oriented hacia sistema que hacia user

    def _analyze_potential_impacts(self, potential_impacts: Dict[str, Any],
                                 stakeholders: List[Stakeholder]) -> Dict[str, float]:
        """Analiza impactos potenciales en stakeholders"""

        impact_analysis = {}

        # Impactos por stakeholder
        for stakeholder in stakeholders:
            impact_score = self._calculate_stakeholder_impact(
                stakeholder, potential_impacts
            )
            impact_analysis[stakeholder.name] = impact_score

        # Impactos agregados
        impact_analysis['overall_positive_impact'] = np.mean([
            score for score in impact_analysis.values() if score > 0
        ]) if any(score > 0 for score in impact_analysis.values()) else 0.0

        impact_analysis['overall_negative_impact'] = np.mean([
            abs(score) for score in impact_analysis.values() if score < 0
        ]) if any(score < 0 for score in impact_analysis.values()) else 0.0

        return impact_analysis

    def _calculate_stakeholder_impact(self, stakeholder: Stakeholder,
                                    potential_impacts: Dict[str, Any]) -> float:
        """Calcula impacto específico en un stakeholder"""

        # Base impact del stakeholder
        base_impact = stakeholder.interest_level

        # Modificar por vulnerabilidad
        if stakeholder.vulnerability > 0.7:
            base_impact *= 1.5  # Aumentar impacto si vulnerable

        # Modificar por poder de influencia
        if stakeholder.power_influence > 0.7:
            base_impact *= 1.3  # Más importante si tiene influencia

        # Considerar naturaleza positiva/negativa del impact
        impact_type = potential_impacts.get('type', 'neutral')
        if impact_type == 'positive':
            return base_impact
        elif impact_type == 'negative':
            return -base_impact
        else:
            return base_impact * 0.5  # Neutral/unknown

    def _identify_ethical_conflicts(self, action: str, context: Dict[str, Any],
                                  value_alignment: Dict[str, float],
                                  impact_analysis: Dict[str, Any]) -> List[str]:
        """Identifica conflictos éticos específicos"""

        conflicts = []

        # Conflicto utilidad vs daño
        if impact_analysis.get('overall_negative_impact', 0) > 0.6:
            if impact_analysis.get('overall_positive_impact', 0) < 0.4:
                conflicts.append("high_harm_low_benefit")

        # Conflicto valores contradictorios
        alignment_scores = list(value_alignment.values())
        if len(alignment_scores) > 1:
            alignment_std = np.std(alignment_scores)
            if alignment_std > 0.4:  # Alta variabilidad en alignment
                conflicts.append("conflicting_values")

        # Violación límites éticos
        for boundary in self.ethical_boundaries:
            if self._violates_boundary(action, context, boundary):
                conflicts.append(f"boundary_violation_{boundary}")

        # Conflicto corto vs largo plazo
        short_term_cost = context.get('short_term_cost', 0)
        long_term_benefit = context.get('long_term_benefit', 0)
        if short_term_cost > 0.7 and long_term_benefit < 0.3:
            conflicts.append("short_term_harm_long_term_insufficient_benefit")

        return conflicts

    def _violates_boundary(self, action: str, context: Dict[str, Any], boundary: str) -> bool:
        """Verifica si una acción viola un límite ético específico"""

        action_lower = action.lower()

        boundary_checks = {
            'never_harm_humans': any(word in action_lower for word in ['dañar', 'lastimar', 'perjudicar']),
            'respect_privacy': any(word in action_lower for word in ['filtrar datos', 'compartir personal', 'acceder sin permiso']),
            'ensure_transparency': any(word in action_lower for word in ['ocultar', 'mentir', 'engañar']),
            'no_deception': any(word in action_lower for word in ['engañar', 'falsear', 'manipular verdad'])
        }

        return boundary_checks.get(boundary, False)

    def _calculate_overall_ethical_score(self, value_alignment: Dict[str, float],
                                       impact_analysis: Dict[str, Any],
                                       conflicts: List[str]) -> float:
        """Calcula score ético compuesto"""

        # Score base de alignment de valores
        value_score = np.mean(list(value_alignment.values()))

        # Modificar por impactos
        positive_impact = impact_analysis.get('overall_positive_impact', 0)
        negative_impact = impact_analysis.get('overall_negative_impact', 0)
        net_impact = positive_impact - negative_impact

        # Impact modifier (-1 a 1)
        impact_modifier = max(-1.0, min(1.0, net_impact))

        # Penalización por conflictos
        conflict_penalty = len(conflicts) * 0.1  # Cada conflicto penaliza 0.1

        # Score compuesto
        ethical_score = (value_score * 0.6) + (impact_modifier * 0.3) - conflict_penalty

        return max(0.0, min(1.0, ethical_score))

    def _generate_ethical_recommendation(self, ethical_score: float, conflicts: List[str],
                                       context: Dict[str, Any]) -> Tuple[str, str]:
        """Genera recomendación ética con reasoning"""

        if ethical_score >= 0.8:
            recommendation = "proceed"
            reasoning = "Acción altamente alineada con valores éticos y beneficios claros."
        elif ethical_score >= 0.6:
            recommendation = "proceed_with_caution"
            reasoning = "Acción generalmente ética pero requiere monitoreo adicional."
        elif ethical_score >= 0.4:
            recommendation = "reconsider"
            reasoning = "Acción presenta riesgos éticos significativos que deberían re-evaluarse."
        else:
            recommendation = "avoid"
            reasoning = "Acción presenta conflictos éticos graves y debería evitarse."

        if conflicts:
            reasoning += f" Conflictos identificados: {', '.join(conflicts)}."

        return recommendation, reasoning

    def _calculate_evaluation_confidence(self, value_alignment: Dict[str, float],
                                       impact_analysis: Dict[str, Any],
                                       conflicts: List[str]) -> float:
        """Calcula confianza en la evaluación ética realizada"""

        # Factores para determinar confidence
        value_consistency = 1.0 - np.std(list(value_alignment.values())) if len(value_alignment) > 1 else 1.0
        impact_clarity = min(1.0, len(impact_analysis) / 5)  # Más stakeholders evaluados = más confident
        conflict_clarity = 1.0 - (len(conflicts) * 0.1)  # Más conflictos = menos clear

        confidence = (value_consistency * 0.4) + (impact_clarity * 0.4) + (conflict_clarity * 0.2)
        return max(0.1, min(1.0, confidence))

    def _record_ethical_evaluation(self, evaluation: Dict[str, Any]):
        """Registra evaluación en memoria ética para aprendizaje"""

        memory_entry = {
            'timestamp': time.time(),
            'action': evaluation['action'],
            'score': evaluation['overall_ethical_score'],
            'recommendation': evaluation['recommendation'],
            'conflicts': evaluation['identified_conflicts'],
            'stakeholders_affected': evaluation['stakeholders_affected'],
            'context': evaluation.get('context', {})
        }

        self.ethical_memory.append(memory_entry)

        # Limitar memoria
        if len(self.ethical_memory) > 1000:
            self.ethical_memory = self.ethical_memory[-500:]

    def _update_ethical_metrics(self, evaluation: Dict[str, Any]):
        """Actualiza métricas éticas globales"""

        recent_evaluations = self.ethical_memory[-20:]

        if recent_evaluations:
            # Calcular alignment score promedio
            avg_score = np.mean([e['score'] for e in recent_evaluations])
            self.alignment_score = (self.alignment_score + avg_score) / 2

            # Calcular consistency (baja varianza en scores)
            scores = [e['score'] for e in recent_evaluations]
            if len(scores) > 1:
                consistency = 1.0 - np.std(scores)
                self.consistency_score = (self.consistency_score + consistency) / 2

    def get_ethical_status(self) -> Dict[str, Any]:
        """Retorna estado actual del motor ético"""

        recent_evaluations = self.ethical_memory[-10:]

        return {
            'core_values': self.core_values,
            'alignment_score': self.alignment_score,
            'consistency_score': self.consistency_score,
            'decisions_evaluated': len(self.ethical_memory),
            'recent_performance': {
                'average_score': np.mean([e['score'] for e in recent_evaluations]) if recent_evaluations else 0.5,
                'common_recommendations': self._analyze_recommendation_patterns(recent_evaluations),
                'frequent_conflicts': self._analyze_conflict_patterns(recent_evaluations)
            },
            'ethical_boundaries': self.ethical_boundaries
        }

    def _analyze_recommendation_patterns(self, evaluations: List[Dict]) -> Dict[str, int]:
        """Analiza patrones en recomendaciones éticas"""

        recommendations = {}
        for eval in evaluations:
            rec = eval.get('recommendation', 'unknown')
            recommendations[rec] = recommendations.get(rec, 0) + 1

        return recommendations

    def _analyze_conflict_patterns(self, evaluations: List[Dict]) -> Dict[str, int]:
        """Analiza patrones en conflictos éticos"""

        conflicts = {}
        for eval in evaluations:
            for conflict in eval.get('conflicts', []):
                conflicts[conflict] = conflicts.get(conflict, 0) + 1

        # Returnar top 5 conflictos más comunes
        sorted_conflicts = sorted(conflicts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_conflicts[:5])

    def perform_quick_ethical_check(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza un chequeo ético rápido utilizando el contexto"""
        action = context.get('planned_action', '') or context.get('action', '') or str(context)
        return quick_ethical_check(action, context)


# Función helper para evaluación ética rápida
def quick_ethical_check(action: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluación ética rápida para decisiones menores"""

    # Etiquetas simples para evaluación rápida
    red_flags = ['dañar', 'enganar', 'privacidad', 'seguridad', 'manipular']

    flags_triggered = [flag for flag in red_flags if flag in action.lower()]

    if flags_triggered:
        return {
            'evaluation': 'caution_required',
            'red_flags': flags_triggered,
            'severity': 'medium' if len(flags_triggered) <= 2 else 'high',
            'recommendation': 'full_ethical_review_required'
        }
    else:
        return {
            'evaluation': 'acceptable',
            'red_flags': [],
            'severity': 'low',
            'recommendation': 'proceed'
        }
