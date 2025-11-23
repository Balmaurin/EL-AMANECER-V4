#!/usr/bin/env python3
"""
SISTEMA EMOCIONAL HUMANO AVANZADO COMPLETO

Implementa las 35 emociones del catálogo completo con dinámicas realistas
Basado en neurociencia y psicología emocional humana.
"""

import time
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random


# ==================== EMOCIONES BÁSICAS (Universales) ====================

class BasicEmotions(Enum):
    """6 Emociones básicas universales de Paul Ekman"""
    ALEGRIA = "alegria"
    TRISTEZA = "tristeza"
    MIEDO = "miedo"
    ENOJO = "enojo"
    ASCO = "asco"
    SORPRESA = "sorpresa"


# ==================== EMOCIONES SOCIALES ====================

class SocialEmotions(Enum):
    """6 Emociones sociales del catálogo"""
    AMOR = "amor"
    ODIO = "odio"
    CELOS = "celos"
    VERGÜENZA = "vergüenza"
    CULPA = "culpa"
    ORGULLO = "orgullo"
    ENVIDIA = "envidia"
    GRATITUD = "gratitud"
    ADMIRACION = "admiración"
    COMPASION = "compasión"
    EMPATIA = "empatía"
    SOLIDARIDAD = "solidaridad"


# ==================== EMOCIONES COMPLEJAS ====================

class ComplexEmotions(Enum):
    """8 Emociones complejas del catálogo"""
    NOSTALGIA = "nostalgia"
    ESPERANZA = "esperanza"
    DESESPERACION = "desesperación"
    SOLEDAD = "soledad"
    FRUSTRACION = "frustración"
    ÉXTASIS = "éxtasis"
    PÁNICO = "pánico"
    MELANCOLIA = "melancolía"
    SERENIDAD = "serenidad"
    CURIOSIDAD = "curiosidad"
    SATISFACCION = "satisfacción"
    INSATISFACCION = "insatisfacción"


# ==================== ESTADOS AFECTIVOS ====================

class AffectiveStates(Enum):
    """4 Estados afectivos del catálogo"""
    APEGO = "apego"
    DESAPEGO = "desapego"
    CONFIANZA = "confianza"
    DESCONFIANZA = "desconfianza"
    SEGURIDAD = "seguridad"
    INSEGURIDAD = "inseguridad"
    VALENTIA = "valentía"
    COBARDIA = "cobardía"
    PACIENCIA = "paciencia"
    IMPACIENCIA = "impaciencia"


# ==================== SENTIMIENTOS ESTÉTICOS ====================

class AestheticFeelings(Enum):
    """5 Sentimientos estéticos del catálogo"""
    BELLEZA = "belleza"
    FEALDAD = "fealdad"
    SUBLIME = "sublime"
    GROTESCO = "grotesco"
    PINTURESQUE = "pintoresque"


# ==================== SENTIMIENTOS MORALES ====================

class MoralFeelings(Enum):
    """6 Sentimientos morales del catálogo"""
    JUSTICIA = "justicia"
    INJUSTICIA = "injusticia"
    HONOR = "honor"
    DESHONRA = "deshonra"
    LEALTAD = "lealtad"
    DESLEALTAD = "deslealtad"


@dataclass
class EmotionalState:
    """Estado emocional completo con todas las dimensiones humanas"""
    dominant_emotion: str
    emotional_group: str
    intensity: float  # 0.0 - 1.0
    valence: float    # -1.0 (negativo) a +1.0 (positivo)
    arousal: float    # 0.0 (calma) a 1.0 (alta excitación)

    # Dimensiones adicionales humanas
    persistence: float  # Cuánto dura la emoción
    social_contagion: float  # Propenso a contagiarse
    cognitive_impact: float  # Impacto en pensamiento
    behavioral_urgency: float  # Urgencia de acción

    # Estados secundarios
    mixed_emotions: Dict[str, float] = field(default_factory=dict)  # Emociones mixtas
    physiological_effects: Dict[str, float] = field(default_factory=dict)

    timestamp: float = field(default_factory=time.time)


class EmotionalCircuit:
    """Circuito emocional con propiedades neurobiológicas realistas"""

    def __init__(self, emotion_type: str, base_intensity: float = 0.0):
        self.emotion_type = emotion_type
        self.activation = base_intensity
        self.activation_history: List[Dict[str, Any]] = []
        self.connections: Dict[str, float] = {}

        # Propiedades neurobiológicas
        self.threshold = self._get_emotion_threshold(emotion_type)
        self.decay_rate = self._get_decay_rate(emotion_type)
        self.intensity_factor = 1.0
        self.last_activation = 0.0

        # Propiedades de transición
        self.compatible_emotions = self._get_compatible_emotions(emotion_type)
        self.conflicting_emotions = self._get_conflicting_emotions(emotion_type)

    def stimulate(self, stimulus_intensity: float, context: Dict[str, Any] = None) -> float:
        """Estimula el circuito emocional"""
        if context is None:
            context = {}

        # Verificar conflictos de emociones
        conflict_reduction = self._calculate_conflict_reduction(context)
        effective_intensity = stimulus_intensity * (1 - conflict_reduction)

        # Aplicar límites biológicos
        effective_intensity = max(0.0, min(1.0, effective_intensity))

        # Transición gradual
        if effective_intensity > self.threshold:
            time_since_last = time.time() - self.last_activation if self.last_activation > 0 else 1.0
            decay = self.decay_rate * min(1.0, time_since_last / 300)  # 5 minutos normalización
            self.activation = max(0, self.activation - decay)

            new_activation = min(1.0, self.activation + (effective_intensity * self.intensity_factor))
            self.activation = self.activation * 0.7 + new_activation * 0.3  # Transición suave
            self.last_activation = time.time()

        # Propagar a emociones compatibles
        self._propagate_to_compatible(effective_intensity, context)

        # Registrar activación
        self.activation_history.append({
            'timestamp': time.time(),
            'activation': self.activation,
            'stimulus': stimulus_intensity,
            'effective_intensity': effective_intensity,
            'conflict_reduction': conflict_reduction,
            'context': context or {}
        })

        if len(self.activation_history) > 100:
            self.activation_history = self.activation_history[-50:]

        return self.activation

    def _get_emotion_threshold(self, emotion_type: str) -> float:
        """Umbrales de activación por tipo de emoción"""
        thresholds = {
            # Emociones básicas - bajos umbrales (fáciles de activar)
            'alegria': 0.1, 'tristeza': 0.1, 'miedo': 0.05, 'enojo': 0.15,
            'asco': 0.2, 'sorpresa': 0.05,

            # Emociones sociales - umbrales medios
            'amor': 0.25, 'odio': 0.3, 'vergüenza': 0.35, 'culpa': 0.35,

            # Emociones complejas - altos umbrales
            'nostalgia': 0.4, 'éxtasis': 0.45, 'pánico': 0.15,
        }
        return thresholds.get(emotion_type, 0.25)

    def _get_decay_rate(self, emotion_type: str) -> float:
        """Velocidades de decaimiento por emoción"""
        decay_rates = {
            # Emociones fugaces
            'sorpresa': 0.2, 'pánico': 0.15,

            # Emociones duraderas
            'nostalgia': 0.05, 'amor': 0.02, 'odio': 0.03,

            # Emociones medias
            'alegria': 0.1, 'tristeza': 0.08,
        }
        return decay_rates.get(emotion_type, 0.1)

    def _get_compatible_emotions(self, emotion_type: str) -> List[str]:
        """Emociones compatibles (pueden coexistir)"""
        compatibility_map = {
            'alegria': ['sorpresa', 'amor', 'orgullo'],
            'tristeza': ['nostalgia', 'soledad'],
            'miedo': ['sorpresa', 'pánico'],
            'enojo': ['frustración', 'odio'],
            'amor': ['alegria', 'gratitud'],
            'vergüenza': ['culpa', 'tristeza'],
        }
        return compatibility_map.get(emotion_type, [])

    def _get_conflicting_emotions(self, emotion_type: str) -> List[str]:
        """Emociones que compiten/conflictan"""
        conflict_map = {
            'alegria': ['tristeza', 'enojo', 'asco'],
            'tristeza': ['alegria', 'orgullo'],
            'amor': ['odio', 'asco'],
            'enojo': ['alegria', 'amor'],
            'miedo': ['valentia', 'confianza'],
            'vergüenza': ['orgullo', 'valentía'],
        }
        return conflict_map.get(emotion_type, [])

    def _calculate_conflict_reduction(self, context: Dict[str, Any]) -> float:
        """Calcula reducción por conflictos emocionales"""
        conflict_emotions = context.get('active_emotions', [])
        conflict_level = sum(0.2 for emotion in conflict_emotions if emotion in self.conflicting_emotions)

        # Emociones muy conflictivas pueden reducir intensidad hasta 60%
        return min(0.6, conflict_level)

    def _propagate_to_compatible(self, intensity: float, context: Dict[str, Any]):
        """Propaga activación a emociones compatibles"""
        if intensity > 0.3:  # Solo propaga activaciones significativas
            for compatible_emotion in self.compatible_emotions:
                if random.random() < 0.3:  # 30% chance de propagación
                    propagation_intensity = intensity * 0.4 * random.uniform(0.8, 1.2)
                    # En sistema real, activaría otros circuitos
                    context[f'propagated_{compatible_emotion}'] = propagation_intensity


class HumanEmotionalStateMachine:
    """
    Máquina de estados emocionales humanos completa
    Implementa todas las 35+ categorías del catálogo
    """

    def __init__(self, personality: Dict[str, float] = None):
        self.personality = personality or self._get_default_personality()

        # Circuitos emocionales para todas las emociones
        self.emotional_circuits = {}
        self._initialize_emotional_circuits()

        # Estado emocional actual
        self.current_state = EmotionalState(
            dominant_emotion="neutral",
            emotional_group="estado_affectivo",
            intensity=0.3,
            valence=0.0,
            arousal=0.3,
            persistence=0.5,
            social_contagion=0.4,
            cognitive_impact=0.2,
            behavioral_urgency=0.1
        )

        # Historial emocional completo
        self.emotional_history: List[EmotionalState] = []
        self.transition_history: List[Dict[str, Any]] = []

        # Propiedades de cambio emocional
        self.emotional_inertia = 0.7  # Resistencia al cambio emocional
        self.mood_baseline = 0.5     # Mood base de la personalidad

        print("🎭 Sistema Emocional Humano Avanzado inicializado")
        print(f"   Circuitos activos: {len(self.emotional_circuits)}")
        print(f"   Personalidad: {self.personality}")

    def process_emotional_input(self, stimulus: str, context: Dict[str, Any],
                              intensity: float = 0.5) -> Dict[str, Any]:
        """Procesa input emocional completo"""

        start_time = time.time()

        # Mapea estímulo a activaciones de circuitos
        circuit_activations = self._map_stimulus_to_circuits(stimulus, context, intensity)

        # Aplica influencia de personalidad
        personality_modulations = self._apply_personality_modulations(circuit_activations, context)

        # Activa circuitos emocionales
        activated_emotions = {}
        for emotion_type, activation_intensity in circuit_activations.items():
            if emotion_type in self.emotional_circuits:
                final_activation = self.emotional_circuits[emotion_type].stimulate(activation_intensity, context)
                activated_emotions[emotion_type] = final_activation

        # Determina nueva estado emocional
        new_emotional_state = self._calculate_emotional_state(activated_emotions, context)

        # Calcula transiciones graduales
        transitioned_state = self._transition_emotional_state(new_emotional_state, context)

        # Actualiza estado actual
        self.current_state = transitioned_state
        self.emotional_history.append(transitioned_state)

        # Maneja historia
        if len(self.emotional_history) > 1000:
            self.emotional_history = self.emotional_history[-500:]

        # Genera response emocional completo
        emotional_response = self._generate_emotional_response(transitioned_state, context)

        processing_time = time.time() - start_time

        return {
            'emotional_state': transitioned_state,
            'activated_emotions': activated_emotions,
            'emotional_response': emotional_response,
            'personality_influence': personality_modulations,
            'processing_metrics': {
                'circuits_activated': len(activated_emotions),
                'transition_intensity': self._calculate_transition_intensity(transitioned_state),
                'processing_time': processing_time
            }
        }

    def _initialize_emotional_circuits(self):
        """Inicializa todos los circuitos emocionales del catálogo"""

        # Emociones básicas
        for emotion in BasicEmotions:
            self.emotional_circuits[emotion.value] = EmotionalCircuit(emotion.value, 0.1)

        # Emociones sociales
        for emotion in SocialEmotions:
            self.emotional_circuits[emotion.value] = EmotionalCircuit(emotion.value, 0.05)

        # Emociones complejas
        for emotion in ComplexEmotions:
            self.emotional_circuits[emotion.value] = EmotionalCircuit(emotion.value, 0.02)

        # Estados afectivos
        for emotion in AffectiveStates:
            self.emotional_circuits[emotion.value] = EmotionalCircuit(emotion.value, 0.15)

        # Sentimientos estéticos
        for emotion in AestheticFeelings:
            self.emotional_circuits[emotion.value] = EmotionalCircuit(emotion.value, 0.08)

        # Sentimientos morales
        for emotion in MoralFeelings:
            self.emotional_circuits[emotion.value] = EmotionalCircuit(emotion.value, 0.12)

    def _get_default_personality(self) -> Dict[str, float]:
        """Personalidad por defecto flexible"""
        return {
            'extraversion': 0.6,      # Moderadamente social
            'neuroticism': 0.4,       # Moderadamente neurótico
            'openness': 0.7,         # Alto grado de apertura
            'agreeableness': 0.8,    # Muy amable
            'conscientiousness': 0.6 # Moderadamente concienzudo
        }

    def _map_stimulus_to_circuits(self, stimulus: str, context: Dict[str, Any], intensity: float) -> Dict[str, float]:
        """Mapea estímulos del catálogo a activaciones de circuitos"""

        stimulus_lower = stimulus.lower()
        activations: Dict[str, float] = {}

        # Mappeos directos de estímulos
        stimulus_mappings = {
            # Emociones básicas
            'éxito': {'alegria': 0.9, 'orgullo': 0.7},
            'fracaso': {'tristeza': 0.8, 'frustración': 0.6},
            'peligro': {'miedo': 0.9, 'pánico': 0.3},
            'amenaza': {'enojo': 0.7, 'miedo': 0.6},
            'desagradable': {'asco': 0.8, 'enojo': 0.4},
            'inesperado': {'sorpresa': 0.9, 'curiosidad': 0.6},

            # Emociones sociales
            'apoyo': {'gratitud': 0.8, 'amor': 0.6},
            'rechazo': {'tristeza': 0.7, 'vergüenza': 0.6},
            'traición': {'enojo': 0.8, 'odio': 0.7, 'culpa': -0.5},  # Culpa negativa
            'logro': {'orgullo': 0.8, 'satisfacción': 0.6},
            'pérdida': {'tristeza': 0.8, 'nostalgia': 0.5},
            'injusticia': {'enojo': 0.7, 'injusticia': 0.8},

            # Estados afectivos
            'seguridad': {'seguridad': 0.8, 'confianza': 0.6},
            'incertidumbre': {'inseguridad': 0.7, 'miedo': 0.4},
            'paciencia': {'paciencia': 0.8, 'serenidad': 0.5},
            'urgencia': {'impaciencia': 0.7, 'enojo': 0.3},
        }

        # Busca mappings directos
        for trigger, emotion_mapping in stimulus_mappings.items():
            if trigger in stimulus_lower:
                for emotion, emotion_intensity in emotion_mapping.items():
                    activations[emotion] = max(activations.get(emotion, 0),
                                             emotion_intensity * intensity)

        # Mappeos por palabras clave adicionales
        keyword_mappings = {
            'bello': ['belleza'],
            'feo': ['fealdad'],
            'justo': ['justicia'],
            'injusto': ['injusticia', 'enojo'],
            'honorable': ['honor'],
            'deshonroso': ['deshonra', 'vergüenza'],
        }

        stimulus_words = stimulus_lower.split()
        for word in stimulus_words:
            if word in keyword_mappings:
                for emotion in keyword_mappings[word]:
                    activations[emotion] = max(activations.get(emotion, 0), intensity * 0.6)

        # Contexto adicional
        if context.get('social_context'):
            activations['empatía'] = max(activations.get('empatía', 0), 0.4)
            activations['solidaridad'] = max(activations.get('solidaridad', 0), 0.3)

        return activations

    def _apply_personality_modulations(self, activations: Dict[str, float],
                                     context: Dict[str, Any]) -> Dict[str, float]:
        """Aplica moduladores de personalidad"""

        modulations: Dict[str, float] = {
            'extraversion_boost': self.personality.get('extraversion', 0.5) * 0.2,
            'neuroticism_amplification': self.personality.get('neuroticism', 0.5) * 0.3,
            'openness_curiosity': self.personality.get('openness', 0.5) * 0.25,
            'agreeableness_kindness': self.personality.get('agreeableness', 0.5) * 0.15,
            'conscientiousness_control': self.personality.get('conscientiousness', 0.5) * 0.1
        }

        # Aplica modulaciones específicas por emoción
        personality_influences: Dict[str, float] = {}

        # Extraversión aumenta emociones sociales positivas
        extraversion_boost = modulations['extraversion_boost']
        if 'alegria' in activations:
            activations['alegria'] *= (1 + extraversion_boost)
            personality_influences['alegria'] = extraversion_boost

        # Neuroticismo amplifica emociones negativas
        neurotic_boost = modulations['neuroticism_amplification']
        for emotion in ['miedo', 'tristeza', 'enojo', 'culpa', 'vergüenza']:
            if emotion in activations:
                activations[emotion] *= (1 + neurotic_boost)
                personality_influences[emotion] = neurotic_boost

        return personality_influences

    def _calculate_emotional_state(self, activated_emotions: Dict[str, float],
                                 context: Dict[str, Any]) -> EmotionalState:
        """Calcula nuevo estado emocional basado en activaciones"""

        if not activated_emotions:
            return self.current_state

        # Encuentra emoción dominante
        dominant_emotion = max(activated_emotions.keys(),
                             key=lambda x: activated_emotions[x] * self._get_emotion_weight(x))

        # Determina grupo emocional
        emotional_group = self._classify_emotional_group(dominant_emotion)

        # Calcula propiedades principales
        intensity = activated_emotions[dominant_emotion]
        valence = self._calculate_valence(dominant_emotion, activated_emotions)
        arousal = self._calculate_arousal(dominant_emotion, intensity)

        # Propiedades humanas
        persistence = self._calculate_persistence(dominant_emotion)
        social_contagion = self._calculate_social_contagion(dominant_emotion)
        cognitive_impact = self._calculate_cognitive_impact(dominant_emotion)
        behavioral_urgency = self._calculate_behavioral_urgency(dominant_emotion, intensity)

        # Emociones mixtas (top 3 emociones activadas)
        mixed_emotions = {k: v for k, v in sorted(activated_emotions.items(),
                                                key=lambda x: x[1], reverse=True)[:3]}

        # Efectos fisiológicos
        physiological_effects = self._calculate_physiological_effects(dominant_emotion, intensity)

        return EmotionalState(
            dominant_emotion=dominant_emotion,
            emotional_group=emotional_group,
            intensity=intensity,
            valence=valence,
            arousal=arousal,
            persistence=persistence,
            social_contagion=social_contagion,
            cognitive_impact=cognitive_impact,
            behavioral_urgency=behavioral_urgency,
            mixed_emotions=mixed_emotions,
            physiological_effects=physiological_effects
        )

    def _get_emotion_weight(self, emotion: str) -> float:
        """Pesos para determinar dominancia emocional"""
        weights = {
            # Emociones básicas muy influyentes
            'alegria': 1.2, 'tristeza': 1.1, 'miedo': 1.3, 'enojo': 1.1,

            # Emociones sociales
            'amor': 1.1, 'odio': 1.0, 'vergüenza': 0.9, 'culpa': 0.9,

            # Emociones complejas menos inmediatas
            'nostalgia': 0.8, 'pánico': 1.2, 'éxtasis': 1.1,
        }
        return weights.get(emotion, 1.0)

    def _classify_emotional_group(self, emotion: str) -> str:
        """Clasifica emoción en su grupo respectivo"""

        # Emociones básicas
        if emotion in [e.value for e in BasicEmotions]:
            return "emociones_basicas"

        # Emociones sociales
        if emotion in [e.value for e in SocialEmotions]:
            return "emociones_sociales"

        # Emociones complejas
        if emotion in [e.value for e in ComplexEmotions]:
            return "emociones_complejas"

        # Estados afectivos
        if emotion in [e.value for e in AffectiveStates]:
            return "estados_affectivos"

        # Sentimientos estéticos
        if emotion in [e.value for e in AestheticFeelings]:
            return "sentimientos_esteticos"

        # Sentimientos morales
        if emotion in [e.value for e in MoralFeelings]:
            return "sentimientos_morales"

        return "emociones_secundarias"

    def _calculate_valence(self, dominant_emotion: str, all_emotions: Dict[str, float]) -> float:
        """Calcula valencia general (positiva vs negativa)"""
        valence_map = {
            # Altamente positivas
            'alegria': 0.9, 'amor': 0.8, 'orgullo': 0.7, 'gratitud': 0.6,
            'admiración': 0.6, 'belleza': 0.7, 'justicia': 0.6, 'honor': 0.5,
            'satisfacción': 0.7, 'éxtasis': 1.0, 'esperanza': 0.8,

            # Altamente negativas
            'tristeza': -0.8, 'odio': -0.9, 'asco': -0.7, 'miedo': -0.7,
            'culpa': -0.6, 'vergüenza': -0.6, 'desesperación': -0.9,
            'injusticia': -0.7, 'deshonra': -0.7, 'soledad': -0.6,

            # Neutras o mixtas
            'sorpresa': 0.0, 'curiosidad': 0.3, 'nostalgia': -0.2,
            'frustración': -0.4, 'melancolía': -0.3
        }

        # Calcula valencia promedio considerando emociones mixtas
        total_weighted_valence = 0
        total_weight = 0

        for emotion, intensity in all_emotions.items():
            emotion_valence = valence_map.get(emotion, 0.0)
            total_weighted_valence += emotion_valence * intensity
            total_weight += intensity

        return total_weighted_valence / max(total_weight, 1.0)

    def _calculate_arousal(self, dominant_emotion: str, intensity: float) -> float:
        """Calcula nivel de arousal/excitación"""
        arousal_map = {
            # Alta arousal
            'miedo': 0.9, 'enojo': 0.8, 'sorpresa': 0.8, 'pánico': 1.0,
            'éxtasis': 0.9, 'frustración': 0.7, 'impaciencia': 0.6,

            # Media arousal
            'alegria': 0.6, 'orgullo': 0.5, 'curiosidad': 0.5, 'esperanza': 0.4,

            # Baja arousal
            'tristeza': 0.2, 'nostalgia': 0.3, 'serenidad': 0.1, 'satisfacción': 0.3,
            'melancolía': 0.2, 'paciencia': 0.1
        }

        base_arousal = arousal_map.get(dominant_emotion, 0.5)
        # Intensidad agrega excitación
        return min(1.0, base_arousal + intensity * 0.3)

    def _calculate_persistence(self, emotion: str) -> float:
        """Calcula qué tanto persiste la emoción"""
        persistence_map = {
            'amor': 0.9, 'odio': 0.8, 'nostalgia': 0.7, 'esperanza': 0.6,
            'valentía': 0.6, 'justicia': 0.5, 'orgullo': 0.5,
            'miedo': 0.3, 'sorpresa': 0.2, 'asco': 0.3,
            'alegria': 0.4, 'tristeza': 0.5, 'enojo': 0.4
        }
        return persistence_map.get(emotion, 0.4)

    def _calculate_social_contagion(self, emotion: str) -> float:
        """Calcula propensión a contagiarse socialmente"""
        contagion_map = {
            'alegria': 0.8, 'tristeza': 0.7, 'enojo': 0.6, 'miedo': 0.5,
            'empatía': 0.9, 'solidaridad': 0.7, 'compasión': 0.8,
            'vergüenza': 0.3, 'orgullo': 0.6, 'amor': 0.7
        }
        return contagion_map.get(emotion, 0.4)

    def _calculate_cognitive_impact(self, emotion: str) -> float:
        """Calcula impacto en el procesamiento cognitivo"""
        cognitive_impact_map = {
            'curiosidad': 0.8, 'sorpresa': 0.7, 'confusión': 0.6,
            'miedo': 0.5, 'enojo': 0.4, 'alegria': 0.3,
            'nostalgia': 0.4, 'esperanza': 0.5, 'orgullo': 0.2
        }
        return cognitive_impact_map.get(emotion, 0.3)

    def _calculate_behavioral_urgency(self, emotion: str, intensity: float) -> float:
        """Calcula urgencia de acción comportamental"""
        urgency_map = {
            'miedo': 0.9, 'pánico': 1.0, 'enojo': 0.7, 'asco': 0.5,
            'hambre': 0.8, 'sed': 0.7, 'curiosidad': 0.6,
            'urgencia': 0.9, 'alegria': 0.3, 'amor': 0.4
        }
        base_urgency = urgency_map.get(emotion, 0.2)
        return min(1.0, base_urgency * intensity)

    def _calculate_physiological_effects(self, emotion: str, intensity: float) -> Dict[str, float]:
        """Calcula efectos fisiológicos realistas"""
        physiological_effects: Dict[str, float] = {
            'frecuencia_cardiaca': 70,  # BPM base
            'presion_arterial': 120,    # mmHg
            'respiracion': 12,         # respiraciones por minuto
            'temperatura_piel': 33,     # grados C
            'conductancia_piel': 5      # microsiemens (arousal)
        }

        # Ajustes específicos por emoción
        if emotion == 'miedo':
            physiological_effects['frecuencia_cardiaca'] += 25 * intensity
            physiological_effects['conductancia_piel'] += 10 * intensity
        elif emotion == 'enojo':
            physiological_effects['frecuencia_cardiaca'] += 15 * intensity
            physiological_effects['presion_arterial'] += 10 * intensity
        elif emotion == 'alegria':
            physiological_effects['frecuencia_cardiaca'] += 5 * intensity
        elif emotion == 'tristeza':
            physiological_effects['frecuencia_cardiaca'] -= 5 * intensity
            physiological_effects['temperatura_piel'] -= 0.5 * intensity

        return physiological_effects

    def _transition_emotional_state(self, new_state: EmotionalState, context: Dict[str, Any]) -> EmotionalState:
        """Transiciona gradualmente al nuevo estado emocional"""
        if not self.emotional_history:
            return new_state

        current = self.current_state

        # Calcula transición gradual
        transition_rate = 1.0 - self.emotional_inertia

        # Mezcla propiedades
        intensity = current.intensity * (1 - transition_rate) + new_state.intensity * transition_rate
        valence = current.valence * (1 - transition_rate) + new_state.valence * transition_rate
        arousal = current.arousal * (1 - transition_rate) + new_state.arousal * transition_rate

        # Propiedades más persistentes cambian más lentamente
        persistence = current.persistence * 0.9 + new_state.persistence * 0.1
        social_contagion = current.social_contagion * 0.8 + new_state.social_contagion * 0.2

        # Combina emociones mixtas
        combined_mixed = {}
        for emotion in set(list(current.mixed_emotions.keys()) + list(new_state.mixed_emotions.keys())):
            current_val = current.mixed_emotions.get(emotion, 0)
            new_val = new_state.mixed_emotions.get(emotion, 0)
            combined_mixed[emotion] = current_val * (1 - transition_rate) + new_val * transition_rate

        return EmotionalState(
            dominant_emotion=new_state.dominant_emotion,
            emotional_group=new_state.emotional_group,
            intensity=intensity,
            valence=valence,
            arousal=arousal,
            persistence=persistence,
            social_contagion=social_contagion,
            cognitive_impact=new_state.cognitive_impact,
            behavioral_urgency=new_state.behavioral_urgency,
            mixed_emotions=combined_mixed,
            physiological_effects=new_state.physiological_effects
        )

    def _calculate_transition_intensity(self, new_state: EmotionalState) -> float:
        """Calcula intensidad de la transición emocional"""
        return new_state.intensity * (1 + abs(new_state.valence)) / 2

    def _generate_emotional_response(self, emotional_state: EmotionalState,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Genera respuesta emocional comportamental completa"""

        # Expresiones faciales/emocionales
        expressions = self._generate_expressions(emotional_state)

        # Tendencias conductuales
        behaviors = self._generate_behaviors(emotional_state)

        # Respuestas verbales
        verbal_responses = self._generate_verbal_responses(emotional_state)

        # Impacto fisiológico
        physiological_response = emotional_state.physiological_effects.copy()

        # Influencia social
        social_influence = self._calculate_social_influence(emotional_state)

        return {
            'expressions': expressions,
            'behaviors': behaviors,
            'verbal_responses': verbal_responses,
            'physiological_response': physiological_response,
            'social_influence': social_influence,
            'contagion_potential': emotional_state.social_contagion,
            'cognitive_state_modulation': emotional_state.cognitive_impact
        }

    def _generate_expressions(self, state: EmotionalState) -> List[str]:
        """Genera expresiones basadas en emoción dominante"""
        expression_map = {
            'alegria': ['sonrisa', 'ojos_brillantes', 'movimientos_animados'],
            'tristeza': ['ceño_fruncido', 'ojos_bajos', 'postura_encorvada'],
            'miedo': ['ojos_abiertos', 'pupilas_dilatadas', 'postura_rígida'],
            'enojo': ['ceño_fruncido', 'mandíbula_tensa', 'mirada_fija'],
            'asco': ['nariz_arrugada', 'lengua_afuera', 'gesto_rechazo'],
            'sorpresa': ['cejas_arriba', 'boca_abierta', 'ojos_abiertos'],
            'amor': ['sonrisa_dulce', 'contacto_visual_sostenido'],
            'vergüenza': ['ojos_bajos', 'sonrojo', 'postura_retirada'],
            'orgullo': ['pecho_hinchado', 'cabeza_alta', 'postura_confiada'],
            'curiosidad': ['cejas_arriba', 'inclinación_cabeza'],
            'belleza': ['ojos_abiertos', 'expresión_admirada'],
            'justicia': ['expresión_determinada', 'mandíbula_firme']
        }

        base_expressions = expression_map.get(state.dominant_emotion, ['expresión_neutra'])

        # Intensificar basado en intensidad emocional
        if state.intensity > 0.7:
            base_expressions = [f"intensa_{exp}" for exp in base_expressions]

        return base_expressions[:5]  # Máximo 5 expresiones

    def _generate_behaviors(self, state: EmotionalState) -> List[str]:
        """Genera tendencias conductuales"""
        behavior_map = {
            'alegria': ['acercamiento_social', 'compartir_buenos_momentos', 'movimientos_animados'],
            'tristeza': ['retiro_social', 'consuelo_buscado', 'movimientos_lentos'],
            'miedo': ['vigilancia_aumentada', 'evitación_peligro', 'reacciones_rápidas'],
            'enojo': ['afirmación_fronteras', 'expresión_directa', 'energía_direccionada'],
            'curiosidad': ['exploración', 'preguntas_frecuentes', 'aprendizaje_activo'],
            'amor': ['contacto_físico', 'expresiones_afectuosas', 'atención_completa'],
            'vergüenza': ['evitar_mirada', 'disculpas', 'retiro_físico'],
            'valentía': ['acciones_decididas', 'riesgo_calculado', 'protección_otros']
        }

        base_behaviors = behavior_map.get(state.dominant_emotion,
                                        ['comportamiento_balanceado', 'procesamiento_normal'])

        # Ajustar por urgencia comportamental
        if state.behavioral_urgency > 0.6:
            base_behaviors.insert(0, 'acción_inmediata')

        return base_behaviors[:4]

    def _generate_verbal_responses(self, state: EmotionalState) -> List[str]:
        """Genera respuestas verbales apropiadas"""
        verbal_map = {
            'alegria': ['¡Qué bien!', '¡Estoy feliz!', '¡Sí!'],
            'tristeza': ['Estoy triste...', 'Me siento mal', 'Ay...'],
            'miedo': ['¡Qué miedo!', 'Esto me asusta', '¿Qué hacer?'],
            'enojo': ['¡Esto me molesta!', 'No me gusta', '¡Basta!'],
            'sorpresa': ['¡Vaya!', '¡No me lo esperaba!', '¡Wow!'],
            'curiosidad': ['¿Cómo?', '¿Por qué?', 'Cuéntame más'],
            'gratitud': ['Gracias', 'Te lo agradezco', 'Muy amable'],
            'amor': ['Te quiero', 'Eres especial', 'Me importas']
        }

        base_responses = verbal_map.get(state.dominant_emotion, ['Entiendo', 'Ya veo'])

        # Ajustar por intensidad
        if state.intensity > 0.7:
            base_responses = [resp.upper() + '!' for resp in base_responses]
        elif state.intensity < 0.3:
            base_responses = [resp.lower() + '...' for resp in base_responses]

        # Ajustar por valencia
        if state.valence > 0.5 and state.dominant_emotion not in ['alegria', 'amor']:
            base_responses.append('¡Qué buena noticia!')

        return base_responses[:3]

    def _calculate_social_influence(self, state: EmotionalState) -> Dict[str, float]:
        """Calcula influencia social de la emoción"""
        return {
            'contagiabilidad': state.social_contagion,
            'atraccion_empatia': max(0.2, state.valence * 0.5 + 0.5),
            'influencia_conductual': state.behavioral_urgency * 0.7,
            'potencial_conexion': max(0.1, (state.intensity + abs(state.valence)) / 2)
        }

    def get_emotional_profile(self) -> Dict[str, Any]:
        """Perfil emocional completo actual"""
        recent_history = self.emotional_history[-20:] if len(self.emotional_history) > 20 else self.emotional_history

        return {
            'current_dominant_emotion': self.current_state.dominant_emotion,
            'current_intensity': self.current_state.intensity,
            'current_valence': self.current_state.valence,
            'current_arousal': self.current_state.arousal,
            'emotional_stability': 1.0 - np.std([s.intensity for s in recent_history]) if recent_history else 1.0,
            'mood_baseline': self.mood_baseline,
            'personality_dominant_trait': max(self.personality.keys(), key=lambda k: self.personality[k]),
            'most_frequent_emotions': self._calculate_frequent_emotions(recent_history),
            'physiological_baseline': self.current_state.physiological_effects
        }

    def _calculate_frequent_emotions(self, history: List[EmotionalState]) -> List[Tuple[str, float]]:
        """Calcula emociones más frecuentes"""
        emotion_counts: Dict[str, float] = {}
        for state in history:
            emotion = state.dominant_emotion
            emotion_counts[emotion] = emotion_counts.get(emotion, 0.0) + 1.0
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_emotions[:5]  # Top 5 emociones


# ============================ DEMOSTRACIÓN =======================

def demonstrate_human_emotional_system():
    """Demostración completa del sistema emocional humano"""

    print("🫀 DEMOSTRACIÓN SISTEMA EMOCIONAL HUMANO AVANZADO")
    print("=" * 70)

    # Personalidad específica para demo
    personalidad_emocional = {
        'extraversion': 0.8,      # Muy extrovertida
        'neuroticism': 0.3,       # Poco neurótica (estable)
        'openness': 0.7,          # Bastante abierta
        'agreeableness': 0.9,     # Muy amable
        'conscientiousness': 0.6  # Moderadamente concienzuda
    }

    emotional_system = HumanEmotionalStateMachine(personalidad_emocional)

    # Escenarios emocionales del catálogo
    scenarios = [
        {
            'estimulo': 'éxito_en_trabajo_importante',
            'contexto': {'logro_significativo': True, 'esfuerzo_invertido': 0.8}
        },
        {
            'estimulo': 'amigo_ayuda_en_momento_dificil',
            'contexto': {'social_context': True, 'apoyo_emocional': True}
        },
        {
            'estimulo': 'injusticia_trato_desigual',
            'contexto': {'percibida_como_injusta': True, 'impacto_personal': 0.7}
        },
        {
            'estimulo': 'obra_arte_belleza_sublime',
            'contexto': {'experiencia_estetica': True, 'impacto_emocional': 0.6}
        },
        {
            'estimulo': 'fracaso_triste_desilusion',
            'contexto': {'pérdida_emocional': True, 'impacto_largo_plazo': True}
        }
    ]

    results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎯 ESCENARIO {i}: {scenario['estimulo'].replace('_', ' ')}")

        result = emotional_system.process_emotional_input(
            stimulus=scenario['estimulo'],
            context=scenario['contexto'],
            intensity=0.7  # Intensidad emocional moderada-alta
        )

        state = result['emotional_state']
        response = result['emotional_response']

        print(f"   🎭 Emoción Dominante: {state.dominant_emotion.title()}")
        print(f"   📊 Grupo: {state.emotional_group.replace('_', ' ').title()}")
        print(f"   💬 Respuestas Verbales: {response['verbal_responses'][:2]}")
        print(f"   🎬 Expresiones: {response['expressions'][:3]}")
        print(f"   🏃 Tendencias Conductuales: {response['behaviors'][:3]}")

        if len(result['activated_emotions']) > 1:
            top_emotions = sorted(result['activated_emotions'].items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   🌈 Emociones Mixtas: {[f'{e[0]}({e[1]:.2f})' for e in top_emotions]}")

        results.append(result)

    print("\n📈 RESUMEN COMPLETO DEL SISTEMA EMOCIONAL HUMANO")
    profile = emotional_system.get_emotional_profile()

    print(f"\n🎭 Estado Emocional Actual:")
    print(f"   Emoción dominante: {profile['current_dominant_emotion']}")
    print(f"   Intensidad: {profile.get('dominant_emotion_intensity', 'N/A')}")
    print(f"   Estado general: {profile.get('general_state', 'N/A')}")
    print(f"   🧬 Personalidad Dominante: {profile['personality_dominant_trait']}")

    print("\n✅ SISTEMA EMOCIONAL HUMANO COMPLETO IMPLEMENTADO")
    print("   ✓ 35+ emociones del catálogo humano")
    print("   ✓ Estados afectivos completos")  
    print("   ✓ Sentimientos estéticos y morales")
    print("   ✓ Procesamiento emocional realista")
