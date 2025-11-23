"""
Sistema Neuronal Emocional: Procesamiento emocional basado en circuitos neuronales reales

Implementa circuitos emocionales análogos a los sistemas límbico y dopaminérgico humanos,
procesando emociones como respuestas adaptativas a estímulos. Sistema completamente funcional
que reconoce y genera respuestas emocionales apropiadas.
"""

import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import random
import json


@dataclass
class NeuralActivation:
    """Representa la activación de un circuito neuronal específico"""
    circuit_type: str  # 'recompensa', 'alerta', 'social', etc.
    intensity: float   # 0.0 a 1.0
    duration: float    # duración en segundos
    trigger: str       # qué lo activó
    timestamp: float   # tiempo de activación


@dataclass
class EmotionalState:
    """Estado emocional completo en un momento dado"""
    dominant_emotion: str
    intensity: float
    valence: float      # positivo(1) a negativo(-1)
    arousal: float      # bajo(0) a alto(1)
    circuit_activations: Dict[str, float]
    physiological_effects: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


class EmotionalCircuit:
    """
    Circuito emocional base con propiedades neurobiológicas realistas

    Modela propiedades de circuitos neuronales:
    - Umbral de activación
    - Decaimiento temporal
    - Propagación a circuitos conectados
    - Sensibilidad al contexto
    """

    def __init__(self, name: str, base_activation: float = 0.0):
        self.name = name
        self.activation = base_activation
        self.activation_history: List[Dict[str, Any]] = []
        self.connections: Dict['EmotionalCircuit', float] = {}
        self.threshold = 0.3  # Umbral mínimo para activarse
        self.decay_rate = 0.1  # Velocidad de decaimiento natural
        self.activation_sensitivity = 1.0  # Sensibilidad a estímulos
        self.last_activation = 0.0

        print(f"⚡ Circuito {name} inicializado - activación base: {base_activation}")

    def stimulate(self, stimulus_intensity: float, context: Dict = None) -> float:
        """
        Estimula el circuito con una intensidad específica

        Retorna la activación final después del procesamiento
        """
        if context is None:
            context = {}

        # Calcular intensidad efectiva considerando contexto
        effective_intensity = self._calculate_effective_intensity(stimulus_intensity, context)

        # Aplicar decaimiento temporal
        time_since_last = time.time() - self.last_activation if self.last_activation > 0 else 1.0
        decay = self.decay_rate * min(1.0, time_since_last / 60)  # Normalizar decay por minuto
        self.activation = max(0, self.activation - decay)
        self.last_activation = time.time()

        # Aplicar estímulo si supera umbral
        if effective_intensity > self.threshold:
            # Actualizar activación con decaimiento suave
            new_activation = min(1.0, self.activation + (effective_intensity * self.activation_sensitivity))

            # Transición gradual para evitar cambios bruscos
            transition_rate = 0.4  # 40% transición inmediata
            self.activation = (self.activation * (1 - transition_rate) + new_activation * transition_rate)

            # Propagar a circuitos conectados
            self._propagate_activation(effective_intensity, context)

        # Registrar activación
        self.activation_history.append({
            'timestamp': time.time(),
            'activation': self.activation,
            'stimulus': stimulus_intensity,
            'effective_intensity': effective_intensity,
            'context': context or {}
        })

        return self.activation

    def _calculate_effective_intensity(self, stimulus: float, context: Dict) -> float:
        """Calcula intensidad efectiva considerando contexto neurobiológico"""

        # Factores moduladores basados en neurociencia
        novelty_mod = context.get('novelty', 0.5) * 1.3      # Novelties amplifican señales
        relevance_mod = context.get('relevance', 0.5) * 1.2  # Relevancia aumenta sensibilidad
        attention_mod = context.get('attention', 0.7)        # Atención filtra señales
        stress_mod = context.get('stress', 0.3) * 1.5        # Estrés amplifica respuestas

        # Efectos de personalidad (simulados)
        baseline_sensitivity = context.get('trait_anxiety', 0.5)
        learned_sensitization = context.get('sensitization_level', 1.0)

        # Combinar factores
        modulation_factors = [novelty_mod, relevance_mod, attention_mod, stress_mod,
                             baseline_sensitivity, learned_sensitization]

        overall_modulation = np.mean(modulation_factors)

        # Aplicar modulación no-lineal (similar a DDT en neuromoduladores)
        effective_intensity = stimulus * overall_modulation

        # Función sigmoide para saturación realista
        effective_intensity = 1 / (1 + np.exp(-(effective_intensity - 1) * 2))

        return max(0.0, min(1.0, effective_intensity))

    def _propagate_activation(self, original_intensity: float, context: Dict):
        """Propaga activación a circuitos conectados con atenuación realista"""

        for connected_circuit, connection_strength in self.connections.items():
            if random.random() < connection_strength:  # Probabilidad de propagación
                propagated_intensity = original_intensity * connection_strength * 0.7

                # Añadir ruido neural para variabilidad biológica
                neural_noise = np.random.normal(0, 0.05)
                propagated_intensity += neural_noise

                # Propagar con retraso neural mínimo
                time.sleep(random.uniform(0.001, 0.005))  # 1-5ms retraso sináptico

                connected_circuit.stimulate(propagated_intensity, context)

    def get_activation_state(self) -> Dict[str, Any]:
        """Retorna estado completo del circuito"""
        return {
            'name': self.name,
            'activation': self.activation,
            'threshold': self.threshold,
            'connections': len(self.connections),
            'history_length': len(self.activation_history),
            'sensitivity': self.activation_sensitivity,
            'last_activation': self.last_activation
        }


class DopamineSystem:
    """
    Sistema de recompensa dopaminérgico

    Modela el sistema dopaminérgico mesolímbico que procesa:
    - Recompensas esperadas vs reales
    - Error de predicción de recompensa
    - Motivación y aprendizaje por refuerzo
    """

    def __init__(self, baseline_dopamine: float = 0.5):
        self.dopamine_level = baseline_dopamine
        self.prediction_error = 0.0
        self.expected_rewards: Dict[str, float] = {}
        self.reward_history: List[Dict[str, Any]] = []

        # Parámetros neurobiológicos
        self.baseline_level = baseline_dopamine
        self.release_rate = 0.8      # Velocidad de liberación de dopamina
        self.decay_rate = 0.1        # Velocidad de recaptación/reuptake
        self.sensitivity = 1.2       # Sensibilidad al reward error

        # Circuitos conectados
        self.reward_circuit = EmotionalCircuit("recompensa_primaria", 0.2)
        self.motivation_circuit = EmotionalCircuit("motivación_comportamental", 0.3)

        # Conectar circuitos para propagación natural
        self.reward_circuit.connections[self.motivation_circuit] = 0.8

        print("🧠 Sistema dopaminérgico inicializado")

    def process_reward(self, actual_reward: float, context: Dict = None,
                      expected_reward: Optional[float] = None) -> Dict[str, Any]:
        """
        Procesa un evento de recompensa y calcula respuesta dopaminérgica

        Imita el sistema de recompensa ventral tegmental → núcleo accumbens
        """
        if context is None:
            context = {}

        # Generar predicción si no se proveyó
        if expected_reward is None:
            context_key = context.get('situation_hash', str(hash(str(sorted(context.items())))))
            expected_reward = self.expected_rewards.get(context_key, self.baseline_level)

        # Calcular error de predicción (core del sistema dopaminérgico)
        self.prediction_error = actual_reward - expected_reward

        # Calcular liberación dopaminérgica
        if self.prediction_error > 0:
            # Recompensa inesperada - pico dopaminérgico fuerte
            dopamine_release = expected_reward + (self.prediction_error * self.release_rate * 2)
        else:
            # Recompensa menor a esperada - disminución dopaminérgica
            dopamine_release = max(0.1, expected_reward + (self.prediction_error * self.release_rate))

        # Aplicar límites biológicos realistas (0.1 - 0.95)
        dopamine_release = max(0.1, min(0.95, dopamine_release))

        # Actualizar nivel dopaminérgico con dinámica temporal
        dopamine_change = (dopamine_release - self.dopamine_level) * 0.7  # 70% cambio inmediato
        self.dopamine_level += dopamine_change

        # Decaimiento gradual
        self.dopamine_level = (self.dopamine_level * (1 - self.decay_rate) +
                              self.baseline_level * self.decay_rate)

        # Activar circuitos de recompensa
        reward_activation = self.reward_circuit.stimulate(dopamine_release, context)

        # Generar señal motivacional
        motivation_signal = self.prediction_error * dopamine_release
        motivation_activation = self.motivation_circuit.stimulate(abs(motivation_signal), context)

        # Aprender para futuras predicciones (similar a plasticidad dopaminérgica)
        self._update_reward_predictions(actual_reward, context)

        # Registrar evento de recompensa
        reward_event = {
            'timestamp': time.time(),
            'actual_reward': actual_reward,
            'expected_reward': expected_reward,
            'prediction_error': self.prediction_error,
            'dopamine_level': self.dopamine_level,
            'context': context,
            'reward_activation': reward_activation,
            'motivation_activation': motivation_activation
        }
        self.reward_history.append(reward_event)

        return {
            'dopamine_level': self.dopamine_level,
            'prediction_error': self.prediction_error,
            'reward_activation': reward_activation,
            'motivation_activation': motivation_activation,
            'reward_type': 'positive' if self.prediction_error > 0 else 'negative'
        }

    def _update_reward_predictions(self, actual_reward: float, context: Dict):
        """Actualiza predicciones de recompensa basado en experiencia actual"""
        context_key = context.get('situation_hash', str(hash(str(sorted(context.items())))))

        current_expectation = self.expected_rewards.get(context_key, self.baseline_level)
        learning_rate = 0.2  # Velocidad de aprendizaje dopaminérgica

        # Actualizar expectativa usando delta rule
        prediction_error = actual_reward - current_expectation
        new_expectation = current_expectation + (learning_rate * prediction_error)

        # Limitar expectativas biológicas razonables
        self.expected_rewards[context_key] = max(0.0, min(1.0, new_expectation))

    def get_dopamine_state(self) -> Dict[str, Any]:
        """Estado completo del sistema dopaminérgico"""
        return {
            'current_level': self.dopamine_level,
            'baseline_level': self.baseline_level,
            'prediction_error': self.prediction_error,
            'learned_expectations': len(self.expected_rewards),
            'reward_events_count': len(self.reward_history),
            'sensitivity': self.sensitivity,
            'circuit_states': {
                'reward': self.reward_circuit.get_activation_state(),
                'motivation': self.motivation_circuit.get_activation_state()
            }
        }


class EmotionalStateMachine:
    """
    Máquina de estados emocionales con transiciones neurobiológicamente realistas

    Implementa modelo de estados emocionales discretos con transiciones
    basadas en activación de circuitos subyacentes.
    """

    def __init__(self):
        self.current_state = "neutral"
        self.emotional_intensity = 0.5
        self.state_history: List[Dict[str, Any]] = []

        # Definir estados emocionales y propiedades neurobiológicas
        self.emotional_states = {
            "alegría": {
                "valence": 0.8, "arousal": 0.7,
                "circuit_weights": {"placer": 0.8, "dolor": -0.5},
                "expression": "risa_facial_muscular", "transition_duration": 1.5,
                "typical_duration": 30,  # segundos hasta decay natural
                "physiological": {"heart_rate": 1.2, "muscle_tension": -0.3}
            },
            "tristeza": {
                "valence": -0.7, "arousal": -0.4,
                "circuit_weights": {"dolor": 0.9, "placer": -0.6},
                "expression": "postura_encorvada_lagrimeo", "transition_duration": 3.0,
                "typical_duration": 120,
                "physiological": {"heart_rate": -0.8, "muscle_tension": -0.4}
            },
            "enojo": {
                "valence": -0.6, "arousal": 0.9,
                "circuit_weights": {"frustración": 0.9, "miedo": 0.3},
                "expression": "ceño_fruncido_voz_alta", "transition_duration": 1.0,
                "typical_duration": 45,
                "physiological": {"heart_rate": 1.5, "muscle_tension": 0.8}
            },
            "miedo": {
                "valence": -0.8, "arousal": 0.8,
                "circuit_weights": {"miedo": 0.8, "alerta": 0.6},
                "expression": "ojos_abiertos_respiración_rápida", "transition_duration": 0.5,
                "typical_duration": 20,
                "physiological": {"heart_rate": 1.4, "respiration": 1.5, "stress_hormones": 1.6}
            },
            "asco": {
                "valence": -0.5, "arousal": 0.3,
                "circuit_weights": {"rechazo": 0.8, "dolor": 0.2},
                "expression": "arruga_nariz_expulsión", "transition_duration": 1.0,
                "typical_duration": 15,
                "physiological": {"stomach_activity": -0.6, "salivation": -0.7}
            },
            "sorpresa": {
                "valence": 0.0, "arousal": 0.9,
                "circuit_weights": {"curiosidad": 0.7, "alerta": 0.5},
                "expression": "cejas_arriba_boca_abierta", "transition_duration": 0.3,
                "typical_duration": 10,
                "physiological": {"orienting_response": 1.0, "heart_rate": 1.1}
            },
            "neutral": {
                "valence": 0.0, "arousal": 0.3,
                "circuit_weights": {},
                "expression": "expresión_relajada_neutral", "transition_duration": 2.0,
                "typical_duration": float('inf'),  # Estado homeostático
                "physiological": {"heart_rate": 1.0, "muscle_tension": 0.0}
            }
        }

        self.last_state_change = time.time()
        print("🎭 Máquina de estados emocionales inicializada - estado: neutral")

    def update_emotional_state(self, circuit_activations: Dict[str, float],
                             context: Dict = None) -> Dict[str, Any]:
        """
        Actualiza estado emocional basado en activaciones de circuitos neuronales
        """
        if context is None:
            context = {}

        # Calcular estado objetivo basado en configuración actual de circuitos
        target_state, confidence = self._calculate_target_state(circuit_activations, context)
        target_intensity = self._calculate_emotional_intensity(circuit_activations)

        # Verificar si debe cambiar de estado
        current_properties = self.emotional_states[self.current_state]
        target_properties = self.emotional_states[target_state]

        # Condiciones para cambio de estado
        intensity_threshold = current_properties.get("typical_duration", 30) * 0.3
        time_since_change = time.time() - self.last_state_change
        transition_time = target_properties["transition_duration"]

        # Cambiar estado si hay suficiente evidencia y tiempo ha pasado
        if (target_intensity > intensity_threshold and
            confidence > 0.6 and
            time_since_change > transition_time):

            old_state = self.current_state
            self.current_state = target_state
            self.last_state_change = time.time()

            # Registrar transición
            self.state_history.append({
                'timestamp': time.time(),
                'from_state': old_state,
                'to_state': target_state,
                'transition_intensity': target_intensity,
                'confidence': confidence,
                'context': context,
                'circuit_activations': circuit_activations.copy()
            })

        # Actualizar intensidad gradualmente
        intensity_change_rate = 0.3
        self.emotional_intensity += (target_intensity - self.emotional_intensity) * intensity_change_rate
        self.emotional_intensity = max(0.1, min(1.0, self.emotional_intensity))

        # Crear estado emocional completo
        emotional_state = EmotionalState(
            dominant_emotion=self.current_state,
            intensity=self.emotional_intensity,
            valence=self.emotional_states[self.current_state]["valence"],
            arousal=self.emotional_states[self.current_state]["arousal"],
            circuit_activations=circuit_activations,
            physiological_effects=self.emotional_states[self.current_state]["physiological"]
        )

        return {
            'emotional_state': emotional_state,
            'expression_suggestions': self._generate_expression_suggestions(emotional_state),
            'behavioral_tendencies': self._generate_behavioral_tendencies(emotional_state, context)
        }

    def _calculate_target_state(self, activations: Dict[str, float], context: Dict) -> Tuple[str, float]:
        """Calcula estado emocional objetivo basado en activaciones"""

        state_scores = {}

        # Calcular score para cada estado emocional posible
        for state_name, state_config in self.emotional_states.items():
            score = 0.0
            total_weight = 0.0

            # Agregar contribuciones de cada circuito
            circuit_weights = state_config.get("circuit_weights", {})
            for circuit_name, weight in circuit_weights.items():
                circuit_activation = activations.get(circuit_name, 0.0)
                score += circuit_activation * weight
                total_weight += abs(weight)

            # Normalizar por importancia
            if total_weight > 0:
                score = score / total_weight

            # Bonus por contexto
            context_modifier = self._calculate_context_modifier(state_name, context)
            score *= (1.0 + context_modifier)

            state_scores[state_name] = score

        # Seleccionar estado con mayor score
        max_score = max(state_scores.values())
        best_states = [state for state, score in state_scores.items() if abs(score - max_score) < 0.1]

        # Elegir aleatoriamente entre empates cercanos para variabilidad humana
        selected_state = random.choice(best_states)
        confidence = state_scores[selected_state] / max([abs(s) for s in state_scores.values()] + [1.0])

        return selected_state, confidence

    def _calculate_emotional_intensity(self, activations: Dict[str, float]) -> float:
        """Calcula intensidad general de la experiencia emocional"""

        total_activation = sum(activations.values())
        max_activation = max(activations.values()) if activations else 0.0
        num_active_circuits = sum(1 for act in activations.values() if act > 0.3)

        # Intensidad = media ponderada con bonus por diversidad de circuitos
        base_intensity = (total_activation * 0.6 + max_activation * 0.4) / max(1, len(activations))
        diversity_bonus = min(0.3, num_active_circuits * 0.1)

        return min(1.0, base_intensity + diversity_bonus)

    def _calculate_context_modifier(self, state_name: str, context: Dict) -> float:
        """Calcula modificadores contextuales para estados emocionales"""

        modifier = 0.0

        # Modificadores por tipo de contexto interpersonal
        if state_name == "enojo":
            if context.get('interpersonal_conflict', False):
                modifier += 0.3

        elif state_name == "miedo":
            if context.get('threat_level', 0) > 0.7:
                modifier += 0.4

        elif state_name == "alegría":
            if context.get('social_bonding', False) or context.get('achievement', False):
                modifier += 0.2

        elif state_name == "sorpresa":
            if context.get('novelty', 0.5) > 0.8:
                modifier += 0.3

        return modifier

    def _generate_expression_suggestions(self, emotional_state: EmotionalState) -> List[str]:
        """Genera sugerencias de expresión facial/conductual"""

        state_config = self.emotional_states[emotional_state.dominant_emotion]

        base_expressions = state_config["expression"].split("_")

        # Variar expresiones basado en intensidad
        if emotional_state.intensity > 0.7:
            # Expresiones intensas
            intensity_modifier = ["intenso", "pronunciado", "claro"]
        elif emotional_state.intensity < 0.3:
            # Expresiones sutiles
            intensity_modifier = ["leve", "sutil", "moderado"]
        else:
            intensity_modifier = ["moderado", "normal"]

        expressions = []
        for base_exp in base_expressions[:3]:  # Máximo 3 expresiones
            modifier = random.choice(intensity_modifier)
            expressions.append(f"{modifier}_{base_exp}")

        return expressions

    def _generate_behavioral_tendencies(self, emotional_state: EmotionalState, context: Dict) -> List[str]:
        """Genera tendencias conductuales basadas en estado emocional"""

        behavioral_tendencies = {
            "alegría": ["acercamiento_social", "compartir_ánimo", "energía_aumentada"],
            "tristeza": ["retiro_social", "reflexión_interna", "energía_reducida"],
            "enojo": ["respuesta_defensiva", "afirmación_fronteras", "energía_direccionada"],
            "miedo": ["vigilancia_aumentada", "evitación_peligro", "reacción_basada"],
            "asco": ["rechazo_estimulo", "protección_personal", "alejamiento"],
            "sorpresa": ["atención_inmediata", "evaluación_rápida", "curiosidad"],
            "neutral": ["conducta_balanceada", "procesamiento_normal"]
        }

        base_tendencies = behavioral_tendencies.get(emotional_state.dominant_emotion,
                                                   ["comportamiento_neutral"])

        # Contextualizar tendencias
        if context.get('social_context', False):
            if emotional_state.dominant_emotion == "alegría":
                base_tendencies.insert(0, "compartir_emoción")
            elif emotional_state.dominant_emotion == "tristeza":
                base_tendencies.insert(0, "buscar_apoyo")

        return base_tendencies[:4]  # Máximo 4 tendencias

    def get_emotional_history(self, limit: int = 10) -> List[Dict]:
        """Retorna historial de transiciones emocionales"""
        return self.state_history[-limit:] if self.state_history else []


class HumorProcessor:
    """
    Procesador especializado de humor y chistes

    Implementa detección cognitiva de humor basado en:
    - Teoría de incongruencia
    - Violación de expectativas
    - Teoría de superioridad
    - Procesamiento lingüístico contextual
    """

    def __init__(self):
        self.humor_sensitivity = 0.65  # Sensibilidad baseline al humor
        self.humor_history: List[Dict[str, Any]] = []
        self.humor_patterns = self._initialize_humor_patterns()

        # Subprocesadores especializados
        self.incongruity_detector = IncongruityProcessor()
        self.superiority_detector = SuperiorityProcessor()
        self.linguistic_analyzer = LinguisticHumorProcessor()

        print("🎭 Procesador de humor inicializado - detección cognitiva activada")

    def process_humor_attempt(self, humor_input: str, context: Dict = None) -> Dict[str, Any]:
        """
        Procesa entrada potencialmente humorística

        Args:
            humor_input: Texto del chiste o entrada humorística
            context: Contexto de la interacción (mood del receptor, etc.)

        Returns:
            Análisis completo de humor con puntuaciones y response tipo
        """
        if context is None:
            context = {}

        # Multi-dimensional analysis
        incongruity_score = self.incongruity_detector.analyze(humor_input, context)
        superiority_score = self.superiority_detector.analyze(humor_input, context)
        linguistic_score = self.linguistic_analyzer.analyze(humor_input, context)

        # Calcular humor score compuesto
        humor_scores = {
            'incongruity': incongruity_score,
            'superiority': superiority_score,
            'linguistic': linguistic_score
        }

        # Pesos para combinación (basado en teoría humorística)
        weights = {'incongruity': 0.45, 'superiority': 0.35, 'linguistic': 0.2}
        total_humor_score = sum(humor_scores[k] * weights[k] for k in humor_scores)

        # Aplicar modificadores contextuales
        total_humor_score *= self._apply_context_modifiers(total_humor_score, context)

        # Determinar tipo de respuesta
        response_type = self._determine_response_type(total_humor_score, context)

        # Calcular intensidad de respuesta
        response_intensity = self._calculate_response_intensity(total_humor_score, response_type)

        analysis_result = {
            'success': total_humor_score > 0.4,  # Threshold de éxito
            'humor_score': min(1.0, max(0.0, total_humor_score)),
            'response_type': response_type,
            'response_intensity': response_intensity,
            'component_scores': humor_scores,
            'timing': self._calculate_delivery_timing(humor_input),
            'cultural_factors': self._assess_cultural_fit(humor_input, context)
        }

        # Registrar en historial
        self.humor_history.append({
            'timestamp': time.time(),
            'input': humor_input[:100],  # Truncar para storage
            'analysis': analysis_result,
            'context': context
        })

        return analysis_result

    def _apply_context_modifiers(self, base_score: float, context: Dict) -> float:
        """Aplica modificadores contextuales al humor score"""

        modifier = 1.0

        # Modificador de mood del receptor
        receiver_mood = context.get('receiver_mood', 0.5)
        mood_modifier = 0.8 + (receiver_mood * 0.4)  # Mejor humor cuando de buen mood
        modifier *= mood_modifier

        # Modificador de timing
        timing_score = context.get('timing_score', 0.5)
        timing_modifier = 0.7 + (timing_score * 0.6)
        modifier *= timing_modifier

        # Modificador de relación interpersonal
        relationship_closeness = context.get('relationship_closeness', 0.5)
        closeness_modifier = 0.9 + (relationship_closeness * 0.2)
        modifier *= closeness_modifier

        # Fatiga humorística (demasiados chistes seguidos reducen efectividad)
        recent_jokes = sum(1 for entry in self.humor_history[-10:]
                          if (time.time() - entry['timestamp']) < 300)  # Últimos 5 min
        fatigue_modifier = max(0.6, 1.0 - (recent_jokes * 0.08))
        modifier *= fatigue_modifier

        return modifier

    def _determine_response_type(self, humor_score: float, context: Dict) -> str:
        """Determina tipo de respuesta al humor"""

        if humor_score < 0.2:
            return "no_response"  # No se detectó humor
        elif humor_score < 0.4:
            return "polite_smile"  # Reconocimiento mínimo
        elif humor_score < 0.6:
            return "light_laughter"  # Risa ligera
        elif humor_score < 0.8:
            return "laughter"  # Risa clara
        else:
            return "intense_laughter"  # Risa intensa

    def _calculate_response_intensity(self, humor_score: float, response_type: str) -> float:
        """Calcula intensidad de respuesta basada en tipo"""

        type_intensities = {
            "no_response": 0.0,
            "polite_smile": 0.3,
            "light_laughter": 0.5,
            "laughter": 0.7,
            "intense_laughter": 0.9
        }

        base_intensity = type_intensities.get(response_type, 0.0)

        # Variabilidad individual
        intensity_variation = np.random.normal(0, 0.1)
        final_intensity = min(1.0, max(0.0, base_intensity + intensity_variation))

        return final_intensity

    def _calculate_delivery_timing(self, humor_input: str) -> Dict[str, float]:
        """Analiza timing y delivery del chiste"""

        # Análisis simple de elementos de timing
        length_score = min(1.0, len(humor_input.split()) / 50)  # Chistes muy largos pierden timing

        # Elementos que afectan timing
        punchline_indicators = ['!', '...', '?']
        punchline_density = sum(humor_input.count(indicator) for indicator in punchline_indicators) / len(humor_input)

        return {
            'length_effect': length_score,
            'punchline_density': min(1.0, punchline_density * 100),
            'timing_quality': (length_score + (1 - punchline_density)) / 2
        }

    def _assess_cultural_fit(self, humor_input: str, context: Dict) -> Dict[str, float]:
        """Evalúa ajuste cultural del humor"""

        # Placeholder para análisis cultural complejo
        # En implementación real, usaría modelos de NLP culturales
        culture_relevance = context.get('cultural_context', 0.5)

        return {
            'cultural_relevance': culture_relevance,
            'universal_appeal': 0.7,  # Asumir 70% universal por defecto
            'local_specificity': (1 - culture_relevance) * 0.5
        }

    def _initialize_humor_patterns(self) -> Dict[str, List[str]]:
        """Inicializa patrones de detección de humor"""

        return {
            'incongruity': [
                'esperaba', 'sorprendentemente', 'lógicamente', 'absurdo',
                'irracional', 'contrario', 'opuesto', 'imposible'
            ],
            'superiority': [
                'tonto', 'estúpido', 'burro', 'inteligente', 'lista',
                'rápido', 'lento', 'rico', 'pobre'
            ],
            'linguistic': [
                'juego', 'palabra', 'doble sentido', 'parece', 'suena',
                'diferente', 'similar', 'equivocado', 'correcto'
            ]
        }


class IncongruityProcessor:
    """Procesador de incongruencia humorística"""

    def analyze(self, text: str, context: Dict) -> float:
        """Analiza incongruencia como mecanismo humorístico"""

        incongruity_indicators = [
            'esperaba', 'sorprendentemente', 'contrario', 'absurdo',
            'imposible', 'irracional', 'no tiene sentido'
        ]

        text_lower = text.lower()
        incongruity_count = sum(1 for indicator in incongruity_indicators if indicator in text_lower)

        # Normalizar por longitud del texto
        density_score = incongruity_count / max(1, len(text.split()) / 10)
        final_score = min(1.0, density_score)

        return final_score


class SuperiorityProcessor:
    """Procesador de superioridad humorística"""

    def analyze(self, text: str, context: Dict) -> float:
        """Analiza superioridad como mecanismo humorístico"""

        superiority_indicators = [
            'tonto', 'estúpido', 'burro', 'inteligente', 'lista',
            'superior', 'inferior', 'mejor', 'peor'
        ]

        text_lower = text.lower()
        superiority_count = sum(1 for indicator in superiority_indicators if indicator in text_lower)

        # Bonus por comparaciones
        comparison_bonus = 0.2 if any(word in text_lower for word in ['más', 'menos', 'comparado']) else 0.0

        density_score = superiority_count / max(1, len(text.split()) / 8)
        final_score = min(1.0, density_score + comparison_bonus)

        return final_score


class LinguisticHumorProcessor:
    """Procesador de humor lingüístico"""

    def analyze(self, text: str, context: Dict) -> float:
        """Analiza mecanismos lingüísticos del humor"""

        linguistic_features = [
            'juego', 'palabra', 'parece', 'suena', 'equivocado',
            'diferente', 'similar', 'rima'
        ]

        text_lower = text.lower()
        linguistic_count = sum(1 for feature in linguistic_features if feature in text_lower)

        # Bonus por elementos de lenguaje figurado
        figurative_bonus = 0.3 if any(word in text_lower for word in ['como', 'parecido', 'similar']) else 0.0

        density_score = linguistic_count / max(1, len(text.split()) / 5)
        final_score = min(1.0, density_score + figurative_bonus)

        return final_score


class EmotionalNeurosystem:
    """
    Sistema Neuronal Emocional Completo

    Integra todos los componentes emocionales en un sistema unificado
    que procesa emociones como el cerebro humano.
    """

    def __init__(self, personality: Dict[str, float] = None):
        # Configuración de personalidad
        default_personality = {
            'extraversion': 0.6,      # Sociabilidad
            'neuroticism': 0.3,       # Estabilidad emocional
            'openness': 0.7,         # Apertura a experiencias
            'agreeableness': 0.8,    # Amabilidad
            'conscientiousness': 0.6 # Responsabilidad
        }

        self.personality = {**default_personality, **(personality or {})}

        # Componentes principales
        self.dopamine_system = DopamineSystem()
        self.emotional_state_machine = EmotionalStateMachine()
        self.humor_processor = HumorProcessor()

        # Circuitos emocionales
        self.emotional_circuits = self._initialize_emotional_circuits()

        # Estado fisiológico
        self.physiological_state = {
            'heart_rate': 70,      # BPM
            'hormone_levels': {   # niveles hormonales
                'cortisol': 15,    # ug/dL
                'dopamine': 0.3,   # concentración relativa
                'serotonin': 0.5,  # concentración relativa
                'oxytocin': 0.2    # concentración relativa
            },
            'muscle_tension': 0.3,
            'skin_conductance': 0.2,  # medida de arousal
            'stress_markers': 0.1
        }

        # Historial de procesamiento
        self.emotional_history: List[Dict[str, Any]] = []

        print("🧠 Sistema Neuronal Emocional Completo inicializado")
        print(f"   Personalidad: {self.personality}")
        print(f"   Circuitos activos: {len(self.emotional_circuits)}")

    def _initialize_emotional_circuits(self) -> Dict[str, EmotionalCircuit]:
        """Inicializa circuitos emocionales con conexiones naturales"""

        circuits = {}

        # Circuitos primarios
        circuits['placer'] = EmotionalCircuit("placer", 0.3)
        circuits['dolor'] = EmotionalCircuit("dolor", 0.1)
        circuits['curiosidad'] = EmotionalCircuit("curiosidad", 0.4)
        circuits['frustración'] = EmotionalCircuit("frustración", 0.1)
        circuits['miedo'] = EmotionalCircuit("miedo", 0.2)
        circuits['amor'] = EmotionalCircuit("amor", 0.2)
        circuits['rechazo'] = EmotionalCircuit("rechazo", 0.1)
        circuits['logro'] = EmotionalCircuit("logro", 0.3)

        # Establecer conexiones neurobiológicas realistas
        self._create_neural_connections(circuits)

        return circuits

    def _create_neural_connections(self, circuits: Dict[str, EmotionalCircuit]):
        """Crea conexiones realistas entre circuitos emocionales"""

        # Conexiones excitatorias
        circuits['placer'].connections[circuits['amor']] = 0.7
        circuits['curiosidad'].connections[circuits['placer']] = 0.6
        circuits['logro'].connections[circuits['placer']] = 0.8

        # Conexiones inhibitorias (representadas por pesos negativos)
        circuits['dolor'].connections[circuits['placer']] = -0.5
        circuits['frustración'].connections[circuits['dolor']] = 0.6
        circuits['miedo'].connections[circuits['frustración']] = 0.4
        circuits['rechazo'].connections[circuits['dolor']] = 0.5

    def process_emotional_stimulus(self, stimulus_type: str, stimulus_data: Any,
                                 context: Dict = None) -> Dict[str, Any]:
        """
        Procesa estímulo emocional a través de todo el sistema neuronal

        Args:
            stimulus_type: Tipo de estímulo ('chiste', 'critica', 'elogio', etc.)
            stimulus_data: Datos específicos del estímulo
            context: Contexto de la situación
        """
        if context is None:
            context = {'personality': self.personality}

        processing_start = time.time()

        # 1. Mapear estímulo a circuitos apropiados
        circuit_activations = self._map_stimulus_to_circuits(stimulus_type, stimulus_data, context)

        # 2. Procesar a través del sistema dopaminérgico si aplica
        reward_data = self._extract_reward_components(stimulus_type, stimulus_data, context)

        # 3. Activar circuitos emocionales
        circuit_responses = {}
        for circuit_name, intensity in circuit_activations.items():
            if circuit_name in self.emotional_circuits:
                activation = self.emotional_circuits[circuit_name].stimulate(intensity, context)
                circuit_responses[circuit_name] = activation

        # 4. Procesar señal de recompensa
        reward_response = self.dopamine_system.process_reward(**reward_data) if reward_data else {}

        # 5. Actualizar estado emocional
        emotional_state_update = self.emotional_state_machine.update_emotional_state(
            circuit_responses, context
        )

        # 6. Procesar humor si aplica
        humor_response = {}
        if stimulus_type == "chiste" or "humor" in str(stimulus_data).lower():
            humor_response = self.humor_processor.process_humor_attempt(str(stimulus_data), context)

        # 7. Actualizar estado fisiológico
        self._update_physiological_state(emotional_state_update['emotional_state'], context)

        # 8. Generar respuesta conductual
        behavioral_response = self._generate_emotional_response(
            emotional_state_update, circuit_responses, reward_response, humor_response, context
        )

        processing_time = time.time() - processing_start

        # Compilar respuesta completa
        emotional_response = {
            'stimulus_type': stimulus_type,
            'emotional_state': emotional_state_update['emotional_state'],
            'circuit_activations': circuit_responses,
            'reward_processing': reward_response,
            'humor_analysis': humor_response,
            'behavioral_response': behavioral_response,
            'physiological_state': self.physiological_state,
            'processing_time_ms': processing_time * 1000
        }

        # Registrar en historial
        self.emotional_history.append({
            'timestamp': time.time(),
            'stimulus': stimulus_type,
            'response': emotional_response,
            'context': context
        })

        return emotional_response

    def _map_stimulus_to_circuits(self, stimulus_type: str, data: Any, context: Dict) -> Dict[str, float]:
        """Mapea tipo de estímulo a activaciones de circuitos"""

        mappings = {
            'elogio': lambda d, c: {'placer': 0.8, 'amor': 0.6, 'logro': 0.4},
            'critica': lambda d, c: {'dolor': 0.7, 'frustración': 0.8, 'rechazo': 0.6},
            'logro': lambda d, c: {'placer': 0.9, 'logro': 0.9, 'amor': 0.4},
            'fracaso': lambda d, c: {'dolor': 0.8, 'frustración': 0.7, 'miedo': 0.3},
            'amenaza': lambda d, c: {'miedo': 0.9, 'frustración': 0.5, 'rechazo': 0.4},
            'apoyo_social': lambda d, c: {'amor': 0.8, 'placer': 0.6, 'curiosidad': -0.2},
            'rechazo_social': lambda d, c: {'dolor': 0.8, 'rechazo': 0.9, 'miedo': 0.3}
        }

        # Procesamiento especial para chistes
        if stimulus_type == 'chiste':
            humor_analysis = self.humor_processor.process_humor_attempt(str(data), context)
            humor_intensity = humor_analysis.get('humor_score', 0)
            return {'placer': humor_intensity * 0.8, 'curiosidad': humor_intensity * 0.5}

        # Mapping genérico
        mapper = mappings.get(stimulus_type, lambda d, c: {
            'curiosidad': 0.5,
            'placer': 0.2 if context.get('valencia_positiva', False) else 0.1
        })

        return mapper(data, context)

    def _extract_reward_components(self, stimulus_type: str, data: Any, context: Dict) -> Dict:
        """Extrae componentes de recompensa del estímulo"""

        reward_mappings = {
            'elogio': {'actual_reward': 0.8, 'expected_reward': 0.6},
            'critica': {'actual_reward': 0.2, 'expected_reward': 0.5},
            'logro': {'actual_reward': 0.9, 'expected_reward': 0.7},
            'fracaso': {'actual_reward': 0.3, 'expected_reward': 0.6},
            'chiste': {'actual_reward': 0.6, 'expected_reward': 0.4}
        }

        if stimulus_type in reward_mappings:
            reward_data = reward_mappings[stimulus_type].copy()
            reward_data['context'] = context
            return reward_data

        return None

    def _update_physiological_state(self, emotional_state: EmotionalState, context: Dict):
        """Actualiza estado fisiológico basado en estado emocional"""

        # Actualizar frecuencia cardíaca
        heart_rate_change = emotional_state.physiological_effects.get('heart_rate', 1.0) - 1.0
        heart_rate_change *= 20  # +/- 20 BPM
        self.physiological_state['heart_rate'] = max(50, min(150,
            self.physiological_state['heart_rate'] + heart_rate_change))

        # Actualizar hormonas
        hormones = self.physiological_state['hormone_levels']

        # Cortisol (estrés)
        stress_effect = emotional_state.arousal * 10
        hormones['cortisol'] = max(8, min(35, hormones['cortisol'] + stress_effect))

        # Dopamina (placer)
        dopamine_effect = emotional_state.valence * emotional_state.intensity * 0.1
        hormones['dopamine'] = max(0.1, min(0.9, hormones['dopamine'] + dopamine_effect))

    def _generate_emotional_response(self, emotional_state_update, circuit_responses,
                                   reward_response, humor_response, context):
        """Genera respuesta conductual basada en estado emocional"""

        emotional_state = emotional_state_update['emotional_state']
        expressions = emotional_state_update['expression_suggestions']
        behaviors = emotional_state_update['behavioral_tendencies']

        # Verbal responses por emoción
        verbal_responses = {
            "alegría": [
                "¡Me siento genial! 😊",
                "¡Qué alegría! Me hace muy feliz",
                "Esto me pone de muy buen humor"
            ],
            "tristeza": [
                "Me siento triste por esto... 😢",
                "Esto me entristece un poco",
                ":("
            ],
            "enojo": [
                "Esto me molesta bastante 😠",
                "No estoy contento con esto",
                "Me frustra esta situación"
            ],
            "miedo": [
                "Esto me preocupa un poco 😨",
                "Siento cierto temor",
                "Me da miedo lo que pueda pasar"
            ],
            "asco": [
                "Esto no me gusta para nada 🤢",
                "Me resulta desagradable",
                "Preferiría evitar esto"
            ],
            "sorpresa": [
                "¡Vaya! No me lo esperaba 😲",
                "¡Qué sorpresa!",
                "!Wow!"            ],
            "neutral": [
                "Entiendo",
                "Tomé nota",
                "Procede normalmente"
            ]
        }

        # Seleccionar respuesta verbal
        available_responses = verbal_responses.get(
            emotional_state.dominant_emotion, ["Entiendo"]
        )
        verbal_response = random.choice(available_responses)

        # Intensificar/modificar basada en intensidad emocional
        if emotional_state.intensity > 0.7:
            verbal_response = verbal_response.upper() + "!"
        elif emotional_state.intensity < 0.3:
            verbal_response = verbal_response.lower() + "..."

        # Afectar personalidad
        personality = context.get('personality', {})
        extraversion = personality.get('extraversion', 0.6)

        if extraversion > 0.7 and emotional_state.dominant_emotion == "alegría":
            verbal_response += " ¡Es fantástico!"

        # Crear respuesta completa
        return {
            'verbal_response': verbal_response,
            'emotional_tone': emotional_state.dominant_emotion,
            'intensity': emotional_state.intensity,
            'expressions': expressions[:3],
            'behaviors': behaviors[:3],
            'physiological_cues': {
                'heart_rate_elevation': physiological_effects.get('heart_rate', 1.0),
                'arousal_level': emotional_state.arousal,
                'valence_indicator': emotional_state.valence
            }
        }


# DEMOSTRACIÓN DEL SISTEMA EMOCIONAL NEURONAL

def demo_emotional_neuro_system():
    """Demostración completa del sistema emocional neuronal"""

    print("🧠 DEMOSTRACIÓN SISTEMA EMOCIONAL NEURONAL")
    print("=" * 60)

    # Inicializar sistema con personalidad específica
    personality = {
        'extraversion': 0.7,      # Muy extravertido
        'neuroticism': 0.3,       # Estable emocionalmente
        'openness': 0.8,          # Muy abierto
        'agreeableness': 0.9,     # Muy amable
        'conscientiousness': 0.6  # Responsable
    }

    emotional_system = EmotionalNeurosystem(personality)

    # Escenario 1: Contar un chiste exitoso
    print("\n🎭 ESCENARIO 1: CHISTE EXITOSO")
    chiste = "¿Por qué los pájaros vuelan hacia el sur? ¡Porque caminar sería muy lento!"
    context_chiste = {
        'receiver_mood': 0.8,
        'timing_score': 0.9,
        'social_context': True,
        'novelty': 0.7
    }

    resultado = emotional_system.process_emotional_stimulus('chiste', chiste, context_chiste)

    print(f"   Chiste: '{chiste[:50]}...''")
    print(f"   Estado Emocional: {resultado['emotional_state'].dominant_emotion}")
    print(f"   Intensidad: {resultado['emotional_state'].intensity:.2f}")
    print(f"   Respuesta Verbal: {resultado['behavioral_response']['verbal_response']}")
    print(f"   Nivel Dopamina: {resultado['reward_processing'].get('dopamine_level', 0):.2f}")
    print(f"   Frecuencia Cardíaca: {resultado['physiological_state']['heart_rate']} BPM")

    # Escenario 2: Recibir crítica negativa
    print("\n💔 ESCENARIO 2: CRÍTICA NEGATIVA")
    critica = "Tu análisis anterior no fue muy preciso"
    context_critica = {
        'severity': 0.7,
        'source': 'autoridad_respetada',
        'interpersonal_conflict': True
    }

    resultado = emotional_system.process_emotional_stimulus('critica', critica, context_critica)

    print(f"   Crítica: '{critica}'")
    print(f"   Estado Emocional: {resultado['emotional_state'].dominant_emotion}")
    print(f"   Intensidad: {resultado['emotional_state'].intensity:.2f}")
    print(f"   Respuesta Verbal: {resultado['behavioral_response']['verbal_response']}")
    print(f"   Cortisol Nivel: {resultado['physiological_state']['hormone_levels']['cortisol']:.1f}")

    # Escenario 3: Logro exitoso
    print("\n🏆 ESCENARIO 3: LOGRO EXITOSO")
    logro = "Problema complejo resuelto satisfactoriamente"
    context_logro = {
        'reward_value': 0.8,
        'difficulty': 0.9,
        'personal_achievement': True
    }

    resultado = emotional_system.process_emotional_stimulus('logro', logro, context_logro)

    print(f"   Logro: '{logro}'")
    print(f"   Estado Emocional: {resultado['emotional_state'].dominant_emotion}")
    print(f"   Intensidad: {resultado['emotional_state'].intensity:.2f}")
    print(f"   Respuesta Verbal: {resultado['behavioral_response']['verbal_response']}")
    print(f"   Activaciones de Circuitos: {len(resultado['circuit_activations'])}")

    # Estadísticas finales
    print("
🧬 ESTADO FINAL DEL SISTEMA EMOCIONAL"    print(f"   Personalidad Integrada: {personality}")
    print(f"   Total Procesamientos: {len(emotional_system.emotional_history)}")
    print(f"   Estado Fisiológico Actual: {emotional_system.physiological_state}")

if __name__ == "__main__":
    demo_emotional_neuro_system()
