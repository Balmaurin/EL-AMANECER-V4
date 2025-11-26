# -*- coding: utf-8 -*-
"""
SISTEMA EMOCIONAL HUMANO COMPLETO - IMPLEMENTACIÓN FUNCIONAL
Basado en neurociencia afectiva real con integración hormonal y neurotransmisores
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import time

# ==================== EMOCIONES BÁSICAS ====================

class BasicEmotions:
    """6 Emociones básicas universales (Ekman)"""
    ALEGRIA = "alegria"
    TRISTEZA = "tristeza"
    MIEDO = "miedo"
    ENOJO = "enojo"
    ASCO = "asco"
    SORPRESA = "sorpresa"

class SocialEmotions:
    """12 Emociones sociales complejas"""
    AMOR = "amor"
    ODIO = "odio"
    CELOS = "celos"
    VERGUENZA = "verguenza"
    CULPA = "culpa"
    ORGULLO = "orgullo"
    ENVIDIA = "envidia"
    GRATITUD = "gratitud"
    ADMIRACION = "admiracion"
    COMPASION = "compasion"
    EMPATIA = "empatia"
    SOLIDARIDAD = "solidaridad"

class ComplexEmotions:
    """12 Emociones complejas meta-cognitivas"""
    NOSTALGIA = "nostalgia"
    ESPERANZA = "esperanza"
    DESESPERACION = "desesperacion"
    SOLEDAD = "soledad"
    FRUSTRACION = "frustracion"
    EXTASIS = "extasis"
    PANICO = "panico"
    MELANCOLIA = "melancolia"
    SERENIDAD = "serenidad"
    CURIOSIDAD = "curiosidad"
    SATISFACCION = "satisfaccion"
    INSATISFACCION = "insatisfaccion"


# ==================== CIRCUITO EMOCIONAL ====================

@dataclass
class EmotionalCircuit:
    """Circuito emocional basado en neurociencia afectiva"""
    emotion_name: str
    valence: float  # -1 (negativo) a +1 (positivo)
    arousal: float  # arousal fisiológico (0-1)
    intensity: float = 0.0  # Intensidad actual (0-1)
    duration: float = 0.0  # Duración en segundos
    onset_time: Optional[float] = None
    
    # Neurotransmisores asociados
    neurotransmitters: Dict[str, float] = field(default_factory=dict)
    
    # Hormonas asociadas
    hormones: Dict[str, float] = field(default_factory=dict)
    
    # Decay rate (qué tan rápido desaparece)
    decay_rate: float = 0.1
    
    def update(self, delta_time: float):
        """Actualiza estado del circuito emocional con decaimiento temporal"""
        if self.intensity > 0:
            # Decaimiento exponencial
            self.intensity *= (1 - self.decay_rate * delta_time)
            self.duration += delta_time
            
            # Umbral mínimo
            if self.intensity < 0.01:
                self.intensity = 0.0
                self.duration = 0.0
                self.onset_time = None


# ==================== SISTEMA EMOCIONAL HUMANO COMPLETO ====================

class HumanEmotionalSystem:
    """
    Sistema Emocional Humano Funcional
    
    Basado en:
    - Circumplex Model of Affect (Russell)
    - Constructivist Theory of Emotion (Barrett)
    - Neurobiología afectiva (LeDoux, Damasio)
    
    Implementa 35 circuitos emocionales con:
    - Neurotransmisores reales (dopamina, serotonina, etc.)
    - Hormonas reales (cortisol, oxitocina, etc.)
    - Decaimiento temporal realista
    - Interacciones emocionales (blending, supresión)
    """
    
    def __init__(self, num_circuits: int = 35, personality: Dict[str, float] = None):
        self.num_circuits = num_circuits
        self.personality = personality or {
            'neuroticism': 0.5,
            'extraversion': 0.5,
            'openness': 0.5,
            'agreeableness': 0.5,
            'conscientiousness': 0.5
        }
        
        # Inicializar circuitos emocionales
        self.circuits = self._initialize_emotional_circuits()
        
        # Estado emocional actual (circumplex: valence + arousal)
        self.current_valence = 0.0  # -1 a +1
        self.current_arousal = 0.5  # 0 a 1
        
        # Historia emocional
        self.emotion_history: List[Dict[str, Any]] = []
        
        # Baseline emocional (influido por personalidad)
        self.baseline_mood = self._calculate_baseline_mood()
        
        # Regulación emocional
        self.regulation_strength = 0.5
        self.suppression_active = False
        
        print(f"❤️  Sistema Emocional Humano inicializado: {len(self.circuits)} circuitos activos")
    
    def _initialize_emotional_circuits(self) -> Dict[str, EmotionalCircuit]:
        """Inicializa todos los circuitos emocionales con sus propiedades neurobiológicas"""
        circuits = {}
        
        # Emociones básicas (Ekman)
        basic_emotions = [
            ("alegria", 0.8, 0.7, {"dopamine": 0.7, "serotonin": 0.6}, {"oxytocin": 0.5}),
            ("tristeza", -0.7, 0.3, {"serotonin": -0.5}, {"cortisol": 0.3}),
            ("miedo", -0.5, 0.9, {"norepinephrine": 0.8}, {"adrenaline": 0.9, "cortisol": 0.7}),
            ("enojo", -0.6, 0.8, {"norepinephrine": 0.7, "testosterone": 0.6}, {"cortisol": 0.4}),
            ("asco", -0.4, 0.5, {}, {}),
            ("sorpresa", 0.0, 0.8, {"dopamine": 0.4}, {"adrenaline": 0.5}),
        ]
        
        # Emociones sociales
        social_emotions = [
            ("amor", 0.9, 0.6, {"dopamine": 0.8, "serotonin": 0.7}, {"oxytocin": 0.9}),
            ("odio", -0.9, 0.7, {}, {"testosterone": 0.6, "cortisol": 0.5}),
            ("celos", -0.6, 0.7, {}, {"cortisol": 0.6}),
            ("verguenza", -0.5, 0.6, {}, {"cortisol": 0.5}),
            ("culpa", -0.4, 0.4, {}, {"cortisol": 0.4}),
            ("orgullo", 0.7, 0.5, {"dopamine": 0.6}, {}),
            ("envidia", -0.5, 0.5, {}, {}),
            ("gratitud", 0.6, 0.4, {"serotonin": 0.5}, {"oxytocin": 0.4}),
            ("admiracion", 0.5, 0.5, {"dopamine": 0.5}, {}),
            ("compasion", 0.4, 0.3, {"serotonin": 0.4}, {"oxytocin": 0.6}),
            ("empatia", 0.3, 0.4, {"serotonin": 0.5}, {"oxytocin": 0.7}),
            ("solidaridad", 0.5, 0.4, {"serotonin": 0.4}, {"oxytocin": 0.5}),
        ]
        
        # Emociones complejas
        complex_emotions = [
            ("nostalgia", 0.2, 0.4, {"serotonin": 0.3}, {}),
            ("esperanza", 0.6, 0.5, {"dopamine": 0.5}, {}),
            ("desesperacion", -0.8, 0.6, {}, {"cortisol": 0.8}),
            ("soledad", -0.6, 0.4, {}, {"cortisol": 0.5}),
            ("frustracion", -0.5, 0.7, {}, {"cortisol": 0.4}),
            ("extasis", 1.0, 0.9, {"dopamine": 0.9, "serotonin": 0.8}, {}),
            ("panico", -0.8, 1.0, {"norepinephrine": 0.9}, {"adrenaline": 1.0, "cortisol": 0.9}),
            ("melancolia", -0.4, 0.3, {"serotonin": -0.4}, {}),
            ("serenidad", 0.5, 0.2, {"serotonin": 0.6}, {}),
            ("curiosidad", 0.3, 0.6, {"dopamine": 0.5}, {}),
            ("satisfaccion", 0.7, 0.3, {"dopamine": 0.6, "serotonin": 0.7}, {}),
            ("insatisfaccion", -0.3, 0.5, {}, {}),
        ]
        
        # Combinar todas las emociones
        all_emotions = basic_emotions + social_emotions + complex_emotions
        
        # Crear circuitos
        for name, valence, arousal, neurotransmitters, hormones in all_emotions[:self.num_circuits]:
            circuits[name] = EmotionalCircuit(
                emotion_name=name,
                valence=valence,
                arousal=arousal,
                neurotransmitters=neurotransmitters,
                hormones=hormones,
                decay_rate=0.1
            )
        
        return circuits
    
    def _calculate_baseline_mood(self) -> float:
        """Calcula humor baseline basado en personalidad"""
        # Neuroticism alto = baseline más negativo
        # Extraversion alto = baseline más positivo
        baseline = (self.personality.get('extraversion', 0.5) - 
                   self.personality.get('neuroticism', 0.5))
        return np.clip(baseline, -1.0, 1.0)
    
    def activate_circuit(self, emotion_name: str, intensity: float, trigger: str = ""):
        """Activa un circuito emocional específico"""
        if emotion_name not in self.circuits:
            return
        
        circuit = self.circuits[emotion_name]
        
        # Modular intensidad por personalidad
        if emotion_name in ["miedo", "tristeza", "desesperacion"]:
            intensity *= (0.5 + self.personality.get('neuroticism', 0.5) * 0.5)
        
        # Activar circuito
        circuit.intensity = min(1.0, circuit.intensity + intensity)
        circuit.onset_time = time.time()
        
        # Registrar en historia
        self.emotion_history.append({
            'emotion': emotion_name,
            'intensity': intensity,
            'trigger': trigger,
            'timestamp': datetime.now()
        })
        
        # Limitar historia
        if len(self.emotion_history) > 100:
            self.emotion_history = self.emotion_history[-50:]
        
        # Actualizar estado circumplex
        self._update_circumplex_state()
    
    def stimulate_emotion(self, emotion: str, intensity: float):
        """Compatibilidad con interfaz antigua"""
        self.activate_circuit(emotion, intensity)
    
    def _update_circumplex_state(self):
        """Actualiza el estado emocional en el espacio circumplex (valence x arousal)"""
        total_valence = 0.0
        total_arousal = 0.0
        total_weight = 0.0
        
        for circuit in self.circuits.values():
            if circuit.intensity > 0:
                weight = circuit.intensity
                total_valence += circuit.valence * weight
                total_arousal += circuit.arousal * weight
                total_weight += weight
        
        if total_weight > 0:
            self.current_valence = total_valence / total_weight
            self.current_arousal = total_arousal / total_weight
        else:
            # Sin emociones activas, regresar a baseline
            self.current_valence = self.baseline_mood
            self.current_arousal = 0.5
    
    def update_state(self, delta_time: float = 1.0):
        """Actualiza todos los circuitos emocionales (decaimiento temporal)"""
        for circuit in self.circuits.values():
            circuit.update(delta_time)
        
        self._update_circumplex_state()
    
    def get_emotion_intensity(self, emotion: str) -> float:
        """Obtiene intensidad actual de una emoción específica"""
        if emotion in self.circuits:
            return self.circuits[emotion].intensity
        return 0.0
    
    def get_emotional_state(self) -> Dict[str, Any]:
        """Retorna estado emocional completo"""
        active_emotions = {
            name: circuit.intensity 
            for name, circuit in self.circuits.items() 
            if circuit.intensity > 0.1
        }
        
        dominant_emotion = None
        if active_emotions:
            dominant_emotion = max(active_emotions, key=active_emotions.get)
        
        return {
            'dominant_emotion': dominant_emotion,
            'valence': self.current_valence,
            'arousal': self.current_arousal,
            'active_emotions': active_emotions,
            'mood_category': self._categorize_mood(),
            'baseline_mood': self.baseline_mood
        }
    
    def _categorize_mood(self) -> str:
        """Categoriza el humor actual en cuadrantes del circumplex"""
        if self.current_valence > 0.3:
            if self.current_arousal > 0.6:
                return "excited"  # Alta valencia, alto arousal
            else:
                return "content"  # Alta valencia, bajo arousal
        elif self.current_valence < -0.3:
            if self.current_arousal > 0.6:
                return "distressed"  # Baja valencia, alto arousal
            else:
                return "depressed"  # Baja valencia, bajo arousal
        else:
            return "neutral"
    
    def get_neurochemical_profile(self) -> Dict[str, float]:
        """Calcula perfil neuroquímico agregado de todas las emociones activas"""
        profile = {
            'dopamine': 0.0,
            'serotonin': 0.0,
            'norepinephrine': 0.0,
            'cortisol': 0.0,
            'oxytocin': 0.0,
            'adrenaline': 0.0
        }
        
        for circuit in self.circuits.values():
            if circuit.intensity > 0:
                # Neurotransmisores
                for nt, effect in circuit.neurotransmitters.items():
                    if nt in profile:
                        profile[nt] += effect * circuit.intensity
                
                # Hormonas
                for hormone, effect in circuit.hormones.items():
                    if hormone in profile:
                        profile[hormone] += effect * circuit.intensity
        
        # Normalizar
        for key in profile:
            profile[key] = np.clip(profile[key], 0.0, 1.0)
        
        return profile
    
    def regulate_emotion(self, strategy: str = "suppression"):
        """Aplica estrategia de regulación emocional"""
        if strategy == "suppression":
            # Reducir intensidad de emociones activas
            for circuit in self.circuits.values():
                circuit.intensity *= (1 - self.regulation_strength * 0.3)
        
        elif strategy == "reappraisal":
            # Cambiar valencia de emociones activas
            for circuit in self.circuits.values():
                if circuit.intensity > 0.5:
                    # Mover hacia valencia más neutral
                    circuit.valence *= 0.8
    
    def blend_emotions(self) -> Optional[str]:
        """Detecta mezclas emocionales complejas"""
        active = [(name, c.intensity) for name, c in self.circuits.items() if c.intensity > 0.4]
        
        if len(active) >= 2:
            # Detectar mezclas conocidas
            active_names = [name for name, _ in active]
            
            if "amor" in active_names and "miedo" in active_names:
                return "anxious_attachment"
            elif "alegria" in active_names and "tristeza" in active_names:
                return "bittersweet"
            elif "orgullo" in active_names and "verguenza" in active_names:
                return "humility"
        
        return None


# ==================== MÉTODOS DE COMPATIBILIDAD ====================

def get_all_emotions() -> List[str]:
    """Retorna lista de todas las emociones disponibles"""
    return (
        [getattr(BasicEmotions, attr) for attr in dir(BasicEmotions) if not attr.startswith('_')] +
        [getattr(SocialEmotions, attr) for attr in dir(SocialEmotions) if not attr.startswith('_')] +
        [getattr(ComplexEmotions, attr) for attr in dir(ComplexEmotions) if not attr.startswith('_')]
    )
