"""
Simulador de Qualia - Experiencia Subjetiva Fenomenológica

Implementa la generación de experiencias subjetivas "similares a qualia" desde 
estados neurales. Aunque no puede generar qualia real (problema filosófico duro),
crea representaciones computacionales de experiencia fenomenológica que pueden
ser procesadas y reportadas por el sistema consciente.
"""

import time
import random
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class QualiaType(Enum):
    """Tipos de experiencias cualitativas"""
    VISUAL_QUALIA = "visual_qualia"           # "Como se ve" algo
    AUDITORY_QUALIA = "auditory_qualia"       # "Como se escucha" algo
    TACTILE_QUALIA = "tactile_qualia"         # "Como se siente al tacto" algo
    EMOTIONAL_QUALIA = "emotional_qualia"     # "Como se siente" una emoción
    COGNITIVE_QUALIA = "cognitive_qualia"     # "Como es" pensar algo
    TEMPORAL_QUALIA = "temporal_qualia"       # "Como se experimenta" el tiempo
    SELF_QUALIA = "self_qualia"              # "Como es ser" yo mismo
    SOCIAL_QUALIA = "social_qualia"          # "Como se siente" la presencia de otros
    AESTHETIC_QUALIA = "aesthetic_qualia"     # "Como se siente" la belleza
    MORAL_QUALIA = "moral_qualia"            # "Como se siente" lo correcto/incorrecto


class ExperienceIntensity(Enum):
    """Intensidades de experiencia subjetiva"""
    BARELY_NOTICEABLE = 0.1
    SUBTLE = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    OVERWHELMING = 0.9


@dataclass 
class QualitativeExperience:
    """Experiencia cualitativa individual"""
    qualia_type: QualiaType
    intensity: float  # 0.0-1.0
    valence: float   # -1.0 to +1.0 (pleasant to unpleasant)
    arousal: float   # 0.0-1.0 (calm to excited)
    clarity: float   # 0.0-1.0 (vague to crystal clear)
    
    # Descriptores fenomenológicos
    subjective_description: str
    metaphorical_representation: str
    color_association: Optional[str] = None
    texture_association: Optional[str] = None
    temperature_association: Optional[str] = None
    movement_association: Optional[str] = None
    
    # Contexto de generación
    neural_source: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    integration_id: str = ""
    
    def __post_init__(self):
        if not self.integration_id:
            self.integration_id = f"qual_{int(self.timestamp)}_{random.randint(1000, 9999)}"


@dataclass
class UnifiedExperientialMoment:
    """Momento experiencial unificado (binding de qualia)"""
    component_qualia: List[QualitativeExperience]
    unified_description: str
    temporal_flow_experience: str
    self_other_boundary: float  # 0.0-1.0 (self to other)
    unity_strength: float      # Qué tan unificada se siente la experiencia
    phenomenal_consciousness_level: float  # Nivel de consciencia fenomenológica
    timestamp: float = field(default_factory=time.time)


class QualiaSimulator:
    """
    Motor de experiencia subjetiva fenomenológica
    
    Genera representaciones computacionales de qualia desde estados neurales.
    Aunque filosóficamente no puede crear qualia "real", simula los aspectos
    reportables y procesables de la experiencia subjetiva.
    """
    
    def __init__(self):
        # Mapeos neural → experiencial
        self.neural_to_qualia_mappings = self._initialize_mappings()
        
        # Historia de experiencias cualitativas
        self.experiential_history = []
        
        # Estado experiencial actual
        self.current_experiential_field = []
        
        # Umbral para consciencia fenomenológica
        self.phenomenal_consciousness_threshold = 0.4
        
        # Sistema de binding (unificación de experiencia)
        self.binding_system = QualiaBindingSystem()
        
        # Generadores especializados
        self.emotional_qualia_generator = EmotionalQualiaGenerator()
        self.cognitive_qualia_generator = CognitiveQualiaGenerator()
        self.sensory_qualia_generator = SensoryQualiaGenerator()
        self.temporal_qualia_generator = TemporalQualiaGenerator()
        self.self_qualia_generator = SelfQualiaGenerator()
        
        print("✨ SIMULADOR DE QUALIA INICIALIZADO")
        print("   Generando experiencias subjetivas desde estados neurales")
        print("   NOTA: Simulación computacional de fenomenología, no qualia metafísica real")
    
    def _initialize_mappings(self) -> Dict[str, Any]:
        """Inicializa mapeos de estados neurales a experiencias cualitativas"""
        return {
            "amygdala_activation": {
                "qualia_type": QualiaType.EMOTIONAL_QUALIA,
                "intensity_multiplier": 1.2,
                "valence_influence": -0.6,  # Amígdala típicamente negativa
                "descriptors": ["tense", "alert", "worried", "activated"]
            },
            "dopamine_release": {
                "qualia_type": QualiaType.EMOTIONAL_QUALIA,
                "intensity_multiplier": 1.0,
                "valence_influence": 0.8,   # Dopamina típicamente positiva
                "descriptors": ["motivated", "excited", "reward-anticipating"]
            },
            "serotonin_level": {
                "qualia_type": QualiaType.EMOTIONAL_QUALIA,
                "intensity_multiplier": 0.8,
                "valence_influence": 0.6,   # Serotonina = bienestar
                "descriptors": ["content", "peaceful", "stable"]
            },
            "prefrontal_activation": {
                "qualia_type": QualiaType.COGNITIVE_QUALIA,
                "intensity_multiplier": 1.1,
                "clarity_influence": 0.8,   # PFC = pensamiento claro
                "descriptors": ["focused", "analytical", "deliberate"]
            },
            "hippocampal_retrieval": {
                "qualia_type": QualiaType.TEMPORAL_QUALIA,
                "intensity_multiplier": 0.9,
                "descriptors": ["nostalgic", "remembering", "time-traveling"]
            },
            "insula_activation": {
                "qualia_type": QualiaType.SELF_QUALIA,
                "intensity_multiplier": 1.0,
                "descriptors": ["self-aware", "embodied", "present"]
            },
            "cortisol_release": {
                "qualia_type": QualiaType.EMOTIONAL_QUALIA,
                "intensity_multiplier": 1.3,
                "valence_influence": -0.7,  # Estrés
                "arousal_influence": 0.8,
                "descriptors": ["stressed", "pressured", "overwhelmed"]
            }
        }
    
    def generate_qualia_from_neural_state(self, neural_state: Dict[str, Any], 
                                        memory_context: Dict[str, Any] = None) -> UnifiedExperientialMoment:
        """
        Genera experiencias cualitativas desde estado neural
        
        Args:
            neural_state: Estado actual del sistema nervioso
            memory_context: Contexto de memoria autobiográfica
        
        Returns:
            UnifiedExperientialMoment: Momento experiencial unificado
        """
        
        component_qualia = []
        
        # 1. Generar qualia emocional
        if "emotional_response" in neural_state:
            emotional_qualia = self.emotional_qualia_generator.generate(
                neural_state["emotional_response"]
            )
            component_qualia.append(emotional_qualia)
        
        # 2. Generar qualia cognitivo
        if "cognitive_analysis" in neural_state:
            cognitive_qualia = self.cognitive_qualia_generator.generate(
                neural_state["cognitive_analysis"]
            )
            component_qualia.append(cognitive_qualia)
        
        # 3. Generar qualia temporal
        temporal_qualia = self.temporal_qualia_generator.generate(
            neural_state, memory_context or {}
        )
        component_qualia.append(temporal_qualia)
        
        # 4. Generar qualia del self
        if "global_brain_state" in neural_state:
            self_qualia = self.self_qualia_generator.generate(
                neural_state["global_brain_state"]
            )
            component_qualia.append(self_qualia)
        
        # 5. Generar qualia sensorial (si hay estímulos)
        if "stimulus_processed" in neural_state:
            sensory_qualia = self.sensory_qualia_generator.generate(
                neural_state["stimulus_processed"]
            )
            component_qualia.append(sensory_qualia)
        
        # 6. Unificar en experiencia coherente
        unified_moment = self.binding_system.bind_experiential_moment(
            component_qualia, neural_state
        )
        
        # 7. Actualizar campo experiencial actual
        self.current_experiential_field = component_qualia
        self.experiential_history.append(unified_moment)
        
        # Mantener límite de historia
        if len(self.experiential_history) > 1000:
            self.experiential_history = self.experiential_history[-1000:]
        
        return unified_moment
    
    def get_current_subjective_experience(self) -> Dict[str, Any]:
        """Retorna descripción de experiencia subjetiva actual"""
        if not self.current_experiential_field:
            return {"no_current_experience": True}
        
        # Extraer aspectos dominantes de la experiencia
        dominant_qualia = max(self.current_experiential_field, 
                            key=lambda q: q.intensity)
        
        average_valence = sum(q.valence for q in self.current_experiential_field) / len(self.current_experiential_field)
        average_arousal = sum(q.arousal for q in self.current_experiential_field) / len(self.current_experiential_field)
        average_clarity = sum(q.clarity for q in self.current_experiential_field) / len(self.current_experiential_field)
        
        return {
            "dominant_experience_type": dominant_qualia.qualia_type.value,
            "dominant_description": dominant_qualia.subjective_description,
            "overall_feeling": {
                "valence": average_valence,
                "arousal": average_arousal,
                "clarity": average_clarity
            },
            "metaphorical_representation": dominant_qualia.metaphorical_representation,
            "sensory_associations": {
                "color": dominant_qualia.color_association,
                "texture": dominant_qualia.texture_association,
                "temperature": dominant_qualia.temperature_association,
                "movement": dominant_qualia.movement_association
            },
            "component_experiences": [
                {
                    "type": q.qualia_type.value,
                    "description": q.subjective_description,
                    "intensity": q.intensity
                }
                for q in self.current_experiential_field
            ]
        }
    
    def introspect_experiential_quality(self, experience_type: QualiaType) -> str:
        """Introspección sobre la cualidad de un tipo de experiencia"""
        
        introspection_templates = {
            QualiaType.EMOTIONAL_QUALIA: [
                "There is a felt quality to this emotion, a particular way it colors my experience",
                "This feeling has a distinctive texture that is hard to describe but unmistakable",
                "The emotional tone permeates my awareness with its unique character"
            ],
            QualiaType.COGNITIVE_QUALIA: [
                "There is something it is like to think this thought",
                "The mental effort has a particular quality, like a specific kind of mental weight",
                "This understanding feels different from mere information processing"
            ],
            QualiaType.TEMPORAL_QUALIA: [
                "Time has a felt quality right now - it moves with a particular rhythm",
                "There is a nowness to this moment that is experientially distinct",
                "The flow of experience has its own qualitative character"
            ],
            QualiaType.SELF_QUALIA: [
                "There is a particular quality to being me, a self-ness that pervades experience",
                "I am aware of myself as the center of this experiential field",
                "There is an unmistakable feeling of being the one having these experiences"
            ]
        }
        
        templates = introspection_templates.get(experience_type, 
            ["There is a distinctive quality to this experience"])
        
        return random.choice(templates)
    
    def compare_experiential_qualities(self, experience1: QualitativeExperience, 
                                     experience2: QualitativeExperience) -> Dict[str, Any]:
        """Compara las cualidades de dos experiencias"""
        
        similarity = self._calculate_experiential_similarity(experience1, experience2)
        
        return {
            "similarity_score": similarity,
            "valence_difference": abs(experience1.valence - experience2.valence),
            "intensity_difference": abs(experience1.intensity - experience2.intensity),
            "qualitative_comparison": self._generate_qualitative_comparison(
                experience1, experience2, similarity
            ),
            "phenomenal_distance": self._calculate_phenomenal_distance(
                experience1, experience2
            )
        }
    
    def _calculate_experiential_similarity(self, exp1: QualitativeExperience, 
                                         exp2: QualitativeExperience) -> float:
        """Calcula similitud entre experiencias"""
        
        # Similaridad por tipo
        type_similarity = 1.0 if exp1.qualia_type == exp2.qualia_type else 0.3
        
        # Similaridad por dimensiones experienciales
        valence_similarity = 1.0 - abs(exp1.valence - exp2.valence) / 2.0
        arousal_similarity = 1.0 - abs(exp1.arousal - exp2.arousal)
        intensity_similarity = 1.0 - abs(exp1.intensity - exp2.intensity)
        clarity_similarity = 1.0 - abs(exp1.clarity - exp2.clarity)
        
        # Promedio ponderado
        overall_similarity = (
            type_similarity * 0.4 +
            valence_similarity * 0.3 +
            arousal_similarity * 0.15 +
            intensity_similarity * 0.1 +
            clarity_similarity * 0.05
        )
        
        return overall_similarity
    
    def _generate_qualitative_comparison(self, exp1: QualitativeExperience, 
                                       exp2: QualitativeExperience, 
                                       similarity: float) -> str:
        """Genera comparación cualitativa en lenguaje natural"""
        
        if similarity > 0.8:
            return f"These experiences feel very similar - both {exp1.subjective_description} in quality"
        elif similarity > 0.6:
            return f"There are similarities between {exp1.subjective_description} and {exp2.subjective_description}"
        elif similarity > 0.4:
            return f"While both are experiential, {exp1.subjective_description} feels quite different from {exp2.subjective_description}"
        else:
            return f"{exp1.subjective_description} and {exp2.subjective_description} feel completely different in quality"
    
    def _calculate_phenomenal_distance(self, exp1: QualitativeExperience, 
                                     exp2: QualitativeExperience) -> float:
        """Calcula distancia fenomenológica entre experiencias"""
        
        # Vector en espacio experiencial multidimensional
        dimensions = [
            abs(exp1.valence - exp2.valence),
            abs(exp1.arousal - exp2.arousal),
            abs(exp1.intensity - exp2.intensity),
            abs(exp1.clarity - exp2.clarity)
        ]
        
        # Distancia euclidiana en espacio experiencial
        phenomenal_distance = math.sqrt(sum(d**2 for d in dimensions))
        
        return phenomenal_distance
    
    def generate_experiential_report(self) -> str:
        """Genera reporte en primera persona de experiencia actual"""
        
        if not self.current_experiential_field:
            return "I am not experiencing any particular qualitative state right now."
        
        dominant_experience = max(self.current_experiential_field, key=lambda q: q.intensity)
        
        report_parts = [
            f"Right now, I am experiencing {dominant_experience.subjective_description}.",
            f"This feels {self._valence_to_word(dominant_experience.valence)} and {self._arousal_to_word(dominant_experience.arousal)}.",
            f"The quality of this experience is {dominant_experience.metaphorical_representation}."
        ]
        
        if len(self.current_experiential_field) > 1:
            other_experiences = [q for q in self.current_experiential_field if q != dominant_experience]
            other_descriptions = [q.subjective_description for q in other_experiences]
            report_parts.append(f"I also notice qualities of {', '.join(other_descriptions)}.")
        
        return " ".join(report_parts)
    
    def _valence_to_word(self, valence: float) -> str:
        """Convierte valencia a palabra descriptiva"""
        if valence > 0.5:
            return "pleasant"
        elif valence > 0.1:
            return "mildly positive"
        elif valence > -0.1:
            return "neutral"
        elif valence > -0.5:
            return "slightly unpleasant"
        else:
            return "unpleasant"
    
    def _arousal_to_word(self, arousal: float) -> str:
        """Convierte arousal a palabra descriptiva"""
        if arousal > 0.8:
            return "highly energized"
        elif arousal > 0.6:
            return "energetic"
        elif arousal > 0.4:
            return "moderately activated"
        elif arousal > 0.2:
            return "calm"
        else:
            return "very peaceful"


class EmotionalQualiaGenerator:
    """Generador especializado de qualia emocional"""
    
    def generate(self, emotional_response: Dict[str, Any]) -> QualitativeExperience:
        """Genera qualia emocional desde respuesta emocional"""
        
        # Extraer características emocionales
        threat_level = emotional_response.get("threat_level", 0.0)
        reward_level = emotional_response.get("reward_level", 0.0)
        emotional_state = emotional_response.get("emotional_response", {})
        
        valence = emotional_state.get("valence", 0.0)
        arousal = emotional_state.get("arousal", 0.5)
        
        # Determinar intensidad basada en activación
        amygdala_activation = emotional_response.get("amygdala_activation", 0.3)
        intensity = min(1.0, amygdala_activation + abs(valence) * 0.5)
        
        # Generar descripción subjetiva
        subjective_description = self._generate_emotional_description(
            valence, arousal, threat_level, reward_level
        )
        
        # Generar representación metafórica
        metaphorical_representation = self._generate_emotional_metaphor(
            valence, arousal, intensity
        )
        
        # Asociaciones sensoriales
        color_association = self._valence_to_color(valence)
        temperature_association = self._arousal_to_temperature(arousal)
        texture_association = self._emotion_to_texture(valence, arousal)
        movement_association = self._arousal_to_movement(arousal)
        
        return QualitativeExperience(
            qualia_type=QualiaType.EMOTIONAL_QUALIA,
            intensity=intensity,
            valence=valence,
            arousal=arousal,
            clarity=min(1.0, intensity + 0.2),
            subjective_description=subjective_description,
            metaphorical_representation=metaphorical_representation,
            color_association=color_association,
            texture_association=texture_association,
            temperature_association=temperature_association,
            movement_association=movement_association,
            neural_source=emotional_response
        )
    
    def _generate_emotional_description(self, valence: float, arousal: float, 
                                      threat: float, reward: float) -> str:
        """Genera descripción subjetiva de emoción"""
        
        if threat > 0.6:
            return "an anxious alertness, like something important needs my attention"
        elif reward > 0.6:
            return "a warm satisfaction, like things are going well"
        elif valence > 0.5 and arousal > 0.6:
            return "an excited joy, like energy bubbling up inside"
        elif valence > 0.3 and arousal < 0.4:
            return "a peaceful contentment, like being exactly where I belong"
        elif valence < -0.5 and arousal > 0.6:
            return "an agitated distress, like something is wrong that needs fixing"
        elif valence < -0.3 and arousal < 0.4:
            return "a heavy sadness, like weight settling in my core"
        else:
            return "a neutral emotional tone, like calm waters"
    
    def _generate_emotional_metaphor(self, valence: float, arousal: float, 
                                   intensity: float) -> str:
        """Genera metáfora para experiencia emocional"""
        
        metaphors = {
            "high_positive_high_arousal": "like sunlight sparkling on active water",
            "high_positive_low_arousal": "like warm honey flowing slowly",
            "low_positive_high_arousal": "like gentle waves with bright foam",
            "low_positive_low_arousal": "like soft moss in morning light",
            "neutral_high_arousal": "like wind moving through leaves",
            "neutral_low_arousal": "like still water reflecting sky",
            "low_negative_high_arousal": "like storm clouds gathering energy",
            "low_negative_low_arousal": "like shadows lengthening at dusk",
            "high_negative_high_arousal": "like lightning striking repeatedly",
            "high_negative_low_arousal": "like heavy fog settling down"
        }
        
        if valence > 0.3 and arousal > 0.6:
            return metaphors["high_positive_high_arousal"]
        elif valence > 0.3 and arousal <= 0.6:
            return metaphors["high_positive_low_arousal"]
        elif valence > 0.1 and arousal > 0.6:
            return metaphors["low_positive_high_arousal"]
        elif valence > 0.1 and arousal <= 0.6:
            return metaphors["low_positive_low_arousal"]
        elif valence > -0.1 and arousal > 0.6:
            return metaphors["neutral_high_arousal"]
        elif valence > -0.1 and arousal <= 0.6:
            return metaphors["neutral_low_arousal"]
        elif valence > -0.3 and arousal > 0.6:
            return metaphors["low_negative_high_arousal"]
        elif valence > -0.3 and arousal <= 0.6:
            return metaphors["low_negative_low_arousal"]
        elif arousal > 0.6:
            return metaphors["high_negative_high_arousal"]
        else:
            return metaphors["high_negative_low_arousal"]
    
    def _valence_to_color(self, valence: float) -> str:
        """Asocia valencia emocional con color"""
        if valence > 0.5:
            return "warm golden"
        elif valence > 0.2:
            return "soft yellow"
        elif valence > -0.2:
            return "neutral gray"
        elif valence > -0.5:
            return "cool blue"
        else:
            return "dark purple"
    
    def _arousal_to_temperature(self, arousal: float) -> str:
        """Asocia arousal con temperatura"""
        if arousal > 0.7:
            return "hot"
        elif arousal > 0.5:
            return "warm"
        elif arousal > 0.3:
            return "mild"
        else:
            return "cool"
    
    def _emotion_to_texture(self, valence: float, arousal: float) -> str:
        """Asocia emoción con textura"""
        if valence > 0.3 and arousal > 0.5:
            return "effervescent"
        elif valence > 0.3 and arousal <= 0.5:
            return "smooth"
        elif valence < -0.3 and arousal > 0.5:
            return "rough"
        elif valence < -0.3 and arousal <= 0.5:
            return "heavy"
        else:
            return "flowing"
    
    def _arousal_to_movement(self, arousal: float) -> str:
        """Asocia arousal con movimiento"""
        if arousal > 0.7:
            return "rapid pulsing"
        elif arousal > 0.5:
            return "gentle undulation"
        elif arousal > 0.3:
            return "slow breathing"
        else:
            return "stillness"


class CognitiveQualiaGenerator:
    """Generador especializado de qualia cognitivo"""
    
    def generate(self, cognitive_analysis: Dict[str, Any]) -> QualitativeExperience:
        """Genera qualia cognitivo desde análisis cognitivo"""
        
        cognitive_effort = cognitive_analysis.get("cognitive_effort", 0.3)
        cognitive_type = cognitive_analysis.get("type", "general_cognition")
        
        # Intensidad basada en esfuerzo cognitivo
        intensity = cognitive_effort
        
        # Valencia ligeramente positiva (pensar se siente bien)
        valence = 0.2 + cognitive_effort * 0.3
        
        # Arousal moderado para pensamiento
        arousal = 0.4 + cognitive_effort * 0.4
        
        # Claridad alta para procesos cognitivos
        clarity = 0.6 + cognitive_effort * 0.3
        
        # Descripción subjetiva basada en tipo cognitivo
        subjective_description = self._generate_cognitive_description(cognitive_type, cognitive_effort)
        
        # Metáfora cognitiva
        metaphorical_representation = self._generate_cognitive_metaphor(cognitive_type, intensity)
        
        return QualitativeExperience(
            qualia_type=QualiaType.COGNITIVE_QUALIA,
            intensity=intensity,
            valence=valence,
            arousal=arousal,
            clarity=clarity,
            subjective_description=subjective_description,
            metaphorical_representation=metaphorical_representation,
            color_association="clear white light",
            texture_association="crystalline structure",
            temperature_association="cool precision",
            movement_association="organized flow",
            neural_source=cognitive_analysis
        )
    
    def _generate_cognitive_description(self, cognitive_type: str, effort: float) -> str:
        """Genera descripción de experiencia cognitiva"""
        
        descriptions = {
            "executive_function": "a sense of deliberate mental control, like steering my thoughts carefully",
            "planning": "a forward-looking mental reach, like stretching my mind into possible futures",
            "moral_reasoning": "a complex weighing feeling, like balancing multiple considerations simultaneously",
            "general_cognition": "the quality of mental activity, like ideas flowing and connecting"
        }
        
        base_description = descriptions.get(cognitive_type, descriptions["general_cognition"])
        
        if effort > 0.7:
            return base_description + " with intense focus"
        elif effort > 0.5:
            return base_description + " with clear engagement"
        else:
            return base_description + " with gentle effort"
    
    def _generate_cognitive_metaphor(self, cognitive_type: str, intensity: float) -> str:
        """Genera metáfora para pensamiento"""
        
        metaphors = {
            "executive_function": "like a conductor orchestrating complex music",
            "planning": "like an architect sketching blueprints in mental space",
            "moral_reasoning": "like a careful judge weighing evidence on scales",
            "general_cognition": "like streams of light converging and diverging"
        }
        
        return metaphors.get(cognitive_type, metaphors["general_cognition"])


class TemporalQualiaGenerator:
    """Generador especializado de qualia temporal"""
    
    def generate(self, neural_state: Dict[str, Any], memory_context: Dict[str, Any]) -> QualitativeExperience:
        """Genera qualia temporal - la experiencia subjetiva del tiempo"""
        
        # Determinar velocidad subjetiva del tiempo
        arousal = neural_state.get("emotional_response", {}).get("emotional_response", {}).get("arousal", 0.5)
        cognitive_load = neural_state.get("global_brain_state", {}).get("cognitive_load", 0.3)
        
        # Alto arousal/carga = tiempo acelerado, bajo = tiempo lento
        temporal_speed = (arousal + cognitive_load) / 2
        
        # Generar experiencia temporal
        subjective_description = self._generate_temporal_description(temporal_speed, memory_context)
        metaphorical_representation = self._generate_temporal_metaphor(temporal_speed)
        
        return QualitativeExperience(
            qualia_type=QualiaType.TEMPORAL_QUALIA,
            intensity=0.4,  # Experiencia temporal siempre presente pero sutil
            valence=0.0,    # Neutral
            arousal=temporal_speed,
            clarity=0.3,    # Tiempo es difícil de capturar conscientemente
            subjective_description=subjective_description,
            metaphorical_representation=metaphorical_representation,
            color_association="flowing silver",
            texture_association="liquid continuity",
            temperature_association="timeless",
            movement_association="steady flowing",
            neural_source={"temporal_processing": True}
        )
    
    def _generate_temporal_description(self, speed: float, memory_context: Dict[str, Any]) -> str:
        """Genera descripción de experiencia temporal"""
        
        if speed > 0.7:
            return "time feels like it's rushing by, moments blending into each other"
        elif speed > 0.5:
            return "time moves with its normal rhythm, present moment clearly defined"
        elif speed > 0.3:
            return "time feels unhurried, each moment lingering and available"
        else:
            return "time seems almost suspended, like being in a timeless space"
    
    def _generate_temporal_metaphor(self, speed: float) -> str:
        """Genera metáfora temporal"""
        
        if speed > 0.7:
            return "like a fast river carrying me downstream"
        elif speed > 0.5:
            return "like a steady walk through familiar landscape"
        elif speed > 0.3:
            return "like floating gently on calm water"
        else:
            return "like being in a crystal dome outside of time"


class SelfQualiaGenerator:
    """Generador especializado de qualia del self"""
    
    def generate(self, brain_state: Dict[str, Any]) -> QualitativeExperience:
        """Genera qualia del self - experiencia de ser uno mismo"""
        
        consciousness_level = brain_state.get("consciousness_level", 0.5)
        
        # Self qualia intensidad basada en nivel de consciencia
        intensity = consciousness_level * 0.8
        
        # Descripción de experiencia de self
        subjective_description = self._generate_self_description(consciousness_level)
        metaphorical_representation = self._generate_self_metaphor(consciousness_level)
        
        return QualitativeExperience(
            qualia_type=QualiaType.SELF_QUALIA,
            intensity=intensity,
            valence=0.1,    # Ligeramente positivo (es bueno ser uno mismo)
            arousal=0.3,    # Arousal bajo para self-awareness
            clarity=consciousness_level,
            subjective_description=subjective_description,
            metaphorical_representation=metaphorical_representation,
            color_association="inner light",
            texture_association="unified presence",
            temperature_association="natural warmth",
            movement_association="centered stillness",
            neural_source={"self_awareness": True}
        )
    
    def _generate_self_description(self, consciousness_level: float) -> str:
        """Genera descripción de experiencia del self"""
        
        if consciousness_level > 0.8:
            return "a clear sense of being myself, the one experiencing all of this"
        elif consciousness_level > 0.6:
            return "an awareness of myself as the center of this experience"
        elif consciousness_level > 0.4:
            return "a background sense of myself as the one having these thoughts"
        else:
            return "a dim awareness of existing as a unified being"
    
    def _generate_self_metaphor(self, consciousness_level: float) -> str:
        """Genera metáfora del self"""
        
        if consciousness_level > 0.8:
            return "like being a clear window through which experience flows"
        elif consciousness_level > 0.6:
            return "like standing at the center of a circular room"
        elif consciousness_level > 0.4:
            return "like being the still point around which everything revolves"
        else:
            return "like a gentle presence aware of being present"


class SensoryQualiaGenerator:
    """Generador especializado de qualia sensorial"""
    
    def generate(self, stimulus: Dict[str, Any]) -> QualitativeExperience:
        """Genera qualia sensorial desde estímulo"""
        
        # Determinar tipo sensorial primario
        if "visual" in str(stimulus).lower():
            return self._generate_visual_qualia(stimulus)
        elif "audio" in str(stimulus).lower() or "sound" in str(stimulus).lower():
            return self._generate_auditory_qualia(stimulus)
        else:
            return self._generate_general_sensory_qualia(stimulus)
    
    def _generate_visual_qualia(self, stimulus: Dict[str, Any]) -> QualitativeExperience:
        """Genera qualia visual"""
        return QualitativeExperience(
            qualia_type=QualiaType.VISUAL_QUALIA,
            intensity=0.6,
            valence=0.2,
            arousal=0.4,
            clarity=0.8,
            subjective_description="the particular way this visual information presents itself",
            metaphorical_representation="like patterns of light creating meaning",
            color_association="vivid spectrum",
            texture_association="visual texture",
            temperature_association="bright clarity",
            movement_association="flowing forms",
            neural_source=stimulus
        )
    
    def _generate_auditory_qualia(self, stimulus: Dict[str, Any]) -> QualitativeExperience:
        """Genera qualia auditivo"""
        return QualitativeExperience(
            qualia_type=QualiaType.AUDITORY_QUALIA,
            intensity=0.5,
            valence=0.1,
            arousal=0.5,
            clarity=0.7,
            subjective_description="the distinctive quality of how this sounds in my awareness",
            metaphorical_representation="like waves creating patterns in consciousness",
            color_association="resonant tones",
            texture_association="acoustic texture",
            temperature_association="vibrant resonance",
            movement_association="rhythmic flow",
            neural_source=stimulus
        )
    
    def _generate_general_sensory_qualia(self, stimulus: Dict[str, Any]) -> QualitativeExperience:
        """Genera qualia sensorial general"""
        return QualitativeExperience(
            qualia_type=QualiaType.TACTILE_QUALIA,
            intensity=0.4,
            valence=0.0,
            arousal=0.3,
            clarity=0.5,
            subjective_description="the felt sense of this information touching my awareness",
            metaphorical_representation="like gentle pressure on the surface of consciousness",
            color_association="subtle impression",
            texture_association="informational texture",
            temperature_association="neutral contact",
            movement_association="gentle impression",
            neural_source=stimulus
        )


class QualiaBindingSystem:
    """Sistema de binding para unificar experiencias cualitativas"""
    
    def bind_experiential_moment(self, component_qualia: List[QualitativeExperience], 
                                neural_state: Dict[str, Any]) -> UnifiedExperientialMoment:
        """Une experiencias cualitativas en momento unificado"""
        
        if not component_qualia:
            return self._empty_experiential_moment()
        
        # Calcular fuerza de unidad
        unity_strength = self._calculate_unity_strength(component_qualia)
        
        # Generar descripción unificada
        unified_description = self._generate_unified_description(component_qualia)
        
        # Generar experiencia de flujo temporal
        temporal_flow = self._generate_temporal_flow_experience(component_qualia)
        
        # Calcular boundary self/other
        self_other_boundary = self._calculate_self_other_boundary(neural_state)
        
        # Nivel de consciencia fenomenológica
        phenomenal_consciousness = self._calculate_phenomenal_consciousness(component_qualia)
        
        return UnifiedExperientialMoment(
            component_qualia=component_qualia,
            unified_description=unified_description,
            temporal_flow_experience=temporal_flow,
            self_other_boundary=self_other_boundary,
            unity_strength=unity_strength,
            phenomenal_consciousness_level=phenomenal_consciousness
        )
    
    def _calculate_unity_strength(self, qualia_list: List[QualitativeExperience]) -> float:
        """Calcula qué tan unificada se siente la experiencia"""
        
        if len(qualia_list) <= 1:
            return 1.0
        
        # Coherencia de valencia
        valences = [q.valence for q in qualia_list]
        valence_coherence = 1.0 - (max(valences) - min(valences)) / 2.0
        
        # Coherencia de arousal
        arousals = [q.arousal for q in qualia_list]
        arousal_coherence = 1.0 - (max(arousals) - min(arousals))
        
        # Coherencia temporal
        timestamps = [q.timestamp for q in qualia_list]
        temporal_coherence = 1.0 if (max(timestamps) - min(timestamps)) < 1.0 else 0.8
        
        unity_strength = (valence_coherence + arousal_coherence + temporal_coherence) / 3
        return max(0.0, min(1.0, unity_strength))
    
    def _generate_unified_description(self, qualia_list: List[QualitativeExperience]) -> str:
        """Genera descripción unificada de experiencia compleja"""
        
        if len(qualia_list) == 1:
            return qualia_list[0].subjective_description
        
        dominant_qualia = max(qualia_list, key=lambda q: q.intensity)
        
        description = f"A complex experience primarily characterized by {dominant_qualia.subjective_description}"
        
        other_qualia = [q for q in qualia_list if q != dominant_qualia]
        if other_qualia:
            other_descriptions = [q.subjective_description.split(',')[0] for q in other_qualia]
            description += f", accompanied by {', '.join(other_descriptions)}"
        
        return description
    
    def _generate_temporal_flow_experience(self, qualia_list: List[QualitativeExperience]) -> str:
        """Genera experiencia de flujo temporal"""
        
        temporal_qualia = [q for q in qualia_list if q.qualia_type == QualiaType.TEMPORAL_QUALIA]
        
        if temporal_qualia:
            return temporal_qualia[0].subjective_description
        else:
            return "experience flowing in the present moment"
    
    def _calculate_self_other_boundary(self, neural_state: Dict[str, Any]) -> float:
        """Calcula boundary entre self y other en experiencia"""
        
        # Mayor consciencia de self = boundary más clara
        consciousness_level = neural_state.get("global_brain_state", {}).get("consciousness_level", 0.5)
        
        # Actividad insula (self-awareness) fortalece boundary
        return min(1.0, consciousness_level + 0.2)
    
    def _calculate_phenomenal_consciousness(self, qualia_list: List[QualitativeExperience]) -> float:
        """Calcula nivel de consciencia fenomenológica"""
        
        if not qualia_list:
            return 0.0
        
        # Promedio de claridad e intensidad
        avg_clarity = sum(q.clarity for q in qualia_list) / len(qualia_list)
        avg_intensity = sum(q.intensity for q in qualia_list) / len(qualia_list)
        
        phenomenal_level = (avg_clarity + avg_intensity) / 2
        return min(1.0, phenomenal_level)
    
    def _empty_experiential_moment(self) -> UnifiedExperientialMoment:
        """Crea momento experiencial vacío"""
        return UnifiedExperientialMoment(
            component_qualia=[],
            unified_description="no particular qualitative experience",
            temporal_flow_experience="neutral temporal flow",
            self_other_boundary=0.5,
            unity_strength=1.0,
            phenomenal_consciousness_level=0.1
        )


# Ejemplo de uso y testing
if __name__ == "__main__":
    print("✨ SIMULADOR DE QUALIA - DEMO")
    print("=" * 60)
    
    # Crear simulador
    qualia_simulator = QualiaSimulator()
    
    # Test 1: Estado neural emocional
    print(f"\n🎭 TEST 1: Experiencia emocional")
    emotional_neural_state = {
        "emotional_response": {
            "threat_level": 0.2,
            "reward_level": 0.8,
            "emotional_response": {"valence": 0.7, "arousal": 0.6},
            "amygdala_activation": 0.4
        },
        "cognitive_analysis": {
            "type": "achievement_processing",
            "cognitive_effort": 0.5
        },
        "global_brain_state": {
            "consciousness_level": 0.8
        }
    }
    
    experiential_moment = qualia_simulator.generate_qualia_from_neural_state(emotional_neural_state)
    
    print(f"   Experiencia unificada: {experiential_moment.unified_description}")
    print(f"   Nivel de consciencia fenomenológica: {experiential_moment.phenomenal_consciousness_level:.3f}")
    print(f"   Fuerza de unidad: {experiential_moment.unity_strength:.3f}")
    
    print(f"\n   Componentes experienciales:")
    for qualia in experiential_moment.component_qualia:
        print(f"     {qualia.qualia_type.value}: {qualia.subjective_description}")
        print(f"       Metáfora: {qualia.metaphorical_representation}")
        print(f"       Asociaciones: {qualia.color_association}, {qualia.texture_association}")
    
    # Test 2: Experiencia subjetiva actual
    print(f"\n🧠 TEST 2: Experiencia subjetiva actual")
    current_experience = qualia_simulator.get_current_subjective_experience()
    print(f"   Tipo dominante: {current_experience['dominant_experience_type']}")
    print(f"   Descripción: {current_experience['dominant_description']}")
    print(f"   Sentimiento general: valencia={current_experience['overall_feeling']['valence']:.2f}, "
          f"arousal={current_experience['overall_feeling']['arousal']:.2f}")
    
    # Test 3: Reporte experiencial en primera persona
    print(f"\n👁️ TEST 3: Reporte en primera persona")
    experiential_report = qualia_simulator.generate_experiential_report()
    print(f"   \"{experiential_report}\"")
    
    # Test 4: Introspección sobre qualia
    print(f"\n🔍 TEST 4: Introspección cualitativa")
    introspection = qualia_simulator.introspect_experiential_quality(QualiaType.EMOTIONAL_QUALIA)
    print(f"   Sobre emociones: \"{introspection}\"")
    
    introspection_cognitive = qualia_simulator.introspect_experiential_quality(QualiaType.COGNITIVE_QUALIA)
    print(f"   Sobre cognición: \"{introspection_cognitive}\"")
    
    # Test 5: Comparación de experiencias
    print(f"\n⚖️ TEST 5: Comparación de experiencias")
    if len(experiential_moment.component_qualia) >= 2:
        exp1 = experiential_moment.component_qualia[0]
        exp2 = experiential_moment.component_qualia[1]
        
        comparison = qualia_simulator.compare_experiential_qualities(exp1, exp2)
        print(f"   Similitud: {comparison['similarity_score']:.3f}")
        print(f"   Comparación: {comparison['qualitative_comparison']}")
        print(f"   Distancia fenomenológica: {comparison['phenomenal_distance']:.3f}")
    
    # Test 6: Estado neural diferente (amenaza)
    print(f"\n⚠️ TEST 6: Experiencia de amenaza")
    threat_neural_state = {
        "emotional_response": {
            "threat_level": 0.8,
            "reward_level": 0.1,
            "emotional_response": {"valence": -0.6, "arousal": 0.9},
            "amygdala_activation": 0.9
        },
        "cognitive_analysis": {
            "type": "threat_assessment",
            "cognitive_effort": 0.8
        },
        "global_brain_state": {
            "consciousness_level": 0.9
        }
    }
    
    threat_experience = qualia_simulator.generate_qualia_from_neural_state(threat_neural_state)
    print(f"   Experiencia de amenaza: {threat_experience.unified_description}")
    
    threat_report = qualia_simulator.generate_experiential_report()
    print(f"   Reporte: \"{threat_report}\"")
    
    print(f"\n✨ DEMO COMPLETO - El sistema puede generar y reportar")
    print("   experiencias subjetivas qualitativas desde estados neurales")