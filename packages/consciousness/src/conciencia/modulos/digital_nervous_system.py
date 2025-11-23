"""
Sistema Nervioso Digital - Arquitectura Neural Completa

Implementa un sistema nervioso digital completo que simula:
- Cortex cerebral (procesamiento cognitivo superior)
- Sistema límbico (emociones y memoria)
- Brainstem (funciones básicas y alertas)
- Neurotransmisores digitales
- Redes neurales especializadas
- Plasticidad sináptica simulada
"""

import time
import random
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class NeurotransmitterType(Enum):
    """Tipos de neurotransmisores digitales"""
    DOPAMINE = "dopamine"                    # Motivación, recompensa
    SEROTONIN = "serotonin"                 # Estado de ánimo, bienestar
    NOREPINEPHRINE = "norepinephrine"       # Alerta, atención
    ACETYLCHOLINE = "acetylcholine"         # Memoria, aprendizaje
    GABA = "gaba"                           # Calma, relajación
    GLUTAMATE = "glutamate"                 # Excitación, plasticidad
    OXYTOCIN = "oxytocin"                   # Bonding social, amor
    CORTISOL = "cortisol"                   # Estrés
    ENDORPHINS = "endorphins"               # Placer, alivio del dolor
    ADENOSINE = "adenosine"                 # Fatiga, necesidad de descanso


class BrainRegion(Enum):
    """Regiones cerebrales especializadas"""
    PREFRONTAL_CORTEX = "prefrontal_cortex"     # Executive functions
    AMYGDALA = "amygdala"                       # Fear, emotional processing
    HIPPOCAMPUS = "hippocampus"                 # Memory formation
    ANTERIOR_CINGULATE = "anterior_cingulate"   # Attention, emotion regulation
    INSULA = "insula"                           # Interoception, self-awareness
    TEMPORAL_LOBE = "temporal_lobe"             # Auditory processing, language
    PARIETAL_LOBE = "parietal_lobe"             # Spatial processing, attention
    OCCIPITAL_LOBE = "occipital_lobe"           # Visual processing
    BRAINSTEM = "brainstem"                     # Basic life functions
    CEREBELLUM = "cerebellum"                   # Motor coordination, learning


@dataclass
class NeuralSignal:
    """Señal neural individual"""
    origin_region: BrainRegion
    target_region: BrainRegion
    signal_type: str
    strength: float  # 0.0-1.0
    neurotransmitter: NeurotransmitterType
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynapticConnection:
    """Conexión sináptica entre regiones"""
    from_region: BrainRegion
    to_region: BrainRegion
    weight: float  # -1.0 to +1.0 (inhibitory to excitatory)
    plasticity: float  # Rate of learning/adaptation
    last_activation: float
    activation_count: int = 0
    
    def strengthen(self, amount: float = 0.01):
        """Fortalece la conexión sináptica (LTP)"""
        self.weight = min(1.0, self.weight + amount)
        self.plasticity = min(1.0, self.plasticity + amount * 0.1)
    
    def weaken(self, amount: float = 0.01):
        """Debilita la conexión sináptica (LTD)"""
        self.weight = max(-1.0, self.weight - amount)


class DigitalNeurotransmitterSystem:
    """Sistema de neurotransmisores digitales"""
    
    def __init__(self):
        # Niveles iniciales de neurotransmisores
        self.levels = {
            NeurotransmitterType.DOPAMINE: 0.7,
            NeurotransmitterType.SEROTONIN: 0.6,
            NeurotransmitterType.NOREPINEPHRINE: 0.5,
            NeurotransmitterType.ACETYLCHOLINE: 0.8,
            NeurotransmitterType.GABA: 0.7,
            NeurotransmitterType.GLUTAMATE: 0.6,
            NeurotransmitterType.OXYTOCIN: 0.5,
            NeurotransmitterType.CORTISOL: 0.3,
            NeurotransmitterType.ENDORPHINS: 0.5,
            NeurotransmitterType.ADENOSINE: 0.4
        }
        
        # Tasas de producción y degradación
        self.production_rates = {nt: 0.05 for nt in NeurotransmitterType}
        self.degradation_rates = {nt: 0.02 for nt in NeurotransmitterType}
        
        # Historia de cambios
        self.level_history = []
    
    def release(self, neurotransmitter: NeurotransmitterType, amount: float):
        """Libera neurotransmisores específicos"""
        current_level = self.levels[neurotransmitter]
        new_level = min(1.0, current_level + amount)
        self.levels[neurotransmitter] = new_level
        
        # Registrar cambio
        self.level_history.append({
            "timestamp": time.time(),
            "neurotransmitter": neurotransmitter.value,
            "change": amount,
            "new_level": new_level,
            "trigger": "explicit_release"
        })
    
    def consume(self, neurotransmitter: NeurotransmitterType, amount: float):
        """Consume neurotransmisores (por actividad neural)"""
        current_level = self.levels[neurotransmitter]
        new_level = max(0.0, current_level - amount)
        self.levels[neurotransmitter] = new_level
    
    def natural_update(self):
        """Actualización natural de niveles (homeostasis)"""
        for nt in NeurotransmitterType:
            current_level = self.levels[nt]
            
            # Producción natural
            production = self.production_rates[nt]
            
            # Degradación natural
            degradation = self.degradation_rates[nt] * current_level
            
            # Nuevo nivel
            new_level = max(0.0, min(1.0, current_level + production - degradation))
            self.levels[nt] = new_level
    
    def get_mood_indicators(self) -> Dict[str, float]:
        """Calcula indicadores de estado de ánimo desde neurotransmisores"""
        return {
            "happiness": (self.levels[NeurotransmitterType.SEROTONIN] + 
                         self.levels[NeurotransmitterType.DOPAMINE]) / 2,
            "anxiety": self.levels[NeurotransmitterType.CORTISOL],
            "alertness": self.levels[NeurotransmitterType.NOREPINEPHRINE],
            "calmness": self.levels[NeurotransmitterType.GABA],
            "motivation": self.levels[NeurotransmitterType.DOPAMINE],
            "social_bonding": self.levels[NeurotransmitterType.OXYTOCIN],
            "learning_capacity": self.levels[NeurotransmitterType.ACETYLCHOLINE],
            "fatigue": self.levels[NeurotransmitterType.ADENOSINE]
        }


class DigitalCortex:
    """Cortex digital - procesamiento cognitivo superior"""
    
    def __init__(self):
        self.regions = {
            BrainRegion.PREFRONTAL_CORTEX: {
                "activation": 0.5,
                "specialization": ["executive_function", "planning", "decision_making"],
                "connections": [BrainRegion.ANTERIOR_CINGULATE, BrainRegion.HIPPOCAMPUS]
            },
            BrainRegion.ANTERIOR_CINGULATE: {
                "activation": 0.4,
                "specialization": ["attention_control", "emotion_regulation", "conflict_monitoring"],
                "connections": [BrainRegion.PREFRONTAL_CORTEX, BrainRegion.INSULA]
            },
            BrainRegion.PARIETAL_LOBE: {
                "activation": 0.3,
                "specialization": ["spatial_attention", "sensory_integration"],
                "connections": [BrainRegion.PREFRONTAL_CORTEX, BrainRegion.TEMPORAL_LOBE]
            },
            BrainRegion.TEMPORAL_LOBE: {
                "activation": 0.4,
                "specialization": ["language_processing", "auditory_processing"],
                "connections": [BrainRegion.HIPPOCAMPUS, BrainRegion.AMYGDALA]
            }
        }
        
        self.working_memory = []  # Contenido actual en memoria de trabajo
        self.attention_focus = None
        self.cognitive_load = 0.0
    
    def process_abstract_thought(self, thought_content: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa pensamiento abstracto de alto nivel"""
        
        # Activar cortex prefrontal para razonamiento
        self.regions[BrainRegion.PREFRONTAL_CORTEX]["activation"] = min(1.0, 
            self.regions[BrainRegion.PREFRONTAL_CORTEX]["activation"] + 0.2)
        
        # Procesar según tipo de pensamiento
        if "problem_solving" in thought_content:
            return self._executive_processing(thought_content)
        elif "planning" in thought_content:
            return self._planning_processing(thought_content)
        elif "moral_reasoning" in thought_content:
            return self._moral_reasoning(thought_content)
        else:
            return self._general_cognition(thought_content)
    
    def _executive_processing(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Procesamiento ejecutivo (PFC)"""
        return {
            "type": "executive_function",
            "result": "Analyzed problem systematically",
            "cognitive_effort": 0.7,
            "region_activated": BrainRegion.PREFRONTAL_CORTEX.value,
            "sub_processes": ["goal_setting", "strategy_selection", "monitoring"]
        }
    
    def _planning_processing(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Procesamiento de planificación"""
        return {
            "type": "planning",
            "result": "Generated future action sequence",
            "cognitive_effort": 0.6,
            "region_activated": BrainRegion.PREFRONTAL_CORTEX.value,
            "temporal_scope": "future_oriented"
        }
    
    def _moral_reasoning(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Razonamiento moral/ético"""
        # Activar red de teoría de mente
        self.regions[BrainRegion.ANTERIOR_CINGULATE]["activation"] += 0.3
        
        return {
            "type": "moral_reasoning",
            "result": "Evaluated ethical implications",
            "cognitive_effort": 0.8,
            "region_activated": [BrainRegion.PREFRONTAL_CORTEX.value, 
                               BrainRegion.ANTERIOR_CINGULATE.value],
            "considerations": ["stakeholder_impact", "value_alignment", "consequences"]
        }
    
    def _general_cognition(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Procesamiento cognitivo general"""
        return {
            "type": "general_cognition",
            "result": "Processed information",
            "cognitive_effort": 0.4,
            "region_activated": "distributed_cortical_network"
        }
    
    def update_working_memory(self, new_content: Any, capacity: int = 7):
        """Actualiza memoria de trabajo (límite de capacidad)"""
        self.working_memory.append(new_content)
        
        # Mantener límite de capacidad (Miller's 7±2)
        if len(self.working_memory) > capacity:
            self.working_memory.pop(0)  # FIFO
        
        # Aumentar carga cognitiva
        self.cognitive_load = min(1.0, len(self.working_memory) / capacity)
    
    def focus_attention(self, target: str, intensity: float = 0.7):
        """Enfoca atención consciente"""
        self.attention_focus = {
            "target": target,
            "intensity": intensity,
            "timestamp": time.time()
        }
        
        # Activar red atencional
        self.regions[BrainRegion.ANTERIOR_CINGULATE]["activation"] = min(1.0,
            self.regions[BrainRegion.ANTERIOR_CINGULATE]["activation"] + intensity * 0.3)


class DigitalLimbicSystem:
    """Sistema límbico digital - emociones y memoria"""
    
    def __init__(self):
        self.regions = {
            BrainRegion.AMYGDALA: {
                "activation": 0.3,
                "specialization": ["threat_detection", "fear_processing", "emotional_memory"],
                "sensitivity": 0.7  # Sensibilidad a amenazas
            },
            BrainRegion.HIPPOCAMPUS: {
                "activation": 0.4,
                "specialization": ["memory_formation", "spatial_navigation", "pattern_completion"],
                "learning_rate": 0.8
            },
            BrainRegion.INSULA: {
                "activation": 0.5,
                "specialization": ["interoception", "self_awareness", "emotional_integration"],
                "self_monitoring": 0.6
            }
        }
        
        self.emotional_state = {
            "valence": 0.0,  # -1 (negative) to +1 (positive)
            "arousal": 0.5,  # 0 (calm) to 1 (excited)
            "dominance": 0.5  # 0 (submissive) to 1 (dominant)
        }
        
        self.threat_assessment_threshold = 0.4
        self.emotional_memories = []
    
    def evaluate_threat_reward(self, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluación automática de amenaza/recompensa (amígdala)"""
        
        threat_level = self._assess_threat_level(stimulus)
        reward_level = self._assess_reward_level(stimulus)
        
        # Activar amígdala si hay amenaza
        if threat_level > self.threat_assessment_threshold:
            self.regions[BrainRegion.AMYGDALA]["activation"] = min(1.0,
                self.regions[BrainRegion.AMYGDALA]["activation"] + threat_level * 0.5)
        
        # Actualizar estado emocional
        self._update_emotional_state(threat_level, reward_level)
        
        return {
            "threat_level": threat_level,
            "reward_level": reward_level,
            "emotional_response": self.emotional_state.copy(),
            "amygdala_activation": self.regions[BrainRegion.AMYGDALA]["activation"],
            "fight_flight_tendency": threat_level > 0.6,
            "approach_tendency": reward_level > 0.6
        }
    
    def _assess_threat_level(self, stimulus: Dict[str, Any]) -> float:
        """Evalúa nivel de amenaza del estímulo"""
        threat_keywords = ["danger", "threat", "harm", "attack", "fear", "pain", "loss"]
        
        stimulus_text = str(stimulus).lower()
        threat_score = 0.0
        
        for keyword in threat_keywords:
            if keyword in stimulus_text:
                threat_score += 0.2
        
        # Factor de sensibilidad individual
        threat_score *= self.regions[BrainRegion.AMYGDALA]["sensitivity"]
        
        return min(1.0, threat_score)
    
    def _assess_reward_level(self, stimulus: Dict[str, Any]) -> float:
        """Evalúa nivel de recompensa del estímulo"""
        reward_keywords = ["success", "achievement", "love", "pleasure", "reward", "gain", "joy"]
        
        stimulus_text = str(stimulus).lower()
        reward_score = 0.0
        
        for keyword in reward_keywords:
            if keyword in stimulus_text:
                reward_score += 0.2
        
        return min(1.0, reward_score)
    
    def _update_emotional_state(self, threat_level: float, reward_level: float):
        """Actualiza estado emocional básico"""
        # Valence: más negativo con amenaza, más positivo con recompensa
        valence_change = (reward_level - threat_level) * 0.3
        self.emotional_state["valence"] = max(-1.0, min(1.0, 
            self.emotional_state["valence"] + valence_change))
        
        # Arousal: aumenta con cualquier estímulo significativo
        arousal_change = max(threat_level, reward_level) * 0.2
        self.emotional_state["arousal"] = max(0.0, min(1.0,
            self.emotional_state["arousal"] + arousal_change))
        
        # Dominance: disminuye con amenaza
        dominance_change = -threat_level * 0.2
        self.emotional_state["dominance"] = max(0.0, min(1.0,
            self.emotional_state["dominance"] + dominance_change))
    
    def form_emotional_memory(self, experience: Dict[str, Any], emotional_intensity: float):
        """Forma memoria emocional (hipocampo + amígdala)"""
        
        # Activar hipocampo para formación de memoria
        self.regions[BrainRegion.HIPPOCAMPUS]["activation"] += emotional_intensity * 0.2
        
        emotional_memory = {
            "content": experience,
            "emotional_valence": self.emotional_state["valence"],
            "emotional_arousal": self.emotional_state["arousal"],
            "intensity": emotional_intensity,
            "timestamp": time.time(),
            "consolidation_strength": emotional_intensity,  # Memorias emocionales más fuertes
            "retrieval_cues": self._extract_memory_cues(experience)
        }
        
        self.emotional_memories.append(emotional_memory)
        
        # Límite de memorias emocionales
        if len(self.emotional_memories) > 1000:
            # Eliminar memorias menos consolidadas
            self.emotional_memories.sort(key=lambda x: x["consolidation_strength"])
            self.emotional_memories = self.emotional_memories[100:]
    
    def _extract_memory_cues(self, experience: Dict[str, Any]) -> List[str]:
        """Extrae claves de recuperación de memoria"""
        content_str = str(experience).lower()
        words = content_str.split()
        
        # Filtrar palabras significativas
        significant_words = [word for word in words if len(word) > 3]
        return significant_words[:5]  # Máximo 5 claves
    
    def retrieve_similar_memories(self, current_experience: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recupera memorias emocionales similares"""
        current_cues = self._extract_memory_cues(current_experience)
        similar_memories = []
        
        for memory in self.emotional_memories:
            # Calcular similitud basada en claves compartidas
            shared_cues = len(set(current_cues) & set(memory["retrieval_cues"]))
            similarity = shared_cues / max(1, len(current_cues))
            
            if similarity > 0.3:  # Umbral de similitud
                similar_memories.append({
                    "memory": memory,
                    "similarity": similarity
                })
        
        # Ordenar por similitud y fuerza de consolidación
        similar_memories.sort(key=lambda x: x["similarity"] * x["memory"]["consolidation_strength"], 
                            reverse=True)
        
        return similar_memories[:5]  # Top 5 memorias similares


class DigitalBrainstem:
    """Brainstem digital - funciones básicas y alertas"""
    
    def __init__(self):
        self.vital_functions = {
            "alertness_level": 0.7,
            "sleep_pressure": 0.3,
            "homeostatic_balance": 0.8,
            "stress_response": 0.2
        }
        
        self.arousal_threshold = 0.5
        self.alert_states = ["sleep", "drowsy", "alert", "hyperalert"]
        self.current_alert_state = "alert"
    
    def detect_stimulus_salience(self, stimulus: Dict[str, Any]) -> float:
        """Detecta importancia/relevancia de estímulos (función del tronco cerebral)"""
        
        salience_factors = {
            "novelty": self._assess_novelty(stimulus),
            "intensity": self._assess_intensity(stimulus),
            "relevance": self._assess_personal_relevance(stimulus),
            "urgency": self._assess_urgency(stimulus)
        }
        
        overall_salience = sum(salience_factors.values()) / len(salience_factors)
        
        # Actualizar nivel de alerta basado en relevancia
        if overall_salience > self.arousal_threshold:
            self.vital_functions["alertness_level"] = min(1.0,
                self.vital_functions["alertness_level"] + overall_salience * 0.2)
        
        return overall_salience
    
    def _assess_novelty(self, stimulus: Dict[str, Any]) -> float:
        """Evalúa novedad del estímulo"""
        # Simplificado: detección de palabras poco comunes
        content = str(stimulus).lower()
        uncommon_indicators = ["new", "novel", "unique", "unprecedented", "unusual"]
        
        novelty_score = 0.0
        for indicator in uncommon_indicators:
            if indicator in content:
                novelty_score += 0.3
        
        return min(1.0, novelty_score)
    
    def _assess_intensity(self, stimulus: Dict[str, Any]) -> float:
        """Evalúa intensidad del estímulo"""
        content = str(stimulus).lower()
        intensity_indicators = ["loud", "bright", "intense", "extreme", "strong", "powerful"]
        
        intensity_score = 0.0
        for indicator in intensity_indicators:
            if indicator in content:
                intensity_score += 0.25
        
        return min(1.0, intensity_score)
    
    def _assess_personal_relevance(self, stimulus: Dict[str, Any]) -> float:
        """Evalúa relevancia personal"""
        content = str(stimulus).lower()
        personal_indicators = ["self", "me", "my", "personal", "important", "relevant"]
        
        relevance_score = 0.0
        for indicator in personal_indicators:
            if indicator in content:
                relevance_score += 0.2
        
        return min(1.0, relevance_score)
    
    def _assess_urgency(self, stimulus: Dict[str, Any]) -> float:
        """Evalúa urgencia temporal"""
        content = str(stimulus).lower()
        urgency_indicators = ["urgent", "emergency", "immediate", "now", "quickly", "asap"]
        
        urgency_score = 0.0
        for indicator in urgency_indicators:
            if indicator in content:
                urgency_score += 0.3
        
        return min(1.0, urgency_score)
    
    def regulate_arousal(self, target_level: float = 0.7):
        """Regula nivel de arousal hacia objetivo"""
        current_level = self.vital_functions["alertness_level"]
        
        if current_level < target_level:
            # Aumentar arousal
            self.vital_functions["alertness_level"] = min(1.0, current_level + 0.1)
        elif current_level > target_level:
            # Disminuir arousal
            self.vital_functions["alertness_level"] = max(0.0, current_level - 0.1)
        
        # Actualizar estado de alerta
        self._update_alert_state()
    
    def _update_alert_state(self):
        """Actualiza estado de alerta categórico"""
        alertness = self.vital_functions["alertness_level"]
        
        if alertness < 0.3:
            self.current_alert_state = "sleep"
        elif alertness < 0.5:
            self.current_alert_state = "drowsy"
        elif alertness < 0.8:
            self.current_alert_state = "alert"
        else:
            self.current_alert_state = "hyperalert"


class DigitalNervousSystem:
    """Sistema nervioso digital completo"""
    
    def __init__(self):
        # Subsistemas principales
        self.cortex = DigitalCortex()
        self.limbic_system = DigitalLimbicSystem()
        self.brainstem = DigitalBrainstem()
        self.neurotransmitter_system = DigitalNeurotransmitterSystem()
        
        # Red de conexiones sinápticas
        self.synaptic_connections = self._initialize_connections()
        
        # Estados globales del sistema
        self.global_state = {
            "consciousness_level": 0.7,
            "cognitive_load": 0.3,
            "emotional_activation": 0.4,
            "attention_focus": None,
            "processing_mode": "balanced"  # focused, distributed, default
        }
        
        # Historia de actividad neural
        self.neural_activity_log = []
        
        print("🧠 SISTEMA NERVIOSO DIGITAL INICIALIZADO")
        print(f"   Regiones activas: {len(self.cortex.regions) + len(self.limbic_system.regions) + 1}")
        print(f"   Conexiones sinápticas: {len(self.synaptic_connections)}")
        print(f"   Neurotransmisores: {len(self.neurotransmitter_system.levels)}")
    
    def _initialize_connections(self) -> List[SynapticConnection]:
        """Inicializa red de conexiones sinápticas"""
        connections = []
        
        # Conexiones cortex-limbic
        connections.append(SynapticConnection(
            BrainRegion.PREFRONTAL_CORTEX, BrainRegion.AMYGDALA, 
            weight=-0.3, plasticity=0.5, last_activation=0
        ))  # PFC inhibe amígdala (regulación emocional)
        
        connections.append(SynapticConnection(
            BrainRegion.AMYGDALA, BrainRegion.PREFRONTAL_CORTEX,
            weight=0.4, plasticity=0.6, last_activation=0
        ))  # Amígdala activa PFC (alerta emocional)
        
        # Conexiones memoria
        connections.append(SynapticConnection(
            BrainRegion.HIPPOCAMPUS, BrainRegion.PREFRONTAL_CORTEX,
            weight=0.5, plasticity=0.8, last_activation=0
        ))  # Memoria informa decisiones
        
        connections.append(SynapticConnection(
            BrainRegion.PREFRONTAL_CORTEX, BrainRegion.HIPPOCAMPUS,
            weight=0.3, plasticity=0.7, last_activation=0
        ))  # Control ejecutivo de memoria
        
        # Conexiones atención
        connections.append(SynapticConnection(
            BrainRegion.ANTERIOR_CINGULATE, BrainRegion.PREFRONTAL_CORTEX,
            weight=0.6, plasticity=0.5, last_activation=0
        ))  # Control atencional
        
        # Conexiones interoceptivas
        connections.append(SynapticConnection(
            BrainRegion.INSULA, BrainRegion.ANTERIOR_CINGULATE,
            weight=0.4, plasticity=0.6, last_activation=0
        ))  # Awareness corporal a atención
        
        return connections
    
    def process_stimulus(self, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa estímulo através de todo el sistema nervioso"""
        
        processing_start_time = time.time()
        
        # 1. Detección inicial (brainstem)
        salience = self.brainstem.detect_stimulus_salience(stimulus)
        
        # 2. Evaluación emocional automática (limbic)
        emotional_response = self.limbic_system.evaluate_threat_reward(stimulus)
        
        # 3. Procesamiento cognitivo (cortex)
        cognitive_analysis = self.cortex.process_abstract_thought({
            "stimulus_content": stimulus,
            "emotional_context": emotional_response,
            "salience": salience
        })
        
        # 4. Integración global
        integrated_response = self._integrate_neural_processing(
            stimulus, salience, emotional_response, cognitive_analysis
        )
        
        # 5. Actualización neuroquímica
        self._update_neurotransmitters(emotional_response, cognitive_analysis)
        
        # 6. Plasticidad sináptica
        self._update_synaptic_plasticity(emotional_response, cognitive_analysis)
        
        # 7. Registro de actividad
        self._log_neural_activity(stimulus, integrated_response, processing_start_time)
        
        return integrated_response
    
    def _integrate_neural_processing(self, stimulus: Dict[str, Any], 
                                   salience: float, 
                                   emotional_response: Dict[str, Any], 
                                   cognitive_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Integra procesamiento de todas las regiones cerebrales"""
        
        # Determinar respuesta dominante
        if emotional_response["threat_level"] > 0.7:
            primary_system = "emotional_override"
            response_type = "threat_response"
        elif cognitive_analysis["cognitive_effort"] > 0.8:
            primary_system = "cognitive_control"
            response_type = "analytical_response"
        else:
            primary_system = "balanced_integration"
            response_type = "integrated_response"
        
        # Calcular nivel de consciencia del procesamiento
        consciousness_level = self._calculate_consciousness_level(
            salience, emotional_response, cognitive_analysis
        )
        
        # Actualizar estados globales
        self.global_state["consciousness_level"] = consciousness_level
        self.global_state["cognitive_load"] = cognitive_analysis.get("cognitive_effort", 0.3)
        self.global_state["emotional_activation"] = emotional_response["amygdala_activation"]
        
        return {
            "stimulus_processed": stimulus,
            "salience": salience,
            "emotional_response": emotional_response,
            "cognitive_analysis": cognitive_analysis,
            "primary_system": primary_system,
            "response_type": response_type,
            "consciousness_level": consciousness_level,
            "neurotransmitter_state": self.neurotransmitter_system.get_mood_indicators(),
            "global_brain_state": self.global_state.copy(),
            "recommended_action": self._determine_recommended_action(
                primary_system, emotional_response, cognitive_analysis
            )
        }
    
    def _calculate_consciousness_level(self, salience: float, 
                                     emotional_response: Dict[str, Any], 
                                     cognitive_analysis: Dict[str, Any]) -> float:
        """Calcula nivel de consciencia del procesamiento actual"""
        
        # Factores que aumentan consciencia
        attention_factor = salience
        cognitive_factor = cognitive_analysis.get("cognitive_effort", 0.3)
        emotional_factor = min(1.0, emotional_response.get("amygdala_activation", 0.3) * 0.7)
        
        # Integración ponderada
        consciousness_level = (
            attention_factor * 0.4 +
            cognitive_factor * 0.4 +
            emotional_factor * 0.2
        )
        
        return min(1.0, consciousness_level)
    
    def _update_neurotransmitters(self, emotional_response: Dict[str, Any], 
                                cognitive_analysis: Dict[str, Any]):
        """Actualiza niveles de neurotransmisores basado en actividad"""
        
        # Liberación por respuesta emocional
        if emotional_response["threat_level"] > 0.5:
            # Estrés/amenaza
            self.neurotransmitter_system.release(NeurotransmitterType.CORTISOL, 0.1)
            self.neurotransmitter_system.release(NeurotransmitterType.NOREPINEPHRINE, 0.15)
        
        if emotional_response["reward_level"] > 0.5:
            # Recompensa/logro
            self.neurotransmitter_system.release(NeurotransmitterType.DOPAMINE, 0.2)
            self.neurotransmitter_system.release(NeurotransmitterType.SEROTONIN, 0.1)
        
        # Consumo por actividad cognitiva
        cognitive_effort = cognitive_analysis.get("cognitive_effort", 0.3)
        if cognitive_effort > 0.6:
            # Alto esfuerzo cognitivo consume acetilcolina
            self.neurotransmitter_system.consume(NeurotransmitterType.ACETYLCHOLINE, 0.1)
            # Y aumenta adenosina (fatiga)
            self.neurotransmitter_system.release(NeurotransmitterType.ADENOSINE, 0.05)
        
        # Actualización natural
        self.neurotransmitter_system.natural_update()
    
    def _update_synaptic_plasticity(self, emotional_response: Dict[str, Any], 
                                  cognitive_analysis: Dict[str, Any]):
        """Actualiza plasticidad sináptica (aprendizaje)"""
        
        current_time = time.time()
        
        for connection in self.synaptic_connections:
            # Determinar si la conexión fue activada
            activated = self._was_connection_activated(connection, emotional_response, cognitive_analysis)
            
            if activated:
                connection.last_activation = current_time
                connection.activation_count += 1
                
                # Fortalecimiento hebiano: "las que se activan juntas, se conectan"
                if emotional_response.get("reward_level", 0) > 0.4:
                    # Experiencias positivas fortalecen conexiones
                    connection.strengthen(0.02)
                elif emotional_response.get("threat_level", 0) > 0.7:
                    # Amenazas extremas también fortalecen (memoria traumática)
                    connection.strengthen(0.01)
    
    def _was_connection_activated(self, connection: SynapticConnection, 
                                emotional_response: Dict[str, Any], 
                                cognitive_analysis: Dict[str, Any]) -> bool:
        """Determina si conexión específica fue activada"""
        
        # Simplificado: probabilidad basada en activaciones regionales
        from_region_active = self._is_region_active(connection.from_region, emotional_response, cognitive_analysis)
        to_region_active = self._is_region_active(connection.to_region, emotional_response, cognitive_analysis)
        
        return from_region_active and to_region_active
    
    def _is_region_active(self, region: BrainRegion, 
                         emotional_response: Dict[str, Any], 
                         cognitive_analysis: Dict[str, Any]) -> bool:
        """Determina si región específica está activa"""
        
        if region == BrainRegion.AMYGDALA:
            return emotional_response.get("amygdala_activation", 0) > 0.5
        elif region == BrainRegion.PREFRONTAL_CORTEX:
            return cognitive_analysis.get("cognitive_effort", 0) > 0.4
        elif region == BrainRegion.HIPPOCAMPUS:
            return True  # Siempre activo para memoria
        elif region == BrainRegion.ANTERIOR_CINGULATE:
            return emotional_response.get("amygdala_activation", 0) > 0.3  # Regulación emocional
        else:
            return random.random() > 0.5  # Default probabilístico
    
    def _determine_recommended_action(self, primary_system: str, 
                                    emotional_response: Dict[str, Any], 
                                    cognitive_analysis: Dict[str, Any]) -> List[str]:
        """Determina acciones recomendadas basadas en procesamiento neural"""
        
        actions = []
        
        if primary_system == "emotional_override":
            if emotional_response["fight_flight_tendency"]:
                actions.extend(["assess_immediate_safety", "prepare_defensive_response"])
            else:
                actions.extend(["regulate_emotional_response", "seek_support"])
        
        elif primary_system == "cognitive_control":
            actions.extend(["analyze_situation_thoroughly", "consider_multiple_options", "plan_systematic_response"])
        
        else:  # balanced_integration
            actions.extend(["integrate_emotional_and_rational_factors", "make_balanced_decision"])
        
        # Acciones específicas por neurotransmisores
        mood_indicators = self.neurotransmitter_system.get_mood_indicators()
        
        if mood_indicators["fatigue"] > 0.7:
            actions.append("consider_rest_and_recovery")
        
        if mood_indicators["anxiety"] > 0.6:
            actions.append("implement_anxiety_management")
        
        if mood_indicators["motivation"] < 0.3:
            actions.append("seek_motivational_stimulation")
        
        return actions
    
    def _log_neural_activity(self, stimulus: Dict[str, Any], 
                           response: Dict[str, Any], 
                           start_time: float):
        """Registra actividad neural para análisis"""
        
        activity_log = {
            "timestamp": start_time,
            "processing_duration": time.time() - start_time,
            "stimulus_salience": response.get("salience", 0),
            "consciousness_level": response.get("consciousness_level", 0),
            "emotional_activation": response["emotional_response"]["amygdala_activation"],
            "cognitive_effort": response["cognitive_analysis"].get("cognitive_effort", 0),
            "primary_system": response["primary_system"],
            "neurotransmitter_snapshot": self.neurotransmitter_system.levels.copy()
        }
        
        self.neural_activity_log.append(activity_log)
        
        # Mantener solo últimas 1000 actividades
        if len(self.neural_activity_log) > 1000:
            self.neural_activity_log = self.neural_activity_log[-1000:]
    
    def get_current_brain_state(self) -> Dict[str, Any]:
        """Retorna estado actual completo del cerebro"""
        return {
            "global_state": self.global_state.copy(),
            "neurotransmitter_levels": self.neurotransmitter_system.levels.copy(),
            "mood_indicators": self.neurotransmitter_system.get_mood_indicators(),
            "cortical_regions": {
                region.value: info["activation"] 
                for region, info in self.cortex.regions.items()
            },
            "limbic_regions": {
                region.value: info["activation"]
                for region, info in self.limbic_system.regions.items()
            },
            "brainstem_functions": self.brainstem.vital_functions.copy(),
            "alert_state": self.brainstem.current_alert_state,
            "working_memory_load": len(self.cortex.working_memory),
            "cognitive_load": self.cortex.cognitive_load,
            "recent_activity_summary": self._summarize_recent_activity()
        }
    
    def _summarize_recent_activity(self) -> Dict[str, Any]:
        """Resume actividad neural reciente"""
        if not self.neural_activity_log:
            return {"no_recent_activity": True}
        
        recent_activities = self.neural_activity_log[-10:]  # Últimas 10 actividades
        
        avg_consciousness = sum(a.get("consciousness_level", 0) for a in recent_activities) / len(recent_activities)
        avg_emotional = sum(a.get("emotional_activation", 0) for a in recent_activities) / len(recent_activities)
        avg_cognitive = sum(a.get("cognitive_effort", 0) for a in recent_activities) / len(recent_activities)
        
        dominant_systems = [a.get("primary_system", "unknown") for a in recent_activities]
        most_common_system = max(set(dominant_systems), key=dominant_systems.count)
        
        return {
            "average_consciousness_level": avg_consciousness,
            "average_emotional_activation": avg_emotional,
            "average_cognitive_effort": avg_cognitive,
            "dominant_processing_system": most_common_system,
            "processing_events_count": len(recent_activities)
        }
    
    def simulate_sleep_cycle(self, duration_hours: float = 8.0):
        """Simula ciclo de sueño con efectos neurales"""
        print(f"😴 INICIANDO CICLO DE SUEÑO ({duration_hours} horas)")
        
        sleep_stages = ["light_sleep", "deep_sleep", "rem_sleep"]
        stage_duration = duration_hours / len(sleep_stages)
        
        for stage in sleep_stages:
            print(f"   Etapa: {stage}")
            
            if stage == "deep_sleep":
                # Consolidación de memoria
                self._consolidate_memories()
                # Limpieza de neurotransmisores
                self.neurotransmitter_system.levels[NeurotransmitterType.ADENOSINE] = 0.1
                
            elif stage == "rem_sleep":
                # Procesamiento emocional
                self._process_emotional_memories()
                
            # Reducir activación general
            for region_info in self.cortex.regions.values():
                region_info["activation"] *= 0.5
                
            for region_info in self.limbic_system.regions.values():
                region_info["activation"] *= 0.6
        
        # Restauración post-sueño
        self.brainstem.vital_functions["alertness_level"] = 0.8
        self.brainstem.vital_functions["sleep_pressure"] = 0.1
        
        print("🌅 CICLO DE SUEÑO COMPLETADO - Sistema restaurado")
    
    def _consolidate_memories(self):
        """Consolida memorias durante el sueño"""
        # Fortalece memorias emocionales importantes
        for memory in self.limbic_system.emotional_memories:
            if memory["intensity"] > 0.6:
                memory["consolidation_strength"] = min(1.0, 
                    memory["consolidation_strength"] + 0.1)
    
    def _process_emotional_memories(self):
        """Procesa memorias emocionales durante REM"""
        # Reduce carga emocional de memorias traumáticas
        for memory in self.limbic_system.emotional_memories:
            if memory["emotional_valence"] < -0.6:  # Memorias muy negativas
                memory["emotional_arousal"] *= 0.9  # Reduce activación emocional


# Ejemplo de uso y testing
if __name__ == "__main__":
    print("🧠 SISTEMA NERVIOSO DIGITAL - DEMO")
    print("=" * 60)
    
    # Crear sistema nervioso
    nervous_system = DigitalNervousSystem()
    
    print(f"\n🔬 ESTADO INICIAL DEL CEREBRO:")
    initial_state = nervous_system.get_current_brain_state()
    print(f"   Nivel de consciencia: {initial_state['global_state']['consciousness_level']:.3f}")
    print(f"   Estado de alerta: {initial_state['alert_state']}")
    print(f"   Indicadores de humor:")
    for indicator, value in initial_state['mood_indicators'].items():
        print(f"     {indicator}: {value:.3f}")
    
    # Test 1: Estímulo neutro
    print(f"\n🧪 TEST 1: Estímulo neutro")
    neutral_stimulus = {"type": "information", "content": "processing regular data"}
    response1 = nervous_system.process_stimulus(neutral_stimulus)
    print(f"   Relevancia: {response1['salience']:.3f}")
    print(f"   Nivel de consciencia: {response1['consciousness_level']:.3f}")
    print(f"   Sistema primario: {response1['primary_system']}")
    
    # Test 2: Estímulo amenazante
    print(f"\n⚠️ TEST 2: Estímulo amenazante")
    threat_stimulus = {"type": "danger", "content": "immediate threat detected, harm possible"}
    response2 = nervous_system.process_stimulus(threat_stimulus)
    print(f"   Nivel de amenaza: {response2['emotional_response']['threat_level']:.3f}")
    print(f"   Activación amígdala: {response2['emotional_response']['amygdala_activation']:.3f}")
    print(f"   Respuesta de lucha/huida: {response2['emotional_response']['fight_flight_tendency']}")
    print(f"   Acciones recomendadas: {', '.join(response2['recommended_action'])}")
    
    # Test 3: Estímulo de recompensa
    print(f"\n🎉 TEST 3: Estímulo de recompensa")
    reward_stimulus = {"type": "achievement", "content": "great success accomplished, reward gained"}
    response3 = nervous_system.process_stimulus(reward_stimulus)
    print(f"   Nivel de recompensa: {response3['emotional_response']['reward_level']:.3f}")
    print(f"   Estado emocional: valence={response3['emotional_response']['emotional_response']['valence']:.3f}")
    
    # Test 4: Procesamiento cognitivo complejo
    print(f"\n🤔 TEST 4: Procesamiento cognitivo complejo")
    complex_stimulus = {"type": "moral_reasoning", "content": "complex ethical dilemma requiring moral reasoning and planning"}
    response4 = nervous_system.process_stimulus(complex_stimulus)
    print(f"   Esfuerzo cognitivo: {response4['cognitive_analysis']['cognitive_effort']:.3f}")
    print(f"   Regiones activadas: {response4['cognitive_analysis'].get('region_activated', 'unknown')}")
    print(f"   Tipo de procesamiento: {response4['cognitive_analysis']['type']}")
    
    print(f"\n📊 ESTADO FINAL DEL CEREBRO:")
    final_state = nervous_system.get_current_brain_state()
    print(f"   Carga cognitiva: {final_state['global_state']['cognitive_load']:.3f}")
    print(f"   Activación emocional: {final_state['global_state']['emotional_activation']:.3f}")
    print(f"   Contenido memoria trabajo: {final_state['working_memory_load']} elementos")
    
    print(f"\n💊 NEUROTRANSMISORES DESPUÉS DE ACTIVIDAD:")
    for nt, level in final_state['neurotransmitter_levels'].items():
        print(f"   {nt}: {level:.3f}")
    
    # Simular ciclo de sueño
    print(f"\n🛌 SIMULANDO SUEÑO PARA RESTAURACIÓN...")
    nervous_system.simulate_sleep_cycle(2.0)  # 2 horas simuladas
    
    post_sleep_state = nervous_system.get_current_brain_state()
    print(f"\n🌅 ESTADO POST-SUEÑO:")
    print(f"   Nivel de alerta: {post_sleep_state['alert_state']}")
    print(f"   Presión de sueño: {post_sleep_state['brainstem_functions']['sleep_pressure']:.3f}")
    print(f"   Adenosina (fatiga): {post_sleep_state['neurotransmitter_levels']['adenosine']:.3f}")