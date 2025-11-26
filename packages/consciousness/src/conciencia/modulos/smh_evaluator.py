"""
Biologically Realistic Somatic Marker Hypothesis (SMH) Implementation
Based on Antonio Damasio's Somatic Marker Hypothesis with full physiological modeling

This implementation models actual biological processes:
1. Complete neural circuit (vmPFC, OFC, Amygdala, Insula, ACC)
2. Autonomic nervous system (sympathetic/parasympathetic)
3. Neurotransmitter dynamics (DA, 5-HT, NE, ACh)
4. Physiological signals (HR, SCR, HRV, cortisol, etc.)
5. Homeostatic regulation
6. Biological timing and delays

References:
- Damasio (1994) "Descartes' Error"
- Bechara et al. (1997) Iowa Gambling Task
- Xu & Huang (2020) Electrophysiological evidence
- Critchley & Harrison (2013) Visceral influences on brain and behavior
- Thayer & Lane (2000) A model of neurovisceral integration
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import time

# ============================================================================
# PHYSIOLOGICAL STATE MODELING
# ============================================================================

@dataclass
class PhysiologicalState:
    """
    Complete physiological state modeling real bodily responses
    
    Based on:
    - Critchley & Harrison (2013): Interoceptive signals
    - Thayer & Lane (2000): Neurovisceral integration
    """
    # Cardiovascular
    heart_rate: float = 70.0           # BPM (60-100 normal)
    heart_rate_variability: float = 50.0  # RMSSD in ms (higher = better regulation)
    blood_pressure_systolic: float = 120.0  # mmHg
    blood_pressure_diastolic: float = 80.0  # mmHg
    
    # Electrodermal (Skin Conductance)
    skin_conductance_level: float = 10.0  # μS (baseline arousal)
    skin_conductance_response: float = 0.0  # μS (phasic response)
    
    # Respiratory
    respiration_rate: float = 15.0     # breaths/min (12-20 normal)
    respiration_depth: float = 0.5     # 0-1 (shallow to deep)
    
    # Hormonal
    cortisol_level: float = 15.0       # μg/dL (morning: 10-20, evening: 3-10)
    adrenaline_level: float = 50.0     # pg/mL (baseline ~50)
    
    # Visceral
    gut_activity: float = 0.5          # 0-1 (gastroparesis to hyperactivity)
    muscle_tension: float = 0.3        # 0-1 (relaxed to tense)
    
    # Temperature
    core_temperature: float = 37.0     # °C (36.5-37.5 normal)
    peripheral_temperature: float = 33.0  # °C (skin temperature)
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)

@dataclass
class NeurotransmitterState:
    """
    Neurotransmitter levels affecting emotional processing
    
    Based on:
    - Berridge & Robinson (2003): Parsing reward
    - Daw et al. (2002): Opponent interactions between serotonin and dopamine
    """
    # Dopamine (motivation, reward prediction)
    dopamine_baseline: float = 1.0     # Relative to baseline
    dopamine_phasic: float = 0.0       # Phasic burst (-1 to +2)
    
    # Serotonin (mood, punishment sensitivity)
    serotonin_level: float = 1.0       # Relative to baseline
    
    # Norepinephrine (arousal, vigilance)
    norepinephrine_level: float = 1.0  # Relative to baseline
    
    # Acetylcholine (attention, learning)
    acetylcholine_level: float = 1.0   # Relative to baseline
    
    # GABA (inhibition, anxiety reduction)
    gaba_level: float = 1.0            # Relative to baseline
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)

# ============================================================================
# NEURAL CIRCUIT MODELING
# ============================================================================

@dataclass
class NeuralRegion:
    """
    Individual brain region with activation and connectivity
    """
    name: str
    activation: float = 0.0            # 0-1
    baseline_activity: float = 0.3     # Resting state
    excitability: float = 1.0          # Sensitivity to inputs
    adaptation: float = 0.0            # Firing rate adaptation
    noise_level: float = 0.05          # Neural noise
    
    # Processing delay (ms)
    processing_delay: int = 50         # Time to process and respond
    
    # Activity history for temporal dynamics
    activity_history: deque = field(default_factory=lambda: deque(maxlen=100))

@dataclass
class NeuralCircuit:
    """
    Complete vmPFC-OFC-Amygdala-Insula circuit for somatic markers
    
    Based on:
    - Bechara & Damasio (2005): Neural basis of SMH
    - Roy et al. (2012): Ventromedial prefrontal-subcortical systems
    """
    # Prefrontal regions
    vmPFC: NeuralRegion = field(default_factory=lambda: NeuralRegion(
        "vmPFC", 
        baseline_activity=0.4,
        processing_delay=80  # Higher-order processing
    ))
    OFC: NeuralRegion = field(default_factory=lambda: NeuralRegion(
        "OFC",
        baseline_activity=0.35,
        processing_delay=70
    ))
    dlPFC: NeuralRegion = field(default_factory=lambda: NeuralRegion(
        "dlPFC",
        baseline_activity=0.3,
        processing_delay=90  # Executive control
    ))
    
    # Limbic regions
    amygdala: NeuralRegion = field(default_factory=lambda: NeuralRegion(
        "Amygdala",
        baseline_activity=0.25,
        processing_delay=30,  # Fast threat detection
        excitability=1.5  # Highly reactive
    ))
    
    # Interoceptive regions  
    insula: NeuralRegion = field(default_factory=lambda: NeuralRegion(
        "Insula",
        baseline_activity=0.4,
        processing_delay=60
    ))
    ACC: NeuralRegion = field(default_factory=lambda: NeuralRegion(
        "ACC",  # Anterior Cingulate Cortex
        baseline_activity=0.35,
        processing_delay=70
    ))
    
    # Subcortical
    VTA: NeuralRegion = field(default_factory=lambda: NeuralRegion(
        "VTA",  # Ventral Tegmental Area (dopamine)
        baseline_activity=0.3,
        processing_delay=40
    ))
    locus_coeruleus: NeuralRegion = field(default_factory=lambda: NeuralRegion(
        "LC",  # Locus Coeruleus (norepinephrine)
        baseline_activity=0.3,
        processing_delay=35
    ))

# ============================================================================
# AUTONOMIC NERVOUS SYSTEM
# ============================================================================

@dataclass
class AutonomicState:
    """
    Sympathetic and parasympathetic nervous system state
    
    Based on:
    - Porges (2007): Polyvagal theory
    - Thayer & Lane (2000): Model of neurovisceral integration
    """
    # Sympathetic (fight-or-flight)
    sympathetic_tone: float = 0.3      # 0-1 (low to high activation)
    
    # Parasympathetic (rest-and-digest)
    parasympathetic_tone: float = 0.7  # 0-1 (low to high activation)
    
    # Vagal tone (cardiac parasympathetic control)
    vagal_tone: float = 0.7            # 0-1 (derived from HRV)
    
    # Balance (-1: parasympathetic dominant, +1: sympathetic dominant)
    @property
    def autonomic_balance(self) -> float:
        return self.sympathetic_tone - self.parasympathetic_tone

# ============================================================================
# BIOLOGICALLY REALISTIC SOMATIC MARKER
# ============================================================================

@dataclass
class SomaticMarker:
    """
    Somatic marker with complete biological representation
    
    Not just valence/arousal, but full physiological signature
    """
    # Situation pattern (as before)
    situation_pattern: Dict[str, float]
    context: str
    
    # Emotional evaluation
    emotional_valence: float           # -1 to +1
    arousal: float                     # 0 to 1
    
    # Physiological signature (what the body does)
    typical_heart_rate_change: float = 0.0      # BPM delta from baseline
    typical_scr_amplitude: float = 0.0          # μS
    typical_hrv_change: float = 0.0             # RMSSD delta
    typical_respiratory_change: float = 0.0     # breaths/min delta
    typical_cortisol_change: float = 0.0        # μg/dL delta
    
    # Neural signature (which regions activate)
    vmPFC_activation: float = 0.0      # Relative activation
    OFC_activation: float = 0.0
    amygdala_activation: float = 0.0
    insula_activation: float = 0.0
    
    # Neurotransmitter signature
    dopamine_response: float = 0.0     # Expected DA change
    serotonin_response: float = 0.0    # Expected 5-HT change
    norepinephrine_response: float = 0.0  # Expected NE change
    
    # Learning
    strength: float = 0.5              # Association strength
    reinforcement_count: int = 0
    last_activated: float = field(default_factory=time.time)
    
    # Temporal dynamics
    onset_latency: float = 0.2         # Seconds to marker activation
    duration: float = 3.0              # Seconds marker persists

# Alias for backward compatibility
DecisionOption = dict  # Will be populated with decision data

# ============================================================================
# MAIN BIOLOGICAL SMH SYSTEM (SMHEvaluator - Compatible Interface)
# ============================================================================

class SMHEvaluator:
    """
    Biologically realistic Somatic Marker Hypothesis implementation
    
    BACKWARD COMPATIBLE with previous SMHEvaluator interface
    
    Now models complete physiological and neural processes:
    - Real-time physiological state
    - Neural circuit dynamics
    - Autonomic nervous system
    - Neurotransmitter dynamics
    - Homeostatic regulation
    """
    
    def __init__(self):
        # Biological states
        self.physiology = PhysiologicalState()
        self.neurotransmitters = NeurotransmitterState()
        self.neural_circuit = NeuralCircuit()
        self.autonomic = AutonomicState()
        
        # Somatic markers (learned associations)
        self.markers: List[SomaticMarker] = []
        
        # Homeostatic regulation
        self.homeostatic_setpoints = {
            'heart_rate': 70.0,
            'blood_pressure_systolic': 120.0,
            'cortisol': 15.0,
            'core_temperature': 37.0
        }
        
        # Processing parameters
        self.dt = 0.01  # 10ms time steps (biologically realistic)
        self.current_time = 0.0
        
        # History for temporal dynamics
        self.physiological_history = deque(maxlen=1000)  # Last 10 seconds at 10ms
        self.neural_history = deque(maxlen=1000)
        
        # Learning parameters (backward compatible)
        self.learning_rate = 0.2
        self.learning_rate_fast = 0.3    # Fast learning (amygdala)
        self.learning_rate_slow = 0.05   # Slow learning (cortex)
        self.decay_rate = 0.98
        self.match_threshold = 0.6
        
        # Decision history (backward compatible)
        self.decision_history: List[Dict[str, Any]] = []
        self.max_history = 100
        
    def evaluate_situation(self, 
                          current_state: Dict[str, float],
                          context: str = "general") -> Dict[str, Any]:
        """
        Evaluate current situation using somatic markers
        
        BACKWARD COMPATIBLE interface with added biological realism
        
        Process (biologically realistic):
        0-30ms:   Amygdala activation (threat detection)
        30-80ms:  Insula activation (interoceptive representation)
        80-150ms: vmPFC/OFC activation (marker retrieval)
        150ms+:   Autonomic and physiological response
        
        Returns:
            Complete biological response (backward compatible format)
        """
        # Process through biological timeline
        response = self._process_stimulus_biological(
            stimulus=current_state,
            context=context,
            duration=0.5  # 500ms processing
        )
        
        # Return backward-compatible format
        return {
            'somatic_valence': response['somatic_valence'],
            'arousal': response['arousal'],
            'confidence': response['confidence'],
            'marker_count': response['marker_count'],
            'recommendation': response['recommendation'],
            'markers_used': response.get('markers_used', []),
            
            # ADDED: Biological extensions
            'physiology': response.get('physiology', {}),
            'neural_activation': response.get('neural_activation', {}),
            'autonomic_state': response.get('autonomic', {})
        }
    
    def _process_stimulus_biological(self, 
                                     stimulus: Dict[str, float],
                                     context: str,
                                     duration: float = 0.5) -> Dict[str, Any]:
        """
        Process stimulus through complete biological pathway
        """
        n_steps = int(duration / self.dt)
        relevant_markers = []
        
        for step in range(min(n_steps, 50)):  # Limit iterations for performance
            self.current_time += self.dt
            
            # PHASE 1: AMYGDALA (0-30ms)
            if step * self.dt < 0.030:
                amygdala_input = self._calculate_amygdala_input(stimulus)
                self.neural_circuit.amygdala.activation = self._activate_region(
                    self.neural_circuit.amygdala,
                    amygdala_input
                )
            
            # PHASE 2: INSULA (30-80ms)
            if 0.030 <= step * self.dt < 0.080:
                insula_input = self._calculate_insula_input(stimulus, self.physiology)
                self.neural_circuit.insula.activation = self._activate_region(
                    self.neural_circuit.insula,
                    insula_input
                )
            
            # PHASE 3: vmPFC/OFC (80-150ms)
            if 0.080 <= step * self.dt < 0.150:
                relevant_markers = self._retrieve_markers_biological(
                    stimulus, context
                )
                
                if relevant_markers:
                    vmPFC_input = np.mean([m.vmPFC_activation for m in relevant_markers])
                    OFC_input = np.mean([m.OFC_activation for m in relevant_markers])
                else:
                    vmPFC_input = 0.3
                    OFC_input = 0.3
                
                self.neural_circuit.vmPFC.activation = self._activate_region(
                    self.neural_circuit.vmPFC,
                    vmPFC_input
                )
                self.neural_circuit.OFC.activation = self._activate_region(
                    self.neural_circuit.OFC,
                    OFC_input
                )
            
            # PHASE 4+: AUTONOMIC AND PHYSIOLOGY (150ms+)
            if step * self.dt >= 0.150:
                self._update_autonomic_from_markers(relevant_markers)
                self._update_physiology(self.dt)
                self._update_neurotransmitters(relevant_markers, self.dt)
            
            # Continuous homeostatic regulation
            self._homeostatic_regulation(self.dt)
        
        # Generate response
        return self._generate_response(relevant_markers)
    
    def _calculate_amygdala_input(self, stimulus: Dict[str, float]) -> float:
        """Calculate amygdala activation (fast threat/reward detection)"""
        # Helper to safely extract float value
        def safe_get(key, default=0.0):
            val = stimulus.get(key, default)
            if isinstance(val, (list, tuple, np.ndarray)):
                try:
                    return float(np.mean(val))
                except:
                    return default
            try:
                return float(val)
            except:
                return default

        threat_level = safe_get('threat')
        if threat_level == 0.0:
            threat_level = safe_get('prediction_error') * 0.5
            
        reward_level = safe_get('reward')
        novelty = safe_get('novelty')

        # Calculate intensity safely - ensure it's always a float
        if 'intensity' in stimulus:
            intensity = safe_get('intensity')
        elif stimulus:
            # Find the maximum absolute value from stimulus values
            try:
                max_val = 0.0
                for v in stimulus.values():
                    if isinstance(v, (int, float)):
                        val = float(v)
                    elif isinstance(v, (list, tuple, np.ndarray)):
                        try:
                            val = float(np.mean(v)) if len(v) > 0 else 0.0
                        except:
                            val = 0.0
                    else:
                        val = 0.0
                    max_val = max(max_val, abs(val))
                intensity = max_val if max_val > 0 else 0.5
            except:
                intensity = 0.5
        else:
            intensity = 0.5

        # Ensure intensity is a proper float between 0 and 1
        intensity = float(max(0.0, min(1.0, intensity)))

        activation = (
            abs(threat_level) * 0.7 * intensity +
            abs(reward_level) * 0.5 * intensity +
            abs(novelty) * 0.3
        )

        return np.clip(activation, 0, 1)
    
    def _calculate_insula_input(self, stimulus: Dict[str, float], physiology: PhysiologicalState) -> float:
        """Calculate insula activation (interoceptive awareness)"""
        interoceptive_intensity = (
            abs(physiology.heart_rate - 70) / 30 * 0.3 +
            physiology.skin_conductance_level / 20 * 0.3 +
            abs(physiology.respiration_rate - 15) / 5 * 0.2 +
            physiology.muscle_tension * 0.2
        )
        
        expected_arousal = stimulus.get('expected_arousal', 0.5)
        actual_arousal = self._calculate_current_arousal()
        prediction_error = abs(expected_arousal - actual_arousal) * 0.4
        
        activation = interoceptive_intensity + prediction_error
        return np.clip(activation, 0, 1)
    
    def _activate_region(self, region: NeuralRegion, input_strength: float) -> float:
        """Activate neural region with realistic dynamics"""
        noisy_input = input_strength + np.random.normal(0, region.noise_level)
        raw_activation = 1 / (1 + np.exp(-10 * (noisy_input - 0.5) / region.excitability))
        adapted_activation = raw_activation * (1 - region.adaptation * 0.5)
        final_activation = region.baseline_activity * 0.3 + adapted_activation * 0.7
        
        if final_activation > region.baseline_activity:
            region.adaptation = min(1.0, region.adaptation + 0.01)
        else:
            region.adaptation = max(0.0, region.adaptation - 0.005)
        
        region.activity_history.append(final_activation)
        return np.clip(final_activation, 0, 1)
    
    def _retrieve_markers(self, state: Dict[str, float], context: str) -> List[SomaticMarker]:
        """Retrieve markers (backward compatible)"""
        return self._retrieve_markers_biological(state, context)
    
    def _retrieve_markers_biological(self, stimulus: Dict[str, float], context: str, threshold: float = None) -> List[SomaticMarker]:
        """Retrieve markers with biological constraints"""
        if threshold is None:
            threshold = self.match_threshold
            
        relevant = []
        
        for marker in self.markers:
            if marker.context != context and marker.context != "general":
                continue
            
            similarity = self._pattern_similarity(stimulus, marker.situation_pattern)
            
            if similarity >= threshold:
                time_since_use = self.current_time - marker.last_activated
                recency_factor = np.exp(-time_since_use / 3600)
                effective_strength = marker.strength * (0.7 + 0.3 * recency_factor)
                
                if effective_strength > 0.3:
                    relevant.append(marker)
        
        relevant.sort(key=lambda m: m.strength, reverse=True)
        return relevant[:5]  # Working memory constraint
    
    def _pattern_similarity(self, pattern1: Dict[str, float], pattern2: Dict[str, float]) -> float:
        """Cosine similarity between patterns"""
        common_keys = set(pattern1.keys()) & set(pattern2.keys())
        if not common_keys:
            return 0.0
        
        vec1 = np.array([pattern1[k] for k in common_keys])
        vec2 = np.array([pattern2[k] for k in common_keys])
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return max(0.0, dot_product / (norm1 * norm2))
    
    def _calculate_similarity(self, state1: Dict[str, float], state2: Dict[str, float]) -> float:
        """Backward compatible alias"""
        return self._pattern_similarity(state1, state2)
    
    def _update_autonomic_from_markers(self, markers: List[SomaticMarker]):
        """Update autonomic state based on activated markers"""
        if not markers:
            self.autonomic.sympathetic_tone *= 0.95
            self.autonomic.parasympathetic_tone += (0.7 - self.autonomic.parasympathetic_tone) * 0.05
            return
        
        avg_valence = np.mean([m.emotional_valence for m in markers])
        avg_arousal = np.mean([m.arousal for m in markers])
        avg_strength = np.mean([m.strength for m in markers])
        
        if avg_valence < 0 and avg_arousal > 0.5:
            target_sympathetic = 0.7 + avg_arousal * 0.3
            target_parasympathetic = 0.3 - avg_arousal * 0.2
        elif avg_valence > 0 and avg_arousal > 0.5:
            target_sympathetic = 0.5 + avg_arousal * 0.2
            target_parasympathetic = 0.5
        else:
            target_sympathetic = 0.2
            target_parasympathetic = 0.8
        
        tau = 0.5
        alpha = self.dt / tau
        
        self.autonomic.sympathetic_tone += (target_sympathetic - self.autonomic.sympathetic_tone) * alpha * avg_strength
        self.autonomic.parasympathetic_tone += (target_parasympathetic - self.autonomic.parasympathetic_tone) * alpha * avg_strength
        
        self.autonomic.sympathetic_tone = np.clip(self.autonomic.sympathetic_tone, 0, 1)
        self.autonomic.parasympathetic_tone = np.clip(self.autonomic.parasympathetic_tone, 0, 1)
        self.autonomic.vagal_tone = self.autonomic.parasympathetic_tone * 0.7 + 0.3
    
    def _update_physiology(self, dt: float):
        """Update physiological state"""
        target_hr = 60 + 40 * self.autonomic.sympathetic_tone - 20 * self.autonomic.parasympathetic_tone
        tau_hr = 2.0
        self.physiology.heart_rate += (target_hr - self.physiology.heart_rate) * (dt / tau_hr)
        
        target_hrv = 30 + 60 * self.autonomic.parasympathetic_tone
        tau_hrv = 5.0
        self.physiology.heart_rate_variability += (target_hrv - self.physiology.heart_rate_variability) * (dt / tau_hrv)
        
        target_scr = 2.0 * self.autonomic.sympathetic_tone
        tau_scr = 0.5
        self.physiology.skin_conductance_response += (target_scr - self.physiology.skin_conductance_response) * (dt / tau_scr)
        
        target_scl = 5 + 15 * self.autonomic.sympathetic_tone
        tau_scl = 10.0
        self.physiology.skin_conductance_level += (target_scl - self.physiology.skin_conductance_level) * (dt / tau_scl)
        
        target_rr = 12 + 8 * self.autonomic.sympathetic_tone
        tau_rr = 3.0
        self.physiology.respiration_rate += (target_rr - self.physiology.respiration_rate) * (dt / tau_rr)
        
        target_bp_sys = 110 + 30 * self.autonomic.sympathetic_tone
        target_bp_dia = 70 + 20 * self.autonomic.sympathetic_tone
        tau_bp = 5.0
        self.physiology.blood_pressure_systolic += (target_bp_sys - self.physiology.blood_pressure_systolic) * (dt / tau_bp)
        self.physiology.blood_pressure_diastolic += (target_bp_dia - self.physiology.blood_pressure_diastolic) * (dt / tau_bp)
        
        target_cortisol = 10 + 15 * self.autonomic.sympathetic_tone
        tau_cortisol = 300.0
        self.physiology.cortisol_level += (target_cortisol - self.physiology.cortisol_level) * (dt / tau_cortisol)
        
    def _update_neurotransmitters(self, markers: List[SomaticMarker], dt: float):
        """Update neurotransmitter levels"""
        if not markers:
            self.neurotransmitters.dopamine_phasic *= 0.95
            return
        
        avg_da_response = np.mean([m.dopamine_response for m in markers])
        vta_drive = self.neural_circuit.VTA.activation
        target_da_phasic = avg_da_response + vta_drive * 0.5
        tau_da = 0.2
        self.neurotransmitters.dopamine_phasic += (target_da_phasic - self.neurotransmitters.dopamine_phasic) * (dt / tau_da)
        
        avg_valence = np.mean([m.emotional_valence for m in markers]) if markers else 0
        target_5ht = 1.0 + avg_valence * 0.3
        tau_5ht = 10.0
        self.neurotransmitters.serotonin_level += (target_5ht - self.neurotransmitters.serotonin_level) * (dt / tau_5ht)
        
        lc_drive = self.neural_circuit.locus_coeruleus.activation
        target_ne = 1.0 + lc_drive * 0.8
        tau_ne = 1.0
        self.neurotransmitters.norepinephrine_level += (target_ne - self.neurotransmitters.norepinephrine_level) * (dt / tau_ne)
        
    def _homeostatic_regulation(self, dt: float):
        """Homeostatic processes"""
        hr_error = self.physiology.heart_rate - self.homeostatic_setpoints['heart_rate']
        self.physiology.heart_rate -= hr_error * 0.001 * dt
        
        bp_error = self.physiology.blood_pressure_systolic - self.homeostatic_setpoints['blood_pressure_systolic']
        self.physiology.blood_pressure_systolic -= bp_error * 0.0005 * dt
        
        temp_error = self.physiology.core_temperature - self.homeostatic_setpoints['core_temperature']
        self.physiology.core_temperature -= temp_error * 0.01 * dt
        
    def _calculate_current_arousal(self) -> float:
        """Calculate current arousal from physiological state"""
        hr_arousal = (self.physiology.heart_rate - 60) / 40
        scl_arousal = self.physiology.skin_conductance_level / 20
        rr_arousal = (self.physiology.respiration_rate - 12) / 8
        
        arousal = hr_arousal * 0.4 + scl_arousal * 0.4 + rr_arousal * 0.2
        return np.clip(arousal, 0, 1)
    
    def _generate_response(self, markers: List[SomaticMarker]) -> Dict[str, Any]:
        """Generate complete response"""
        if not markers:
            avg_valence = 0.0
            avg_arousal = self._calculate_current_arousal()
            confidence = 0.0
        else:
            avg_valence = np.mean([m.emotional_valence for m in markers])
            avg_arousal = np.mean([m.arousal for m in markers])
            confidence = min(1.0, np.mean([m.strength for m in markers]))
        
        return {
            'somatic_valence': avg_valence,
            'arousal': avg_arousal,
            'confidence': confidence,
            'marker_count': len(markers),
            'recommendation': 'approach' if avg_valence > 0.3 else ('avoid' if avg_valence < -0.3 else 'neutral'),
            'markers_used': [
                {
                    'valence': m.emotional_valence,
                    'arousal': m.arousal,
                    'strength': m.strength,
                    'context': m.context
                }
                for m in markers[:3]
            ],
            'physiology': {
                'heart_rate': self.physiology.heart_rate,
                'hrv': self.physiology.heart_rate_variability,
                'scr': self.physiology.skin_conductance_response,
                'scl': self.physiology.skin_conductance_level,
                'respiration_rate': self.physiology.respiration_rate,
                'blood_pressure': f"{self.physiology.blood_pressure_systolic:.0f}/{self.physiology.blood_pressure_diastolic:.0f}",
                'cortisol': self.physiology.cortisol_level
            },
            'neural_activation': {
                'vmPFC': self.neural_circuit.vmPFC.activation,
                'OFC': self.neural_circuit.OFC.activation,
                'amygdala': self.neural_circuit.amygdala.activation,
                'insula': self.neural_circuit.insula.activation,
                'ACC': self.neural_circuit.ACC.activation
            },
            'neurotransmitters': {
                'dopamine_phasic': self.neurotransmitters.dopamine_phasic,
                'serotonin': self.neurotransmitters.serotonin_level,
                'norepinephrine': self.neurotransmitters.norepinephrine_level
            },
            'autonomic': {
                'sympathetic_tone': self.autonomic.sympathetic_tone,
                'parasympathetic_tone': self.autonomic.parasympathetic_tone,
                'autonomic_balance': self.autonomic.autonomic_balance,
                'vagal_tone': self.autonomic.vagal_tone
            }
        }
    
    def reinforce_marker(self, situation: Dict[str, float], outcome_valence: float, outcome_arousal: float, context: str = "general"):
        """
        Learn from outcome with biological realism (BACKWARD COMPATIBLE)
        """
        similar_marker = None
        best_similarity = 0.0
        
        for marker in self.markers:
            if marker.context != context:
                continue
            
            sim = self._pattern_similarity(situation, marker.situation_pattern)
            if sim > best_similarity and sim >= self.match_threshold:
                best_similarity = sim
                similar_marker = marker
        
        if similar_marker:
            # Reconsolidation
            alpha = self.learning_rate_fast if abs(outcome_valence) > 0.5 else self.learning_rate_slow
            
            similar_marker.emotional_valence += (outcome_valence - similar_marker.emotional_valence) * alpha
            similar_marker.arousal += (outcome_arousal - similar_marker.arousal) * alpha
            similar_marker.strength = min(1.0, similar_marker.strength + 0.1)
            similar_marker.reinforcement_count += 1
            similar_marker.last_activated = self.current_time
            
            # Update biological signatures
            similar_marker.typical_heart_rate_change = (
                similar_marker.typical_heart_rate_change * (1 - alpha) +
                (self.physiology.heart_rate - 70.0) * alpha
            )
            similar_marker.typical_scr_amplitude = (
                similar_marker.typical_scr_amplitude * (1 - alpha) +
                self.physiology.skin_conductance_response * alpha
            )
        else:
            # Create new marker with biological signature
            new_marker = SomaticMarker(
                situation_pattern=situation.copy(),
                context=context,
                emotional_valence=outcome_valence,
                arousal=outcome_arousal,
                strength=0.5,
                reinforcement_count=1,
                typical_heart_rate_change=self.physiology.heart_rate - 70.0,
                typical_scr_amplitude=self.physiology.skin_conductance_response,
                typical_hrv_change=self.physiology.heart_rate_variability - 50.0,
                vmPFC_activation=self.neural_circuit.vmPFC.activation,
                OFC_activation=self.neural_circuit.OFC.activation,
                amygdala_activation=self.neural_circuit.amygdala.activation,
                insula_activation=self.neural_circuit.insula.activation,
                dopamine_response=self.neurotransmitters.dopamine_phasic,
                serotonin_response=self.neurotransmitters.serotonin_level - 1.0,
                norepinephrine_response=self.neurotransmitters.norepinephrine_level - 1.0
            )
            self.markers.append(new_marker)
    
    def decay_markers(self):
        """Decay unused markers (BACKWARD COMPATIBLE)"""
        for marker in self.markers:
            marker.strength *= self.decay_rate
        
        self.markers = [m for m in self.markers if m.strength > 0.1]
    
    def evaluate_options(self, options: List[Dict[str, float]], context: str = "decision") -> List[Dict[str, Any]]:
        """Evaluate multiple options (BACKWARD COMPATIBLE)"""
        evaluated_options = []
        
        for i, option_state in enumerate(options):
            evaluation = self.evaluate_situation(option_state, context)
            
            decision_option = {
                'option_id': f"option_{i}",
                'predicted_state': option_state,
                'somatic_value': evaluation['somatic_valence'],
                'arousal': evaluation['arousal'],
                'confidence': evaluation['confidence']
            }
            
            evaluated_options.append(decision_option)
        
        evaluated_options.sort(key=lambda x: x['somatic_value'], reverse=True)
        return evaluated_options
    
    def get_emotional_bias(self) -> Dict[str, float]:
        """Get emotional bias for GWT (BACKWARD COMPATIBLE)"""
        if not self.markers:
            return {}
        
        subsystem_biases = {}
        
        for marker in self.markers:
            for subsystem, value in marker.situation_pattern.items():
                if subsystem not in subsystem_biases:
                    subsystem_biases[subsystem] = []
                
                bias = marker.emotional_valence * marker.strength
                subsystem_biases[subsystem].append(bias)
        
        final_biases = {k: np.mean(v) for k, v in subsystem_biases.items()}
        return final_biases
    
    def get_summary(self) -> Dict[str, Any]:
        """Get system summary (BACKWARD COMPATIBLE + BIOLOGICAL)"""
        if not self.markers:
            return {
                'total_markers': 0,
                'average_strength': 0.0,
                'positive_markers': 0,
                'negative_markers': 0,
                'physiology': self._get_physiological_summary(),
                'neural': self._get_neural_summary()
            }
        
        positive = sum(1 for m in self.markers if m.emotional_valence > 0)
        negative = sum(1 for m in self.markers if m.emotional_valence < 0)
        avg_strength = np.mean([m.strength for m in self.markers])
        
        return {
            'total_markers': len(self.markers),
            'average_strength': avg_strength,
            'positive_markers': positive,
            'negative_markers': negative,
            'contexts': list(set(m.context for m in self.markers)),
            'physiology': self._get_physiological_summary(),
            'neural': self._get_neural_summary(),
            'autonomic': {
                'sympathetic': self.autonomic.sympathetic_tone,
                'parasympathetic': self.autonomic.parasympathetic_tone,
                'balance': self.autonomic.autonomic_balance
            }
        }
    
    def _get_physiological_summary(self) -> Dict[str, str]:
        """Get physiological state summary"""
        return {
            'heart_rate': f"{self.physiology.heart_rate:.1f} BPM",
            'hrv': f"{self.physiology.heart_rate_variability:.1f} ms",
            'blood_pressure': f"{self.physiology.blood_pressure_systolic:.0f}/{self.physiology.blood_pressure_diastolic:.0f} mmHg",
            'respiration': f"{self.physiology.respiration_rate:.1f} /min",
            'scl': f"{self.physiology.skin_conductance_level:.2f} μS",
            'cortisol': f"{self.physiology.cortisol_level:.1f} μg/dL"
        }
    
    def _get_neural_summary(self) -> Dict[str, str]:
        """Get neural activation summary"""
        return {
            'vmPFC': f"{self.neural_circuit.vmPFC.activation:.2f}",
            'amygdala': f"{self.neural_circuit.amygdala.activation:.2f}",
            'insula': f"{self.neural_circuit.insula.activation:.2f}",
            'OFC': f"{self.neural_circuit.OFC.activation:.2f}"
        }
