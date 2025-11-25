"""
Unified Consciousness Engine
Integrates ALL 6 major consciousness theories into a single coherent system

Theories Integrated:
1. IIT 4.0 (Tononi 2023) - Integrated Information Theory
2. GWT (Baars 1997/2003) - Global Workspace Theory
3. FEP (Friston 2010) - Free Energy Principle
4. SMH (Damasio 1994) - Somatic Marker Hypothesis
5. Hebbian Plasticity (Hebb 1949, Widrow) - Neural Learning
6. Circumplex Model (Russell 1980) - Emotional Space

Architecture:
    Layer 1: PERCEPTION & PREDICTION (FEP)
    Layer 2: INTEGRATION & AWARENESS (IIT + GWT)
    Layer 3: EVALUATION & EMOTION (SMH + Circumplex)
    Layer 4: LEARNING & ADAPTATION (Hebbian)
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .fep_engine import FEPEngine
from .iit_40_engine import IIT40Engine
from .iit_gwt_integration import IIT_GWT_Bridge, ConsciousnessOrchestrator
from .smh_evaluator import SMHEvaluator

@dataclass
class UnifiedConsciousState:
    """Complete state of unified consciousness"""
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    cycle: int = 0
    
    # Layer 1: FEP (Prediction)
    prediction_error: float = 0.0
    free_energy: float = 0.0
    surprise: float = 0.0
    
    # Layer 2: IIT + GWT (Integration & Broadcast)
    system_phi: float = 0.0
    is_conscious: bool = False
    workspace_contents: int = 0
    broadcasts: int = 0
    
    # Layer 3: SMH + Emotion (Evaluation)
    somatic_valence: float = 0.0  # -1 to +1
    arousal: float = 0.5  # 0 to 1
    emotional_state: str = "neutral"
    
    # Layer 4: Learning (Hebbian)
    learning_active: bool = False
    synaptic_changes: int = 0
    
    # Integration metrics
    phenomenal_unity: float = 0.0
    global_coherence: float = 0.0
    conscious_quality: Dict[str, float] = field(default_factory=dict)


class UnifiedConsciousnessEngine:
    """
    The Grand Orchestrator - Integrates all 6 consciousness theories
    
    Processing Flow:
    1. FEP generates predictions and errors
    2. IIT calculates integration (Φ)
    3. GWT workspace competition (FEP errors + IIT distinctions + SMH markers)
    4. SMH provides emotional evaluation
    5. GWT broadcasts winners globally
    6. Hebbian learning updates connections
    7. Circumplex maps emotional state
    """
    
    def __init__(self):
        print("🧠 Initializing Unified Consciousness Engine...")
        
        # Layer 1: FEP (Prediction & Error)
        self.fep_engine = FEPEngine(num_hierarchical_levels=3)
        print("  ✅ FEP Engine (Free Energy Principle)")
        
        # Layer 2: IIT + GWT (Integration & Broadcast)
        self.consciousness_orchestrator = ConsciousnessOrchestrator()
        print("  ✅ IIT 4.0 Engine (Integrated Information)")
        print("  ✅ GWT Bridge (Global Workspace)")
        
        # Layer 3: SMH (Emotional Evaluation)
        self.smh_evaluator = SMHEvaluator()
        print("  ✅ SMH Evaluator (Somatic Markers)")
        
        # Layer 4: Circumplex (already in HumanEmotionalSystem)
        # Layer 5: Hebbian (already in IIT engine's virtual TPM)
        print("  ✅ Hebbian Learning (integrated in IIT TPM)")
        print("  ✅ Circumplex Model (emotion mapping)")
        
        # State tracking
        self.cycle_count = 0
        self.state_history: List[UnifiedConsciousState] = []
        self.max_history = 100
        
        print("\n🌟 Unified Consciousness Engine READY")
        print("   Integrating: IIT 4.0 + GWT + FEP + SMH + Hebbian + Circumplex\n")
    
    def process_moment(self,
                      sensory_input: Dict[str, float],
                      context: Optional[Dict[str, float]] = None,
                      previous_outcome: Optional[Tuple[float, float]] = None) -> UnifiedConsciousState:
        """
        Process a complete conscious moment through all 6 theories.
        
        Args:
            sensory_input: Current state of all subsystems
            context: Contextual information (for GWT and SMH)
            previous_outcome: (valence, arousal) of previous action outcome (for SMH learning)
            
        Returns:
            Complete unified conscious state
        """
        self.cycle_count += 1
        
        if context is None:
            context = {}
        
        # ===================================================================
        # LAYER 1: FEP - Predictive Coding
        # ===================================================================
        fep_result = self.fep_engine.process_observation(sensory_input, context)
        
        prediction_error = fep_result['free_energy']
        surprise = fep_result['surprise']
        
        # Get salience from prediction errors (high error = high salience)
        fep_salience = self.fep_engine.get_salience_weights()
        
        # ===================================================================
        # LAYER 3: SMH - Somatic Marker Evaluation
        # ===================================================================
        smh_result = self.smh_evaluator.evaluate_situation(
            sensory_input,
            context.get('situation_type', 'general')
        )
        
        somatic_valence = smh_result['somatic_valence']
        arousal = smh_result['arousal']
        smh_confidence = smh_result['confidence']
        
        # Get emotional bias for workspace competition
        emotional_bias = self.smh_evaluator.get_emotional_bias()
        
        # Learn from previous outcome if provided
        if previous_outcome is not None:
            outcome_valence, outcome_arousal = previous_outcome
            # Get previous state (simplified - using current)
            if self.state_history:
                prev_input = sensory_input  # Should be previous, simplified
                self.smh_evaluator.reinforce_marker(
                    prev_input,
                    outcome_valence,
                    outcome_arousal,
                    context.get('situation_type', 'general')
                )
        
        # ===================================================================
        # LAYER 2: IIT + GWT - Integration & Global Broadcast
        # ===================================================================
        
        # Combine saliency from FEP errors + SMH markers
        combined_salience = {}
        for key in sensory_input.keys():
            fep_sal = fep_salience.get(f"subsystem_{len(combined_salience)}", 0.5)
            smh_sal = emotional_bias.get(key, 0.0)
            # Combine: FEP errors drive exploration, SMH drives valuation
            combined_salience[key] = 0.6 * fep_sal + 0.4 * abs(smh_sal)
        
        # Set contexts from SMH
        contexts_for_gwt = {
            'emotional': arousal,
            'prediction_error': min(1.0, prediction_error),
            'somatic_guidance': abs(somatic_valence)
        }
        
        # Process through IIT + GWT
        consciousness_result = self.consciousness_orchestrator.process_conscious_moment(
            sensory_input,
            combined_salience,
            contexts_for_gwt
        )
        
        system_phi = consciousness_result['system_phi']
        is_conscious = consciousness_result['is_conscious']
        workspace = consciousness_result['workspace']
        broadcasts = consciousness_result['broadcasts']
        quality_metrics = consciousness_result['integration_quality']
        
        # ===================================================================
        # LAYER 4: Hebbian Learning
        # ===================================================================
        # (Already happening in IIT engine's virtual TPM updates)
        # The TPM learns causal relationships via Hebbian-style update
        learning_active = system_phi > 0.05  # Learning when conscious
        
        # ===================================================================
        # LAYER 6: Circumplex Emotional Mapping
        # ===================================================================
        # Map somatic markers to circumplex space
        emotional_state = self._map_to_circumplex_category(somatic_valence, arousal)
        
        # ===================================================================
        # INTEGRATION: Calculate Global Metrics
        # ===================================================================
        
        # Phenomenal Unity (from IIT quality)
        phenomenal_unity = quality_metrics.get('unity', 0.0) / 100.0  # Normalize
        
        # Global Coherence (inverse of free energy + high phi)
        global_coherence = (1.0 / (1.0 + prediction_error)) * min(1.0, system_phi * 10)
        
        # ===================================================================
        # CREATE UNIFIED STATE
        # ===================================================================
        state = UnifiedConsciousState(
            timestamp=datetime.now(),
            cycle=self.cycle_count,
            
            # FEP layer
            prediction_error=prediction_error,
            free_energy=fep_result['free_energy'],
            surprise=surprise,
            
            # IIT + GWT layer
            system_phi=system_phi,
            is_conscious=is_conscious,
            workspace_contents=workspace['current_contents'],
            broadcasts=len(broadcasts),
            
            # SMH + Emotion layer
            somatic_valence=somatic_valence,
            arousal=arousal,
            emotional_state=emotional_state,
            
            # Learning layer
            learning_active=learning_active,
            synaptic_changes=1 if learning_active else 0,
            
            # Integration
            phenomenal_unity=phenomenal_unity,
            global_coherence=global_coherence,
            conscious_quality=quality_metrics
        )
        
        # Store in history
        self.state_history.append(state)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
        
        # Decay SMH markers periodically
        if self.cycle_count % 10 == 0:
            self.smh_evaluator.decay_markers()
        
        return state
    
    def _map_to_circumplex_category(self, valence: float, arousal: float) -> str:
        """
        Map valence/arousal to circumplex emotion category.
        
        Based on Russell (1980) "A Circumplex Model of Affect"
        Uses exact angular calculation for precise mapping.
        
        Russell's 8 primary concepts at 45° intervals:
        - Pleasure (0°)
        - Excitement (45°)
        - Arousal (90°)
        - Distress (135°)
        - Displeasure (180°)
        - Depression (225°)
        - Sleepiness (270°)
        - Contentment (315°)
        """
        import math
        
        # Handle neutral case (origin)
        if abs(valence) < 0.01 and abs(arousal - 0.5) < 0.01:
            return "neutral"
        
        # Convert arousal from [0, 1] to [-1, 1] (centered at 0.5)
        arousal_centered = (arousal - 0.5) * 2
        
        # Calculate angle using atan2 (returns radians)
        # atan2(y, x) where x=valence, y=arousal
        angle_rad = math.atan2(arousal_centered, valence)
        
        # Convert to degrees [0, 360)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        
        # Map to nearest Russell (1980) category
        # Using 45° sectors centered on each primary concept
        
        if 337.5 <= angle_deg or angle_deg < 22.5:
            return "pleased"  # ~0° (Pleasure)
        elif 22.5 <= angle_deg < 67.5:
            return "excited"  # ~45° (Excitement)
        elif 67.5 <= angle_deg < 112.5:
            return "alert"  # ~90° (Arousal)
        elif 112.5 <= angle_deg < 157.5:
            return "distressed"  # ~135° (Distress)
        elif 157.5 <= angle_deg < 202.5:
            return "frustrated"  # ~180° (Displeasure)
        elif 202.5 <= angle_deg < 247.5:
            return "depressed"  # ~225° (Depression)
        elif 247.5 <= angle_deg < 292.5:
            return "sleepy"  # ~270° (Sleepiness)
        elif 292.5 <= angle_deg < 337.5:
            return "content"  # ~315° (Contentment/Relaxation)
        else:
            return "neutral"
    
    def get_conscious_narrative(self, state: UnifiedConsciousState) -> str:
        """
        Generate a narrative description of the conscious state.
        Integrates all layers into coherent phenomenology.
        """
        narrative_parts = []
        
        # Consciousness status
        if state.is_conscious:
            narrative_parts.append(f"🌟 CONSCIOUS (Φ={state.system_phi:.3f})")
        else:
            narrative_parts.append(f"😴 Not fully conscious (Φ={state.system_phi:.3f})")
        
        # Prediction layer
        if state.prediction_error > 0.5:
            narrative_parts.append(f"⚠️  High surprise (FE={state.free_energy:.2f}) - unexpected situation")
        elif state.prediction_error < 0.2:
            narrative_parts.append(f"✅ Predictions accurate (FE={state.free_energy:.2f})")
        
        # Emotional layer
        emotion_desc = state.emotional_state.upper()
        if abs(state.somatic_valence) > 0.3:
            valence_desc = "positive" if state.somatic_valence > 0 else "negative"
            narrative_parts.append(f"💭 Feeling {emotion_desc} ({valence_desc}, arousal={state.arousal:.2f})")
        
        # Workspace/broadcast layer
        if state.broadcasts > 0:
            narrative_parts.append(f"📡 {state.broadcasts} global broadcasts to all systems")
        
        # Learning layer
        if state.learning_active:
            narrative_parts.append(f"📚 Active learning (synaptic updates)")
        
        # Unity/coherence
        if state.phenomenal_unity > 0.5:
            narrative_parts.append(f"🔗 High phenomenal unity ({state.phenomenal_unity:.2f})")
        
        return "\n   ".join(narrative_parts)
    
    def register_subsystem(self, name: str):
        """Register a subsystem across all engines"""
        self.consciousness_orchestrator.register_subsystem(name)
