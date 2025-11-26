"""
STDP Learning Module - Spike-Timing-Dependent Plasticity
Enhanced Hebbian learning with temporal causality

Based on:
- Keysers & Gazzola (2014): Hebbian learning and mirror neurons
- Bi & Poo (2001): Synaptic modification by correlated activity
- Widrow & Kim (2015): Hebbian-LMS with homeostasis

This module can be used to upgrade any Hebbian learning system
to use temporally-accurate STDP instead of simple correlation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TimestampedState:
    """State with timestamp for STDP"""
    timestamp: datetime
    state: Dict[str, float]

class STDPLearner:
    """
    Spike-Timing-Dependent Plasticity learner
    
    Key insight: "Cells that fire IN SEQUENCE wire together"
    Not: "Cells that fire TOGETHER wire together"
    
    STDP Window (Bi & Poo 2001):
    - Pre fires 0-40ms BEFORE Post → LTP (strengthen)
    - Post fires 0-40ms BEFORE Pre → LTD (weaken)
    - Outside window → No change
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 stdp_window_ms: float = 40.0,
                 tau_ltp: float = 20.0,
                 tau_ltd: float = 20.0):
        """
        Initialize STDP learner
        
        Args:
            learning_rate: Base learning rate
            stdp_window_ms: Asymmetric window (±40ms typical)
            tau_ltp: Time constant for LTP (Long-Term Potentiation)
            tau_ltd: Time constant for LTD (Long-Term Depression)
        """
        self.learning_rate = learning_rate
        self.stdp_window_ms = stdp_window_ms
        self.tau_ltp = tau_ltp
        self.tau_ltd = tau_ltd
        
        # Connection weights: {(pre_unit, post_unit): weight}
        self.weights: Dict[Tuple[str, str], float] = {}
        
        # State history with timestamps
        self.history: List[TimestampedState] = []
        self.max_history = 100
        
        # Contingency tracking (Bauer et al. 2001)
        self.contingency_window = timedelta(minutes=10)
        
        # Homeostatic bounds
        self.min_weight = 0.01
        self.max_weight = 1.0
    
    def update(self, state: Dict[str, float]):
        """
        Update with new state and apply STDP learning
        
        Args:
            state: Current activation state {unit_name: activation}
        """
        timestamped = TimestampedState(
            timestamp=datetime.now(),
            state=state
        )
        
        self.history.append(timestamped)
        
        # Trim history
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Apply STDP if we have at least 2 states
        if len(self.history) >= 2:
            self._apply_stdp()
    
    def _apply_stdp(self):
        """
        Apply STDP learning rule to recent state pairs
        
        STDP Rule:
        - If pre fires BEFORE post (causal): LTP (strengthen)
        - If post fires BEFORE pre (reverse): LTD (weaken)
        - Exponential decay with distance from optimal timing
        """
        # Look at recent consecutive pairs
        for i in range(len(self.history) - 1):
            t1_state = self.history[i]
            t2_state = self.history[i + 1]
            
            # Calculate time difference in milliseconds
            dt = (t2_state.timestamp - t1_state.timestamp).total_seconds() * 1000
            
            # Only learn if within STDP window
            if abs(dt) > self.stdp_window_ms:
                continue
            
            # Update weights based on STDP  
            for pre_unit in t1_state.state.keys():
                for post_unit in t2_state.state.keys():
                    self._update_weight_stdp(
                        pre_unit,
                        post_unit,
                        t1_state.state[pre_unit],
                        t2_state.state[post_unit],
                        dt
                    )
    
    def _update_weight_stdp(self,
                           pre_unit: str,
                           post_unit: str,
                           pre_act: float,
                           post_act: float,
                           dt_ms: float):
        """
        Update single weight using STDP rule

        Args:
            pre_unit: Presynaptic unit name
            post_unit: Postsynaptic unit name
            pre_act: Presynaptic activation
            post_act: Postsynaptic activation
            dt_ms: Time difference in milliseconds (positive = pre before post)
        """
        # Safely extract scalar values for activation - handle complex data types
        def safe_scalar(val):
            try:
                if np.isscalar(val):
                    return float(val)
                elif isinstance(val, (list, tuple, np.ndarray)):
                    return float(np.mean([float(x) if np.isscalar(x) else 0.0 for x in val])) if len(val) > 0 else 0.0
                else:
                    return 0.0
            except (ValueError, TypeError):
                return 0.0

        pre_act_scalar = safe_scalar(pre_act)
        post_act_scalar = safe_scalar(post_act)

        # Skip if no activation
        if pre_act_scalar < 0.01 or post_act_scalar < 0.01:
            return
        
        key = (pre_unit, post_unit)
        
        # Initialize if needed
        if key not in self.weights:
            self.weights[key] = self.min_weight
        
        # STDP asymmetric learning rule
        if dt_ms > 0:
            # Pre BEFORE Post → LTP (causality!)
            stdp_factor = np.exp(-dt_ms / self.tau_ltp)
            delta = self.learning_rate * stdp_factor * pre_act_scalar * post_act_scalar
            self.weights[key] += delta
        else:
            # Post BEFORE Pre → LTD (reverse)
            stdp_factor = np.exp(abs(dt_ms) / self.tau_ltd)
            delta = self.learning_rate * 0.5 * stdp_factor * pre_act_scalar * post_act_scalar
            self.weights[key] -= delta
        
        # Homeostatic bounds (Widrow & Kim 2015)
        self.weights[key] = np.clip(
            self.weights[key],
            self.min_weight,
            self.max_weight
        )
    
    def get_weight(self, pre_unit: str, post_unit: str) -> float:
        """Get connection weight"""
        return self.weights.get((pre_unit, post_unit), self.min_weight)
    
    def predict_next(self, current_state: Dict[str, float]) -> Dict[str, float]:
        """
        Predict next state based on learned STDP weights
        
        The 200ms delay in sensorimotor system means STDP learns
        to predict ~200ms into the future (Keysers & Gazzola 2014)
        
        Args:
            current_state: Current activation state
            
        Returns:
            Predicted next state
        """
        prediction = {}
        
        for post_unit in current_state.keys():
            # Sum weighted inputs
            total_input = 0.0
            for pre_unit, pre_act in current_state.items():
                weight = self.get_weight(pre_unit, post_unit)
                total_input += weight * pre_act
            
            # Sigmoid activation
            prediction[post_unit] = 1.0 / (1.0 + np.exp(-total_input))
        
        return prediction
    
    def get_connection_strength(self) -> Dict[Tuple[str, str], float]:
        """Get all connection weights"""
        return self.weights.copy()
    
    def get_summary(self) -> Dict[str, any]:
        """Get learning summary statistics"""
        if not self.weights:
            return {
                'total_connections': 0,
                'avg_weight': 0.0,
                'max_weight': 0.0,
                'states_seen': len(self.history)
            }
        
        weights_array = np.array(list(self.weights.values()))
        
        return {
            'total_connections': len(self.weights),
            'avg_weight': float(np.mean(weights_array)),
            'max_weight': float(np.max(weights_array)),
            'min_weight': float(np.min(weights_array)),
            'std_weight': float(np.std(weights_array)),
            'states_seen': len(self.history),
            'learning_active': len(self.history) >= 2
        }


class ContingencyTracker:
    """
    Tracks contingency for STDP learning (Bauer et al. 2001)
    
    Contingency: p(post|pre) >> p(post|~pre)
    
    STDP requires both:
    1. Contiguity: Close in time (~40ms)
    2. Contingency: Predictive over longer window (~10 min)
    """
    
    def __init__(self, window_minutes: float = 10.0):
        """
        Initialize contingency tracker
        
        Args:
            window_minutes: Time window for contingency (default 10 min)
        """
        self.window = timedelta(minutes=window_minutes)
        self.events: List[Tuple[datetime, str, bool]] = []  # (time, event, occurred)
    
    def add_event(self, event_name: str, occurred: bool):
        """Add event occurrence"""
        self.events.append((datetime.now(), event_name, occurred))
        
        # Clean old events
        cutoff = datetime.now() - self.window
        self.events = [e for e in self.events if e[0] > cutoff]
    
    def get_contingency(self, pre_event: str, post_event: str) -> float:
        """
        Calculate contingency: p(post|pre) / p(post|~pre)
        
        Returns:
            Contingency ratio (>1 = contingent, ~1 = not contingent)
        """
        # Count occurrences
        pre_and_post = 0
        pre_not_post = 0
        not_pre_post = 0
        not_pre_not_post = 0
        
        # Simplified: look at consecutive events
        for i in range(len(self.events) - 1):
            t1, e1, occ1 = self.events[i]
            t2, e2, occ2 = self.events[i + 1]
            
            if e1 == pre_event and e2 == post_event:
                if occ1 and occ2:
                    pre_and_post += 1
                elif occ1 and not occ2:
                    pre_not_post += 1
            elif e1 != pre_event and e2 == post_event:
                if occ2:
                    not_pre_post += 1
                else:
                    not_pre_not_post += 1
        
        # Calculate conditional probabilities
        if pre_and_post + pre_not_post > 0:
            p_post_given_pre = pre_and_post / (pre_and_post + pre_not_post)
        else:
            p_post_given_pre = 0.5
        
        if not_pre_post + not_pre_not_post > 0:
            p_post_given_not_pre = not_pre_post / (not_pre_post + not_pre_not_post)
        else:
            p_post_given_not_pre = 0.5
        
        # Contingency ratio
        if p_post_given_not_pre > 0:
            contingency = p_post_given_pre / p_post_given_not_pre
        else:
            contingency = 2.0 if p_post_given_pre > 0.5 else 1.0
        
        return contingency
