"""
IIT Engine with STDP Enhancement
Wraps existing IIT40Engine with STDP learning capabilities

This provides backward compatibility while enabling STDP
"""

from typing import Dict, Any
from .stdp_learner import STDPLearner

try:
    from .iit_40_engine import IIT40Engine as BaseIITEngine
except:
    # Fallback if IIT engine has issues
    class BaseIITEngine:
        def __init__(self, min_phi_conscious=0.1, learning_rate=0.01):
            self.min_phi_conscious = min_phi_conscious
            self.learning_rate = learning_rate
            self.virtual_tpm = {}
            self.state_history = []
        
        def calculate_system_phi(self, current_subsystems, debug=False):
            # Simple phi calculation as fallback
            if len(current_subsystems) < 2:
                return 0.0
            
            import numpy as np
            values = list(current_subsystems.values())
            integration = np.std(values) * len(values) ** 0.5
            return min(integration / 10.0, 1.0)


class IITEngineSTDP(BaseIITEngine):
    """
    IIT Engine enhanced with STDP learning
    
    Combines:
    - IIT 4.0 Phi calculation (from base)
    - STDP temporal learning (new)
    
    Usage:
        engine = IITEngineSTDP()
        engine.update_state(current_state)
        phi = engine.calculate_system_phi(current_state)
    """
    
    def __init__(self, 
                 min_phi_conscious: float = 0.1,
                 learning_rate: float = 0.01,
                 use_stdp: bool = True):
        """
        Initialize IIT Engine with optional STDP
        
        Args:
            min_phi_conscious: Minimum Φ for consciousness
            learning_rate: Learning rate
            use_stdp: Use STDP (True) or simple Hebbian (False)
        """
        super().__init__(min_phi_conscious, learning_rate)
        
        self.use_stdp = use_stdp
        
        if use_stdp:
            # Initialize STDP learner
            self.stdp = STDPLearner(
                learning_rate=learning_rate,
                stdp_window_ms=40.0,  # Bi & Poo 2001
                tau_ltp=20.0,
                tau_ltd=20.0
            )
        else:
            self.stdp = None
    
    def update_state(self, current_state: Dict[str, float]):
        """
        Update with new state
        
        If STDP enabled, uses temporal learning
        Otherwise falls back to base Hebbian
        """
        if self.use_stdp and self.stdp is not None:
            # STDP learning with timestamps
            self.stdp.update(current_state)
            
            # Sync STDP weights to virtual_tpm for IIT calculations
            self.virtual_tpm = self.stdp.get_connection_strength()
        else:
            # Fall back to base update
            super().update_state(current_state)
    
    def predict_next_state(self, current_state: Dict[str, float]) -> Dict[str, float]:
        """
        Predict next state using STDP-learned weights
        
        This uses the ~200ms predictive horizon that emerges
        from sensorimotor delays (Keysers & Gazzola 2014)
        """
        if self.use_stdp and self.stdp is not None:
            return self.stdp.predict_next(current_state)
        else:
            # No prediction without STDP
            return current_state
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress"""
        if self.use_stdp and self.stdp is not None:
            return self.stdp.get_summary()
        else:
            return {
                'learning_type': 'simple_hebbian',
                'connections': len(self.virtual_tpm),
                'states_seen': len(self.state_history)
            }
    
    def get_connection_weights(self) -> Dict[tuple, float]:
        """Get all learned connection weights"""
        return self.virtual_tpm.copy()
