"""
Free Energy Principle (FEP) Engine
Based on Karl Friston's Free Energy Principle (2010)

Key Concepts:
1. The brain minimizes "surprise" (prediction error)
2. Free Energy = upper bound on surprise
3. Predictive Coding: top-down predictions vs bottom-up sensory input
4. Active Inference: act to confirm predictions

References:
- Friston (2010) "The free-energy principle: a unified brain theory"
- Friston & Kiebel (2009) "Predictive coding under the free-energy principle"
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class PredictiveState:
    """
    Represents a predictive state in the hierarchical generative model
    """
    level: int  # Hierarchical level (0=sensory, higher=abstract)
    prediction: np.ndarray  # Top-down prediction
    prediction_error: np.ndarray  # Bottom-up error
    precision: float  # Confidence in prediction (inverse variance)
    free_energy: float  # Free energy at this level

@dataclass
class GenerativeModel:
    """
    Hierarchical generative model for predictions
    Each level predicts the level below
    """
    num_levels: int
    states: List[PredictiveState] = field(default_factory=list)
    learning_rate: float = 0.1
    precision_decay: float = 0.95

class FEPEngine:
    """
    Free Energy Principle Engine
    
    Implements:
    1. Predictive coding (hierarchical prediction)
    2. Prediction error minimization
    3. Free energy calculation
    4. Active inference (action selection)
    
    Integration with IIT/GWT:
    - Prediction errors have high salience for GWT competition
    - Minimized free energy indicates integrated coherent state (high Φ)
    """
    
    def __init__(self, num_hierarchical_levels: int = 3):
        self.num_levels = num_hierarchical_levels
        
        # Generative model (internal world model)
        self.generative_model = GenerativeModel(num_levels=num_hierarchical_levels)
        
        # History of predictions and errors
        self.prediction_history: List[Dict[str, np.ndarray]] = []
        self.max_history = 100
        
        # Learning parameters
        self.learning_rate = 0.1
        self.precision_learning_rate = 0.05
        
        # Initialize hierarchical states
        self._initialize_hierarchy()
        
    def _initialize_hierarchy(self):
        """Initialize the hierarchical predictive states"""
        for level in range(self.num_levels):
            state = PredictiveState(
                level=level,
                prediction=np.zeros(1),  # Will be resized on first input
                prediction_error=np.zeros(1),
                precision=1.0,  # Start with high confidence
                free_energy=0.0
            )
            self.generative_model.states.append(state)
    
    def process_observation(self, 
                           observation: Dict[str, float],
                           context: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Process a new observation through predictive coding.
        
        Steps:
        1. Generate top-down predictions
        2. Calculate prediction errors (bottom-up)
        3. Update beliefs to minimize free energy
        4. Calculate total free energy
        
        Args:
            observation: Current sensory input (subsystem states)
            context: Optional contextual information
            
        Returns:
            Dictionary with predictions, errors, and free energy
        """
        # Convert observation to array with safe scalar extraction
        obs_keys = sorted(observation.keys())
        obs_values = []

        def safe_numeric(val):
            """Convert anything to a float safely."""
            try:
                if isinstance(val, (int, float, np.number)):
                    return float(val)
                elif isinstance(val, (list, tuple, np.ndarray)):
                    # Extract only numeric values
                    nums = [float(x) for x in val if isinstance(x, (int, float, np.number))]
                    return float(np.mean(nums)) if nums else 0.0
                else:
                    return 0.0
            except Exception:
                return 0.0

        for key in obs_keys:
            val = observation[key]
            numeric_val = safe_numeric(val)
            obs_values.append(numeric_val)

        obs_vector = np.array(obs_values)
        
        # STEP 1: Generate predictions (top-down)
        predictions = self._generate_predictions(obs_vector)
        
        # STEP 2: Calculate prediction errors (bottom-up)
        errors = self._calculate_prediction_errors(obs_vector, predictions)
        
        # STEP 3: Update internal model (minimize free energy)
        self._update_beliefs(errors)
        
        # STEP 4: Calculate total free energy
        total_free_energy = self._calculate_free_energy(obs_vector, predictions, errors)
        
        # Store in history
        self.prediction_history.append({
            'observation': obs_vector,
            'predictions': predictions,
            'errors': errors,
            'free_energy': total_free_energy
        })
        if len(self.prediction_history) > self.max_history:
            self.prediction_history.pop(0)
        
        # Prepare output
        result = {
            'observations': observation,
            'predictions': {obs_keys[i]: predictions[0][i] for i in range(len(obs_keys))},
            'prediction_errors': {obs_keys[i]: errors[0][i] for i in range(len(obs_keys))},
            'free_energy': total_free_energy,
            'surprise': total_free_energy,  # Free energy upper-bounds surprise
            'precision': self.generative_model.states[0].precision,
            'hierarchical_states': [
                {
                    'level': state.level,
                    'prediction': state.prediction.tolist(),
                    'error': state.prediction_error.tolist(),
                    'precision': state.precision,
                    'free_energy': state.free_energy
                }
                for state in self.generative_model.states
            ]
        }
        
        return result
    
    def _generate_predictions(self, observation: np.ndarray) -> List[np.ndarray]:
        """
        Generate top-down predictions at each hierarchical level.

        Higher levels predict lower levels based on learned patterns.
        """
        predictions = []

        # Ensure all states have the correct size for current observation
        obs_size = len(observation)
        for state in self.generative_model.states:
            if len(state.prediction) != obs_size:
                state.prediction = np.zeros(obs_size)
                state.prediction_error = np.zeros(obs_size)

        # Generate predictions from each level
        for level, state in enumerate(self.generative_model.states):
            if level == self.num_levels - 1:
                # Highest level: use learned prior
                prediction = state.prediction.copy()  # Ensure it's the right size
            else:
                # Lower levels: predict from level above + own state
                higher_state = self.generative_model.states[level + 1]
                # Combine higher-level influence with current state
                prediction = 0.7 * state.prediction + 0.3 * higher_state.prediction
                # Ensure prediction is the right size (safety check)
                if len(prediction) != obs_size:
                    prediction = np.zeros(obs_size)

            predictions.append(prediction)

        return predictions
    
    def _calculate_prediction_errors(self, 
                                    observation: np.ndarray,
                                    predictions: List[np.ndarray]) -> List[np.ndarray]:
        """
        Calculate prediction errors at each level.
        
        Prediction Error = Observation - Prediction (weighted by precision)
        """
        errors = []
        
        for level, state in enumerate(self.generative_model.states):
            if level == 0:
                # Sensory level: error = observation - prediction
                error = observation - predictions[level]
            else:
                # Higher levels: error from level below
                lower_error = errors[level - 1]
                error = lower_error - predictions[level]
            
            # Weight by precision (confidence)
            error = error * state.precision
            
            errors.append(error)
            
            # Store in state
            state.prediction_error = error
        
        return errors
    
    def _update_beliefs(self, errors: List[np.ndarray]):
        """
        Update internal beliefs to minimize prediction error.
        
        This is the core of predictive coding:
        - Adjust predictions to reduce error
        - Update precision based on error magnitude
        """
        for level, state in enumerate(self.generative_model.states):
            error = errors[level]
            
            # Update prediction (move toward reducing error)
            state.prediction = state.prediction + self.learning_rate * error
            
            # Update precision (confidence)
            # High error → lower precision
            # Low error → higher precision
            error_magnitude = np.mean(np.abs(error))
            
            if error_magnitude > 0.1:
                # High error: reduce confidence
                state.precision *= self.generative_model.precision_decay
            else:
                # Low error: increase confidence
                state.precision = min(2.0, state.precision * 1.05)
    
    def _calculate_free_energy(self,
                               observation: np.ndarray,
                               predictions: List[np.ndarray],
                               errors: List[np.ndarray]) -> float:
        """
        Calculate free energy (upper bound on surprise).
        
        Free Energy = Σ (precision * error²) + KL_divergence
        
        Simplified version:
        F ≈ weighted sum of squared prediction errors
        """
        total_free_energy = 0.0
        
        for level, state in enumerate(self.generative_model.states):
            error = errors[level]
            
            # Precision-weighted squared error
            precision_weighted_error = state.precision * np.sum(error ** 2)
            
            # Free energy at this level
            level_free_energy = precision_weighted_error
            
            total_free_energy += level_free_energy
            state.free_energy = level_free_energy
        
        # Normalize by number of dimensions
        total_free_energy /= len(observation)
        
        return total_free_energy
    
    def select_action(self, current_state: Dict[str, float]) -> Dict[str, Any]:
        """
        Active Inference: Select action to minimize expected free energy.
        
        The agent should act to:
        1. Resolve uncertainty (explore)
        2. Fulfill prior preferences (exploit)
        
        Returns action recommendations based on prediction errors.
        """
        # Process current state
        result = self.process_observation(current_state)
        
        # High prediction error → need to act to resolve uncertainty
        errors = result['prediction_errors']
        
        # Identify subsystems with highest error
        error_magnitudes = {k: abs(v) for k, v in errors.items()}
        sorted_errors = sorted(error_magnitudes.items(), key=lambda x: x[1], reverse=True)
        
        # Action recommendations
        actions = []
        for subsystem, error_mag in sorted_errors[:3]:  # Top 3
            if error_mag > 0.1:
                actions.append({
                    'subsystem': subsystem,
                    'error': error_mag,
                    'action': 'increase_precision' if error_mag > 0.3 else 'gather_info',
                    'urgency': min(1.0, error_mag)
                })
        
        return {
            'free_energy': result['free_energy'],
            'recommended_actions': actions,
            'exploration_drive': result['free_energy']  # High FE → explore
        }
    
    def get_salience_weights(self) -> Dict[str, float]:
        """
        Get salience weights for GWT integration.
        
        Prediction errors are highly salient and should compete
        strongly for global workspace access.
        
        Returns:
            Dictionary of salience weights per subsystem
        """
        if not self.prediction_history:
            return {}
        
        latest = self.prediction_history[-1]
        errors = latest['errors'][0]  # Sensory level errors
        
        # Convert to salience (higher error = higher salience)
        # But cap at 1.0
        salience = np.clip(np.abs(errors), 0.0, 1.0)
        
        # Convert back to dictionary
        # (Assuming we stored the keys somewhere - simplified here)
        return {f"subsystem_{i}": salience[i] for i in range(len(salience))}
