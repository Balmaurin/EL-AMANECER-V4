"""
IIT 4.0 + GWT Integration Module
Integrates Integrated Information Theory 4.0 with Global Workspace Theory

Based on:
- Albantakis et al. (2023) - IIT 4.0
- Baars (1997) - Global Workspace Theory

Key Insight:
- IIT 4.0 determines WHAT is conscious (integrated information Φ)
- GWT determines HOW that information is broadcast globally
"""

import numpy as np
from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from .iit_stdp_engine import IITEngineSTDP

@dataclass
class WorkspaceContent:
    """Content in the global workspace (the 'bright spot' on stage)"""
    content_id: str
    source_mechanism: List[str]  # Which IIT distinction generated this
    information: Dict[str, Any]
    phi_d: float  # Distinction's integrated information
    salience: float  # Attention weight (spotlight intensity)
    timestamp: float

@dataclass
class BroadcastEvent:
    """A global broadcast event from the workspace"""
    broadcast_id: str
    content: WorkspaceContent
    audience: Set[str]  # Which subsystems receive this
    broadcast_strength: float
    timestamp: float

class IIT_GWT_Bridge:
    """
    Bridges IIT 4.0 and Global Workspace Theory
    
    IIT 4.0 provides:
    - Φ-structure (distinctions + relations)
    - Measure of integration (what is conscious)
    - Quality of consciousness
    
    GWT provides:
    - Global broadcasting mechanism
    - Competition for workspace access
    - Audience of unconscious processors
    """
    
    def __init__(self, iit_engine: IITEngineSTDP):
        self.iit_engine = iit_engine
        
        # Global Workspace (the 'stage')
        self.workspace_capacity = 3  # Limited capacity (Baars: ~7±2 items)
        self.current_workspace: List[WorkspaceContent] = []
        
        # Attention spotlight
        self.spotlight_focus: Optional[str] = None
        self.spotlight_intensity = 1.0
        
        # Audience (unconscious processors that receive broadcasts)
        self.audience_members: Set[str] = set()
        
        # Context operators (behind-the-scenes influences)
        self.context_operators: Dict[str, float] = {}
        
        # Broadcast history
        self.broadcast_history: List[BroadcastEvent] = []
        self.max_history = 100
        
    def register_audience_member(self, name: str):
        """Register an unconscious processor as audience member"""
        self.audience_members.add(name)
        
    def set_context(self, context_name: str, strength: float):
        """Set a context operator (unconscious influence)"""
        self.context_operators[context_name] = strength
        
    def compete_for_workspace(self, 
                              phi_structure: Dict[str, Any],
                              external_saliency: Dict[str, float] = None) -> List[WorkspaceContent]:
        """
        Competition for global workspace access.
        
        IIT distinctions compete based on:
        1. Integrated information (φd) - intrinsic salience
        2. External saliency (attention, relevance) - extrinsic salience
        3. Context operators (unconscious biases)
        
        Returns: Winners that enter the workspace
        """
        if external_saliency is None:
            external_saliency = {}
        
        # Extract distinctions from Φ-structure
        distinctions = phi_structure.get('distinctions', [])
        
        if not distinctions:
            return []
        
        # Calculate competition scores
        candidates = []
        for i, distinction in enumerate(distinctions):
            mechanism = distinction['mechanism']
            phi_d = distinction['phi_d']
            
            # Intrinsic salience (from IIT)
            intrinsic_salience = phi_d
            
            # Extrinsic salience (from attention/relevance)
            mechanism_key = '_'.join(mechanism)
            extrinsic_salience = external_saliency.get(mechanism_key, 0.5)
            
            # Context modulation
            context_modulation = 1.0
            for ctx, strength in self.context_operators.items():
                if any(ctx.lower() in m.lower() for m in mechanism):
                    context_modulation *= (1.0 + strength)
            
            # Total competition score
            competition_score = intrinsic_salience * extrinsic_salience * context_modulation
            
            content = WorkspaceContent(
                content_id=f"content_{i}",
                source_mechanism=mechanism,
                information=distinction.get('effect_state', {}),
                phi_d=phi_d,
                salience=competition_score,
                timestamp=np.datetime64('now').astype(float)
            )
            
            candidates.append((competition_score, content))
        
        # Sort by competition score (descending)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Select top N for workspace (limited capacity)
        winners = [content for score, content in candidates[:self.workspace_capacity]]
        
        return winners
        
    def update_workspace(self, 
                        phi_structure: Dict[str, Any],
                        external_saliency: Dict[str, float] = None) -> List[BroadcastEvent]:
        """
        Update global workspace with new content.
        
        Process:
        1. Competition among IIT distinctions
        2. Winners enter the workspace
        3. Content is broadcast to audience
        4. Unconscious processors receive and process broadcast
        
        Returns: List of broadcast events
        """
        # Competition for workspace access
        winners = self.compete_for_workspace(phi_structure, external_saliency)
        
        # Update workspace (replace old content)
        self.current_workspace = winners
        
        # Broadcast to audience
        broadcasts = []
        for content in self.current_workspace:
            # Determine broadcast strength
            # Higher phi_d = stronger broadcast
            broadcast_strength = content.phi_d * content.salience
            
            # Create broadcast event
            broadcast = BroadcastEvent(
                broadcast_id=f"broadcast_{len(self.broadcast_history)}",
                content=content,
                audience=self.audience_members.copy(),
                broadcast_strength=broadcast_strength,
                timestamp=np.datetime64('now').astype(float)
            )
            
            broadcasts.append(broadcast)
            
            # Store in history
            self.broadcast_history.append(broadcast)
            if len(self.broadcast_history) > self.max_history:
                self.broadcast_history.pop(0)
        
        return broadcasts
    
    def get_workspace_summary(self) -> Dict[str, Any]:
        """Get summary of current workspace state"""
        return {
            "capacity": self.workspace_capacity,
            "current_contents": len(self.current_workspace),
            "contents": [
                {
                    "mechanism": c.source_mechanism,
                    "phi_d": c.phi_d,
                    "salience": c.salience,
                    "info": c.information
                }
                for c in self.current_workspace
            ],
            "spotlight_focus": self.spotlight_focus,
            "audience_size": len(self.audience_members),
            "context_operators": self.context_operators,
            "recent_broadcasts": len(self.broadcast_history)
        }
    
    def apply_spotlight(self, content_id: str, intensity: float = 1.0):
        """
        Apply attention spotlight to specific workspace content.
        This amplifies the broadcast of that content.
        """
        self.spotlight_focus = content_id
        self.spotlight_intensity = intensity
        
        # Amplify salience of focused content
        for content in self.current_workspace:
            if content.content_id == content_id:
                content.salience *= (1.0 + intensity)


class ConsciousnessOrchestrator:
    """
    Master orchestrator combining IIT 4.0 and GWT
    
    Architecture:
    1. IIT Engine computes Φ-structure (integrated information)
    2. GWT Bridge selects content for global workspace (competition)
    3. Content is broadcast to unconscious processors (global access)
    4. Results feed back to influence next cycle
    """
    
    def __init__(self, iit_engine: IITEngineSTDP = None):
        # Use shared IIT+STDP engine if provided, otherwise create a new one
        self.iit_engine = iit_engine or IITEngineSTDP()
        self.gwt_bridge = IIT_GWT_Bridge(self.iit_engine)
        
        # Cycle counter
        self.cycle_count = 0
        
    def register_subsystem(self, name: str):
        """Register a subsystem as both IIT unit and GWT audience member"""
        self.gwt_bridge.register_audience_member(name)
        
    def process_conscious_moment(self,
                                 subsystem_states: Dict[str, float],
                                 external_attention: Dict[str, float] = None,
                                 contexts: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Process a complete conscious moment.
        
        Steps:
        1. Update IIT engine with current states
        2. Calculate Φ-structure (distinctions + relations)
        3. Compete for global workspace
        4. Broadcast winning content
        
        Returns complete state of consciousness
        """
        self.cycle_count += 1

        # Set contexts if provided
        if contexts:
            for ctx_name, strength in contexts.items():
                self.gwt_bridge.set_context(ctx_name, strength)

        # DEBUG: Check GWT → IIT states
        print("DEBUG - GWT → IIT states:", subsystem_states)

        # FIX: Expand GWT vector output into individual IIT nodes
        # IIT needs a distributed system of nodes, not a single vector
        if len(subsystem_states) == 1 and isinstance(next(iter(subsystem_states.values())), (list, tuple, np.ndarray)):
            # Expand single vector node into individual scalar nodes
            vector_key = next(iter(subsystem_states.keys()))
            vector_data = subsystem_states[vector_key]

            # Convert to individual named nodes
            expanded_states = {}
            for i, value in enumerate(vector_data):
                try:
                    numeric_value = float(value) if np.isscalar(value) else 0.0
                    expanded_states[f"unit_{i}"] = numeric_value
                except (ValueError, TypeError):
                    expanded_states[f"unit_{i}"] = 0.5

            print(f"DEBUG - IIT states expanded from {len(vector_data)} vector elements to {len(expanded_states)} discrete nodes")
            subsystem_states = expanded_states

        # 1. IIT: Update causal history
        self.iit_engine.update_state(subsystem_states)
        
        # 2. IIT: Calculate system Phi
        system_phi = self.iit_engine.calculate_system_phi(subsystem_states)
        
        # 3. Create simplified phi_structure from subsystem states
        # Since calculate_phi_structure doesn't exist, we create a minimal structure
        distinctions = []
        for i, (subsystem_name, value) in enumerate(subsystem_states.items()):
            # Ensure value is numeric for phi calculation
            try:
                numeric_value = float(value) if np.isscalar(value) else np.mean([float(v) for v in value if np.isscalar(v)]) if hasattr(value, '__iter__') and not isinstance(value, str) else 0.5
            except (ValueError, TypeError):
                numeric_value = 0.5  # Default fallback

            distinctions.append({
                'mechanism': [subsystem_name],
                'phi_d': numeric_value * system_phi,  # Individual contribution
                'effect_state': {subsystem_name: value},
                'cause_repertoire': {subsystem_name: value},
                'effect_repertoire': {subsystem_name: value}
            })
        
        # Safely compute informativeness from subsystem values
        try:
            numeric_values = []
            for v in subsystem_states.values():
                try:
                    numeric_values.append(float(v) if np.isscalar(v) else np.mean([float(x) for x in v if np.isscalar(x)]) if hasattr(v, '__iter__') and not isinstance(v, str) else 0.0)
                except (ValueError, TypeError):
                    numeric_values.append(0.0)
            informativeness = sum(numeric_values) / max(1, len(numeric_values))
        except (ValueError, TypeError):
            informativeness = 0.5

        phi_structure = {
            'system_phi': system_phi,
            'num_distinctions': len(distinctions),
            'distinctions': distinctions,
            'relations': [],  # Simplified - no relations computed
            'quality_metrics': {
                'unity': system_phi * 100,
                'complexity': len(distinctions) * system_phi,
                'informativeness': informativeness
            }
        }
        
        # 4. GWT: Competition and workspace update
        broadcasts = self.gwt_bridge.update_workspace(phi_structure, external_attention)
        
        # 5. GWT: Get workspace state
        workspace_summary = self.gwt_bridge.get_workspace_summary()
        
        return {
            "cycle": self.cycle_count,
            "is_conscious": system_phi > 0.1,
            "system_phi": system_phi,
            "phi_structure": phi_structure,
            "workspace": workspace_summary,
            "broadcasts": [
                {
                    "source": b.content.source_mechanism,
                    "strength": b.broadcast_strength,
                    "audience_size": len(b.audience)
                }
                for b in broadcasts
            ],
            "integration_quality": phi_structure.get('quality_metrics', {}),
            "global_access": len(broadcasts) > 0
        }
