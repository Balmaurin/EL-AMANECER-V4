"""
Advanced Theory of Mind API Endpoints
======================================

REST API endpoints for ToM Levels 8-10:
- Level 8: Multi-agent belief hierarchies
- Level 9: Strategic social reasoning
- Level 10: Cultural context modeling
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Initialize router
router = APIRouter(prefix="/api/v1/tom", tags=["Theory of Mind"])

# Import ToM system
try:
    import sys
    from pathlib import Path
    
    # Add consciousness package to path
    project_root = Path(__file__).parent.parent.parent.parent.parent
    sys.path.insert(0, str(project_root / "packages" / "consciousness" / "src"))
    
    from conciencia.modulos.teoria_mente import get_unified_tom
    
    # Initialize global ToM instance
    unified_tom = get_unified_tom(enable_advanced=True)
    TOM_AVAILABLE = True
    
except ImportError as e:
    print(f"⚠️  Theory of Mind not available: {e}")
    unified_tom = None
    TOM_AVAILABLE = False


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class BeliefHierarchyRequest(BaseModel):
    """Request to create a belief hierarchy"""
    agent_chain: List[str] = Field(..., description="Chain of agents [A, B, C, ...]", min_items=2)
    final_content: str = Field(..., description="Content of the deepest belief")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="Confidence level")


class BeliefHierarchyResponse(BaseModel):
    """Response with belief hierarchy details"""
    belief_id: str
    depth: int
    natural_language: str
    confidence: float
    agents_involved: List[str]
    created_at: str


class StrategicActionRequest(BaseModel):
    """Request to evaluate a strategic action"""
    actor: str = Field(..., description="Agent performing the action")
    target: str = Field(..., description="Target agent")
    strategy_type: str = Field(..., description="cooperation, competition, deception, etc.")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")


class StrategicActionResponse(BaseModel):
    """Response with strategic evaluation"""
    strategy: str
    expected_payoff: float
    risk_level: float
    ethical_score: float
    description: str
    predicted_responses: Dict[str, float]


class SocialInteractionRequest(BaseModel):
    """Request to process a social interaction"""
    actor: str = Field(..., description="Actor agent")
    target: str = Field(..., description="Target agent")
    interaction_type: str = Field(..., description="Type of interaction")
    content: Dict[str, Any] = Field(..., description="Interaction content")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SocialInteractionResponse(BaseModel):
    """Complete social interaction analysis"""
    cultural_context: Dict[str, Any]
    belief_analysis: Dict[str, Any]
    strategic_analysis: Dict[str, Any]
    suggested_response: Optional[str]
    tom_levels_active: List[int]
    timestamp: str


class CulturalAssignmentRequest(BaseModel):
    """Request to assign cultural background"""
    agent_id: str
    cultures: List[str]


class ToMStatusResponse(BaseModel):
    """Theory of Mind system status"""
    available: bool
    has_advanced: bool
    tom_level: float
    description: str
    statistics: Dict[str, Any]


# =============================================================================
# LEVEL 8: MULTI-AGENT BELIEF HIERARCHIES
# =============================================================================

@router.post("/belief-hierarchy", response_model=BeliefHierarchyResponse)
async def create_belief_hierarchy(request: BeliefHierarchyRequest):
    """
    Create a multi-agent belief hierarchy (Level 8 ToM).
    
    Example: "Alice believes that Bob believes that Charlie believes X"
    
    Args:
        agent_chain: List of agents in order [A, B, C, ...]
        final_content: The actual content of the deepest belief
        confidence: Confidence level (0.0 to 1.0)
        
    Returns:
        Belief hierarchy with ID and natural language description
    """
    if not TOM_AVAILABLE or not unified_tom.has_advanced_capabilities:
        raise HTTPException(
            status_code=503,
            detail="Advanced Theory of Mind (Level 8+) not available"
        )
    
    try:
        # Create hierarchy
        hierarchy_id = unified_tom.create_belief_hierarchy(
            agent_chain=request.agent_chain,
            final_content=request.final_content,
            confidence=request.confidence
        )
        
        # Get hierarchy details
        hierarchy = unified_tom.advanced_tom.belief_tracker.belief_networks[hierarchy_id]
        
        return BeliefHierarchyResponse(
            belief_id=hierarchy_id,
            depth=hierarchy.get_depth(),
            natural_language=hierarchy.to_natural_language(),
            confidence=request.confidence,
            agents_involved=request.agent_chain,
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating belief hierarchy: {str(e)}")


@router.get("/beliefs/{agent_id}")
async def get_agent_beliefs(agent_id: str, about_agent: Optional[str] = None):
    """
    Get beliefs held by an agent, optionally filtered by target agent.
    
    Args:
        agent_id: Agent whose beliefs to retrieve
        about_agent: Optional filter for epistemic beliefs about specific agent
        
    Returns:
        List of beliefs
    """
    if not TOM_AVAILABLE or not unified_tom.has_advanced_capabilities:
        raise HTTPException(
            status_code=503,
            detail="Advanced Theory of Mind not available"
        )
    
    try:
        tracker = unified_tom.advanced_tom.belief_tracker
        
        if about_agent:
            # Get epistemic beliefs about specific agent
            beliefs = tracker.query_beliefs_about_agent(agent_id, about_agent)
        else:
            # Get all beliefs
            belief_ids = tracker.agent_beliefs.get(agent_id, set())
            beliefs = [tracker.beliefs[bid] for bid in belief_ids]
        
        return {
            "agent_id": agent_id,
            "about_agent": about_agent,
            "belief_count": len(beliefs),
            "beliefs": [b.to_dict() for b in beliefs]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving beliefs: {str(e)}")


@router.get("/shared-beliefs")
async def get_shared_beliefs(agent_a: str, agent_b: str, threshold: float = 0.7):
    """
    Find shared beliefs between two agents (mutual knowledge).
    
    Args:
        agent_a: First agent
        agent_b: Second agent
        threshold: Minimum similarity threshold
        
    Returns:
        List of shared belief pairs
    """
    if not TOM_AVAILABLE or not unified_tom.has_advanced_capabilities:
        raise HTTPException(
            status_code=503,
            detail="Advanced Theory of Mind not available"
        )
    
    try:
        tracker = unified_tom.advanced_tom.belief_tracker
        shared = tracker.get_shared_beliefs(agent_a, agent_b, threshold)
        
        return {
            "agent_a": agent_a,
            "agent_b": agent_b,
            "threshold": threshold,
            "shared_count": len(shared),
            "shared_beliefs": [
                {
                    "agent_a_belief": belief_a.to_dict(),
                    "agent_b_belief": belief_b.to_dict()
                }
                for belief_a, belief_b in shared
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding shared beliefs: {str(e)}")


# =============================================================================
# LEVEL 9: STRATEGIC SOCIAL REASONING
# =============================================================================

@router.post("/strategic-action", response_model=StrategicActionResponse)
async def evaluate_strategic_action(request: StrategicActionRequest):
    """
    Evaluate a strategic social action using game theory (Level 9 ToM).
    
    Available strategies:
    - cooperation, competition, deception, manipulation
    - alliance, betrayal, negotiation, intimidation
    
    Returns:
        Strategic evaluation with payoff, risk, ethics, and predicted responses
    """
    if not TOM_AVAILABLE or not unified_tom.has_advanced_capabilities:
        raise HTTPException(
            status_code=503,
            detail="Advanced Theory of Mind (Level 9+) not available"
        )
    
    try:
        # Evaluate strategy
        result = unified_tom.evaluate_strategic_action(
            actor=request.actor,
            target=request.target,
            strategy_type=request.strategy_type,
            context=request.context
        )
        
        if result is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy type: {request.strategy_type}"
            )
        
        return StrategicActionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating strategy: {str(e)}")


@router.post("/detect-deception")
async def detect_deception(
    actor: str,
    stated_belief: str,
    actual_behavior: str
):
    """
    Detect potential deception by comparing stated beliefs with behavior.
    
    Args:
        actor: Agent being evaluated
        stated_belief: What the agent claims to believe
        actual_behavior: What the agent actually does
        
    Returns:
        Deception detection result with confidence
    """
    if not TOM_AVAILABLE or not unified_tom.has_advanced_capabilities:
        raise HTTPException(
            status_code=503,
            detail="Advanced Theory of Mind (Level 9+) not available"
        )
    
    try:
        reasoner = unified_tom.advanced_tom.strategic_reasoner
        is_deceptive, confidence = reasoner.detect_deception(
            actor, stated_belief, actual_behavior
        )
        
        return {
            "actor": actor,
            "is_deceptive": is_deceptive,
            "confidence": confidence,
            "stated_belief": stated_belief,
            "actual_behavior": actual_behavior,
            "analysis": "Contradiction detected" if is_deceptive else "Consistent behavior"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting deception: {str(e)}")


@router.get("/recommend-strategy")
async def recommend_strategy(
    actor: str,
    target: str,
    goal: str,
    ethical_constraint: float = 0.5
):
    """
    Recommend optimal strategy for achieving a goal with ethical constraints.
    
    Args:
        actor: Agent seeking recommendation
        target: Target agent
        goal: Desired outcome
        ethical_constraint: Minimum ethical score (0.0 to 1.0)
        
    Returns:
        Recommended strategic action
    """
    if not TOM_AVAILABLE or not unified_tom.has_advanced_capabilities:
        raise HTTPException(
            status_code=503,
            detail="Advanced Theory of Mind (Level 9+) not available"
        )
    
    try:
        reasoner = unified_tom.advanced_tom.strategic_reasoner
        recommended = reasoner.recommend_strategy(
            actor, target, goal, ethical_constraint
        )
        
        return {
            "actor": actor,
            "target": target,
            "goal": goal,
            "ethical_constraint": ethical_constraint,
            "recommended_strategy": recommended.action_type.value,
            "expected_payoff": recommended.expected_payoff,
            "risk_level": recommended.risk_level,
            "ethical_score": recommended.ethical_score,
            "description": recommended.description
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recommending strategy: {str(e)}")


# =============================================================================
# LEVEL 10: CULTURAL CONTEXT MODELING
# =============================================================================

@router.post("/assign-culture")
async def assign_culture(request: CulturalAssignmentRequest):
    """
    Assign cultural background to an agent (Level 10 ToM).
    
    Available cultures: western, eastern, professional, casual, etc.
    
    Args:
        agent_id: Agent identifier
        cultures: List of culture identifiers
        
    Returns:
        Confirmation of assignment
    """
    if not TOM_AVAILABLE or not unified_tom.has_advanced_capabilities:
        raise HTTPException(
            status_code=503,
            detail="Advanced Theory of Mind (Level 10) not available"
        )
    
    try:
        unified_tom.assign_culture(request.agent_id, request.cultures)
        
        return {
            "agent_id": request.agent_id,
            "cultures": request.cultures,
            "status": "assigned",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error assigning culture: {str(e)}")


@router.get("/cultural-context")
async def get_cultural_context(
    agent_a: str,
    agent_b: str,
    situation: str = "general"
):
    """
    Get cultural context for interaction between two agents.
    
    Args:
        agent_a: First agent
        agent_b: Second agent
        situation: Type of situation (business, casual, etc.)
        
    Returns:
        Cultural context with formality level and active norms
    """
    if not TOM_AVAILABLE or not unified_tom.has_advanced_capabilities:
        raise HTTPException(
            status_code=503,
            detail="Advanced Theory of Mind (Level 10) not available"
        )
    
    try:
        engine = unified_tom.advanced_tom.cultural_engine
        context = engine.get_cultural_context(agent_a, agent_b, situation)
        
        return {
            "agent_a": agent_a,
            "agent_b": agent_b,
            "situation": situation,
            "cultures": context.culture_ids,
            "formality_level": context.formality_level,
            "active_norms": len(context.active_norms),
            "norms": [
                {
                    "culture": norm.culture,
                    "category": norm.category,
                    "description": norm.description,
                    "importance": norm.importance
                }
                for norm in context.active_norms
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cultural context: {str(e)}")


# =============================================================================
# INTEGRATED: COMPLETE SOCIAL INTERACTION
# =============================================================================

@router.post("/social-interaction", response_model=SocialInteractionResponse)
async def process_social_interaction(request: SocialInteractionRequest):
    """
    Process a complete social interaction using all ToM levels (8-10).
    
    This endpoint combines:
    - Level 8: Belief modeling
    - Level 9: Strategic analysis
    - Level 10: Cultural appropriateness
    
    Returns:
        Complete analysis with cultural context, beliefs, strategy, and response
    """
    if not TOM_AVAILABLE or not unified_tom.has_advanced_capabilities:
        raise HTTPException(
            status_code=503,
            detail="Advanced Theory of Mind (Levels 8-10) not available"
        )
    
    try:
        # Process interaction asynchronously
        result = await unified_tom.process_social_interaction(
            actor=request.actor,
            target=request.target,
            interaction_type=request.interaction_type,
            content=request.content,
            context=request.context
        )
        
        return SocialInteractionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing interaction: {str(e)}")


# =============================================================================
# SYSTEM STATUS & MANAGEMENT
# =============================================================================

@router.get("/status", response_model=ToMStatusResponse)
async def get_tom_status():
    """
    Get comprehensive Theory of Mind system status.
    
    Returns:
        System status including ToM level, capabilities, and statistics
    """
    if not TOM_AVAILABLE:
        return ToMStatusResponse(
            available=False,
            has_advanced=False,
            tom_level=0.0,
            description="Theory of Mind not available not",
            statistics={}
        )
    
    try:
        # Get ToM level
        tom_level, description = unified_tom.get_tom_level()
        
        # Get comprehensive status
        status = unified_tom.get_comprehensive_status()
        
        return ToMStatusResponse(
            available=True,
            has_advanced=unified_tom.has_advanced_capabilities,
            tom_level=tom_level,
            description=description,
            statistics=status
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")


@router.get("/statistics")
async def get_belief_statistics():
    """Get detailed statistics about the belief tracking system."""
    if not TOM_AVAILABLE or not unified_tom.has_advanced_capabilities:
        raise HTTPException(
            status_code=503,
            detail="Advanced Theory of Mind not available"
        )
    
    try:
        tracker = unified_tom.advanced_tom.belief_tracker
        stats = tracker.get_statistics()
        
        return {
            "belief_tracking": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")


# Export router
__all__ = ["router"]
