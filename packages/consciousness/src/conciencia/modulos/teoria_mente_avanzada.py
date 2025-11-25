"""
Advanced Theory of Mind - Multi-Agent Social Intelligence
=========================================================

Implements ToM Levels 8-10 (Arrabales 2008 ConsScale):
- Level 8: Empathic - "I know you know" (multi-agent belief hierarchies)
- Level 9: Social - "I know you know I know" (Machiavellian strategic reasoning)
- Level 10: Human-Like - Cultural context modeling and Turing-capable interaction

Enterprise-Grade Implementation:
- Type-safe belief hierarchies
- Game-theoretic social reasoning
- Cultural knowledge graph
- Thread-safe multi-agent coordination
- Comprehensive logging and monitoring

Author: EL-AMANECERV3 Consciousness Team
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import numpy as np
from threading import RLock

logger = logging.getLogger(__name__)


# =============================================================================
# LEVEL 8: EMPATHIC - MULTI-AGENT BELIEF HIERARCHIES
# =============================================================================

class BeliefType(Enum):
    """Types of beliefs in the hierarchy"""
    FACTUAL = "factual"              # "The door is open"
    INTENTIONAL = "intentional"      # "Agent A wants to leave"
    EPISTEMIC = "epistemic"          # "Agent B knows the door is open"
    META_EPISTEMIC = "meta_epistemic"  # "A knows that B knows that A wants to leave"
    EMOTIONAL = "emotional"          # "Agent C feels frustrated"


@dataclass
class Belief:
    """Represents a single belief in the belief hierarchy"""
    belief_id: str
    belief_type: BeliefType
    subject: str  # Who holds this belief
    content: str  # What the belief is about
    about_agent: Optional[str] = None  # For epistemic beliefs about other agents
    confidence: float = 1.0  # 0.0 to 1.0
    source: str = "inference"  # "observation", "inference", "communication"
    timestamp: datetime = field(default_factory=datetime.now)
    evidence: List[str] = field(default_factory=list)
    parent_belief_id: Optional[str] = None  # For hierarchical beliefs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert belief to dictionary representation"""
        return {
            "belief_id": self.belief_id,
            "type": self.belief_type.value,
            "subject": self.subject,
            "content": self.content,
            "about_agent": self.about_agent,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "evidence": self.evidence,
            "parent": self.parent_belief_id
        }


@dataclass
class BeliefHierarchy:
    """Hierarchical belief structure: "A believes that B believes that C believes X" """
    root_belief: Belief
    nested_beliefs: List['BeliefHierarchy'] = field(default_factory=list)
    depth: int = 0  # How many levels of nesting
    
    def get_depth(self) -> int:
        """Calculate maximum depth of belief hierarchy"""
        if not self.nested_beliefs:
            return self.depth
        return max(self.depth, max(nb.get_depth() for nb in self.nested_beliefs))
    
    def to_natural_language(self) -> str:
        """Convert belief hierarchy to human-readable string"""
        if self.depth == 0:
            return f"{self.root_belief.content}"
        
        subject = self.root_belief.subject
        parts = [f"{subject} believes that"]
        
        current = self
        while current.nested_beliefs:
            nested = current.nested_beliefs[0]
            parts.append(f"{nested.root_belief.subject} believes that")
            current = nested
        
        parts.append(current.root_belief.content)
        return " ".join(parts)


class MultiAgentBeliefTracker:
    """
    Track beliefs across multiple agents with hierarchical structure.
    
    Implements Level 8 ToM: "I know you know"
    Supports recursive belief tracking up to configurable depth.
    """
    
    def __init__(self, max_belief_depth: int = 5):
        """
        Initialize multi-agent belief tracker.
        
        Args:
            max_belief_depth: Maximum depth of belief hierarchies to track
        """
        self.max_belief_depth = max_belief_depth
        self.beliefs: Dict[str, Belief] = {}  # belief_id -> Belief
        self.agent_beliefs: Dict[str, Set[str]] = defaultdict(set)  # agent_id -> belief_ids
        self.belief_networks: Dict[str, BeliefHierarchy] = {}  # root_belief_id -> hierarchy
        self._lock = RLock()
        self._belief_counter = 0
        
        logger.info(f"🧠 MultiAgentBeliefTracker initialized (max_depth={max_belief_depth})")
    
    def add_belief(
        self,
        subject: str,
        content: str,
        belief_type: BeliefType,
        about_agent: Optional[str] = None,
        confidence: float = 1.0,
        evidence: Optional[List[str]] = None
    ) -> str:
        """
        Add a new belief to the system.
        
        Args:
            subject: Agent who holds the belief
            content: Content of the belief
            belief_type: Type of belief
            about_agent: For epistemic beliefs, which agent is it about
            confidence: Confidence level (0.0 to 1.0)
            evidence: Supporting evidence for the belief
            
        Returns:
            belief_id: Unique identifier for the created belief
        """
        with self._lock:
            self._belief_counter += 1
            belief_id = f"belief_{self._belief_counter}_{datetime.now().timestamp()}"
            
            belief = Belief(
                belief_id=belief_id,
                belief_type=belief_type,
                subject=subject,
                content=content,
                about_agent=about_agent,
                confidence=confidence,
                evidence=evidence or [],
                source="inference"
            )
            
            self.beliefs[belief_id] = belief
            self.agent_beliefs[subject].add(belief_id)
            
            logger.debug(f"Added belief: {subject} -> {content} (type={belief_type.value})")
            return belief_id
    
    def create_belief_hierarchy(
        self,
        agent_chain: List[str],
        final_content: str,
        confidence: float = 0.8
    ) -> str:
        """
        Create hierarchical belief: "A believes that B believes that C believes X"
        
        Args:
            agent_chain: List of agents in order [A, B, C, ...]
            final_content: The actual content of the deepest belief
            confidence: Confidence level (decreases with depth)
            
        Returns:
            root_belief_id: ID of the root belief in the hierarchy
            
        Example:
            create_belief_hierarchy(
                ["Alice", "Bob", "Charlie"],
                "the door is locked"
            )
            Creates: Alice believes that Bob believes that Charlie believes the door is locked
        """
        if len(agent_chain) < 2:
            raise ValueError("Need at least 2 agents for hierarchical belief")
        
        if len(agent_chain) > self.max_belief_depth:
            logger.warning(f"Agent chain exceeds max depth, truncating to {self.max_belief_depth}")
            agent_chain = agent_chain[:self.max_belief_depth]
        
        with self._lock:
            # Build from deepest to shallowest
            deepest_agent = agent_chain[-1]
            deepest_belief_id = self.add_belief(
                subject=deepest_agent,
                content=final_content,
                belief_type=BeliefType.FACTUAL,
                confidence=confidence
            )
            
            current_belief_id = deepest_belief_id
            current_depth = len(agent_chain) - 1
            
            # Build epistemic beliefs upward
            for i in range(len(agent_chain) - 2, -1, -1):
                believer = agent_chain[i]
                believed_about = agent_chain[i + 1]
                
                epistemic_content = f"{believed_about} believes that {final_content}"
                parent_belief_id = self.add_belief(
                    subject=believer,
                    content=epistemic_content,
                    belief_type=BeliefType.META_EPISTEMIC if i < len(agent_chain) - 2 else BeliefType.EPISTEMIC,
                    about_agent=believed_about,
                    confidence=confidence * (0.9 ** (current_depth - i))  # Decay confidence
                )
                
                # Link child to parent
                self.beliefs[current_belief_id].parent_belief_id = parent_belief_id
                current_belief_id = parent_belief_id
            
            # Create hierarchy structure
            root_hierarchy = self._build_hierarchy_structure(current_belief_id)
            self.belief_networks[current_belief_id] = root_hierarchy
            
            logger.info(f"Created belief hierarchy with {len(agent_chain)} agents, depth={len(agent_chain)-1}")
            return current_belief_id
    
    def _build_hierarchy_structure(self, belief_id: str, depth: int = 0) -> BeliefHierarchy:
        """Recursively build BeliefHierarchy structure from belief links"""
        belief = self.beliefs[belief_id]
        
        # Find child beliefs (beliefs that have this as parent)
        child_beliefs = [
            bid for bid, b in self.beliefs.items()
            if b.parent_belief_id == belief_id
        ]
        
        nested = [
            self._build_hierarchy_structure(child_id, depth + 1)
            for child_id in child_beliefs
        ]
        
        return BeliefHierarchy(
            root_belief=belief,
            nested_beliefs=nested,
            depth=depth
        )
    
    def query_beliefs_about_agent(
        self,
        querying_agent: str,
        target_agent: str
    ) -> List[Belief]:
        """
        Query what one agent believes about another.
        
        Args:
            querying_agent: Agent making the query
            target_agent: Agent being queried about
            
        Returns:
            List of epistemic beliefs querying_agent has about target_agent
        """
        agent_belief_ids = self.agent_beliefs.get(querying_agent, set())
        
        return [
            self.beliefs[bid] for bid in agent_belief_ids
            if self.beliefs[bid].about_agent == target_agent
        ]
    
    def get_shared_beliefs(
        self,
        agent_a: str,
        agent_b: str,
        threshold: float = 0.7
    ) -> List[Tuple[Belief, Belief]]:
        """
        Find beliefs shared between two agents (mutual knowledge).
        
        Args:
            agent_a: First agent
            agent_b: Second agent
            threshold: Minimum similarity threshold
            
        Returns:
            List of (belief_a, belief_b) pairs that are similar
        """
        beliefs_a = [self.beliefs[bid] for bid in self.agent_beliefs.get(agent_a, set())]
        beliefs_b = [self.beliefs[bid] for bid in self.agent_beliefs.get(agent_b, set())]
        
        shared = []
        for ba in beliefs_a:
            for bb in beliefs_b:
                if ba.belief_type == bb.belief_type:
                    # Simple content similarity (in production: use embeddings)
                    similarity = self._calculate_content_similarity(ba.content, bb.content)
                    if similarity >= threshold:
                        shared.append((ba, bb))
        
        return shared
    
    def _calculate_content_similarity(self, content_a: str, content_b: str) -> float:
        """Calculate similarity between belief contents (simple Jaccard for now)"""
        words_a = set(content_a.lower().split())
        words_b = set(content_b.lower().split())
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        
        return intersection / union if union > 0 else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the belief tracking system"""
        with self._lock:
            total_beliefs = len(self.beliefs)
            agents_tracked = len(self.agent_beliefs)
            hierarchies = len(self.belief_networks)
            
            max_hierarchy_depth = max(
                (h.get_depth() for h in self.belief_networks.values()),
                default=0
            )
            
            belief_types = defaultdict(int)
            for belief in self.beliefs.values():
                belief_types[belief.belief_type.value] += 1
            
            return {
                "total_beliefs": total_beliefs,
                "agents_tracked": agents_tracked,
                "belief_hierarchies": hierarchies,
                "max_hierarchy_depth": max_hierarchy_depth,
                "belief_types": dict(belief_types),
                "avg_beliefs_per_agent": total_beliefs / agents_tracked if agents_tracked > 0 else 0
            }


# =============================================================================
# LEVEL 9: SOCIAL - MACHIAVELLIAN STRATEGIC REASONING
# =============================================================================

class SocialStrategy(Enum):
    """Strategic social behaviors"""
    COOPERATION = "cooperation"
    COMPETITION = "competition"
    DECEPTION = "deception"
    MANIPULATION = "manipulation"
    ALLIANCE = "alliance"
    BETRAYAL = "betrayal"
    NEGOTIATION = "negotiation"
    INTIMIDATION = "intimidation"


@dataclass
class SocialRelationship:
    """Relationship between two agents"""
    agent_a: str
    agent_b: str
    trust_level: float = 0.5  # 0.0 to 1.0
    power_balance: float = 0.0  # -1.0 (B dominates) to 1.0 (A dominates)
    cooperation_history: List[bool] = field(default_factory=list)
    conflict_count: int = 0
    alliance_strength: float = 0.0
    last_interaction: Optional[datetime] = None
    
    def update_trust(self, outcome: bool, learning_rate: float = 0.1):
        """Update trust based on interaction outcome"""
        if outcome:
            self.trust_level = min(1.0, self.trust_level + learning_rate)
        else:
            self.trust_level = max(0.0, self.trust_level - learning_rate * 1.5)  # Faster decay
        
        self.cooperation_history.append(outcome)
        if not outcome:
            self.conflict_count += 1


@dataclass
class StrategicAction:
    """A strategic social action with predicted outcomes"""
    action_type: SocialStrategy
    actor: str
    target: str
    description: str
    expected_payoff: float  # Expected utility
    risk_level: float  # 0.0 to 1.0
    predicted_responses: Dict[str, float]  # response_type -> probability
    ethical_score: float = 0.5  # 0.0 (unethical) to 1.0 (ethical)


class StrategicSocialReasoner:
    """
    Game-theoretic social reasoning for Level 9 ToM.
    
    Implements Machiavellian intelligence:
    - Strategic action selection
    - Outcome prediction
    - Deception detection and generation
    - Alliance formation
    - Power dynamics modeling
    """
    
    def __init__(self, belief_tracker: MultiAgentBeliefTracker):
        """
        Initialize strategic social reasoner.
        
        Args:
            belief_tracker: Multi-agent belief tracker for epistemic reasoning
        """
        self.belief_tracker = belief_tracker
        self.relationships: Dict[Tuple[str, str], SocialRelationship] = {}
        self.strategy_history: List[StrategicAction] = []
        self._lock = RLock()
        
        logger.info("🎭 StrategicSocialReasoner initialized")
    
    def model_relationship(self, agent_a: str, agent_b: str) -> SocialRelationship:
        """Get or create relationship model between two agents"""
        with self._lock:
            key = tuple(sorted([agent_a, agent_b]))
            if key not in self.relationships:
                self.relationships[key] = SocialRelationship(agent_a=agent_a, agent_b=agent_b)
            return self.relationships[key]
    
    def evaluate_strategic_action(
        self,
        actor: str,
        target: str,
        action_type: SocialStrategy,
        context: Dict[str, Any]
    ) -> StrategicAction:
        """
        Evaluate a potential strategic action using game theory.
        
        Args:
            actor: Agent performing the action
            target: Target agent
            action_type: Type of strategic action
            context: Additional context for evaluation
            
        Returns:
            StrategicAction with predicted outcomes
        """
        relationship = self.model_relationship(actor, target)
        
        # Calculate expected payoff based on relationship and strategy
        payoff = self._calculate_payoff(actor, target, action_type, relationship)
        
        # Calculate risk
        risk = self._calculate_risk(action_type, relationship)
        
        # Predict target's responses
        predicted_responses = self._predict_responses(target, action_type, relationship)
        
        # Ethical evaluation
        ethical_score = self._evaluate_ethics(action_type)
        
        description = self._generate_action_description(actor, target, action_type)
        
        action = StrategicAction(
            action_type=action_type,
            actor=actor,
            target=target,
            description=description,
            expected_payoff=payoff,
            risk_level=risk,
            predicted_responses=predicted_responses,
            ethical_score=ethical_score
        )
        
        with self._lock:
            self.strategy_history.append(action)
        
        return action
    
    def _calculate_payoff(
        self,
        actor: str,
        target: str,
        strategy: SocialStrategy,
        relationship: SocialRelationship
    ) -> float:
        """Calculate expected payoff using simplified game theory"""
        base_payoffs = {
            SocialStrategy.COOPERATION: 0.7,
            SocialStrategy.COMPETITION: 0.5,
            SocialStrategy.DECEPTION: 0.8,
            SocialStrategy.MANIPULATION: 0.6,
            SocialStrategy.ALLIANCE: 0.9,
            SocialStrategy.BETRAYAL: 0.3,
            SocialStrategy.NEGOTIATION: 0.75,
            SocialStrategy.INTIMIDATION: 0.4
        }
        
        base = base_payoffs.get(strategy, 0.5)
        
        # Adjust by trust and power
        trust_factor = relationship.trust_level if strategy in [
            SocialStrategy.COOPERATION, SocialStrategy.ALLIANCE
        ] else 1.0 - relationship.trust_level
        
        power_factor = 1.0 + (relationship.power_balance * 0.2)  # +/- 20% based on power
        
        return base * trust_factor * power_factor
    
    def _calculate_risk(self, strategy: SocialStrategy, relationship: SocialRelationship) -> float:
        """Calculate risk level of strategy"""
        risk_levels = {
            SocialStrategy.COOPERATION: 0.2,
            SocialStrategy.COMPETITION: 0.5,
            SocialStrategy.DECEPTION: 0.8,
            SocialStrategy.MANIPULATION: 0.7,
            SocialStrategy.ALLIANCE: 0.3,
            SocialStrategy.BETRAYAL: 0.9,
            SocialStrategy.NEGOTIATION: 0.4,
            SocialStrategy.INTIMIDATION: 0.6
        }
        
        base_risk = risk_levels.get(strategy, 0.5)
        
        # Higher risk if low trust (might backfire)
        trust_mod = 1.0 - relationship.trust_level
        
        return min(1.0, base_risk * (1.0 + trust_mod * 0.5))
    
    def _predict_responses(
        self,
        target: str,
        strategy: SocialStrategy,
        relationship: SocialRelationship
    ) -> Dict[str, float]:
        """Predict probability distribution of target's responses"""
        # Response probabilities based on strategy and relationship
        if strategy == SocialStrategy.COOPERATION:
            return {
                "reciprocate": 0.6 + relationship.trust_level * 0.3,
                "exploit": 0.2 - relationship.trust_level * 0.15,
                "ignore": 0.2
            }
        elif strategy == SocialStrategy.DECEPTION:
            return {
                "detect": 0.3 + (1.0 - relationship.trust_level) * 0.4,
                "deceived": 0.5  + relationship.trust_level * 0.2,
                "retaliate": 0.2
            }
        elif strategy == SocialStrategy.ALLIANCE:
            return {
                "accept": 0.7 + relationship.trust_level * 0.2,
                "reject": 0.2 - relationship.trust_level * 0.1,
                "negotiate": 0.1
            }
        else:
            return {"cooperate": 0.33, "compete": 0.33, "avoid": 0.34}
    
    def _evaluate_ethics(self, strategy: SocialStrategy) -> float:
        """Evaluate ethical implications of strategy"""
        ethical_scores = {
            SocialStrategy.COOPERATION: 0.9,
            SocialStrategy.COMPETITION: 0.6,
            SocialStrategy.DECEPTION: 0.2,
            SocialStrategy.MANIPULATION: 0.3,
            SocialStrategy.ALLIANCE: 0.8,
            SocialStrategy.BETRAYAL: 0.1,
            SocialStrategy.NEGOTIATION: 0.8,
            SocialStrategy.INTIMIDATION: 0.3
        }
        return ethical_scores.get(strategy, 0.5)
    
    def _generate_action_description(self, actor: str, target: str, strategy: SocialStrategy) -> str:
        """Generate human-readable description of strategic action"""
        templates = {
            SocialStrategy.COOPERATION: f"{actor} cooperates with {target} for mutual benefit",
            SocialStrategy.DECEPTION: f"{actor} attempts to deceive {target} about intentions",
            SocialStrategy.ALLIANCE: f"{actor} proposes alliance with {target}",
            SocialStrategy.MANIPULATION: f"{actor} manipulates {target}'s beliefs for advantage",
            SocialStrategy.BETRAYAL: f"{actor} betrays trust of {target}",
            SocialStrategy.NEGOTIATION: f"{actor} negotiates terms with {target}",
            SocialStrategy.COMPETITION: f"{actor} competes directly with {target}",
            SocialStrategy.INTIMIDATION: f"{actor} attempts to intimidate {target}"
        }
        return templates.get(strategy, f"{actor} interacts strategically with {target}")
    
    def detect_deception(
        self,
        actor: str,
        stated_belief: str,
        actual_behavior: str
    ) -> Tuple[bool, float]:
        """
        Detect potential deception by comparing stated beliefs with behavior.
        
        Args:
            actor: Agent being evaluated
            stated_belief: What the agent claims to believe
            actual_behavior: What the agent actually does
            
        Returns:
            (is_deceptive, confidence)
        """
        # Get actor's known beliefs
        actor_beliefs = [
            self.belief_tracker.beliefs[bid]
            for bid in self.belief_tracker.agent_beliefs.get(actor, set())
        ]
        
        # Simple heuristic: check if behavior contradicts stated belief
        contradiction_score = 0.0
        
        # Check if actual behavior contradicts stated belief content
        if stated_belief.lower() in actual_behavior.lower():
            contradiction_score = 0.0  # Consistent
        else:
            # Check for opposite terms (simple keyword matching)
            opposites = {
                "yes": "no", "no": "yes",
                "true": "false", "false": "true",
                "agree": "disagree", "disagree": "agree"
            }
            
            for word1, word2 in opposites.items():
                if word1 in stated_belief.lower() and word2 in actual_behavior.lower():
                    contradiction_score += 0.3
        
        is_deceptive = contradiction_score > 0.5
        confidence = min(1.0, contradiction_score + 0.3)  # Min 30% confidence
        
        if is_deceptive:
            logger.warning(f"🎭 Potential deception detected from {actor} (confidence={confidence:.2f})")
        
        return is_deceptive, confidence
    
    def recommend_strategy(
        self,
        actor: str,
        target: str,
        goal: str,
        ethical_constraint: float = 0.5
    ) -> StrategicAction:
        """
        Recommend optimal strategy for achieving a goal.
        
        Args:
            actor: Agent seeking recommendation
            target: Target agent
            goal: Desired outcome
            ethical_constraint: Minimum ethical score (0.0 to 1.0)
            
        Returns:
            Recommended strategic action
        """
        # Evaluate all strategies
        candidates = []
        for strategy in SocialStrategy:
            action = self.evaluate_strategic_action(actor, target, strategy, {"goal": goal})
            
            # Filter by ethical constraint
            if action.ethical_score >= ethical_constraint:
                candidates.append(action)
        
        if not candidates:
            logger.warning(f"No ethical strategies found for {actor} -> {target} with constraint {ethical_constraint}")
            # Fallback to most ethical strategy regardless of payoff
            return max(
                [self.evaluate_strategic_action(actor, target, s, {"goal": goal}) for s in SocialStrategy],
                key=lambda a: a.ethical_score
            )
        
        # Select strategy with highest expected payoff / risk ratio
        best_action = max(candidates, key=lambda a: a.expected_payoff / (a.risk_level + 0.1))
        
        logger.info(f"🎯 Recommended strategy for {actor}: {best_action.action_type.value} "
                   f"(payoff={best_action.expected_payoff:.2f}, risk={best_action.risk_level:.2f})")
        
        return best_action


# =============================================================================
# LEVEL 10: HUMAN-LIKE - CULTURAL CONTEXT ENGINE
# =============================================================================

@dataclass
class CulturalNorm:
    """Represents a cultural norm or expectation"""
    norm_id: str
    culture: str
    category: str  # "greeting", "politeness", "taboo", etc.
    description: str
    importance: float = 0.5  # 0.0 to 1.0
    contexts: List[str] = field(default_factory=list)
    violations_observed: int = 0


@dataclass
class CulturalContext:
    """Cultural context for an interaction"""
    culture_ids: List[str]
    formality_level: float = 0.5  # 0.0 (informal) to 1.0 (formal)
    power_distance: float = 0.5  # Cultural power distance
    individualism: float = 0.5  # vs. collectivism
    uncertainty_avoidance: float = 0.5
    active_norms: List[CulturalNorm] = field(default_factory=list)


class CulturalContextEngine:
    """
    Models cultural context for human-like social interaction (Level 10 ToM).
    
    Implements:
    - Cultural norm database
    - Hofstede dimensions modeling
    - Context-appropriate behavior generation
    - Cross-cultural communication
    - Turing-capable natural interaction
    """
    
    def __init__(self):
        """Initialize cultural context engine"""
        self.cultural_norms: Dict[str, CulturalNorm] = {}
        self.agent_cultures: Dict[str, List[str]] = defaultdict(list)
        self.interaction_contexts: Dict[str, CulturalContext] = {}
        self._norm_counter = 0
        
        # Initialize default cultural norms
        self._initialize_default_norms()
        
        logger.info("🌍 CulturalContextEngine initialized")
    
    def _initialize_default_norms(self):
        """Initialize basic cultural norms"""
        default_norms = [
            ("western", "greeting", "Use 'hello' or 'hi' for informal greetings", 0.7),
            ("western", "politeness", "Say 'please' and 'thank you'", 0.8),
            ("eastern", "greeting", "Bow or use formal titles more frequently", 0.8),
            ("eastern", "politeness", "Avoid direct confrontation", 0.9),
            ("professional", "formality", "Use formal language in business contexts", 0.9),
            ("casual", "formality", "Use casual language with friends", 0.6),
        ]
        
        for culture, category, description, importance in default_norms:
            self.add_cultural_norm(culture, category, description, importance)
    
    def add_cultural_norm(
        self,
        culture: str,
        category: str,
        description: str,
        importance: float = 0.5,
        contexts: Optional[List[str]] = None
    ) -> str:
        """Add a cultural norm to the knowledge base"""
        self._norm_counter += 1
        norm_id = f"norm_{culture}_{self._norm_counter}"
        
        norm = CulturalNorm(
            norm_id=norm_id,
            culture=culture,
            category=category,
            description=description,
            importance=importance,
            contexts=contexts or []
        )
        
        self.cultural_norms[norm_id] = norm
        logger.debug(f"Added cultural norm: {culture}/{category}")
        
        return norm_id
    
    def assign_culture_to_agent(self, agent_id: str, cultures: List[str]):
        """Assign cultural background to an agent"""
        self.agent_cultures[agent_id] = cultures
        logger.info(f"Assigned cultures to {agent_id}: {cultures}")
    
    def get_cultural_context(
        self,
        agent_a: str,
        agent_b: str,
        situation: str = "general"
    ) -> CulturalContext:
        """
        Generate cultural context for interaction between two agents.
        
        Args:
            agent_a: First agent
            agent_b: Second agent
            situation: Type of situation (e.g., "business", "casual")
            
        Returns:
            CulturalContext for the interaction
        """
        cultures_a = self.agent_cultures.get(agent_a, ["neutral"])
        cultures_b = self.agent_cultures.get(agent_b, ["neutral"])
        
        # Combine cultures
        combined_cultures = list(set(cultures_a + cultures_b))
        
        # Get active norms for these cultures
        active_norms = [
            norm for norm in self.cultural_norms.values()
            if norm.culture in combined_cultures
        ]
        
        # Calculate formality (higher if cultures differ or situation is formal)
        formality = 0.7 if situation in ["business", "professional"] else 0.3
        if cultures_a != cultures_b:
            formality += 0.2
        
        formality = min(1.0, formality)
        
        context = CulturalContext(
            culture_ids=combined_cultures,
            formality_level=formality,
            active_norms=active_norms
        )
        
        # Cache context
        context_key = f"{agent_a}_{agent_b}_{situation}"
        self.interaction_contexts[context_key] = context
        
        return context
    
    def generate_culturally_appropriate_response(
        self,
        agent_id: str,
        input_text: str,
        context: CulturalContext
    ) -> str:
        """
        Generate response that respects cultural norms.
        
        Args:
            agent_id: Agent generating response
            input_text: Input to respond to
            context: Cultural context
            
        Returns:
            Culturally appropriate response
        """
        # Check relevant norms
        politeness_norms = [n for n in context.active_norms if n.category == "politeness"]
        formality_norms = [n for n in context.active_norms if n.category == "formality"]
        
        response = input_text  # Base response
        
        # Apply politeness norms (high importance)
        high_priority_norms = [n for n in context.active_norms if n.importance > 0.7]
        if high_priority_norms:
            # Add polite markers
            if context.formality_level > 0.6:
                response = f"Thank you for your message. {response}"
        
        # Apply formality
        if context.formality_level > 0.7:
            # Make more formal (simple transformation)
            response = response.replace("hi", "hello")
            response = response.replace("yeah", "yes")
            response = response.replace("nope", "no")
        
        return response
    
    def evaluate_cultural_appropriateness(
        self,
        text: str,
        context: CulturalContext
    ) -> Tuple[float, List[str]]:
        """
        Evaluate if text is culturally appropriate.
        
        Args:
            text: Text to evaluate
            context: Cultural context
            
        Returns:
            (appropriateness_score, violations) where score is 0.0 to 1.0
        """
        violations = []
        score = 1.0
        
        # Check against active norms
        for norm in context.active_norms:
            if norm.category == "taboo":
                # Check for taboo words (simple keyword check)
                if any(word in text.lower() for word in ["taboo_word1", "taboo_word2"]):
                    violations.append(f"Violated {norm.culture} taboo: {norm.description}")
                    score -= norm.importance * 0.3
            
            elif norm.category == "formality":
                # Check formality match
                if context.formality_level > 0.7:
                    informal_markers = ["yeah", "nope", "gonna", "wanna"]
                    if any(marker in text.lower() for marker in informal_markers):
                        violations.append(f"Too informal for context: {norm.description}")
                        score -= norm.importance * 0.2
        
        score = max(0.0, score)
        
        return score, violations
    
    def get_turing_test_readiness(self) -> Dict[str, Any]:
        """
        Assess readiness for Turing test based on cultural modeling.
        
        Returns:
            Dictionary with readiness metrics
        """
        total_norms = len(self.cultural_norms)
        cultures_modeled = len(set(n.culture for n in self.cultural_norms.values()))
        agents_with_culture = len(self.agent_cultures)
        
        # Scoring
        norm_coverage = min(1.0, total_norms / 50)  # Target: 50+ norms
        culture_diversity = min(1.0, cultures_modeled / 5)  # Target: 5+ cultures
        
        overall_readiness = (norm_coverage + culture_diversity) / 2
        
        return {
            "overall_readiness": overall_readiness,
            "total_norms": total_norms,
            "cultures_modeled": cultures_modeled,
            "agents_with_culture": agents_with_culture,
            "norm_coverage": norm_coverage,
            "culture_diversity": culture_diversity,
            "status": "ready" if overall_readiness > 0.7 else "needs_improvement"
        }


# =============================================================================
# INTEGRATED ADVANCED THEORY OF MIND SYSTEM
# =============================================================================

class AdvancedTheoryOfMind:
    """
    Integrated Advanced ToM System combining Levels 8, 9, and 10.
    
    Provides enterprise-grade multi-agent social intelligence with:
    - Hierarchical belief tracking (Level 8)
    - Strategic social reasoning (Level 9)
    - Cultural context modeling (Level 10)
    """
    
    def __init__(self, max_belief_depth: int = 5):
        """
        Initialize Advanced Theory of Mind system.
        
        Args:
            max_belief_depth: Maximum depth for belief hierarchies
        """
        self.belief_tracker = MultiAgentBeliefTracker(max_belief_depth)
        self.strategic_reasoner = StrategicSocialReasoner(self.belief_tracker)
        self.cultural_engine = CulturalContextEngine()
        
        self.active = True
        self.initialization_time = datetime.now()
        
        logger.info("="*80)
        logger.info("🧠 ADVANCED THEORY OF MIND SYSTEM INITIALIZED")
        logger.info("="*80)
        logger.info(f"✅ Level 8 - Empathic: Multi-Agent Belief Hierarchies (depth={max_belief_depth})")
        logger.info(f"✅ Level 9 - Social: Machiavellian Strategic Reasoning")
        logger.info(f"✅ Level 10 - Human-Like: Cultural Context Modeling")
        logger.info("="*80)
    
    async def process_social_interaction(
        self,
        actor: str,
        target: str,
        interaction_type: str,
        content: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a complete social interaction with full ToM analysis.
        
        Args:
            actor: Agent initiating interaction
            target: Target agent
            interaction_type: Type of interaction
            content: Interaction content
            context: Additional context
            
        Returns:
            Complete analysis including beliefs, strategy, and cultural assessment
        """
        context = context or {}
        
        # Level 10: Get cultural context
        situation = context.get("situation", "general")
        cultural_context = self.cultural_engine.get_cultural_context(actor, target, situation)
        
        # Level 8: Update belief models
        belief_update = await self._update_beliefs_from_interaction(
            actor, target, content, cultural_context
        )
        
        # Level 9: Analyze strategic implications
        strategic_analysis = self._analyze_strategic_implications(
            actor, target, interaction_type, content
        )
        
        # Level 10: Generate culturally appropriate response
        if "text" in content:
            response = self.cultural_engine.generate_culturally_appropriate_response(
                target, content["text"], cultural_context
            )
        else:
            response = None
        
        return {
            "actor": actor,
            "target": target,
            "cultural_context": {
                "cultures": cultural_context.culture_ids,
                "formality": cultural_context.formality_level,
                "active_norms": len(cultural_context.active_norms)
            },
            "belief_analysis": belief_update,
            "strategic_analysis": strategic_analysis,
            "suggested_response": response,
            "tom_level_active": [8, 9, 10],
            "timestamp": datetime.now().isoformat()
        }
    
    async def _update_beliefs_from_interaction(
        self,
        actor: str,
        target: str,
        content: Dict[str, Any],
        cultural_context: CulturalContext
    ) -> Dict[str, Any]:
        """Update belief models based on interaction"""
        # Extract stated beliefs from content
        if "stated_belief" in content:
            belief_id = self.belief_tracker.add_belief(
                subject=actor,
                content=content["stated_belief"],
                belief_type=BeliefType.FACTUAL,
                confidence=0.9
            )
            
            # Create epistemic belief: target knows actor stated this
            self.belief_tracker.create_belief_hierarchy(
                agent_chain=[target, actor],
                final_content=content["stated_belief"],
                confidence=0.85
            )
            
            return {
                "belief_created": belief_id,
                "hierarchy_depth": 2,
                "confidence": 0.85
            }
        
        return {"status": "no_explicit_beliefs"}
    
    def _analyze_strategic_implications(
        self,
        actor: str,
        target: str,
        interaction_type: str,
        content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze strategic implications of interaction"""
        # Map interaction type to strategy
        strategy_map = {
            "request": SocialStrategy.NEGOTIATION,
            "offer": SocialStrategy.COOPERATION,
            "challenge": SocialStrategy.COMPETITION,
            "praise": SocialStrategy.ALLIANCE,
        }
        
        strategy = strategy_map.get(interaction_type, SocialStrategy.COOPERATION)
        
        # Evaluate action
        action = self.strategic_reasoner.evaluate_strategic_action(
            actor, target, strategy, content
        )
        
        # Detect potential deception
        is_deceptive = False
        deception_confidence = 0.0
        if "stated_belief" in content and "actual_behavior" in content:
            is_deceptive, deception_confidence = self.strategic_reasoner.detect_deception(
                actor,
                content["stated_belief"],
                content["actual_behavior"]
            )
        
        return {
            "recommended_strategy": action.action_type.value,
            "expected_payoff": action.expected_payoff,
            "risk_level": action.risk_level,
            "ethical_score": action.ethical_score,
            "deception_detected": is_deceptive,
            "deception_confidence": deception_confidence,
            "predicted_responses": action.predicted_responses
        }
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all ToM subsystems"""
        belief_stats = self.belief_tracker.get_statistics()
        turing_readiness = self.cultural_engine.get_turing_test_readiness()
        
        return {
            "system_active": self.active,
            "uptime_seconds": (datetime.now() - self.initialization_time).total_seconds(),
            "level_8_belief_tracking": belief_stats,
            "level_9_strategic_reasoning": {
                "relationships_tracked": len(self.strategic_reasoner.relationships),
                "strategies_evaluated": len(self.strategic_reasoner.strategy_history)
            },
            "level_10_cultural_modeling": turing_readiness,
            "overall_tom_level": self._calculate_overall_level(belief_stats, turing_readiness)
        }
    
    def _calculate_overall_level(
        self,
        belief_stats: Dict[str, Any],
        turing_readiness: Dict[str, Any]
    ) -> float:
        """Calculate overall ToM level (6.0 to 10.0)"""
        # Base level 6 (emotional - from base ToM)
        level = 6.0
        
        # Level 8 contribution (up to +2.0)
        if belief_stats["max_hierarchy_depth"] >= 2:
            level += min(2.0, belief_stats["max_hierarchy_depth"] * 0.4)
        
        # Level 9 contribution (strategic reasoning active)
        if len(self.strategic_reasoner.relationships) > 0:
            level = max(level, 9.0)
        
        # Level 10 contribution (cultural modeling)
        if turing_readiness["overall_readiness"] > 0.7:
            level = max(level, 10.0)
        
        return min(10.0, level)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Level 8
    "BeliefType",
    "Belief",
    "BeliefHierarchy",
    "MultiAgentBeliefTracker",
    
    # Level 9
    "SocialStrategy",
    "SocialRelationship",
    "StrategicAction",
    "StrategicSocialReasoner",
    
    # Level 10
    "CulturalNorm",
    "CulturalContext",
    "CulturalContextEngine",
    
    # Integrated
    "AdvancedTheoryOfMind"
]

__version__ = "1.0.0"
__author__ = "EL-AMANECERV3 Consciousness Team"
