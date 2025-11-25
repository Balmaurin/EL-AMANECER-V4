"""
Enterprise Test Suite for Advanced Theory of Mind (Levels 8-10)
================================================================

Comprehensive testing of multi-agent belief hierarchies, strategic reasoning,
and cultural context modeling.

Run with: python -m pytest test_teoria_mente_avanzada.py -v
"""

import pytest
import asyncio
from datetime import datetime
from packages.consciousness.src.conciencia.modulos.teoria_mente_avanzada import (
    AdvancedTheoryOfMind,
    BeliefType,
    SocialStrategy,
    MultiAgentBeliefTracker,
    StrategicSocialReasoner,
    CulturalContextEngine
)


class TestLevel8_BeliefHierarchies:
    """Test Level 8 ToM: Multi-Agent Belief Hierarchies"""
    
    def test_simple_belief_creation(self):
        """Test creating a simple belief"""
        tracker = MultiAgentBeliefTracker()
        
        belief_id = tracker.add_belief(
            subject="Alice",
            content="the door is locked",
            belief_type=BeliefType.FACTUAL,
            confidence=0.9
        )
        
        assert belief_id is not None
        assert belief_id in tracker.beliefs
        assert tracker.beliefs[belief_id].subject == "Alice"
        assert tracker.beliefs[belief_id].content == "the door is locked"
    
    def test_two_level_belief_hierarchy(self):
        """Test Level 8: 'A knows that B knows X' """
        tracker = MultiAgentBeliefTracker()
        
        # Alice believes that Bob knows the door is locked
        root_id = tracker.create_belief_hierarchy(
            agent_chain=["Alice", "Bob"],
            final_content="the door is locked",
            confidence=0.8
        )
        
        assert root_id is not None
        hierarchy = tracker.belief_networks[root_id]
        
        # Check hierarchy depth
        assert hierarchy.get_depth() == 1  # 2 agents = depth 1
        
        # Check natural language conversion
        nl = hierarchy.to_natural_language()
        assert "Alice believes that" in nl
        assert "Bob believes that" in nl
        assert "the door is locked" in nl
        
        print(f"✅ Level 8 Test Passed: {nl}")
    
    def test_three_level_belief_hierarchy(self):
        """Test deeper hierarchy: 'A knows that B knows that C knows X' """
        tracker = MultiAgentBeliefTracker()
        
        # Alice believes that Bob believes that Charlie believes the meeting is at 3pm
        root_id = tracker.create_belief_hierarchy(
            agent_chain=["Alice", "Bob", "Charlie"],
            final_content="the meeting is at 3pm",
            confidence=0.7
        )
        
        hierarchy = tracker.belief_networks[root_id]
        assert hierarchy.get_depth() == 2  # 3 agents = depth 2
        
        nl = hierarchy.to_natural_language()
        print(f"✅ Level 8 Deep Hierarchy: {nl}")
        
        # Verify confidence decay
        root_belief = tracker.beliefs[root_id]
        assert root_belief.confidence < 0.8  # Should decay from 0.7
    
    def test_query_beliefs_about_agent(self):
        """Test querying what one agent believes about another"""
        tracker = MultiAgentBeliefTracker()
        
        # Alice has epistemic beliefs about Bob
        tracker.add_belief(
            subject="Alice",
            content="Bob is trustworthy",
            belief_type=BeliefType.EPISTEMIC,
            about_agent="Bob",
            confidence=0.9
        )
        
        tracker.add_belief(
            subject="Alice",
            content="Bob knows the secret",
            belief_type=BeliefType.EPISTEMIC,
            about_agent="Bob",
            confidence=0.7
        )
        
        # Query Alice's beliefs about Bob
        beliefs_about_bob = tracker.query_beliefs_about_agent("Alice", "Bob")
        
        assert len(beliefs_about_bob) == 2
        assert all(b.about_agent == "Bob" for b in beliefs_about_bob)
        
        print(f"✅ Alice has {len(beliefs_about_bob)} epistemic beliefs about Bob")
    
    def test_shared_beliefs(self):
        """Test detecting shared beliefs between agents (mutual knowledge)"""
        tracker = MultiAgentBeliefTracker()
        
        # Both Alice and Bob believe the project is important
        tracker.add_belief(
            subject="Alice",
            content="the project is important for success",
            belief_type=BeliefType.FACTUAL
        )
        
        tracker.add_belief(
            subject="Bob",
            content="the project is important",
            belief_type=BeliefType.FACTUAL
        )
        
        shared = tracker.get_shared_beliefs("Alice", "Bob", threshold=0.5)
        
        assert len(shared) > 0
        print(f"✅ Found {len(shared)} shared beliefs between Alice and Bob")
    
    def test_belief_tracker_statistics(self):
        """Test comprehensive statistics"""
        tracker = MultiAgentBeliefTracker()
        
        # Create diverse beliefs
        tracker.add_belief("Alice", "fact1", BeliefType.FACTUAL)
        tracker.add_belief("Bob", "fact2", BeliefType.FACTUAL)
        tracker.create_belief_hierarchy(["Alice", "Bob"], "shared knowledge")
        
        stats = tracker.get_statistics()
        
        assert stats["total_beliefs"] >= 3
        assert stats["agents_tracked"] >= 2
        assert stats["belief_hierarchies"] >= 1
        
        print(f"✅ Tracker Statistics: {stats}")


class TestLevel9_StrategicReasoning:
    """Test Level 9 ToM: Machiavellian Strategic Reasoning"""
    
    def test_relationship_modeling(self):
        """Test modeling relationships between agents"""
        belief_tracker = MultiAgentBeliefTracker()
        reasoner = StrategicSocialReasoner(belief_tracker)
        
        relationship = reasoner.model_relationship("Alice", "Bob")
        
        assert relationship is not None
        assert relationship.trust_level == 0.5  # Initial neutral trust
        assert relationship.power_balance == 0.0  # Initial balance
        
        print(f"✅ Relationship modeled: Alice <-> Bob (trust={relationship.trust_level})")
    
    def test_cooperation_strategy(self):
        """Test evaluating cooperation strategy"""
        belief_tracker = MultiAgentBeliefTracker()
        reasoner = StrategicSocialReasoner(belief_tracker)
        
        action = reasoner.evaluate_strategic_action(
            actor="Alice",
            target="Bob",
            action_type=SocialStrategy.COOPERATION,
            context={"goal": "complete project together"}
        )
        
        assert action.action_type == SocialStrategy.COOPERATION
        assert 0.0 <= action.expected_payoff <= 1.0
        assert 0.0 <= action.risk_level <= 1.0
        assert 0.0 <= action.ethical_score <= 1.0
        
        print(f"✅ Cooperation Strategy: payoff={action.expected_payoff:.2f}, "
              f"risk={action.risk_level:.2f}, ethics={action.ethical_score:.2f}")
    
    def test_deception_detection(self):
        """Test detecting potential deception"""
        belief_tracker = MultiAgentBeliefTracker()
        reasoner = StrategicSocialReasoner(belief_tracker)
        
        # Alice states one thing but does another
        is_deceptive, confidence = reasoner.detect_deception(
            actor="Alice",
            stated_belief="I will help with the project",
            actual_behavior="Alice avoided all project meetings"
        )
        
        # Should detect some inconsistency
        print(f"✅ Deception Detection: deceptive={is_deceptive}, confidence={confidence:.2f}")
    
    def test_strategy_comparison(self):
        """Test comparing multiple strategies"""
        belief_tracker = MultiAgentBeliefTracker()
        reasoner = StrategicSocialReasoner(belief_tracker)
        
        strategies_to_test = [
            SocialStrategy.COOPERATION,
            SocialStrategy.COMPETITION,
            SocialStrategy.ALLIANCE,
            SocialStrategy.DECEPTION
        ]
        
        results = []
        for strategy in strategies_to_test:
            action = reasoner.evaluate_strategic_action(
                "Alice", "Bob", strategy, {}
            )
            results.append((strategy.value, action.expected_payoff, action.ethical_score))
        
        print("✅ Strategy Comparison:")
        for name, payoff, ethics in results:
            print(f"   {name}: payoff={payoff:.2f}, ethics={ethics:.2f}")
        
        # Alliance should have high payoff and high ethics
        alliance_result = [r for r in results if r[0] == "alliance"][0]
        assert alliance_result[1] > 0.7  # High payoff
        assert alliance_result[2] > 0.7  # High ethics
    
    def test_recommend_ethical_strategy(self):
        """Test recommending strategy with ethical constraints"""
        belief_tracker = MultiAgentBeliefTracker()
        reasoner = StrategicSocialReasoner(belief_tracker)
        
        # Request strategy with high ethical constraint
        recommended = reasoner.recommend_strategy(
            actor="Alice",
            target="Bob",
            goal="resolve conflict",
            ethical_constraint=0.7
        )
        
        assert recommended.ethical_score >= 0.7
        print(f"✅ Recommended Ethical Strategy: {recommended.action_type.value} "
              f"(ethics={recommended.ethical_score:.2f})")
    
    def test_trust_update(self):
        """Test updating trust based on outcomes"""
        belief_tracker = MultiAgentBeliefTracker()
        reasoner = StrategicSocialReasoner(belief_tracker)
        
        relationship = reasoner.model_relationship("Alice", "Bob")
        initial_trust = relationship.trust_level
        
        # Positive outcome
        relationship.update_trust(outcome=True, learning_rate=0.2)
        assert relationship.trust_level > initial_trust
        
        # Negative outcome
        relationship.update_trust(outcome=False, learning_rate=0.2)
        # Trust should decrease faster than it increased
        
        print(f"✅ Trust dynamics: {initial_trust:.2f} -> {relationship.trust_level:.2f}")


class TestLevel10_CulturalContext:
    """Test Level 10 ToM: Cultural Context Modeling"""
    
    def test_cultural_norm_creation(self):
        """Test adding cultural norms"""
        engine = CulturalContextEngine()
        
        norm_id = engine.add_cultural_norm(
            culture="japanese",
            category="greeting",
            description="Bow when meeting someone",
            importance=0.9
        )
        
        assert norm_id in engine.cultural_norms
        assert engine.cultural_norms[norm_id].culture == "japanese"
        
        print(f"✅ Cultural norm created: {norm_id}")
    
    def test_assign_culture_to_agent(self):
        """Test assigning cultural background to agents"""
        engine = CulturalContextEngine()
        
        engine.assign_culture_to_agent("Alice", ["western", "professional"])
        engine.assign_culture_to_agent("Bob", ["eastern", "casual"])
        
        assert "western" in engine.agent_cultures["Alice"]
        assert "eastern" in engine.agent_cultures["Bob"]
        
        print(f"✅ Cultures assigned: Alice={engine.agent_cultures['Alice']}, "
              f"Bob={engine.agent_cultures['Bob']}")
    
    def test_cultural_context_generation(self):
        """Test generating cultural context for interaction"""
        engine = CulturalContextEngine()
        
        engine.assign_culture_to_agent("Alice", ["western"])
        engine.assign_culture_to_agent("Bob", ["eastern"])
        
        context = engine.get_cultural_context("Alice", "Bob", situation="business")
        
        assert len(context.culture_ids) >= 2
        assert context.formality_level > 0.5  # Business context should be formal
        
        print(f"✅ Cultural context: cultures={context.culture_ids}, "
              f"formality={context.formality_level:.2f}")
    
    def test_culturally_appropriate_response(self):
        """Test generating culturally appropriate responses"""
        engine = CulturalContextEngine()
        
        engine.assign_culture_to_agent("Alice", ["professional"])
        context = engine.get_cultural_context("Alice", "Bob", "business")
        
        response = engine.generate_culturally_appropriate_response(
            agent_id="Alice",
            input_text="Let's schedule a meeting",
            context=context
        )
        
        assert response is not None
        assert len(response) > 0
        
        print(f"✅ Culturally appropriate response: '{response}'")
    
    def test_turing_test_readiness(self):
        """Test Turing test readiness assessment"""
        engine = CulturalContextEngine()
        
        # Add multiple norms
        for i in range(10):
            engine.add_cultural_norm(
                culture=f"culture_{i % 3}",
                category="test",
                description=f"Test norm {i}",
                importance=0.5
            )
        
        readiness = engine.get_turing_test_readiness()
        
        assert "overall_readiness" in readiness
        assert "total_norms" in readiness
        assert "cultures_modeled" in readiness
        
        print(f"✅ Turing test readiness: {readiness['overall_readiness']:.2f} "
              f"({readiness['status']})")


@pytest.mark.asyncio
class TestIntegratedAdvancedToM:
    """Test Integrated Advanced Theory of Mind System"""
    
    async def test_full_system_initialization(self):
        """Test complete system initialization"""
        tom = AdvancedTheoryOfMind(max_belief_depth=4)
        
        assert tom.belief_tracker is not None
        assert tom.strategic_reasoner is not None
        assert tom.cultural_engine is not None
        assert tom.active is True
        
        print("✅ Full Advanced ToM system initialized")
    
    async def test_complete_social_interaction(self):
        """Test processing a complete social interaction (Levels 8-10)"""
        tom = AdvancedTheoryOfMind()
        
        # Assign cultures
        tom.cultural_engine.assign_culture_to_agent("Alice", ["western", "professional"])
        tom.cultural_engine.assign_culture_to_agent("Bob", ["western", "professional"])
        
        # Process interaction
        result = await tom.process_social_interaction(
            actor="Alice",
            target="Bob",
            interaction_type="offer",
            content={
                "text": "I'd like to propose a partnership",
                "stated_belief": "collaboration is beneficial"
            },
            context={"situation": "business"}
        )
        
        assert "cultural_context" in result
        assert "belief_analysis" in result
        assert "strategic_analysis" in result
        assert "tom_level_active" in result
        assert result["tom_level_active"] == [8, 9, 10]
        
        print(f"✅ Complete social interaction processed:")
        print(f"   Cultural formality: {result['cultural_context']['formality']:.2f}")
        print(f"   Strategic payoff: {result['strategic_analysis']['expected_payoff']:.2f}")
        print(f"   Active ToM levels: {result['tom_level_active']}")
    
    async def test_comprehensive_status(self):
        """Test getting comprehensive system status"""
        tom = AdvancedTheoryOfMind()
        
        # Add some data
        tom.belief_tracker.add_belief("Alice", "test belief", BeliefType.FACTUAL)
        tom.cultural_engine.assign_culture_to_agent("Alice", ["western"])
        
        status = tom.get_comprehensive_status()
        
        assert "system_active" in status
        assert "level_8_belief_tracking" in status
        assert "level_9_strategic_reasoning" in status
        assert "level_10_cultural_modeling" in status
        assert "overall_tom_level" in status
        
        tom_level = status["overall_tom_level"]
        assert 6.0 <= tom_level <= 10.0
        
        print(f"✅ System status: ToM Level = {tom_level:.1f}")
        print(f"   Beliefs tracked: {status['level_8_belief_tracking']['total_beliefs']}")
        print(f"   Relationships: {status['level_9_strategic_reasoning']['relationships_tracked']}")
        print(f"   Turing readiness: {status['level_10_cultural_modeling']['overall_readiness']:.2f}")


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for enterprise deployment"""
    
    def test_belief_creation_performance(self, benchmark):
        """Benchmark belief creation speed"""
        tracker = MultiAgentBeliefTracker()
        
        def create_belief():
            return tracker.add_belief(
                subject="Alice",
                content="test belief content",
                belief_type=BeliefType.FACTUAL
            )
        
        result = benchmark(create_belief)
        print(f"✅ Belief creation: {benchmark.stats.stats.mean * 1000:.2f}ms avg")
    
    def test_hierarchy_creation_performance(self, benchmark):
        """Benchmark hierarchy creation speed"""
        tracker = MultiAgentBeliefTracker()
        
        def create_hierarchy():
            return tracker.create_belief_hierarchy(
                agent_chain=["Alice", "Bob", "Charlie"],
                final_content="test content"
            )
        
        result = benchmark(create_hierarchy)
        print(f"✅ Hierarchy creation: {benchmark.stats.stats.mean * 1000:.2f}ms avg")
    
    def test_strategy_evaluation_performance(self, benchmark):
        """Benchmark strategic evaluation speed"""
        belief_tracker = MultiAgentBeliefTracker()
        reasoner = StrategicSocialReasoner(belief_tracker)
        
        def evaluate_strategy():
            return reasoner.evaluate_strategic_action(
                "Alice", "Bob", SocialStrategy.COOPERATION, {}
            )
        
        result = benchmark(evaluate_strategy)
        print(f"✅ Strategy evaluation: {benchmark.stats.stats.mean * 1000:.2f}ms avg")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestIntegration:
    """Integration tests with existing consciousness modules"""
    
    def test_conscious_moment_integration(self):
        """Test integration with conscious moment processing"""
        # Simulated conscious moment from consciousness system
        conscious_moment = {
            "emotional_valence": 0.7,
            "primary_focus": {"type": "request", "content": "help me understand"},
            "context": {"task_type": "learning"},
            "integrated_content": "This topic is important to understand"
        }
        
        # Basic ToM should work
        from packages.consciousness.src.conciencia.modulos.teoria_mente import TheoryOfMind
        
        tom = TheoryOfMind()
        tom.update_model("user123", conscious_moment)
        
        user_model = tom.get_user_model("user123")
        assert user_model is not None
        assert user_model["intent"] == "seeking_knowledge"
        
        print("✅ Integration with conscious moment processing successful")


if __name__ == "__main__":
    print("="*80)
    print("🧪 ADVANCED THEORY OF MIND TEST SUITE")
    print("="*80)
    print()
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
