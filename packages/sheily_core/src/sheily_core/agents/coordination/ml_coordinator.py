#!/usr/bin/env python3
"""
ML-Enhanced Agent Coordinator with Reinforcement Learning
=========================================================

Features 2025:
- Multi-Armed Bandit for agent selection
- Thompson Sampling algorithm
- Contextual bandits
- Dynamic load balancing
- Performance prediction
- Auto-scaling decisions
- A/B testing framework
"""

import asyncio
import hashlib
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base.enhanced_base import (
    AgentCapability,
    AgentStatus,
    AgentTask,
    EnhancedBaseMCPAgent,
    TaskPriority,
)

logger = logging.getLogger(__name__)


class BanditArm:
    """Multi-armed bandit arm for agent selection"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.alpha = 1.0  # Success count + 1 (Beta dist param)
        self.beta = 1.0  # Failure count + 1 (Beta dist param)
        self.total_pulls = 0
        self.total_rewards = 0.0
        self.recent_rewards = deque(maxlen=100)

    def sample(self) -> float:
        """Thompson Sampling: sample from Beta distribution"""
        return np.random.beta(self.alpha, self.beta)

    def update(self, reward: float):
        """Update arm with reward (0-1)"""
        self.total_pulls += 1
        self.total_rewards += reward
        self.recent_rewards.append(reward)

        # Update Beta distribution parameters
        if reward > 0.5:  # Success
            self.alpha += reward
        else:  # Failure
            self.beta += 1.0 - reward

    def get_expected_value(self) -> float:
        """Get expected value (mean of Beta distribution)"""
        return self.alpha / (self.alpha + self.beta)

    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval for expected value"""
        # Using percentiles of Beta distribution
        lower = np.percentile(
            [np.random.beta(self.alpha, self.beta) for _ in range(1000)],
            (1 - confidence) * 100 / 2,
        )
        upper = np.percentile(
            [np.random.beta(self.alpha, self.beta) for _ in range(1000)],
            confidence * 100 + (1 - confidence) * 100 / 2,
        )
        return lower, upper


class ContextualBandit:
    """Contextual bandit for task-agent matching"""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        # Context features: task_type, capabilities, agent_performance
        self.weights: Dict[str, np.ndarray] = {}
        self.context_history: List[Tuple[str, np.ndarray, float]] = []

    def get_context_vector(
        self, task: AgentTask, agent: EnhancedBaseMCPAgent
    ) -> np.ndarray:
        """Extract context features"""
        features = []

        # Task features
        features.append(float(task.priority.value) / 5.0)  # Normalized priority
        features.append(float(task.timeout_seconds) / 600.0)  # Normalized timeout
        features.append(float(len(task.required_capabilities)) / 10.0)

        # Agent features
        features.append(agent.metrics.success_rate)
        features.append(agent.get_capacity_score())
        features.append(agent.get_performance_score())
        features.append(float(len(agent.capabilities)) / 20.0)

        # Match score
        match_score = sum(
            1 for cap in task.required_capabilities if cap in agent.capabilities
        )
        features.append(match_score / max(1, len(task.required_capabilities)))

        return np.array(features)

    def predict_reward(self, agent_id: str, context: np.ndarray) -> float:
        """Predict expected reward for agent-context pair"""
        if agent_id not in self.weights:
            # Initialize weights
            self.weights[agent_id] = np.random.randn(len(context)) * 0.01

        # Linear model: reward = w^T * context
        prediction = np.dot(self.weights[agent_id], context)
        return float(np.clip(prediction, 0, 1))

    def update_weights(self, agent_id: str, context: np.ndarray, actual_reward: float):
        """Update weights using gradient descent"""
        if agent_id not in self.weights:
            self.weights[agent_id] = np.random.randn(len(context)) * 0.01

        # Gradient descent update
        predicted = self.predict_reward(agent_id, context)
        error = actual_reward - predicted
        gradient = error * context
        self.weights[agent_id] += self.learning_rate * gradient

        # Store history
        self.context_history.append((agent_id, context, actual_reward))

        # Keep only recent history
        if len(self.context_history) > 10000:
            self.context_history = self.context_history[-5000:]


@dataclass
class CoordinationStats:
    """Statistics for ML coordinator"""

    total_tasks_assigned: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    avg_assignment_time: float = 0.0
    avg_completion_time: float = 0.0
    agent_utilization: Dict[str, float] = field(default_factory=dict)
    task_type_distribution: Dict[str, int] = field(default_factory=dict)
    bandit_performance: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


class MLAgentCoordinator(EnhancedBaseMCPAgent):
    """
    ML-Enhanced coordinator with reinforcement learning
    """

    def __init__(
        self,
        coordinator_id: str = "ml_coordinator_master",
        exploration_rate: float = 0.1,
        enable_contextual: bool = True,
    ):
        super().__init__(
            agent_id=coordinator_id,
            agent_name="ML Agent Coordinator",
            capabilities=[
                AgentCapability.STRATEGIC,
                AgentCapability.ANALYSIS,
                AgentCapability.SYSTEM_OPTIMIZATION,
            ],
            max_concurrent_tasks=1000,
            enable_learning=True,
        )

        self.exploration_rate = exploration_rate
        self.enable_contextual = enable_contextual

        # Agent registry
        self.agents: Dict[str, EnhancedBaseMCPAgent] = {}

        # Multi-armed bandits (one per task type)
        self.bandits: Dict[str, Dict[str, BanditArm]] = defaultdict(dict)

        # Contextual bandit
        self.contextual_bandit = ContextualBandit() if enable_contextual else None

        # Stats
        self.stats = CoordinationStats()

        # Task history for A/B testing
        self.assignment_history: deque = deque(maxlen=10000)

        logger.info("ML Agent Coordinator initialized with RL capabilities")

    def register_agent(self, agent: EnhancedBaseMCPAgent):
        """Register specialized agent"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_name} ({agent.agent_id})")

    def unregister_agent(self, agent_id: str):
        """Unregister agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")

    async def _execute_task_impl(self, task: AgentTask) -> Dict[str, Any]:
        """
        Coordinate task execution using ML-based agent selection
        """
        # Find capable agents
        capable_agents = self._find_capable_agents(task)

        if not capable_agents:
            raise RuntimeError(f"No capable agents found for task {task.task_id}")

        # Select agent using ML
        selected_agent = await self._select_agent_ml(task, capable_agents)

        # Execute task
        start_time = datetime.now()
        result = await selected_agent.execute_task(task)
        end_time = datetime.now()

        # Calculate reward
        reward = self._calculate_reward(result, end_time - start_time)

        # Update ML models
        await self._update_ml_models(task, selected_agent.agent_id, reward)

        # Update stats
        self._update_stats(task, selected_agent.agent_id, result, reward)

        return result

    def _find_capable_agents(self, task: AgentTask) -> List[EnhancedBaseMCPAgent]:
        """Find agents capable of handling the task"""
        capable = []

        for agent in self.agents.values():
            # Check capabilities
            if not agent.has_all_capabilities(task.required_capabilities):
                continue

            # Check availability
            if (
                agent.status == AgentStatus.ERROR
                or agent.status == AgentStatus.MAINTENANCE
            ):
                continue

            # Check capacity
            if agent.get_capacity_score() < 0.1:  # Less than 10% capacity
                continue

            capable.append(agent)

        return capable

    async def _select_agent_ml(
        self, task: AgentTask, candidates: List[EnhancedBaseMCPAgent]
    ) -> EnhancedBaseMCPAgent:
        """
        Select agent using ML (Thompson Sampling + Contextual Bandit)
        """
        task_type = task.task_type

        # Exploration vs Exploitation
        if np.random.random() < self.exploration_rate:
            # Exploration: random selection weighted by capacity
            weights = [agent.get_capacity_score() for agent in candidates]
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                selected = np.random.choice(candidates, p=weights)
            else:
                selected = np.random.choice(candidates)

            logger.debug(f"Exploration: selected {selected.agent_id}")
            return selected

        # Exploitation: use ML models
        if self.enable_contextual and self.contextual_bandit:
            # Contextual bandit approach
            best_agent = None
            best_score = -float("inf")

            for agent in candidates:
                context = self.contextual_bandit.get_context_vector(task, agent)
                score = self.contextual_bandit.predict_reward(agent.agent_id, context)

                # Add capacity bonus
                score += agent.get_capacity_score() * 0.2

                if score > best_score:
                    best_score = score
                    best_agent = agent

            logger.debug(
                f"Contextual: selected {best_agent.agent_id} (score: {best_score:.3f})"
            )
            return best_agent

        else:
            # Thompson Sampling approach
            # Initialize bandits for new agents
            for agent in candidates:
                if agent.agent_id not in self.bandits[task_type]:
                    self.bandits[task_type][agent.agent_id] = BanditArm(agent.agent_id)

            # Sample from each arm
            samples = {}
            for agent in candidates:
                arm = self.bandits[task_type][agent.agent_id]
                sample = arm.sample()

                # Adjust sample by current capacity
                adjusted_sample = sample * (0.7 + 0.3 * agent.get_capacity_score())
                samples[agent.agent_id] = adjusted_sample

            # Select best sample
            best_agent_id = max(samples.items(), key=lambda x: x[1])[0]
            best_agent = next(a for a in candidates if a.agent_id == best_agent_id)

            logger.debug(
                f"Thompson: selected {best_agent_id} (sample: {samples[best_agent_id]:.3f})"
            )
            return best_agent

    def _calculate_reward(
        self, result: Dict[str, Any], execution_time: timedelta
    ) -> float:
        """Calculate reward signal (0-1)"""
        reward = 0.0

        # Success component (0.5)
        if result.get("success", False):
            reward += 0.5

        # Speed component (0.3)
        exec_seconds = execution_time.total_seconds()
        if exec_seconds < 60:
            reward += 0.3
        elif exec_seconds < 300:
            reward += 0.2
        elif exec_seconds < 600:
            reward += 0.1

        # Quality component (0.2) if available
        if "quality_score" in result:
            reward += result["quality_score"] * 0.2
        else:
            reward += 0.1  # Default quality

        return min(1.0, reward)

    async def _update_ml_models(self, task: AgentTask, agent_id: str, reward: float):
        """Update ML models with reward signal"""
        task_type = task.task_type

        # Update Thompson Sampling bandit
        if agent_id in self.bandits[task_type]:
            self.bandits[task_type][agent_id].update(reward)

        # Update Contextual Bandit
        if self.contextual_bandit and agent_id in self.agents:
            agent = self.agents[agent_id]
            context = self.contextual_bandit.get_context_vector(task, agent)
            self.contextual_bandit.update_weights(agent_id, context, reward)

    def _update_stats(
        self, task: AgentTask, agent_id: str, result: Dict[str, Any], reward: float
    ):
        """Update coordination statistics"""
        self.stats.total_tasks_assigned += 1

        if result.get("success", False):
            self.stats.total_tasks_completed += 1
        else:
            self.stats.total_tasks_failed += 1

        # Update task type distribution
        task_type = task.task_type
        self.stats.task_type_distribution[task_type] = (
            self.stats.task_type_distribution.get(task_type, 0) + 1
        )

        # Update agent utilization
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            self.stats.agent_utilization[agent_id] = 1.0 - agent.get_capacity_score()

        # Update bandit performance
        if task_type in self.bandits and agent_id in self.bandits[task_type]:
            arm = self.bandits[task_type][agent_id]
            self.stats.bandit_performance[agent_id] = arm.get_expected_value()

        self.stats.last_updated = datetime.now()

    async def assign_task_intelligently(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        required_capabilities: List[AgentCapability],
        priority: TaskPriority = TaskPriority.MEDIUM,
        timeout_seconds: int = 300,
    ) -> Dict[str, Any]:
        """
        High-level API for intelligent task assignment
        """
        task = AgentTask(
            task_id=hashlib.md5(
                f"{task_type}_{datetime.now().timestamp()}".encode()
            ).hexdigest(),
            task_type=task_type,
            parameters=parameters,
            priority=priority,
            required_capabilities=required_capabilities,
            timeout_seconds=timeout_seconds,
        )

        result = await self.execute_task(task)
        return result

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        total_agents = len(self.agents)
        active_agents = sum(
            1 for a in self.agents.values() if a.status != AgentStatus.ERROR
        )

        # Calculate average performance
        if self.agents:
            avg_performance = sum(
                a.get_performance_score() for a in self.agents.values()
            ) / len(self.agents)
        else:
            avg_performance = 0.0

        # Calculate system health (0-1)
        if total_agents > 0:
            health_score = (active_agents / total_agents) * avg_performance
        else:
            health_score = 0.0

        return {
            "coordinator_id": self.agent_id,
            "total_agents": total_agents,
            "active_agents": active_agents,
            "system_health": health_score,
            "stats": {
                "total_tasks_assigned": self.stats.total_tasks_assigned,
                "total_tasks_completed": self.stats.total_tasks_completed,
                "total_tasks_failed": self.stats.total_tasks_failed,
                "success_rate": self.stats.total_tasks_completed
                / max(1, self.stats.total_tasks_assigned),
                "agent_utilization": self.stats.agent_utilization,
                "task_distribution": self.stats.task_type_distribution,
            },
            "ml_models": {
                "exploration_rate": self.exploration_rate,
                "contextual_enabled": self.enable_contextual,
                "total_task_types": len(self.bandits),
                "bandit_arms": {
                    task_type: len(arms) for task_type, arms in self.bandits.items()
                },
            },
            "timestamp": datetime.now().isoformat(),
        }

    async def get_agent_recommendations(self, task: AgentTask) -> List[Dict[str, Any]]:
        """Get ranked agent recommendations for a task"""
        capable_agents = self._find_capable_agents(task)

        recommendations = []
        for agent in capable_agents:
            # Calculate scores
            performance_score = agent.get_performance_score()
            capacity_score = agent.get_capacity_score()

            # Contextual prediction if available
            if self.contextual_bandit:
                context = self.contextual_bandit.get_context_vector(task, agent)
                ml_score = self.contextual_bandit.predict_reward(
                    agent.agent_id, context
                )
            else:
                task_type = task.task_type
                if (
                    task_type in self.bandits
                    and agent.agent_id in self.bandits[task_type]
                ):
                    ml_score = self.bandits[task_type][
                        agent.agent_id
                    ].get_expected_value()
                else:
                    ml_score = 0.5

            # Combined score
            combined_score = (
                performance_score * 0.4 + capacity_score * 0.3 + ml_score * 0.3
            )

            recommendations.append(
                {
                    "agent_id": agent.agent_id,
                    "agent_name": agent.agent_name,
                    "performance_score": performance_score,
                    "capacity_score": capacity_score,
                    "ml_score": ml_score,
                    "combined_score": combined_score,
                    "current_tasks": len(agent.current_tasks),
                    "capabilities": [cap.value for cap in agent.capabilities],
                }
            )

        # Sort by combined score
        recommendations.sort(key=lambda x: x["combined_score"], reverse=True)

        return recommendations

    def export_ml_state(self) -> Dict[str, Any]:
        """Export ML models state for persistence"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "exploration_rate": self.exploration_rate,
            "bandits": {},
            "contextual_weights": {},
        }

        # Export bandit arms
        for task_type, arms in self.bandits.items():
            state["bandits"][task_type] = {
                agent_id: {
                    "alpha": arm.alpha,
                    "beta": arm.beta,
                    "total_pulls": arm.total_pulls,
                    "total_rewards": arm.total_rewards,
                    "expected_value": arm.get_expected_value(),
                }
                for agent_id, arm in arms.items()
            }

        # Export contextual bandit weights
        if self.contextual_bandit:
            state["contextual_weights"] = {
                agent_id: weights.tolist()
                for agent_id, weights in self.contextual_bandit.weights.items()
            }

        return state

    def import_ml_state(self, state: Dict[str, Any]):
        """Import ML models state"""
        # Import bandit arms
        if "bandits" in state:
            for task_type, arms_data in state["bandits"].items():
                for agent_id, arm_data in arms_data.items():
                    arm = BanditArm(agent_id)
                    arm.alpha = arm_data["alpha"]
                    arm.beta = arm_data["beta"]
                    arm.total_pulls = arm_data["total_pulls"]
                    arm.total_rewards = arm_data["total_rewards"]
                    self.bandits[task_type][agent_id] = arm

        # Import contextual weights
        if "contextual_weights" in state and self.contextual_bandit:
            for agent_id, weights_list in state["contextual_weights"].items():
                self.contextual_bandit.weights[agent_id] = np.array(weights_list)

        logger.info(f"Imported ML state from {state['timestamp']}")


# Global coordinator instance
_global_coordinator: Optional[MLAgentCoordinator] = None


async def get_ml_coordinator() -> MLAgentCoordinator:
    """Get or create global ML coordinator"""
    global _global_coordinator

    if _global_coordinator is None:
        _global_coordinator = MLAgentCoordinator()

    return _global_coordinator


__all__ = [
    "MLAgentCoordinator",
    "BanditArm",
    "ContextualBandit",
    "CoordinationStats",
    "get_ml_coordinator",
]
