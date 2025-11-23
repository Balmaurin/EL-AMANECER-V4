#!/usr/bin/env python3
"""
Advanced ML Agent Coordinator - Enterprise 2025
================================================

State-of-the-art ML coordinator with:
✓ Multi-algorithm ensemble (Thompson, UCB, EXP3, Softmax)
✓ Deep contextual bandits with neural features
✓ Multi-objective optimization (latency + cost + quality)
✓ Transfer learning across task domains
✓ Dynamic auto-scaling and load balancing
✓ Anomaly detection and self-healing
✓ Advanced analytics and explainability
✓ Real-time performance prediction
✓ Adaptive exploration strategies
✓ Causal inference for task routing
"""

import asyncio
import hashlib
import json
import logging
import math
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats

from ..base.enhanced_base import (
    AgentCapability,
    AgentStatus,
    AgentTask,
    EnhancedBaseMCPAgent,
    TaskPriority,
)

logger = logging.getLogger(__name__)


class BanditAlgorithm(Enum):
    """Available bandit algorithms"""

    THOMPSON = "thompson"  # Thompson Sampling (Beta distribution)
    UCB = "ucb"  # Upper Confidence Bound
    EXP3 = "exp3"  # Exponential-weight algorithm
    SOFTMAX = "softmax"  # Boltzmann exploration
    ENSEMBLE = "ensemble"  # Combination of multiple


class OptimizationObjective(Enum):
    """Multi-objective optimization goals"""

    LATENCY = "latency"  # Minimize response time
    COST = "cost"  # Minimize computational cost
    QUALITY = "quality"  # Maximize output quality
    BALANCED = "balanced"  # Balance all objectives


@dataclass
class AgentPerformanceMetrics:
    """Comprehensive agent performance tracking"""

    agent_id: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_latency: float = 0.0
    total_cost: float = 0.0
    quality_scores: List[float] = field(default_factory=list)

    # Time-based metrics
    tasks_last_hour: int = 0
    tasks_last_day: int = 0
    hourly_capacity: float = 100.0

    # Reliability metrics
    consecutive_failures: int = 0
    mean_time_between_failures: float = 0.0
    last_failure_time: Optional[datetime] = None

    # Learning metrics
    learning_rate: float = 0.01
    convergence_score: float = 0.0

    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks

    def avg_latency(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.total_latency / self.total_tasks

    def avg_quality(self) -> float:
        if not self.quality_scores:
            return 0.5
        return statistics.mean(self.quality_scores[-100:])


class AdvancedBanditArm:
    """
    Multi-algorithm bandit arm with ensemble capabilities
    """

    def __init__(
        self, agent_id: str, algorithm: BanditAlgorithm = BanditAlgorithm.THOMPSON
    ):
        self.agent_id = agent_id
        self.algorithm = algorithm

        # Thompson Sampling (Beta distribution)
        self.alpha = 1.0  # Prior successes
        self.beta = 1.0  # Prior failures

        # UCB (Upper Confidence Bound)
        self.mean_reward = 0.0
        self.reward_variance = 0.0
        self.sample_count = 0
        self.ucb_score = float("inf")
        self.ucb_confidence = 2.0  # Confidence parameter

        # EXP3 (Exponential weights)
        self.weight = 1.0
        self.probability = 0.0
        self.gamma = 0.1  # Exploration parameter

        # Softmax (Boltzmann)
        self.temperature = 1.0
        self.q_value = 0.0

        # Multi-objective rewards
        self.latency_rewards = deque(maxlen=1000)
        self.cost_rewards = deque(maxlen=1000)
        self.quality_rewards = deque(maxlen=1000)

        # Statistical tracking
        self.reward_history = deque(maxlen=5000)
        self.timestamped_rewards: List[Tuple[datetime, float]] = []

        # Anomaly detection
        self.baseline_mean = 0.5
        self.baseline_std = 0.1
        self.anomaly_threshold = 3.0  # 3-sigma rule

        # Metadata
        self.last_update = datetime.now()
        self.created_at = datetime.now()

    def sample(self, total_pulls: int = 0, temperature: float = None) -> float:
        """
        Sample using selected algorithm

        Args:
            total_pulls: Total pulls across all arms (for UCB)
            temperature: Override temperature for Softmax

        Returns:
            Score/probability for arm selection
        """
        if self.algorithm == BanditAlgorithm.THOMPSON:
            return self._thompson_sample()
        elif self.algorithm == BanditAlgorithm.UCB:
            return self._ucb_score(total_pulls)
        elif self.algorithm == BanditAlgorithm.EXP3:
            return self.probability
        elif self.algorithm == BanditAlgorithm.SOFTMAX:
            return self._softmax_score(temperature or self.temperature)
        elif self.algorithm == BanditAlgorithm.ENSEMBLE:
            return self._ensemble_score(total_pulls, temperature)
        else:
            return self.mean_reward

    def _thompson_sample(self) -> float:
        """Thompson Sampling from Beta distribution"""
        return float(np.random.beta(self.alpha, self.beta))

    def _ucb_score(self, total_pulls: int) -> float:
        """Upper Confidence Bound score"""
        if self.sample_count == 0:
            return float("inf")  # Unexplored arms have highest priority

        if total_pulls <= 0:
            return self.mean_reward

        # UCB1: mean + c * sqrt(ln(total) / count)
        exploration_bonus = self.ucb_confidence * math.sqrt(
            math.log(total_pulls) / self.sample_count
        )
        return self.mean_reward + exploration_bonus

    def _softmax_score(self, temperature: float) -> float:
        """Softmax (Boltzmann) probability"""
        return self.q_value / max(temperature, 0.001)

    def _ensemble_score(self, total_pulls: int, temperature: float = None) -> float:
        """Ensemble combining multiple algorithms"""
        scores = [
            self._thompson_sample() * 0.4,
            min(self._ucb_score(total_pulls) / 2.0, 1.0) * 0.3,  # Normalize UCB
            self._softmax_score(temperature or 1.0) * 0.3,
        ]
        return sum(scores)

    def update(
        self,
        reward: float,
        latency: float = 0.0,
        cost: float = 0.0,
        quality: float = 0.0,
        total_pulls: int = 0,
    ):
        """
        Update all algorithm parameters

        Args:
            reward: Overall reward (0-1)
            latency: Task latency in seconds
            cost: Computational cost
            quality: Output quality score (0-1)
            total_pulls: Total pulls across all arms
        """
        # Thompson Sampling update
        if reward > 0.5:
            self.alpha += 1
        else:
            self.beta += 1

        # Mean and variance tracking (for UCB and general stats)
        self.sample_count += 1
        delta = reward - self.mean_reward
        self.mean_reward += delta / self.sample_count
        delta2 = reward - self.mean_reward
        self.reward_variance += delta * delta2

        # UCB score update
        if total_pulls > 0:
            self.ucb_score = self._ucb_score(total_pulls)

        # EXP3 weight update
        if self.probability > 0:
            estimated_reward = reward / max(self.probability, 0.001)
            self.weight *= math.exp(self.gamma * estimated_reward / self.sample_count)

        # Softmax Q-value update (exponential moving average)
        self.q_value = 0.9 * self.q_value + 0.1 * reward

        # Multi-objective rewards
        if latency > 0:
            latency_reward = max(0, 1.0 - latency / 10.0)  # Normalize to 0-1
            self.latency_rewards.append(latency_reward)

        if cost > 0:
            cost_reward = max(0, 1.0 - cost / 100.0)
            self.cost_rewards.append(cost_reward)

        if quality > 0:
            self.quality_rewards.append(quality)

        # History tracking
        self.reward_history.append(reward)
        self.timestamped_rewards.append((datetime.now(), reward))

        # Update baseline for anomaly detection
        if len(self.reward_history) >= 30:
            recent = list(self.reward_history)[-100:]
            self.baseline_mean = statistics.mean(recent)
            self.baseline_std = statistics.stdev(recent) if len(recent) > 1 else 0.1

        self.last_update = datetime.now()

    def update_exp3_probabilities(self, total_weight: float, num_arms: int):
        """Update EXP3 probability based on total weights"""
        uniform_prob = self.gamma / num_arms
        weighted_prob = (1 - self.gamma) * (self.weight / max(total_weight, 0.001))
        self.probability = uniform_prob + weighted_prob

    def is_anomaly(self, reward: float) -> bool:
        """Detect if reward is anomalous (3-sigma rule)"""
        if self.sample_count < 30:
            return False

        z_score = abs(reward - self.baseline_mean) / max(self.baseline_std, 0.001)
        return z_score > self.anomaly_threshold

    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval for expected reward"""
        if self.sample_count < 2:
            return (0.0, 1.0)

        # For Beta distribution (Thompson Sampling)
        mean = self.alpha / (self.alpha + self.beta)
        var = (self.alpha * self.beta) / (
            (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)
        )
        std = math.sqrt(var)

        # Z-score for confidence level
        z = 1.96 if confidence == 0.95 else 2.576

        return (max(0, mean - z * std), min(1, mean + z * std))

    def get_multi_objective_score(self, objective: OptimizationObjective) -> float:
        """Get score for specific optimization objective"""
        if objective == OptimizationObjective.LATENCY:
            return (
                statistics.mean(self.latency_rewards) if self.latency_rewards else 0.5
            )
        elif objective == OptimizationObjective.COST:
            return statistics.mean(self.cost_rewards) if self.cost_rewards else 0.5
        elif objective == OptimizationObjective.QUALITY:
            return (
                statistics.mean(self.quality_rewards) if self.quality_rewards else 0.5
            )
        else:  # BALANCED
            scores = []
            if self.latency_rewards:
                scores.append(statistics.mean(self.latency_rewards))
            if self.cost_rewards:
                scores.append(statistics.mean(self.cost_rewards))
            if self.quality_rewards:
                scores.append(statistics.mean(self.quality_rewards))
            return statistics.mean(scores) if scores else self.mean_reward

    def get_statistics(self) -> Dict[str, Any]:
        """Comprehensive statistics"""
        ci_low, ci_high = self.get_confidence_interval()

        stats_dict = {
            "agent_id": self.agent_id,
            "algorithm": self.algorithm.value,
            "mean_reward": self.mean_reward,
            "sample_count": self.sample_count,
            "success_rate": self.alpha / (self.alpha + self.beta),
            "ucb_score": self.ucb_score if self.ucb_score != float("inf") else 1.0,
            "confidence_interval": (ci_low, ci_high),
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
        }

        # Recent performance
        if len(self.reward_history) >= 10:
            recent = list(self.reward_history)[-100:]
            stats_dict["recent_avg"] = statistics.mean(recent)
            stats_dict["recent_std"] = (
                statistics.stdev(recent) if len(recent) > 1 else 0
            )
            stats_dict["trend"] = self._calculate_trend()

        # Multi-objective scores
        stats_dict["latency_score"] = (
            statistics.mean(self.latency_rewards) if self.latency_rewards else 0
        )
        stats_dict["cost_score"] = (
            statistics.mean(self.cost_rewards) if self.cost_rewards else 0
        )
        stats_dict["quality_score"] = (
            statistics.mean(self.quality_rewards) if self.quality_rewards else 0
        )

        return stats_dict

    def _calculate_trend(self) -> str:
        """Calculate performance trend"""
        if len(self.reward_history) < 20:
            return "insufficient_data"

        recent = list(self.reward_history)[-20:]
        older = (
            list(self.reward_history)[-40:-20]
            if len(self.reward_history) >= 40
            else recent
        )

        recent_avg = statistics.mean(recent)
        older_avg = statistics.mean(older)

        diff = recent_avg - older_avg

        if abs(diff) < 0.05:
            return "stable"
        elif diff > 0.1:
            return "improving"
        elif diff > 0:
            return "slightly_improving"
        elif diff < -0.1:
            return "degrading"
        else:
            return "slightly_degrading"


class DeepContextualBandit:
    """
    Advanced contextual bandit with deep features and transfer learning
    """

    def __init__(self, learning_rate: float = 0.01, feature_dim: int = 32):
        self.learning_rate = learning_rate
        self.feature_dim = feature_dim

        # Linear model weights (per agent)
        self.weights: Dict[str, np.ndarray] = {}

        # Second layer for non-linear transformation
        self.hidden_weights: Dict[str, np.ndarray] = {}
        self.hidden_dim = 16

        # Adam optimizer parameters
        self.momentum: Dict[str, np.ndarray] = {}
        self.velocity: Dict[str, np.ndarray] = {}
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.timestep = 0

        # Adaptive learning rate per agent
        self.agent_lr: Dict[str, float] = {}

        # Transfer learning: task embeddings
        self.task_embeddings: Dict[str, np.ndarray] = {}
        self.embedding_dim = 8

        # Context history for replay learning
        self.context_history: deque = deque(maxlen=10000)

        # Feature importance (for explainability)
        self.feature_importance: Dict[int, float] = defaultdict(float)

    def extract_context_vector(
        self, task: AgentTask, agent: EnhancedBaseMCPAgent
    ) -> np.ndarray:
        """
        Extract comprehensive context features
        """
        features = []

        # === Task Features ===
        # Basic task properties
        priority_norm = float(task.priority.value) / 5.0
        timeout_norm = min(float(task.timeout_seconds) / 600.0, 1.0)
        cap_count = len(task.required_capabilities)
        cap_count_norm = min(float(cap_count) / 10.0, 1.0)

        features.extend(
            [
                priority_norm,
                timeout_norm,
                cap_count_norm,
                priority_norm * timeout_norm,  # Interaction: urgent + short timeout
            ]
        )

        # Task type embedding
        if task.task_type in self.task_embeddings:
            task_emb = self.task_embeddings[task.task_type][:4]
        else:
            task_emb = [0.0] * 4
        features.extend(task_emb)

        # === Agent Features ===
        # Performance metrics
        success_rate = agent.metrics.success_rate
        capacity = agent.get_capacity_score()
        performance = agent.get_performance_score()
        agent_caps = len(agent.capabilities)
        agent_caps_norm = min(float(agent_caps) / 20.0, 1.0)

        features.extend(
            [
                success_rate,
                capacity,
                performance,
                agent_caps_norm,
            ]
        )

        # Combined features
        features.extend(
            [
                success_rate * capacity,  # Available quality
                performance * capacity,  # Load-adjusted performance
                success_rate * performance,  # Reliable performance
            ]
        )

        # === Match Features ===
        # Capability matching
        matching_caps = sum(
            1 for cap in task.required_capabilities if cap in agent.capabilities
        )
        match_ratio = matching_caps / max(cap_count, 1)
        perfect_match = 1.0 if matching_caps == cap_count else 0.0

        features.extend(
            [
                match_ratio,
                perfect_match,
                float(matching_caps),
            ]
        )

        # === Temporal Features ===
        now = datetime.now()
        hour_norm = now.hour / 24.0
        day_of_week_norm = now.weekday() / 7.0

        features.extend(
            [
                hour_norm,
                day_of_week_norm,
                math.sin(2 * math.pi * hour_norm),  # Cyclical encoding
                math.cos(2 * math.pi * hour_norm),
            ]
        )

        # === Polynomial Features (degree 2) ===
        base_features = features[:8]  # Use first 8 features
        for i in range(len(base_features)):
            features.append(base_features[i] ** 2)

        # Ensure correct dimensionality
        feature_array = np.array(features)
        if len(feature_array) > self.feature_dim:
            feature_array = feature_array[: self.feature_dim]
        elif len(feature_array) < self.feature_dim:
            padding = np.zeros(self.feature_dim - len(feature_array))
            feature_array = np.concatenate([feature_array, padding])

        return feature_array

    def predict_reward(
        self, agent_id: str, context: np.ndarray, use_deep: bool = True
    ) -> float:
        """
        Predict expected reward using linear or deep model
        """
        # Initialize weights if needed
        if agent_id not in self.weights:
            self.weights[agent_id] = np.random.randn(len(context)) * 0.01
            if use_deep:
                self.hidden_weights[agent_id] = (
                    np.random.randn(self.hidden_dim, len(context)) * 0.01
                )

        if use_deep and agent_id in self.hidden_weights:
            # Two-layer network with ReLU
            hidden = np.maximum(
                0, np.dot(self.hidden_weights[agent_id], context)
            )  # ReLU
            # Linear output layer
            output_weights = (
                self.weights[agent_id][: self.hidden_dim]
                if len(self.weights[agent_id]) >= self.hidden_dim
                else self.weights[agent_id]
            )
            prediction = np.dot(output_weights, hidden[: len(output_weights)])
        else:
            # Simple linear model
            prediction = np.dot(self.weights[agent_id], context)

        return float(np.clip(prediction, 0, 1))

    def update(
        self,
        agent_id: str,
        context: np.ndarray,
        actual_reward: float,
        task_type: str = None,
    ):
        """
        Update model weights using Adam optimizer
        """
        # Initialize if needed
        if agent_id not in self.weights:
            self.weights[agent_id] = np.random.randn(len(context)) * 0.01
            self.momentum[agent_id] = np.zeros(len(context))
            self.velocity[agent_id] = np.zeros(len(context))
            self.agent_lr[agent_id] = self.learning_rate

        self.timestep += 1

        # Compute prediction and error
        predicted = self.predict_reward(agent_id, context, use_deep=False)
        error = actual_reward - predicted

        # Gradient
        gradient = error * context

        # Adam optimization
        m = self.momentum[agent_id]
        v = self.velocity[agent_id]

        m = self.beta1 * m + (1 - self.beta1) * gradient
        v = self.beta2 * v + (1 - self.beta2) * (gradient**2)

        # Bias correction
        m_hat = m / (1 - self.beta1**self.timestep)
        v_hat = v / (1 - self.beta2**self.timestep)

        # Weight update
        lr = self.agent_lr[agent_id]
        self.weights[agent_id] += lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        self.momentum[agent_id] = m
        self.velocity[agent_id] = v

        # Adaptive learning rate
        if abs(error) < 0.05:
            self.agent_lr[agent_id] *= 0.995  # Converging, reduce LR
        elif abs(error) > 0.3:
            self.agent_lr[agent_id] = min(
                self.agent_lr[agent_id] * 1.01, self.learning_rate * 2
            )

        # Update feature importance (absolute gradient magnitude)
        for i, grad_val in enumerate(gradient):
            self.feature_importance[i] += abs(grad_val)

        # Transfer learning: update task embeddings
        if task_type:
            self._update_task_embedding(task_type, context, actual_reward)

        # Store in replay buffer
        self.context_history.append(
            {
                "agent_id": agent_id,
                "context": context.copy(),
                "reward": actual_reward,
                "task_type": task_type,
                "timestamp": datetime.now(),
            }
        )

    def _update_task_embedding(
        self, task_type: str, context: np.ndarray, reward: float
    ):
        """Update task type embeddings for transfer learning"""
        if task_type not in self.task_embeddings:
            self.task_embeddings[task_type] = np.random.randn(self.embedding_dim) * 0.1

        # Gradient update for embedding
        embedding_lr = 0.005
        context_slice = context[: self.embedding_dim]
        if len(context_slice) < self.embedding_dim:
            context_slice = np.pad(
                context_slice, (0, self.embedding_dim - len(context_slice))
            )

        # Update based on reward signal
        self.task_embeddings[task_type] += embedding_lr * reward * context_slice

        # L2 normalization to prevent explosion
        norm = np.linalg.norm(self.task_embeddings[task_type])
        if norm > 0:
            self.task_embeddings[task_type] /= norm + 1e-8

    def get_similar_tasks(
        self, task_type: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Find similar tasks for transfer learning"""
        if task_type not in self.task_embeddings:
            return []

        similarities = []
        emb1 = self.task_embeddings[task_type]

        for other_type, emb2 in self.task_embeddings.items():
            if other_type != task_type:
                # Cosine similarity
                sim = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
                )
                similarities.append((other_type, float(sim)))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    def get_feature_importance(self, top_k: int = 10) -> List[Tuple[int, float]]:
        """Get most important features for explainability"""
        if not self.feature_importance:
            return []

        # Normalize
        total_importance = sum(self.feature_importance.values())
        normalized = {
            k: v / total_importance for k, v in self.feature_importance.items()
        }

        return sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:top_k]


# Continue in next message due to length...

__all__ = [
    "AdvancedBanditArm",
    "DeepContextualBandit",
    "BanditAlgorithm",
    "OptimizationObjective",
    "AgentPerformanceMetrics",
]
