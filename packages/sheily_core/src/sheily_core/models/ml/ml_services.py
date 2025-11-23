#!/usr/bin/env python3
"""
ML SERVICES - Enterprise ML Orchestration Layer
===============================================

Centralized ML services for Sheily MCP Enterprise:
✓ ML Coordinator Advanced management
✓ Deep contextual bandit services
✓ Transfer learning orchestration
✓ Model optimization and deployment
✓ Real-time ML model monitoring
✓ Adaptive learning strategies
✓ Enterprise ML model registry

@Author: Sheily MCP Enterprise
@Version: 2025.1.0
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from sheily_core.agents.base.enhanced_base import AgentCapability, AgentTask, TaskPriority

# Enterprise ML imports
from sheily_core.agents.coordination.ml_coordinator_advanced import (
    AdvancedBanditArm,
    BanditAlgorithm,
    DeepContextualBandit,
    OptimizationObjective,
)

try:
    from sheily_core.models.ml.qora_fine_tuning import QLoRAFineTuner
except ImportError:
    QLoRAFineTuner = None
    logging.getLogger(__name__).warning("⚠️ QLoRAFineTuner not available")

logger = logging.getLogger(__name__)


@dataclass
class MLModelMetrics:
    """Comprehensive ML model performance tracking"""

    model_id: str
    bandit_algorithm: str
    contextual_accuracy: float = 0.0
    bandit_regret: float = 0.0
    convergence_score: float = 0.0
    learning_rate: float = 0.01
    total_updates: int = 0
    active_experiments: int = 0
    last_training: Optional[datetime] = None
    model_health_score: float = 1.0


@dataclass
class MLServiceConfig:
    """ML services configuration"""

    max_concurrent_models: int = 100
    model_cache_size: int = 1000
    training_batch_size: int = 32
    learning_rate: float = 0.001
    monitoring_interval: int = 60
    auto_save_interval: int = 300


class MLModelRegistry:
    """Enterprise ML model registry with versioning and governance"""

    def __init__(self):
        self._models: Dict[str, MLModelMetrics] = {}
        self._model_versions: Dict[str, List[str]] = defaultdict(list)
        self._active_models: Set[str] = set()

    def register_model(self, model_id: str, algorithm: BanditAlgorithm) -> None:
        """Register a new ML model"""
        metrics = MLModelMetrics(
            model_id=model_id,
            bandit_algorithm=algorithm.value,
            last_training=datetime.now(),
        )
        self._models[model_id] = metrics
        self._model_versions[model_id].append(str(uuid.uuid4()))
        self._active_models.add(model_id)

        logger.info(f"📝 Registered ML model: {model_id} ({algorithm.value})")

    def unregister_model(self, model_id: str) -> None:
        """Unregister an ML model"""
        if model_id in self._models:
            del self._models[model_id]
            self._active_models.discard(model_id)
            logger.info(f"🗑️ Unregistered ML model: {model_id}")

    def get_model_status(self, model_id: str) -> Optional[MLModelMetrics]:
        """Get model status and metrics"""
        return self._models.get(model_id)

    def get_active_models(self) -> List[str]:
        """Get list of active models"""
        return list(self._active_models)

    def update_model_metrics(self, model_id: str, **metrics) -> None:
        """Update model performance metrics"""
        if model_id in self._models:
            model = self._models[model_id]
            for key, value in metrics.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            model.total_updates += 1
            logger.debug(f"📊 Updated metrics for {model_id}: {metrics}")


class MLCoordinatorService:
    """Central ML coordination service"""

    def __init__(self, config: MLServiceConfig = None):
        self.config = config or MLServiceConfig()
        self.registry = MLModelRegistry()

        # ML Model instances
        self._bandit_models: Dict[str, AdvancedBanditArm] = {}
        self._contextual_models: Dict[str, DeepContextualBandit] = {}
        self._qlora_tuner: Optional[QLoRAFineTuner] = None

        # Performance monitoring
        self._model_performance: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Model persistence
        self._model_cache_path = "models/ml_models_cache.json"
        self._last_save = datetime.now()

    async def start_service(self) -> None:
        """Start the ML coordination service"""
        logger.info("🚀 Starting ML Coordination Service")

        # Initialize QLoRA fine-tuner
        if QLoRAFineTuner:
            self._qlora_tuner = QLoRAFineTuner()
        else:
            self._qlora_tuner = None

        # Load cached models
        await self._load_cached_models()

        # Start background monitoring
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("✅ ML Coordination Service started")

    async def stop_service(self) -> None:
        """Stop the ML coordination service"""
        logger.info("🛑 Stopping ML Coordination Service")

        # Cancel background tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Save model state
        await self._save_model_cache()

        # Shutdown executor
        self._executor.shutdown(wait=True)

        logger.info("✅ ML Coordination Service stopped")

    async def create_ml_model(
        self,
        model_type: str,
        algorithm: BanditAlgorithm,
        model_id: Optional[str] = None,
    ) -> str:
        """Create a new ML model instance"""
        model_id = model_id or f"{model_type}_{uuid.uuid4().hex[:8]}"

        if model_type == "bandit":
            bandit = AdvancedBanditArm(model_id, algorithm)
            self._bandit_models[model_id] = bandit
            self.registry.register_model(model_id, algorithm)

        elif model_type == "contextual":
            contextual = DeepContextualBandit(learning_rate=self.config.learning_rate)
            self._contextual_models[model_id] = contextual
            self.registry.register_model(model_id, algorithm)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        logger.info(f"🎯 Created ML model: {model_id} ({model_type})")
        return model_id

    async def execute_ml_prediction(
        self, model_id: str, context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute ML prediction with real-time context"""
        try:
            start_time = time.time()

            if model_id in self._bandit_models:
                model = self._bandit_models[model_id]
                total_pulls = sum(m.sample_count for m in self._bandit_models.values())

                score = model.sample(total_pulls=total_pulls)
                prediction = {
                    "model_type": "bandit",
                    "algorithm": model.algorithm.value,
                    "score": score,
                    "confidence_interval": model.get_confidence_interval(),
                    "statistics": model.get_statistics(),
                }

            elif model_id in self._contextual_models:
                model = self._contextual_models[model_id]
                # Create mock task and agent for context extraction
                task = self._create_mock_task(context_data)
                agent = self._create_mock_agent(f"predict_{model_id}")

                context_vector = model.extract_context_vector(task, agent)
                reward_prediction = model.predict_reward(model_id, context_vector)

                prediction = {
                    "model_type": "contextual",
                    "reward_prediction": reward_prediction,
                    "feature_importance": model.get_feature_importance(top_k=5),
                    "similar_tasks": model.get_similar_tasks(task.task_type, top_k=3),
                }

            else:
                raise ValueError(f"Model not found: {model_id}")

            processing_time = time.time() - start_time

            # Update metrics
            self.registry.update_model_metrics(
                model_id, contextual_accuracy=1.0
            )  # Placeholder

            result = {
                "model_id": model_id,
                "prediction": prediction,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
            }

            # Cache performance
            self._model_performance[model_id].append(
                {
                    "processing_time": processing_time,
                    "success": True,
                    "timestamp": datetime.now(),
                }
            )

            return result

        except Exception as e:
            logger.error(f"❌ ML prediction failed for {model_id}: {e}")
            return {
                "model_id": model_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
            }

    async def train_ml_model(
        self, model_id: str, training_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Train ML model with new data"""
        try:
            if model_id in self._contextual_models:
                model = self._contextual_models[model_id]

                updates_count = 0
                for data_point in training_data:
                    task = self._create_mock_task(data_point)
                    agent = self._create_mock_agent(f"train_{model_id}_{updates_count}")

                    context = model.extract_context_vector(task, agent)
                    reward = data_point.get("reward", 0.5)

                    model.update(agent.agent_id, context, reward, task.task_type)
                    updates_count += 1

                # Update registry
                self.registry.update_model_metrics(
                    model_id,
                    total_updates=self.registry._models[model_id].total_updates
                    + updates_count,
                    last_training=datetime.now(),
                    convergence_score=0.8,  # Placeholder
                )

                return {
                    "model_id": model_id,
                    "updates_processed": updates_count,
                    "training_time": datetime.now().isoformat(),
                    "status": "success",
                }

            else:
                raise ValueError(f"Model {model_id} not suitable for training")

        except Exception as e:
            logger.error(f"❌ ML training failed for {model_id}: {e}")
            return {"model_id": model_id, "error": str(e), "status": "failed"}

    async def get_ml_model_report(self, model_id: str) -> Dict[str, Any]:
        """Generate comprehensive ML model report"""
        try:
            # Get basic metrics
            metrics = self.registry.get_model_status(model_id)
            if not metrics:
                raise ValueError(f"Model not found: {model_id}")

            # Get performance history
            performance_history = (
                list(self._model_performance[model_id])[-10:]
                if self._model_performance[model_id]
                else []
            )

            # Get model specific stats
            if model_id in self._bandit_models:
                bandit = self._bandit_models[model_id]
                model_stats = bandit.get_statistics()
                model_type = "bandit"
            elif model_id in self._contextual_models:
                contextual = self._contextual_models[model_id]
                model_stats = {
                    "feature_importance": contextual.get_feature_importance(top_k=10),
                    "timestep": contextual.timestep,
                    "active_embeddings": len(
                        [t for t in contextual.task_embeddings.keys() if t != ""]
                    ),
                }
                model_type = "contextual"
            else:
                model_stats = {}
                model_type = "unknown"

            report = {
                "model_id": model_id,
                "model_type": model_type,
                "metrics": {
                    "contextual_accuracy": metrics.contextual_accuracy,
                    "convergence_score": metrics.convergence_score,
                    "total_updates": metrics.total_updates,
                    "model_health_score": metrics.model_health_score,
                    "active_experiments": metrics.active_experiments,
                    "last_training": (
                        metrics.last_training.isoformat()
                        if metrics.last_training
                        else None
                    ),
                },
                "performance_history": performance_history,
                "model_statistics": model_stats,
                "performance_score": self._calculate_performance_score(metrics),
                "generated_at": datetime.now().isoformat(),
            }

            return report

        except Exception as e:
            logger.error(f"❌ Report generation failed for {model_id}: {e}")
            return {"error": str(e), "model_id": model_id, "status": "failed"}

    # ==================================================
    # MONITORING AND OPTIMIZATION
    # ==================================================

    async def _monitoring_loop(self) -> None:
        """Continuous ML model monitoring"""
        while True:
            try:
                await asyncio.sleep(self.config.monitoring_interval)

                # Monitor all active models
                for model_id in self.registry.get_active_models():
                    await self._check_model_health(model_id)

                # Auto-save if needed
                if (
                    datetime.now() - self._last_save
                ).seconds > self.config.auto_save_interval:
                    await self._save_model_cache()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ ML monitoring error: {e}")

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of old model states"""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour

                # Clean old performance data
                for model_id in list(self._model_performance.keys()):
                    if (
                        len(self._model_performance[model_id])
                        > self.config.model_cache_size
                    ):
                        # Keep only recent entries
                        self._model_performance[model_id] = deque(
                            list(self._model_performance[model_id])[
                                -self.config.model_cache_size :
                            ],
                            maxlen=self.config.model_cache_size,
                        )

                logger.info("🧹 ML model cache cleanup completed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ ML cleanup error: {e}")

    async def _check_model_health(self, model_id: str) -> None:
        """Check individual model health"""
        try:
            metrics = self.registry.get_model_status(model_id)
            if not metrics:
                return

            # Health checks
            health_score = 1.0

            # Recency check (models should be updated recently)
            if (
                metrics.last_training
                and (datetime.now() - metrics.last_training).days > 7
            ):
                health_score -= 0.2

            # Performance check
            if metrics.convergence_score < 0.5:
                health_score -= 0.3

            # Update capability check
            if metrics.total_updates < 10:
                health_score -= 0.1

            # Update health score
            self.registry.update_model_metrics(
                model_id, model_health_score=max(0.0, health_score)
            )

            if health_score < 0.5:
                logger.warning(
                    f"⚠️ Model {model_id} health degraded: {health_score:.2f}"
                )

        except Exception as e:
            logger.error(f"❌ Health check failed for {model_id}: {e}")

    async def _load_cached_models(self) -> None:
        """Load previously cached models"""
        try:
            if not os.path.exists(self._model_cache_path):
                return

            with open(self._model_cache_path, "r") as f:
                cache = json.load(f)

            # Restore registry from cache
            for model_id, model_data in cache.get("registry", {}).items():
                self.registry._models[model_id] = MLModelMetrics(**model_data)

            logger.info(f"💾 Loaded {len(self.registry._models)} cached ML models")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load ML model cache: {e}")

    async def _save_model_cache(self) -> None:
        """Save current model state to cache"""
        try:
            cache_data = {
                "registry": {mid: vars(m) for mid, m in self.registry._models.items()},
                "last_saved": datetime.now().isoformat(),
            }

            with open(self._model_cache_path, "w") as f:
                json.dump(cache_data, f, indent=2, default=str)

            self._last_save = datetime.now()
            logger.info(f"💾 Saved {len(self.registry._models)} ML models to cache")

        except Exception as e:
            logger.error(f"❌ Failed to save ML model cache: {e}")

    # ==================================================
    # UTILITY METHODS
    # ==================================================

    def _create_mock_task(self, data: Dict[str, Any]) -> AgentTask:
        """Create mock task for ML operations"""
        from sheily_core.agents.base.enhanced_base import AgentTask

        return AgentTask(
            task_id=f"mock_{uuid.uuid4().hex[:8]}",
            task_type=data.get("task_type", "general"),
            priority=TaskPriority(data.get("priority", 3)),
            required_capabilities=data.get("capabilities", []),
            timeout_seconds=data.get("timeout", 300),
            data=data,
        )

    def _create_mock_agent(self, agent_id: str) -> "EnhancedBaseMCPAgent":
        """Create mock agent for ML operations"""
        from sheily_core.agents.base.enhanced_base import (
            AgentPerformanceMetrics,
            AgentStatus,
            EnhancedBaseMCPAgent,
        )

        class MockAgent(EnhancedBaseMCPAgent):
            def __init__(self, agent_id: str):
                super().__init__(agent_id, [])
                self.status = AgentStatus.RUNNING

            async def process_task_impl(self, task):
                return {"result": "mock_processed", "quality_score": 0.8}

        return MockAgent(agent_id)

    def _calculate_performance_score(self, metrics: MLModelMetrics) -> float:
        """Calculate overall performance score"""
        score = 0.0

        # Contextual accuracy (40%)
        score += metrics.contextual_accuracy * 0.4

        # Convergence (30%)
        score += metrics.convergence_score * 0.3

        # Recency bonus (20%) - models updated recently get bonus
        if metrics.last_training:
            days_since_training = (datetime.now() - metrics.last_training).days
            recency_score = max(
                0, 1.0 - days_since_training / 30.0
            )  # Decay over 30 days
            score += recency_score * 0.2

        # Activity bonus (10%) - more updates = better
        activity_score = min(1.0, metrics.total_updates / 100.0)
        score += activity_score * 0.1

        return min(score, 1.0)


# ================================
# SERVICE INSTANCE MANAGEMENT
# ================================

_global_ml_service = None


def get_ml_service() -> MLCoordinatorService:
    """Get global ML service instance"""
    global _global_ml_service
    if _global_ml_service is None:
        config = MLServiceConfig()
        _global_ml_service = MLCoordinatorService(config)
    return _global_ml_service


async def initialize_ml_services():
    """Initialize ML services"""
    service = get_ml_service()
    await service.start_service()
    return service


if __name__ == "__main__":
    print("🤖 Sheily MCP Enterprise - ML Services")
    print(
        "Run: python -c \"from sheily_core.ml_services import get_ml_service; print('✅ ML Services loaded')\""
    )
