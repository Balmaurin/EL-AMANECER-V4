"""
Sheily AI - Multi-Modal AI Models Registry
Phase 2: Intelligence - Multi-Modal AI Integration

This module provides a comprehensive registry system for managing
multi-modal AI models from different providers with intelligent
routing and optimization capabilities.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported AI model types"""

    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    CODE = "code"
    EMBEDDING = "embedding"


class Provider(Enum):
    """Supported AI providers"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    META = "meta"
    MICROSOFT = "microsoft"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class ModelCapability(Enum):
    """Model capabilities"""

    TEXT_GENERATION = "text_generation"
    CODE_ANALYSIS = "code_analysis"
    CODE_GENERATION = "code_generation"
    IMAGE_ANALYSIS = "image_analysis"
    OCR = "ocr"
    VISUAL_REASONING = "visual_reasoning"
    SPEECH_RECOGNITION = "speech_recognition"
    AUDIO_ANALYSIS = "audio_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    EMBEDDING_GENERATION = "embedding_generation"
    MULTIMODAL_UNDERSTANDING = "multimodal_understanding"
    REASONING = "reasoning"
    MATHEMATICAL = "mathematical"
    SCIENTIFIC = "scientific"


@dataclass
class ModelMetrics:
    """Model performance metrics"""

    accuracy: float = 0.0
    latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    cost_per_token: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ModelConstraints:
    """Model operational constraints"""

    max_tokens: int = 4096
    max_images: int = 1
    max_audio_duration_sec: int = 300
    supported_languages: List[str] = field(default_factory=lambda: ["en"])
    rate_limit_per_minute: int = 60
    context_window: int = 4096


class ModelRegistration(BaseModel):
    """Model registration data"""

    name: str = Field(..., description="Unique model name")
    type: ModelType = Field(..., description="Model type")
    provider: Provider = Field(..., description="AI provider")
    version: str = Field(..., description="Model version")
    capabilities: List[ModelCapability] = Field(
        default_factory=list, description="Model capabilities"
    )
    constraints: ModelConstraints = Field(
        default_factory=ModelConstraints, description="Operational constraints"
    )
    metrics: ModelMetrics = Field(
        default_factory=ModelMetrics, description="Performance metrics"
    )
    api_endpoint: str = Field(..., description="API endpoint")
    api_key_required: bool = Field(
        default=True, description="Whether API key is required"
    )
    is_active: bool = Field(default=True, description="Whether model is active")
    tags: List[str] = Field(default_factory=list, description="Model tags")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("Model name cannot be empty")
        return v.strip()

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, v: List[ModelCapability]) -> List[ModelCapability]:
        if not v:
            raise ValueError("Model must have at least one capability")
        return v


class MultiModalModelRegistry:
    """
    Registry system for managing multi-modal AI models with intelligent
    routing and optimization capabilities.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.models: Dict[str, ModelRegistration] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize the registry system"""
        if self._initialized:
            return

        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis.from_url(
                self.redis_url, decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Connected to Redis for model registry")

            # Load existing models from Redis
            await self._load_models_from_cache()

            self._initialized = True
            logger.info("Multi-modal model registry initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize model registry: {e}")
            raise

    async def _load_models_from_cache(self):
        """Load models from Redis cache"""
        try:
            keys = await self.redis_client.keys("model:*")
            for key in keys:
                model_data = await self.redis_client.get(key)
                if model_data:
                    model_dict = json.loads(model_data)
                    # Convert string timestamps back to datetime
                    if (
                        "metrics" in model_dict
                        and "last_updated" in model_dict["metrics"]
                    ):
                        model_dict["metrics"]["last_updated"] = datetime.fromisoformat(
                            model_dict["metrics"]["last_updated"]
                        )
                    model = ModelRegistration(**model_dict)
                    self.models[model.name] = model

            logger.info(f"Loaded {len(self.models)} models from cache")

        except Exception as e:
            logger.warning(f"Failed to load models from cache: {e}")

    async def register_model(self, model_data: Dict[str, Any]) -> str:
        """
        Register a new AI model in the registry

        Args:
            model_data: Model registration data

        Returns:
            Model name
        """
        await self.initialize()

        try:
            # Create model registration
            model = ModelRegistration(**model_data)
            model_name = model.name

            # Store in memory
            self.models[model_name] = model

            # Store in Redis cache
            model_dict = model.model_dump()
            # Convert datetime to ISO string for JSON serialization
            if "metrics" in model_dict and "last_updated" in model_dict["metrics"]:
                model_dict["metrics"][
                    "last_updated"
                ] = model.metrics.last_updated.isoformat()

            await self.redis_client.setex(
                f"model:{model_name}",
                timedelta(days=30),  # Cache for 30 days
                json.dumps(model_dict),
            )

            # Update model index
            await self._update_model_index(model)

            logger.info(f"Registered model: {model_name}")
            return model_name

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    async def _update_model_index(self, model: ModelRegistration):
        """Update search index for model discovery"""
        try:
            # Index by capabilities
            for capability in model.capabilities:
                await self.redis_client.sadd(
                    f"capability:{capability.value}", model.name
                )

            # Index by provider
            await self.redis_client.sadd(f"provider:{model.provider.value}", model.name)

            # Index by type
            await self.redis_client.sadd(f"type:{model.type.value}", model.name)

            # Index by tags
            for tag in model.tags:
                await self.redis_client.sadd(f"tag:{tag}", model.name)

        except Exception as e:
            logger.warning(f"Failed to update model index: {e}")

    async def get_model(self, model_name: str) -> Optional[ModelRegistration]:
        """Get model by name"""
        await self.initialize()
        return self.models.get(model_name)

    async def list_models(
        self,
        model_type: Optional[ModelType] = None,
        provider: Optional[Provider] = None,
        capability: Optional[ModelCapability] = None,
        tag: Optional[str] = None,
        active_only: bool = True,
    ) -> List[ModelRegistration]:
        """List models with optional filtering"""
        await self.initialize()

        models = list(self.models.values())

        if active_only:
            models = [m for m in models if m.is_active]

        if model_type:
            models = [m for m in models if m.type == model_type]

        if provider:
            models = [m for m in models if m.provider == provider]

        if capability:
            models = [m for m in models if capability in m.capabilities]

        if tag:
            models = [m for m in models if tag in m.tags]

        return models

    async def update_model_metrics(self, model_name: str, metrics: Dict[str, Any]):
        """Update model performance metrics"""
        await self.initialize()

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]

        # Update metrics
        for key, value in metrics.items():
            if hasattr(model.metrics, key):
                setattr(model.metrics, key, value)

        model.metrics.last_updated = datetime.now()

        # Update cache
        model_dict = model.model_dump()
        model_dict["metrics"]["last_updated"] = model.metrics.last_updated.isoformat()

        await self.redis_client.setex(
            f"model:{model_name}", timedelta(days=30), json.dumps(model_dict)
        )

        logger.info(f"Updated metrics for model: {model_name}")

    async def deactivate_model(self, model_name: str):
        """Deactivate a model"""
        await self.initialize()

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        self.models[model_name].is_active = False

        # Update cache
        model_dict = self.models[model_name].model_dump()
        model_dict["metrics"]["last_updated"] = self.models[
            model_name
        ].metrics.last_updated.isoformat()

        await self.redis_client.setex(
            f"model:{model_name}", timedelta(days=30), json.dumps(model_dict)
        )

        logger.info(f"Deactivated model: {model_name}")

    async def find_best_model(
        self, task_requirements: Dict[str, Any]
    ) -> Optional[ModelRegistration]:
        """
        Find the best model for a given task based on requirements

        Args:
            task_requirements: Task requirements including type, capabilities, constraints

        Returns:
            Best matching model or None
        """
        await self.initialize()

        candidates = await self.list_models(active_only=True)

        # Filter by basic requirements
        if "type" in task_requirements:
            candidates = [
                m for m in candidates if m.type.value == task_requirements["type"]
            ]

        if "capabilities" in task_requirements:
            required_caps = set(task_requirements["capabilities"])
            candidates = [
                m
                for m in candidates
                if required_caps.issubset(set(c.value for c in m.capabilities))
            ]

        if not candidates:
            return None

        # Score candidates based on various factors
        scored_candidates = []
        for model in candidates:
            score = await self._score_model(model, task_requirements)
            scored_candidates.append((model, score))

        # Return highest scoring model
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        best_model = scored_candidates[0][0]

        logger.info(
            f"Selected best model '{best_model.name}' with score {scored_candidates[0][1]:.2f}"
        )
        return best_model

    async def _score_model(
        self, model: ModelRegistration, requirements: Dict[str, Any]
    ) -> float:
        """Score a model based on requirements"""
        score = 0.0

        # Base score for being active and functional
        score += 50.0

        # Performance score (higher accuracy, lower latency = higher score)
        if model.metrics.accuracy > 0:
            score += model.metrics.accuracy * 20  # 0-20 points

        if model.metrics.latency_ms > 0:
            # Lower latency = higher score (max 15 points)
            latency_score = max(0, 15 - (model.metrics.latency_ms / 100))
            score += latency_score

        # Cost efficiency (lower cost = higher score)
        if model.metrics.cost_per_token > 0:
            cost_score = max(0, 10 - (model.metrics.cost_per_token * 10000))
            score += cost_score

        # Capability match bonus
        if "capabilities" in requirements:
            required_caps = set(requirements["capabilities"])
            model_caps = set(c.value for c in model.capabilities)
            match_ratio = len(required_caps.intersection(model_caps)) / len(
                required_caps
            )
            score += match_ratio * 10  # 0-10 points

        # Recency bonus (newer metrics = slight preference)
        days_since_update = (datetime.now() - model.metrics.last_updated).days
        recency_score = max(0, 5 - days_since_update)
        score += recency_score

        return score

    async def get_model_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        await self.initialize()

        total_models = len(self.models)
        active_models = len([m for m in self.models.values() if m.is_active])

        provider_stats = {}
        type_stats = {}
        capability_stats = {}

        for model in self.models.values():
            # Provider stats
            provider = model.provider.value
            provider_stats[provider] = provider_stats.get(provider, 0) + 1

            # Type stats
            model_type = model.type.value
            type_stats[model_type] = type_stats.get(model_type, 0) + 1

            # Capability stats
            for cap in model.capabilities:
                cap_name = cap.value
                capability_stats[cap_name] = capability_stats.get(cap_name, 0) + 1

        return {
            "total_models": total_models,
            "active_models": active_models,
            "inactive_models": total_models - active_models,
            "provider_distribution": provider_stats,
            "type_distribution": type_stats,
            "capability_distribution": capability_stats,
        }

    async def cleanup_expired_models(self, max_age_days: int = 90):
        """Clean up models with old metrics"""
        await self.initialize()

        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        expired_models = []

        for name, model in self.models.items():
            if model.metrics.last_updated < cutoff_date:
                expired_models.append(name)

        for name in expired_models:
            await self.redis_client.delete(f"model:{name}")
            del self.models[name]

        if expired_models:
            logger.info(f"Cleaned up {len(expired_models)} expired models")

        return len(expired_models)


# Global registry instance
model_registry = MultiModalModelRegistry()


async def get_model_registry() -> MultiModalModelRegistry:
    """Get the global model registry instance"""
    await model_registry.initialize()
    return model_registry


# Example usage and initialization
async def initialize_default_models():
    """Initialize registry with default models"""
    registry = await get_model_registry()

    # GPT-5 Text Model
    await registry.register_model(
        {
            "name": "gpt5-text",
            "type": ModelType.TEXT,
            "provider": Provider.OPENAI,
            "version": "gpt-5-turbo",
            "capabilities": [
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE_ANALYSIS,
                ModelCapability.REASONING,
                ModelCapability.MATHEMATICAL,
            ],
            "constraints": {
                "max_tokens": 128000,
                "context_window": 128000,
                "rate_limit_per_minute": 10000,
            },
            "metrics": {
                "accuracy": 0.95,
                "latency_ms": 250,
                "throughput_tokens_per_sec": 150,
                "cost_per_token": 0.00015,
            },
            "api_endpoint": "https://api.openai.com/v1/chat/completions",
            "tags": ["text", "reasoning", "code", "premium"],
        }
    )

    # Claude 3 Vision Model
    await registry.register_model(
        {
            "name": "claude3-vision",
            "type": ModelType.VISION,
            "provider": Provider.ANTHROPIC,
            "version": "claude-3-opus",
            "capabilities": [
                ModelCapability.IMAGE_ANALYSIS,
                ModelCapability.OCR,
                ModelCapability.VISUAL_REASONING,
                ModelCapability.MULTIMODAL_UNDERSTANDING,
            ],
            "constraints": {
                "max_images": 20,
                "max_tokens": 4096,
                "supported_languages": [
                    "en",
                    "es",
                    "fr",
                    "de",
                    "it",
                    "pt",
                    "zh",
                    "ja",
                    "ko",
                ],
            },
            "metrics": {
                "accuracy": 0.92,
                "latency_ms": 800,
                "throughput_tokens_per_sec": 80,
                "cost_per_token": 0.00025,
            },
            "api_endpoint": "https://api.anthropic.com/v1/messages",
            "tags": ["vision", "multimodal", "analysis", "premium"],
        }
    )

    # Gemini Pro Audio Model
    await registry.register_model(
        {
            "name": "gemini-pro-audio",
            "type": ModelType.AUDIO,
            "provider": Provider.GOOGLE,
            "version": "gemini-pro",
            "capabilities": [
                ModelCapability.SPEECH_RECOGNITION,
                ModelCapability.AUDIO_ANALYSIS,
                ModelCapability.SENTIMENT_ANALYSIS,
            ],
            "constraints": {
                "max_audio_duration_sec": 300,
                "supported_languages": [
                    "en",
                    "es",
                    "fr",
                    "de",
                    "it",
                    "pt",
                    "zh",
                    "ja",
                    "ko",
                ],
                "sample_rate": 16000,
            },
            "metrics": {
                "accuracy": 0.94,
                "latency_ms": 500,
                "throughput_tokens_per_sec": 120,
                "cost_per_token": 0.00012,
            },
            "api_endpoint": "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent",
            "tags": ["audio", "speech", "sentiment", "multilingual"],
        }
    )

    logger.info("Default models initialized successfully")


if __name__ == "__main__":
    # Example usage
    async def main():
        try:
            # Initialize registry
            registry = await get_model_registry()

            # Initialize default models
            await initialize_default_models()

            # List all models
            models = await registry.list_models()
            print(f"Registered models: {len(models)}")

            for model in models:
                print(f"- {model.name} ({model.type.value}) - {model.provider.value}")

            # Find best model for text generation
            best_text_model = await registry.find_best_model(
                {"type": "text", "capabilities": ["text_generation", "reasoning"]}
            )

            if best_text_model:
                print(f"\nBest text model: {best_text_model.name}")

            # Get registry statistics
            stats = await registry.get_model_stats()
            print(f"\nRegistry stats: {stats}")

        except Exception as e:
            logger.error(f"Error in main: {e}")

    asyncio.run(main())
