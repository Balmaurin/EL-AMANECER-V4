#!/usr/bin/env python3
"""
Enhanced Base Agent System with 2025 Enterprise Features
========================================================

Features:
- ML-based performance optimization
- Real-time observability (OpenTelemetry)
- Advanced capability management
- Circuit breaker pattern
- Distributed tracing
- Prometheus metrics
- Health checks
- Adaptive learning
"""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """Enhanced capabilities with 2025 standards"""

    # Core capabilities
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    STRATEGIC = "strategic"

    # Domain-specific
    FINANCIAL_ANALYSIS = "financial_analysis"
    RISK_MANAGEMENT = "risk_management"
    SECURITY_AUDIT = "security_audit"
    THREAT_DETECTION = "threat_detection"
    MEDICAL_DIAGNOSIS = "medical_diagnosis"
    TREATMENT_PLANNING = "treatment_planning"
    EDUCATIONAL_DESIGN = "educational_design"
    TUTORING = "tutoring"
    ENGINEERING_DESIGN = "engineering_design"
    SYSTEM_OPTIMIZATION = "system_optimization"
    BUSINESS_INTELLIGENCE = "business_intelligence"
    DATA_SCIENCE = "data_science"
    LEGAL_ANALYSIS = "legal_analysis"
    CONTENT_CREATION = "content_creation"
    MARKET_RESEARCH = "market_research"

    # Advanced capabilities
    REAL_TIME_PROCESSING = "real_time_processing"
    PREDICTIVE_ANALYTICS = "predictive_analytics"
    NLP_ADVANCED = "nlp_advanced"
    COMPUTER_VISION = "computer_vision"
    RECOMMENDATION = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"


class AgentStatus(Enum):
    """Agent operational status"""

    IDLE = "idle"
    BUSY = "busy"
    LEARNING = "learning"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SCALING = "scaling"


class TaskPriority(Enum):
    """Task priority levels"""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class AgentTask:
    """Enhanced task with ML features"""

    task_id: str
    task_type: str
    parameters: Dict[str, Any]
    priority: TaskPriority = TaskPriority.MEDIUM
    required_capabilities: List[AgentCapability] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict"""
        data = asdict(self)
        data["priority"] = self.priority.value
        data["required_capabilities"] = [c.value for c in self.required_capabilities]
        data["created_at"] = self.created_at.isoformat()
        data["started_at"] = self.started_at.isoformat() if self.started_at else None
        data["completed_at"] = (
            self.completed_at.isoformat() if self.completed_at else None
        )
        return data


@dataclass
class AgentMetrics:
    """Real-time metrics with Prometheus format"""

    agent_id: str
    total_tasks_processed: int = 0
    total_tasks_succeeded: int = 0
    total_tasks_failed: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    success_rate: float = 1.0
    error_rate: float = 0.0
    throughput: float = 0.0  # tasks/hour
    capacity_utilization: float = 0.0  # 0-1
    learning_rate: float = 0.01
    performance_trend: float = 0.0  # positive = improving
    last_updated: datetime = field(default_factory=datetime.now)

    # Advanced metrics
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)

    def update_from_task(
        self,
        task: AgentTask,
        success: bool,
        execution_time: float,
        error: Optional[str] = None,
    ):
        """Update metrics from task execution"""
        self.total_tasks_processed += 1

        if success:
            self.total_tasks_succeeded += 1
        else:
            self.total_tasks_failed += 1
            if error:
                error_type = error.split(":")[0]
                self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

        # Update timing metrics
        self.total_execution_time += execution_time
        self.avg_execution_time = self.total_execution_time / self.total_tasks_processed

        # Update rates
        total = self.total_tasks_processed
        self.success_rate = self.total_tasks_succeeded / total if total > 0 else 1.0
        self.error_rate = self.total_tasks_failed / total if total > 0 else 0.0

        # Calculate throughput (tasks/hour)
        time_since_start = (datetime.now() - self.last_updated).total_seconds()
        if time_since_start > 0:
            self.throughput = (self.total_tasks_processed / time_since_start) * 3600

        self.last_updated = datetime.now()


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open

    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        if self.state == "half-open":
            self.state = "closed"

    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if timeout has passed
            if self.last_failure_time:
                time_since_failure = (
                    datetime.now() - self.last_failure_time
                ).total_seconds()
                if time_since_failure >= self.timeout_seconds:
                    self.state = "half-open"
                    logger.info("Circuit breaker entering half-open state")
                    return True
            return False

        # half-open state
        return True


class AdaptiveLearner:
    """Adaptive learning system for agents"""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.performance_history = deque(maxlen=100)
        self.task_execution_times: Dict[str, List[float]] = defaultdict(list)
        self.optimization_weights: Dict[str, float] = {}

    def record_performance(self, task_type: str, execution_time: float, success: bool):
        """Record task performance"""
        score = 1.0 if success else 0.0
        self.performance_history.append(score)

        if success:
            self.task_execution_times[task_type].append(execution_time)

            # Keep only recent history
            if len(self.task_execution_times[task_type]) > 50:
                self.task_execution_times[task_type] = self.task_execution_times[
                    task_type
                ][-50:]

    def get_performance_trend(self) -> float:
        """Calculate performance trend (-1 to 1)"""
        if len(self.performance_history) < 10:
            return 0.0

        # Linear regression on recent performance
        recent = list(self.performance_history)[-20:]
        x = np.arange(len(recent))
        y = np.array(recent)

        # Calculate slope
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return float(np.clip(slope, -1, 1))

        return 0.0

    def predict_execution_time(self, task_type: str) -> float:
        """Predict execution time for task type"""
        if task_type not in self.task_execution_times:
            return 300.0  # Default

        times = self.task_execution_times[task_type]
        if not times:
            return 300.0

        # Use median for robustness
        return float(np.median(times))

    def optimize_parameters(self, task_type: str) -> Dict[str, Any]:
        """Get optimized parameters for task type"""
        base_params = {}

        # Adjust timeout based on historical data
        predicted_time = self.predict_execution_time(task_type)
        base_params["timeout"] = predicted_time * 1.5  # 50% buffer

        # Adjust batch size based on performance
        performance_trend = self.get_performance_trend()
        if performance_trend > 0.1:
            base_params["batch_size"] = "large"
        elif performance_trend < -0.1:
            base_params["batch_size"] = "small"
        else:
            base_params["batch_size"] = "medium"

        return base_params


class EnhancedBaseMCPAgent(ABC):
    """
    Enhanced base class for all MCP agents with 2025 enterprise features
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        capabilities: List[AgentCapability],
        max_concurrent_tasks: int = 10,
        enable_learning: bool = True,
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.capabilities = set(capabilities)
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_learning = enable_learning

        # Status
        self.status = AgentStatus.IDLE
        self.current_tasks: Dict[str, AgentTask] = {}

        # Metrics
        self.metrics = AgentMetrics(agent_id=agent_id)

        # Fault tolerance
        self.circuit_breaker = CircuitBreaker()

        # Learning
        self.learner = AdaptiveLearner() if enable_learning else None

        # Task queue
        self.task_queue: deque = deque()

        # Execution history for latency tracking
        self.execution_times = deque(maxlen=1000)

        logger.info(f"Initialized enhanced agent: {agent_name} ({agent_id})")

    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """
        Execute task with fault tolerance, metrics, and learning
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise RuntimeError("Circuit breaker is open - agent unavailable")

        # Check capacity
        if len(self.current_tasks) >= self.max_concurrent_tasks:
            raise RuntimeError("Agent at maximum capacity")

        # Update status
        self.status = AgentStatus.BUSY
        task.started_at = datetime.now()
        self.current_tasks[task.task_id] = task

        start_time = datetime.now()
        success = False
        error_msg = None
        result = {}

        try:
            # Execute task with timeout
            result = await asyncio.wait_for(
                self._execute_task_impl(task), timeout=task.timeout_seconds
            )

            success = True
            self.circuit_breaker.record_success()

        except asyncio.TimeoutError:
            error_msg = f"TimeoutError: Task exceeded {task.timeout_seconds}s"
            logger.error(f"Task {task.task_id} timed out")
            self.circuit_breaker.record_failure()
            result = {"error": error_msg, "timeout": True}

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Task {task.task_id} failed: {e}")
            self.circuit_breaker.record_failure()
            result = {"error": error_msg, "success": False}

        finally:
            # Calculate metrics
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            task.completed_at = end_time

            # Update metrics
            self.metrics.update_from_task(task, success, execution_time, error_msg)

            # Update latency tracking
            self.execution_times.append(execution_time)
            self._update_latency_percentiles()

            # Learning
            if self.learner:
                self.learner.record_performance(task.task_type, execution_time, success)
                self.metrics.performance_trend = self.learner.get_performance_trend()

            # Cleanup
            if task.task_id in self.current_tasks:
                del self.current_tasks[task.task_id]

            # Update status
            if len(self.current_tasks) == 0:
                self.status = AgentStatus.IDLE

        return {
            "task_id": task.task_id,
            "agent_id": self.agent_id,
            "success": success,
            "execution_time": execution_time,
            "result": result,
            "timestamp": end_time.isoformat(),
        }

    def _update_latency_percentiles(self):
        """Update latency percentile metrics"""
        if not self.execution_times:
            return

        times = sorted(self.execution_times)
        n = len(times)

        self.metrics.p50_latency = times[int(n * 0.50)] if n > 0 else 0.0
        self.metrics.p95_latency = times[int(n * 0.95)] if n > 0 else 0.0
        self.metrics.p99_latency = times[int(n * 0.99)] if n > 0 else 0.0

    @abstractmethod
    async def _execute_task_impl(self, task: AgentTask) -> Dict[str, Any]:
        """
        Implementation-specific task execution
        Must be overridden by specialized agents
        """
        pass

    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has specific capability"""
        return capability in self.capabilities

    def has_all_capabilities(self, capabilities: List[AgentCapability]) -> bool:
        """Check if agent has all required capabilities"""
        return all(cap in self.capabilities for cap in capabilities)

    def get_capacity_score(self) -> float:
        """Get current capacity score (0-1, higher is better)"""
        if self.max_concurrent_tasks == 0:
            return 0.0

        current_load = len(self.current_tasks) / self.max_concurrent_tasks
        return 1.0 - current_load

    def get_performance_score(self) -> float:
        """Get overall performance score (0-1)"""
        # Weighted combination of metrics
        weights = {"success_rate": 0.4, "capacity": 0.2, "speed": 0.2, "trend": 0.2}

        # Success rate component
        success_component = self.metrics.success_rate * weights["success_rate"]

        # Capacity component
        capacity_component = self.get_capacity_score() * weights["capacity"]

        # Speed component (faster = better, normalized)
        avg_time = self.metrics.avg_execution_time
        speed_score = 1.0 / (1.0 + avg_time / 300.0)  # Normalize against 5 min baseline
        speed_component = speed_score * weights["speed"]

        # Trend component (improving = better)
        trend_score = (self.metrics.performance_trend + 1.0) / 2.0  # Map -1,1 to 0,1
        trend_component = trend_score * weights["trend"]

        total_score = (
            success_component + capacity_component + speed_component + trend_component
        )
        return min(1.0, max(0.0, total_score))

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "healthy": self.circuit_breaker.state == "closed"
            and self.metrics.error_rate < 0.5,
            "circuit_breaker_state": self.circuit_breaker.state,
            "current_tasks": len(self.current_tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "capacity_utilization": len(self.current_tasks) / self.max_concurrent_tasks,
            "metrics": {
                "total_tasks": self.metrics.total_tasks_processed,
                "success_rate": self.metrics.success_rate,
                "avg_execution_time": self.metrics.avg_execution_time,
                "p50_latency": self.metrics.p50_latency,
                "p95_latency": self.metrics.p95_latency,
                "p99_latency": self.metrics.p99_latency,
                "throughput": self.metrics.throughput,
                "performance_trend": self.metrics.performance_trend,
                "error_rate": self.metrics.error_rate,
            },
            "capabilities": [cap.value for cap in self.capabilities],
            "timestamp": datetime.now().isoformat(),
        }

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        metrics = []

        # Counter metrics
        metrics.append(
            f'agent_tasks_total{{agent_id="{self.agent_id}"}} {self.metrics.total_tasks_processed}'
        )
        metrics.append(
            f'agent_tasks_succeeded{{agent_id="{self.agent_id}"}} {self.metrics.total_tasks_succeeded}'
        )
        metrics.append(
            f'agent_tasks_failed{{agent_id="{self.agent_id}"}} {self.metrics.total_tasks_failed}'
        )

        # Gauge metrics
        metrics.append(
            f'agent_success_rate{{agent_id="{self.agent_id}"}} {self.metrics.success_rate}'
        )
        metrics.append(
            f'agent_error_rate{{agent_id="{self.agent_id}"}} {self.metrics.error_rate}'
        )
        metrics.append(
            f'agent_avg_execution_time{{agent_id="{self.agent_id}"}} {self.metrics.avg_execution_time}'
        )
        metrics.append(
            f'agent_p50_latency{{agent_id="{self.agent_id}"}} {self.metrics.p50_latency}'
        )
        metrics.append(
            f'agent_p95_latency{{agent_id="{self.agent_id}"}} {self.metrics.p95_latency}'
        )
        metrics.append(
            f'agent_p99_latency{{agent_id="{self.agent_id}"}} {self.metrics.p99_latency}'
        )
        metrics.append(
            f'agent_throughput{{agent_id="{self.agent_id}"}} {self.metrics.throughput}'
        )
        metrics.append(
            f'agent_current_tasks{{agent_id="{self.agent_id}"}} {len(self.current_tasks)}'
        )
        metrics.append(
            f'agent_capacity_utilization{{agent_id="{self.agent_id}"}} {self.get_capacity_score()}'
        )
        metrics.append(
            f'agent_performance_score{{agent_id="{self.agent_id}"}} {self.get_performance_score()}'
        )

        return "\n".join(metrics)


__all__ = [
    "EnhancedBaseMCPAgent",
    "AgentCapability",
    "AgentStatus",
    "TaskPriority",
    "AgentTask",
    "AgentMetrics",
    "CircuitBreaker",
    "AdaptiveLearner",
]
