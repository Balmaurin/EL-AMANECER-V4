#!/usr/bin/env python3
"""
Sheily Enterprise Event-Driven Architecture
==========================================

Sistema de eventos enterprise integrado con ii-agent patterns para
coordinación real-time entre los 4 agentes especializados core (Finance, Security, Healthcare, Business) del sistema Sheily MCP.

Características:
- AsyncEventStream con hook registry
- Real-time event propagation
- Thread-safe subscriber management
- Event filtering y modification
- Integration con MCP Enterprise Master
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class SheìlyEventType(str, Enum):
    """Tipos de eventos Sheily Enterprise"""

    # Agent Events
    AGENT_STATUS_UPDATE = "agent_status_update"
    AGENT_TASK_STARTED = "agent_task_started"
    AGENT_TASK_COMPLETED = "agent_task_completed"
    AGENT_ERROR = "agent_error"

    # Neural Events
    NEURAL_WEIGHT_UPDATE = "neural_weight_update"
    NEURAL_TRAINING_STATUS = "neural_training_status"

    # MCP Events
    MCP_COORDINATION = "mcp_coordination"
    MCP_LAYER_SYNC = "mcp_layer_sync"

    # System Events
    SYSTEM_HEALTH = "system_health"
    METRICS_UPDATE = "metrics_update"
    PERFORMANCE_ALERT = "performance_alert"

    # Integration Events (from ii-agent)
    TOOL_EXECUTION = "tool_execution"
    SESSION_UPDATE = "session_update"
    REALTIME_COMMUNICATION = "realtime_communication"


@dataclass
class SheìlyRealtimeEvent:
    """Evento real-time Sheily Enterprise"""

    session_id: Optional[UUID]
    run_id: Optional[UUID]
    type: SheìlyEventType
    content: Dict[str, Any]
    timestamp: datetime = None
    agent_id: Optional[str] = None
    source: str = "sheily_enterprise"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class EventHook(ABC):
    """Abstract base class for event hooks"""

    @abstractmethod
    async def process_event(
        self, event: SheìlyRealtimeEvent
    ) -> Optional[SheìlyRealtimeEvent]:
        """Process event before propagation"""
        pass

    @abstractmethod
    def should_process(self, event: SheìlyRealtimeEvent) -> bool:
        """Determine if hook should process event"""
        pass


class EventSubscriber(ABC):
    """Abstract base class for event subscribers"""

    @abstractmethod
    async def handle_event(self, event: SheìlyRealtimeEvent) -> None:
        """Handle incoming event"""
        pass

    async def should_handle(self, event: SheìlyRealtimeEvent) -> bool:
        """Determine if subscriber should handle event"""
        return True


class EventHookRegistry:
    """Registry for managing event hooks"""

    def __init__(self):
        self._hooks: List[EventHook] = []

    def register_hook(self, hook: EventHook) -> None:
        """Register an event hook"""
        self._hooks.append(hook)

    def unregister_hook(self, hook: EventHook) -> None:
        """Unregister an event hook"""
        if hook in self._hooks:
            self._hooks.remove(hook)

    async def process_event(
        self, event: SheìlyRealtimeEvent
    ) -> Optional[SheìlyRealtimeEvent]:
        """Process event through all registered hooks"""
        current_event = event

        for hook in self._hooks:
            if hook.should_process(current_event):
                processed_event = await hook.process_event(current_event)
                if processed_event is None:
                    return None
                current_event = processed_event

        return current_event

    def clear_hooks(self) -> None:
        """Remove all registered hooks"""
        self._hooks.clear()


class SheìlyEventStream:
    """Enterprise Event Stream for Sheily"""

    def __init__(self, logger: logging.Logger = None):
        self._subscribers: Set[EventSubscriber] = set()
        self._lock = Lock()
        self._logger = logger or logging.getLogger(__name__)
        self._hook_registry = EventHookRegistry()
        self._event_history: List[SheìlyRealtimeEvent] = []
        self._max_history = 1000

    async def publish(self, event: SheìlyRealtimeEvent) -> None:
        """Publish event to all subscribers"""
        try:
            # Process event through hooks first
            processed_event = await self._hook_registry.process_event(event)
        except Exception as e:
            self._logger.error(f"Error processing event hooks: {e}")
            processed_event = event

        # If event was filtered out by hooks, don't propagate
        if processed_event is None:
            return

        # Add to history
        with self._lock:
            self._event_history.append(processed_event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

        # Notify all subscribers
        with self._lock:
            subscribers = self._subscribers.copy()

        for subscriber in subscribers:
            try:
                if await subscriber.should_handle(processed_event):
                    asyncio.create_task(subscriber.handle_event(processed_event))
            except Exception as e:
                self._logger.error(
                    f"Error in event subscriber {type(subscriber).__name__}: {e}"
                )

    def subscribe(self, subscriber: EventSubscriber) -> None:
        """Subscribe to events"""
        with self._lock:
            self._subscribers.add(subscriber)

    def unsubscribe(self, subscriber: EventSubscriber) -> None:
        """Unsubscribe from events"""
        with self._lock:
            self._subscribers.discard(subscriber)

    def clear_subscribers(self) -> None:
        """Remove all subscribers"""
        with self._lock:
            self._subscribers.clear()

    def get_subscriber_count(self) -> int:
        """Get number of active subscribers"""
        with self._lock:
            return len(self._subscribers)

    def register_hook(self, hook: EventHook) -> None:
        """Register an event hook"""
        self._hook_registry.register_hook(hook)

    def get_event_history(
        self, event_type: Optional[SheìlyEventType] = None, limit: int = 100
    ) -> List[SheìlyRealtimeEvent]:
        """Get event history with optional filtering"""
        with self._lock:
            events = self._event_history.copy()

        if event_type:
            events = [e for e in events if e.type == event_type]

        return events[-limit:] if limit else events


# ========================================
# SPECIALIZED EVENT HOOKS
# ========================================


class AgentCoordinationHook(EventHook):
    """Hook para coordinación entre agentes MCP"""

    def __init__(self, mcp_manager):
        self.mcp_manager = mcp_manager

    async def process_event(
        self, event: SheìlyRealtimeEvent
    ) -> Optional[SheìlyRealtimeEvent]:
        """Process agent coordination events"""
        if event.type == SheìlyEventType.AGENT_TASK_STARTED:
            # Notify MCP coordinator
            if self.mcp_manager:
                await self.mcp_manager.notify_task_started(
                    event.agent_id, event.content
                )

        # Always propagate event
        return event

    def should_process(self, event: SheìlyRealtimeEvent) -> bool:
        """Process all agent-related events"""
        return event.type.value.startswith("agent_")


class MetricsCollectionHook(EventHook):
    """Hook para recolección de métricas enterprise"""

    def __init__(self):
        self.metrics = {}

    async def process_event(
        self, event: SheìlyRealtimeEvent
    ) -> Optional[SheìlyRealtimeEvent]:
        """Collect metrics from events"""
        metric_key = f"{event.type.value}_{event.agent_id or 'system'}"

        if metric_key not in self.metrics:
            self.metrics[metric_key] = 0
        self.metrics[metric_key] += 1

        # Add metrics to event content
        event.content["_metrics"] = {
            "total_events": sum(self.metrics.values()),
            "event_count": self.metrics[metric_key],
        }

        return event

    def should_process(self, event: SheìlyRealtimeEvent) -> bool:
        """Process all events for metrics"""
        return True


# ========================================
# SPECIALIZED SUBSCRIBERS
# ========================================


class AgentStatusSubscriber(EventSubscriber):
    """Subscriber para status de agentes"""

    def __init__(self):
        self.agent_status = {}

    async def handle_event(self, event: SheìlyRealtimeEvent) -> None:
        """Handle agent status events"""
        if event.agent_id:
            self.agent_status[event.agent_id] = {
                "last_event": event.type.value,
                "timestamp": event.timestamp,
                "content": event.content,
            }

    async def should_handle(self, event: SheìlyRealtimeEvent) -> bool:
        """Handle agent-related events"""
        return event.type.value.startswith("agent_")


class NeuralSystemSubscriber(EventSubscriber):
    """Subscriber para eventos del sistema neural"""

    def __init__(self):
        self.neural_status = {
            "training_active": False,
            "last_weight_update": None,
            "performance_metrics": {},
        }

    async def handle_event(self, event: SheìlyRealtimeEvent) -> None:
        """Handle neural system events"""
        if event.type == SheìlyEventType.NEURAL_WEIGHT_UPDATE:
            self.neural_status["last_weight_update"] = event.timestamp
            self.neural_status["performance_metrics"].update(
                event.content.get("metrics", {})
            )

        elif event.type == SheìlyEventType.NEURAL_TRAINING_STATUS:
            self.neural_status["training_active"] = event.content.get("active", False)

    async def should_handle(self, event: SheìlyRealtimeEvent) -> bool:
        """Handle neural-related events"""
        return event.type.value.startswith("neural_")


# ========================================
# GLOBAL INSTANCE
# ========================================

# Global event stream instance
_sheily_event_stream: Optional[SheìlyEventStream] = None


def get_event_stream() -> SheìlyEventStream:
    """Get global event stream instance"""
    global _sheily_event_stream
    if _sheily_event_stream is None:
        _sheily_event_stream = SheìlyEventStream()

        # Register default hooks
        _sheily_event_stream.register_hook(MetricsCollectionHook())

        # Register default subscribers
        _sheily_event_stream.subscribe(AgentStatusSubscriber())
        _sheily_event_stream.subscribe(NeuralSystemSubscriber())

        logger.info("Sheily Enterprise Event Stream initialized")

    return _sheily_event_stream


async def publish_event(
    event_type: SheìlyEventType,
    content: Dict[str, Any],
    session_id: Optional[UUID] = None,
    agent_id: Optional[str] = None,
) -> None:
    """Convenience function to publish events"""
    event = SheìlyRealtimeEvent(
        session_id=session_id,
        run_id=uuid4(),
        type=event_type,
        content=content,
        agent_id=agent_id,
    )

    stream = get_event_stream()
    await stream.publish(event)


# ========================================
# INTEGRATION UTILITIES
# ========================================


async def initialize_event_system() -> SheìlyEventStream:
    """Initialize the complete event system"""
    stream = get_event_stream()

    # Publish system startup event
    await publish_event(
        SheìlyEventType.SYSTEM_HEALTH,
        {
            "status": "event_system_initialized",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )

    logger.info("✅ Sheily Enterprise Event System initialized")
    return stream


__all__ = [
    "SheìlyEventType",
    "SheìlyRealtimeEvent",
    "EventHook",
    "EventSubscriber",
    "SheìlyEventStream",
    "get_event_stream",
    "publish_event",
    "initialize_event_system",
    "AgentCoordinationHook",
    "MetricsCollectionHook",
    "AgentStatusSubscriber",
    "NeuralSystemSubscriber",
]
