#!/usr/bin/env python3
"""
Sheily Enterprise Background Processing & Scheduling System
==========================================================

Sistema de procesamiento en background y scheduling avanzado
integrado con ii-agent patterns para self-healing y automation.

Características:
- AsyncIOScheduler para tareas programadas
- Automatic cleanup y recovery de agentes
- Health monitoring automation
- Resource optimization schedules
- Integration con Event System
"""

import asyncio
import logging

# Use threading.Timer as fallback for scheduling
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from threading import Timer
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sheily_core.core.events.event_system import (
    SheìlyEventType,
    get_event_stream,
    publish_event,
)

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Prioridad de tareas programadas"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Estado de las tareas"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTask:
    """Tarea programada enterprise"""

    task_id: str
    name: str
    function: Callable[[], Awaitable[None]]
    priority: TaskPriority
    trigger: str  # 'interval' or 'cron'
    trigger_config: Dict[str, Any]
    max_retries: int = 3
    timeout_seconds: int = 300
    status: TaskStatus = TaskStatus.PENDING
    created_at: Optional[datetime] = None
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    error_count: int = 0
    timer: Optional[Timer] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class SheìlyScheduler:
    """Enterprise scheduler para Sheily usando asyncio"""

    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self.event_stream = None
        self._task_handlers: Dict[str, asyncio.Task] = {}

    async def initialize(self) -> None:
        """Initialize scheduler system"""
        try:
            self.event_stream = get_event_stream()

            # Register default maintenance tasks
            await self._register_default_tasks()

            self.running = True

            await publish_event(
                SheìlyEventType.SYSTEM_HEALTH,
                {"status": "scheduler_initialized", "task_count": len(self.tasks)},
            )

            logger.info("✅ Sheily Enterprise Scheduler initialized")

        except Exception as e:
            logger.error(f"Error initializing scheduler: {e}")
            raise

    async def _register_default_tasks(self) -> None:
        """Register default system maintenance tasks"""

        # Agent health monitoring every 5 minutes
        await self.add_task(
            task_id="agent_health_check",
            name="Agent Health Monitor",
            function=self._monitor_agent_health,
            priority=TaskPriority.HIGH,
            trigger="interval",
            trigger_config={"minutes": 5},
        )

        # Cleanup stale resources every 30 minutes
        await self.add_task(
            task_id="resource_cleanup",
            name="Resource Cleanup",
            function=self._cleanup_stale_resources,
            priority=TaskPriority.NORMAL,
            trigger="interval",
            trigger_config={"minutes": 30},
        )

        # Neural weights backup every 2 hours
        await self.add_task(
            task_id="neural_backup",
            name="Neural Weights Backup",
            function=self._backup_neural_weights,
            priority=TaskPriority.HIGH,
            trigger="interval",
            trigger_config={"hours": 2},
        )

        # System metrics collection every minute
        await self.add_task(
            task_id="metrics_collection",
            name="System Metrics Collection",
            function=self._collect_system_metrics,
            priority=TaskPriority.NORMAL,
            trigger="interval",
            trigger_config={"minutes": 1},
        )

        # Daily system optimization at 3 AM
        await self.add_task(
            task_id="daily_optimization",
            name="Daily System Optimization",
            function=self._daily_system_optimization,
            priority=TaskPriority.CRITICAL,
            trigger="cron",
            trigger_config={"hour": 3, "minute": 0},
        )

    async def add_task(
        self,
        task_id: str,
        name: str,
        function: Callable[[], Awaitable[None]],
        priority: TaskPriority,
        trigger: str,
        trigger_config: Dict[str, Any],
        max_retries: int = 3,
        timeout_seconds: int = 300,
    ) -> bool:
        """Add scheduled task"""
        try:
            task = ScheduledTask(
                task_id=task_id,
                name=name,
                function=function,
                priority=priority,
                trigger=trigger,
                trigger_config=trigger_config,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
            )

            # Start the task loop using asyncio
            if trigger == "interval":
                interval_seconds = self._calculate_interval_seconds(trigger_config)
                task_handler = asyncio.create_task(
                    self._run_interval_task(task, interval_seconds)
                )
                self._task_handlers[task_id] = task_handler

            elif trigger == "cron":
                task_handler = asyncio.create_task(
                    self._run_cron_task(task, trigger_config)
                )
                self._task_handlers[task_id] = task_handler

            self.tasks[task_id] = task

            logger.info(f"✅ Task '{name}' scheduled successfully")
            return True

        except Exception as e:
            logger.error(f"Error adding task {task_id}: {e}")
            return False

    def _calculate_interval_seconds(self, config: Dict[str, Any]) -> int:
        """Calculate interval in seconds from config"""
        seconds = config.get("seconds", 0)
        minutes = config.get("minutes", 0)
        hours = config.get("hours", 0)
        return seconds + (minutes * 60) + (hours * 3600)

    async def _run_interval_task(self, task: ScheduledTask, interval_seconds: int):
        """Run task at regular intervals"""
        while self.running and task.task_id in self.tasks:
            try:
                await asyncio.sleep(interval_seconds)
                if self.running and task.task_id in self.tasks:
                    await self._execute_task_safely(task)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in interval task {task.name}: {e}")

    async def _run_cron_task(self, task: ScheduledTask, cron_config: Dict[str, Any]):
        """Run task based on cron-like schedule"""
        while self.running and task.task_id in self.tasks:
            try:
                next_run_time = self._calculate_next_cron_run(cron_config)
                wait_seconds = (
                    next_run_time - datetime.now(timezone.utc)
                ).total_seconds()

                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)

                if self.running and task.task_id in self.tasks:
                    await self._execute_task_safely(task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cron task {task.name}: {e}")

    def _calculate_next_cron_run(self, config: Dict[str, Any]) -> datetime:
        """Calculate next run time for cron task (simplified)"""
        now = datetime.now(timezone.utc)
        hour = config.get("hour", now.hour)
        minute = config.get("minute", now.minute)

        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        # If time has passed today, schedule for tomorrow
        if next_run <= now:
            next_run += timedelta(days=1)

        return next_run

    async def _execute_task_safely(self, task: ScheduledTask) -> None:
        """Execute task with error handling and metrics"""
        task.status = TaskStatus.RUNNING
        task.last_run = datetime.now(timezone.utc)

        try:
            # Publish task start event
            await publish_event(
                SheìlyEventType.SYSTEM_HEALTH,
                {
                    "action": "task_started",
                    "task_id": task.task_id,
                    "task_name": task.name,
                },
            )

            # Execute with timeout
            await asyncio.wait_for(task.function(), timeout=task.timeout_seconds)

            task.status = TaskStatus.COMPLETED
            task.error_count = 0

            # Publish success event
            await publish_event(
                SheìlyEventType.SYSTEM_HEALTH,
                {
                    "action": "task_completed",
                    "task_id": task.task_id,
                    "task_name": task.name,
                },
            )

        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error_count += 1

            await publish_event(
                SheìlyEventType.PERFORMANCE_ALERT,
                {
                    "alert_type": "task_timeout",
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "timeout_seconds": task.timeout_seconds,
                },
            )

            logger.error(f"Task {task.name} timed out after {task.timeout_seconds}s")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_count += 1

            await publish_event(
                SheìlyEventType.PERFORMANCE_ALERT,
                {
                    "alert_type": "task_error",
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "error": str(e),
                    "error_count": task.error_count,
                },
            )

            logger.error(f"Task {task.name} failed: {e}")

            # Remove task if too many failures
            if task.error_count >= task.max_retries:
                await self.remove_task(task.task_id)

    # ========================================
    # DEFAULT MAINTENANCE TASKS
    # ========================================

    async def _monitor_agent_health(self) -> None:
        """Monitor health of all active agents"""
        try:
            # Get MCP agent manager
            from sheily_core.core.mcp.mcp_agent_manager import get_mcp_agent_manager

            agent_manager = await get_mcp_agent_manager()
            if not agent_manager:
                return

            # Check agent health
            unhealthy_agents = []
            for agent_id, agent in getattr(agent_manager, "active_agents", {}).items():
                try:
                    # Simple health check - can be enhanced
                    if not hasattr(agent, "status") or agent.status != "healthy":
                        unhealthy_agents.append(agent_id)
                except Exception:
                    unhealthy_agents.append(agent_id)

            if unhealthy_agents:
                await publish_event(
                    SheìlyEventType.PERFORMANCE_ALERT,
                    {
                        "alert_type": "unhealthy_agents",
                        "unhealthy_agents": unhealthy_agents,
                        "count": len(unhealthy_agents),
                    },
                )

                # Attempt to restart unhealthy agents
                for agent_id in unhealthy_agents:
                    try:
                        await agent_manager.restart_agent(agent_id)
                        logger.info(f"Restarted unhealthy agent: {agent_id}")
                    except Exception as e:
                        logger.error(f"Failed to restart agent {agent_id}: {e}")

        except Exception as e:
            logger.error(f"Error in agent health monitoring: {e}")

    async def _cleanup_stale_resources(self) -> None:
        """Cleanup stale resources and optimize memory"""
        try:
            import gc

            import psutil

            # Force garbage collection
            collected = gc.collect()

            # Get memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            await publish_event(
                SheìlyEventType.METRICS_UPDATE,
                {
                    "metric_type": "resource_cleanup",
                    "objects_collected": collected,
                    "memory_usage_mb": round(memory_mb, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Cleanup old event history
            if self.event_stream and hasattr(self.event_stream, "_event_history"):
                history_len = len(self.event_stream._event_history)
                if history_len > 1000:
                    # Keep only last 500 events
                    self.event_stream._event_history = self.event_stream._event_history[
                        -500:
                    ]
                    logger.info(
                        f"Cleaned up {history_len - 500} old events from history"
                    )

        except Exception as e:
            logger.error(f"Error in resource cleanup: {e}")

    async def _backup_neural_weights(self) -> None:
        """Backup neural network weights"""
        try:
            import shutil
            from pathlib import Path

            # Neural weights backup logic
            weights_dir = Path("real_neural_weights")
            if weights_dir.exists():
                backup_dir = Path(
                    f"neural_backups/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                backup_dir.mkdir(parents=True, exist_ok=True)

                # Copy weights
                for weights_file in weights_dir.glob("*.pt"):
                    shutil.copy2(weights_file, backup_dir)

                await publish_event(
                    SheìlyEventType.NEURAL_TRAINING_STATUS,
                    {
                        "action": "weights_backup_completed",
                        "backup_path": str(backup_dir),
                        "files_backed_up": len(list(backup_dir.glob("*.pt"))),
                    },
                )

                logger.info(f"Neural weights backed up to {backup_dir}")

        except Exception as e:
            logger.error(f"Error in neural weights backup: {e}")

    async def _collect_system_metrics(self) -> None:
        """Collect comprehensive system metrics"""
        try:
            import psutil

            # CPU and Memory metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / 1024**3, 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / 1024**3, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await publish_event(
                SheìlyEventType.METRICS_UPDATE,
                {"metric_type": "system_performance", **metrics},
            )

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    async def _daily_system_optimization(self) -> None:
        """Daily comprehensive system optimization"""
        try:
            logger.info("🔧 Starting daily system optimization...")

            # Run all maintenance tasks
            await self._cleanup_stale_resources()
            await self._monitor_agent_health()

            # Optimize database if exists
            await self._optimize_database()

            # Cleanup old logs
            await self._cleanup_old_logs()

            await publish_event(
                SheìlyEventType.SYSTEM_HEALTH,
                {
                    "action": "daily_optimization_completed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            logger.info("✅ Daily system optimization completed")

        except Exception as e:
            logger.error(f"Error in daily optimization: {e}")

    async def _optimize_database(self) -> None:
        """Optimize database performance"""
        try:
            # Database optimization logic would go here
            # For now, just publish that we're doing maintenance
            await publish_event(
                SheìlyEventType.SYSTEM_HEALTH,
                {"action": "database_optimization", "status": "simulated"},
            )
        except Exception as e:
            logger.error(f"Error in database optimization: {e}")

    async def _cleanup_old_logs(self) -> None:
        """Cleanup old log files"""
        try:
            from pathlib import Path

            log_dirs = [Path("logs"), Path("security_logs_*")]
            cutoff_date = datetime.now() - timedelta(days=7)

            files_cleaned = 0
            for log_pattern in log_dirs:
                if "*" in str(log_pattern):
                    for log_dir in Path(".").glob(str(log_pattern)):
                        if log_dir.is_dir():
                            files_cleaned += await self._clean_directory(
                                log_dir, cutoff_date
                            )
                else:
                    if log_pattern.exists():
                        files_cleaned += await self._clean_directory(
                            log_pattern, cutoff_date
                        )

            if files_cleaned > 0:
                await publish_event(
                    SheìlyEventType.SYSTEM_HEALTH,
                    {"action": "log_cleanup", "files_cleaned": files_cleaned},
                )

        except Exception as e:
            logger.error(f"Error in log cleanup: {e}")

    async def _clean_directory(self, directory: Path, cutoff_date: datetime) -> int:
        """Clean files older than cutoff date"""
        files_cleaned = 0
        try:
            for file_path in directory.rglob("*.log"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    files_cleaned += 1
        except Exception as e:
            logger.error(f"Error cleaning directory {directory}: {e}")
        return files_cleaned

    # ========================================
    # TASK MANAGEMENT
    # ========================================

    async def remove_task(self, task_id: str) -> bool:
        """Remove scheduled task"""
        try:
            if task_id in self.tasks:
                self.scheduler.remove_job(task_id)
                del self.tasks[task_id]
                logger.info(f"Task {task_id} removed")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing task {task_id}: {e}")
            return False

    async def get_task_status(self, task_id: str) -> Optional[ScheduledTask]:
        """Get task status"""
        return self.tasks.get(task_id)

    async def get_all_tasks(self) -> Dict[str, ScheduledTask]:
        """Get all tasks"""
        return self.tasks.copy()

    async def shutdown(self) -> None:
        """Shutdown scheduler"""
        try:
            self.scheduler.shutdown()
            self.running = False

            await publish_event(
                SheìlyEventType.SYSTEM_HEALTH, {"status": "scheduler_shutdown"}
            )

            logger.info("🛑 Sheily Scheduler shut down")

        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}")


# ========================================
# GLOBAL INSTANCE
# ========================================

_scheduler: Optional[SheìlyScheduler] = None


async def get_scheduler() -> SheìlyScheduler:
    """Get global scheduler instance"""
    global _scheduler
    if _scheduler is None:
        _scheduler = SheìlyScheduler()
        await _scheduler.initialize()
    return _scheduler


async def initialize_scheduler_system() -> SheìlyScheduler:
    """Initialize the complete scheduler system"""
    scheduler = await get_scheduler()
    logger.info("✅ Sheily Enterprise Scheduler System initialized")
    return scheduler


# Alias for compatibility
BackgroundSchedulerSystem = SheìlyScheduler


__all__ = [
    "SheìlyScheduler",
    "ScheduledTask",
    "TaskPriority",
    "TaskStatus",
    "get_scheduler",
    "initialize_scheduler_system",
]
