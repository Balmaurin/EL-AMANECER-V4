"""
Agent Orchestrator - Core Coordination System
===========================================

Orchestrates specialized AI agents across enterprise domains.
Provides intelligent agent selection, task distribution, and coordination.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from apps.backend.src.config.settings import settings


class AgentType(Enum):
    """Types of specialized agents"""

    FINANCE = "finance"
    SECURITY = "security"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    ENGINEERING = "engineering"
    BUSINESS = "business"
    CREATIVE = "creative"
    SPECIALIZED = "specialized"


class AgentCapability(Enum):
    """Agent capabilities and specializations"""

    ANALYTICS = "analytics"
    OPTIMIZATION = "optimization"
    PREDICTION = "prediction"
    DIAGNOSIS = "diagnosis"
    DESIGN = "design"
    COMPLIANCE = "compliance"
    MANAGEMENT = "management"
    STRATEGY = "strategy"
    SECURITY = "security"
    INNOVATION = "innovation"


@dataclass
class AgentDefinition:
    """Definition of a specialized agent"""

    id: str
    name: str
    domain: AgentType
    capabilities: List[AgentCapability]
    priority: int = 1  # 1-10 scale
    load_capacity: int = 5  # Concurrent tasks
    specialization_score: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentInstance:
    """Runtime instance of an agent"""

    definition: AgentDefinition
    status: str = "idle"  # idle, busy, offline, error
    active_tasks: int = 0
    last_activity: Optional[datetime] = None
    performance_score: float = 0.0
    error_count: int = 0

    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = datetime.utcnow()


class Task:
    """Task definition for agent execution"""

    def __init__(
        self,
        task_id: str,
        title: str,
        domain: AgentType,
        requirements: Dict[str, Any],
        context: Optional[Dict] = None,
    ):
        self.task_id = task_id
        self.title = title
        self.domain = domain
        self.requirements = requirements
        self.context = context or {}
        self.created_at = datetime.utcnow()
        self.assigned_to: Optional[AgentInstance] = None
        self.status = "pending"  # pending, assigned, running, completed, failed
        self.priority = requirements.get("priority", 5)  # 1-10
        self.estimated_complexity = requirements.get("complexity", 1)  # 1-5
        self.tags: Set[str] = set()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "domain": self.domain.value,
            "status": self.status,
            "priority": self.priority,
            "complexity": self.estimated_complexity,
            "assigned_to": self.assigned_to.definition.id if self.assigned_to else None,
            "created_at": self.created_at.isoformat(),
            "tags": list(self.tags),
            "requirements": self.requirements,
            "context": self.context,
        }


class AgentOrchestrator:
    """Main orchestration system for specialized agents"""

    def __init__(self):
        # Specialized agents across domains
        self.agents: Dict[str, AgentDefinition] = self._initialize_agents()
        self.agent_instances: Dict[str, AgentInstance] = {}

        # Task management
        self.pending_tasks: List[Task] = []
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}

        # Coordination metrics
        self.orchestration_metrics = {
            "total_tasks_processed": 0,
            "success_rate": 0.0,
            "average_response_time": 0.0,
            "agent_utilization": {},
            "domain_performance": {},
        }

        # Initialize agent instances
        self._load_agent_instances()

        # Start orchestration loop
        asyncio.create_task(self.orchestration_loop())

    def _initialize_agents(self) -> Dict[str, AgentDefinition]:
        """Initialize the specialized agents across domains"""
        agents = {}

        # Define core agents
        core_agents = [
            (
                "finance/financial_analyst",
                "Financial Analyst",
                AgentType.FINANCE,
                [AgentCapability.ANALYTICS, AgentCapability.PREDICTION],
                "Analyzes financial data and provides insights",
            ),
            (
                "security/security_expert",
                "Security Expert",
                AgentType.SECURITY,
                [AgentCapability.SECURITY, AgentCapability.COMPLIANCE],
                "Ensures system security and compliance",
            ),
            (
                "healthcare/medical_advisor",
                "Medical Advisor",
                AgentType.HEALTHCARE,
                [AgentCapability.DIAGNOSIS, AgentCapability.ANALYTICS],
                "Provides medical insights and analysis",
            ),
            (
                "business/general_assistant",
                "General Assistant",
                AgentType.BUSINESS,
                [AgentCapability.STRATEGY, AgentCapability.MANAGEMENT],
                "Assists with general business tasks",
            ),
        ]

        for i, (agent_id, name, domain, capabilities, description) in enumerate(core_agents):
            agents[agent_id] = AgentDefinition(
                id=agent_id,
                name=name,
                domain=domain,
                capabilities=capabilities,
                priority=i + 1,
                specialization_score={"general": 0.9},
                description=description,
                metadata={"type": "core_agent"},
            )

        return agents














    def _load_agent_instances(self):
        """Initialize agent instances"""
        for agent_id, definition in self.agents.items():
            self.agent_instances[agent_id] = AgentInstance(definition)

    async def orchestration_loop(self):
        """Main orchestration loop for task assignment and monitoring"""
        while True:
            try:
                await self._process_pending_tasks()
                await self._monitor_running_tasks()
                await self._cleanup_completed_tasks()
                self._update_metrics()

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                print(f"Orchestration loop error: {e}")
                await asyncio.sleep(5)

    async def _process_pending_tasks(self):
        """Process and assign pending tasks to appropriate agents with robust error handling"""
        try:
            if not hasattr(self, "pending_tasks") or not self.pending_tasks:
                return

            # Sort tasks by priority (highest first) with validation
            try:
                sorted_tasks = sorted(
                    self.pending_tasks,
                    key=lambda t: getattr(t, "priority", 5),
                    reverse=True,
                )
            except (AttributeError, TypeError) as e:
                print(f"Error sorting tasks by priority: {e}")
                sorted_tasks = self.pending_tasks  # Continue without sorting

            for task in sorted_tasks:
                try:
                    best_agent = self._select_optimal_agent(task)
                    if best_agent:
                        await self._assign_task_to_agent(task, best_agent)
                        # Safer removal using task object comparison
                        if task in self.pending_tasks:
                            self.pending_tasks.remove(task)
                except Exception as e:
                    print(
                        f"Error processing task {getattr(task, 'task_id', 'unknown')}: {e}"
                    )
                    # Continue with next task instead of breaking
                    continue

        except Exception as e:
            print(f"Critical error in _process_pending_tasks: {e}")
            # Graceful degradation - continue operation

    def _is_valid_candidate(self, agent: AgentInstance, target_domain: AgentType) -> bool:
        """Check if agent is valid for the domain."""
        try:
            if not agent or not hasattr(agent, "status") or agent.status != "idle":
                return False
            
            definition = getattr(agent, "definition", None)
            if not definition or getattr(definition, "domain", None) != target_domain:
                return False
                
            if getattr(agent, "active_tasks", 0) >= getattr(definition, "load_capacity", 0):
                return False
                
            return True
        except Exception:
            return False

    def _find_candidates(self, task: Task) -> List[AgentInstance]:
        """Find potential agent candidates for a task."""
        candidates = []
        if not hasattr(self, "agent_instances") or not isinstance(self.agent_instances, dict):
            return []
            
        # Check same domain
        for agent in self.agent_instances.values():
            if self._is_valid_candidate(agent, task.domain):
                candidates.append(agent)
                
        # If no candidates, check specialized
        if not candidates:
            for agent in self.agent_instances.values():
                if self._is_valid_candidate(agent, AgentType.SPECIALIZED):
                    candidates.append(agent)
                    
        return candidates

    def _select_optimal_agent(self, task: Task) -> Optional[AgentInstance]:
        """Select the best available agent for a given task with robust error handling"""
        try:
            if not task or not hasattr(task, "domain"):
                return None

            candidates = self._find_candidates(task)
            scored_candidates = []
            
            for agent in candidates:
                try:
                    score = self._calculate_agent_suitability(agent, task)
                    scored_candidates.append((agent, score))
                except Exception:
                    continue
            
            if scored_candidates:
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                return scored_candidates[0][0]
                
            return None
        except Exception as e:
            print(f"Critical error in _select_optimal_agent: {e}")
            return None

    def _calculate_agent_suitability(self, agent: AgentInstance, task: Task) -> float:
        """Calculate how suitable an agent is for a task"""
        score = 0.0

        # Base domain match score
        if agent.definition.domain == task.domain:
            score += 50
        elif agent.definition.domain == AgentType.SPECIALIZED:
            score += 25

        # Capability match score
        relevant_capabilities = task.requirements.get("capabilities", [])
        for capability in relevant_capabilities:
            if capability in [c.value for c in agent.definition.capabilities]:
                score += 20

        # Performance score
        score += agent.performance_score * 10

        # Load balance score (prefer less loaded agents)
        max_capacity = agent.definition.load_capacity
        current_load = agent.active_tasks
        load_score = (max_capacity - current_load) / max_capacity * 15
        score += load_score

        # Specialization score
        task_keywords = task.requirements.get("keywords", [])
        for keyword in task_keywords:
            domain_specs = agent.definition.specialization_score
            matches = [
                score for key, score in domain_specs.items() if keyword.lower() in key
            ]
            if matches:
                score += max(matches) * 5

        return score

    async def _assign_task_to_agent(self, task: Task, agent: AgentInstance):
        """Assign a task to an agent and simulate execution"""
        task.assigned_to = agent
        task.status = "assigned"
        self.running_tasks[task.task_id] = task

        # Update agent status
        agent.status = "busy"
        agent.active_tasks += 1

        # Simulate task execution (in production, this would actually execute)
        asyncio.create_task(self._execute_task_simulation(task, agent))

    async def _execute_task_simulation(self, task: Task, agent: AgentInstance):
        """Simulate task execution"""
        # Simulate processing time based on complexity
        processing_time = task.estimated_complexity * 2 + 1  # 1-11 seconds
        await asyncio.sleep(processing_time)

        # Mark task as completed (90% success rate simulation)
        if task.task_id.endswith("1") or task.task_id.endswith(
            "2"
        ):  # 20% failure simulation
            task.status = "failed"
            self.failed_tasks[task.task_id] = task
            agent.error_count += 1
        else:
            task.status = "completed"
            self.completed_tasks[task.task_id] = task

        # Update agent status
        agent.status = "idle"
        agent.active_tasks -= 1
        agent.last_activity = datetime.utcnow()
        agent.performance_score = min(
            100, agent.performance_score + 0.1
        )  # Small performance boost

    async def _monitor_running_tasks(self):
        """Monitor running tasks for timeouts, etc."""
        # In production, this would check for stuck tasks and handle failures
        pass

    async def _cleanup_completed_tasks(self):
        """Clean up old completed tasks"""
        current_time = datetime.utcnow()

        # Keep only last 1000 completed tasks
        if len(self.completed_tasks) > 1000:
            to_remove = len(self.completed_tasks) - 1000
            sorted_tasks = sorted(
                self.completed_tasks.items(),
                key=lambda x: x[1].created_at,
                reverse=True,
            )
            for _, task in sorted_tasks[-to_remove:]:
                del self.completed_tasks[task.task_id]

        # Clean up failed tasks older than 7 days
        failed_to_remove = []
        for task_id, task in self.failed_tasks.items():
            if (current_time - task.created_at).days > 7:
                failed_to_remove.append(task_id)

        for task_id in failed_to_remove:
            del self.failed_tasks[task_id]

    def _update_metrics(self):
        """Update orchestration metrics"""
        total_tasks = len(self.completed_tasks) + len(self.failed_tasks)
        if total_tasks > 0:
            success_rate = len(self.completed_tasks) / total_tasks
            self.orchestration_metrics["success_rate"] = success_rate

            # Calculate average response time (simplified)
            avg_time = (
                sum(
                    [
                        (t.created_at - t.created_at).seconds
                        for t in self.completed_tasks.values()
                    ]
                )
                / len(self.completed_tasks)
                if self.completed_tasks
                else 0
            )
            self.orchestration_metrics["average_response_time"] = avg_time

        # Update agent utilization
        utilization = {}
        for agent_id, agent in self.agent_instances.items():
            domain = agent.definition.domain.value
            if domain not in utilization:
                utilization[domain] = {"busy": 0, "total": 0}

            utilization[domain]["total"] += 1
            if agent.status == "busy":
                utilization[domain]["busy"] += 1

        self.orchestration_metrics["agent_utilization"] = utilization

    def submit_task(
        self,
        title: str,
        domain: AgentType,
        requirements: Dict[str, Any],
        context: Optional[Dict] = None,
    ) -> str:
        """Submit a new task for agent processing with robust error handling"""
        try:
            # Validate inputs
            if not title or not isinstance(title, str):
                raise ValueError("Task title must be a non-empty string")

            if not isinstance(domain, AgentType):
                raise ValueError("Domain must be a valid AgentType")

            if not requirements or not isinstance(requirements, dict):
                raise ValueError("Requirements must be a non-empty dictionary")

            task_id = str(uuid.uuid4())

            # Create task safely
            task = Task(task_id, title, domain, requirements, context)

            # Extract tags safely
            try:
                task.tags.update(requirements.get("tags", []))
                task.tags.add(domain.value)
            except (AttributeError, TypeError):
                # Continue without tags if there's an issue
                pass

            # Append to pending tasks safely
            if hasattr(self, "pending_tasks") and isinstance(self.pending_tasks, list):
                self.pending_tasks.append(task)

            # Update metrics safely
            try:
                if hasattr(self, "orchestration_metrics") and isinstance(
                    self.orchestration_metrics, dict
                ):
                    self.orchestration_metrics["total_tasks_processed"] = (
                        self.orchestration_metrics.get("total_tasks_processed", 0) + 1
                    )
            except (AttributeError, KeyError):
                pass

            return task_id

        except Exception as e:
            print(f"Error submitting task '{title}': {e}")
            # Return a fallback task ID to maintain operation
            return f"error_{str(uuid.uuid4())[:8]}"

    def _find_task(self, task_id: str) -> Optional[Task]:
        """Search for task in all collections."""
        collections = [self.running_tasks, self.completed_tasks, self.failed_tasks]
        
        for collection in collections:
            if isinstance(collection, dict) and task_id in collection:
                return collection[task_id]
                
        if isinstance(self.pending_tasks, list):
            for task in self.pending_tasks:
                if hasattr(task, "task_id") and task.task_id == task_id:
                    return task
        return None

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task with robust error handling"""
        try:
            if not task_id or not isinstance(task_id, str):
                return None
                
            task = self._find_task(task_id)
            return task.to_dict() if task else None
        except Exception as e:
            print(f"Critical error getting task status for {task_id}: {e}")
            return None

    def _format_agent_status(self, agent: AgentInstance) -> Dict[str, Any]:
        """Format agent status safely."""
        try:
            definition = getattr(agent, "definition", None)
            last_activity = getattr(agent, "last_activity", None)
            
            last_activity_str = "unknown"
            if last_activity:
                try:
                    last_activity_str = last_activity.isoformat()
                except AttributeError:
                    last_activity_str = "invalid_datetime"

            return {
                "agent_id": getattr(definition, "id", "unknown"),
                "name": getattr(definition, "name", "Unknown Agent"),
                "domain": getattr(getattr(definition, "domain", None), "value", "unknown"),
                "status": getattr(agent, "status", "unknown"),
                "active_tasks": getattr(agent, "active_tasks", 0),
                "performance_score": getattr(agent, "performance_score", 0.0),
                "last_activity": last_activity_str
            }
        except Exception:
            return {}

    def get_agent_status(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of agents with robust error handling and nullable safety"""
        try:
            if agent_id and isinstance(agent_id, str):
                agent = self.agent_instances.get(agent_id)
                return self._format_agent_status(agent) if agent else {}
            
            return {
                aid: self._format_agent_status(agent) 
                for aid, agent in self.agent_instances.items()
            }
        except Exception as e:
            print(f"Critical error getting agent status: {e}")
            return {}

    def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get orchestration performance metrics"""
        return self.orchestration_metrics.copy()


# Global agent orchestrator instance
agent_orchestrator = AgentOrchestrator()

__all__ = [
    "agent_orchestrator",
    "AgentOrchestrator",
    "AgentType",
    "AgentCapability",
    "AgentDefinition",
    "AgentInstance",
    "Task",
]
