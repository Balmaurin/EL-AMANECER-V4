"""
Agent Orchestration System
==========================
Manages the 4 Specialized Agents (Hive Mind) and their access to Tools.
Agents:
1. Core Agent (System Architecture & Code)
2. Business Agent (Strategy & Data)
3. Infrastructure Agent (DevOps & Security)
4. Meta-Cognition Agent (Reflection & Improvement)
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel

# Import Tool System
from sheily_core.tools.tool_creation import ToolCreationSystem, ToolCreationRequest, ToolType

# Configure logging
logger = logging.getLogger("agent_orchestrator")

class AgentType(Enum):
    CORE = "core_agent"
    BUSINESS = "business_agent"
    INFRASTRUCTURE = "infrastructure_agent"
    META_COGNITION = "meta_cognition_agent"

class AgentTask(BaseModel):
    task_id: str
    description: str
    agent_type: Optional[str] = None
    priority: int = 1
    context: Dict[str, Any] = {}

class AgentOrchestrator:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentOrchestrator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.tool_system = ToolCreationSystem()
        self.agents = {
            AgentType.CORE: self._init_core_agent(),
            AgentType.BUSINESS: self._init_business_agent(),
            AgentType.INFRASTRUCTURE: self._init_infra_agent(),
            AgentType.META_COGNITION: self._init_meta_agent()
        }
        self._initialized = True
        logger.info("🐝 Agent Hive Mind Initialized")

    def _init_core_agent(self):
        return {"name": "Core Architect", "capabilities": ["code", "architecture", "refactor"]}

    def _init_business_agent(self):
        return {"name": "Business Strategist", "capabilities": ["data_analysis", "planning", "reporting"]}

    def _init_infra_agent(self):
        return {"name": "Infra Guardian", "capabilities": ["security", "deployment", "monitoring"]}

    def _init_meta_agent(self):
        return {"name": "Meta Thinker", "capabilities": ["reflection", "optimization", "learning"]}

    async def dispatch_task(self, task: AgentTask) -> Dict[str, Any]:
        """
        Route task to the appropriate agent.
        If agent needs a tool, it requests it from ToolCreationSystem.
        """
        agent_type = self._determine_agent(task)
        agent = self.agents.get(agent_type)
        
        logger.info(f"🤖 Dispatching task '{task.description}' to {agent['name']}")
        
        # 1. Check if tool is needed
        tool_needed = self._analyze_tool_needs(task)
        
        tool_result = None
        if tool_needed:
            logger.info(f"🛠️ Agent {agent['name']} requesting tool: {tool_needed}")
            tool_result = await self._request_tool_creation(tool_needed, agent_type)
            
        # 2. Execute Task (Simulated for now)
        # In full version, this calls the actual LLM Agent implementation
        result = {
            "status": "completed",
            "agent": agent['name'],
            "output": f"Processed '{task.description}' using capabilities: {agent['capabilities']}",
            "tool_used": tool_result
        }
        
        return result

    def _determine_agent(self, task: AgentTask) -> AgentType:
        """Simple heuristic to route tasks"""
        desc = task.description.lower()
        if any(w in desc for w in ["code", "python", "bug", "api", "function"]):
            return AgentType.CORE
        elif any(w in desc for w in ["server", "deploy", "security", "db", "database"]):
            return AgentType.INFRASTRUCTURE
        elif any(w in desc for w in ["plan", "strategy", "cost", "user"]):
            return AgentType.BUSINESS
        else:
            return AgentType.META_COGNITION

    def _analyze_tool_needs(self, task: AgentTask) -> Optional[str]:
        """Check if task requires a new tool"""
        if "create tool" in task.description.lower() or "generate script" in task.description.lower():
            return "dynamic_tool_request"
        return None

    async def _request_tool_creation(self, tool_desc: str, requester: AgentType) -> Dict[str, Any]:
        """Interact with ToolCreationSystem"""
        # Create a request object
        req = ToolCreationRequest(
            requester_agent=requester.value,
            tool_name=f"tool_{requester.value}_{asyncio.get_event_loop().time()}",
            description=tool_desc,
            tool_type=ToolType.PYTHON_FUNCTION
        )
        
        # In a real scenario, we would await the generation
        # code = await self.tool_system.generate_tool(req)
        return {"tool_status": "requested", "request_id": req.request_id}

# Global Accessor
_orchestrator = None

def get_agent_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
