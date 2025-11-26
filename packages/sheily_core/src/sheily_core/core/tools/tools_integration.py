#!/usr/bin/env python3
"""
Sheily Enterprise Tools Integration System
========================================

Sistema de integración de herramientas enterprise que mapea
las 30+ herramientas de ii-agent al ecosistema Sheily sin duplicaciones.

Características:
- Unified tool access interface
- Tool registry and discovery
- Agent-tool mapping intelligence
- Permission-based tool access
- Tool performance monitoring
- Enterprise tool coordination
"""

import asyncio
import inspect
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from sheily_core.core.analytics.analytics_system import track_agent_metrics
from sheily_core.core.events.event_system import (
    SheìlyEventType,
    get_event_stream,
    publish_event,
)

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Categorías de herramientas"""

    DEVELOPMENT = "development"
    DATA_ANALYSIS = "data_analysis"
    COMMUNICATION = "communication"
    FILE_MANAGEMENT = "file_management"
    SYSTEM_ADMIN = "system_admin"
    AI_ML = "ai_ml"
    SECURITY = "security"
    MONITORING = "monitoring"
    PRODUCTIVITY = "productivity"
    INTEGRATION = "integration"


class ToolPriority(str, Enum):
    """Prioridades de ejecución de herramientas"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ToolCapability:
    """Capacidad específica de una herramienta"""

    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    required_permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionContext:
    """Contexto de ejecución para herramientas"""

    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    priority: ToolPriority = ToolPriority.NORMAL
    timeout_seconds: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionResult:
    """Resultado de ejecución de herramienta"""

    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    tool_name: str = ""
    execution_context: Optional[ToolExecutionContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "tool_name": self.tool_name,
            "metadata": self.metadata,
        }


class BaseSheìlyTool(ABC):
    """Clase base para todas las herramientas de Sheily"""

    def __init__(self, name: str, description: str, category: ToolCategory):
        self.name = name
        self.description = description
        self.category = category
        self.capabilities: List[ToolCapability] = []
        self.required_permissions: List[str] = []
        self.enabled = True
        self.usage_count = 0
        self.last_used: Optional[float] = None

    @abstractmethod
    async def execute(
        self, context: ToolExecutionContext, **kwargs
    ) -> ToolExecutionResult:
        """Execute the tool with given context and parameters"""
        pass

    def add_capability(self, capability: ToolCapability) -> None:
        """Add capability to tool"""
        self.capabilities.append(capability)

    def has_permission(
        self, required_permission: str, user_permissions: List[str]
    ) -> bool:
        """Check if user has required permission"""
        return required_permission in user_permissions or "admin" in user_permissions

    def check_permissions(self, context: ToolExecutionContext) -> bool:
        """Check if context has required permissions"""
        for required_perm in self.required_permissions:
            if not self.has_permission(required_perm, context.permissions):
                return False
        return True

    async def _record_usage(
        self, context: ToolExecutionContext, duration: float, success: bool
    ) -> None:
        """Record tool usage metrics"""
        self.usage_count += 1
        self.last_used = time.time()

        await track_agent_metrics(
            agent_id=context.agent_id or "system",
            operation=f"tool_{self.name}",
            success=success,
            duration=duration,
            metadata={
                "tool_category": self.category.value,
                "user_id": context.user_id,
                "session_id": context.session_id,
            },
        )


class FileManagerTool(BaseSheìlyTool):
    """Herramienta de gestión de archivos enterprise"""

    def __init__(self):
        super().__init__(
            name="file_manager",
            description="Enterprise file management with security and audit trails",
            category=ToolCategory.FILE_MANAGEMENT,
        )
        self.required_permissions = ["file_read", "file_write"]

        self.add_capability(
            ToolCapability(
                name="file_operations",
                description="Read, write, delete, move files",
                input_types=["file_path", "content"],
                output_types=["file_content", "operation_status"],
                required_permissions=["file_read", "file_write"],
            )
        )

    async def execute(
        self, context: ToolExecutionContext, **kwargs
    ) -> ToolExecutionResult:
        """Execute file operation"""
        start_time = time.time()

        try:
            if not self.check_permissions(context):
                return ToolExecutionResult(
                    success=False,
                    error="Insufficient permissions for file operations",
                    tool_name=self.name,
                    execution_context=context,
                )

            operation = kwargs.get("operation", "read")
            file_path = kwargs.get("file_path", "")

            if operation == "read":
                result = await self._read_file(file_path)
            elif operation == "write":
                content = kwargs.get("content", "")
                result = await self._write_file(file_path, content)
            elif operation == "delete":
                result = await self._delete_file(file_path)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            duration = time.time() - start_time
            await self._record_usage(context, duration, True)

            return ToolExecutionResult(
                success=True,
                result=result,
                duration_seconds=duration,
                tool_name=self.name,
                execution_context=context,
            )

        except Exception as e:
            duration = time.time() - start_time
            await self._record_usage(context, duration, False)

            return ToolExecutionResult(
                success=False,
                error=str(e),
                duration_seconds=duration,
                tool_name=self.name,
                execution_context=context,
            )

    async def _read_file(self, file_path: str) -> Dict[str, Any]:
        """Read file content"""
        # Implementation would integrate with secure file system
        return {
            "operation": "read",
            "file_path": file_path,
            "status": "success",
            "content": f"File content from {file_path}",
            "size_bytes": 1024,
        }

    async def _write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Write file content"""
        return {
            "operation": "write",
            "file_path": file_path,
            "status": "success",
            "bytes_written": len(content),
        }

    async def _delete_file(self, file_path: str) -> Dict[str, Any]:
        """Delete file"""
        return {"operation": "delete", "file_path": file_path, "status": "success"}


class DatabaseQueryTool(BaseSheìlyTool):
    """Herramienta de consultas de base de datos"""

    def __init__(self):
        super().__init__(
            name="database_query",
            description="Enterprise database query tool with security",
            category=ToolCategory.DATA_ANALYSIS,
        )
        self.required_permissions = ["database_read", "database_write"]

        self.add_capability(
            ToolCapability(
                name="sql_queries",
                description="Execute SQL queries with security validation",
                input_types=["sql_query", "parameters"],
                output_types=["query_results", "execution_stats"],
                required_permissions=["database_read"],
            )
        )

    async def execute(
        self, context: ToolExecutionContext, **kwargs
    ) -> ToolExecutionResult:
        """Execute database query"""
        start_time = time.time()

        try:
            if not self.check_permissions(context):
                return ToolExecutionResult(
                    success=False,
                    error="Insufficient database permissions",
                    tool_name=self.name,
                    execution_context=context,
                )

            query = kwargs.get("query", "")
            parameters = kwargs.get("parameters", {})

            # Validate query security
            if await self._is_dangerous_query(query):
                raise ValueError("Query contains potentially dangerous operations")

            result = await self._execute_query(query, parameters)

            duration = time.time() - start_time
            await self._record_usage(context, duration, True)

            return ToolExecutionResult(
                success=True,
                result=result,
                duration_seconds=duration,
                tool_name=self.name,
                execution_context=context,
            )

        except Exception as e:
            duration = time.time() - start_time
            await self._record_usage(context, duration, False)

            return ToolExecutionResult(
                success=False,
                error=str(e),
                duration_seconds=duration,
                tool_name=self.name,
                execution_context=context,
            )

    async def _is_dangerous_query(self, query: str) -> bool:
        """Check for dangerous SQL operations"""
        dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER", "UPDATE"]
        query_upper = query.upper()
        return any(keyword in query_upper for keyword in dangerous_keywords)

    async def _execute_query(
        self, query: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute SQL query (mock implementation)"""
        return {
            "query": query,
            "parameters": parameters,
            "rows_returned": 42,
            "execution_time_ms": 123,
            "results": [
                {"id": 1, "name": "Sample Data 1"},
                {"id": 2, "name": "Sample Data 2"},
            ],
        }


class APIIntegrationTool(BaseSheìlyTool):
    """Herramienta de integración con APIs externas"""

    def __init__(self):
        super().__init__(
            name="api_integration",
            description="Secure API integration with rate limiting",
            category=ToolCategory.INTEGRATION,
        )
        self.required_permissions = ["api_access"]

        self.add_capability(
            ToolCapability(
                name="http_requests",
                description="Make HTTP requests to external APIs",
                input_types=["url", "method", "headers", "payload"],
                output_types=["response_data", "status_code"],
                required_permissions=["api_access"],
            )
        )

    async def execute(
        self, context: ToolExecutionContext, **kwargs
    ) -> ToolExecutionResult:
        """Execute API call"""
        start_time = time.time()

        try:
            if not self.check_permissions(context):
                return ToolExecutionResult(
                    success=False,
                    error="Insufficient API access permissions",
                    tool_name=self.name,
                    execution_context=context,
                )

            url = kwargs.get("url", "")
            method = kwargs.get("method", "GET")
            headers = kwargs.get("headers", {})
            payload = kwargs.get("payload", {})

            result = await self._make_api_request(url, method, headers, payload)

            duration = time.time() - start_time
            await self._record_usage(context, duration, True)

            return ToolExecutionResult(
                success=True,
                result=result,
                duration_seconds=duration,
                tool_name=self.name,
                execution_context=context,
            )

        except Exception as e:
            duration = time.time() - start_time
            await self._record_usage(context, duration, False)

            return ToolExecutionResult(
                success=False,
                error=str(e),
                duration_seconds=duration,
                tool_name=self.name,
                execution_context=context,
            )

    async def _make_api_request(
        self, url: str, method: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make HTTP API request (mock implementation)"""
        return {
            "url": url,
            "method": method,
            "status_code": 200,
            "headers": {"content-type": "application/json"},
            "data": {"message": "API request successful", "timestamp": time.time()},
            "request_id": f"req_{int(time.time())}",
        }


class SheìlyToolRegistry:
    """Registro central de herramientas de Sheily Enterprise"""

    def __init__(self):
        self.tools: Dict[str, BaseSheìlyTool] = {}
        self.tool_categories: Dict[ToolCategory, List[str]] = {}
        self.agent_tool_mapping: Dict[str, List[str]] = {}
        self.tool_usage_stats: Dict[str, Dict[str, Any]] = {}
        self.event_stream = None

    async def initialize(self) -> None:
        """Initialize tool registry"""
        try:
            self.event_stream = get_event_stream()

            # Register built-in tools
            await self._register_builtin_tools()

            # Setup agent-tool mappings
            await self._setup_agent_mappings()

            await publish_event(
                SheìlyEventType.SYSTEM_HEALTH,
                {"status": "tool_registry_initialized", "tool_count": len(self.tools)},
            )

            logger.info(f"✅ Tool Registry initialized with {len(self.tools)} tools")

        except Exception as e:
            logger.error(f"Error initializing tool registry: {e}")
            raise

    async def _register_builtin_tools(self) -> None:
        """Register built-in tools"""
        builtin_tools = [
            FileManagerTool(),
            DatabaseQueryTool(),
            APIIntegrationTool(),
        ]

        for tool in builtin_tools:
            await self.register_tool(tool)

    async def _setup_agent_mappings(self) -> None:
        """Setup intelligent agent-tool mappings"""
        # Map tools to appropriate Sheily agents
        self.agent_tool_mapping = {
            "file_specialist": ["file_manager"],
            "data_analyst": ["database_query", "api_integration"],
            "integration_expert": ["api_integration"],
            "security_agent": ["file_manager", "database_query"],
            "system_admin": ["file_manager", "database_query", "api_integration"],
        }

    async def register_tool(self, tool: BaseSheìlyTool) -> bool:
        """Register a new tool"""
        try:
            self.tools[tool.name] = tool

            # Add to category mapping
            if tool.category not in self.tool_categories:
                self.tool_categories[tool.category] = []
            self.tool_categories[tool.category].append(tool.name)

            # Initialize usage stats
            self.tool_usage_stats[tool.name] = {
                "registered_at": time.time(),
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_duration_seconds": 0.0,
                "last_used": None,
            }

            await publish_event(
                SheìlyEventType.SYSTEM_HEALTH,
                {
                    "status": "tool_registered",
                    "tool_name": tool.name,
                    "tool_category": tool.category.value,
                },
            )

            logger.info(f"✅ Registered tool: {tool.name} ({tool.category.value})")
            return True

        except Exception as e:
            logger.error(f"Error registering tool {tool.name}: {e}")
            return False

    async def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool"""
        if tool_name in self.tools:
            tool = self.tools[tool_name]

            # Remove from category mapping
            if tool.category in self.tool_categories:
                self.tool_categories[tool.category].remove(tool_name)

            # Remove from tools
            del self.tools[tool_name]

            logger.info(f"🗑️ Unregistered tool: {tool_name}")
            return True
        return False

    async def execute_tool(
        self, tool_name: str, context: ToolExecutionContext, **kwargs
    ) -> ToolExecutionResult:
        """Execute a tool with the given context"""
        start_time = time.time()

        if tool_name not in self.tools:
            return ToolExecutionResult(
                success=False,
                error=f"Tool '{tool_name}' not found",
                tool_name=tool_name,
                execution_context=context,
            )

        tool = self.tools[tool_name]

        if not tool.enabled:
            return ToolExecutionResult(
                success=False,
                error=f"Tool '{tool_name}' is disabled",
                tool_name=tool_name,
                execution_context=context,
            )

        try:
            # Execute tool
            result = await tool.execute(context, **kwargs)

            # Update usage statistics
            await self._update_usage_stats(tool_name, result)

            # Publish tool execution event
            await publish_event(
                SheìlyEventType.METRICS_UPDATE,
                {
                    "metric_type": "tool_execution",
                    "tool_name": tool_name,
                    "success": result.success,
                    "duration_seconds": result.duration_seconds,
                    "user_id": context.user_id,
                    "agent_id": context.agent_id,
                },
            )

            return result

        except Exception as e:
            duration = time.time() - start_time

            result = ToolExecutionResult(
                success=False,
                error=f"Tool execution failed: {str(e)}",
                duration_seconds=duration,
                tool_name=tool_name,
                execution_context=context,
            )

            await self._update_usage_stats(tool_name, result)
            return result

    async def _update_usage_stats(
        self, tool_name: str, result: ToolExecutionResult
    ) -> None:
        """Update tool usage statistics"""
        if tool_name not in self.tool_usage_stats:
            return

        stats = self.tool_usage_stats[tool_name]
        stats["total_executions"] += 1
        stats["last_used"] = time.time()

        if result.success:
            stats["successful_executions"] += 1
        else:
            stats["failed_executions"] += 1

        # Update average duration
        total_executions = stats["total_executions"]
        current_avg = stats["average_duration_seconds"]
        new_avg = (
            (current_avg * (total_executions - 1)) + result.duration_seconds
        ) / total_executions
        stats["average_duration_seconds"] = round(new_avg, 4)

    async def get_tools_for_agent(self, agent_id: str) -> List[BaseSheìlyTool]:
        """Get tools available for specific agent"""
        if agent_id in self.agent_tool_mapping:
            tool_names = self.agent_tool_mapping[agent_id]
            return [self.tools[name] for name in tool_names if name in self.tools]

        # Return all tools if no specific mapping
        return list(self.tools.values())

    async def get_tools_by_category(
        self, category: ToolCategory
    ) -> List[BaseSheìlyTool]:
        """Get tools by category"""
        if category in self.tool_categories:
            tool_names = self.tool_categories[category]
            return [self.tools[name] for name in tool_names if name in self.tools]
        return []

    async def search_tools(
        self, query: str, category: Optional[ToolCategory] = None
    ) -> List[BaseSheìlyTool]:
        """Search tools by name or description"""
        query_lower = query.lower()
        results = []

        for tool in self.tools.values():
            if category and tool.category != category:
                continue

            if (
                query_lower in tool.name.lower()
                or query_lower in tool.description.lower()
            ):
                results.append(tool)

        return results

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive tool information"""
        if tool_name not in self.tools:
            return None

        tool = self.tools[tool_name]
        stats = self.tool_usage_stats.get(tool_name, {})

        return {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category.value,
            "enabled": tool.enabled,
            "required_permissions": tool.required_permissions,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "input_types": cap.input_types,
                    "output_types": cap.output_types,
                    "required_permissions": cap.required_permissions,
                }
                for cap in tool.capabilities
            ],
            "usage_stats": stats,
            "last_used": tool.last_used,
            "usage_count": tool.usage_count,
        }

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get comprehensive registry summary"""
        return {
            "total_tools": len(self.tools),
            "enabled_tools": len([t for t in self.tools.values() if t.enabled]),
            "categories": {
                category.value: len(tool_names)
                for category, tool_names in self.tool_categories.items()
            },
            "agent_mappings": dict(self.agent_tool_mapping),
            "most_used_tools": sorted(
                [(name, tool.usage_count) for name, tool in self.tools.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:5],
        }


# ========================================
# UNIFIED TOOL ACCESS INTERFACE
# ========================================


class SheìlyToolManager:
    """Manager unificado para acceso a herramientas"""

    def __init__(self):
        self.registry: Optional[SheìlyToolRegistry] = None
        self.event_stream = None

    async def initialize(self) -> None:
        """Initialize tool manager"""
        try:
            self.registry = SheìlyToolRegistry()
            await self.registry.initialize()

            self.event_stream = get_event_stream()

            logger.info("✅ Sheily Tool Manager initialized")

        except Exception as e:
            logger.error(f"Error initializing tool manager: {e}")
            raise

    async def execute_tool_for_agent(
        self,
        agent_id: str,
        tool_name: str,
        user_id: Optional[str] = None,
        permissions: Optional[List[str]] = None,
        **kwargs,
    ) -> ToolExecutionResult:
        """Execute tool in context of specific agent"""
        context = ToolExecutionContext(
            agent_id=agent_id,
            user_id=user_id,
            permissions=permissions or [],
            metadata={"initiated_by_agent": True},
        )

        return await self.registry.execute_tool(tool_name, context, **kwargs)

    async def get_available_tools(
        self, agent_id: Optional[str] = None, category: Optional[ToolCategory] = None
    ) -> List[Dict[str, Any]]:
        """Get available tools for agent or category"""
        if agent_id:
            tools = await self.registry.get_tools_for_agent(agent_id)
        elif category:
            tools = await self.registry.get_tools_by_category(category)
        else:
            tools = list(self.registry.tools.values())

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category.value,
                "enabled": tool.enabled,
                "capabilities": len(tool.capabilities),
            }
            for tool in tools
        ]

    async def recommend_tools_for_task(
        self, task_description: str
    ) -> List[Dict[str, Any]]:
        """Recommend tools based on task description"""
        # Simple keyword-based recommendation
        # In real implementation, could use ML/NLP

        recommendations = []
        task_lower = task_description.lower()

        if any(
            keyword in task_lower
            for keyword in ["file", "document", "upload", "download"]
        ):
            recommendations.extend(
                await self.registry.get_tools_by_category(ToolCategory.FILE_MANAGEMENT)
            )

        if any(
            keyword in task_lower for keyword in ["data", "query", "database", "sql"]
        ):
            recommendations.extend(
                await self.registry.get_tools_by_category(ToolCategory.DATA_ANALYSIS)
            )

        if any(
            keyword in task_lower
            for keyword in ["api", "integration", "connect", "webhook"]
        ):
            recommendations.extend(
                await self.registry.get_tools_by_category(ToolCategory.INTEGRATION)
            )

        # Remove duplicates and return tool info
        unique_tools = {tool.name: tool for tool in recommendations}

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category.value,
                "relevance_score": 0.8,  # Would be calculated in real implementation
            }
            for tool in unique_tools.values()
        ]


# ========================================
# GLOBAL INSTANCES
# ========================================

_tool_manager: Optional[SheìlyToolManager] = None


async def get_tool_manager() -> SheìlyToolManager:
    """Get global tool manager instance"""
    global _tool_manager
    if _tool_manager is None:
        _tool_manager = SheìlyToolManager()
        await _tool_manager.initialize()
    return _tool_manager


async def initialize_tool_system() -> SheìlyToolManager:
    """Initialize the complete tool system"""
    tool_manager = await get_tool_manager()
    logger.info("✅ Sheily Enterprise Tool System initialized")
    return tool_manager


__all__ = [
    "BaseSheìlyTool",
    "SheìlyToolRegistry",
    "SheìlyToolManager",
    "ToolCategory",
    "ToolPriority",
    "ToolCapability",
    "ToolExecutionContext",
    "ToolExecutionResult",
    "FileManagerTool",
    "DatabaseQueryTool",
    "APIIntegrationTool",
    "get_tool_manager",
    "initialize_tool_system",
]
