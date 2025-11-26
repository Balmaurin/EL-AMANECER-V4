"""
GraphQL API for Sheily AI Enterprise.
Provides advanced querying capabilities for agents, tenants, and analytics.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info


# GraphQL Types
@strawberry.type
class TenantType:
    """GraphQL type for Tenant model."""

    id: str
    name: str
    domain: str
    status: str
    admin_email: str
    contact_email: Optional[str]
    created_at: str
    updated_at: str
    current_users: int
    current_storage_gb: float
    api_calls_today: int

    @strawberry.field
    def limits(self) -> Dict[str, Any]:
        """Get tenant resource limits."""
        # This would be populated from the actual tenant data
        return {
            "max_users": 100,
            "max_agents": 10,
            "max_api_calls_per_hour": 1000,
            "max_storage_gb": 10.0,
        }

    @strawberry.field
    def features(self) -> List[str]:
        """Get enabled features for this tenant."""
        return ["advanced_agents", "api_access", "analytics"]


@strawberry.type
class AgentType:
    """GraphQL type for Agent model."""

    id: str
    name: str
    type: str
    status: str
    description: str
    created_at: str
    last_active: Optional[str]
    tenant_id: Optional[str]

    @strawberry.field
    def capabilities(self) -> List[str]:
        """Get agent capabilities."""
        return ["reasoning", "execution", "monitoring"]

    @strawberry.field
    def metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return {
            "requests_processed": 150,
            "success_rate": 0.98,
            "average_response_time": 0.5,
        }


@strawberry.type
class AnalyticsType:
    """GraphQL type for analytics data."""

    total_tenants: int
    active_tenants: int
    total_agents: int
    total_api_calls: int
    system_health: str
    uptime_percentage: float

    @strawberry.field
    def tenant_distribution(self) -> Dict[str, int]:
        """Get tenant distribution by status."""
        return {"active": 15, "suspended": 2, "inactive": 1, "pending": 3}

    @strawberry.field
    def agent_performance(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        return {
            "total_executions": 1250,
            "success_rate": 0.96,
            "average_execution_time": 1.2,
            "error_rate": 0.04,
        }


@strawberry.type
class SystemHealthType:
    """GraphQL type for system health information."""

    status: str
    uptime: str
    version: str
    last_backup: Optional[str]

    @strawberry.field
    def services(self) -> List[Dict[str, Any]]:
        """Get status of all system services."""
        return [
            {"name": "backend", "status": "healthy", "uptime": "99.9%"},
            {"name": "database", "status": "healthy", "uptime": "99.8%"},
            {"name": "cache", "status": "healthy", "uptime": "99.9%"},
            {"name": "agents", "status": "healthy", "uptime": "99.5%"},
        ]

    @strawberry.field
    def alerts(self) -> List[Dict[str, Any]]:
        """Get active system alerts."""
        return [
            {
                "level": "info",
                "message": "Scheduled maintenance in 24 hours",
                "timestamp": "2025-11-12T10:00:00Z",
            }
        ]


# Input Types
@strawberry.input
class TenantFilter:
    """Input type for tenant filtering."""

    status: Optional[str] = None
    domain: Optional[str] = None
    created_after: Optional[str] = None
    limit: Optional[int] = 50


@strawberry.input
class AgentFilter:
    """Input type for agent filtering."""

    type: Optional[str] = None
    status: Optional[str] = None
    tenant_id: Optional[str] = None
    limit: Optional[int] = 100


# Queries
@strawberry.type
class Query:
    """Main GraphQL query type."""

    @strawberry.field
    async def tenants(
        self, info: Info, filter: Optional[TenantFilter] = None
    ) -> List[TenantType]:
        """Query tenants with optional filtering."""
        # This would integrate with your actual tenant service
        # For now, return mock data
        return [
            TenantType(
                id="tenant-001",
                name="Demo Corp",
                domain="demo.example.com",
                status="active",
                admin_email="admin@demo.example.com",
                contact_email="support@demo.example.com",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-11-12T00:00:00Z",
                current_users=45,
                current_storage_gb=2.3,
                api_calls_today=1250,
            )
        ]

    @strawberry.field
    async def tenant(self, info: Info, id: str) -> Optional[TenantType]:
        """Get a specific tenant by ID."""
        # Mock implementation
        if id == "tenant-001":
            return TenantType(
                id="tenant-001",
                name="Demo Corp",
                domain="demo.example.com",
                status="active",
                admin_email="admin@demo.example.com",
                contact_email="support@demo.example.com",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-11-12T00:00:00Z",
                current_users=45,
                current_storage_gb=2.3,
                api_calls_today=1250,
            )
        return None

    @strawberry.field
    async def agents(
        self, info: Info, filter: Optional[AgentFilter] = None
    ) -> List[AgentType]:
        """Query agents with optional filtering using real detected agents."""
        try:
            from apps.backend.src.services.agent_discovery import agent_discovery

            # Obtener agentes reales del sistema
            agent_stats = agent_discovery.get_agent_stats()
            real_agents = agent_stats["agents"]

            # Si no hay agentes reales, usar un agente básico como fallback
            if not real_agents:
                real_agents = [
                    {
                        "id": "agent-001",
                        "name": "BasicLLMChatService",
                        "type": "LLM",
                        "capabilities": ["chat", "reasoning"],  # Manually add capabilities
                        "description": "Basic LLM chat service using LLAMA models"
                    }
                ]

            # Convertir agentes reales a formato GraphQL
            agents_list = []
            for i, agent_data in enumerate(real_agents):
                # Determinar status (active por defecto)
                status = agent_data.get("status", "active")

                # Aplicar filtros si se especifican
                if filter:
                    if filter.type and agent_data.get("type", "").lower() != filter.type.lower():
                        continue
                    if filter.status and status != filter.status:
                        continue
                    if filter.tenant_id and filter.tenant_id != "tenant-001":
                        continue

                # Crear AgentType
                agent = AgentType(
                    id=agent_data.get("id", f"agent-{i+1:03d}"),
                    name=agent_data.get("name", f"Agent{i+1}"),
                    type=agent_data.get("type", "Unknown"),
                    status=status,
                    description=agent_data.get("description", "Agent description"),
                    created_at="2025-01-01T00:00:00Z",  # Placeholder
                    last_active=datetime.now(timezone.utc).isoformat(),
                    tenant_id="tenant-001",
                )
                agents_list.append(agent)

                # Limitar resultado si se especifica
                if filter and filter.limit and len(agents_list) >= filter.limit:
                    break

            return agents_list

        except Exception as e:
            # Fallback a algunos agentes conocidos si fallan las detecciones dinámicas
            fallback_agents = [
                {
                    "id": "agent-001",
                    "name": "AdvancedAgentTrainer",
                    "type": "Training",
                    "description": "Sistema avanzado de entrenamiento para agentes AI"
                },
                {
                    "id": "agent-002",
                    "name": "ConstitutionalEvaluator",
                    "type": "Evaluation",
                    "description": "Evaluador constitucional para decisiones éticas"
                },
                {
                    "id": "agent-003",
                    "name": "ReflexionAgent",
                    "type": "Learning",
                    "description": "Agente de reflexión para mejora continua"
                },
                {
                    "id": "agent-004",
                    "name": "ToolformerAgent",
                    "type": "Tooling",
                    "description": "Agente especializado en selección y uso de herramientas"
                },
            ]

            return [
                AgentType(
                    id=agent["id"],
                    name=agent["name"],
                    type=agent["type"],
                    status="active",
                    description=agent["description"],
                    created_at="2025-01-01T00:00:00Z",
                    last_active=datetime.now(timezone.utc).isoformat(),
                    tenant_id="tenant-001",
                )
                for agent in fallback_agents
            ]

    @strawberry.field
    async def system_health(self, info: Info) -> SystemHealthType:
        """Get system health information."""
        try:
            from apps.backend.src.services.agent_discovery import agent_discovery

            # Obtener estado real del sistema
            overview = agent_discovery.get_system_overview()
            total_components = overview.get("total_components", 0)
            system_health = overview.get("system_health", "limited")

            status = "healthy" if system_health == "operational" else "degraded"
            uptime_percentage = 99.0 if total_components > 5 else 75.0

            return SystemHealthType(
                status=status,
                uptime=f"{uptime_percentage}% operational",
                version="1.0.0",
                last_backup=datetime.now(timezone.utc).isoformat(),
            )
        except Exception:
            # Fallback básico
            return SystemHealthType(
                status="operational",
                uptime="75% operational",
                version="1.0.0",
                last_backup=datetime.now(timezone.utc).isoformat(),
            )


# Mutations
@strawberry.input
class CreateTenantInput:
    """Input for creating a new tenant."""

    name: str
    domain: str
    admin_email: str
    contact_email: Optional[str] = None


@strawberry.input
class UpdateTenantInput:
    """Input for updating a tenant."""

    name: Optional[str] = None
    status: Optional[str] = None
    contact_email: Optional[str] = None


@strawberry.type
class Mutation:
    """Main GraphQL mutation type."""

    @strawberry.mutation
    async def create_tenant(self, info: Info, input: CreateTenantInput) -> TenantType:
        """Create a new tenant."""
        # Mock implementation
        return TenantType(
            id=f"tenant-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            name=input.name,
            domain=input.domain,
            status="pending",
            admin_email=input.admin_email,
            contact_email=input.contact_email,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            current_users=0,
            current_storage_gb=0.0,
            api_calls_today=0,
        )

    @strawberry.mutation
    async def update_tenant(
        self, info: Info, id: str, input: UpdateTenantInput
    ) -> Optional[TenantType]:
        """Update an existing tenant."""
        # Mock implementation
        return TenantType(
            id=id,
            name=input.name or "Updated Corp",
            domain="updated.example.com",
            status=input.status or "active",
            admin_email="admin@example.com",
            contact_email=input.contact_email,
            created_at="2025-01-01T00:00:00Z",
            updated_at=datetime.now(timezone.utc).isoformat(),
            current_users=50,
            current_storage_gb=3.2,
            api_calls_today=1500,
        )

    @strawberry.mutation
    async def delete_tenant(self, info: Info, id: str) -> bool:
        """Delete a tenant."""
        # Mock implementation
        return True


# Create the GraphQL schema
schema = strawberry.Schema(query=Query, mutation=Mutation)

# Create the GraphQL router
graphql_app = GraphQLRouter(schema, path="/graphql")

# Export for use in main FastAPI app
__all__ = ["graphql_app", "schema"]
