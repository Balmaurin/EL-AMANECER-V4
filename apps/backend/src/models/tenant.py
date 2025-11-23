"""
Multi-tenancy models for enterprise deployment.
Provides tenant isolation and management capabilities.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TenantStatus(str, Enum):
    """Tenant status enumeration."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"
    PENDING = "pending"


class TenantLimits(BaseModel):
    """Resource limits for a tenant."""

    max_users: int = Field(default=100, description="Maximum number of users")
    max_agents: int = Field(default=10, description="Maximum number of agents")
    max_api_calls_per_hour: int = Field(
        default=1000, description="API calls per hour limit"
    )
    max_storage_gb: float = Field(default=10.0, description="Storage limit in GB")
    max_concurrent_sessions: int = Field(
        default=50, description="Concurrent sessions limit"
    )

    # AI-specific limits
    max_tokens_per_request: int = Field(
        default=4000, description="Max tokens per AI request"
    )
    max_requests_per_minute: int = Field(
        default=60, description="AI requests per minute"
    )
    allowed_models: list[str] = Field(
        default=["default-model", "llama"], description="Allowed AI models"
    )


class TenantConfig(BaseModel):
    """Tenant-specific configuration."""

    theme: Dict[str, Any] = Field(
        default_factory=dict, description="UI theme configuration"
    )
    features: Dict[str, bool] = Field(
        default_factory=lambda: {
            "advanced_agents": True,
            "custom_integrations": False,
            "api_access": True,
            "analytics": True,
        },
        description="Enabled features",
    )
    custom_settings: Dict[str, Any] = Field(
        default_factory=dict, description="Custom tenant settings"
    )


class Tenant(BaseModel):
    """Multi-tenant organization model."""

    id: str = Field(..., description="Unique tenant identifier")
    name: str = Field(..., description="Tenant display name")
    domain: str = Field(..., description="Primary domain for the tenant")
    status: TenantStatus = Field(
        default=TenantStatus.ACTIVE, description="Tenant status"
    )

    # Contact information
    admin_email: str = Field(..., description="Administrator email")
    contact_email: Optional[str] = Field(default=None, description="Contact email")

    # Resource limits and configuration
    limits: TenantLimits = Field(
        default_factory=TenantLimits, description="Resource limits"
    )
    config: TenantConfig = Field(
        default_factory=TenantConfig, description="Tenant configuration"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )
    created_by: Optional[str] = Field(
        default=None, description="User who created the tenant"
    )

    # Usage tracking
    current_users: int = Field(default=0, description="Current number of active users")
    current_storage_gb: float = Field(
        default=0.0, description="Current storage usage in GB"
    )
    api_calls_today: int = Field(default=0, description="API calls made today")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    def is_within_limits(self, resource_type: str, value: float) -> bool:
        """Check if a resource usage is within tenant limits."""
        limits = self.limits

        if resource_type == "users":
            return value <= limits.max_users
        elif resource_type == "agents":
            return value <= limits.max_agents
        elif resource_type == "api_calls_per_hour":
            return value <= limits.max_api_calls_per_hour
        elif resource_type == "storage_gb":
            return value <= limits.max_storage_gb
        elif resource_type == "concurrent_sessions":
            return value <= limits.max_concurrent_sessions
        elif resource_type == "tokens_per_request":
            return value <= limits.max_tokens_per_request
        elif resource_type == "requests_per_minute":
            return value <= limits.max_requests_per_minute

        return True

    def has_feature_access(self, feature: str) -> bool:
        """Check if tenant has access to a specific feature."""
        return self.config.features.get(feature, False)

    def update_usage(self, resource_type: str, value: float):
        """Update resource usage tracking."""
        if resource_type == "users":
            self.current_users = int(value)
        elif resource_type == "storage_gb":
            self.current_storage_gb = value
        elif resource_type == "api_calls_today":
            self.api_calls_today = int(value)

        self.updated_at = datetime.now(timezone.utc)


class TenantContext:
    """Context manager for tenant operations."""

    def __init__(self, tenant: Tenant):
        self.tenant = tenant
        self.start_time = datetime.now(timezone.utc)

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        # Log tenant operation completion
        duration = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        # Here you would typically log to monitoring system
        pass

    def check_limits(self, operation: str) -> bool:
        """Check if operation is allowed within tenant limits."""
        # Implementation would check specific operation limits
        return True
