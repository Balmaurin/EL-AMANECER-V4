from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel


class SystemMetrics(BaseModel):
    """System metrics data model"""

    timestamp: datetime
    performance_score: float
    security_score: float
    stability_score: float
    user_satisfaction: float
    resource_utilization: float
    ai_accuracy: float
    response_time_avg: float
    error_rate: float
    active_users: int
    total_requests: int
    cache_hit_rate: float

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class HealthStatus(BaseModel):
    """Health status data model"""

    status: str
    timestamp: float
    system: Dict[str, float]
    services: Dict[str, str]
    metadata: Optional[Dict[str, Any]] = None
