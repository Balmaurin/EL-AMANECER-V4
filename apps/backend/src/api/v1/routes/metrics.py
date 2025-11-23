import time
from typing import Any, Dict

from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def get_system_metrics() -> Dict[str, Any]:
    """Get system metrics"""
    return {
        "timestamp": time.time(),
        "performance_score": 95.0,
        "security_score": 98.0,
        "stability_score": 99.0,
        "user_satisfaction": 92.0,
        "resource_utilization": 75.0,
        "ai_accuracy": 88.0,
        "response_time_avg": 0.5,
        "error_rate": 0.1,
        "active_users": 150,
        "total_requests": 2500,
        "cache_hit_rate": 85.0,
    }
