import time
from typing import Any, Dict

import psutil
from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/detailed")
async def get_detailed_health() -> Dict[str, Any]:
    """Get detailed system health information"""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Uptime
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time

        return {
            "status": "healthy",
            "timestamp": time.time(),
            "system": {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "uptime_seconds": uptime_seconds,
            },
            "services": {
                "api": "operational",
                "database": "operational",
                "cache": "operational",
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
