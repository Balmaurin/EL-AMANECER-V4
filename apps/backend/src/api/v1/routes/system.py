from typing import Any, Dict

from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def system_status() -> Dict[str, Any]:
    """System status"""
    return {
        "message": "System status endpoint",
        "status": "operational",
        "timestamp": time.time(),
    }
