from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter()

@router.get("/tokens")
async def get_user_tokens() -> Dict[str, Any]:
    """
    Get user token balance and usage.
    Mocked for now to satisfy frontend requirements.
    """
    return {
        "total_tokens": 1000000,
        "used_tokens": 5000,
        "available_tokens": 995000,
        "plan": "enterprise"
    }
