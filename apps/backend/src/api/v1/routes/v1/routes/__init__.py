"""
Sheily MCP API v1 Routes
=========================

All API route modules for version 1 of the Sheily MCP Enterprise API.
Contains authentication, chat, agents, datasets, and other enterprise endpoints.
"""

from .auth import router as auth_router
from .chat import router as chat_router
from .datasets import router as datasets_router

__all__ = ["auth_router", "chat_router", "datasets_router"]
