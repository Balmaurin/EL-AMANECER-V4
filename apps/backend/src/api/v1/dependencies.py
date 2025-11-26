"""
API Dependencies Module
Common dependencies for FastAPI endpoints
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from apps.backend.src.api.models.user import User

# Security scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Get current authenticated user from JWT token
    
    This is a stub implementation. In production, you should:
    1. Verify the JWT token
    2. Extract user_id from token
    3. Load user from database
    4. Return User object
    """
    # TODO: Implement proper JWT verification
    # For now, return mock user
    return User(
        id="mock_user_id",
        username="mock_user",
        email="mock@example.com",
        created_at="2025-01-01T00:00:00",
        updated_at="2025-01-01T00:00:00",
        is_active=True,
        is_verified=True
    )


async def get_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current user and verify admin permissions
    
    This is a stub implementation. In production, add:
    1. Role/permissions check
    2. Admin verification
    3. Proper authorization logic
    """
    # TODO: Check if user is admin
    # For now, just return the user
    return current_user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    )
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise
    Useful for endpoints that work for both authenticated and anonymous users
    """
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials)
    except:
        return None


__all__ = [
    "get_current_user",
    "get_admin_user",
    "get_optional_user",
    "security"
]
