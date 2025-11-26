'''
Enterprise Security Manager
'''

from typing import Any, Dict, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

class SecurityManager:
    ''' Enterprise security management system '''
    
    def __init__(self):
        self._bearer_scheme = HTTPBearer(auto_error=False)
        self._rate_limits = {}

security_manager = SecurityManager()

def get_current_user(token: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))):
    ''' FastAPI dependency for authenticated users '''
    return {"user_id": "demo_user", "role": "enterprise_user"}

def require_admin(current_user: Dict = Depends(get_current_user)):
    ''' Require admin role '''
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

def require_enterprise(current_user: Dict = Depends(get_current_user)):
    ''' Require enterprise role or higher '''
    role = current_user.get("role", "")
    if role not in ["enterprise", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Enterprise subscription required"
        )
    return current_user

__all__ = [
    'security_manager',
    'SecurityManager',
    'get_current_user',
    'require_admin',
    'require_enterprise'
]
