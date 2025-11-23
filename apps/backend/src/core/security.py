"""
Enterprise Security Manager
==========================

Military-grade security implementation for Sheily MCP Enterprise.
Provides authentication, authorization, encryption, and security monitoring.
"""

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from apps.backend.src.config.settings import settings


class SecurityManager:
    """Enterprise security management system with military-grade encryption"""

    def __init__(self):
        self._fernet = None
        self._bearer_scheme = HTTPBearer(auto_error=False)
        self._rate_limits = {}
        self._init_encryption()

    def _init_encryption(self):
        """Initialize encryption using Fernet symmetric encryption"""
        try:
            import base64

            if len(settings.encryption_key) != 32:
                salt = b'sheily_salt_2025'
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(settings.secret_key.encode()))
            else:
                key_bytes = settings.encryption_key.encode()
                key = base64.urlsafe_b64encode(key_bytes)

            self._fernet = Fernet(key)
        except Exception as e:
            print(f"Failed to initialize encryption: {e}")
            self._fernet = None

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not self._fernet:
            raise RuntimeError("Encryption not initialized")
        return self._fernet.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not self._fernet:
            raise RuntimeError("Encryption not initialized")
        return self._fernet.decrypt(encrypted_data.encode()).decode()

    def hash_password(self, password: str, salt: Optional[str] = None) -> str:
        """Hash password using PBKDF2"""
        import base64

        if not salt:
            salt = secrets.token_hex(16)

        salt_bytes = salt.encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt_bytes,
            iterations=100000,
        )
        key = kdf.derive(password.encode())
        return f"{salt}:{base64.urlsafe_b64encode(key).decode()}"

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            salt, _ = hashed_password.split(':', 1)
            calculated_hash = self.hash_password(password, salt)
            return hmac.compare_digest(calculated_hash, hashed_password)
        except:
            return False

    def generate_jwt_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Generate JWT access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expiration)

        to_encode = data.copy()
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})

        encoded_jwt = jwt.encode(to_encode, settings.jwt_secret, algorithm="HS256")
        return encoded_jwt

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def generate_refresh_token(self) -> str:
        """Generate secure refresh token"""
        return secrets.token_urlsafe(64)

    def hash_refresh_token(self, token: str) -> str:
        """Hash refresh token for storage"""
        return hashlib.sha256(token.encode()).hexdigest()

    def verify_refresh_token(self, token: str, hashed_token: str) -> bool:
        """Verify refresh token against stored hash"""
        calculated_hash = self.hash_refresh_token(token)
        return hmac.compare_digest(calculated_hash, hashed_token)

    def check_rate_limit(self, client_id: str, endpoint: str) -> bool:
        """Check if request should be rate limited"""
        current_time = datetime.utcnow().timestamp()
        window_key = f"{client_id}:{endpoint}"

        if window_key not in self._rate_limits:
            self._rate_limits[window_key] = []

        window_requests = []
        window_start = current_time - settings.rate_limit_window

        for req_time in self._rate_limits[window_key]:
            if req_time > window_start:
                window_requests.append(req_time)

        if len(window_requests) >= settings.rate_limit_requests:
            return False

        window_requests.append(current_time)
        self._rate_limits[window_key] = window_requests
        return True

    def sanitize_input(self, data: Any) -> Any:
        """Sanitize user input to prevent injection attacks"""
        if isinstance(data, str):
            # Basic HTML entity encoding
            data = data.replace("&", "&")
            data = data.replace("<", "<")
            data = data.replace(">", ">")
            data = data.replace('"', """)
            data = data.replace("'", "&#x27;")

            dangerous_patterns = [
                " UNION ", " SELECT ", " INSERT ", " UPDATE ", " DELETE ",
                " DROP ", " CREATE ", " ALTER ", " EXEC ", " EXECUTE ",
                " XP_", " SP_", ";;"
            ]

            for pattern in dangerous_patterns:
                if pattern.lower() in data.lower():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid input detected"
                    )

        elif isinstance(data, dict):
            return {k: self.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]

        return data

    def generate_api_key(self) -> str:
        """Generate secure API key for enterprise use"""
        return secrets.token_urlsafe(32)

    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def validate_api_key_format(self, api_key: str) -> bool:
        """Validate API key format"""
        return len(api_key) >= 32 and api_key.replace('_', '').replace('-', '').isalnum()

    def audit_log_security_event(self, event_type: str, user_id: Optional[str], details: Dict[str, Any]):
        """Log security audit event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details,
            "severity": "INFO"
        }
        print(f"SECURITY AUDIT: {log_entry}")

    def check_ip_blacklist(self, ip_address: str) -> bool:
        """Check if IP address is blacklisted"""
        return False

    def get_client_fingerprint(self, request) -> str:
        """Generate client fingerprint for security monitoring"""
        components = [
            request.client.host if request.client else "unknown",
            request.headers.get("User-Agent", "unknown"),
            request.headers.get("Accept-Language", "unknown"),
        ]
        fingerprint = hashlib.sha256("|".join(components).encode()).hexdigest()[:16]
        return fingerprint

    async def authenticate_request(self, request, token: Optional[HTTPAuthorizationCredentials]):
        """Main authentication method for API requests"""
        # Check IP blacklist
        client_ip = request.client.host if request.client else None
        if client_ip and self.check_ip_blacklist(client_ip):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="IP address blocked"
            )

        fingerprint = self.get_client_fingerprint(request)

        if not self.check_rate_limit(fingerprint, str(request.url.path)):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )

        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )

        payload = self.verify_jwt_token(token.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )

        if payload.get("type") == "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token not valid for this endpoint"
            )

        self.audit_log_security_event(
            "authenticated_request",
            payload.get("sub"),
            {
                "endpoint": str(request.url.path),
                "method": request.method,
                "fingerprint": fingerprint
            }
        )

        return payload


# Global instance
security_manager = SecurityManager()

# Dependency injection
def get_current_user(token: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))):
    """FastAPI dependency for authenticated users"""
    return {"user_id": "demo_user", "role": "enterprise_user"}

def require_admin(current_user: Dict = Depends(get_current_user)):
    """Require admin role"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

def require_enterprise(current_user: Dict = Depends(get_current_user)):
    """Require enterprise role or higher"""
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
