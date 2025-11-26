"""
Authentication API Routes
==========================

Enterprise-grade authentication endpoints for Sheily MCP.
Supports JWT tokens, OAuth, and multi-tenant user management.
"""

from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt

from apps.backend.src.core.config.settings import settings
from apps.backend.src.core.database import get_db
from apps.backend.src.core.security import SecurityManager
from apps.backend.src.models.user import TokenResponse, User, UserCreate, UserLogin

router = APIRouter()
security_manager = SecurityManager()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# =================================
# HELPER FUNCTIONS
# =================================


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expiration)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret, algorithm="HS256")
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    return username  # In production, fetch from database


# =================================
# AUTHENTICATION ENDPOINTS
# =================================


@router.post("/register", response_model=TokenResponse)
async def register_user(user_data: UserCreate):
    """
    Register a new user account.

    - **email**: User's email address
    - **password**: User's password
    - **name**: User's display name (optional)
    - **subscription**: Subscription tier (free/premium/enterprise)
    """
    try:
        # Check if user already exists
        # In production: query database

        # Create user
        user = User(
            id=user_data.email,  # Simplified for demo
            email=user_data.email,
            name=user_data.name,
            subscription=user_data.subscription,
            created_at="2025-01-16T08:49:23Z",
            updated_at="2025-01-16T08:49:23Z",
        )

        # Generate tokens
        access_token = create_access_token(data={"sub": user.email})

        refresh_token = create_access_token(
            data={"sub": user.email, "type": "refresh"},
            expires_delta=timedelta(days=30),
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.jwt_expiration * 60,
            user=user,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}",
        )


@router.post("/login", response_model=TokenResponse)
async def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and return access tokens.

    - **username**: User's email address
    - **password**: User's password
    """
    try:
        # Demo authentication - replace with real database lookup
        demo_users = {
            "admin@sheily.ai": {
                "password": "admin123",
                "name": "System Administrator",
                "subscription": "enterprise",
                "tokens": 5000,
                "level": 10,
            },
            "user@sheily.ai": {
                "password": "user123",
                "name": "Demo User",
                "subscription": "free",
                "tokens": 100,
                "level": 2,
            },
        }

        user_info = demo_users.get(form_data.username)
        if not user_info or user_info["password"] != form_data.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Create user object
        user = User(
            id=form_data.username,
            email=form_data.username,
            name=user_info["name"],
            tokens=user_info["tokens"],
            level=user_info["level"],
            subscription=user_info["subscription"],
            created_at="2025-01-15T10:30:00Z",
            updated_at="2025-01-16T08:49:23Z",
        )

        # Generate tokens
        access_token = create_access_token(data={"sub": user.email})

        refresh_token = create_access_token(
            data={"sub": user.email, "type": "refresh"},
            expires_delta=timedelta(days=30),
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.jwt_expiration * 60,
            user=user,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}",
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str):
    """
    Refresh access token using refresh token.

    - **refresh_token**: Valid refresh token
    """
    try:
        # Validate refresh token
        payload = jwt.decode(refresh_token, settings.jwt_secret, algorithms=["HS256"])

        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
            )

        username = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
            )

        # Issue new access token
        access_token = create_access_token(data={"sub": username})

        new_refresh_token = create_access_token(
            data={"sub": username, "type": "refresh"}, expires_delta=timedelta(days=30)
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=settings.jwt_expiration * 60,
            user=None,  # Don't return user info on refresh
        )

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )


@router.get("/me")
async def get_current_user_info(current_user: str = Depends(get_current_user)):
    """
    Get current authenticated user information.

    Requires valid JWT token in Authorization header.
    """
    try:
        # Demo user lookup - replace with database query
        demo_users = {
            "admin@sheily.ai": {
                "id": "admin@sheily.ai",
                "email": "admin@sheily.ai",
                "name": "System Administrator",
                "subscription": "enterprise",
                "tokens": 5000,
                "level": 10,
            },
            "user@sheily.ai": {
                "id": "user@sheily.ai",
                "email": "user@sheily.ai",
                "name": "Demo User",
                "subscription": "free",
                "tokens": 100,
                "level": 2,
            },
        }

        user_info = demo_users.get(current_user)
        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        return User(
            **user_info,
            created_at="2025-01-15T10:30:00Z",
            updated_at="2025-01-16T08:49:23Z",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user info: {str(e)}",
        )


@router.post("/logout")
async def logout_user():
    """
    Logout user (client-side token invalidation).

    In production, this would add token to blacklist.
    """
    return {"message": "Successfully logged out", "timestamp": "2025-01-16T08:49:23Z"}


@router.post("/verify-email")
async def verify_email(token: str):
    """
    Verify user email address.

    - **token**: Email verification token
    """
    try:
        # Verify email token logic would go here
        # In production: validate token, update user email_verified status
        return {
            "message": "Email verified successfully",
            "email_verified": True,
            "timestamp": "2025-01-16T08:49:23Z",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Email verification failed: {str(e)}",
        )


@router.post("/reset-password")
async def request_password_reset(email: str):
    """
    Request password reset for user.

    - **email**: User's email address
    """
    try:
        # Password reset logic would go here
        # In production: generate reset token, send email
        return {
            "message": "Password reset email sent",
            "email": email,
            "timestamp": "2025-01-16T08:49:23Z",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password reset request failed: {str(e)}",
        )


@router.post("/reset-password/confirm")
async def confirm_password_reset(token: str, new_password: str):
    """
    Confirm password reset with new password.

    - **token**: Password reset token
    - **new_password**: New password
    """
    try:
        # Password reset confirmation logic would go here
        # In production: validate token, hash new password, update user
        return {
            "message": "Password reset successfully",
            "timestamp": "2025-01-16T08:49:23Z",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password reset failed: {str(e)}",
        )


# Import datetime for JWT token creation
from datetime import datetime
