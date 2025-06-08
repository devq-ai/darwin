"""
Authentication API Routes for Darwin Security System

This module provides FastAPI endpoints for user authentication, including login,
logout, token refresh, password management, and user registration.

Features:
- User login with JWT token generation
- Token refresh endpoint
- User logout with token blacklisting
- Password change functionality
- User registration
- Security audit logging
- Rate limiting integration
- Comprehensive error handling
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from darwin.security.auth import AuthManager
from darwin.security.exceptions import (
    AuthenticationError,
    InvalidCredentialsError,
    TokenExpiredError,
)
from darwin.security.models import (
    AuthenticationRequest,
    AuthenticationResponse,
    PasswordChangeRequest,
    TokenRefreshRequest,
    TokenRefreshResponse,
    UserInfo,
    UserRegistrationRequest,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/auth", tags=["Authentication"])

# Security scheme
security = HTTPBearer()

# This would be injected via dependency injection in a real application
auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get authentication manager dependency."""
    if auth_manager is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication manager not initialized",
        )
    return auth_manager


def set_auth_manager(manager: AuthManager):
    """Set the authentication manager (for initialization)."""
    global auth_manager
    auth_manager = manager


def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    # Check for forwarded headers first
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fall back to direct connection
    return request.client.host if request.client else "unknown"


@router.post(
    "/login",
    response_model=AuthenticationResponse,
    status_code=status.HTTP_200_OK,
    summary="User Login",
    description="Authenticate user and return access and refresh tokens",
)
async def login(
    request: Request,
    auth_request: AuthenticationRequest,
    auth_mgr: AuthManager = Depends(get_auth_manager),
):
    """
    Authenticate a user and return JWT tokens.

    Args:
        request: FastAPI request object
        auth_request: Authentication request with credentials
        auth_mgr: Authentication manager dependency

    Returns:
        Authentication response with tokens and user info

    Raises:
        HTTPException: If authentication fails
    """
    try:
        client_ip = get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "unknown")

        # Authenticate user
        access_token, refresh_token = auth_mgr.authenticate_user(
            credentials=auth_request, ip_address=client_ip, user_agent=user_agent
        )

        # Get user information
        user_data = auth_mgr._get_user_by_username(auth_request.username)

        user_info = {
            "id": user_data["id"],
            "username": user_data["username"],
            "roles": user_data.get("roles", []),
            "permissions": user_data.get("permissions", []),
        }

        logger.info(f"User logged in successfully: {auth_request.username}")

        return AuthenticationResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user_info=user_info,
            expires_in=access_token.expires_in_seconds,
        )

    except InvalidCredentialsError as e:
        logger.warning(f"Login failed for {auth_request.username}: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    except AuthenticationError as e:
        logger.error(f"Authentication error for {auth_request.username}: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post(
    "/refresh",
    response_model=TokenRefreshResponse,
    status_code=status.HTTP_200_OK,
    summary="Refresh Token",
    description="Refresh access token using refresh token",
)
async def refresh_token(
    refresh_request: TokenRefreshRequest,
    auth_mgr: AuthManager = Depends(get_auth_manager),
):
    """
    Refresh access and refresh tokens.

    Args:
        refresh_request: Token refresh request
        auth_mgr: Authentication manager dependency

    Returns:
        New access and refresh tokens

    Raises:
        HTTPException: If token refresh fails
    """
    try:
        new_access_token, new_refresh_token = auth_mgr.refresh_tokens(
            refresh_request.refresh_token
        )

        logger.info("Tokens refreshed successfully")

        return TokenRefreshResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            expires_in=new_access_token.expires_in_seconds,
        )

    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token has expired"
        )
    except AuthenticationError as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )
    except Exception as e:
        logger.error(f"Unexpected error during token refresh: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="User Logout",
    description="Logout user and invalidate tokens",
)
async def logout(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_mgr: AuthManager = Depends(get_auth_manager),
):
    """
    Logout user and blacklist tokens.

    Args:
        request: FastAPI request object
        credentials: HTTP authorization credentials
        auth_mgr: Authentication manager dependency

    Returns:
        Success message

    Raises:
        HTTPException: If logout fails
    """
    try:
        access_token = credentials.credentials

        # Validate token first
        payload = auth_mgr.validate_token(access_token)
        username = payload.get("username", "unknown")

        # Blacklist the access token
        auth_mgr.logout_user(access_token)

        logger.info(f"User logged out successfully: {username}")

        return {"message": "Logged out successfully"}

    except AuthenticationError as e:
        logger.warning(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"Unexpected error during logout: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post(
    "/change-password",
    status_code=status.HTTP_200_OK,
    summary="Change Password",
    description="Change user password",
)
async def change_password(
    password_request: PasswordChangeRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_mgr: AuthManager = Depends(get_auth_manager),
):
    """
    Change user password.

    Args:
        password_request: Password change request
        credentials: HTTP authorization credentials
        auth_mgr: Authentication manager dependency

    Returns:
        Success message

    Raises:
        HTTPException: If password change fails
    """
    try:
        access_token = credentials.credentials

        # Validate token and get user ID
        payload = auth_mgr.validate_token(access_token)
        user_id = payload["sub"]
        username = payload["username"]

        # Change password
        auth_mgr.change_password(
            user_id=user_id,
            current_password=password_request.current_password.get_secret_value(),
            new_password=password_request.new_password.get_secret_value(),
        )

        logger.info(f"Password changed successfully for user: {username}")

        return {"message": "Password changed successfully"}

    except InvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )
    except AuthenticationError as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during password change: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post(
    "/register",
    response_model=UserInfo,
    status_code=status.HTTP_201_CREATED,
    summary="User Registration",
    description="Register a new user account",
)
async def register_user(
    registration_request: UserRegistrationRequest,
    auth_mgr: AuthManager = Depends(get_auth_manager),
):
    """
    Register a new user.

    Args:
        registration_request: User registration request
        auth_mgr: Authentication manager dependency

    Returns:
        Created user information

    Raises:
        HTTPException: If registration fails
    """
    try:
        # Check if username already exists
        existing_user = auth_mgr._get_user_by_username(registration_request.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists",
            )

        # Validate password strength
        is_valid, errors = auth_mgr.password_manager.validate_password_strength(
            registration_request.password.get_secret_value()
        )
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password validation failed: {', '.join(errors)}",
            )

        # Hash password
        password_hash = auth_mgr.password_manager.hash_password(
            registration_request.password.get_secret_value()
        )

        # Create user (this would save to database in production)
        user_data = {
            "id": f"user-{datetime.now().timestamp()}",
            "username": registration_request.username,
            "email": registration_request.email,
            "password_hash": password_hash,
            "first_name": registration_request.first_name,
            "last_name": registration_request.last_name,
            "roles": ["user"],  # Default role
            "permissions": ["read", "write"],
            "is_active": True,
            "is_verified": False,
            "created_at": datetime.now(timezone.utc),
        }

        logger.info(f"User registered successfully: {registration_request.username}")

        return UserInfo(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            first_name=user_data["first_name"],
            last_name=user_data["last_name"],
            roles=user_data["roles"],
            permissions=user_data["permissions"],
            is_active=user_data["is_active"],
            is_verified=user_data["is_verified"],
            created_at=user_data["created_at"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/me",
    response_model=UserInfo,
    status_code=status.HTTP_200_OK,
    summary="Get Current User",
    description="Get current authenticated user information",
)
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_mgr: AuthManager = Depends(get_auth_manager),
):
    """
    Get current authenticated user information.

    Args:
        credentials: HTTP authorization credentials
        auth_mgr: Authentication manager dependency

    Returns:
        Current user information

    Raises:
        HTTPException: If token is invalid
    """
    try:
        access_token = credentials.credentials

        # Validate token
        payload = auth_mgr.validate_token(access_token)
        user_id = payload["sub"]

        # Get user data
        user_data = auth_mgr._get_user_by_id(user_id)
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        return UserInfo(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data.get("email"),
            first_name=user_data.get("first_name"),
            last_name=user_data.get("last_name"),
            roles=user_data.get("roles", []),
            permissions=user_data.get("permissions", []),
            is_active=True,
            is_verified=True,
            created_at=datetime.now(timezone.utc),
            last_login=datetime.now(timezone.utc),
        )

    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/validate",
    status_code=status.HTTP_200_OK,
    summary="Validate Token",
    description="Validate access token",
)
async def validate_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_mgr: AuthManager = Depends(get_auth_manager),
):
    """
    Validate access token.

    Args:
        credentials: HTTP authorization credentials
        auth_mgr: Authentication manager dependency

    Returns:
        Token validation result

    Raises:
        HTTPException: If token is invalid
    """
    try:
        access_token = credentials.credentials

        # Validate token
        payload = auth_mgr.validate_token(access_token)

        return {
            "valid": True,
            "user_id": payload["sub"],
            "username": payload["username"],
            "roles": payload.get("roles", []),
            "permissions": payload.get("permissions", []),
            "expires_at": payload["exp"],
        }

    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
        )
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"Error validating token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
