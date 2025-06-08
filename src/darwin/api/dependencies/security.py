"""
Security Dependencies for Darwin API

This module provides FastAPI dependency functions for authentication, authorization,
and security validation. These dependencies can be used to protect API endpoints
and ensure proper access control.

Features:
- JWT token validation dependencies
- User authentication dependencies
- Role-based authorization dependencies
- Permission checking dependencies
- API key authentication dependencies
- Rate limiting dependencies
"""

import logging
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from darwin.security.auth import AuthManager
from darwin.security.exceptions import (
    AuthenticationError,
    InsufficientPermissionsError,
    TokenExpiredError,
)
from darwin.security.models import AuthToken
from darwin.security.rbac import AccessControl, PermissionType, ResourceType

logger = logging.getLogger(__name__)

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)

# Global references to security managers (set during app startup)
_auth_manager: Optional[AuthManager] = None
_access_control: Optional[AccessControl] = None


def set_security_managers(auth_manager: AuthManager, access_control: AccessControl):
    """Set global security managers for use in dependencies."""
    global _auth_manager, _access_control
    _auth_manager = auth_manager
    _access_control = access_control


def get_auth_manager() -> AuthManager:
    """Dependency to get the authentication manager."""
    if _auth_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available",
        )
    return _auth_manager


def get_access_control() -> AccessControl:
    """Dependency to get the access control system."""
    if _access_control is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Access control service not available",
        )
    return _access_control


async def get_current_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)],
    auth_manager: Annotated[AuthManager, Depends(get_auth_manager)],
) -> AuthToken:
    """
    Dependency to extract and validate JWT token from Authorization header.

    Args:
        credentials: HTTP Bearer token credentials
        auth_manager: Authentication manager instance

    Returns:
        Validated AuthToken object

    Raises:
        HTTPException: If token is invalid, expired, or missing
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        token = auth_manager.jwt_manager.validate_token(credentials.credentials)
        return token
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    token: Annotated[AuthToken, Depends(get_current_token)]
) -> dict:
    """
    Dependency to get current authenticated user information.

    Args:
        token: Validated authentication token

    Returns:
        User information dictionary
    """
    return {
        "user_id": token.user_id,
        "username": token.username,
        "roles": token.roles,
        "permissions": token.permissions,
    }


def require_permission(permission: PermissionType, resource: ResourceType):
    """
    Factory function to create a dependency that requires specific permission.

    Args:
        permission: Required permission type
        resource: Resource type the permission applies to

    Returns:
        FastAPI dependency function
    """

    async def permission_dependency(
        current_user: Annotated[dict, Depends(get_current_user)],
        access_control: Annotated[AccessControl, Depends(get_access_control)],
    ) -> dict:
        """Check if current user has required permission."""
        try:
            access_control.require_permission(
                current_user["user_id"], permission, resource
            )
            return current_user
        except InsufficientPermissionsError as e:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

    return permission_dependency


def require_role(required_role: str):
    """
    Factory function to create a dependency that requires specific role.

    Args:
        required_role: Required role name

    Returns:
        FastAPI dependency function
    """

    async def role_dependency(
        current_user: Annotated[dict, Depends(get_current_user)]
    ) -> dict:
        """Check if current user has required role."""
        if required_role not in current_user["roles"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required",
            )
        return current_user

    return role_dependency


# Common permission dependencies
RequireReadAccess = require_permission(PermissionType.READ, ResourceType.OPTIMIZATION)
RequireWriteAccess = require_permission(PermissionType.WRITE, ResourceType.OPTIMIZATION)
RequireDeleteAccess = require_permission(
    PermissionType.DELETE, ResourceType.OPTIMIZATION
)
RequireAdminAccess = require_permission(PermissionType.ADMIN, ResourceType.SYSTEM)

# Common role dependencies
RequireAdminRole = require_role("admin")
RequireUserRole = require_role("user")


async def get_optional_user(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)
    ],
    auth_manager: Annotated[AuthManager, Depends(get_auth_manager)],
) -> Optional[dict]:
    """
    Dependency to optionally get current user (doesn't require authentication).

    Args:
        credentials: Optional HTTP Bearer token credentials
        auth_manager: Authentication manager instance

    Returns:
        User information if authenticated, None otherwise
    """
    if not credentials:
        return None

    try:
        token = auth_manager.jwt_manager.validate_token(credentials.credentials)
        return {
            "user_id": token.user_id,
            "username": token.username,
            "roles": token.roles,
            "permissions": token.permissions,
        }
    except (TokenExpiredError, AuthenticationError):
        return None


async def validate_api_key(
    request: Request, auth_manager: Annotated[AuthManager, Depends(get_auth_manager)]
) -> dict:
    """
    Dependency to validate API key from X-API-Key header.

    Args:
        request: FastAPI request object
        auth_manager: Authentication manager instance

    Returns:
        API key information

    Raises:
        HTTPException: If API key is invalid or missing
    """
    api_key = request.headers.get("X-API-Key")

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # TODO: Implement API key validation logic
    # This would typically involve checking the key against a database
    # For now, we'll return a placeholder
    return {
        "api_key_id": "placeholder",
        "permissions": ["read", "write"],
        "rate_limit": "1000/hour",
    }


async def rate_limit_check(
    request: Request,
    current_user: Annotated[Optional[dict], Depends(get_optional_user)] = None,
) -> None:
    """
    Dependency to check rate limits for the current request.

    Args:
        request: FastAPI request object
        current_user: Optional current user information

    Raises:
        HTTPException: If rate limit is exceeded
    """
    # Get client identifier
    client_id = None
    if current_user:
        client_id = current_user["user_id"]
    else:
        # Use IP address for unauthenticated requests
        client_id = request.client.host if request.client else "unknown"

    # TODO: Implement actual rate limiting logic
    # This would typically involve checking a cache/database for request counts
    # For now, we'll just pass through
    pass


async def security_audit_log(
    request: Request,
    current_user: Annotated[Optional[dict], Depends(get_optional_user)] = None,
) -> None:
    """
    Dependency to log security-relevant events.

    Args:
        request: FastAPI request object
        current_user: Optional current user information
    """
    # Log the request for security auditing
    logger.info(
        f"API Request: {request.method} {request.url.path}",
        extra={
            "user_id": current_user.get("user_id") if current_user else None,
            "username": current_user.get("username") if current_user else None,
            "ip_address": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "endpoint": f"{request.method} {request.url.path}",
        },
    )


# Convenience type aliases for common dependency combinations
AuthenticatedUser = Annotated[dict, Depends(get_current_user)]
OptionalUser = Annotated[Optional[dict], Depends(get_optional_user)]
AdminUser = Annotated[dict, Depends(RequireAdminRole)]
ReadOnlyUser = Annotated[dict, Depends(RequireReadAccess)]
WriteUser = Annotated[dict, Depends(RequireWriteAccess)]
