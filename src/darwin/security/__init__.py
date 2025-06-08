"""
Darwin Security & Authentication Package

This package provides comprehensive security features for the Darwin genetic algorithm
platform, including JWT-based authentication, role-based access control (RBAC),
API security middleware, session management, and security monitoring.

Features:
- JWT token generation, validation, and refresh
- Role-based access control with permissions
- API route protection and middleware
- Password hashing and validation using Argon2
- Session management and security
- Rate limiting and throttling
- Security audit logging
- CORS and security headers
- API key management
- OAuth2 integration support
"""

from .auth import AuthManager, JWTManager, PasswordManager, SessionManager
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    InsufficientPermissionsError,
    InvalidCredentialsError,
    RateLimitExceededError,
    SecurityConfigError,
    TokenExpiredError,
)
from .middleware import (
    AuthenticationMiddleware,
    AuthorizationMiddleware,
    CustomCORSMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from .models import (
    AuthenticationRequest,
    AuthenticationResponse,
    AuthToken,
    SecurityConfig,
    UserCredentials,
)
from .rbac import AccessControl, Permission, PermissionManager, Role, RoleManager, User
from .utils import PasswordValidator, SecurityLogger, SecurityUtils, TokenValidator

__version__ = "1.0.0"
__author__ = "DevQ.ai"

__all__ = [
    # Core authentication
    "AuthManager",
    "JWTManager",
    "PasswordManager",
    "SessionManager",
    # Role-based access control
    "RoleManager",
    "PermissionManager",
    "AccessControl",
    "Role",
    "Permission",
    "User",
    # Middleware
    "AuthenticationMiddleware",
    "AuthorizationMiddleware",
    "SecurityHeadersMiddleware",
    "RateLimitMiddleware",
    "CustomCORSMiddleware",
    # Models
    "AuthToken",
    "UserCredentials",
    "SecurityConfig",
    "AuthenticationRequest",
    "AuthenticationResponse",
    # Exceptions
    "AuthenticationError",
    "AuthorizationError",
    "TokenExpiredError",
    "InvalidCredentialsError",
    "InsufficientPermissionsError",
    "RateLimitExceededError",
    "SecurityConfigError",
    # Utilities
    "SecurityUtils",
    "TokenValidator",
    "PasswordValidator",
    "SecurityLogger",
]

# Default security configuration
DEFAULT_SECURITY_CONFIG = {
    "jwt": {
        "algorithm": "HS256",
        "access_token_expire_minutes": 30,
        "refresh_token_expire_days": 7,
        "issuer": "darwin-platform",
        "audience": "darwin-users",
    },
    "password": {
        "min_length": 8,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_numbers": True,
        "require_special_chars": True,
        "max_age_days": 90,
    },
    "rate_limiting": {
        "default_rate": "100/minute",
        "auth_rate": "10/minute",
        "burst_rate": "1000/hour",
    },
    "session": {
        "cookie_secure": True,
        "cookie_httponly": True,
        "cookie_samesite": "lax",
        "max_age_seconds": 3600,
    },
    "cors": {
        "allow_origins": ["http://localhost:3000", "http://localhost:8080"],
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Authorization", "Content-Type", "X-Requested-With"],
        "allow_credentials": True,
    },
    "security_headers": {
        "x_frame_options": "DENY",
        "x_content_type_options": "nosniff",
        "x_xss_protection": "1; mode=block",
        "strict_transport_security": "max-age=31536000; includeSubDomains",
        "content_security_policy": "default-src 'self'",
    },
}


def get_version():
    """Get the current version of the security package."""
    return __version__


def create_auth_manager(**kwargs):
    """Factory function to create a configured authentication manager."""
    config = DEFAULT_SECURITY_CONFIG.copy()
    config.update(kwargs)
    security_config = SecurityConfig(**config)
    return AuthManager(config=security_config)


def create_access_control(**kwargs):
    """Factory function to create a configured access control system."""
    config = DEFAULT_SECURITY_CONFIG.copy()
    config.update(kwargs)
    return AccessControl(config=config)


# Security constants
class SecurityConstants:
    """Security-related constants."""

    # Token types
    ACCESS_TOKEN = "access"
    REFRESH_TOKEN = "refresh"
    API_KEY = "api_key"

    # Default roles
    ADMIN_ROLE = "admin"
    USER_ROLE = "user"
    VIEWER_ROLE = "viewer"

    # Permissions
    READ_PERMISSION = "read"
    WRITE_PERMISSION = "write"
    DELETE_PERMISSION = "delete"
    ADMIN_PERMISSION = "admin"

    # Security headers
    AUTHORIZATION_HEADER = "Authorization"
    API_KEY_HEADER = "X-API-Key"
    CSRF_TOKEN_HEADER = "X-CSRF-Token"

    # Rate limiting
    RATE_LIMIT_HEADER = "X-RateLimit-Limit"
    RATE_LIMIT_REMAINING_HEADER = "X-RateLimit-Remaining"
    RATE_LIMIT_RESET_HEADER = "X-RateLimit-Reset"
